"""Random-layout synaptic training and frozen generalization evaluation.

Teacher labels are local, training-only. Every rollout action comes from the
full connectome's unchanged motor decoder. No controller at evaluation.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .layout_world import LayoutSwarmWorld,color_sensory,random_layout
from .layout_brain import LayoutConnectome
from .supervised_steering import TrainableConnectome
from .supervised_joint import teacher_label,raw_readout
from .evaluate_supervised import FrozenPolicy
from .plastic_brain import digest,DT

KINDS=('rotated','permuted','compact','wide')
LABEL_VERSION='retinal-visibility-0.002-wall-escape-v2'

def local_label(agent,stock=False):
    # A target inside the distance limit can still lie in the rear retinal gap.
    # Do not teach precise steering from only microscopic Gaussian tails.
    retina=color_sensory(agent)[30:126].reshape(4,24)
    visible=[s for i,s in enumerate(agent.stations) if float(retina[i].max())>=.002]
    if agent.cargo:
        target=agent.stations[agent.cargo]
        if any(s is target for s in visible):return teacher_label(agent,target)
    else:
        candidates=[s for s in visible if s is not agent.stations[3] and s['stock']>0]
        if candidates:
            target=min(candidates,key=lambda s:math.hypot(s['x']-agent.x,s['y']-agent.y)+.3*s['timer'])
            return teacher_label(agent,target)
    # When nothing useful is visible, labels cannot depend on hidden targets.
    # Straight exploration, local wall/contact turning; no global route.
    near=agent.sensory()[26]>.4
    # A .15-speed / 1-turn circle cannot leave the .6-unit wall-proximity
    # zone (radius .15). Keep enough forward motion to actually turn away.
    return np.asarray([.7 if near else .8,1.2 if near else 0.,2.2 if agent.cargo else .2],np.float32)

def sample_batch(rng,count,stock=False,contrast=False):
    observations=[];labels=[]
    for _ in range(count):
        seed=int(rng.integers(6100000,6200000));kind=str(rng.choice(KINDS))
        world=LayoutSwarmWorld(seed,flies=1,kind=kind);a=world.agents[0]
        stage=int(rng.integers(0,4));a.cargo=stage
        target=world.stations[stage]
        # Visible approach / pickup / drop / blind exploration, all headings.
        for _ in range(100):
            d=float(rng.uniform(.25,5.8));angle=rng.uniform(-math.pi,math.pi)
            x=target['x']+d*math.cos(angle);y=target['y']+d*math.sin(angle)
            if .22<x<19.78 and .22<y<13.78:break
        a.x,a.y,a.heading=x,y,float(rng.uniform(-math.pi,math.pi))
        for s in world.stations[1:3]:s['stock']=int(rng.integers(0,3))
        a.speed=float(rng.uniform(0,1.3));a.turn=float(rng.uniform(-1,1))
        world.refresh_landmarks()
        for cargo in (range(4) if contrast else [stage]):
            a.cargo=cargo;observations.append(color_sensory(a,stock));labels.append(local_label(a,stock))
    return np.asarray(observations),np.asarray(labels)


def focus_batch(rng,count,stock=True,missing=0,varied=False):
    """Identical visible scene, four cargo states and conflicting correct turns.

    Training curriculum only; the deployed senses/world never select a target.
    Positions, box identities and heading are independently randomized.
    """
    if missing not in (0,1,2):raise ValueError('Hide zero, one or two boxes in this curriculum')
    observations=[];labels=[]
    for _ in range(count):
        w=LayoutSwarmWorld(int(rng.integers(6100000,6200000)),flies=1,stock=stock)
        a=w.agents[0];center=rng.uniform([5.,5.],[15.,9.]);heading=rng.uniform(-math.pi,math.pi)
        bearings=np.array([-2.,-.7,.7,2.])+rng.uniform(-.1,.1,4)
        distance=rng.uniform(2.,3.8,4);angles=bearings+heading
        p=center+distance[:,None]*np.c_[np.cos(angles),np.sin(angles)]
        if varied:
            for attempt in range(10000):
                bearings=rng.uniform(-2.6,2.6,4);distance=rng.uniform(.4,5.5,4)
                angles=bearings+heading
                p=center+distance[:,None]*np.c_[np.cos(angles),np.sin(angles)]
                try:w.set_layout(p[rng.permutation(4)])
                except ValueError:continue
                break
            else:raise ValueError('Could not create varied visible scene')
        else:w.set_layout(p[rng.permutation(4)])
        a.x,a.y=center;a.heading=heading;a.speed=rng.uniform(0,1.3);a.turn=rng.uniform(-1.,1.)
        if missing:
            # Real layout changes, not an encoded "target absent" instruction.
            # Every cargo sees the same scene, including every remaining box.
            roles=rng.choice(4,missing,replace=False)
            positions=np.asarray([[s['x'],s['y']] for s in w.stations])
            for attempt in range(10000):
                moved=positions.copy();moved[roles]=rng.uniform([.8,.8],[19.2,13.2],size=(missing,2))
                if np.any(np.linalg.norm(moved[roles]-center,axis=1)<=6.1):continue
                try:w.set_layout(moved)
                except ValueError:continue
                break
            else:raise ValueError('Could not place hidden curriculum boxes')
        for s in w.stations[1:3]:s['stock']=int(rng.integers(0,4))
        w.refresh_landmarks()
        for cargo in range(4):
            a.cargo=cargo;observations.append(color_sensory(a,stock));labels.append(local_label(a,stock))
    return np.asarray(observations),np.asarray(labels)


def demonstration_sequences(rng,world_count=16,steps=2400,stock=False):
    """Offline teacher examples, explicitly NOT a measure of brain performance.

    Includes moving retinal input, pickup/drop transitions and fly collisions.
    Training seeds are separate from the frozen development/confirmation sets.
    """
    if world_count<1 or steps<1:raise ValueError('Positive demonstration bounds required')
    worlds=[LayoutSwarmWorld(int(rng.integers(6100000,6200000)),kind=KINDS[i%4],stock=stock)
            for i in range(world_count)]
    x=np.empty((steps,4*world_count,201 if stock else 129),np.float32)
    y=np.empty((steps,4*world_count,3),np.float32)
    for tick in range(steps):
        x[tick]=np.concatenate([w.sensory() for w in worlds])
        y[tick]=[local_label(a,stock) for w in worlds for a in w.agents]
        for i,w in enumerate(worlds):
            w.advance([{'speed':float(label[0]),'turn':float(label[1]),'interact':bool(label[2]>1)}
                       for label in y[tick,i*4:i*4+4]])
    report={'teacherGeneratedExamples':True,'brainControlled':False,'steps':steps,
            'worlds':world_count,'labelVersion':LABEL_VERSION,'productsByTeacher':[w.deliveries for w in worlds]}
    return x,y,report

@torch.no_grad()
def evaluate(model,seed=6300000,cases=4,seconds=240,colored=True,progress=True,kinds=KINDS):
    worlds=[(kind,seed+k*1000+i,LayoutSwarmWorld(seed+k*1000+i,kind=kind,colored=colored,stock=getattr(model,'stock_sensing',False)))
            for k,kind in enumerate(kinds) for i in range(cases)]
    policy=FrozenPolicy(model,4*len(worlds));before=digest(policy.gains);times=[[] for _ in worlds]
    began=time.perf_counter()
    for tick in range(round(seconds/DT)):
        actions=policy.act(np.concatenate([w.sensory() for _,_,w in worlds]))
        for i,(_,_,w) in enumerate(worlds):
            old=w.deliveries;w.advance(actions[i*4:i*4+4]);times[i].extend([(tick+1)*DT]*(w.deliveries-old))
        if progress and (tick+1)%1200==0:
            print(json.dumps({'evalSeconds':(tick+1)*DT,'wallSeconds':time.perf_counter()-began,
                  'products':{kind:sum(w.deliveries for k,_,w in worlds if k==kind) for kind in kinds}}),flush=True)
    trials=[{'kind':kind,'seed':s,'positions':[[a['x'],a['y']] for a in w.stations],
             'products':w.deliveries,'pickups':w.pickups,'transfers':w.transfers,'bumps':w.bump_events,
             'deliveryTimes':t,'finalStock':[a['stock'] for a in w.stations],
             'finalAvatars':[{'x':a.x,'y':a.y,'heading':a.heading,'cargo':a.cargo,
                              'speed':a.speed,'turn':a.turn,'distance':a.distance} for a in w.agents]}
            for (kind,s,w),t in zip(worlds,times)]
    assert before==digest(model.log_gains.detach().cpu().numpy())
    return {'checkpointHash':before,'colored':colored,'sensoryInterface':getattr(model,'interface','original-30'),'seconds':seconds,'teacher':False,'learning':False,'seed':seed,
            'summary':{kind:{'cases':cases,'products':sum(t['products'] for t in trials if t['kind']==kind),
                        'worldsWithProduct':sum(t['products']>0 for t in trials if t['kind']==kind),
                        'worldsWithThree':sum(t['products']>=3 for t in trials if t['kind']==kind)} for kind in kinds},
            'trials':trials}

def main(args):
    if args.out.exists():raise FileExistsError('Keep prior experiments')
    args.out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    with np.load(args.candidate,allow_pickle=False) as archive:initial=archive['gains'].copy()
    cls=TrainableConnectome if args.legacy else LayoutConnectome
    model=cls(args.root,initial,**({'overlay':args.overlay,'stock':args.stock} if not args.legacy else {}))
    if args.evaluate:
        model.eval().requires_grad_(False)
        result=evaluate(model,args.eval_seed,args.cases,args.eval_seconds,not args.legacy,kinds=args.kinds.split(','))
        (args.out/'evaluation.json').write_text(json.dumps(result,indent=2));print('RESULT '+json.dumps(result['summary']),flush=True);return
    if args.legacy:raise ValueError('Training requires the color interface')
    rng=np.random.default_rng(args.seed)
    x,y=(focus_batch(rng,args.samples,stock=args.stock) if args.focus else
         sample_batch(rng,args.samples,stock=args.stock,contrast=args.contrast))
    x=torch.as_tensor(x,device='cuda');y=torch.as_tensor(y,device='cuda')
    # Fixed held-out local scenes for teacher-loss diagnostics, not service proof.
    vx,vy=sample_batch(np.random.default_rng(6400000),128,stock=args.stock);vx=torch.as_tensor(vx,device='cuda');vy=torch.as_tensor(vy,device='cuda')
    optimizer=torch.optim.Adam([model.log_gains],lr=args.lr,eps=1e-10)
    worlds=[LayoutSwarmWorld(int(rng.integers(6100000,6200000)),kind=KINDS[i%4],stock=args.stock) for i in range(args.batch//4)]
    state=None;history=[];began=time.perf_counter();completed=0
    for update in range(1,args.updates+1):
        optimizer.zero_grad(set_to_none=True);weights=model.weights()
        static=args.static_only or (update%4!=0 if args.static_first else update%4==0)
        contrast_loss=None
        if static:
            indices=(4*rng.integers(0,len(x)//4,args.batch//4)[:,None]+np.arange(4)).ravel() if args.contrast or args.focus else rng.integers(0,len(x),args.batch)
            indices=torch.as_tensor(indices,device='cuda')
            with torch.no_grad():_,hidden=model(x[indices],64,weights=weights)
            _,hidden=model(x[indices],args.unroll,hidden,weights)
            pred=raw_readout(model,hidden);errors=(pred-y[indices]).square().mean(0)
            if args.focus:
                # Remove the shared scene response: reward cargo-specific action
                # differences, not a turn toward the scene's visual average.
                delta=(pred-y[indices]).reshape(-1,4,3)[:,1:,:2]
                contrast_loss=(delta-delta.mean(1,keepdim=True)).square().mean()
        else:
            if state is not None:state=state.detach()
            losses=[]
            for _ in range(8):
                obs=np.concatenate([w.sensory() for w in worlds])
                target=np.asarray([local_label(a,args.stock) for w in worlds for a in w.agents])
                action,state=model(torch.as_tensor(obs,device='cuda'),4,state,weights)
                losses.append((raw_readout(model,state)-torch.as_tensor(target,device='cuda')).square().mean(0))
                actions=action.detach().cpu().numpy()
                reset=[]
                for i,w in enumerate(worlds):
                    w.advance([{'speed':float(a[0]),'turn':float(a[1]),'interact':bool(a[2]>.025)} for a in actions[i*4:i*4+4]])
                    if w.steps>=1200:
                        completed+=w.deliveries;worlds[i]=LayoutSwarmWorld(int(rng.integers(6100000,6200000)),kind=str(rng.choice(KINDS)),stock=args.stock)
                        reset.extend(range(i*4,i*4+4))
                if reset:
                    mask=torch.ones(args.batch,device='cuda');mask[reset]=0;state=state*mask[None]
            errors=torch.stack(losses).mean(0)
        loss=errors.sum()+(2*contrast_loss if contrast_loss is not None else 0)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward()
        if not torch.isfinite(model.log_gains.grad).all():raise FloatingPointError('Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_([model.log_gains],1.);optimizer.step()
        with torch.no_grad():model.log_gains.clamp_(-2,2)
        if state is not None:state=state.detach()
        if update%20==0 or update==1:
            row={'update':update,'loss':float(loss.detach()),'byHead':errors.detach().cpu().tolist(),
                 'seconds':time.perf_counter()-began,'products':completed+sum(w.deliveries for w in worlds)}
            print(json.dumps(row),flush=True);history.append(row)
        if update%args.save_every==0 or update==args.updates:
            gains=model.log_gains.detach().cpu().numpy();np.savez_compressed(args.out/f'candidate-{update}.npz',gains=gains)
            with torch.no_grad():
                weights=model.weights();_,h=model(vx,128,weights=weights)
                diagnostic=(raw_readout(model,h)-vy).square().mean(0).cpu().tolist()
            print('LOCAL_CHECK '+json.dumps({'update':update,'mse':diagnostic}),flush=True)
            if args.focus:
                fx,fy=focus_batch(np.random.default_rng(6800000),32,stock=args.stock)
                with torch.no_grad():
                    _,fh=model(torch.as_tensor(fx,device='cuda'),128,weights=model.weights())
                    fp=raw_readout(model,fh).cpu().numpy()
                errors=((fp-fy)**2).mean(0)
                print('FOCUS_CHECK '+json.dumps({'update':update,'mse':errors.tolist(),
                    'cargoTurnRange':float(np.ptp(fp.reshape(-1,4,3)[:,1:,1],axis=1).mean()),
                    'targetCargoTurnRange':float(np.ptp(fy.reshape(-1,4,3)[:,1:,1],axis=1).mean())}),flush=True)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
    result={'audit':model.audit(initial),'updates':args.updates,'seconds':time.perf_counter()-began,
            'sensoryInterface':model.interface,'graphAndDecoderPreserved':True,
            'training':'supervised full-synapse backpropagation; not dopamine','requiresFrozenVerification':True}
    (args.out/'result.json').write_text(json.dumps(result,indent=2));print('TRAINING_FINISHED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--evaluate',action='store_true');p.add_argument('--legacy',action='store_true')
    p.add_argument('--overlay',action='store_true');p.add_argument('--kinds',default=','.join(KINDS))
    p.add_argument('--stock',action='store_true');p.add_argument('--contrast',action='store_true')
    p.add_argument('--focus',action='store_true');p.add_argument('--static-only',action='store_true')
    p.add_argument('--unroll',type=int,default=32)
    p.add_argument('--seed',type=int,default=6500000);p.add_argument('--eval-seed',type=int,default=6300000)
    p.add_argument('--cases',type=int,default=4);p.add_argument('--eval-seconds',type=int,default=240)
    p.add_argument('--updates',type=int,default=400);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--samples',type=int,default=2048);p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--static-first',action='store_true')
    args=p.parse_args()
    if args.batch%4 or args.batch<4:raise ValueError('Batch must be divisible by four')
    if any(k not in KINDS for k in args.kinds.split(',')):raise ValueError('Unknown evaluation layout kind')
    main(args)
