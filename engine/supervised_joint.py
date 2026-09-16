"""Joint synaptic backprop: locomotion, steering, interaction; fixed interface.

Geometry provides TRAINING LABELS only. Frozen evaluation uses the original
30 senses, recurrent connectome and motor decoder, with actual world.advance.
Single-visible-station tasks are explicitly a curriculum, not a full factory.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .conditioning_experiment import evaluate as pickup_evaluation,worlds_for
from .evaluate_supervised import FrozenPolicy
from .haul_world import HaulWorld,wrap
from .plastic_brain import digest
from .supervised_steering import TrainableConnectome,brief
from .validate_candidate import haul


TASKS=('approach','carry','pickup','line')


def curriculum_world(seed,kind,bearing=None,distance=None):
    rng=np.random.default_rng(seed);w=HaulWorld(seed,horizon=400)
    for s in w.stations:s['stock']=0;s['timer']=0.
    stage=1 if kind=='carry' else 0
    if kind=='line':stage=int(rng.integers(0,4))
    if stage==0:w.stations[0]['stock']=3
    w.cargo=stage;target=w.stations[stage]
    angle=float(rng.uniform(-2.4,2.4)) if bearing is None else bearing
    d=float(rng.uniform(.3,4.5)) if distance is None else distance
    # Keep avatar inside the original arena, retain the actual station spacing.
    w.x=target['x']-d*math.cos(angle);w.y=target['y']-d*math.sin(angle);w.heading=0.
    if kind!='line':w.landmarks=[target]
    w.trajectory.clear();w.trajectory.append([w.x,w.y])
    return w,target


def teacher_label(world,target):
    """Labels never enter sensory() or the frozen actor."""
    d=math.hypot(target['x']-world.x,target['y']-world.y)
    bearing=wrap(math.atan2(target['y']-world.y,target['x']-world.x)-world.heading)
    proximity=float(np.clip((d-.55)/.6,0,1))
    speed=proximity*max(0.,math.cos(bearing))
    turn=float(np.clip(1.2*bearing,-1,1))*proximity
    # Cargo produces the existing taste signal too: allow repeated attempts
    # while carrying. Only the world, not this label, decides valid transfers.
    interact=.055 if world.sensory()[27]>.5 else .005
    return np.asarray([speed,turn,40*interact],np.float32)


def dataset(seed,per_task=1024):
    rng=np.random.default_rng(seed);observations=[];labels=[];tasks=[]
    for task,kind in enumerate(TASKS):
        for _ in range(per_task):
            scene=int(rng.integers(100000000,200000000))
            if kind=='pickup':
                near=bool(rng.integers(0,2));w=worlds_for([scene],[near],40)[0]
                # Preserve near/far discrimination even when nothing is visible.
                target=w.stations[0]
                label=np.asarray([0.,0.,2.2 if near else .2],np.float32)
            else:
                w,target=curriculum_world(scene,kind)
                label=teacher_label(w,target)
            w.speed=float(rng.uniform(0,1.2));w.turn=float(rng.uniform(-1.2,1.2))
            observations.append(w.sensory());labels.append(label);tasks.append(task)
    return np.asarray(observations,np.float32),np.asarray(labels,np.float32),np.asarray(tasks)


def raw_readout(model,state):
    # Unclipped training readouts retain corrective gradients at saturation.
    # They never replace the fixed clipped motor decoder during execution.
    return torch.stack([80*state[model.motor_forward].mean(0),
                        160*(state[model.motor_right].mean(0)-state[model.motor_left].mean(0)),
                        40*state[model.motor_interact].mean(0)],dim=1)


def mirror_observations(observations):
    """Reflection of original senses; no added channel or task information."""
    mirrored=observations.copy()
    mirrored[:,:24]=observations[:,:24][:,::-1]
    mirrored[:,24]=observations[:,25];mirrored[:,25]=observations[:,24]
    return mirrored


@torch.no_grad()
def static_check(model,observations,labels,tasks,batch=16,substeps=160):
    device=model.log_gains.device;weights=model.weights();predictions=[]
    for start in range(0,len(observations),batch):
        action,_=model(torch.as_tensor(observations[start:start+batch],device=device),substeps,weights=weights)
        action[:,2]*=40;predictions.append(action.cpu().numpy())
    p=np.concatenate(predictions);report={}
    for task,kind in enumerate(TASKS):
        select=tasks==task;target=labels[select];pred=p[select];turn=np.abs(target[:,1])>.2
        report[kind]={'cases':int(select.sum()),'mseByHead':((pred-target)**2).mean(0).tolist(),
            'meanSpeed':float(pred[:,0].mean()),'targetMeanSpeed':float(target[:,0].mean()),
            'directionCases':int(turn.sum()),
            'directionAccuracy':float(np.mean(np.sign(pred[turn,1])==np.sign(target[turn,1]))) if turn.any() else None,
            'interactionAccuracy':float(np.mean((pred[:,2]>1)==(target[:,2]>1)))}
    return report


@torch.no_grad()
def transport_evaluation(model,seeds,kind,batch=16,horizon=400,blank=False):
    """Actual travel + pickup/drop; no teacher actions or synthetic rewards."""
    trials=[];policy=FrozenPolicy(model,batch)
    for start in range(0,len(seeds),batch):
        group=seeds[start:start+batch];scenes=[]
        for seed in group:
            rng=np.random.default_rng(int(seed))
            scenes.append(curriculum_world(int(seed),kind,distance=float(rng.uniform(1.5,4.5))))
        worlds=[w for w,_ in scenes];policy.reset();done=np.zeros(len(group),bool)
        initial=[math.hypot(t['x']-w.x,t['y']-w.y) for w,t in scenes]
        ticks=np.full(len(group),horizon)
        for tick in range(horizon):
            observations=np.asarray([w.sensory() for w in worlds])
            if blank:observations[:,:24]=.025
            motors=policy.act(observations,explore=False)
            for i,(w,motor) in enumerate(zip(worlds,motors)):
                if done[i]:continue
                w.advance(motor)
                if (w.transfers if kind=='carry' else w.pickups)>0:
                    done[i]=True;ticks[i]=tick+1
        for i,(seed,(w,target)) in enumerate(zip(group,scenes)):
            trials.append({'seed':int(seed),'passed':bool(done[i]),'initialDistance':initial[i],
                'finalDistance':math.hypot(target['x']-w.x,target['y']-w.y),
                'travel':w.distance,'pickups':w.pickups,'transfers':w.transfers,'interactions':w.interactions,
                'seconds':float(ticks[i]*.05)})
    return {'cases':len(trials),'successes':sum(t['passed'] for t in trials),
        'meanTravel':float(np.mean([t['travel'] for t in trials])),
        'meanFinalDistance':float(np.mean([t['finalDistance'] for t in trials])),
        'horizon':horizon,'blankVision':blank,'teacher':False,'learning':False,'noise':False,
        'singleVisibleStation':True,'trials':trials}


def evaluate_all(model,cases,seed,batch=16,full_line=False,blank=False):
    seeds=list(range(seed,seed+cases));before=digest(model.log_gains.detach().cpu().numpy())
    report={'gainsHash':before,'seed':seed,'cases':cases,'teacher':False,'learning':False,
        'approach':transport_evaluation(model,seeds,'approach',batch),
        'carry':transport_evaluation(model,seeds,'carry',batch),
        'pickup':pickup_evaluation(FrozenPolicy(model,batch),40,seed+10000,cases*2)}
    if full_line:report['fullLine']=haul(FrozenPolicy(model,batch),np.arange(seed+20000,seed+20000+cases),800)
    if blank:
        report['blankApproach']=transport_evaluation(model,seeds,'approach',batch,blank=True)
        report['blankCarry']=transport_evaluation(model,seeds,'carry',batch,blank=True)
        report['blankPickup']=pickup_evaluation(FrozenPolicy(model,batch),40,seed+10000,cases*2,blank=True)
    assert before==digest(model.log_gains.detach().cpu().numpy())
    return report


def brief_evaluation(report):
    return {k:brief(v) if isinstance(v,dict) else v for k,v in report.items()}


def run(args):
    if args.out.exists():raise FileExistsError('Preserve old experiments; choose a new output')
    args.out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    initial=np.load(args.candidate,allow_pickle=False)['gains'].copy()
    model=TrainableConnectome(args.root,initial);device=model.log_gains.device
    protocol={**vars(args),'initialHash':digest(initial),'trainable':'all existing signed synaptic log gains only',
        'lossHeads':['speed','turn','40 * interaction rate'],'taskMix':list(TASKS),
        'labels':'scripted teacher during optimization only; never an inference input or action override',
        'input':'original 30 senses','teacherAtEvaluation':False,'dopamineLearning':False,
        'fixedInterfaceHash':model.fixed_hash,'sourceHash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'protocol.json').write_text(json.dumps(protocol,default=str,indent=2))
    vx,vy,vt=dataset(4811123,64)
    if args.evaluate_only:
        result=evaluate_all(model,args.cases,args.eval_seed,args.batch,full_line=True,blank=True)
        result['audit']={**model.audit(initial),'backpropagation':False}
        (args.out/'result.json').write_text(json.dumps(result,indent=2))
        print('FINAL '+json.dumps(brief_evaluation(result)),flush=True);return
    history=[];initial_metrics=evaluate_all(model,args.cases,4820000,args.batch,full_line=True)
    initial_metrics['static']=static_check(model,vx,vy,vt,args.batch,args.warmup+args.substeps)
    (args.out/'initial.json').write_text(json.dumps(initial_metrics,indent=2))
    print('BASELINE '+json.dumps(brief_evaluation(initial_metrics)),flush=True)
    x,y,t=dataset(args.seed)
    if args.paired:
        # Balance each individual minibatch, not merely the dataset. The
        # earlier steering pilot needed this to expose weak directional input.
        mirrored=mirror_observations(x);mirror_y=y.copy();mirror_y[:,1]*=-1
        x=np.stack([x,mirrored],axis=1).reshape(-1,30)
        y=np.stack([y,mirror_y],axis=1).reshape(-1,3);t=np.repeat(t,2)
    x=torch.as_tensor(x,device=device);y=torch.as_tensor(y,device=device)
    optimizer=torch.optim.Adam([model.log_gains],lr=args.lr,eps=1e-10)
    rng=np.random.default_rng(args.seed+99);began=time.perf_counter();gradient_audit=None;steps=0
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,args.updates+1):
        if time.perf_counter()-began>args.seconds:break
        # Every minibatch includes every task; none of the heads is frozen.
        if args.paired:
            pairs=np.concatenate([rng.choice(np.flatnonzero(t[::2]==task),args.batch//8) for task in range(4)])
            indices=np.stack([2*pairs,2*pairs+1],axis=1).flatten()
        else:indices=np.concatenate([rng.choice(np.flatnonzero(t==task),args.batch//4) for task in range(4)])
        indices=torch.as_tensor(indices,device=device)
        optimizer.zero_grad(set_to_none=True);weights=model.weights();state=None
        warmup=int(rng.choice([0,64,128,256])) if args.vary_warmup else args.warmup
        if warmup:
            with torch.no_grad():_,state=model(x[indices],warmup,weights=weights)
        _,state=model(x[indices],args.substeps,state,weights)
        prediction=raw_readout(model,state);by_head=(prediction-y[indices]).square().mean(0)
        loss=by_head.sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();gradient=model.log_gains.grad
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite gradient')
        if gradient_audit is None:
            gradient_audit={'eligible':gradient.numel(),'nonzero':int(torch.count_nonzero(gradient)),
                'maxAbsolute':float(gradient.abs().max()),'finite':True}
            print('GRADIENT '+json.dumps(gradient_audit),flush=True)
        torch.nn.utils.clip_grad_norm_([model.log_gains],1.)
        optimizer.step()
        with torch.no_grad():model.log_gains.clamp_(-2,2)
        steps=step
        if step==1 or step%10==0:
            print(json.dumps({'update':step,'lossByHead':by_head.detach().cpu().tolist(),
                'seconds':time.perf_counter()-began,'gpuGB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
        if step%args.eval_every==0 or step==args.updates:
            report=static_check(model,vx,vy,vt,args.batch,args.warmup+args.substeps)
            history.append({'update':step,'static':report})
            gains=model.log_gains.detach().cpu().numpy();np.savez_compressed(args.out/f'candidate-{step}.npz',gains=gains)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
            print('HELD_OUT '+json.dumps(history[-1]),flush=True)
    gains=model.log_gains.detach().cpu().numpy();np.savez_compressed(args.out/f'candidate-final-{steps}.npz',gains=gains)
    torch.save({'gains':model.log_gains.detach().cpu(),'optimizer':optimizer.state_dict(),'steps':steps,
        'rng':torch.get_rng_state(),'cudaRng':torch.cuda.get_rng_state(),'samplingRng':rng.bit_generator.state,
        'protocol':protocol},args.out/'resume.pt')
    training_seconds=time.perf_counter()-began
    result={'updates':steps,'initial':initial_metrics,'final':evaluate_all(model,args.cases,4820000,args.batch,full_line=True,blank=True),
        'static':static_check(model,vx,vy,vt,args.batch,args.warmup+args.substeps),
        'audit':model.audit(initial),'gradientAudit':gradient_audit,'history':history,
        'trainingSeconds':training_seconds,'peakGPUAllocatedGB':torch.cuda.max_memory_allocated()/1e9,
        'status':'bounded experiment complete; no promotion'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print('FINAL '+json.dumps({'updates':steps,'final':brief_evaluation(result['final']),'audit':result['audit']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=4800123);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--updates',type=int,default=300);p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--substeps',type=int,default=32);p.add_argument('--lr',type=float,default=.005)
    p.add_argument('--warmup',type=int,default=128);p.add_argument('--vary-warmup',action='store_true')
    p.add_argument('--paired',action='store_true')
    p.add_argument('--eval-every',type=int,default=50);p.add_argument('--cases',type=int,default=32)
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--eval-seed',type=int,default=4900000)
    args=p.parse_args()
    if args.batch%4 or args.batch<4 or args.cases%args.batch:raise ValueError('Use four-task batches and complete evaluations')
    if args.paired and args.batch%8:raise ValueError('Mirrored four-task batches require multiples of eight')
    run(args)
