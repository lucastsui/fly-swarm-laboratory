"""Sequence-supervised synaptic learning with old-task replay and service tests.

The student controls every training/evaluation action. Expert geometry supplies
training labels only. Recurrent state continues between updates; gradients are
truncated at segment boundaries. Original graph, senses and decoder stay fixed.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .conditioning_experiment import evaluate as pickup_evaluation
from .evaluate_supervised import FrozenPolicy
from .haul_world import HaulWorld
from .plastic_brain import digest
from .supervised_joint import dataset,raw_readout,teacher_label,curriculum_world,transport_evaluation
from .supervised_steering import TrainableConnectome,brief


def service_target(world):
    """Privileged TRAINING supervisor; never called by the evaluator/actor."""
    if world.cargo:return world.stations[world.cargo]
    choices=[s for s in world.stations[:3] if s['stock']>0 or s['timer']>0]
    # Waiting briefly for a nearby processing station can be more efficient
    # than abandoning the item to go back to the source.
    return min(choices,key=lambda s:math.hypot(world.x-s['x'],world.y-s['y'])+.3*s['timer']) if choices else None


def service_label(world):
    target=service_target(world)
    return teacher_label(world,target) if target is not None else np.asarray([0.,0.,.2],np.float32)


def training_world(seed,kind):
    if kind in ('approach','carry'):return curriculum_world(seed,kind)[0]
    w=HaulWorld(seed,horizon=2400)
    if kind=='final':
        rng=np.random.default_rng(seed)
        # Full, unmasked factory with preloaded final-stage cargo. Recovery
        # positions span both sides of the line and arbitrary headings.
        for station in w.stations:station['stock']=0
        w.cargo=3;w.x=float(rng.uniform(7.5,14.));w.y=float(rng.uniform(4.5,9.5))
        w.heading=float(rng.uniform(-math.pi,math.pi));w.horizon=400
        w.trajectory.clear();w.trajectory.append([w.x,w.y])
    return w


class SequenceBatch:
    def __init__(self,batch,seed,natural_recovery=False):
        if batch%8:raise ValueError('Use batches divisible by eight')
        self.rng=np.random.default_rng(seed)
        self.kinds=['line']*(batch//2)+['line' if natural_recovery else 'final']*(batch//4)+['approach']*(batch//8)+['carry']*(batch//8)
        self.worlds=[self.new_world(kind) for kind in self.kinds]
        self.completed={kind:0 for kind in set(self.kinds)}
        self.resets={kind:0 for kind in set(self.kinds)}
        self.actions=0

    def new_world(self,kind):return training_world(int(self.rng.integers(210000000,220000000)),kind)

    def observations_and_labels(self):
        return np.asarray([w.sensory() for w in self.worlds]),np.asarray([service_label(w) for w in self.worlds])

    def advance(self,actions):
        reset=np.zeros(len(self.worlds),bool)
        for i,(world,kind,action) in enumerate(zip(self.worlds,self.kinds,actions)):
            # Only decoded student actions reach the existing environment.
            world.advance({'speed':float(action[0]),'turn':float(action[1]),'interact':bool(action[2]>.025)})
            success=(world.deliveries>=3 if kind=='line' else world.pickups>0 if kind=='approach' else world.transfers>0)
            if success or world.steps>=world.horizon:
                self.completed[kind]+=int(success);self.resets[kind]+=1
                self.worlds[i]=self.new_world(kind);reset[i]=True
        self.actions+=len(self.worlds)
        return reset


def mask_reset_state(state,reset):
    # Multiplication is differentiable; resetting an episode must not retain
    # either its neural state or its gradient history in the replacement.
    return state*(~torch.as_tensor(reset,device=state.device))[None]


@torch.no_grad()
def service_evaluation(model,seeds,batch=16,horizon=4800,replenish=False,blank=False):
    policy=FrozenPolicy(model,batch);initial_hash=digest(policy.gains);trials=[]
    for offset in range(0,len(seeds),batch):
        group=seeds[offset:offset+batch];worlds=[HaulWorld(int(seed),horizon=horizon) for seed in group]
        policy.reset();done=np.zeros(len(group),bool);first=np.full(len(group),-1);cleared=np.full(len(group),-1)
        milestones=[{} for _ in group];delivery_times=[[] for _ in group]
        for tick in range(horizon):
            obs=np.asarray([w.sensory() for w in worlds])
            if blank:obs[:]=0
            actions=policy.act(obs,explore=False)
            for i,(w,a) in enumerate(zip(worlds,actions)):
                if not done[i]:
                    before=w.deliveries;w.advance(a)
                    if w.deliveries>before:
                        delivery_times[i].append((tick+1)*.05)
                        if first[i]<0:first[i]=tick+1
                    if replenish:
                        # Exogenous source supply, not an agent action or
                        # automatic pickup/delivery. Rewards/terminal flags
                        # from the original finite task are not used/scored.
                        w.stations[0]['stock']=3
                    elif w.deliveries>=3:
                        done[i]=True;cleared[i]=tick+1
                if (tick+1)%800==0:
                    milestones[i][str((tick+1)//20)]={'products':w.deliveries,'pickups':w.pickups,'transfers':w.transfers}
        trials.extend([{'seed':int(seed),'products':w.deliveries,'pickups':w.pickups,'transfers':w.transfers,
            'firstProductSeconds':float(first[i]*.05) if first[i]>=0 else None,
            'clearSeconds':float(cleared[i]*.05) if cleared[i]>=0 else None,
            'cargoAtEnd':w.cargo,'distance':w.distance,'deliveryTimes':delivery_times[i],
            'milestones':milestones[i]} for i,(seed,w) in enumerate(zip(group,worlds))])
    assert initial_hash==digest(model.log_gains.detach().cpu().numpy())
    checkpoints={}
    for second in range(40,horizon//20+1,40):
        rows=[t['milestones'][str(second)] for t in trials]
        checkpoints[str(second)]={'episodesWithProduct':sum(r['products']>0 for r in rows),
            'episodesAllThree':sum(r['products']>=3 for r in rows),'products':sum(r['products'] for r in rows),
            'pickups':sum(r['pickups'] for r in rows),'transfers':sum(r['transfers'] for r in rows)}
    return {'gainsHash':initial_hash,'cases':len(trials),'horizon':horizon,'seconds':horizon*.05,
        'continuousSupply':replenish,'blankSenses':blank,'teacher':False,'learning':False,'noise':False,
        'checkpoints':checkpoints,'productsPerFactoryMinute':sum(t['products'] for t in trials)/(len(trials)*horizon*.05/60),
        'trials':trials}


def retention(model,seed,batch=16):
    return {'pickup':pickup_evaluation(FrozenPolicy(model,batch),40,seed+10000,64),
        'approach':transport_evaluation(model,list(range(seed,seed+32)),'approach',batch),
        'carry':transport_evaluation(model,list(range(seed,seed+32)),'carry',batch)}


def short(report):return {k:brief(v) if isinstance(v,dict) else v for k,v in report.items()}


def development_rank(report):
    """Whole-line completion first, total production next, early service last.

    This chooses a candidate for independent verification, not deployment.
    Training loss and within-training events are deliberately excluded.
    """
    last=report['checkpoints'][str(max(map(int,report['checkpoints'])))]
    early=report['checkpoints']['40']
    return last['episodesAllThree'],last['products'],early['episodesWithProduct']


def main(args):
    if args.out.exists():raise FileExistsError('Choose a new output; preserve earlier experiments')
    args.out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    initial=np.load(args.candidate,allow_pickle=False)['gains'].copy()
    model=TrainableConnectome(args.root,initial);device=model.log_gains.device
    protocol={**vars(args),'initialHash':digest(initial),'fixedInterfaceHash':model.fixed_hash,
        'sourceHash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input':'original 30 senses only','actions':'student fixed motor decoder, always',
        'training':'sequence supervised backprop through existing synaptic gains; detached state between segments',
        'dopamineLearning':False,'teacherAtEvaluation':False}
    (args.out/'protocol.json').write_text(json.dumps(protocol,default=str,indent=2))
    if args.evaluate_only:
        model.eval();model.requires_grad_(False)
        report={'retention':retention(model,args.eval_seed+10000,args.batch),
            'audit':{**model.audit(initial),'backpropagation':False}}
        if not args.continuous_only:
            report['finite']=service_evaluation(model,list(range(args.eval_seed,args.eval_seed+args.cases)),args.batch,args.horizon)
        if args.continuous or args.continuous_only:
            report['continuous']=service_evaluation(model,list(range(args.eval_seed+100000,args.eval_seed+100000+args.cases)),args.batch,args.horizon,True)
        (args.out/'result.json').write_text(json.dumps(report,indent=2))
        print('EVALUATION '+json.dumps({k:short(v) if k=='retention' else brief(v) for k,v in report.items()}),flush=True)
        return
    x,y,t=dataset(5100123);x=torch.as_tensor(x,device=device);y=torch.as_tensor(y,device=device)
    rng=np.random.default_rng(args.seed+99);worlds=SequenceBatch(args.batch,args.seed,args.natural_recovery);state=None
    optimizer=torch.optim.Adam([model.log_gains],lr=args.lr,eps=1e-10)
    history=[];gradient_audit=None;selection=None;best_rank=None
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();steps=0
    for step in range(1,args.updates+1):
        if time.perf_counter()-started>args.seconds:break
        optimizer.zero_grad(set_to_none=True);weights=model.weights()
        replay=step%args.replay_every==0
        if replay:
            indices=np.concatenate([rng.choice(np.flatnonzero(t==task),args.batch//4) for task in range(4)])
            ix=torch.as_tensor(indices,device=device);replay_state=None
            warmup=int(rng.choice([0,64,128,256]))
            if warmup:
                with torch.no_grad():_,replay_state=model(x[ix],warmup,weights=weights)
            _,replay_state=model(x[ix],args.segment*4,replay_state,weights)
            by_head=(raw_readout(model,replay_state)-y[ix]).square().mean(0)
        else:
            losses=[]
            if state is not None:state=state.detach()
            for _ in range(args.segment):
                obs,labels=worlds.observations_and_labels()
                importance=torch.as_tensor([args.final_weight if w.cargo==3 else 1. for w in worlds.worlds],device=device)
                action,state=model(torch.as_tensor(obs,device=device),4,state,weights)
                error=(raw_readout(model,state)-torch.as_tensor(labels,device=device)).square()
                losses.append((error*importance[:,None]).sum(0)/importance.sum())
                reset=worlds.advance(action.detach().cpu().numpy())
                if reset.any():state=mask_reset_state(state,reset)
            by_head=torch.stack(losses).mean(0)
        loss=by_head.sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();gradient=model.log_gains.grad
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite gradient')
        if gradient_audit is None:
            gradient_audit={'eligible':gradient.numel(),'nonzero':int(torch.count_nonzero(gradient)),'finite':True}
            print('GRADIENT '+json.dumps(gradient_audit),flush=True)
        torch.nn.utils.clip_grad_norm_([model.log_gains],1.)
        optimizer.step()
        with torch.no_grad():model.log_gains.clamp_(-2,2)
        if state is not None:state=state.detach()
        steps=step
        if step==1 or step%10==0:
            print(json.dumps({'update':step,'replay':replay,'lossByHead':by_head.detach().cpu().tolist(),
                'studentActions':worlds.actions,'trainingCompletions':worlds.completed,'trainingResets':worlds.resets,
                'seconds':time.perf_counter()-started,'gpuGB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
        if step%args.eval_every==0 or step==args.updates:
            gains=model.log_gains.detach().cpu().numpy();np.savez_compressed(args.out/f'candidate-{step}.npz',gains=gains)
            report=service_evaluation(model,list(range(4840000,4840016)),args.batch,2400)
            history.append({'update':step,'development':brief(report)})
            rank=development_rank(report)
            if best_rank is None or rank>best_rank:
                best_rank=rank;selection={'checkpoint':str(args.out/f'candidate-{step}.npz'),
                    'gainsHash':digest(gains),'update':step,'developmentRank':rank,
                    'requiresFrozenConfirmation':True,'automaticPromotion':False}
                (args.out/'selection.json').write_text(json.dumps(selection,indent=2))
            (args.out/f'evaluation-{step}.json').write_text(json.dumps(report,indent=2))
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
            print('DEVELOPMENT '+json.dumps(history[-1]),flush=True)
    gains=model.log_gains.detach().cpu().numpy();np.savez_compressed(args.out/f'candidate-final-{steps}.npz',gains=gains)
    torch.save({'gains':model.log_gains.detach().cpu(),'optimizer':optimizer.state_dict(),'steps':steps,
        'samplingRng':rng.bit_generator.state,'worldRng':worlds.rng.bit_generator.state,'protocol':protocol},args.out/'optimizer.pt')
    report={'updates':steps,'audit':model.audit(initial),'gradientAudit':gradient_audit,'history':history,
        'selectedForConfirmation':selection,
        'trainingSeconds':time.perf_counter()-started,'peakGPUAllocatedGB':torch.cuda.max_memory_allocated()/1e9,
        'retention':retention(model,5200000,args.batch),'status':'bounded experiment finished; no automatic promotion'}
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print('FINAL '+json.dumps({**{k:v for k,v in report.items() if k!='retention'},'retention':short(report['retention'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=5100001);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--updates',type=int,default=300);p.add_argument('--seconds',type=int,default=1200)
    p.add_argument('--lr',type=float,default=.0005);p.add_argument('--segment',type=int,default=8)
    p.add_argument('--replay-every',type=int,default=4);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--evaluate-only',action='store_true');p.add_argument('--eval-seed',type=int,default=5300000)
    p.add_argument('--cases',type=int,default=64);p.add_argument('--horizon',type=int,default=4800)
    p.add_argument('--continuous',action='store_true')
    p.add_argument('--continuous-only',action='store_true')
    p.add_argument('--natural-recovery',action='store_true')
    p.add_argument('--final-weight',type=float,default=1.)
    args=p.parse_args()
    if args.batch%8 or args.cases%args.batch or args.segment<1 or args.replay_every<2:raise ValueError('Invalid batch/segment/replay setting')
    if args.final_weight<=0:raise ValueError('Positive final-stage loss weight required')
    main(args)
