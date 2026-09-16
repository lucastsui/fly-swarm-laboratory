"""Paired closed-loop swarm evaluation, one immutable brain and no teacher."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .swarm_world import SwarmWorld
from .supervised_steering import TrainableConnectome
from .evaluate_supervised import FrozenPolicy
from .plastic_brain import digest,DT

@torch.inference_mode()
def evaluate(args):
    if args.out.exists():raise FileExistsError('Preserve earlier reports')
    torch.set_num_threads(4)
    with np.load(args.candidate,allow_pickle=False) as archive:gains=archive['gains'].copy()
    model=TrainableConnectome(args.root,gains).eval().requires_grad_(False)
    groups=[('solo',1,True),('swarm',4,True),('ghostSwarm',4,False)]
    if args.swarm_only:groups=groups[1:2]
    worlds=[(label,seed,SwarmWorld(seed,flies=n,collisions=c))
            for label,n,c in groups for seed in range(args.seed,args.seed+args.cases)]
    policy=FrozenPolicy(model,sum(len(w.agents) for _,_,w in worlds))
    products=[[] for _ in worlds];milestones=[];min_separation=100.;began=time.perf_counter()
    for step in range(round(args.seconds/DT)):
        observations=np.concatenate([w.sensory() for _,_,w in worlds])
        actions=policy.act(observations);offset=0
        for i,(_,_,w) in enumerate(worlds):
            before=w.deliveries;n=len(w.agents)
            w.advance(actions[offset:offset+n]);offset+=n
            products[i].extend([(step+1)*DT]*(w.deliveries-before))
            if w.collisions_enabled:
                for j,a in enumerate(w.agents):
                    for b in w.agents[:j]:min_separation=min(min_separation,float(np.hypot(a.x-b.x,a.y-b.y)))
        if (step+1)%600==0:
            entry={'seconds':(step+1)*DT,'wallSeconds':round(time.perf_counter()-began,2),
                   **{label:{'products':sum(w.deliveries for name,_,w in worlds if name==label),
                            'worldsWithProduct':sum(w.deliveries>0 for name,_,w in worlds if name==label)}
                      for label,_,_ in groups}}
            milestones.append(entry);print(json.dumps(entry),flush=True)
    unchanged=digest(model.log_gains.detach().cpu().numpy())==digest(gains)
    assert unchanged and min_separation>=.44-1e-6,(unchanged,min_separation)
    trials=[{'group':label,'seed':seed,'products':w.deliveries,'pickups':w.pickups,'transfers':w.transfers,
             'deliveryTimes':times,'perFlyProducts':[a.deliveries for a in w.agents],
             'perFlyTransfers':[a.transfers for a in w.agents],'bumpEvents':w.bump_events,
             'solverHolds':w.solver_holds,'cargo':[a.cargo for a in w.agents]}
            for (label,seed,w),times in zip(worlds,products)]
    summary={label:{'cases':args.cases,'products':sum(t['products'] for t in trials if t['group']==label),
                  'atLeastThree':sum(t['products']>=3 for t in trials if t['group']==label),
                  'withProduct':sum(t['products']>0 for t in trials if t['group']==label),
                  'withLateProduct':sum(any(s>args.seconds/2 for s in t['deliveryTimes']) for t in trials if t['group']==label),
                  'bumpEvents':sum(t['bumpEvents'] for t in trials if t['group']==label)} for label,_,_ in groups}
    report={'checkpointHash':digest(gains),'candidate':str(args.candidate),'seed':args.seed,'seconds':args.seconds,
            'teacher':False,'learning':False,'noise':False,'weightsUnchanged':unchanged,
            'peerVision':True,'flyRadius':.22,'minSeparation':min_separation,'sharedFactoryClock':True,
            'summary':summary,'trials':trials,'milestones':milestones,'wallSeconds':time.perf_counter()-began}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print('COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,default=5600000);p.add_argument('--cases',type=int,default=8)
    p.add_argument('--seconds',type=int,default=240);p.add_argument('--swarm-only',action='store_true')
    evaluate(p.parse_args())
