"""Frozen extended-duration diagnostic, never used to teach or choose actions."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .evaluate_supervised import FrozenPolicy
from .haul_world import HaulWorld
from .plastic_brain import digest
from .supervised_steering import TrainableConnectome


def main(args):
    if args.out.exists():raise FileExistsError('Preserve existing traces')
    torch.set_num_threads(4);initial=np.load(args.candidate,allow_pickle=False)['gains'].copy()
    model=TrainableConnectome(args.root,initial);model.requires_grad_(False)
    policy=FrozenPolicy(model,args.batch);trials=[]
    for start in range(0,args.cases,args.batch):
        seeds=range(args.seed+start,args.seed+start+args.batch)
        worlds=[HaulWorld(seed,horizon=args.horizon) for seed in seeds];policy.reset()
        traces=[[] for _ in worlds];completed=np.zeros(args.batch,bool)
        for tick in range(args.horizon):
            actions=policy.act([w.sensory() for w in worlds])
            for i,(w,a) in enumerate(zip(worlds,actions)):
                if not completed[i]:
                    w.advance(a);completed[i]=w.deliveries>=3
                if tick%20==19 or tick==args.horizon-1:
                    targets=[w.stations[w.cargo]] if w.cargo else [s for s in w.stations[:3] if s['stock']>0]
                    traces[i].append({'seconds':(tick+1)*.05,'x':w.x,'y':w.y,'heading':w.heading,
                        'cargo':w.cargo,'speed':w.speed,'turn':w.turn,'pickups':w.pickups,
                        'transfers':w.transfers,'products':w.deliveries,'interactions':w.interactions,
                        'targetDistance':min((math.hypot(s['x']-w.x,s['y']-w.y) for s in targets),default=None),
                        'stock':[s['stock'] for s in w.stations],'timers':[s['timer'] for s in w.stations]})
        trials.extend([{'seed':seed,'trace':trace,'completedAllThree':bool(done)} for seed,trace,done in zip(seeds,traces,completed)])
    checkpoints={}
    for seconds in range(40,int(args.horizon*.05)+1,40):
        rows=[next(r for r in t['trace'] if r['seconds']==seconds) for t in trials]
        checkpoints[str(seconds)]={'episodesWithProduct':sum(r['products']>0 for r in rows),
            'episodesAllThree':sum(r['products']>=3 for r in rows),'products':sum(r['products'] for r in rows),
            'pickups':sum(r['pickups'] for r in rows),'transfers':sum(r['transfers'] for r in rows)}
    assert digest(initial)==digest(model.log_gains.detach().cpu().numpy())
    report={'gainsHash':digest(initial),'cases':args.cases,'seed':args.seed,'horizon':args.horizon,
        'teacher':False,'noise':False,'learning':False,'checkpoints':checkpoints,'trials':trials}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='trials'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=int,default=32);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=4840000);p.add_argument('--horizon',type=int,default=2400)
    args=p.parse_args()
    if args.cases%args.batch:raise ValueError('Complete batches required')
    main(args)
