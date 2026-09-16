"""Locked, paired noise-free task evaluation. No updates or action supervision."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .plastic_brain import PlasticBrain,digest
from .haul_world import HaulWorld
from .conditioning_experiment import evaluate


def haul(brain,seeds,horizon,blank=False):
    trials=[]; initial=brain.gains.copy(); start=time.time()
    for offset in range(0,len(seeds),brain.batch):
        batch_seeds=seeds[offset:offset+brain.batch]
        assert len(batch_seeds)==brain.batch
        worlds=[HaulWorld(int(s),horizon=horizon) for s in batch_seeds]; brain.reset()
        for tick in range(horizon):
            observations=[np.zeros(30,np.float32) if blank else w.sensory() for w in worlds]
            motors=brain.act(observations,explore=False)
            for world,motor in zip(worlds,motors): world.advance(motor)
        trials.extend([{'seed':int(seed),'pickups':w.pickups,'transfers':w.transfers,'products':w.deliveries,
            'reward':w.total_reward,'interactions':w.interactions,'distance':w.distance}
            for seed,w in zip(batch_seeds,worlds)])
    assert np.array_equal(initial,brain.gains)
    return {'episodes':len(trials),'horizon':horizon,'blankSensory':blank,'noise':False,'learning':False,
        'wallSeconds':time.time()-start,'gainsHash':digest(brain.gains),'trials':trials,
        **{k:sum(t[k] for t in trials) for k in ('pickups','transfers','products')},
        'pickupSuccessRate':float(np.mean([t['pickups']>0 for t in trials])),
        'transferSuccessRate':float(np.mean([t['transfers']>0 for t in trials])),
        'productSuccessRate':float(np.mean([t['products']>0 for t in trials])),
        'meanReward':float(np.mean([t['reward'] for t in trials]))}


def main(args):
    torch.set_num_threads(4); b=PlasticBrain(args.root,16,401)
    learned=np.load(args.candidate,allow_pickle=False)['gains'].copy()
    # This seed range is disjoint from assay training and validation ranges.
    protocol={'seedStart':args.seed_start,'cases':args.cases,'horizon':args.horizon,
        'candidate':str(args.candidate),'candidateHash':digest(learned),'frozen':True,'noise':False,
        'decoderFixed':True,'sensoryProjectionFixed':True}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results={'protocol':protocol}
    for label,gains,blank in [('original',np.zeros_like(learned),False),('learned',learned,False),('blank',learned,True)]:
        b.load_gains(gains)
        report=haul(b,np.arange(args.seed_start,args.seed_start+args.cases),args.horizon,blank)
        results[label]=report
        print(label.upper()+' '+json.dumps({k:v for k,v in report.items() if k!='trials'}),flush=True)
        (args.out/'result.json').write_text(json.dumps(results,indent=2))
    b.load_gains(learned)
    results['pickupDiscrimination']=evaluate(b,40,start=args.seed_start+10000,cases=256)
    results['audit']=b.audit()
    (args.out/'result.json').write_text(json.dumps(results,indent=2))
    print('AUDIT '+json.dumps(results['audit']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed-start',type=int,default=1300000);p.add_argument('--cases',type=int,default=64)
    p.add_argument('--horizon',type=int,default=800)
    main(p.parse_args())
