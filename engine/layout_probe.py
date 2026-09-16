"""Frozen counterfactual cargo tests. No test-time training or action override."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .layout_brain import LayoutConnectome
from .layout_training import sample_batch
from .supervised_joint import raw_readout
from .supervised_steering import TrainableConnectome

@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    gains=np.load(args.candidate,allow_pickle=False)['gains']
    model=(TrainableConnectome(args.root,gains) if args.legacy else
           LayoutConnectome(args.root,gains,overlay=args.overlay,stock=args.stock))
    model.eval().requires_grad_(False)
    x,y=sample_batch(np.random.default_rng(6900000),32,stock=args.stock,contrast=True)
    if args.legacy:x=x[:,:30]
    predictions=[];weights=model.weights()
    for offset in range(0,len(x),16):
        _,h=model(torch.as_tensor(x[offset:offset+16],device='cuda'),128,weights=weights)
        predictions.extend(raw_readout(model,h).cpu().tolist())
    p=np.asarray(predictions);mask=np.abs(y[:,1])>.2
    # Only compare cargo stages 1..3; original generic cargo input is identical.
    change=np.ptp(p.reshape(-1,4,3)[:,1:,1],axis=1)
    report={'sensoryInterface':getattr(model,'interface','original-30'),
            'cases':len(p),'mse':((p-y)**2).mean(0).tolist(),
            'directionAccuracy':float(np.mean(np.sign(p[mask,1])==np.sign(y[mask,1]))),
            'meanCargoSpecificTurnRange':float(change.mean()),
            'targetCargoSpecificTurnRange':float(np.ptp(y.reshape(-1,4,3)[:,1:,1],axis=1).mean()),
            'rows':[{'target':a.tolist(),'actual':b.tolist()} for a,b in zip(y[:16],p[:16])]}
    if args.out:
        if args.out.exists():raise FileExistsError('Preserve old probes')
        args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--legacy',action='store_true')
    p.add_argument('--overlay',action='store_true');p.add_argument('--stock',action='store_true')
    run(p.parse_args())
