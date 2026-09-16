"""Read-only circuit diagnostics; artificial positive controls are NOT learned results."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .plastic_brain import PlasticBrain
from .conditioning_experiment import evaluate,worlds_for


def probe(root,candidate):
    torch.set_num_threads(4); b=PlasticBrain(root,16,seed=101)
    gains=np.load(candidate)['gains'] if candidate else np.zeros_like(b.gains)
    for label,g in [('baseline',np.zeros_like(gains)),('candidate',gains),('artificial motor activation control',None)]:
        if g is None:
            g=np.zeros_like(gains); ix=np.isin(b.spec['post'],b.spec['motor_interact'])
            g[ix]=2*np.sign(b.spec['base'][ix])
        b.load_gains(g); b.reset()
        near=np.arange(16)%2==0; worlds=worlds_for(np.arange(710000,710016),near,40)
        records=[]
        for tick in range(40):
            motors=b.act([w.sensory() for w in worlds],explore=False)
            if tick in (0,9,19,39):
                records.append({'tick':tick,'nearInteract':float(np.mean([m['rates']['interact'] for m,n in zip(motors,near) if n])),
                    'farInteract':float(np.mean([m['rates']['interact'] for m,n in zip(motors,near) if not n]))})
        inputs=b.state[b.t['pre']].cpu().numpy()
        ix=np.isin(b.spec['post'],b.spec['motor_interact'])
        print(json.dumps({'label':label,'rates':records,'inputActivityNear':float(inputs[ix][:,near].mean()),
            'inputActivityFar':float(inputs[ix][:,~near].mean()),'evaluation':{k:v for k,v in evaluate(b,40).items() if k!='trials'}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path)
    a=p.parse_args(); probe(a.root,a.candidate)
