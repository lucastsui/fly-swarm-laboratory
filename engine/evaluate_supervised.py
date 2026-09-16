"""Frozen sensory/closed-loop/full-line checks for supervised graph candidates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .supervised_steering import TrainableConnectome,physical_evaluation,static_evaluation,samples,brief
from .validate_candidate import haul
from .plastic_brain import digest


class FrozenPolicy:
    def __init__(self,model,batch):
        self.model=model;self.batch=batch;self.gains=model.log_gains.detach().cpu().numpy().copy()
        with torch.no_grad():self.weights=model.weights()
        self.state=None

    def reset(self):self.state=None

    @torch.no_grad()
    def act(self,observations,explore=False):
        if explore:raise ValueError('Frozen evaluator never explores')
        actions,self.state=self.model(torch.as_tensor(np.asarray(observations),device=self.model.log_gains.device),
                                      4,self.state,self.weights)
        return [{'speed':float(v[0]),'turn':float(v[1]),'interact':bool(v[2]>.025)} for v in actions.cpu().numpy()]


def main(args):
    if args.out.exists():raise FileExistsError('Keep prior evaluations')
    torch.set_num_threads(4);gains=np.load(args.candidate,allow_pickle=False)['gains']
    model=TrainableConnectome(args.root,gains);model.eval();model.requires_grad_(False)
    x,y,angles=samples(args.seed+10000,args.cases//2)
    x=torch.as_tensor(x,device='cuda');y=torch.as_tensor(y,device='cuda')
    report={'candidate':str(args.candidate),'gainsHash':digest(gains),'seed':args.seed,
            'learning':False,'noise':False,'teacher':False,'device':torch.cuda.get_device_name()}
    report['static']={str(steps):static_evaluation(model,x,y,angles,args.batch,steps) for steps in (32,128,256)}
    seeds=list(range(args.seed,args.seed+args.cases))
    report['physical']=physical_evaluation(model,seeds,args.batch)
    report['blankVision']=physical_evaluation(model,seeds,args.batch,blank=True)
    if args.full_line:
        report['fullLine']=haul(FrozenPolicy(model,args.batch),np.arange(args.seed+20000,args.seed+20000+args.cases),800)
    report['audit']={**model.audit(gains),'backpropagation':False}
    assert report['gainsHash']==digest(model.log_gains.detach().cpu().numpy())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({**{k:brief(v) for k,v in report.items() if k in ('physical','blankVision','fullLine')},
                      'static':{k:brief(v) for k,v in report['static'].items()},'audit':report['audit']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=int,default=64);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=4500000);p.add_argument('--full-line',action='store_true')
    args=p.parse_args()
    if args.cases%args.batch or args.cases%2:raise ValueError('Cases must be divisible by batch and two')
    main(args)
