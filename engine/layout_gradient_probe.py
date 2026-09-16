"""Read-only gradient diagnostic: does the gain bound prevent optimization?"""
import argparse,json
import numpy as np
import torch
from .layout_brain import LayoutConnectome
from .layout_training import focus_batch
from .supervised_joint import raw_readout

def main(args):
    torch.set_num_threads(4)
    gains=np.load(args.candidate,allow_pickle=False)['gains']
    model=LayoutConnectome(args.root,gains,stock=True)
    x,y=focus_batch(np.random.default_rng(7100000),4)
    x=torch.as_tensor(x,device='cuda');y=torch.as_tensor(y,device='cuda')
    weights=model.weights()
    with torch.no_grad():_,state=model(x,64,weights=weights)
    _,state=model(x,32,state,weights)
    error=raw_readout(model,state)-y;delta=error.reshape(-1,4,3)[:,1:,:2]
    loss=error.square().mean(0).sum()+2*(delta-delta.mean(1,keepdim=True)).square().mean()
    loss.backward();g=model.log_gains.detach();grad=model.log_gains.grad.detach()
    blocked=((g>=1.999)&(grad<0))|((g<=-1.999)&(grad>0))
    print(json.dumps({'loss':float(loss.detach()),'blockedEdges':int(blocked.sum()),
          'blockedGradientL1Fraction':float(grad[blocked].abs().sum()/grad.abs().sum()),
          'blockedGradientL2Fraction':float(grad[blocked].square().sum()/grad.square().sum()),
          'nonzeroGradientEdges':int(torch.count_nonzero(grad))}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--candidate',required=True)
    main(p.parse_args())
