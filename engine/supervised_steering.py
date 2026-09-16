"""Supervised steering through the existing full, signed connectome.

Only existing synaptic log gains are optimized. No learned decoder, new edges,
goal input, or teacher action at inference. This is backpropagation, explicitly
NOT dopamine-style biological learning. Old runtime/checkpoints are untouched.
The sparse derivative uses the edge-local outer-product identity; see Flyhard's
published sparse-gradient approach for the motivation, not biological validity.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .haul_world import HaulWorld,wrap
from .plastic_brain import digest


class SynapseMM(torch.autograd.Function):
    """Exact first derivatives; temporary memory O(edge_chunk * batch), not N²."""
    @staticmethod
    def forward(ctx,values,state,crow,col,post,tcrow,tcol,tvalues):
        n=len(crow)-1
        matrix=torch.sparse_csr_tensor(crow,col,values,size=(n,n),check_invariants=False)
        ctx.save_for_backward(state,post,col,tcrow,tcol,tvalues)
        return torch.sparse.mm(matrix,state)

    @staticmethod
    @once_differentiable
    def backward(ctx,adjoint):
        state,post,col,tcrow,tcol,tvalues=ctx.saved_tensors
        edge_gradient=None;state_gradient=None
        if ctx.needs_input_grad[0]:
            edge_gradient=torch.empty(len(col),device=state.device,dtype=state.dtype)
            for first in range(0,len(col),262144):
                part=slice(first,first+262144)
                edge_gradient[part]=(adjoint[post[part]]*state[col[part]]).sum(dim=1)
        if ctx.needs_input_grad[1]:
            n=len(tcrow)-1
            transpose=torch.sparse_csr_tensor(tcrow,tcol,tvalues,size=(n,n),check_invariants=False)
            state_gradient=torch.sparse.mm(transpose,adjoint)
        return edge_gradient,state_gradient,None,None,None,None,None,None


class TrainableConnectome(nn.Module):
    def __init__(self,root,initial,device='cuda'):
        super().__init__();root=Path(root)
        info=json.loads((root/'brain-info.json').read_text())
        with np.load(root/'graph.npz',allow_pickle=False) as graph:
            crow=graph['crow'].copy();col=graph['col'].copy();values=graph['values'].copy()
            self.n=len(graph['body_ids'])
        if digest(values)!=info['initialWeightHash']:raise ValueError('Wrong graph fingerprint')
        if initial.shape!=values.shape or not np.isfinite(initial).all() or np.abs(initial).max()>2.001:
            raise ValueError('Wrong initial gains')
        post=np.repeat(np.arange(self.n),np.diff(crow))
        # The transpose structure/permutation is immutable and computed once.
        order=np.argsort(col,kind='stable')
        transpose_crow=np.r_[np.int64(0),np.cumsum(np.bincount(col,minlength=self.n))]
        for name,value in [('crow',crow),('col',col),('post',post),('base',values),
                ('tcrow',transpose_crow),('tcol',post[order]),('torder',order)]:
            self.register_buffer(name,torch.as_tensor(value.copy(),device=device))
        with np.load(root/'brain-spec.npz',allow_pickle=False) as spec:
            for name in ['sensory_indices','sensory_channels','tonic','fast_mask',
                    'motor_forward','motor_left','motor_right','motor_interact']:
                self.register_buffer(name,torch.as_tensor(spec[name].copy(),device=device))
        self.log_gains=nn.Parameter(torch.as_tensor(initial.copy(),device=device))
        self.input_channels=30
        self.graph_hash=info['initialWeightHash']
        self.fixed_hash=self.fingerprint()

    def fingerprint(self):
        h=hashlib.sha256()
        for name,buffer in self.named_buffers():
            h.update(name.encode());h.update(buffer.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def weights(self):
        values=self.base*torch.exp(self.log_gains)
        with torch.no_grad():transpose_values=values[self.torder]
        return values,transpose_values

    def sensory_drive(self,observations):
        return .5*observations.T[self.sensory_channels]

    def activation(self,current):
        return torch.tanh(torch.relu(current))

    def forward(self,observations,substeps=32,state=None,weights=None):
        if observations.ndim!=2 or observations.shape[1]!=self.input_channels:raise ValueError(f'Expected {self.input_channels} sensory channels')
        if state is None:state=torch.zeros((self.n,len(observations)),device=observations.device)
        drive=torch.zeros_like(state)
        drive[self.sensory_indices]=self.sensory_drive(observations)
        values,transpose_values=self.weights() if weights is None else weights
        for _ in range(substeps):
            signal=SynapseMM.apply(values,state*self.fast_mask,self.crow,self.col,self.post,
                                   self.tcrow,self.tcol,transpose_values)
            target=self.activation(1.5*signal+drive+self.tonic)
            state=.75*state+.25*target
        rates=torch.stack([state[getattr(self,'motor_'+k)].mean(0) for k in ('forward','left','right','interact')],dim=1)
        # The same fixed decoder as the observation/plasticity brain.
        outputs=torch.stack([(80*rates[:,0]).clamp(0,2),
                             (160*(rates[:,2]-rates[:,1])).clamp(-2,2),rates[:,3]],dim=1)
        return outputs,state

    @torch.no_grad()
    def audit(self,initial):
        gains=self.log_gains.cpu().numpy();values=(self.base*torch.exp(self.log_gains)).cpu().numpy()
        return {'neurons':self.n,'eligibleSynapses':len(gains),'changedFromInitial':int(np.count_nonzero(gains!=initial)),
            'finite':bool(np.isfinite(values).all()),'signsPreserved':bool(np.array_equal(np.sign(values),np.sign(self.base.cpu().numpy()))),
            'fixedGraphSensoryDecoderDynamicsUnchanged':self.fingerprint()==self.fixed_hash,
            'backpropagation':True,'decoderTrained':False,'dopamineLearning':False,'gainsHash':digest(gains)}


def landmark_world(seed,bearing=None):
    """Single visible landmark, ordinary retinal encoding; no goal feature."""
    w=HaulWorld(seed,horizon=240);rng=np.random.default_rng(seed)
    angle=float(rng.uniform(.2,2.2)*(1 if seed%2 else -1)) if bearing is None else bearing
    distance=float(rng.uniform(1.5,4.5))
    w.x,w.y,w.heading=10.,7.,0.;w.cargo=1
    for station in w.stations:station['stock']=0;station['timer']=0.
    marker={'x':10+distance*math.cos(angle),'y':7+distance*math.sin(angle),
            'radius':float(rng.uniform(.35,.65)),'kind':'steering landmark','stock':0,'timer':0.}
    w.landmarks=[marker];w.trajectory.clear();w.trajectory.append([w.x,w.y])
    return w,marker


def samples(seed,pairs):
    rng=np.random.default_rng(seed);observations=[];targets=[];bearings=[]
    for _ in range(pairs):
        scene_seed=int(rng.integers(4100000,4200000));angle=float(rng.uniform(.0,2.2))
        # Mirrored pairs share all non-directional features. Balanced batches
        # prevent a constant left/right bias from looking like a learned skill.
        speed=float(rng.uniform(0,.3));turn=float(rng.uniform(0,1.))
        for bearing in (-angle,angle):
            world,_=landmark_world(scene_seed,bearing);world.speed=speed;world.turn=turn
            observations.append(world.sensory());targets.append(np.clip(1.2*bearing,-1,1));bearings.append(bearing)
    return np.asarray(observations,np.float32),np.asarray(targets,np.float32),np.asarray(bearings,np.float32)


@torch.no_grad()
def static_evaluation(model,observations,targets,bearings,batch,substeps):
    outputs=[];weights=model.weights()
    for start in range(0,len(observations),batch):
        outputs.append(model(observations[start:start+batch],substeps,weights=weights)[0].cpu().numpy())
    output=np.concatenate(outputs);expected=targets.cpu().numpy();mask=np.abs(bearings)>.2
    return {'cases':len(output),'directionCases':int(mask.sum()),'meanSquaredTurnError':float(np.mean((output[:,1]-expected)**2)),
            'directionAccuracy':float(np.mean(np.sign(output[mask,1])==np.sign(expected[mask]))),
            'turnRange':float(np.ptp(output[:,1])),'meanSpeed':float(output[:,0].mean()),
            'outputs':output.tolist()}


@torch.no_grad()
def physical_evaluation(model,seeds,batch,horizon=240,blank=False):
    """No teacher, desired turn, optimizer, or exploration in this closed loop."""
    trials=[];weights=model.weights();device=model.log_gains.device
    for offset in range(0,len(seeds),batch):
        group=seeds[offset:offset+batch];scenes=[landmark_world(int(seed)) for seed in group];state=None
        initial=[abs(wrap(math.atan2(m['y']-w.y,m['x']-w.x)-w.heading)) for w,m in scenes]
        last=[]
        for tick in range(horizon):
            observations=np.asarray([w.sensory() for w,_ in scenes])
            if blank:observations[:,:24]=.025
            actions,state=model(torch.as_tensor(observations,device=device),4,state,weights)
            actions=actions.cpu().numpy()
            for (world,_),action in zip(scenes,actions):
                # Pure planar kinematics. No automatic steering/interaction.
                world.heading=wrap(world.heading+float(action[1])*.05)
                world.x+=float(action[0])*math.cos(world.heading)*.05
                world.y+=float(action[0])*math.sin(world.heading)*.05
                world.speed,world.turn=map(float,action[:2])
            if tick>=horizon-20:
                last.append([abs(wrap(math.atan2(m['y']-w.y,m['x']-w.x)-w.heading)) for w,m in scenes])
        for i,seed in enumerate(group):
            worst=float(np.max(np.asarray(last)[:,i]));final=float(last[-1][i])
            trials.append({'seed':int(seed),'initialBearingError':initial[i],'finalBearingError':final,
                           'worstLastSecondError':worst,'passed':worst<=.2})
    return {'cases':len(trials),'horizon':horizon,'seconds':horizon*.05,'teacher':False,'learning':False,
            'noise':False,'blankVision':blank,'successes':sum(t['passed'] for t in trials),
            'meanFinalBearingError':float(np.mean([t['finalBearingError'] for t in trials])),'trials':trials}


def brief(report):return {k:v for k,v in report.items() if k not in ('outputs','trials')}


def run(args):
    if args.out.exists():raise FileExistsError('Use a new directory; preserve previous trials')
    args.out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(args.seed)
    initial=np.load(args.candidate,allow_pickle=False)['gains'].copy()
    model=TrainableConnectome(args.root,initial);device=model.log_gains.device
    protocol={**vars(args),'initialHash':digest(initial),'subtask':'turn toward a single visible landmark',
        'target':'clip(1.2 * landmark bearing, -1, 1), training labels only',
        'input':'original 30 senses; no geometry, desired bearing or action supplied',
        'teacherAtEvaluation':False,'fullLineReliability':False,'dopamineLearning':False,
        'fixedInterfaceHash':model.fixed_hash,'sourceHash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'protocol.json').write_text(json.dumps(protocol,default=str,indent=2))
    x,y,angles=samples(args.seed,1024);vx,vy,vangles=samples(4200123,64)
    x=torch.as_tensor(x,device=device);y=torch.as_tensor(y,device=device)
    vx=torch.as_tensor(vx,device=device);vy=torch.as_tensor(vy,device=device)
    evaluation_steps=args.substeps+args.warmup
    initial_static=static_evaluation(model,vx,vy,vangles,args.batch,evaluation_steps)
    initial_physical=physical_evaluation(model,list(range(4300000,4300032)),args.batch)
    history=[{'update':0,'static':brief(initial_static),'physical':brief(initial_physical)}]
    (args.out/'initial.json').write_text(json.dumps({'static':initial_static,'physical':initial_physical},indent=2))
    print('BASELINE '+json.dumps(history[-1]),flush=True)
    optimizer=torch.optim.Adam([model.log_gains],lr=args.lr,eps=1e-10)
    began=time.perf_counter();torch.cuda.reset_peak_memory_stats();gradient_audit=None;steps=0
    warmup_rng=np.random.default_rng(args.seed+99)
    for step in range(1,args.updates+1):
        if time.perf_counter()-began>args.seconds:break
        pair=torch.randint(0,len(x)//2,(args.batch//2,),device=device)
        indices=torch.stack([2*pair,2*pair+1],dim=1).flatten()
        optimizer.zero_grad(set_to_none=True)
        weights=model.weights();state=None
        if args.warmup:
            # Match sustained activity instead of always training from a silent
            # brain. Gradients are truncated only at this warm-up boundary;
            # deployment retains its original continuously evolving state.
            with torch.no_grad():
                warmup=int(warmup_rng.choice([args.warmup//2,args.warmup,args.warmup*2])) if args.vary_warmup else args.warmup
                _,state=model(x[indices],warmup,weights=weights)
        output,state=model(x[indices],args.substeps,state,weights)
        # The actual action decoder remains clipped and unchanged. Optionally
        # penalize its pre-clamp readout during training so saturated actions
        # still receive a corrective gradient instead of becoming dead zones.
        prediction=160*(state[model.motor_right].mean(0)-state[model.motor_left].mean(0)) if args.raw_loss else output[:,1]
        loss=(prediction-y[indices]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; candidate not promoted')
        loss.backward()
        gradient=model.log_gains.grad
        if not torch.isfinite(gradient).all():raise FloatingPointError('Nonfinite gradient')
        if gradient_audit is None:
            gradient_audit={'eligible':gradient.numel(),'nonzero':int(torch.count_nonzero(gradient)),
                            'maxAbsolute':float(gradient.abs().max()),'norm':float(gradient.norm()),'finite':True}
            print('GRADIENT '+json.dumps(gradient_audit),flush=True)
        torch.nn.utils.clip_grad_norm_([model.log_gains],1.)
        if not args.no_learning:
            optimizer.step()
            with torch.no_grad():model.log_gains.clamp_(-2,2)
        steps=step
        if step==1 or step%10==0:
            print(json.dumps({'update':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-began,
                             'gpuGB':torch.cuda.max_memory_allocated()/1e9}),flush=True)
        if step%args.eval_every==0 or step==args.updates:
            report=static_evaluation(model,vx,vy,vangles,args.batch,evaluation_steps)
            history.append({'update':step,'static':brief(report)})
            gains=model.log_gains.detach().cpu().numpy();np.savez_compressed(args.out/f'candidate-{step}.npz',gains=gains)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
            print('HELD_OUT '+json.dumps(history[-1]),flush=True)
    gains=model.log_gains.detach().cpu().numpy();np.savez_compressed(args.out/f'candidate-final-{steps}.npz',gains=gains)
    torch.save({'gains':model.log_gains.detach().cpu(),'optimizer':optimizer.state_dict(),'steps':steps,
                'rng':torch.get_rng_state(),'cudaRng':torch.cuda.get_rng_state(),'protocol':protocol},args.out/'resume.pt')
    final_static=static_evaluation(model,vx,vy,vangles,args.batch,evaluation_steps)
    final_physical=physical_evaluation(model,list(range(4300000,4300032)),args.batch)
    blank=physical_evaluation(model,list(range(4300000,4300032)),args.batch,blank=True)
    result={'updates':steps,'initial':{'static':initial_static,'physical':initial_physical},
            'final':{'static':final_static,'physical':final_physical,'blankVision':blank},
            'gradientAudit':gradient_audit,'audit':model.audit(initial),'history':history,
            'trainingWallSeconds':time.perf_counter()-began,'peakGPUAllocatedGB':torch.cuda.max_memory_allocated()/1e9,
            'status':'bounded experiment complete; no automatic promotion'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2))
    print('FINAL '+json.dumps({'updates':steps,'static':brief(final_static),'physical':brief(final_physical),
                              'blank':brief(blank),'audit':result['audit']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=4400123);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--updates',type=int,default=300);p.add_argument('--seconds',type=int,default=900)
    p.add_argument('--substeps',type=int,default=32);p.add_argument('--lr',type=float,default=.02)
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--vary-warmup',action='store_true');p.add_argument('--raw-loss',action='store_true')
    p.add_argument('--eval-every',type=int,default=50);p.add_argument('--no-learning',action='store_true')
    args=p.parse_args()
    if args.batch<2 or args.batch%2:raise ValueError('Use an even batch for mirrored pairs')
    run(args)
