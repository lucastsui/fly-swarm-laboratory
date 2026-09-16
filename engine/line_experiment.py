"""All-edge signed three-factor plasticity and physical full-line curricula.

No target action is generated: only ordinary sensory observations reach the brain.
Goal distance is used by the experimenter to assign reinforcement, never as input.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .carry_training import CarryBrain,carry_world,carry_evaluation
from .haul_world import HaulWorld,wrap
from .validate_candidate import haul
from .plastic_brain import digest

def odd_perturbation(current,noise,positive=None):
    """Local antithetic response, odd in noise even at rectifier thresholds."""
    if positive is None:positive=torch.tanh(torch.relu(current+noise))
    return .5*(positive-torch.tanh(torch.relu(current-noise)))

def presynaptic_contrast(activity):
    """Training-only baseline/scale for the SAME neuron across replicas.

    No observation, goal or action is added to inference. This statistical
    shared-training normalization is engineered, not biological circuitry.
    """
    if activity.shape[1]<2:raise ValueError('Covariance learning requires at least two replicas')
    contrast=activity-activity.mean(1,keepdim=True)
    scale=contrast.square().mean(1,keepdim=True).sqrt().clamp_min(.0005)
    return (contrast/scale).clamp(-3,3)


class RewardBaseline:
    """Causal per-replica reward expectation; never observes goals or actions.

    Read the old expectation before incorporating the current reward. This is
    an engineered reward-prediction error, not a reconstructed dopamine model.
    """
    def __init__(self,batch,alpha=.025):
        if not 0<alpha<=1:raise ValueError('Invalid baseline rate')
        self.mean=np.zeros(batch,np.float32);self.alpha=alpha

    def reset(self,indices=None):
        if indices is None:self.mean.fill(0)
        else:self.mean[indices]=0

    def advance(self,rewards):
        rewards=np.asarray(rewards,dtype=np.float32)
        if rewards.shape!=self.mean.shape or not np.isfinite(rewards).all():
            raise ValueError('Invalid rewards')
        prediction_error=rewards-self.mean
        self.mean+=self.alpha*prediction_error
        return prediction_error

class SignedBrain(CarryBrain):
    def configure(self,focus='all',seed=401):
        if focus!='all':raise ValueError('Signed experiments require the full graph')
        super().configure('all',seed)
        self.exploration_sigma=torch.full((self.n,1),.0005,device='cuda')
        for key in ('forward','left','right','interact'):
            self.exploration_sigma[self.t['motor_'+key]]=.02 if key=='forward' else .008
        self.reward_baseline=RewardBaseline(self.batch)
        self.held_noise=torch.zeros_like(self.state);self.exploration_age=0

    @torch.inference_mode()
    def reset(self,indices=None):
        super().reset(indices)
        if hasattr(self,'reward_baseline'):
            self.reward_baseline.reset(indices)
            if indices is None:self.held_noise.zero_();self.exploration_age=0
            else:self.held_noise[:,indices]=0

    @torch.inference_mode()
    def reinforce(self,rewards,enabled=True):
        if enabled and getattr(self,'eligibility_mode','signed') in ('differential','coherent'):
            # Subtract expected reinforcement BEFORE stimulating real DAN cells.
            # Both signs pass through the existing measured PAM/PPL response.
            if self.eligibility_mode=='differential':
                rewards=self.reward_baseline.advance(rewards)
        super().reinforce(rewards,enabled)

    def exploration(self,mode):
        if mode not in ('differential','coherent'):
            return torch.randn(self.state.shape,device=self.state.device,generator=self.generator)*self.exploration_sigma
        # Coherent half-second neural perturbations allow physical turning to
        # accumulate. This adds no direction labels and changes no frozen policy.
        if self.exploration_age%10==0:
            self.held_noise=torch.randn(self.state.shape,device=self.state.device,generator=self.generator)*self.exploration_sigma
            self.held_noise[self.t['motor_forward']]*=.125
            self.held_noise[self.t['motor_interact']]*=.125
        self.exploration_age+=1
        return self.held_noise

    @torch.inference_mode()
    def act(self,observations,explore=True):
        obs=torch.as_tensor(np.asarray(observations).T,device='cuda',dtype=torch.float32)
        drive=torch.zeros_like(self.state)
        drive[self.t['sensory_indices']]=.5*obs[self.t['sensory_channels']]
        mode=getattr(self,'eligibility_mode','signed')
        centered=mode in ('antithetic','covariance','differential','coherent')
        pre_state=self.state.clone() if explore and centered else None
        if explore and mode=='covariance':pre_state=presynaptic_contrast(pre_state)
        noise=self.exploration(mode) if explore else 0.
        fluct=torch.zeros_like(self.state)
        for _ in range(4):
            current=1.5*torch.sparse.mm(self.wiring,self.state*self.t['fast_mask'])+drive+self.t['tonic']
            clean=torch.tanh(torch.relu(current))
            target=torch.tanh(torch.relu(current+noise)) if explore else clean
            if explore:
                response=odd_perturbation(current,noise,target) if centered else target-clean
                fluct.add_(response,alpha=.25)
            self.state.mul_(.75).add_(target,alpha=.25)
        if explore:
            fluct.div_(self.exploration_sigma)
            if mode in ('differential','coherent'):
                fluct[self.t['motor_forward']]*=8
                fluct[self.t['motor_interact']]*=8
            for start in range(0,len(self.local_pre),self.edge_chunk):
                s=slice(start,start+self.edge_chunk)
                pre=pre_state if centered else self.state
                factor=pre[self.local_pre[s]] if mode=='covariance' else pre[self.local_pre[s]].clamp(0,.5)/.02
                pair=(factor*fluct[self.local_post[s]]).clamp_(-5,5)
                self.local_elig[s].mul_(.95).add_(pair,alpha=.05)
        rates=torch.stack([self.state[self.t['motor_'+k]].mean(0) for k in ('forward','left','right','interact')]).cpu().numpy()
        self.ticks+=1
        return [{'speed':float(np.clip(80*rates[0,i],0,2)),
            'turn':float(np.clip(160*(rates[2,i]-rates[1,i]),-2,2)),
            'interact':bool(rates[3,i]>.025),
            'rates':dict(zip(('forward','left','right','interact'),map(float,rates[:,i])))} for i in range(self.batch)]

def line_world(seed,horizon,stage='aligned',level=0):
    w=HaulWorld(seed,level=level,horizon=horizon)
    if stage=='carry':return carry_world(seed,horizon)
    if stage=='mixed':
        if w.rng.random()<.25:return line_world(seed,horizon,'aligned',level)
        # Valid post-processing INITIAL states expose every transport leg.
        # They are curriculum trials, never counted as full-line test success.
        leg=int(w.rng.integers(1,4));w.cargo=leg;w.stations[0]['stock']=2
        target=w.stations[leg]
        w.x=target['x']-float(w.rng.uniform(1.45,1.85));w.y=7+float(w.rng.uniform(-.3,.3))
        w.heading=wrap(math.atan2(target['y']-w.y,target['x']-w.x)+w.rng.uniform(-.35,.35))
        w.trajectory.clear();w.trajectory.append([w.x,w.y])
    if stage in ('aligned','wide'):
        # Curriculum INITIAL CONDITIONS only. No per-step steering or teleporting.
        w.x=w.stations[0]['x']+float(w.rng.uniform(-.3,.3))
        w.y=7+float(w.rng.uniform(-.2,.2))
        w.heading=float(w.rng.uniform(-.2,.2) if stage=='aligned' else w.rng.uniform(-1.2,1.2))
        w.trajectory.clear();w.trajectory.append([w.x,w.y])
    return w

def target_position(w):
    if w.cargo:return w.stations[w.cargo]['x'],w.stations[w.cargo]['y']
    choices=[s for s in w.stations[:3] if s['stock']>0 or s['timer']>0]
    s=min(choices,key=lambda s:math.hypot(w.x-s['x'],w.y-s['y'])) if choices else w.stations[0]
    return s['x'],s['y']

def advance_reward(w,motor):
    tx,ty=target_position(w);before=math.hypot(w.x-tx,w.y-ty)
    pickups,transfers,products=w.pickups,w.transfers,w.deliveries
    _,done=w.advance(motor)
    # Hold the target fixed across this transition: no reward from goal switches
    # or terminal potential resets. Station events are counted only when real.
    reward=.08*(before-math.hypot(w.x-tx,w.y-ty))-.0002
    reward+=.1*(w.pickups-pickups)+.4*(w.transfers-transfers)+1.6*(w.deliveries-products)
    if w.contact:reward-=.002
    return reward,done

def load_candidate(brain,path):
    values=np.load(path,allow_pickle=False)['gains']
    if values.shape==brain.gains.shape:return values.copy()
    original=np.load(brain.root/'brain-spec.npz',allow_pickle=False)['plastic_edges']
    gains=np.zeros_like(brain.gains);gains[original]=values
    return gains

def evaluate_stage(brain,stage,cases=32,horizon=400,start=3200000):
    saved=brain.gains.copy();counts=[]
    for offset in range(0,cases,brain.batch):
        worlds=[line_world(start+offset+i,horizon,stage) for i in range(brain.batch)];brain.reset()
        for _ in range(horizon):
            motors=brain.act([w.sensory() for w in worlds],explore=False)
            for w,m in zip(worlds,motors):w.advance(m)
        counts.extend([{'pickups':w.pickups,'transfers':w.transfers,'products':w.deliveries} for w in worlds])
    assert np.array_equal(saved,brain.gains)
    return {'stage':stage,'episodes':cases,'horizon':horizon,'noise':False,'learning':False,
        'productSuccessRate':float(np.mean([v['products']>0 for v in counts])),
        **{k:sum(v[k] for v in counts) for k in ('pickups','transfers','products')}}

def experiment(args):
    if (args.out/'history.json').exists():raise FileExistsError('Retain evaluated candidates; use a new experiment directory')
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=True)
    brain=SignedBrain(args.root,args.batch,871,all_synapses=True);brain.configure('all')
    brain.eligibility_mode=args.eligibility
    brain.load_gains(load_candidate(brain,args.candidate));initial=brain.gains.copy()
    (args.out/'protocol.json').write_text(json.dumps({**vars(args),'initialHash':digest(initial)},default=str,indent=2))
    history=[];rng=np.random.default_rng(7301)
    for round_number in range(args.rounds+1):
        if round_number%args.eval_every==0 or round_number==args.rounds:
            report=evaluate_stage(brain,args.stage,cases=32,horizon=args.horizon)
            report['fullLine']=haul(brain,np.arange(2500000,2500032),800)
            report.update(round=round_number,weightHash=digest(brain.gains))
            history.append(report)
            # Keep actual weights for every evaluated candidate, not only hashes.
            np.savez_compressed(args.out/f'candidate-{round_number}.npz',gains=brain.gains)
            (args.out/'history.json').write_text(json.dumps(history,indent=2))
            print('FROZEN_STAGE '+json.dumps({k:v for k,v in report.items() if k!='fullLine'} |
                {'fullProducts':report['fullLine']['products'],'fullTransfers':report['fullLine']['transfers']}),flush=True)
        if round_number==args.rounds:break
        brain.reset();brain.local_proposal.zero_()
        worlds=[line_world(int(rng.integers(3000000,3100000)),args.horizon,args.stage) for _ in range(args.batch)]
        finished=np.zeros(args.batch,bool);started=time.perf_counter()
        for tick in range(args.horizon):
            motors=brain.act([w.sensory() for w in worlds],explore=tick>=8)
            rewards=np.zeros(args.batch,np.float32)
            for i,(w,m) in enumerate(zip(worlds,motors)):
                if not finished[i]:rewards[i],finished[i]=advance_reward(w,m)
            brain.reinforce(rewards,enabled=not args.no_learning)
            if finished.any():brain.local_elig[:,finished]=0
        delta=brain.proposal_delta(args.eta)
        brain.load_gains(np.clip(brain.gains+np.clip(delta,-.005,.005),-2,2))
        print(json.dumps({'round':round_number+1,'proposalL1':float(np.abs(delta).sum()),
            'motorProposalMeanAbs':{k:float(np.abs(delta[np.isin(brain.spec['post'],brain.t['motor_'+k].cpu().numpy())]).mean())
                for k in ('forward','left','right','interact')},
            'noisyProducts':sum(w.deliveries for w in worlds),'noisyTransfers':sum(w.transfers for w in worlds),
            'changed':int(np.count_nonzero(brain.gains!=initial)),'seconds':time.perf_counter()-started}),flush=True)
    (args.out/'audit.json').write_text(json.dumps(brain.audit(),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--rounds',type=int,default=12);p.add_argument('--eval-every',type=int,default=4)
    p.add_argument('--eta',type=float,default=1.);p.add_argument('--horizon',type=int,default=400)
    p.add_argument('--eligibility',choices=['signed','antithetic','covariance','coherent','differential'],default='signed')
    p.add_argument('--stage',choices=['carry','aligned','mixed','wide','full'],default='aligned')
    p.add_argument('--no-learning',action='store_true');experiment(p.parse_args())
