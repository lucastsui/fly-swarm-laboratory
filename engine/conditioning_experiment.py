"""Controlled pickup conditioning. Experimental candidates never overwrite live weights.

Original decoder, sensory projection, topology and rate dynamics are unchanged.
Trials end after 2 s; half start within pickup reach, half away from every station.
Frozen/no-reward controls and fresh, held-out positions separate exploratory motion
from retained, sensory-dependent behavior. No rule selects actions from task labels.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .plastic_brain import PlasticBrain,digest
from .haul_world import HaulWorld


class SubstepBrain(PlasticBrain):
    """Eligibility follows each neural integration substep, including cue onset."""
    @torch.inference_mode()
    def act(self,observations,explore=True):
        obs=torch.as_tensor(np.asarray(observations).T,device='cuda',dtype=torch.float32)
        drive=torch.zeros_like(self.state)
        drive[self.t['sensory_indices']]=.5*obs[self.t['sensory_channels']]
        self.noise=(torch.randn(self.noise.shape,device='cuda',generator=self.generator)*self.sigma
                    if explore else torch.zeros_like(self.noise))
        for _ in range(4):
            pre=self.state[self.t['pre']].clone() if explore else None
            current=1.5*torch.sparse.mm(self.wiring,self.state*self.t['fast_mask'])+drive+self.t['tonic']
            clean=torch.tanh(torch.relu(current[self.t['plastic_rows']]))
            current[self.t['plastic_rows']]+=self.noise
            target=torch.tanh(torch.relu(current))
            if explore:
                fluct=(self.noise if getattr(self,'raw_perturbation',False)
                       else target[self.t['plastic_rows']]-clean)
                pair=pre.clamp(0,.5)/.02 * fluct[self.t['post_rank']]/.03
                self.eligibility.mul_(.9875).add_(pair,alpha=.0125).clamp_(-5,5)
            self.state.mul_(.75).add_(target,alpha=.25)
        rates=torch.stack([self.state[self.t['motor_'+k]].mean(0) for k in ('forward','left','right','interact')]).cpu().numpy()
        self.ticks+=1
        return [{'speed':float(np.clip(80*rates[0,i],0,2)),
                 'turn':float(np.clip(160*(rates[2,i]-rates[1,i]),-2,2)),
                 'interact':bool(rates[3,i]>.025),
                 'rates':dict(zip(('forward','left','right','interact'),map(float,rates[:,i])))} for i in range(self.batch)]


def worlds_for(seeds,near_flags,horizon):
    worlds=[]
    for seed,near in zip(seeds,near_flags):
        w=HaulWorld(int(seed),horizon=horizon)
        if not near:
            rng=np.random.default_rng(int(seed))
            w.x=float(rng.uniform(1.,2.)); w.y=float(rng.uniform(11.,12.))
            w.trajectory.clear(); w.trajectory.append([w.x,w.y])
        worlds.append(w)
    return worlds


@torch.inference_mode()
def evaluate(brain,horizon,start=510000,cases=64,blank=False):
    results=[]
    initial=brain.gains.copy()
    for offset in range(0,cases,brain.batch):
        seeds=np.arange(start+offset,start+offset+brain.batch)
        near=np.arange(brain.batch)%2==0
        worlds=worlds_for(seeds,near,horizon); brain.reset()
        completed=np.zeros(brain.batch,bool); latencies=np.full(brain.batch,horizon*.05)
        for tick in range(horizon):
            obs=[np.zeros(30,np.float32) if blank else w.sensory() for w in worlds]
            motors=brain.act(obs,explore=False)
            for i,(w,m) in enumerate(zip(worlds,motors)):
                if completed[i]: continue
                w.advance(m)
                if w.pickups:
                    completed[i]=True; latencies[i]=(tick+1)*.05
        results.extend([{'seed':int(seed),'near':bool(n),'picked':bool(w.pickups),'attempts':w.interactions,
                         'latency':float(latency)} for seed,n,w,latency in zip(seeds,near,worlds,latencies)])
    assert np.array_equal(initial,brain.gains)
    near=[r for r in results if r['near']]; far=[r for r in results if not r['near']]
    return {'cases':len(results),'nearPickupRate':float(np.mean([r['picked'] for r in near])),
            'farAttemptRate':float(np.mean([r['attempts']>0 for r in far])),
            'nearMeanLatency':float(np.mean([r['latency'] for r in near])),
            'blankSensory':blank,'noise':False,'weightsChangedDuringEvaluation':False,
            'gainsHash':digest(brain.gains),'trials':results}


def train(args):
    torch.set_num_threads(4)
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    brain=(SubstepBrain if args.variant in ('substep','impulse','raw_impulse','hebb') else PlasticBrain)(args.root,args.batch,args.seed)
    brain.raw_perturbation=args.variant=='raw_impulse'
    brain.sigma*=args.noise_scale
    rng=np.random.default_rng(args.seed)
    if args.initial:
        with np.load(args.initial,allow_pickle=False) as f: brain.load_gains(f['gains'])
    sign=np.sign(brain.spec['base'])
    focus=(np.isin(brain.spec['post'],brain.spec['motor_interact']) if args.focus=='interact'
           else np.ones(len(sign),bool))
    # Parameters and evaluation seed splits are recorded before collecting data.
    protocol={'variant':args.variant,'seed':args.seed,'rounds':args.rounds,'batch':args.batch,'horizon':args.horizon,
              'rewardPickup':1.,'rewardFarAttempt':-args.far_penalty,'trainingSeeds':'0..499999','noiseScale':args.noise_scale,
              'validationSeeds':'510000..510063','heldOutSeeds':'960000..960255',
              'eta':args.eta,'signedGainUpdate':args.variant!='original','decoderFixed':True,'initial':str(args.initial),
              'explorationStartsAfterTicks':args.explore_after,'plasticityFocus':args.focus,
              'eligibleEdgesThisAssay':int(focus.sum()),'rewardDisabled':args.disable_reward}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    baseline=evaluate(brain,args.horizon)
    history=[{'round':0,**{k:v for k,v in baseline.items() if k!='trials'}}]
    print('BASELINE '+json.dumps(history[-1]),flush=True)
    reward_average=0.
    start=time.time()
    for trial in range(1,args.rounds+1):
        brain.reset(); brain.proposal.zero_()
        seeds=rng.integers(0,500000,args.batch); near=np.arange(args.batch)%2==0
        worlds=worlds_for(seeds,near,args.horizon)
        done=np.zeros(args.batch,bool); outcomes=np.zeros(args.batch,np.float32)
        for tick in range(args.horizon):
            motors=brain.act([w.sensory() for w in worlds],explore=tick>=args.explore_after)
            rewards=np.zeros(args.batch,np.float32)
            for i,(w,m) in enumerate(zip(worlds,motors)):
                if done[i]: continue
                previous=w.interactions; w.advance(m)
                if w.pickups:
                    rewards[i]=1.; outcomes[i]=1.; done[i]=True
                elif not near[i] and w.interactions>previous:
                    rewards[i]=-args.far_penalty; outcomes[i]-=args.far_penalty
            if args.variant=='centered':
                # Causal expected-reward baseline; labels never enter the brain.
                active=~done | (rewards!=0)
                rewards[active]-=reward_average
                reward_average=.99*reward_average+.01*float(rewards.mean()+reward_average)
            if args.variant=='hebb':
                # Reward-gated local coactivity: same eligible synapses and fixed decoder.
                brain.eligibility.copy_((brain.state[brain.t['pre']]/.02 *
                    brain.state[brain.t['post']]/.03).clamp(0,5))
            brain.reinforce(rewards,enabled=args.variant!='no_reward' and not args.disable_reward)
            if args.variant in ('impulse','raw_impulse','hebb'):
                # Symmetric dopamine impulse, no unequal negative afterglow in longer trials.
                brain.evoked.zero_()
            if done.any():
                brain.eligibility[:,done]=0; brain.evoked[:,done]=0
        delta=brain.take_proposal()
        if args.variant=='original': delta/=args.horizon
        else: delta*=sign*args.eta
        delta[~focus]=0
        brain.load_gains(np.clip(brain.gains+np.clip(delta,-.1,.1),-2,2).astype(np.float32))
        if trial%args.check_every==0 or trial==args.rounds:
            report=evaluate(brain,args.horizon)
            record={'round':trial,'wallSeconds':time.time()-start,'noisySuccessRate':float(np.mean(outcomes[near]>0)),
                    'meanAbsGain':float(np.abs(brain.gains).mean()),'maxAbsGain':float(np.abs(brain.gains).max()),
                    **{k:v for k,v in report.items() if k!='trials'}}
            history.append(record); print('VALIDATION '+json.dumps(record),flush=True)
            np.savez_compressed(out/'candidate.npz',gains=brain.gains)
            (out/'history.json').write_text(json.dumps(history,indent=2))
    held=evaluate(brain,args.horizon,960000,cases=256)
    blank=evaluate(brain,args.horizon,970000,cases=64,blank=True)
    audit=brain.audit()
    result={'protocol':protocol,'baseline':baseline,'history':history,'heldOut':held,'blankControl':blank,'audit':audit}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('RESULT '+json.dumps({k:v for k,v in held.items() if k!='trials'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=['original','signed','centered','no_reward','substep','impulse','raw_impulse','hebb'],default='signed')
    p.add_argument('--seed',type=int,default=101); p.add_argument('--batch',type=int,default=16)
    p.add_argument('--rounds',type=int,default=100); p.add_argument('--horizon',type=int,default=40)
    p.add_argument('--eta',type=float,default=16.); p.add_argument('--check-every',type=int,default=20)
    p.add_argument('--far-penalty',type=float,default=.5); p.add_argument('--noise-scale',type=float,default=1.)
    p.add_argument('--explore-after',type=int,default=0)
    p.add_argument('--focus',choices=['all','interact'],default='all')
    p.add_argument('--disable-reward',action='store_true')
    p.add_argument('--initial',type=Path)
    train(p.parse_args())
