"""Continue one shared brain with dopamine-gated locomotor plasticity.

The proven pickup checkpoint is immutable. Carrying trials are a subtask
curriculum, not evidence of end-to-end hauling. The viewer remains frozen.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import uuid
import numpy as np
import torch
import uvicorn
from fastapi import Request,HTTPException
from .plastic_brain import PlasticBrain,SCHEMA,ALL_SYNAPSES_SCHEMA,digest
from .haul_world import HaulWorld,wrap
from .haul_protocol import Client
from .haul_server import Learner,create_app
from .experiment_viewer import Viewer,app_for
from .validate_candidate import haul

LINE_RULES=('signed','antithetic','covariance','coherent','differential')


class CarryBrain(PlasticBrain):
    def configure(self,focus='forward',seed=401):
        names=['forward'] if focus in ('forward','all') else ['forward','left','right']
        self.moving=torch.cat([self.t['motor_'+k] for k in names])
        self.focus=np.ones(len(self.gains),bool) if focus=='all' else np.isin(self.spec['post'],self.moving.cpu().numpy())
        if focus=='all':
            self.local_pre=self.t['pre'];self.local_post=self.t['post']
            self.local_elig=self.eligibility;self.local_proposal=self.proposal
        else:
            index=torch.as_tensor(np.flatnonzero(self.focus),device='cuda')
            self.local_pre=self.t['pre'][index];self.local_post=self.t['post'][index]
            self.local_elig=torch.zeros((len(index),self.batch),device='cuda')
            self.local_proposal=torch.zeros(len(index),device='cuda')
        self.edge_chunk=524288
        self.noise_scale=.025

    @torch.inference_mode()
    def reset(self,indices=None):
        super().reset(indices)
        if hasattr(self,'local_elig'):
            if indices is None:self.local_elig.zero_()
            else:self.local_elig[:,indices]=0

    @torch.inference_mode()
    def act(self,observations,explore=True):
        obs=torch.as_tensor(np.asarray(observations).T,device='cuda',dtype=torch.float32)
        drive=torch.zeros_like(self.state)
        drive[self.t['sensory_indices']]=.5*obs[self.t['sensory_channels']]
        noise=torch.randn((len(self.moving),self.batch),device='cuda',generator=self.generator)*self.noise_scale if explore else 0.
        for _ in range(4):
            current=1.5*torch.sparse.mm(self.wiring,self.state*self.t['fast_mask'])+drive+self.t['tonic']
            current[self.moving]+=noise
            self.state.mul_(.75).add_(torch.tanh(torch.relu(current)),alpha=.25)
        if explore:
            # Chunk temporary gathers; the eligibility trace still covers ALL
            # edges simultaneously, with independent state for every replica.
            for start in range(0,len(self.local_pre),self.edge_chunk):
                s=slice(start,start+self.edge_chunk)
                pair=(self.state[self.local_pre[s]]/.02*self.state[self.local_post[s]]/.03).clamp_(0,5)
                self.local_elig[s].mul_(.9).add_(pair,alpha=.1)
        rates=torch.stack([self.state[self.t['motor_'+k]].mean(0) for k in ('forward','left','right','interact')]).cpu().numpy()
        self.ticks+=1
        return [{'speed':float(np.clip(80*rates[0,i],0,2)),
            'turn':float(np.clip(160*(rates[2,i]-rates[1,i]),-2,2)),
            'interact':bool(rates[3,i]>.025),
            'rates':dict(zip(('forward','left','right','interact'),map(float,rates[:,i])))} for i in range(self.batch)]

    @torch.inference_mode()
    def reinforce(self,rewards,enabled=True):
        if not enabled:return
        r=torch.as_tensor(rewards,device='cuda',dtype=torch.float32).clamp(-1,1)
        increments=[]
        for name,pulse in [('appetitive',torch.relu(r)),('aversive',torch.relu(-r))]:
            indices=self.t['dan_'+name];before=self.state[indices].clone()
            self.state[indices]=(before+.5*pulse[None,:]*(1-before)).clamp(0,1)
            increments.append((self.state[indices]-before).mean(0))
        modulator=(increments[0]-increments[1]).clamp(-1,1)
        for start in range(0,len(self.local_pre),self.edge_chunk):
            s=slice(start,start+self.edge_chunk)
            self.local_proposal[s]+=(self.local_elig[s]*modulator[None,:]).mean(1)*.12
        self.last_modulator=modulator.cpu().numpy()

    def proposal_delta(self,eta):
        delta=np.zeros_like(self.gains)
        delta[self.focus]=self.local_proposal.cpu().numpy()*np.sign(self.spec['base'][self.focus])*eta
        self.local_proposal.zero_()
        return np.clip(delta,-.1,.1).astype(np.float32)


def carry_world(seed,horizon,angle=.35):
    w=HaulWorld(seed,horizon=horizon)
    # Valid post-pickup state for a carrying subtask; no automatic actions occur.
    w.cargo=1;w.stations[0]['stock']-=1
    target=w.stations[1];rng=np.random.default_rng(seed)
    w.x=target['x']-float(rng.uniform(1.45,1.85));w.y=7+float(rng.uniform(-.3,.3))
    w.heading=wrap(math.atan2(target['y']-w.y,target['x']-w.x)+rng.uniform(-angle,angle))
    w.trajectory.clear();w.trajectory.append([w.x,w.y])
    return w


def distance(w):return math.hypot(w.x-w.stations[1]['x'],w.y-w.stations[1]['y'])


def carry_evaluation(brain,start=2400000,cases=32,horizon=160):
    saved=brain.gains.copy();trials=[]
    for offset in range(0,cases,brain.batch):
        worlds=[carry_world(start+offset+i,horizon) for i in range(brain.batch)];brain.reset()
        done=np.zeros(brain.batch,bool);latency=np.full(brain.batch,horizon*.05)
        for tick in range(horizon):
            motors=brain.act([w.sensory() for w in worlds],explore=False)
            for i,(w,m) in enumerate(zip(worlds,motors)):
                if done[i]:continue
                w.advance(m)
                if w.transfers:done[i]=True;latency[i]=(tick+1)*.05
        trials.extend([{'seed':start+offset+i,'delivered':bool(w.transfers),'latency':float(latency[i])}
                       for i,w in enumerate(worlds)])
    assert np.array_equal(saved,brain.gains)
    return {'episodes':len(trials),'successRate':float(np.mean([t['delivered'] for t in trials])),
        'latency':float(np.mean([t['latency'] for t in trials])),'noise':False,'learning':False,
        'weightHash':digest(brain.gains),'trials':trials}


class CarryLearner(Learner):
    def __init__(self,root,brain_root,initial,max_updates=900,all_synapses=False,configuration=None):
        self.configuration=configuration or {'rule':'hebbian','stage':'carry','horizon':160,'eta':.05,'stepLimit':.025}
        self.configuration_hash=digest(np.frombuffer(json.dumps(self.configuration,sort_keys=True).encode(),np.uint8))
        if root.resolve()==brain_root.resolve():raise ValueError('Use a separate training checkpoint directory')
        root.mkdir(parents=True,exist_ok=True)
        self.all_synapses=all_synapses
        if (root/'shared-brain.npz').exists():
            expected_schema=ALL_SYNAPSES_SCHEMA if all_synapses else SCHEMA
            if json.loads((root/'brain-info.json').read_text()).get('schema')!=expected_schema:
                raise ValueError('Existing training directory uses another schema; it was not modified')
            from .haul_protocol import decode
            previous,_=decode((root/'shared-brain.npz').read_bytes())
            if previous.get('configurationHash',self.configuration_hash)!=self.configuration_hash:
                raise ValueError('Existing checkpoint has another learning configuration; use a separate training directory')
        for name in ('brain-info.json','worker-token'):
            if not (root/name).exists():shutil.copy2(brain_root/name,root/name)
        existed=(root/'shared-brain.npz').exists()
        self.max_updates=max_updates
        with np.load(brain_root/'brain-spec.npz') as spec:
            old_edges=spec['plastic_edges']
            self.allowed=np.isin(spec['post'],spec['motor_forward'])
        initial_gains=np.load(initial,allow_pickle=False)['gains'].copy()
        if all_synapses:
            info=json.loads((brain_root/'brain-info.json').read_text())
            edges=np.arange(info['edges'],dtype=np.int64)
            info.update(schema=ALL_SYNAPSES_SCHEMA,plasticSynapses=len(edges),plasticCells=info['neurons'],plasticMaskHash=digest(edges))
            (root/'brain-info.json').write_text(json.dumps(info))
            self.allowed=np.ones(len(edges),bool)
            if initial_gains.shape==(len(edges),):self.initial=initial_gains.copy()
            else:
                self.initial=np.zeros(len(edges),np.float32);self.initial[old_edges]=initial_gains
        else:self.initial=initial_gains
        self.initial_hash=digest(self.initial)
        super().__init__(root)
        if not existed:self.gains=self.initial.copy()
        self.running=self.version<max_updates
        self.save()
    def model_meta(self):
        return {**super().model_meta(),'curriculum':'carry-all-synapses-v2' if self.all_synapses else 'carry-forward-v1',
                'initialHash':self.initial_hash,'focus':'all' if self.all_synapses else 'forward','maxUpdates':self.max_updates,
                'configuration':self.configuration,'configurationHash':self.configuration_hash}
    def submit(self,meta,arrays):
        delta=arrays.get('delta')
        if self.configuration['rule'] in LINE_RULES:
            if meta.get('runId')!=self.run_id or meta.get('configurationHash')!=self.configuration_hash:
                raise ValueError('Run or learning configuration mismatch')
            if delta is not None:arrays={**arrays,'delta':np.clip(delta,-self.configuration['stepLimit'],self.configuration['stepLimit'])}
        if not self.all_synapses and delta is not None and delta.shape==self.gains.shape and np.any(delta[~self.allowed]):
            raise ValueError('Pickup and non-locomotor synapses are protected')
        result=super().submit(meta,arrays)
        if not self.all_synapses:assert np.array_equal(self.gains[~self.allowed],self.initial[~self.allowed])
        if self.version>=self.max_updates:self.running=False;self.save()
        return {**result,**self.model_meta()}


class CarryViewer(Viewer):
    def public(self):
        data=super().public()
        if hasattr(self,'learner'):
            with self.learner.lock:
                if getattr(self,'_stats_version',None)!=self.learner.version:
                    self._stats_version=self.learner.version
                    self._weight_stats={'changedSynapses':int(np.count_nonzero(self.learner.gains)),
                        'meanAbsGain':float(np.abs(self.learner.gains).mean())}
                data['training']={**self.learner.model_meta(),'workers':[
                    {'id':k,'accepted':w.get('accepted',0),'lastSeen':w['lastSeen'],
                     'actionsPerSecond':w.get('stepsPerSecond',0),'device':w.get('device'),
                     'contributionL1':w.get('contributionL1',0),'audit':w.get('audit')}
                    for k,w in self.learner.workers.items()],
                    'episodes':self.learner.episodes,'transitions':self.learner.transitions,
                    'carryDeliveries':self.learner.transfers,'evaluations':list(self.learner.evaluations),
                    'plasticSynapses':int(self.learner.allowed.sum()),
                    **self._weight_stats}
        return data


def server(args):
    configuration={'rule':args.rule,'stage':args.stage,'horizon':args.horizon,'eta':args.eta,
                   'stepLimit':.005 if args.rule in LINE_RULES else .025}
    learner=CarryLearner(args.training_root,args.root,args.initial or args.candidate,args.max_updates,args.all_synapses,configuration)
    viewer=CarryViewer(args);viewer.learner=learner
    app=create_app(learner)
    # Retain authenticated worker endpoints/middleware, replace only viewer APIs.
    app.router.routes=[r for r in app.router.routes if not getattr(r,'path','').startswith('/api/plane/')]
    app.router.routes.extend([r for r in app_for(viewer).router.routes if getattr(r,'path','').startswith('/api/plane/')])
    @app.post('/api/carry/control')
    async def training_control(request:Request):
        value=await request.json()
        with learner.lock:
            if value=={'action':'pause'}:learner.running=False
            elif value=={'action':'start'} and learner.version<learner.max_updates:learner.running=True
            else:raise HTTPException(400,'This bounded stage permits pause/resume before its update limit.')
            learner.save()
            return learner.model_meta()
    print('SHARED_CARRY_TRAINING_READY '+json.dumps(learner.model_meta()),flush=True)
    uvicorn.run(app,host='127.0.0.1',port=8769,access_log=False,log_level='warning')


def worker(args):
    torch.set_num_threads(4)
    lock=open(args.root/('carry-'+args.worker+'.lock'),'a+b')
    try:
        if os.name=='nt':
            import msvcrt
            if lock.tell()==0:lock.write(b'0');lock.flush()
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:raise SystemExit('Carry worker already running')
    seed={'laptop':4404,'spark1':5505,'spark2':6606}[args.worker]
    client=Client(args.url,(args.root/'worker-token').read_text().strip())
    model,arrays=client.call('/worker/model')
    configuration=model.get('configuration',{'rule':'hebbian','stage':'carry','horizon':args.horizon,'eta':args.eta})
    configuration_hash=model.get('configurationHash')
    uses_line_rule=configuration['rule'] in LINE_RULES
    if configuration['rule']=='covariance' and args.batch<2:raise ValueError('Covariance learning requires replicas')
    if uses_line_rule:
        from .line_experiment import SignedBrain,line_world,advance_reward,evaluate_stage
        BrainClass=SignedBrain
    else:BrainClass=CarryBrain
    args.horizon=configuration['horizon'];args.eta=configuration['eta']
    brain=BrainClass(args.root,args.batch,seed,all_synapses=args.all_synapses)
    brain.configure('all' if args.all_synapses else 'forward')
    if uses_line_rule:brain.eligibility_mode=configuration['rule']
    expected='carry-all-synapses-v2' if args.all_synapses else 'carry-forward-v1'
    if model.get('curriculum')!=expected or model['mask']!=brain.info['plasticMaskHash']:raise RuntimeError('Wrong training service')
    brain.load_gains(arrays['gains']);initial=brain.gains.copy()
    rng=np.random.default_rng(seed);session=str(uuid.uuid4());counter=0;last_eval=0
    while True:
        model,arrays=client.call('/worker/model');brain.load_gains(arrays['gains'])
        if model.get('configurationHash')!=configuration_hash:raise RuntimeError('Configuration changed; restart this worker')
        final_evaluation=args.worker=='laptop' and model['version']>=model['maxUpdates']
        if not model['running'] and not final_evaluation:
            client.call('/worker/heartbeat',{'worker':args.worker,'paused':True})
            if model['version']>=model['maxUpdates']:break
            time.sleep(1);continue
        if args.worker=='laptop' and (final_evaluation or counter==0 or time.time()-last_eval>300):
            before=brain.gains.copy()
            carry=carry_evaluation(brain,cases=32)
            complete=haul(brain,np.arange(2500000,2500032),800)
            stage_test=evaluate_stage(brain,configuration['stage'],horizon=args.horizon) if uses_line_rule else None
            checkpoint_dir=args.root/'evaluated'/model['runId'];checkpoint_dir.mkdir(parents=True,exist_ok=True)
            checkpoint_path=checkpoint_dir/('v'+str(model['version'])+'.npz')
            np.savez_compressed(checkpoint_path,gains=before)
            report={'worker':'laptop','schema':brain.info['schema'],'version':model['version'],'kind':'carrying and full-line frozen test',
                'runId':model['runId'],'configurationHash':configuration_hash,
                'carry':carry,'fullLine':complete,'stageTest':stage_test,'configuration':configuration,
                'checkpoint':str(checkpoint_path),'noise':False,'learning':False,'weightHash':digest(before)}
            client.call('/worker/evaluation',report);print('FROZEN_TEST '+json.dumps({
                'version':model['version'],'carrySuccess':carry['successRate'],'pickups':complete['pickups'],
                'transfers':complete['transfers'],'products':complete['products']}),flush=True)
            last_eval=time.time()
            model,arrays=client.call('/worker/model');brain.load_gains(arrays['gains'])
            if not model['running']:
                if model['version']>=model['maxUpdates']:break
                continue
        brain.reset();brain.local_proposal.zero_()
        worlds=[line_world(int(rng.integers(3000000,3100000)),args.horizon,configuration['stage'])
                if uses_line_rule else carry_world(int(rng.integers(1800000,2200000)),args.horizon)
                for _ in range(args.batch)]
        done=np.zeros(args.batch,bool);total=np.zeros(args.batch);began=time.perf_counter()
        for tick in range(args.horizon):
            motors=brain.act([w.sensory() for w in worlds],explore=tick>=8)
            reward=np.zeros(args.batch,np.float32)
            for i,(w,m) in enumerate(zip(worlds,motors)):
                if done[i]:continue
                if uses_line_rule:
                    reward[i],done[i]=advance_reward(w,m);continue
                before=distance(w);w.advance(m)
                # Actual progress only; no target change or terminal/reset reward jump.
                reward[i]=.5*(before-distance(w))-.0001
                if w.transfers:reward[i]+=1.;done[i]=True
            brain.reinforce(reward);total+=reward
            if done.any():brain.local_elig[:,done]=0
        delta=brain.proposal_delta(args.eta);counter+=1;elapsed=time.perf_counter()-began
        meta={'schema':brain.info['schema'],'mask':brain.info['plasticMaskHash'],'worker':args.worker,
            'runId':model['runId'],'configurationHash':configuration_hash,
            'version':model['version'],'updateId':session+':'+str(counter),'loadedHash':digest(brain.gains),
            'batch':args.batch,'stepsPerSecond':args.batch*args.horizon/elapsed,'device':torch.cuda.get_device_name(),
            'metrics':{'transitions':args.batch*args.horizon,'episodes':args.batch,'transfers':sum(w.transfers for w in worlds),
                       'products':sum(w.deliveries for w in worlds),'pickups':sum(w.pickups for w in worlds),'rewards':float(total.sum())},
            'gpuMemoryGB':torch.cuda.max_memory_allocated()/1e9,
            'audit':brain.audit() if counter<=2 or counter%20==0 else None}
        response,_=client.call('/worker/update',meta,delta=delta)
        print(json.dumps({'worker':args.worker,'version':response['version'],'accepted':response['accepted'],
            'noisyTransferRate':float(np.mean([w.transfers>0 for w in worlds])),
            'noisyProductRate':float(np.mean([w.deliveries>0 for w in worlds])),'proposalL1':float(np.abs(delta).sum()),
            'actionsPerSecond':round(args.batch*args.horizon/elapsed)}),flush=True)
    print('BOUNDED_TRAINING_COMPLETE: checkpoint retained; no automatic promotion.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('server');s.add_argument('--root',type=Path,required=True)
    s.add_argument('--training-root',type=Path,required=True);s.add_argument('--candidate',type=Path,required=True)
    s.add_argument('--results',type=Path,required=True);s.add_argument('--history',type=Path,required=True)
    s.add_argument('--max-updates',type=int,default=900)
    s.add_argument('--all-synapses',action='store_true');s.add_argument('--initial',type=Path)
    s.add_argument('--rule',choices=['hebbian',*LINE_RULES],default='hebbian')
    s.add_argument('--stage',choices=['carry','aligned','mixed','wide','full'],default='carry')
    s.add_argument('--horizon',type=int,default=160);s.add_argument('--eta',type=float,default=.05)
    w=sub.add_parser('worker');w.add_argument('--root',type=Path,required=True)
    w.add_argument('--worker',choices=['laptop','spark1','spark2'],required=True)
    w.add_argument('--url',default='http://127.0.0.1:8769');w.add_argument('--batch',type=int,default=16)
    w.add_argument('--horizon',type=int,default=160);w.add_argument('--eta',type=float,default=8.)
    w.add_argument('--all-synapses',action='store_true')
    args=p.parse_args();server(args) if args.command=='server' else worker(args)
