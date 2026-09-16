"""GPU experience worker. One shared parameter version, independent neural states."""
import argparse
import json
from pathlib import Path
import time
import uuid
import os
import numpy as np
import torch
from .plastic_brain import PlasticBrain,SCHEMA,digest,DT
from .haul_world import HaulWorld
from .haul_protocol import Client


def evaluate(brain,client,worker,version,level,kind):
    """Paired, fixed seeds; weights frozen; exploratory noise disabled."""
    brain.reset(); worlds=[HaulWorld(800000+i,level,horizon=600) for i in range(brain.batch)]
    for _ in range(600):
        motors=brain.act([w.sensory() for w in worlds],explore=False)
        for w,m in zip(worlds,motors): w.advance(m)
    report={'schema':SCHEMA,'worker':worker,'version':version,'kind':kind,'level':level,
        'episodes':len(worlds),'horizon':600,'seedsStart':800000,'noise':False,'learning':False,
        'meanReward':float(np.mean([w.total_reward for w in worlds])),
        'products':sum(w.deliveries for w in worlds),'transfers':sum(w.transfers for w in worlds),
        'pickups':sum(w.pickups for w in worlds),'successRate':float(np.mean([w.deliveries>0 for w in worlds])),
        'weightHash':digest(brain.gains),'wallTime':time.time()}
    client.call('/worker/evaluation',report)
    print('FROZEN_EVALUATION '+json.dumps(report),flush=True)
    brain.reset(); return report


def main(args):
    torch.set_num_threads(4)
    client=Client(args.url,(args.root/'worker-token').read_text().strip())
    brain=PlasticBrain(args.root,args.batch,seed={'laptop':1101,'spark1':2202,'spark2':3303}[args.worker])
    info=brain.info
    model,arrays=client.call('/worker/model'); brain.load_gains(arrays['gains'])
    if args.worker=='laptop':
        if model['version']==0: evaluate(brain,client,args.worker,0,model['level'],'initial baseline')
        else:
            learned=brain.gains.copy(); brain.load_gains(np.zeros_like(learned))
            evaluate(brain,client,args.worker,0,model['level'],'initial baseline')
            brain.load_gains(learned)
    seeds=np.random.default_rng({'laptop':101,'spark1':202,'spark2':303}[args.worker])
    worlds=[HaulWorld(int(seeds.integers(1,1_000_000)),model['level']) for _ in range(args.batch)]
    last_eval=time.time(); counter=0; last_audit=None; session=str(uuid.uuid4())
    while True:
        model,arrays=client.call('/worker/model')
        if model['schema']!=SCHEMA or model['mask']!=info['plasticMaskHash']: raise RuntimeError('Wrong shared brain')
        brain.load_gains(arrays['gains'])
        if not model['running']:
            client.call('/worker/heartbeat',{'worker':args.worker,'paused':True})
            time.sleep(1); continue
        version=model['version']; loaded_hash=digest(brain.gains)
        metrics=dict.fromkeys(('transitions','episodes','products','transfers','pickups','rewards'),0)
        begin=time.perf_counter()
        for step in range(args.rollout):
            motors=brain.act([w.sensory() for w in worlds])
            rewards=[]; done=[]
            for i,(world,motor) in enumerate(zip(worlds,motors)):
                products,transfers,pickups=world.deliveries,world.transfers,world.pickups
                reward,terminal=world.advance(motor)
                metrics['products']+=world.deliveries-products
                metrics['transfers']+=world.transfers-transfers
                metrics['pickups']+=world.pickups-pickups
                rewards.append(reward)
                if terminal:
                    metrics['episodes']+=1; metrics['rewards']+=world.total_reward; done.append(i)
            brain.reinforce(rewards)
            if done:
                brain.reset(done)
                for i in done: worlds[i]=HaulWorld(int(seeds.integers(1,1_000_000)),model['level'])
            metrics['transitions']+=args.batch
        duration=time.perf_counter()-begin
        proposal=brain.take_proposal()/args.rollout
        # Keep the rule sensitive without amplifying an all-zero proposal.
        counter+=1
        if counter==1 or counter%50==0: last_audit={**brain.audit(),'version':version}
        snapshot=worlds[0].snapshot(motors[0])
        snapshot.update(simSeconds=worlds[0].steps*DT,tickMilliseconds=duration/args.rollout*1000,
            meanActivity=float(brain.state[:,0].mean().item()),
            dopamine={'modulator':float(brain.last_modulator[0]),'plasticity':'three-factor eligibility',
                      'appetitiveCells':len(brain.t['dan_appetitive']),'aversiveCells':len(brain.t['dan_aversive'])},
            brainView={'nodes':info['nodes'],'edges':[],'activity':brain.state[brain.t['view_indices'],0].cpu().tolist(),'bounds':[]})
        payload={'schema':SCHEMA,'mask':info['plasticMaskHash'],'worker':args.worker,
            'version':version,'updateId':session+':'+str(counter),'loadedHash':loaded_hash,'batch':args.batch,
            'metrics':metrics,'stepsPerSecond':metrics['transitions']/duration,
            'snapshot':snapshot,'audit':last_audit,'device':torch.cuda.get_device_name(),
            'gpuMemoryGB':torch.cuda.memory_allocated()/1e9}
        # Exact same updateId and bytes are retried by Client after a lost response.
        result,_=client.call('/worker/update',payload,delta=proposal.astype(np.float32))
        print(json.dumps({'worker':args.worker,'accepted':result['accepted'],'version':result['version'],
            'actionsPerSecond':round(metrics['transitions']/duration),'proposalL1':float(np.abs(proposal).sum()),
            'products':metrics['products'],'transfers':metrics['transfers'],'pickups':metrics['pickups']}),flush=True)
        if args.max_updates and counter>=args.max_updates: break
        if args.worker=='laptop' and time.time()-last_eval>args.eval_seconds:
            evaluate(brain,client,args.worker,version,model['level'],'frozen learned checkpoint')
            worlds=[HaulWorld(int(seeds.integers(1,1_000_000)),model['level']) for _ in range(args.batch)]
            last_eval=time.time()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',choices=['laptop','spark1','spark2'],required=True)
    p.add_argument('--url',default='http://127.0.0.1:8769'); p.add_argument('--batch',type=int,default=16)
    p.add_argument('--rollout',type=int,default=128); p.add_argument('--eval-seconds',type=int,default=300)
    p.add_argument('--max-updates',type=int,default=0)
    args=p.parse_args()
    # One worker per named machine, even when the recovery launcher is run twice.
    lock=open(args.root/(args.worker+'.lock'),'a+b')
    try:
        if os.name=='nt':
            import msvcrt
            if lock.tell()==0: lock.write(b'0'); lock.flush()
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        print('This named GPU worker is already running.',flush=True)
        raise SystemExit(0)
    main(args)
