"""Live full-connectome inference from a verified, immutable experiment checkpoint.

This is NOT training. It does not load or overwrite the old shared learner.
The laptop runs fresh 40-second worlds at real-time speed; Spark tests are separate.
"""
import argparse
import json
from pathlib import Path
import threading
import time
import numpy as np
import torch
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import JSONResponse,Response
import uvicorn
from .plastic_brain import PlasticBrain,digest,DT
from .haul_world import HaulWorld
from .haul_server import ORIGINS


class Viewer:
    def __init__(self,args):
        torch.set_num_threads(4)
        self.args=args; self.running=True; self.error=None; self.snapshot={}; self.lock=threading.RLock()
        self.brain=PlasticBrain(args.root,batch=1,seed=401)
        with np.load(args.candidate,allow_pickle=False) as f: self.brain.load_gains(f['gains'])
        self.weight_hash=digest(self.brain.gains); self.audit=self.brain.audit()
        self.results=json.loads(args.results.read_text())
        if self.results['protocol']['candidateHash']!=self.weight_hash: raise ValueError('Evaluation/checkpoint mismatch')
        self.history=json.loads(args.history.read_text())
        matches=[r['round'] for r in self.history if r['gainsHash']==self.weight_hash]
        if len(matches)!=1: raise ValueError('Checkpoint must match exactly one recorded training round')
        self.round=matches[0]
        self.worker={'id':'laptop','label':'Laptop · frozen inference','online':True,'accepted':0,
            'contributionL1':0.,'stepsPerSecond':20.,'batch':1,'audit':self.audit}
        self.evaluations=[{'version':0 if k=='original' else self.round,'kind':label,
            **{a:self.results[k][a] for a in ('meanReward','products','pickups','transfers','episodes')},
            'successRate':self.results[k]['productSuccessRate']}
            for k,label in [('original','Original brain'),('learned','Learned brain'),('blank','Learned brain, sensory input removed')]]
        self.study={'selectedRound':self.round,'checkpointHash':self.weight_hash,
            'cases':self.results['learned']['episodes'],
            'pickups':self.results['learned']['pickups'],'transfers':self.results['learned']['transfers'],
            'products':self.results['learned']['products'],
            'discrimination':{k:v for k,v in self.results['pickupDiscrimination'].items() if k!='trials'},
            'history':[{'round':r['round'],'pickup':100*r['nearPickupRate'],'falseAttempt':100*r['farAttemptRate']}
                       for r in self.history]}
        self.thread=threading.Thread(target=self.loop,daemon=True); self.thread.start()

    @torch.inference_mode()
    def loop(self):
        seed=1300000; world=HaulWorld(seed); ticks=episodes=pickups=transfers=products=0
        try:
            while True:
                began=time.perf_counter()
                with self.lock:
                    if self.running:
                        before=(world.pickups,world.transfers,world.deliveries)
                        motor=self.brain.act([world.sensory()],explore=False)[0]
                        _,terminal=world.advance(motor); ticks+=1
                        pickups+=world.pickups-before[0]; transfers+=world.transfers-before[1]; products+=world.deliveries-before[2]
                        info=self.brain.info
                        self.snapshot={**world.snapshot(motor),'steps':ticks,'simSeconds':world.steps*DT,
                            'episodes':episodes,'totalPickups':pickups,'totalTransfers':transfers,'totalProducts':products,
                            'tickMilliseconds':(time.perf_counter()-began)*1000,
                            'brain':{**{k:info[k] for k in ('neurons','edges','sensory','initialWeightHash')},
                                'meanActivity':float(self.brain.state.mean()),'weightChange':float(np.abs(self.brain.gains).mean()),
                                'plasticSynapses':415,'changedSynapses':self.audit['changedSynapses']},
                            'brainView':{'nodes':info['nodes'],'edges':[],'bounds':[],
                                'activity':self.brain.state[self.brain.t['view_indices'],0].cpu().tolist()}}
                        if terminal:
                            episodes+=1; seed+=1; world=HaulWorld(seed); self.brain.reset()
                time.sleep(max(.001,DT-(time.perf_counter()-began)))
        except Exception as error:
            self.error=str(error); self.running=False

    def public(self):
        with self.lock:
            return {**self.snapshot,'experiment':'verified-pickup-v1','mode':'frozen','runId':self.weight_hash,
                'ready':bool(self.snapshot),'running':self.running,'learning':False,'rewardEnabled':False,
                'updates':self.round,'activeWorkers':1,'selectedWorker':'laptop','level':0,
                'workers':[self.worker],'actionsPerSecond':20. if self.running else 0.,
                'device':torch.cuda.get_device_name(),'gpuMemoryGB':torch.cuda.memory_allocated()/1e9,
                'history':[],'evaluations':self.evaluations,'study':self.study,'error':self.error,
                'motorNeurons':self.brain.info['motorNeurons']}


def app_for(viewer):
    app=FastAPI(title='Verified pickup: frozen full-brain inference')
    @app.middleware('http')
    async def protect(request,call_next):
        origin=request.headers.get('origin','')
        if request.url.hostname not in ('localhost','127.0.0.1') or (origin and origin not in ORIGINS):
            return Response(status_code=403)
        response=Response(status_code=204) if request.method=='OPTIONS' else await call_next(request)
        response.headers['Cache-Control']='no-store'
        if origin: response.headers.update({'Access-Control-Allow-Origin':origin,'Vary':'Origin',
            'Access-Control-Allow-Methods':'GET, POST, OPTIONS','Access-Control-Allow-Headers':'Content-Type',
            'Access-Control-Allow-Private-Network':'true'})
        return response
    @app.get('/api/plane/state')
    def state(): return viewer.public()
    @app.get('/api/plane/health')
    def health(): return {'ready':bool(viewer.snapshot),'experiment':'verified-pickup-v1','learning':False,'mode':'frozen'}
    @app.get('/api/plane/audit')
    def audit():
        return {**viewer.audit,'checkpointHash':viewer.weight_hash,'learning':False,
            'workers':{'laptop':{'accepted':0,'contributionL1':0.,'audit':viewer.audit}}}
    @app.post('/api/plane/control')
    async def control(request:Request):
        value=await request.json()
        with viewer.lock:
            if value=={'action':'pause'}: viewer.running=False
            elif value=={'action':'start'}: viewer.running=True
            elif value=={'action':'save'}:
                np.savez_compressed(viewer.args.candidate.with_name('verified-pickup-saved.npz'),gains=viewer.brain.gains)
            elif value!={'action':'watch','worker':'laptop'}: raise HTTPException(400,'Frozen viewer: play, pause, save; laptop only.')
        return {'running':viewer.running,'learning':False,'checkpointHash':viewer.weight_hash}
    return app


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--history',type=Path,required=True)
    p.add_argument('--port',type=int,default=8769);a=p.parse_args()
    v=Viewer(a); print('FROZEN_VIEWER_READY '+v.weight_hash,flush=True)
    uvicorn.run(app_for(v),host='127.0.0.1',port=a.port,access_log=False,log_level='warning')
