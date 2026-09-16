"""One canonical synaptic model; asynchronous local-plasticity worker aggregation.

Listens on laptop loopback. Remote workers use authenticated SSH reverse tunnels.
Checkpoint state is data-only NPZ, saved atomically, never the old actor-critic.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
import numpy as np
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import JSONResponse,Response
from starlette.concurrency import run_in_threadpool
import uvicorn
from .haul_protocol import encode,decode,MAX_MESSAGE_BYTES
from .plastic_brain import SCHEMA,digest

from .deployment import ORIGINS
LABELS={'laptop':'Laptop','spark1':'DGX Spark 1','spark2':'DGX Spark 2'}


class Learner:
    def __init__(self,root,resume=True):
        self.root=Path(root); self.info=json.loads((self.root/'brain-info.json').read_text())
        self.schema=self.info.get('schema',SCHEMA)
        self.token_path=self.root/'worker-token'
        if not self.token_path.exists(): self.token_path.write_text(secrets.token_hex(32))
        self.token=self.token_path.read_text().strip()
        self.lock=threading.RLock(); self.running=True; self.run_id=str(uuid.uuid4())
        self.gains=np.zeros(self.info['plasticSynapses'],np.float32); self.version=0; self.level=0
        self.workers={}; self.seen={}; self.history=deque(maxlen=400); self.evaluations=deque(maxlen=50)
        self.transitions=0; self.episodes=0; self.products=0; self.transfers=0; self.pickups=0; self.rewards=0.
        self.started=time.time(); self.last_history=time.time(); self.last_checkpoint=0.; self.selected='laptop'
        self.initialized_from='original connectome'; self.last_error=None
        if resume and (self.root/'shared-brain.npz').exists():
            meta,arrays=decode((self.root/'shared-brain.npz').read_bytes())
            if meta['schema']!=self.schema or meta['mask']!=self.info['plasticMaskHash']: raise ValueError('Checkpoint schema mismatch')
            self.gains=arrays['gains']; self.version=meta['version']; self.run_id=meta['runId']; self.level=meta['level']
            if self.gains.shape!=(self.info['plasticSynapses'],) or not np.isfinite(self.gains).all(): raise ValueError('Invalid checkpoint weights')
            for k in ('transitions','episodes','products','transfers','pickups','rewards'): setattr(self,k,meta.get(k,0))
            self.history.extend(meta.get('history',[])); self.evaluations.extend(meta.get('evaluations',[]))
            self.seen=meta.get('seen',{}); self.started=time.time()-meta.get('elapsed',0.)
            self.workers=meta.get('workers',{})
            self.initialized_from='new dopamine experiment checkpoint'
        self.save()

    def model_meta(self):
        cache_key=(self.version,id(self.gains))
        if getattr(self,'_hash_key',None)!=cache_key:
            self._weight_hash=digest(self.gains);self._hash_key=cache_key
        return {'schema':self.schema,'mask':self.info['plasticMaskHash'],'version':self.version,'runId':self.run_id,
                'running':self.running,'level':self.level,'weightHash':self._weight_hash}

    def save(self):
        meta={**self.model_meta(),**{k:getattr(self,k) for k in ('transitions','episodes','products','transfers','pickups','rewards')},
              'history':list(self.history),'evaluations':list(self.evaluations),'seen':self.seen,
              'elapsed':time.time()-self.started,
              'workers':{k:{a:b for a,b in w.items() if a!='snapshot'} for k,w in self.workers.items()}}
        raw=encode(meta,gains=self.gains)
        temp=self.root/'shared-brain.tmp'; temp.write_bytes(raw)
        os.replace(temp,self.root/'shared-brain.npz'); self.last_checkpoint=time.time()

    def submit(self,meta,arrays):
        worker=meta.get('worker'); update=meta.get('updateId')
        if worker not in LABELS or not isinstance(update,str) or len(update)>100: raise ValueError('Invalid worker/update')
        if meta.get('mask')!=self.info['plasticMaskHash'] or meta.get('schema')!=self.schema: raise ValueError('Brain mismatch')
        if update in self.seen: return {'accepted':False,'duplicate':True,**self.model_meta()}
        delta=arrays.get('delta')
        if delta is None or delta.shape!=self.gains.shape or delta.dtype!=np.float32 or not np.isfinite(delta).all(): raise ValueError('Invalid delta')
        lag=self.version-int(meta.get('version',-999))
        accepted=self.running and 0<=lag<=12
        old=self.workers.get(worker,{})
        record={**old,'label':LABELS[worker],'lastSeen':time.time(),'device':meta.get('device'),
                'batch':meta.get('batch'),'stepsPerSecond':meta.get('stepsPerSecond',0),'lag':lag,
                'accepted':old.get('accepted',0)+int(accepted),'rejected':old.get('rejected',0)+int(not accepted),
                'loadedVersion':meta.get('version'),'loadedHash':meta.get('loadedHash'),'audit':meta.get('audit') or old.get('audit'),
                'snapshot':meta.get('snapshot',old.get('snapshot')),'gpuMemoryGB':meta.get('gpuMemoryGB',0)}
        self.workers[worker]=record
        if accepted:
            # Freshness weighting + bounded changes, no second optimizer.
            applied=np.clip(delta,-.025,.025)/(1+.25*lag)
            self.gains=np.clip(self.gains+applied,-2,2).astype(np.float32)
            self.version+=1
            for k in ('transitions','episodes','products','transfers','pickups','rewards'):
                setattr(self,k,getattr(self,k)+meta.get('metrics',{}).get(k,0))
            record['contributionL1']=record.get('contributionL1',0.)+float(np.abs(applied).sum())
            self.seen[update]=self.version
            if len(self.seen)>12000: self.seen.pop(next(iter(self.seen)))
        if time.time()-self.last_history>3:
            self.history.append({'seconds':round(time.time()-self.started,1),'reward':self.rewards/max(1,self.episodes),
                'products':self.products,'transfers':self.transfers,'updates':self.version,'level':self.level,
                'actionsPerSecond':sum(w['stepsPerSecond'] for w in self.workers.values() if time.time()-w['lastSeen']<30)})
            self.last_history=time.time()
        if time.time()-self.last_checkpoint>30: self.save()
        return {'accepted':accepted,'lag':lag,**self.model_meta()}

    def public(self):
        now=time.time(); live=[key for key,w in self.workers.items() if now-w['lastSeen']<45]
        watched=self.workers.get(self.selected,{})
        snap=watched.get('snapshot') or next((w.get('snapshot') for w in self.workers.values() if w.get('snapshot')), {})
        return {**snap,'experiment':SCHEMA,'runId':self.run_id,'ready':bool(snap),'running':self.running,
            'learning':True,'rewardEnabled':True,'updates':self.version,'level':self.level,'selectedWorker':self.selected,
            'steps':self.transitions,'simSeconds':snap.get('simSeconds',0),'device':watched.get('device','Waiting for GPUs'),
            'gpuMemoryGB':watched.get('gpuMemoryGB',0),'tickMilliseconds':snap.get('tickMilliseconds',0),
            'brain':{**{k:self.info[k] for k in ('neurons','edges','sensory','initialWeightHash')},
                'plasticSynapses':len(self.gains),'changedSynapses':int(np.count_nonzero(self.gains)),
                'meanActivity':snap.get('meanActivity',0),'weightChange':float(np.abs(self.gains).mean()),
                'backpropagation':False,'decoderTrained':False},
            'history':list(self.history),'evaluations':list(self.evaluations),'totalProducts':self.products,
            'totalTransfers':self.transfers,'totalPickups':self.pickups,'episodes':self.episodes,
            'motorNeurons':self.info['motorNeurons'],'actionsPerSecond':sum(self.workers[k]['stepsPerSecond'] for k in live),
            'workers':[{k:v for k,v in w.items() if k not in ('snapshot','loadedHash')}|{'id':key,'online':key in live,'ageSeconds':now-w['lastSeen']} for key,w in self.workers.items()],
            'activeWorkers':len(live),'error':self.last_error,'checkpointVersion':self.version,'initializedFrom':self.initialized_from}


def create_app(learner):
    app=FastAPI(title='One shared dopamine-plastic brain')
    @app.middleware('http')
    async def protect(request,call_next):
        origin=request.headers.get('origin','')
        worker=request.url.path.startswith('/worker/')
        if worker:
            if origin or not secrets.compare_digest(request.headers.get('authorization',''),'Bearer '+learner.token):
                return JSONResponse({'detail':'Unauthorized worker'},status_code=403)
        elif origin and origin not in ORIGINS:
            return JSONResponse({'detail':'Origin not allowed'},status_code=403)
        if request.url.hostname not in ('127.0.0.1','localhost'): return Response(status_code=403)
        if request.method=='OPTIONS': response=Response(status_code=204)
        else: response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        if origin: response.headers.update({'Access-Control-Allow-Origin':origin,'Vary':'Origin',
            'Access-Control-Allow-Methods':'GET, POST, OPTIONS','Access-Control-Allow-Headers':'Content-Type',
            'Access-Control-Allow-Private-Network':'true'})
        return response
    @app.get('/worker/model')
    def model():
        # Snapshot atomically, but do not hold the dashboard lock while deflating
        # a 102 MB full-connectome vector for each remote worker.
        with learner.lock: meta,gains=learner.model_meta(),learner.gains.copy()
        return Response(encode(meta,gains=gains),media_type='application/octet-stream')
    def accept_update(raw):
        meta,arrays=decode(raw)
        with learner.lock: result=learner.submit(meta,arrays)
        return encode(result)
    @app.post('/worker/update')
    async def update(request:Request):
        if int(request.headers.get('content-length','0'))>MAX_MESSAGE_BYTES: raise HTTPException(413)
        try:
            raw=await request.body()
            return Response(await run_in_threadpool(accept_update,raw),media_type='application/octet-stream')
        except (ValueError,KeyError,TypeError) as e: raise HTTPException(400,str(e))
    @app.post('/worker/evaluation')
    async def evaluation(request:Request):
        meta,_=decode(await request.body())
        if meta.get('worker')!='laptop' or meta.get('schema')!=learner.schema: raise HTTPException(400)
        if getattr(learner,'configuration',{}).get('rule') in ('signed','antithetic','covariance') and (
                meta.get('runId')!=learner.run_id or meta.get('configurationHash')!=learner.configuration_hash):
            raise HTTPException(400,'Evaluation run/configuration mismatch')
        def retain_evaluation():
            with learner.lock:
                learner.evaluations.append(meta)
                learner.save()
        await run_in_threadpool(retain_evaluation)
        return Response(encode({'accepted':True}))
    @app.post('/worker/heartbeat')
    async def heartbeat(request:Request):
        meta,_=decode(await request.body()); key=meta.get('worker')
        if key not in LABELS: raise HTTPException(400)
        with learner.lock:
            if key in learner.workers: learner.workers[key]['lastSeen']=time.time()
        return Response(encode({'ok':True}))
    @app.get('/api/plane/state')
    def state():
        with learner.lock: return learner.public()
    @app.get('/api/plane/health')
    def health(): return {'ready':True,'experiment':SCHEMA,'learning':True,'workers':len(learner.workers)}
    @app.get('/api/plane/audit')
    def audit():
        with learner.lock:
            return {**learner.model_meta(),'plasticSynapses':len(learner.gains),'changedSynapses':int(np.count_nonzero(learner.gains)),
                'finite':bool(np.isfinite(learner.gains).all()),'optimizerPresent':False,'decoderTrained':False,
                'workers':{k:{'accepted':v['accepted'],'contributionL1':v.get('contributionL1',0),'audit':v.get('audit')} for k,v in learner.workers.items()}}
    @app.post('/api/plane/control')
    async def control(request:Request):
        value=await request.json()
        with learner.lock:
            if set(value)=={'action'} and value['action'] in ('start','pause','save'):
                if value['action']!='save': learner.running=value['action']=='start'
                learner.save()
            elif set(value)=={'action','worker'} and value['action']=='watch' and value['worker'] in LABELS:
                learner.selected=value['worker']
            else: raise HTTPException(400,'Allowed: start, pause, save, watch. No destructive reset.')
        return learner.model_meta()
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--port',type=int,default=8769)
    args=parser.parse_args(); learner=Learner(args.root)
    print('SHARED_DOPAMINE_LEARNER_READY',flush=True)
    uvicorn.run(create_app(learner),host='127.0.0.1',port=args.port,access_log=False,log_level='warning')
