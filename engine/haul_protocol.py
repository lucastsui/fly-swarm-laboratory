"""Authenticated, bounded NumPy messages. No pickle or executable payloads."""
import io
import json
import time
import urllib.request
import urllib.error
import numpy as np

# A full 25,582,938-edge float32 vector is 102,331,752 bytes.
# Messages remain authenticated, data-only, bounded, and never use pickle.
MAX_MESSAGE_BYTES=128_000_000
MAX_ARRAY_BYTES=110_000_000


def encode(meta,**arrays):
    out=io.BytesIO()
    np.savez_compressed(out,meta=np.frombuffer(json.dumps(meta,allow_nan=False).encode(),np.uint8),**arrays)
    return out.getvalue()


def decode(raw):
    if len(raw)>MAX_MESSAGE_BYTES: raise ValueError('Message too large')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        values={};total=0
        for key in archive.files:
            value=archive[key];total+=value.nbytes
            if value.nbytes>MAX_ARRAY_BYTES or total>MAX_MESSAGE_BYTES: raise ValueError('Array too large')
            values[key]=value
        meta=json.loads(values.pop('meta').tobytes().decode())
        arrays=values
    return meta,arrays


class Client:
    def __init__(self,url,token):
        self.url=url.rstrip('/'); self.token=token
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(self,path,meta=None,**arrays):
        payload=None if meta is None else encode(meta,**arrays)
        attempt=0
        while True:
            try:
                req=urllib.request.Request(self.url+path,data=payload,headers={'Authorization':'Bearer '+self.token,'Content-Type':'application/octet-stream'})
                with self.opener.open(req,timeout=180) as response: return decode(response.read(MAX_MESSAGE_BYTES+1))
            except urllib.error.HTTPError: raise
            except (OSError,TimeoutError):
                attempt+=1
                if attempt==1 or attempt%12==0: print('Shared learner unavailable; retaining the exact pending update for retry.',flush=True)
                time.sleep(min(15.,.5*attempt))
