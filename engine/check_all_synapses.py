"""Isolated eligibility/invariant checks; never submits or changes a checkpoint."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .carry_training import CarryBrain,carry_world
from .plastic_brain import digest
from .haul_protocol import encode,decode

def main(args):
    torch.set_num_threads(4)
    started=time.perf_counter()
    brain=CarryBrain(args.root,args.batch,731,all_synapses=True);brain.configure('all')
    old=np.load(args.root/'brain-spec.npz')['plastic_edges']
    seed_gains=np.load(args.candidate,allow_pickle=False)['gains']
    initial=np.zeros_like(brain.gains);initial[old]=seed_gains
    brain.load_gains(initial)
    assert np.array_equal(brain.spec['plastic_edges'],np.arange(brain.info['edges']))
    # Prove the setter reaches every edge, including both ends and outside v1.
    brain.load_gains(initial+.001)
    actual=brain.wiring.values().cpu().numpy()
    assert np.allclose(actual,brain.spec['base']*np.exp(initial+.001),rtol=1e-5)
    assert np.count_nonzero(actual!=brain.spec['base']*np.exp(initial))==len(initial)
    brain.load_gains(initial);brain.reset()
    worlds=[carry_world(2800000+i,160) for i in range(args.batch)]
    for _ in range(12):
        motors=brain.act([w.sensory() for w in worlds],explore=True)
        for w,m in zip(worlds,motors):w.advance(m)
        brain.reinforce(np.zeros(args.batch))
    assert not np.any(brain.proposal_delta(.05)), 'Zero dopamine must not update gains'
    brain.reinforce(np.ones(args.batch)*.1)
    delta=brain.proposal_delta(.05)
    outside=np.ones(len(initial),bool);outside[old]=False
    assert np.count_nonzero(delta[outside])>0
    brain.load_gains(np.clip(initial+delta,-2,2))
    audit=brain.audit()
    assert audit['finite'] and audit['signsPreserved'] and audit['sharedWeightsMatch']
    assert np.array_equal(brain.wiring.crow_indices().cpu().numpy(),np.load(args.root/'graph.npz')['crow'])
    assert np.array_equal(brain.wiring.col_indices().cpu().numpy(),np.load(args.root/'graph.npz')['col'])
    payload=encode({'schema':brain.info['schema']},gains=brain.gains)
    _,decoded=decode(payload)
    assert np.array_equal(decoded['gains'],brain.gains)
    report={**audit,'everyGainIndependentlySettable':True,'zeroDopamineNoUpdate':True,
        'routesPreserved':True,'eligibleOutsideOldMask':int(outside.sum()),
        'dopamineChangedOutsideOldMask':int(np.count_nonzero(delta[outside])),
        'dopamineChangedSynapses':int(np.count_nonzero(delta)),
        'sourceGainsHash':digest(seed_gains),'batch':args.batch,
        'peakGpuGB':torch.cuda.max_memory_allocated()/1e9,
        'seconds':time.perf_counter()-started,'compressedModelBytes':len(payload),
        'note':'Synthetic reward is used only for this isolated test, not submitted to training.'}
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch',type=int,default=16);main(p.parse_args())
