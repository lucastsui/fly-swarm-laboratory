import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import VERSION, operation_specs, parent_motion, CONTEXTS, KINDS
from .layout_operation_grip_train import operation_heads, train, fit
from .layout_recovery_world import INTERFACE, PHYSICS
from .layout_recovery_teacher import LABEL_VERSION
from .layout_recovery_demonstrations import file_hash
from .test_layout_operation_prefix import operations
from .test_layout_microfit_flow import TinyTrainBrain


class OperationTrainingTests(unittest.TestCase):
    def make_cache(self, folder, brain):
        e=operations()
        for index,(a,m) in enumerate(e): a['observations'][...,:4]=1+index*.01
        specs=operation_specs(e); x=np.stack([e[s['episode']][0]['observations'][s['start']:s['stop'],s['fly']:s['fly']+1] for s in specs])
        y=np.stack([e[s['episode']][0]['labels'][s['start']:s['stop'],s['fly']:s['fly']+1] for s in specs])
        states=np.zeros((len(specs),4,1),np.float32)
        motion=parent_motion(brain,x,states)
        np.savez_compressed(folder/'cache.npz',observations=x,labels=y,states=states,parent_motion=motion)
        data={'episodes':[],'parameterHash':brain.checkpoint_hash(),'fixedHash':brain.fixed_hash,
              'canonicalRun':'fixture','canonicalVersion':60}
        dataset=folder/'dataset'; dataset.mkdir(); (dataset/'manifest.json').write_text(json.dumps(data))
        names=('layout_operation_prefix.py','layout_correction_prefix.py','supervised_joint.py')
        m=dict(schema=VERSION,parameterHash=brain.checkpoint_hash(),fixedHash=brain.fixed_hash,
               finished=True,parametersUnchanged=True,interface=INTERFACE,physics=PHYSICS,
               control='learner-only',teacherActions=False,optimizerUsed=False,labelsUnchanged=True,
               labelVersion=LABEL_VERSION,isServiceEvidence=False,burn=64,lossFrames=32,fullHistoryPrefix=True,
               sourceHashes={n:file_hash(Path(__file__).parent/n) for n in names},
               datasetManifestHash=file_hash(dataset/'manifest.json'),datasetFiles=[],
               behaviorParameterHash=brain.checkpoint_hash(),behaviorCanonicalRun='fixture',behaviorCanonicalVersion=60,
               windows=specs,cacheFileHash=file_hash(folder/'cache.npz'))
        (folder/'manifest.json').write_text(json.dumps(m))
        return e,data,dataset,m

    def test_exact_loader_actor_contexts_and_real_optimizer_steps(self):
        b=TinyTrainBrain().eval(); before=b.checkpoint_hash()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); e,data,dataset,m=self.make_cache(folder,b)
            with patch('engine.layout_operation_sampling.load_histories',return_value=(e,data)):
                bank=OperationCache(folder,dataset,before,b.fixed_hash,neurons=4)
            initial=fit(b,bank); self.assertLess(max(initial['parentMotionMSE']),1e-10)
            rng=np.random.default_rng(24); x,y,state,motion,selection=bank.sample(rng,'cpu')
            self.assertEqual(x.shape,(96,16,297)); self.assertEqual(motion.shape,(32,16,2))
            self.assertEqual({(s['kind'],s['category']) for s in selection['selections']}, {(k,c) for k in KINDS for c in CONTEXTS})
            opt=torch.optim.Adam([{'params':[b.log_gains],'lr':.0003},{'params':[b.tonic],'lr':.000001}])
            saved=[]
            with contextlib.redirect_stdout(io.StringIO()): rows=train(b,bank,opt,rng,4,lambda u,r:saved.append(u))
            self.assertEqual([r['update'] for r in rows],[61,62,63,64]); self.assertEqual(saved,[64])
            self.assertTrue(all(np.isfinite(r['lossBeforeUpdate']) for r in rows))
            self.assertNotEqual(b.checkpoint_hash(),before)

    def test_rejects_rehashed_altered_labels_observations_and_incompatible_cache(self):
        for case in ('labels','observations','finished','teacherActions','parameterHash','source'):
            b=TinyTrainBrain().eval()
            with tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp); e,data,dataset,m=self.make_cache(folder,b)
                if case in ('labels','observations'):
                    with np.load(folder/'cache.npz') as z: arrays={k:z[k].copy() for k in z.files}
                    arrays[case][0,64,0,0]+=.5; np.savez_compressed(folder/'cache.npz',**arrays)
                    m['cacheFileHash']=file_hash(folder/'cache.npz')
                elif case=='source': m['sourceHashes']['layout_operation_prefix.py']='wrong'
                else: m[case]={'finished':False,'teacherActions':True,'parameterHash':'other'}[case]
                (folder/'manifest.json').write_text(json.dumps(m))
                with patch('engine.layout_operation_sampling.load_histories',return_value=(e,data)):
                    with self.assertRaises(ValueError,msg=case): OperationCache(folder,dataset,b.checkpoint_hash(),b.fixed_hash,4)

    def test_threshold_loss_keeps_correct_actions_and_anchors_motion(self):
        y=torch.tensor([[[.1,.2,2.2],[.2,.3,.2]]]); p=torch.tensor([[[.1,.2,1.2],[.2,.3,.8]]],requires_grad=True)
        anchor=p[...,:2].detach().clone()
        torch.testing.assert_close(operation_heads(p,y,anchor),torch.zeros(3))
        wrong=p.detach().clone(); wrong[...,2]=1.5; wrong.requires_grad_(True)
        loss=operation_heads(wrong,y,anchor).sum(); self.assertGreater(float(loss.detach()),0)
        loss.backward(); self.assertGreater(float(wrong.grad[0,1,2]),0)
        self.assertEqual(float(wrong.grad[0,0,2]),0)
        with self.assertRaises(ValueError): operation_heads(p,y,anchor[...,:1])


if __name__=='__main__': unittest.main()
