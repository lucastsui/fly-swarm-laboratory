import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .layout_correction_sampling import CorrectionCache,MixedCorrectionCurriculum
from .layout_correction_prefix import KINDS,CONTEXTS
from .layout_correction_prefix import VERSION,correction_specs
from .layout_recovery_teacher import LABEL_VERSION
from .layout_recovery_world import INTERFACE,PHYSICS
from .layout_recovery_demonstrations import file_hash
from .test_layout_correction_prefix import fixtures


class CorrectionSamplingTests(unittest.TestCase):
    def bank(self,tag):
        b=object.__new__(CorrectionCache)
        b.x=np.zeros((16,96,1,297),np.float32)
        b.y=np.zeros((16,96,1,3),np.float32)
        b.states=np.zeros((16,4,1),np.float32)
        b.manifest={'parameterHash':'parent','windows':[]}
        b.groups={}; b.record={'source':tag,'labelVersion':'explicit-test','noNewBrainInputs':True}
        for i,(k,c) in enumerate((k,c) for k in KINDS for c in CONTEXTS):
            b.x[i]=i; b.y[i]=i+1; b.states[i]=i+2
            b.groups[k,c]=[i]; b.manifest['windows'].append({'kind':k,'category':c,'fly':i%4})
        return b

    def test_independent_actor_history_and_explicit_one_quarter_mix(self):
        d,c=self.bank('demo'),self.bank('learner-only-correction')
        mix=MixedCorrectionCurriculum(d,c); rng=np.random.default_rng(7)
        for n in range(1,9):
            x,y,state,meta=mix.sample(rng,'cpu')
            self.assertEqual(x.shape,(96,16,297)); self.assertEqual(state.shape,(4,16))
            for i in range(16):
                self.assertTrue(torch.all(x[:,i]==i))
                self.assertTrue(torch.all(y[:,i]==i+1))
                self.assertTrue(torch.all(state[:,i]==i+2))
            self.assertEqual(meta['source'],'learner-only-correction' if n%4==0 else 'service-balanced-demonstration')
            self.assertEqual(meta['correctionSamples'],n//4)
        self.assertEqual(mix.correction_samples,2)

    def test_mismatched_prefix_parents_rejected(self):
        a,b=self.bank('a'),self.bank('b'); b.manifest['parameterHash']='other'
        with self.assertRaises(ValueError): MixedCorrectionCurriculum(a,b)

    def test_loader_rejects_rehashed_altered_inputs_labels_and_bad_parent(self):
        episodes=fixtures(); specs=correction_specs(episodes)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'manifest.json').write_text('{}')
            data={'episodes':[],'parameterHash':'behavior','canonicalRun':'run','canonicalVersion':60,'fixedHash':'fixed'}
            folder=p/'cache'; folder.mkdir()
            x=np.stack([episodes[s['episode']][0]['observations'][s['start']:s['stop'],s['fly']:s['fly']+1] for s in specs])
            y=np.stack([episodes[s['episode']][0]['labels'][s['start']:s['stop'],s['fly']:s['fly']+1] for s in specs])
            state=np.zeros((len(specs),4,1),np.float32)
            source_names=('layout_correction_prefix.py','layout_batched_prefix_cache.py','layout_demonstration_cache.py')
            m={'schema':VERSION,'finished':True,'parametersUnchanged':True,'parameterHash':'parent','fixedHash':'fixed',
               'control':'learner-only','teacherActions':False,'optimizerUsed':False,'labelsUnchanged':True,
               'labelVersion':LABEL_VERSION,'interface':INTERFACE,'physics':PHYSICS,'isServiceEvidence':False,
               'burn':64,'gradientFrames':32,'datasetManifestHash':file_hash(p/'manifest.json'),'datasetFiles':[],
               'behaviorParameterHash':'behavior','behaviorCanonicalRun':'run','behaviorCanonicalVersion':60,
               'windows':specs,'sourceHashes':{n:file_hash(Path(__file__).parent/n) for n in source_names}}
            def publish(xx,yy):
                np.savez_compressed(folder/'cache.npz',observations=xx,labels=yy,states=state)
                m['cacheFileHash']=file_hash(folder/'cache.npz')
                (folder/'manifest.json').write_text(json.dumps(m))
            with patch('engine.layout_correction_sampling.load_histories',return_value=(episodes,data)):
                publish(x,y)
                bank=CorrectionCache(folder,p,'parent','fixed',4)
                self.assertEqual(len(bank.diagnostic_indexes()),16)
                with self.assertRaises(ValueError): CorrectionCache(folder,p,'other','fixed',4)
                altered=x.copy(); altered[0,0,0,0]=.1; publish(altered,y)
                with self.assertRaises(ValueError): CorrectionCache(folder,p,'parent','fixed',4)
                altered=y.copy(); altered[0,0,0,2]=.3; publish(x,altered)
                with self.assertRaises(ValueError): CorrectionCache(folder,p,'parent','fixed',4)
                for key,value in (('finished',False),('teacherActions',True),('labelsUnchanged',False),
                                  ('labelVersion','incompatible'),('behaviorCanonicalVersion',99)):
                    old=m[key]; m[key]=value; publish(x,y)
                    with self.assertRaises(ValueError): CorrectionCache(folder,p,'parent','fixed',4)
                    m[key]=old


if __name__=='__main__': unittest.main()
