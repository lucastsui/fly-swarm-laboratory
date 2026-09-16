import copy
import contextlib
import io
import unittest
import numpy as np
import torch
from .layout_correction_curriculum_train import restore_state
from .layout_service_curriculum_train import train_curriculum
from .layout_correction_sampling import MixedCorrectionCurriculum
from . import test_layout_correction_sampling as sampling_tests
from .test_layout_microfit_flow import TinyTrainBrain


class CorrectionCurriculumTests(unittest.TestCase):
    def payload(self,brain):
        rates=[.0003,.000001]
        opt=torch.optim.Adam([{'params':[brain.log_gains],'lr':rates[0],'eps':1e-14},
                              {'params':[brain.tonic],'lr':rates[1],'eps':1e-10}])
        for _ in range(60):
            opt.zero_grad(); (brain.log_gains.square().sum()+brain.tonic.square().sum()).backward(); opt.step()
        rng=np.random.default_rng(51); rng.random(12)
        return {'optimizer':copy.deepcopy(opt.state_dict()),'updates':60,'rng':rng.bit_generator.state},rates

    def test_restore_exact_adam_rng_and_actual_mixed_updates(self):
        b=TinyTrainBrain(); payload,rates=self.payload(b)
        opt,rng=restore_state(b,payload,rates)
        for key,val in payload['optimizer']['state'].items():
            for name,x in val.items(): torch.testing.assert_close(opt.state_dict()['state'][key][name],x,rtol=0,atol=0)
        expected=np.random.default_rng(); expected.bit_generator.state=payload['rng']
        np.testing.assert_array_equal(rng.random(10),expected.random(10))
        factory=sampling_tests.CorrectionSamplingTests(); d,c=factory.bank('demo'),factory.bank('correction')
        sampler=MixedCorrectionCurriculum(d,c)
        before=b.checkpoint_hash(); saved=[]
        with contextlib.redirect_stdout(io.StringIO()):
            rows=train_curriculum(b,sampler,opt,rng,60,4,lambda u,r:saved.append(u))
        self.assertEqual([r['update'] for r in rows],[61,62,63,64]); self.assertEqual(saved,[64])
        self.assertEqual(sampler.correction_samples,1)
        self.assertTrue(all(np.isfinite(r['lossBeforeUpdate']) for r in rows))
        self.assertNotEqual(before,b.checkpoint_hash())
        self.assertTrue(all(float(s['step'])==64 for s in opt.state_dict()['state'].values()))

    def test_reject_wrong_optimizer_version_rng_or_lr(self):
        b=TinyTrainBrain(); p,rates=self.payload(b)
        for wrong in ({**p,'updates':59},{k:v for k,v in p.items() if k!='rng'}):
            with self.assertRaises(ValueError): restore_state(b,wrong,rates)
        with self.assertRaises(ValueError): restore_state(b,p,[.003,.00001])


if __name__=='__main__': unittest.main()
