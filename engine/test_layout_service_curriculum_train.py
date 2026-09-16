from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .layout_service_curriculum_train import train_curriculum
from .test_layout_microfit_flow import TinyTrainBrain


class ServiceCurriculumTrainTests(unittest.TestCase):
    def test_actual_updates_use_teacher_motion_and_preserve_physical_training_arrays(self):
        model = TinyTrainBrain().eval().requires_grad_(False)
        before = model.checkpoint_hash()
        x = torch.ones(96,16,297)
        x[:,::2,0] = .5
        y = torch.full((96,16,3), .2)
        y[:,:,0] = .7
        y[:,::2,1] = -.4
        y[:,1::2,2] = 2.2
        state = torch.full((4,16), .001)
        originals = [v.clone() for v in (x,y,state)]
        sampler = SimpleNamespace(
            sample=lambda rng, device:(x,y,state,{'fixture':'not physical evidence'}),
            batch=lambda indexes, device:(x,y,state), diagnostic_indexes=lambda:[0],
            manifest={'parameterHash':before})
        optimizer = torch.optim.Adam([{'params':[model.log_gains],'lr':.0003,'eps':1e-14},
                                      {'params':[model.tonic],'lr':.000001,'eps':1e-10}])
        records=[]
        with patch('builtins.print'):
            history = train_curriculum(model,sampler,optimizer,np.random.default_rng(7),60,2,
                                       lambda u,r:records.append((u,r)))
        self.assertEqual([r['update'] for r in history],[61,62])
        self.assertEqual([u for u,_ in records],[62])
        self.assertNotEqual(before,model.checkpoint_hash())
        self.assertEqual(len(optimizer.state),2)
        self.assertTrue(all(float(s['step'])==2 for s in optimizer.state.values()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        for original, value in zip(originals,(x,y,state)):
            torch.testing.assert_close(original,value,rtol=0,atol=0)
        self.assertFalse(records[0][1]['isServiceEvidence'])
        self.assertIn('teacherMotionMSE',records[0][1]['trainingFit'])
        self.assertTrue(all(np.isfinite(r['lossBeforeUpdate']) for r in history))

    def test_reject_unbounded_or_wrong_recurrent_batch(self):
        model = TinyTrainBrain()
        for updates in (0,81,1.5,True):
            with self.assertRaises(ValueError):
                train_curriculum(model,None,None,None,60,updates,None)
        optimizer=torch.optim.Adam(model.parameters())
        sampler=SimpleNamespace(sample=lambda *_:(torch.ones(32,16,297),torch.ones(32,16,3),None,{}))
        with self.assertRaisesRegex(ValueError,'96-frame'):
            train_curriculum(model,sampler,optimizer,None,60,1,None)


if __name__ == '__main__':
    unittest.main()
