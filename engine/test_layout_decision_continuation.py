import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .layout_decision_resume import validate_records, restore_state, AnchoredOperationCache
from .layout_decision_continuation import train_segment
from .layout_operation_focus import VERSION, focus_fit
from .layout_operation_prefix import parent_motion
from .layout_operation_sampling import OperationCache
from .test_layout_operation_training import OperationTrainingTests
from .test_layout_microfit_flow import TinyTrainBrain


def records():
    m = dict(experiment=VERSION,startUpdate=60,finalUpdate=100,gradientFrames=96,gripLossFrames=1,
             motionLossFrames=32,gripFocusFrameRelativeToLossStart=16,maximumPrefixAgeUpdates=40,
             datasetEpisodes=16,serializedPhysicalReplayOnTrainingHostPassed=True,labelsUnchanged=True,
             optimizerMomentsPreserved=True,samplerRNGPreserved=True,teacherAtInference=False,
             externalDecisionNetwork=False,decoderTrained=False,dopamineLearning=False)
    history = [dict(update=i,lossBeforeUpdate=1.,gradientNormsBeforeClipping={'log_gains':1.,'tonic':2.})
               for i in range(61,101)]
    history[-1]['parameterHash'] = 'candidate'
    r = dict(startUpdate=60,updates=100,sourceFilesUnchanged=True,history=history,
             finalParameterHash='candidate',audit=dict(finite=True,signsPreserved=True,
                                                      fixedGraphSensoryDecoderDynamicsUnchanged=True))
    return m,r,dict(finished=True,update=100,parameterHash='candidate'),copy.deepcopy(history[-1])


class DecisionContinuationTests(unittest.TestCase):
    def test_original_completed_records_required(self):
        validate_records(*records())
        for change in ('unfinished','missing','nan','fixed','candidate','objective'):
            m,r,s,f = records()
            if change == 'unfinished': s['finished'] = False
            elif change == 'missing': r['history'].pop(0)
            elif change == 'nan': r['history'][1]['gradientNormsBeforeClipping']['tonic'] = float('nan')
            elif change == 'fixed': r['audit']['signsPreserved'] = False
            elif change == 'candidate': f['parameterHash'] = 'other'
            else: m['gripLossFrames'] = 32
            with self.assertRaises(ValueError,msg=change): validate_records(m,r,s,f)

    def test_adam_rng_next_update_matches_uninterrupted(self):
        b = TinyTrainBrain(); rates = [.003,1e-5]
        opt = torch.optim.Adam([{'params':[b.log_gains],'lr':rates[0],'eps':1e-14},
                                {'params':[b.tonic],'lr':rates[1],'eps':1e-10}])
        def step(model,optimizer):
            optimizer.zero_grad(); loss = model.log_gains.square().sum()+model.tonic.square().sum()
            loss.backward(); optimizer.step()
        for _ in range(100): step(b,opt)
        rng = np.random.default_rng(26); rng.random(19)
        payload = copy.deepcopy({'optimizer':opt.state_dict(),'updates':100,'rng':rng.bit_generator.state})
        other = copy.deepcopy(b)
        resumed, rr = restore_state(other,payload,rates)
        np.testing.assert_array_equal(rng.integers(1000,size=16),rr.integers(1000,size=16))
        step(b,opt); step(other,resumed)
        for p,q in zip(b.parameters(),other.parameters()): torch.testing.assert_close(p,q,rtol=0,atol=0)
        for kind in ('counter','rate','variance','shape'):
            bad = copy.deepcopy(payload)
            if kind == 'counter': bad['updates'] = 80
            elif kind == 'rate': bad['optimizer']['param_groups'][0]['lr'] *= 10
            elif kind == 'variance': bad['optimizer']['state'][0]['exp_avg_sq'].fill_(-1)
            else: bad['optimizer']['state'][0]['exp_avg'] = torch.zeros(1)
            with self.assertRaises(ValueError,msg=kind): restore_state(copy.deepcopy(b),bad,rates)

    def test_refresh_retains_original_anchor_and_real_training(self):
        b = TinyTrainBrain().eval()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            episodes,data,dataset,m = OperationTrainingTests().make_cache(folder,b)
            with patch('engine.layout_operation_sampling.load_histories',return_value=(episodes,data)):
                original = OperationCache(folder,dataset,b.checkpoint_hash(),b.fixed_hash,4)
            fresh = copy.deepcopy(original)
            with torch.no_grad(): b.log_gains.add_(.02)
            fresh.manifest['parameterHash'] = b.checkpoint_hash()
            fresh.states += .01
            fresh.motion = parent_motion(b,fresh.x,fresh.states)
            before_original = copy.deepcopy(original.__dict__)
            original_fresh_motion = fresh.motion.copy()
            bank = AnchoredOperationCache(fresh,original)
            self.assertEqual(bank.manifest['motionAnchorParameterHash'],original.manifest['parameterHash'])
            np.testing.assert_array_equal(bank.motion,original.motion)
            np.testing.assert_array_equal(bank.states,fresh.states)
            self.assertLess(max(focus_fit(b,fresh)['parentMotionMSE']),1e-10)
            self.assertGreater(max(focus_fit(b,bank)['parentMotionMSE']),0)
            opt = torch.optim.Adam([{'params':[b.log_gains],'lr':.0003},{'params':[b.tonic],'lr':.000001}])
            saved = []; before = b.checkpoint_hash()
            with contextlib.redirect_stdout(io.StringIO()):
                result = train_segment(b,bank,opt,np.random.default_rng(26),4,lambda u,r:saved.append(u))
            self.assertEqual(saved,[104]); self.assertEqual([r['update'] for r in result],[101,102,103,104])
            self.assertNotEqual(before,b.checkpoint_hash())
            for name in ('x','y','states','motion'): np.testing.assert_array_equal(getattr(original,name),before_original[name])
            np.testing.assert_array_equal(fresh.motion,original_fresh_motion)
            self.assertEqual(original.manifest,before_original['manifest'])
            for key in ('x','y','windows','parent'):
                altered = copy.deepcopy(fresh)
                if key in ('x','y'): getattr(altered,key).flat[0] += .1
                elif key == 'windows': altered.manifest['windows'][0]['fly'] += 1
                else: altered.manifest['parameterHash'] = original.manifest['parameterHash']
                with self.assertRaises(ValueError,msg=key): AnchoredOperationCache(altered,original)


if __name__ == '__main__': unittest.main()
