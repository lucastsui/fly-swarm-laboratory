import contextlib
import copy
import io
import unittest
from unittest.mock import patch
import numpy as np
from .layout_loaded_transition_probe import active_constraints,probe
from .layout_pickup_preservation_probe import probe as original_probe
from .test_layout_pickup_preservation_probe import fixture
from .layout_demonstration_step_probe import frozen_predictions


class LoadedTransitionTests(unittest.TestCase):
    def original(self):
        b,bank=fixture()
        with contextlib.redirect_stdout(io.StringIO()):r=original_probe(b,bank)
        return b,bank,r

    def test_selects_one_actual_worst_loaded_frame_per_violating_actor(self):
        mask=np.zeros((32,16),bool);mask[0,10]=True;mask[19,4]=True;mask[20,4]=True
        original={'windowIndexes':list(range(16)),'initial':{'loadedMask':mask.tolist(),'protectedLoadedGrip':[1.8,1.8,1.8]},
            'trials':[{'backtrack':1.,'allTrainingGuards':False,
                       'measured':{'loadedMask':mask.tolist(),'protectedLoadedGrip':[1.806,1.804,1.803]}}]}
        rows=active_constraints(original)
        self.assertEqual([(a['actorInBatch'],a['scoredFrame']) for a in rows],[(4,19),(10,0)])
        broken=copy.deepcopy(original);broken['trials'][0]['allTrainingGuards']=True
        with self.assertRaises(ValueError):active_constraints(broken)
        broken=copy.deepcopy(original);broken['trials'][0]['measured']['protectedLoadedGrip']=[float('nan')]*3
        with self.assertRaises(ValueError):active_constraints(broken)

    def test_unmocked_augmented_jacobian_and_same_guards_restore_original_brain(self):
        b,bank,r=self.original();before=b.checkpoint_hash()
        with contextlib.redirect_stdout(io.StringIO()):out=probe(b,bank,r)
        self.assertEqual(before,b.checkpoint_hash());self.assertEqual(len(out['trials']),6)
        self.assertEqual(len(out['linearProposal']['requestedOutputChange']),18+len(out['activeLoadedFrames']))
        self.assertTrue(all(v==0. for v in out['linearProposal']['requestedOutputChange'][16:]))
        self.assertTrue(out['oldTrainingAndServiceGuardsUnchanged']);self.assertFalse(out['candidateSaved'])

    def test_error_after_temporary_change_restores_original_mode_and_parameters(self):
        b,bank,r=self.original();before=b.checkpoint_hash();b.eval()
        def fail(*args):
            if b.checkpoint_hash()!=before:raise RuntimeError('Injected temporary-forward failure')
            return frozen_predictions(*args)
        with contextlib.redirect_stdout(io.StringIO()),patch('engine.layout_loaded_transition_probe.frozen_predictions',side_effect=fail):
            with self.assertRaises(RuntimeError):probe(b,bank,r)
        self.assertEqual(before,b.checkpoint_hash());self.assertFalse(b.training)


if __name__=='__main__':unittest.main()
