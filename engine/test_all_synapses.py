"""Small CPU fixtures for all-edge aggregation and checkpoint isolation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .carry_training import CarryLearner
from .plastic_brain import SCHEMA,ALL_SYNAPSES_SCHEMA

class AllSynapseLearnerTests(unittest.TestCase):
    def test_all_edges_update_and_resume_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'base';base.mkdir()
            info={'schema':SCHEMA,'plasticSynapses':3,'plasticMaskHash':'old','edges':10,'neurons':3}
            source=json.dumps(info);(base/'brain-info.json').write_text(source)
            (base/'worker-token').write_text('local-test-token')
            np.savez(base/'brain-spec.npz',plastic_edges=[0,3,7],post=[0,1,2],motor_forward=[1])
            candidate=base/'candidate.npz';np.savez(candidate,gains=np.array([.1,.2,.3],np.float32))
            before=candidate.read_bytes();root=Path(temp)/'all'
            learner=CarryLearner(root,base,candidate,900,True)
            self.assertEqual(learner.schema,ALL_SYNAPSES_SCHEMA)
            self.assertTrue(learner.allowed.all());self.assertEqual(len(learner.gains),10)
            initial=learner.gains.copy();np.testing.assert_allclose(initial[[0,3,7]],[.1,.2,.3])
            meta={'schema':learner.schema,'mask':learner.info['plasticMaskHash'],'worker':'spark1','updateId':'a','version':0}
            self.assertTrue(learner.submit(meta,{'delta':np.full(10,.01,np.float32)})['accepted'])
            self.assertTrue(np.all(learner.gains!=initial))
            with self.assertRaises(ValueError):learner.submit({**meta,'schema':SCHEMA,'updateId':'b'},{'delta':np.ones(10,np.float32)})
            learner.save();restored=CarryLearner(root,base,candidate,900,True)
            np.testing.assert_array_equal(restored.gains,learner.gains)
            self.assertEqual(restored.version,1);self.assertEqual(candidate.read_bytes(),before)
            self.assertEqual((base/'brain-info.json').read_text(),source)
            metadata=(root/'brain-info.json').read_bytes()
            with self.assertRaises(ValueError):CarryLearner(root,base,candidate,900,False)
            self.assertEqual((root/'brain-info.json').read_bytes(),metadata)
            configuration={'rule':'signed','stage':'mixed','horizon':400,'eta':1.,'stepLimit':.005}
            signed_root=Path(temp)/'signed'
            signed=CarryLearner(signed_root,base,candidate,900,True,configuration)
            signed_meta={**meta,'mask':signed.info['plasticMaskHash'],'schema':signed.schema,
                         'runId':signed.run_id,'configurationHash':signed.configuration_hash}
            for wrong in ({'runId':'another-run'},{'configurationHash':'another-rule'}):
                with self.assertRaises(ValueError):signed.submit({**signed_meta,**wrong},{'delta':np.ones(10,np.float32)})
            prior=signed.gains.copy()
            self.assertTrue(signed.submit(signed_meta,{'delta':np.ones(10,np.float32)})['accepted'])
            np.testing.assert_allclose(signed.gains-prior,.005,atol=1e-7)
            signed.save();checkpoint=(signed_root/'shared-brain.npz').read_bytes()
            with self.assertRaises(ValueError):
                CarryLearner(signed_root,base,candidate,900,True,{**configuration,'eta':.2})
            self.assertEqual((signed_root/'shared-brain.npz').read_bytes(),checkpoint)
            anti=CarryLearner(Path(temp)/'antithetic',base,candidate,900,True,{**configuration,'rule':'antithetic'})
            with self.assertRaises(ValueError):
                anti.submit(signed_meta,{'delta':np.ones(10,np.float32)})
            anti_meta={**signed_meta,'runId':anti.run_id,'configurationHash':anti.configuration_hash}
            prior=anti.gains.copy()
            self.assertTrue(anti.submit(anti_meta,{'delta':np.ones(10,np.float32)})['accepted'])
            np.testing.assert_allclose(anti.gains-prior,.005,atol=1e-7)
            for rule in ('covariance','coherent','differential'):
                current=CarryLearner(Path(temp)/rule,base,candidate,900,True,{**configuration,'rule':rule})
                with self.assertRaises(ValueError):current.submit(signed_meta,{'delta':np.ones(10,np.float32)})
                prior=current.gains.copy()
                current_meta={**signed_meta,'runId':current.run_id,'configurationHash':current.configuration_hash}
                self.assertTrue(current.submit(current_meta,{'delta':np.ones(10,np.float32)})['accepted'])
                np.testing.assert_allclose(current.gains-prior,.005,atol=1e-7)

if __name__=='__main__':unittest.main()
