import copy
import unittest
from unittest.mock import patch
import numpy as np
from .layout_learner_correction_data import collect,replay_episode


class TestPolicy:
    def act(self,x):
        return [{'speed':.3,'turn':.2,'interact':False} for _ in x]


class LearnerCorrectionDataTests(unittest.TestCase):
    def test_real_physics_replays_and_teacher_labels_do_not_control_bodies(self):
        specs=[{'seed':9370100,'kind':'compact','randomStarts':False}]
        a,m=collect(TestPolicy(),specs,150)[0]
        audit=replay_episode(a,m)
        self.assertFalse(audit['teacherControlledBodies'])
        self.assertFalse(audit['isServiceEvidence'])
        with patch('engine.layout_learner_correction_data.LocalTeacher.label',
                   return_value=np.asarray([1.,-1.,2.2],np.float32)):
            changed,_=collect(TestPolicy(),specs,150)[0]
        for key in ('observations','bodies','cooldowns','appliedActions'):
            np.testing.assert_array_equal(a[key],changed[key])
        self.assertFalse(np.array_equal(a['labels'],changed['labels']))
        self.assertTrue(np.all(a['appliedActions'][...,2]==0))

    def test_corrupted_commands_labels_and_reset_provenance_rejected(self):
        a,m=collect(TestPolicy(),[{'seed':9371100,'kind':'wide','randomStarts':True}],50)[0]
        for key in ('appliedActions','labels','observations','bodies','cooldowns'):
            bad={k:v.copy() for k,v in a.items()}
            bad[key].flat[0]+=.5
            with self.assertRaises(ValueError):
                replay_episode(bad,m)
        for change in ({'teacherActions':True},{'seed':10100000},{'deliveryResets':True}):
            with self.assertRaises(ValueError):
                replay_episode(a,{**m,**change})
        with self.assertRaises(ValueError):
            collect(TestPolicy(),[{'seed':10100000,'kind':'wide','randomStarts':False}],50)


if __name__=='__main__':
    unittest.main()
