import unittest
import numpy as np
from .layout_world import LayoutSwarmWorld,color_sensory,validate_layout,random_layout

class LayoutTests(unittest.TestCase):
    def test_role_swap_is_observable(self):
        a=LayoutSwarmWorld(1,kind='original');b=LayoutSwarmWorld(1,kind='original')
        a.agents[0].cargo=b.agents[0].cargo=1
        p=np.array([[s['x'],s['y']] for s in b.stations]);p[[1,2]]=p[[2,1]];b.set_layout(p)
        np.testing.assert_allclose(a.agents[0].sensory(),b.agents[0].sensory(),atol=1e-7)
        self.assertGreater(float(np.max(np.abs(a.sensory()-b.sensory()))),.05)

    def test_local_only_and_cargo_identity(self):
        w=LayoutSwarmWorld(1,kind='original');a=w.agents[0];a.x,a.y=1.,1.
        a.cargo=2;x=color_sensory(a)
        self.assertEqual(x.shape,(129,));np.testing.assert_array_equal(x[30:126],0)
        np.testing.assert_array_equal(x[126:],[0,1,0])

    def test_visible_material_sensing(self):
        w=LayoutSwarmWorld(1,kind='original',stock=True);a=w.agents[0]
        a.x,a.y,a.heading=8.,7.,0.;a.cargo=2
        w.stations[1]['stock']=3
        x=color_sensory(a,stock=True)
        self.assertEqual(x.shape,(201,))
        np.testing.assert_array_equal(x[153:177],x[54:78])
        w.stations[1]['stock']=0
        y=color_sensory(a,stock=True)
        np.testing.assert_array_equal(y[153:177],0)
        a.x,a.y=1.,1.
        np.testing.assert_array_equal(color_sensory(a,stock=True)[129:],0)

    def test_random_layouts_and_spawn(self):
        for kind in ('original','rotated','permuted','compact','wide'):
            for seed in range(20):
                p=validate_layout(random_layout(seed,kind));w=LayoutSwarmWorld(seed,kind=kind)
                np.testing.assert_array_equal(p,[[s['x'],s['y']] for s in w.stations])
                for i,a in enumerate(w.agents):
                    self.assertTrue(.22<=a.x<=19.78 and .22<=a.y<=13.78)
                    for b in w.agents[:i]:self.assertGreater(np.hypot(a.x-b.x,a.y-b.y),.44)

    def test_invalid_positions_rejected_atomically(self):
        w=LayoutSwarmWorld(1);before=[s.copy() for s in w.stations]
        for p in ([],[[1,1]]*4,[[float('nan'),1]]*4,[[-1,1],[5,5],[10,10],[18,12]]):
            with self.assertRaises(ValueError):w.set_layout(p)
            self.assertEqual(w.stations,before)

    def test_layout_change_keeps_live_factory_and_bodies(self):
        w=LayoutSwarmWorld(2);a=w.agents[0];a.cargo=2;w.stations[1]['timer']=.75
        before=(a.x,a.y,a.heading,a.cargo,w.steps)
        w.set_layout(random_layout(7))
        self.assertEqual((a.x,a.y,a.heading,a.cargo,w.steps),before)
        self.assertEqual(w.stations[1]['timer'],.75)
        self.assertIs(w.agents[1].stations,w.stations)

    def test_blind_training_teacher_can_leave_wall_zone(self):
        from .layout_training import local_label
        w=LayoutSwarmWorld(1,flies=1,kind='original');a=w.agents[0]
        a.x,a.y,a.heading,a.cargo=.22,7.,np.pi,3
        for _ in range(200):
            label=local_label(a)
            w.advance([{'speed':float(label[0]),'turn':float(label[1]),'interact':False}])
        self.assertGreater(a.x,1.)

    def test_demonstrations_are_labelled_offline_teacher_data(self):
        from .layout_training import demonstration_sequences
        x,y,report=demonstration_sequences(np.random.default_rng(17),world_count=2,steps=24)
        self.assertEqual(x.shape,(24,8,129));self.assertEqual(y.shape,(24,8,3))
        self.assertTrue(np.isfinite(x).all());self.assertTrue(np.isfinite(y).all())
        self.assertTrue(report['teacherGeneratedExamples']);self.assertFalse(report['brainControlled'])
        self.assertGreater(float(np.max(np.abs(x[0]-x[-1]))),0)

    def test_teacher_does_not_require_steering_from_rear_retinal_tails(self):
        from .layout_training import local_label
        w=LayoutSwarmWorld(1,flies=1,kind='original');a=w.agents[0]
        a.x,a.y,a.heading,a.cargo=10.,7.,0.,1
        w.set_layout([[10.,12.],[5.,7.01],[15.,10.],[15.,3.]])
        self.assertLess(float(color_sensory(a)[54:78].max()),.00001)
        np.testing.assert_allclose(local_label(a),[.8,0.,2.2])

    def test_partial_curriculum_has_real_hidden_boxes_and_blind_labels(self):
        from .layout_training import focus_batch
        for missing in (1,2):
            x,y=focus_batch(np.random.default_rng(42),8,stock=False,missing=missing)
            peaks=x[:,30:126].reshape(-1,4,24).max(2)
            np.testing.assert_array_equal((peaks>.002).sum(1),4-missing)
            for i in range(len(x)):
                cargo=i%4
                if cargo and peaks[i,cargo]==0:
                    np.testing.assert_allclose(y[i],[.8,0.,2.2])

    def test_varied_curriculum_covers_continuous_turns_and_missing_boxes(self):
        from .layout_training import focus_batch
        x,y=focus_batch(np.random.default_rng(42),32,stock=False,missing=1,varied=True)
        self.assertEqual(x.shape,(128,129))
        peaks=x[:,30:126].reshape(-1,4,24).max(2)
        np.testing.assert_array_equal((peaks>.002).sum(1),3)
        self.assertTrue(np.any((np.abs(y[:,1])>.01)&(np.abs(y[:,1])<.3)))
        self.assertTrue(np.any((y[:,0]>.79)&(y[:,1]==0)))

if __name__=='__main__':unittest.main()
