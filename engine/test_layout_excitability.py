"""Regression checks for the brain-internal experimental interface."""
from pathlib import Path
import unittest
import numpy as np
import torch
from .layout_excitability import ExcitableConnectome,SurrogateReLU


class InterfaceTests(unittest.TestCase):
    def test_annotated_projection_is_fixed_and_does_not_select_a_target(self):
        # Every supplied category remains present regardless of carried cargo.
        model=ExcitableConnectome.__new__(ExcitableConnectome)
        torch.nn.Module.__init__(model)
        model.annotated=True
        model.register_buffer('annotated_channels',torch.tensor([30,54,78,102,126,127,128]))
        observations=torch.zeros((4,129))
        observations[:,[30,54,78,102]]=torch.tensor([.2,.4,.6,.8])
        observations[1:,126:129]=torch.eye(3)
        drive=model.sensory_drive(observations)
        torch.testing.assert_close(drive[:4],torch.tensor([.1,.2,.3,.4])[:,None].expand(4,4))
        torch.testing.assert_close(drive[4:,1:],.5*torch.eye(3))
        self.assertEqual(list(model.parameters()),[])

    def test_surrogate_forward_is_relu_but_derivative_is_explicitly_different(self):
        x=torch.tensor([-.01,0.,.01],requires_grad=True)
        y=SurrogateReLU.apply(x)
        torch.testing.assert_close(y,torch.relu(x))
        y.sum().backward()
        torch.testing.assert_close(x.grad,torch.sigmoid(x.detach()/.005))


ROOT=Path(__file__).resolve().parents[1]/'.runtime/dopamine-haul'
CANDIDATE=ROOT.parent/'layout-training/annotated-four/candidate-160.npz'


@unittest.skipUnless(torch.cuda.is_available() and CANDIDATE.exists(),'Full annotated local candidate required')
class FullBrainTests(unittest.TestCase):
    def test_only_existing_synapses_and_excitability_are_trainable(self):
        with np.load(CANDIDATE,allow_pickle=False) as archive:
            gains=archive['gains'].copy();tonic=archive['tonic'].copy()
        model=ExcitableConnectome(ROOT,gains,tonic,annotated=True)
        self.assertEqual(set(dict(model.named_parameters())),{'log_gains','tonic'})
        self.assertEqual(model.n,166700)
        self.assertEqual(model.input_channels,129)
        self.assertEqual(model.interface,'annotated-color-cargo-v1')
        self.assertFalse(model.gated)
        self.assertFalse(model.surrogate_training)
        before=model.checkpoint_hash();fixed=model.fingerprint()
        with torch.no_grad():model.tonic[0]+=.0001
        self.assertNotEqual(before,model.checkpoint_hash())
        self.assertEqual(fixed,model.fingerprint())
        self.assertTrue(model.audit(gains)['signsPreserved'])


if __name__=='__main__':unittest.main()
