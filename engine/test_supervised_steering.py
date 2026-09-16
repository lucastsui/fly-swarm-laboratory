"""Gradient, fixed-interface and original-forward regression checks."""
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from .supervised_steering import SynapseMM,TrainableConnectome,samples


class SparseDerivativeTests(unittest.TestCase):
    def compare(self,device):
        crow=torch.tensor([0,2,3,5],device=device);col=torch.tensor([0,2,1,0,1],device=device)
        post=torch.tensor([0,0,1,2,2],device=device)
        transpose=sp.csr_matrix((np.arange(5),[0,2,1,0,1],[0,2,3,5]),shape=(3,3)).T.tocsr()
        tcrow=torch.tensor(transpose.indptr.astype(np.int64),device=device)
        tcol=torch.tensor(transpose.indices.astype(np.int64),device=device)
        order=torch.tensor(transpose.data,device=device)
        values=torch.tensor([.4,-.3,.2,.1,-.2],device=device,dtype=torch.float64,requires_grad=True)
        state=torch.tensor([[.1,.2],[-.3,.6],[.7,-.1]],device=device,dtype=torch.float64,requires_grad=True)
        def function(v,s):return SynapseMM.apply(v,s,crow,col,post,tcrow,tcol,v.detach()[order])
        self.assertTrue(torch.autograd.gradcheck(function,(values,state),eps=1e-6,atol=1e-5))
        actual=function(values,state);dense=torch.zeros((3,3),device=device,dtype=values.dtype)
        dense[post,col]=values
        torch.testing.assert_close(actual,dense@state)
        grad1=torch.autograd.grad(actual.square().sum(),(values,state))
        grad2=torch.autograd.grad((dense@state).square().sum(),(values,state))
        for a,b in zip(grad1,grad2):torch.testing.assert_close(a,b)

    def test_cpu_exact_sparse_gradient(self):self.compare('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_exact_sparse_gradient(self):self.compare('cuda')

    def test_teacher_labels_not_in_original_senses(self):
        observations,targets,angles=samples(4411001,8)
        self.assertEqual(observations.shape,(16,30))
        np.testing.assert_array_equal(observations[::2,24:],observations[1::2,24:])
        np.testing.assert_allclose(targets[::2],-targets[1::2])
        np.testing.assert_allclose(angles[::2],-angles[1::2])
        self.assertFalse(np.array_equal(observations[::2,:24],observations[1::2,:24]))


ROOT=Path(__file__).resolve().parents[1]/'.runtime/dopamine-haul'
CANDIDATE=ROOT.parent/'line-training/best-full-v45.npz'


@unittest.skipUnless(torch.cuda.is_available() and CANDIDATE.exists(),'Full local connectome required')
class OriginalForwardTests(unittest.TestCase):
    def test_supervised_forward_matches_existing_frozen_brain(self):
        from .line_experiment import SignedBrain
        torch.set_num_threads(4)
        initial=np.load(CANDIDATE,allow_pickle=False)['gains']
        model=TrainableConnectome(ROOT,initial)
        brain=SignedBrain(ROOT,2,871,all_synapses=True);brain.configure('all');brain.load_gains(initial);brain.reset()
        observations=samples(4411002,1)[0]
        with torch.no_grad():
            actual,state=model(torch.as_tensor(observations,device='cuda'),32)
            for _ in range(8):expected=brain.act(observations,explore=False)
        torch.testing.assert_close(state,brain.state,atol=1e-7,rtol=1e-5)
        np.testing.assert_allclose(actual[:,:2].cpu().numpy(),[[m['speed'],m['turn']] for m in expected],rtol=1e-4,atol=1e-6)
        self.assertEqual(list(dict(model.named_parameters())),['log_gains'])
        audit=model.audit(initial)
        self.assertTrue(audit['fixedGraphSensoryDecoderDynamicsUnchanged'])
        self.assertTrue(audit['signsPreserved']);self.assertEqual(audit['changedFromInitial'],0)


if __name__=='__main__':unittest.main()
