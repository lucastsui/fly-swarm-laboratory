import unittest
import contextlib
import io
from types import SimpleNamespace
import numpy as np
import torch
from .layout_context_credit_probe import geometry, probe
from .layout_operation_prefix import CONTEXTS, KINDS
from .test_layout_microfit_flow import TinyTrainBrain


class GeometryTests(unittest.TestCase):
    def test_exact_recurrent_probe_is_read_only(self):
        b = TinyTrainBrain().train()
        b.surrogate_training = False
        b.fingerprint = lambda: b.fixed_hash
        before = b.checkpoint_hash()
        specs = [dict(kind=k,category=c,start=20,stop=116,lossStart=84,focusTick=100) for k in KINDS for c in CONTEXTS]
        x = torch.ones((96,16,297)); y = torch.zeros((96,16,3))
        for i,s in enumerate(specs): y[80,i,2] = .2 if s['category'] == CONTEXTS[2] else 2.2
        bank = SimpleNamespace(manifest={'parameterHash':before,'windows':specs},
            groups={(s['kind'],s['category']):[i] for i,s in enumerate(specs)},
            batch=lambda ids,device:(x,y,torch.zeros((4,16)),torch.zeros((32,16,2))))
        with contextlib.redirect_stdout(io.StringIO()): result = probe(b,bank)
        self.assertEqual(before,b.checkpoint_hash())
        self.assertTrue(all(p.grad is None for p in b.parameters()))
        self.assertEqual(len(result['windows']),16)
        self.assertEqual(len(result['scaledCosines']),6)
        self.assertGreater(result['scaledRowNorms'][0],0)
        np.testing.assert_allclose(np.asarray(result['scaledCosines'])[:4,:4],1,atol=1e-12)
        bank.manifest['parameterHash'] = 'stale'
        with self.assertRaises(ValueError): probe(b,bank)

    def test_parallel_and_opposed_outputs(self):
        rows = [[torch.tensor([1.,0.]), torch.tensor([0.])],
                [torch.tensor([2.,0.]), torch.tensor([0.])],
                [torch.tensor([-1.,0.]), torch.tensor([0.])]]
        g = geometry(rows, (1.,1.))
        np.testing.assert_allclose(g['scaledCosines'], [[1,1,-1],[1,1,-1],[-1,-1,1]])
        np.testing.assert_allclose(g['normalizedGramEigenvalues'], [0,0,3], atol=1e-14)
        self.assertEqual(g['parameterBlocks']['tonic']['zeroRows'], [0,1,2])

    def test_orthogonal_blocks_and_zero(self):
        rows = [[torch.tensor([1.]),torch.tensor([0.])],
                [torch.tensor([0.]),torch.tensor([2.])],
                [torch.tensor([0.]),torch.tensor([0.])]]
        g = geometry(rows, (.5,.25))
        np.testing.assert_allclose(g['scaledOutputGram'], np.diag([.25,.25,0]))
        np.testing.assert_allclose(g['scaledCosines'], np.diag([1,1,0]))

    def test_reject_invalid(self):
        good = [[torch.ones(2),torch.ones(1)]]*2
        for scales in ((0,1),(float('nan'),1),(1,)):
            with self.assertRaises(ValueError): geometry(good, scales)
        with self.assertRaises(ValueError): geometry([[torch.ones(1)]]*2)
        with self.assertRaises(ValueError): geometry([[torch.tensor([float('nan')]),torch.ones(1)]]*2)


if __name__ == '__main__':
    unittest.main()
