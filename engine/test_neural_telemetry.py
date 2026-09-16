import struct
import unittest
import numpy as np
import torch

from .fast_frozen_policy import FastFrozenPolicy
from .neural_telemetry import NeuralTelemetry
from .test_fast_frozen_policy import tiny_model


class TelemetryTests(unittest.TestCase):
    def test_edges_and_signals_match_actual_sparse_input(self):
        model=tiny_model()
        policy=FastFrozenPolicy(model,1)
        indices=torch.tensor([2,0,3,1])  # Deliberately not graph order.
        telemetry=NeuralTelemetry(model,policy,indices,[{'id':int(i)+100} for i in indices],'fixed')
        before=[p.clone() for p in model.parameters()]
        obs=torch.tensor([[.1,.2,.3]])
        state=torch.zeros((4,1));drive=torch.zeros_like(state)
        drive[model.sensory_indices]=model.sensory_drive(obs)
        for _ in range(4):
            source=state*model.fast_mask
            total=torch.sparse.mm(policy.matrix,source)
            state=.75*state+.25*model.activation(1.5*total+drive+model.tonic)
        policy.act(obs.numpy());telemetry.capture(7)
        torch.testing.assert_close(policy.last_transmitted,source)
        torch.testing.assert_close(policy.state,state)
        magic,step,_,count=struct.unpack_from('<IddI',telemetry.packet)
        self.assertEqual((magic,step,count),(0x31425346,7.,4))
        values=np.frombuffer(telemetry.packet,dtype='<f4',offset=24)
        np.testing.assert_allclose(values[:4],state[indices,0].numpy())
        np.testing.assert_allclose(values[4:],source[indices,0].numpy())
        received=np.zeros(4)
        m=telemetry.metadata
        for src,dst,w,edge in zip(m['source'],m['target'],m['weight'],m['edgeIndex']):
            self.assertEqual(int(model.col[edge]),int(indices[src]))
            self.assertEqual(int(model.post[edge]),int(indices[dst]))
            received[dst]+=w*values[4+src]
        np.testing.assert_allclose(received,1.5*total[indices,0].numpy(),atol=1e-8)
        for a,b in zip(before,model.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_subset_has_only_real_internal_edges_and_zero_reset_sample(self):
        model=tiny_model();policy=FastFrozenPolicy(model,1)
        telemetry=NeuralTelemetry(model,policy,torch.tensor([0,1]),[{'id':0},{'id':1}],'fixed')
        self.assertEqual(telemetry.metadata['source'],[1])
        self.assertEqual(telemetry.metadata['target'],[0])
        self.assertEqual(telemetry.metadata['edgeIndex'],[0])
        telemetry.capture(0)
        self.assertFalse(np.frombuffer(telemetry.packet,dtype='<f4',offset=24).any())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_replay_updates_the_exact_retained_source_buffer(self):
        model=tiny_model('cuda');policy=FastFrozenPolicy(model,1,cuda_graph=True,index32=True)
        reference=FastFrozenPolicy(model,1)
        for i in range(8):
            obs=np.full((1,3),i*.06,np.float32)
            policy.act(obs);reference.act(obs)
            torch.testing.assert_close(policy.last_transmitted,reference.last_transmitted)
            torch.testing.assert_close(policy.state,reference.state)


if __name__=='__main__':unittest.main()
