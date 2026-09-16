"""Full graph computation with a shared trainable input adapter, neural gains and motor readout."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class FixedWiringMultiply(torch.autograd.Function):
    """Only dense activity is differentiable; reuse the immutable CSR transpose."""
    @staticmethod
    def forward(ctx, wiring, transpose, activity):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(wiring, activity)

    @staticmethod
    def backward(ctx, gradient):
        (transpose,) = ctx.saved_tensors
        return None, None, torch.sparse.mm(transpose, gradient.contiguous())

class Brain(nn.Module):
    def __init__(self, root:Path, observations:int, device='cuda'):
        super().__init__()
        self.meta=json.loads((root/'metadata.json').read_text())
        graph=np.load(root/'graph.npz')
        n=self.meta['neurons']
        self.register_buffer('wiring',torch.sparse_csr_tensor(torch.as_tensor(graph['crow'],device=device),torch.as_tensor(graph['col'],device=device),torch.as_tensor(graph['values'],device=device),size=(n,n)))
        # Anatomical edges are fixed. Build this once, not three times per update.
        self.register_buffer('wiring_transpose',self.wiring.transpose(0,1).to_sparse_csr(),persistent=False)
        self.register_buffer('sensory',torch.as_tensor(graph['sensory'],device=device))
        self.register_buffer('motor',torch.as_tensor(graph['motor'],device=device))
        self.encoder=nn.Linear(observations,len(self.sensory),device=device)
        self.log_gain=nn.Parameter(torch.zeros(n,device=device))
        self.bias=nn.Parameter(torch.zeros(n,device=device))
        self.readout=nn.Sequential(nn.LayerNorm(len(self.motor),device=device),nn.Linear(len(self.motor),64,device=device),nn.Tanh())
        self.actor=nn.Linear(64,6,device=device)
        self.critic=nn.Linear(64,1,device=device)
        nn.init.orthogonal_(self.actor.weight,gain=.01); nn.init.zeros_(self.actor.bias)
        self.n=n;self.device=device

    def initial_state(self,flies:int):
        return torch.zeros((self.n,flies),device=self.device)

    def forward(self,observation,state):
        drive=torch.zeros_like(state).index_copy(0,self.sensory,torch.tanh(self.encoder(observation)).T)
        gain=torch.exp(.5*torch.tanh(self.log_gain))[:,None]
        bias_drive=.05*torch.tanh(self.bias)[:,None]
        # Four full-graph integration steps per action. No hidden global planner.
        for _ in range(4):
            recurrent=FixedWiringMultiply.apply(self.wiring,self.wiring_transpose,state)
            state=.4*state+.6*torch.tanh(1.5*gain*recurrent+drive+bias_drive)
        features=self.readout(state[self.motor].T)
        return self.actor(features),self.critic(features).squeeze(-1),state
