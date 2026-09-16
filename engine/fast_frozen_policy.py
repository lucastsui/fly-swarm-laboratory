"""Inference-only scheduling optimization; original float32 brain equations.

Cache the immutable CSR matrix and optionally replay a CUDA graph. Never
change the connectome, neuron update, sensory projection, or motor decoder.
Kept separate from the original training/evaluation implementation.
"""
import numpy as np
import torch

from .evaluate_supervised import FrozenPolicy


class FastFrozenPolicy(FrozenPolicy):
    def __init__(self, model, batch, cuda_graph=False, index32=False):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Fast policy is restricted to a frozen inference model')
        super().__init__(model, batch)
        if index32 and (model.n >= 2**31 or model.col.numel() >= 2**31):
            raise ValueError('Graph exceeds exact int32 indexing range')
        crow, col = (model.crow.int(), model.col.int()) if index32 else (model.crow, model.col)
        self.matrix = torch.sparse_csr_tensor(crow, col, self.weights[0],
                                             size=(model.n, model.n), check_invariants=False)
        self.graph = None
        self.backend = 'cached-csr'
        if cuda_graph:
            self._capture()
        if index32:
            self.backend += '-int32'

    @torch.inference_mode()
    def forward(self, observations, state):
        model = self.model
        drive = torch.zeros_like(state)
        drive[model.sensory_indices] = model.sensory_drive(observations)
        for _ in range(4):
            # Retain the exact source tensor consumed by the final sparse MM.
            # CUDA graph replay refreshes this same buffer; no extra brain step.
            self.last_transmitted = state * model.fast_mask
            signal = torch.sparse.mm(self.matrix, self.last_transmitted)
            target = model.activation(1.5 * signal + drive + model.tonic)
            state = .75 * state + .25 * target
        rates = torch.stack([state[getattr(model, 'motor_' + k)].mean(0)
                             for k in ('forward', 'left', 'right', 'interact')], dim=1)
        outputs = torch.stack([(80 * rates[:, 0]).clamp(0, 2),
                               (160 * (rates[:, 2] - rates[:, 1])).clamp(-2, 2),
                               rates[:, 3]], dim=1)
        return outputs, state

    @torch.inference_mode()
    def _capture(self):
        device = self.model.log_gains.device
        self.inputs = torch.zeros((self.batch, self.model.input_channels), device=device)
        self.recurrent = torch.zeros((self.model.n, self.batch), device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                output, state = self.forward(self.inputs, self.recurrent)
                self.recurrent.copy_(state)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs, state = self.forward(self.inputs, self.recurrent)
            self.recurrent.copy_(state)
        self.backend = 'cuda-graph'
        self.reset()

    @torch.inference_mode()
    def reset(self):
        self.state = None
        if self.graph is not None:
            self.recurrent.zero_()

    @torch.inference_mode()
    def act(self, observations, explore=False):
        if explore:
            raise ValueError('Frozen inference never explores')
        array = np.asarray(observations, dtype=np.float32)
        if array.shape != (self.batch, self.model.input_channels):
            raise ValueError('Wrong observation shape')
        tensor = torch.as_tensor(array, device=self.model.log_gains.device)
        actions = self.submit(tensor)
        return [{'speed': float(v[0]), 'turn': float(v[1]), 'interact': bool(v[2] > .025)}
                for v in actions.cpu().numpy()]

    @torch.inference_mode()
    def submit(self, tensor):
        """GPU-only advance, allowing independent flies to run concurrently."""
        if self.graph is not None:
            self.inputs.copy_(tensor)
            self.graph.replay()
            actions, self.state = self.outputs, self.recurrent
        else:
            state = self.state
            if state is None:
                state = torch.zeros((self.model.n, self.batch), device=tensor.device)
            actions, self.state = self.forward(tensor, state)
        return actions


class SplitFrozenPolicy:
    """Independent neural states on parallel GPU streams; one shared factory."""
    backend = 'cuda-graph-per-fly'

    def __init__(self, model, batch):
        self.model, self.batch = model, batch
        self.policies = [FastFrozenPolicy(model, 1, cuda_graph=True, index32=True) for _ in range(batch)]
        self.streams = [torch.cuda.Stream(device=model.log_gains.device) for _ in range(batch)]
        self.state = None

    def reset(self):
        # All worker streams are joined before act returns.
        for policy in self.policies:
            policy.reset()
        self.state = None

    @torch.inference_mode()
    def act(self, observations, explore=False):
        if explore:
            raise ValueError('Frozen inference never explores')
        array = np.asarray(observations, dtype=np.float32)
        if array.shape != (self.batch, self.model.input_channels):
            raise ValueError('Wrong observation shape')
        inputs = torch.as_tensor(array, device=self.model.log_gains.device)
        main = torch.cuda.current_stream()
        outputs = []
        for i, (policy, stream) in enumerate(zip(self.policies, self.streams)):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                outputs.append(policy.submit(inputs[i:i+1]))
        for stream in self.streams:
            main.wait_stream(stream)
        actions = torch.cat(outputs, dim=0)
        self.state = torch.cat([p.state for p in self.policies], dim=1)
        return [{'speed': float(v[0]), 'turn': float(v[1]), 'interact': bool(v[2] > .025)}
                for v in actions.cpu().numpy()]
