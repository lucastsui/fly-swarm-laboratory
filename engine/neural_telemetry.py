"""Read-only, sampled telemetry of actual frozen-model connection signals.

The graph contains neuron-pair connections, not spatially localized synapses.
Only edges with BOTH endpoints in the existing anatomy sample are published.
"""
import json
import struct
import time

import numpy as np
import torch


class NeuralTelemetry:
    def __init__(self, model, policy, indices, nodes, checkpoint_hash):
        self.indices = indices
        self.model = model
        self.policy = policy
        selected = indices.cpu().numpy()
        lookup = np.full(model.n, -1, np.int32)
        lookup[selected] = np.arange(len(selected))
        crow = model.crow.cpu().numpy()
        col = model.col.cpu().numpy()
        edge_ids, sources, targets = [], [], []
        for target, row in enumerate(selected):
            start, end = int(crow[row]), int(crow[row+1])
            owners = lookup[col[start:end]]
            offsets = np.flatnonzero(owners >= 0)
            edge_ids.extend((offsets+start).tolist())
            sources.extend(owners[offsets].tolist())
            targets.extend([target]*len(offsets))
        first = policy.policies[0] if hasattr(policy, 'policies') else policy
        weights = first.weights[0][torch.as_tensor(edge_ids, device=indices.device)].cpu().numpy()
        self.metadata = {
            'schema': 'connection-signals-v1', 'checkpointHash': checkpoint_hash,
            'nodes': nodes, 'source': sources, 'target': targets, 'edgeIndex': edge_ids,
            'weight': (weights*np.float32(1.5)).tolist(),
            'connectionCount': len(edge_ids), 'totalConnections': int(model.col.numel()),
            'neuronCount': len(nodes), 'totalNeurons': model.n, 'fly': 1,
            'definition': '1.5 * learned signed weight * masked presynaptic state entering neural substep 4',
            'geometry': 'Schematic links between neuron locations; not localized synapses or traced axons',
            'sampled': True, 'targetSampleHz': 20,
        }
        self.metadata_bytes = json.dumps(self.metadata, separators=(',', ':')).encode()
        self.packet = b''

    @torch.inference_mode()
    def capture(self, step):
        first = self.policy.policies[0] if hasattr(self.policy, 'policies') else self.policy
        state = self.policy.state
        if state is None:
            activity = source = np.zeros(len(self.indices), np.float32)
        else:
            # No approximation using the *next* recurrent state: require the
            # exact tensor actually consumed by the sparse multiplication.
            transmitted = getattr(first, 'last_transmitted', None)
            if transmitted is None:
                raise ValueError('Connection telemetry requires the instrumented frozen policy')
            activity = state[self.indices, 0].cpu().numpy()
            source = transmitted[self.indices, 0].cpu().numpy()
        self.packet = (struct.pack('<IddI', 0x31425346, float(step), time.monotonic()*1000, len(activity))
                       + np.asarray(activity, dtype='<f4').tobytes()
                       + np.asarray(source, dtype='<f4').tobytes())
