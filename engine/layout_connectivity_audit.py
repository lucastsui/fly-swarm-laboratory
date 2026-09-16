"""Read-only directed reachability audit; topology is not proof of learning.

CSR rows are postsynaptic and columns presynaptic. Traversing that matrix from
a motor cell therefore finds its ancestors. Disabled presynaptic cells and
zero-valued edges cannot transmit. No weights, masks, model or body are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from .layout_recovery_protocol import atomic_json


def active_reverse_graph(crow, col, values, fast_mask):
    n = len(crow)-1
    fast_mask = np.asarray(fast_mask).reshape(-1)
    if len(fast_mask) != n or len(values) != len(col):
        raise ValueError('Graph and mask dimensions differ')
    allowed = (values != 0) & (fast_mask[col] != 0)
    graph = csr_matrix((allowed.astype(np.uint8), col.copy(), crow.copy()), shape=(n, n))
    graph.eliminate_zeros()
    return graph


def ancestor_distances(graph, motor_indices):
    return dijkstra(graph, directed=True, indices=np.asarray(motor_indices),
                    unweighted=True, min_only=True)


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve prior audit')
    with np.load(args.root/'graph.npz', allow_pickle=False) as data:
        crow, col, values = (data[k].copy() for k in ('crow', 'col', 'values'))
    graph_hash = hashlib.sha256(values.tobytes()).hexdigest()
    info = json.loads((args.root/'brain-info.json').read_text())
    if graph_hash != info['initialWeightHash']:
        raise ValueError('Unexpected graph values')
    with np.load(args.root/'brain-spec.npz', allow_pickle=False) as spec:
        graph = active_reverse_graph(crow, col, values, spec['fast_mask'])
        sensory_indices = spec['sensory_indices'].copy()
        motors = {k: spec['motor_'+k].copy() for k in ('forward', 'left', 'right', 'interact')}
    with np.load(args.root/'annotated-inputs.npz', allow_pickle=False) as data:
        if not np.array_equal(data['indices'], sensory_indices):
            raise ValueError('Sensory mapping mismatch')
        channels = data['channels'].copy()
    result = {'initialWeightHash': graph_hash, 'neurons': len(crow)-1,
              'signedEdges': len(values), 'transmittingEdges': graph.nnz,
              'parametersChanged': False, 'isServiceEvidence': False,
              'caveat': 'Reachability ignores actual ReLU gating, saturation, learned magnitude and task discrimination.',
              'sourceHash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'motors': {}}
    for name, motor_indices in motors.items():
        distance = ancestor_distances(graph, motor_indices)
        rows = []
        for channel in np.unique(channels):
            selected = distance[sensory_indices[channels == channel]]
            reachable = selected[np.isfinite(selected)]
            rows.append({'channel': int(channel), 'sensoryCells': len(selected),
                         'reachableCells': len(reachable),
                         'shortestEdges': int(reachable.min()) if len(reachable) else None,
                         'medianEdges': float(np.median(reachable)) if len(reachable) else None,
                         'longestShortestPathEdges': int(reachable.max()) if len(reachable) else None})
        result['motors'][name] = {'indices': motor_indices.tolist(),
                                  'allReachableAncestors': int(np.isfinite(distance).sum()), 'channels': rows}
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'motors'}), flush=True)
    for name, motor in result['motors'].items():
        print(json.dumps({'motor': name, 'ancestors': motor['allReachableAncestors'],
                          'cargoAndTaste': [r for r in motor['channels'] if r['channel'] in (27, 126, 127, 128)],
                          'colorCells': sum(r['sensoryCells'] for r in motor['channels'] if 30 <= r['channel'] < 126),
                          'reachableColorCells': sum(r['reachableCells'] for r in motor['channels'] if 30 <= r['channel'] < 126)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
