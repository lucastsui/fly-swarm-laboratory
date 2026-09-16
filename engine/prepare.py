"""Prepare the full released MaleCNS neuronal graph; no edge threshold/circuit cropping."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from scipy import sparse

def prepare(root: Path):
    started = time.time()
    annotations = feather.read_feather(root / 'annotations.feather')
    neurons = annotations[annotations.superclass.notna() & (annotations.status.fillna('') != 'Glia')].sort_values('bodyId').reset_index(drop=True)
    assert len(neurons) == 166700, f'Unexpected neuron count: {len(neurons)}'
    identifiers = pd.Index(neurons.bodyId)
    pre, post, weights = [], [], []
    with pa.memory_map(str(root / 'edges.feather'), 'r') as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            src = identifiers.get_indexer(batch.column('body_pre').to_numpy())
            dst = identifiers.get_indexer(batch.column('body_post').to_numpy())
            keep = (src >= 0) & (dst >= 0)
            pre.append(src[keep].astype(np.int32)); post.append(dst[keep].astype(np.int32))
            weights.append(batch.column('weight').to_numpy()[keep].astype(np.float32))
    pre = np.concatenate(pre); post = np.concatenate(post); contacts = np.concatenate(weights)
    assert len(contacts) == 25582938, f'Unexpected edge count {len(contacts)}'
    assert int(contacts.sum(dtype=np.float64)) == 124177617
    nt = feather.read_feather(root / 'neurotransmitters.feather').set_index('body').reindex(identifiers)
    transmitter = nt.consensus_nt.fillna(nt.predicted_nt).fillna('unknown')
    sign = np.where(transmitter.isin(['gaba', 'glutamate', 'histamine']), -1., 1.).astype(np.float32)
    # Degree normalization is an engineered stability choice, not measured physiology.
    norm = np.maximum(np.bincount(post, weights=contacts, minlength=len(neurons)), 1).astype(np.float32)
    values = contacts * sign[pre] / norm[post]
    matrix = sparse.coo_matrix((values, (post, pre)), shape=(len(neurons), len(neurons))).tocsr()
    assert matrix.nnz == len(contacts), 'Duplicate rows unexpectedly merged'
    sensory = np.where(neurons.superclass.str.contains('sensory').to_numpy())[0].astype(np.int64)
    motor = np.where(neurons.superclass.isin(['descending_neuron', 'vnc_motor', 'cb_motor']).to_numpy())[0].astype(np.int64)
    np.savez(root / 'graph.npz', crow=matrix.indptr.astype(np.int64), col=matrix.indices.astype(np.int64), values=matrix.data,
             body_ids=neurons.bodyId.to_numpy(), sensory=sensory, motor=motor)
    meta = dict(neurons=len(neurons), edges=len(contacts), contacts=int(contacts.sum(dtype=np.float64)),
                source='https://male-cns.janelia.org/download/', license='CC BY 4.0', release='MaleCNS v1.0',
                paper='https://doi.org/10.1016/j.cell.2026.08.015', sensory=len(sensory), motor=len(motor),
                node_policy='All annotated superclasses, including tbc; exclude explicit Glia status.',
                edge_policy='Every released edge between retained neurons; weak edges and autapses retained.',
                dynamics='Signed contact-count weighted, input-normalized recurrent rate model. Unknown transmitter sign defaults to excitatory.',
                transmitters=transmitter.value_counts().to_dict(), sha256={})
    for name in ['annotations.feather','neurotransmitters.feather','edges.feather','graph.npz']:
        with open(root/name,'rb') as stream: meta['sha256'][name] = hashlib.file_digest(stream,'sha256').hexdigest()
    (root/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({**meta, 'seconds':round(time.time()-started,2)},indent=2),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);prepare(parser.parse_args().root)
