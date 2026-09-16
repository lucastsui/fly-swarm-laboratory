"""Fixed identity-preserving sensory projection using existing receptor types.

Box categories and carried-material odors are artificial labels. The selected
cell-type populations are anatomical annotations, not random new neurons.
No location, route, desired direction, or target selection is encoded here.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow.feather as feather

def main(args):
    output=args.root/'annotated-inputs.npz'
    if output.exists():raise FileExistsError('Preserve the existing sensory mapping')
    with np.load(args.root/'graph.npz',allow_pickle=False) as graph:ids=graph['body_ids'].copy()
    with np.load(args.root/'brain-spec.npz',allow_pickle=False) as spec:
        indices=spec['sensory_indices'].copy();original=spec['sensory_channels'].copy()
    a=feather.read_feather(args.annotations).set_index('bodyId').reindex(ids[indices])
    types=a['type'].fillna('').to_numpy();channels=original.copy();counts={}
    for role,cell_type in enumerate(('R7p','R8p','R7y','R8y')):
        mask=(types==cell_type)&(original<24)
        if not mask.any():raise ValueError('Missing color sensory population '+cell_type)
        channels[mask]=30+24*role+original[mask];counts[cell_type]=int(mask.sum())
    # Three distinct annotated receptor families, not a modulo split that mixes
    # each synthetic odor into every olfactory receptor family.
    for cargo,cell_type in enumerate(('ORN_DA1','ORN_VA1d','ORN_VA1v')):
        mask=(types==cell_type)&np.isin(original,[24,25])
        if not mask.any():raise ValueError('Missing odor sensory population '+cell_type)
        channels[mask]=126+cargo;counts[cell_type]=int(mask.sum())
    np.savez_compressed(output,channels=channels,indices=indices)
    report={'interface':'annotated-color-cargo-v1','cellCounts':counts,'sensoryCells':len(channels),
            'channels':129,'mapping':'fixed artificial category-to-annotated-receptor mapping',
            'targetSelection':False,'globalMap':False,'newNeurons':False}
    (args.root/'annotated-inputs.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);main(p.parse_args())
