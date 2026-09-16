"""Versioned offline correction histories, not teacher-controlled rollouts.

Original teacher labels remain unchanged. The behavior checkpoint can differ
from the prefix checkpoint: this is explicit off-policy supervised TRAINING.
Every prefix encodes the entire recorded sensory history with the current
frozen parent; neither labels nor sampling metadata enter the brain.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_learner_correction_data import SCHEMA, TRAIN_LOW, TRAIN_HIGH, replay_episode
from .layout_batched_prefix_cache import fill_states
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_teacher import KINDS, LABEL_VERSION
from .layout_recovery_world import INTERFACE, PHYSICS, CHANNELS
from .layout_recovery_brain import load_model
from .layout_recovery_protocol import atomic_json

VERSION = 'learner-only-full-history-correction-prefix-v1'
CONTEXTS = ('pickup-needed', 'loaded-grip-needed', 'loaded-release', 'steering-error')
DATA_SOURCES = ('layout_learner_correction_data.py', 'evaluate_supervised.py',
                'layout_recovery_brain.py', 'layout_excitability.py', 'layout_brain.py',
                'supervised_steering.py', 'layout_recovery_teacher.py', 'layout_recovery_world.py',
                'layout_world.py', 'swarm_world.py', 'haul_world.py',
                'layout_recovery_demonstrations.py', 'layout_interaction_trace.py')


def load_histories(folder, replay=True):
    folder = Path(folder).resolve()
    m = json.loads((folder/'manifest.json').read_text())
    expected = {'schema': SCHEMA, 'control': 'learner-only', 'teacherActions': False,
                'optimizerUsed': False, 'finished': True, 'parametersUnchanged': True,
                'fullPhysicalReplayPassed': True, 'isServiceEvidence': False,
                'interface': INTERFACE, 'physics': PHYSICS, 'labelVersion': LABEL_VERSION}
    if (any(m.get(k) != v for k,v in expected.items()) or (folder/'failure.json').exists()
            or not 4 <= len(m['episodes']) <= 16 or set(m['sourceHashes']) != set(DATA_SOURCES)):
        raise ValueError('Incomplete learner-only training data')
    if any(file_hash(Path(__file__).parent/n) != h for n,h in m['sourceHashes'].items()):
        raise ValueError('Behavior/data source changed')
    episodes, names, seeds = [], set(), set()
    for e in m['episodes']:
        p = (folder/e['file']).resolve()
        if (p.parent != folder or p.suffix != '.npz' or p.name in names or e['seed'] in seeds
                or file_hash(p) != e['sha256']):
            raise ValueError('Invalid, duplicated or changed learner episode')
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            a = {k:z[k].copy() for k in z.files if k != 'metadata'}
        if (any(meta[k] != m[k] for k in ('parameterHash', 'fixedHash', 'canonicalRun', 'canonicalVersion'))
                or any(meta[k] != e[k] for k in ('kind','seed'))
                or not TRAIN_LOW <= meta['seed'] < TRAIN_HIGH
                or meta['teacherLabelsUsedForTrainingOnly'] is not True
                or e['audit']['allPhysicalActionsObservationsBodiesLabelsEventsMatched'] is not True):
            raise ValueError('Wrong canonical behavior provenance')
        # This exact replay is performed on the Linux source/training host.
        # It deliberately does not relax checks for Windows double rounding.
        if replay:
            replay_episode(a, meta)
        n = meta['frames']
        for key, shape in {'observations':(n,4,CHANNELS),'labels':(n,4,3),
                           'appliedActions':(n,4,3),'bodies':(n+1,4,6),
                           'cooldowns':(n+1,4)}.items():
            if (a[key].shape != shape or not np.isfinite(a[key]).all()
                    or a[key].dtype != (np.float64 if key=='cooldowns' else np.float32)):
                raise ValueError('Malformed numeric correction history')
        if not np.array_equal(a['observations'][...,126:129], a['bodies'][:-1,:,5,None] == np.arange(1,4)):
            raise ValueError('Cargo/observation mismatch')
        episodes.append((a,meta)); names.add(p.name); seeds.add(meta['seed'])
    if {s['kind'] for _,s in episodes} != set(KINDS):
        raise ValueError('Every training family required')
    return episodes, m


def correction_specs(episodes, per_actor_context=2):
    if type(per_actor_context) is not int or not 1 <= per_actor_context <= 4:
        raise ValueError('Bounded correction selections required')
    result = []
    for episode, (a,m) in enumerate(episodes):
        if m['frames'] < 96 or not TRAIN_LOW <= m['seed'] < TRAIN_HIGH or m['kind'] not in KINDS:
            raise ValueError('Wrong correction window provenance')
        rng = np.random.default_rng(m['seed']+410099)
        for fly in range(4):
            loaded = a['bodies'][:-1,fly,5] > 0
            positive = a['labels'][:,fly,2] > 1.
            ready = a['cooldowns'][:-1,fly] <= .05
            masks = {'pickup-needed':~loaded & positive & ready,
                     'loaded-grip-needed':loaded & positive & ready,
                     'loaded-release':loaded & ~positive & ready & (a['appliedActions'][:,fly,2] > .5),
                     'steering-error':np.abs(a['labels'][:,fly,1]-a['appliedActions'][:,fly,1]) > .4}
            for context, mask in masks.items():
                ticks = np.flatnonzero(mask & (np.arange(len(mask)) >= 80)
                                       & (np.arange(len(mask)) < len(mask)-16))
                if not len(ticks):
                    continue
                for tick in sorted(rng.choice(ticks,min(per_actor_context,len(ticks)),replace=False)):
                    result.append({'episode':episode,'seed':m['seed'],'kind':m['kind'],'fly':fly,
                                   'category':context,'focusTick':int(tick),
                                   'start':int(tick)-80,'stop':int(tick)+16,'lossStart':int(tick)-16})
    for kind in KINDS:
        for context in CONTEXTS:
            if not any(s['kind']==kind and s['category']==context for s in result):
                raise ValueError('Missing real correction family/context: '+kind+'/'+context)
    return result


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve every correction cache')
    episodes, data = load_histories(args.dataset, replay=True)
    specs = correction_specs(episodes)
    torch.set_num_threads(4)
    model = load_model(args.root,args.candidate).eval().requires_grad_(False)
    before = model.checkpoint_hash()
    if model.fixed_hash != data['fixedHash'] or before != args.expected_parameter_hash:
        raise ValueError('Wrong frozen prefix parent')
    engine = Path(__file__).parent
    names = set(DATA_SOURCES) | {'layout_correction_prefix.py','layout_batched_prefix_cache.py',
                               'layout_demonstration_cache.py','layout_demonstration_train.py',
                               'layout_demonstration_curriculum.py'}
    m = {'schema':VERSION,'parameterHash':before,'fixedHash':model.fixed_hash,
         'candidateFileHash':file_hash(args.candidate),'datasetManifestHash':file_hash(args.dataset/'manifest.json'),
         'datasetFiles':data['episodes'],'behaviorParameterHash':data['parameterHash'],
         'behaviorCanonicalRun':data['canonicalRun'],'behaviorCanonicalVersion':data['canonicalVersion'],
         'prefixCanonicalRun':args.canonical_run,'prefixCanonicalVersion':args.version,
         'sourceHashes':{n:file_hash(engine/n) for n in sorted(names)},
         'control':'learner-only','teacherActions':False,'optimizerUsed':False,
         'labelVersion':LABEL_VERSION,'labelsUnchanged':True,'interface':INTERFACE,'physics':PHYSICS,
         'windows':specs,'finished':False,'isServiceEvidence':False,'burn':64,'gradientFrames':32,
         'prefixSemantics':'Full recorded history with frozen prefix parent, BEFORE window; behavior parent explicit'}
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json',m)
    try:
        states = fill_states(model,episodes,specs,32)
        x = np.stack([episodes[s['episode']][0]['observations'][s['start']:s['stop'],s['fly']:s['fly']+1] for s in specs])
        y = np.stack([episodes[s['episode']][0]['labels'][s['start']:s['stop'],s['fly']:s['fly']+1] for s in specs])
        one_actor = np.stack([states[i,:,s['fly']:s['fly']+1] for i,s in enumerate(specs)])
        if (model.checkpoint_hash()!=before or model.fingerprint()!=model.fixed_hash
                or file_hash(args.candidate)!=m['candidateFileHash']
                or file_hash(args.dataset/'manifest.json')!=m['datasetManifestHash']
                or any(file_hash(args.dataset/e['file'])!=e['sha256'] for e in m['datasetFiles'])
                or any(file_hash(engine/n)!=h for n,h in m['sourceHashes'].items())):
            raise ValueError('Prefix model/data/source changed')
        with (args.out/'cache.npz').open('xb') as f:
            np.savez_compressed(f,observations=x,labels=y,states=one_actor)
        m.update(finished=True,parametersUnchanged=True,cacheFileHash=file_hash(args.out/'cache.npz'),
                 audit=model.audit(model.log_gains.detach().cpu().numpy()))
        atomic_json(args.out/'manifest.json',m)
        print('CORRECTION_PREFIX_FINISHED '+json.dumps({'windows':len(specs),'parameterHash':before,
                                                       'cacheFileHash':m['cacheFileHash']}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','candidate','dataset','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--canonical-run',required=True)
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--expected-parameter-hash',required=True)
    main(p.parse_args())
