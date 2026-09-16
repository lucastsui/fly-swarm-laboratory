"""Versioned learner-only physical TRAINING histories for later corrections.

Spark2 runs one frozen canonical checkpoint; labels never control its bodies.
Actual applied actions, sensory frames, cargo, cooldowns and events are saved
and fully replayed. This is not held-out service evidence or an optimizer.
The original teacher-label version is explicit; no timing-label substitution
is silently made, and the current offline V22 trainer does not consume these.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .evaluate_supervised import FrozenPolicy
from .layout_recovery_brain import load_model
from .layout_recovery_world import RecoveryWorld, CHANNELS, INTERFACE, PHYSICS
from .layout_recovery_teacher import LocalTeacher, KINDS, LABEL_VERSION
from .layout_recovery_demonstrations import body_state, file_hash
from .layout_interaction_trace import before_interaction, interaction_events
from .layout_recovery_protocol import atomic_json
from .plastic_brain import DT

SCHEMA = 'versioned-learner-only-physical-training-data-v1'
TRAIN_LOW, TRAIN_HIGH = 9370000, 9380000


def commands(values):
    return [{'speed':float(v[0]), 'turn':float(v[1]), 'interact':bool(v[2] > .5)} for v in values]


def collect(policy, specs, frames):
    if not 1 <= frames <= 12000 or not 1 <= len(specs) <= 16:
        raise ValueError('Bounded real training episodes required')
    if (len({s['seed'] for s in specs}) != len(specs)
            or any(not TRAIN_LOW <= s['seed'] < TRAIN_HIGH or s['kind'] not in KINDS for s in specs)):
        raise ValueError('Unique dedicated learner TRAINING seeds required')
    worlds = [RecoveryWorld(s['seed'],kind=s['kind'],random_starts=s['randomStarts']) for s in specs]
    teachers = [[LocalTeacher() for _ in range(4)] for _ in worlds]
    arrays = []
    for w in worlds:
        a = {'observations':np.empty((frames,4,CHANNELS),np.float32),
             'labels':np.empty((frames,4,3),np.float32),
             'appliedActions':np.empty((frames,4,3),np.float32),
             'bodies':np.empty((frames+1,4,6),np.float32),
             'cooldowns':np.empty((frames+1,4),np.float64)}
        a['bodies'][0] = body_state(w)
        a['cooldowns'][0] = [f.cooldown for f in w.agents]
        arrays.append(a)
    events = [[] for _ in worlds]
    began = time.perf_counter()
    for tick in range(frames):
        observations = [w.sensory() for w in worlds]
        labels = [np.asarray([t.label(f) for t,f in zip(ts,w.agents)],np.float32)
                  for ts,w in zip(teachers,worlds)]
        # Only the policy receives observations. Teacher labels never enter
        # this call or the physical action path.
        motors = policy.act(np.concatenate(observations))
        applied = np.asarray([[m['speed'],m['turn'],float(m['interact'])] for m in motors],np.float32)
        if applied.shape != (4*len(worlds),3) or not np.isfinite(applied).all():
            raise ValueError('Invalid fixed-decoder commands')
        for i,w in enumerate(worlds):
            a = arrays[i]
            a['observations'][tick], a['labels'][tick] = observations[i], labels[i]
            a['appliedActions'][tick] = applied[4*i:4*i+4]
            previous = before_interaction(w)
            w.advance(commands(a['appliedActions'][tick]))
            a['bodies'][tick+1] = body_state(w)
            a['cooldowns'][tick+1] = [f.cooldown for f in w.agents]
            events[i].extend(interaction_events(w,previous,tick+1))
        if (tick+1) % 1200 == 0:
            print('LEARNER_TRAINING_TRAJECTORY '+json.dumps({'seconds':(tick+1)*DT,
                  'wallSeconds':time.perf_counter()-began,'productsNotVerification':sum(w.deliveries for w in worlds)}),flush=True)
    metadata = []
    for s,w,e in zip(specs,worlds,events):
        metadata.append({**s,'schema':SCHEMA,'control':'learner-only','teacherActions':False,
                         'teacherLabelsUsedForTrainingOnly':True,'optimizerUsed':False,
                         'injectedCargo':False,'deliveryResets':False,'neuralResetsInsideEpisode':False,
                         'frames':frames,'dt':DT,'labelVersion':LABEL_VERSION,'interface':INTERFACE,'physics':PHYSICS,
                         'interactionEvents':e,'productsNotVerification':w.deliveries,
                         'finalStock':[b['stock'] for b in w.stations],
                         'finalReturned':[b['returned'] for b in w.stations],'isServiceEvidence':False})
    return list(zip(arrays,metadata))


def replay_episode(a,m):
    expected = {'schema':SCHEMA,'control':'learner-only','teacherActions':False,'optimizerUsed':False,
                'injectedCargo':False,'deliveryResets':False,'neuralResetsInsideEpisode':False,
                'labelVersion':LABEL_VERSION,'dt':DT,'interface':INTERFACE,'physics':PHYSICS,'isServiceEvidence':False}
    if any(m.get(k)!=v for k,v in expected.items()) or not TRAIN_LOW<=m['seed']<TRAIN_HIGH or m['kind'] not in KINDS:
        raise ValueError('Wrong learner-only physical training provenance')
    n=m['frames']
    shapes={'observations':(n,4,CHANNELS),'labels':(n,4,3),'appliedActions':(n,4,3),
            'bodies':(n+1,4,6),'cooldowns':(n+1,4)}
    if set(a)!=set(shapes) or not 1<=n<=12000:
        raise ValueError('Invalid learner episode arrays')
    for key,shape in shapes.items():
        if (a[key].shape!=shape or a[key].dtype!=(np.float64 if key=='cooldowns' else np.float32)
                or not np.isfinite(a[key]).all()):
            raise ValueError('Invalid numeric learner history')
    if not np.isin(a['appliedActions'][...,2],(0.,1.)).all():
        raise ValueError('Applied interaction must be binary')
    w=RecoveryWorld(m['seed'],kind=m['kind'],random_starts=m['randomStarts'])
    teachers=[LocalTeacher() for _ in range(4)]
    event_map={}
    for e in m['interactionEvents']:
        event_map.setdefault(round(e['time']/DT)-1,[]).append(e)
    for tick in range(n):
        labels=np.asarray([t.label(f) for t,f in zip(teachers,w.agents)],np.float32)
        if (not np.array_equal(w.sensory(),a['observations'][tick])
                or not np.array_equal(body_state(w),a['bodies'][tick])
                or not np.array_equal(labels,a['labels'][tick])
                or not np.array_equal(np.asarray([f.cooldown for f in w.agents]),a['cooldowns'][tick])):
            raise ValueError('Learner action/sensory/body/label history changed')
        previous=before_interaction(w)
        w.advance(commands(a['appliedActions'][tick]))
        if interaction_events(w,previous,tick+1)!=event_map.get(tick,[]):
            raise ValueError('Learner interaction history changed')
    if (not np.array_equal(body_state(w),a['bodies'][-1])
            or not np.array_equal(np.asarray([f.cooldown for f in w.agents]),a['cooldowns'][-1])
            or w.deliveries!=m['productsNotVerification'] or [b['stock'] for b in w.stations]!=m['finalStock']
            or [b['returned'] for b in w.stations]!=m['finalReturned']):
        raise ValueError('Final learner physical accounting changed')
    return {'seed':m['seed'],'frames':n,'allPhysicalActionsObservationsBodiesLabelsEventsMatched':True,
            'teacherControlledBodies':False,'isServiceEvidence':False}


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve versioned experiences')
    if not 1<=args.cases<=4 or not 1<=args.seconds<=600 or args.version<0:
        raise ValueError('Bounded versioned experience generation required')
    torch.set_num_threads(4)
    model=load_model(args.root,args.candidate).eval().requires_grad_(False)
    before=model.checkpoint_hash()
    if before!=args.expected_parameter_hash:
        raise ValueError('Canonical checkpoint identity mismatch')
    engine=Path(__file__).parent
    sources=('layout_learner_correction_data.py','evaluate_supervised.py','layout_recovery_brain.py',
             'layout_excitability.py','layout_brain.py','supervised_steering.py','layout_recovery_teacher.py',
             'layout_recovery_world.py','layout_world.py','swarm_world.py','haul_world.py',
             'layout_recovery_demonstrations.py','layout_interaction_trace.py')
    manifest={'schema':SCHEMA,'control':'learner-only','canonicalRun':args.canonical_run,
              'canonicalVersion':args.version,'parameterHash':before,'fixedHash':model.fixed_hash,
              'candidateFileHash':file_hash(args.candidate),'sourceHashes':{n:file_hash(engine/n) for n in sources},
              'labelVersion':LABEL_VERSION,'interface':INTERFACE,'physics':PHYSICS,
              'teacherActions':False,'optimizerUsed':False,'isServiceEvidence':False,
              'consumedByCurrentTrainer':False,'finished':False,'episodes':[]}
    specs=[{'seed':args.seed+1000*j+i,'kind':kind,'randomStarts':bool(i%2)}
           for j,kind in enumerate(KINDS) for i in range(args.cases)]
    manifest.update(seconds=args.seconds,casesPerFamily=args.cases,trainingSpecs=specs,
                    neuralStatePolicy='One zero initialization at the real episode start; no mid-episode reset')
    args.out.mkdir(parents=True)
    atomic_json(args.out/'manifest.json',manifest)
    try:
        episodes=collect(FrozenPolicy(model,len(specs)*4),specs,round(args.seconds/DT))
        if (model.checkpoint_hash()!=before or model.fingerprint()!=model.fixed_hash
                or file_hash(args.candidate)!=manifest['candidateFileHash']):
            raise ValueError('Frozen canonical brain changed')
        for a,m in episodes:
            audit=replay_episode(a,m)
            m.update(parameterHash=before,fixedHash=model.fixed_hash,canonicalRun=args.canonical_run,
                     canonicalVersion=args.version)
            name=f"{m['kind']}-{m['seed']}.npz"
            with (args.out/name).open('xb') as stream:
                np.savez_compressed(stream,**a,metadata=np.asarray(json.dumps(m)))
            manifest['episodes'].append({'file':name,'sha256':file_hash(args.out/name),
                                         'kind':m['kind'],'seed':m['seed'],'audit':audit})
            atomic_json(args.out/'manifest.json',manifest)
        if any(file_hash(engine/n)!=h for n,h in manifest['sourceHashes'].items()):
            raise ValueError('Generator/physics/teacher source changed')
        manifest.update(finished=True,parametersUnchanged=True,fullPhysicalReplayPassed=True)
        atomic_json(args.out/'manifest.json',manifest)
        print('LEARNER_CORRECTION_DATA_FINISHED '+json.dumps({'episodes':len(episodes),'parameterHash':before,
                                                            'isServiceEvidence':False}),flush=True)
    except BaseException as error:
        atomic_json(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','candidate','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--canonical-run',required=True)
    p.add_argument('--version',type=int,required=True)
    p.add_argument('--expected-parameter-hash',required=True)
    p.add_argument('--seed',type=int,default=9370100)
    p.add_argument('--cases',type=int,default=2)
    p.add_argument('--seconds',type=int,default=600)
    main(p.parse_args())
