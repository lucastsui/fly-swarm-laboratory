"""Read-only age of teacher return-memory cues in ORIGINAL physical histories.

Teacher state is inspected for training diagnosis only, never supplied to a
brain or body. Replay every original sensory/body/label frame and applied act.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .layout_correction_prefix import load_histories
from .layout_operation_prefix import operation_specs, CONTEXTS
from .layout_recovery_teacher import LocalTeacher, KINDS
from .layout_recovery_world import RecoveryWorld
from .layout_learner_correction_data import commands
from .layout_recovery_demonstrations import body_state, file_hash
from .layout_recovery_protocol import atomic_json


def cue_row(spec, last_full):
    if last_full is None or not 0 <= last_full <= spec['focusTick']:
        raise ValueError('An actual earlier visible-full cue is required')
    return {**spec, 'lastVisibleFullTick': last_full,
            'cueAgeSeconds': .05*(spec['focusTick']-last_full),
            'outsideDifferentiatedHistory': last_full < spec['start']}


def audit_episode(arrays, meta, specs):
    w = RecoveryWorld(meta['seed'],kind=meta['kind'],random_starts=meta['randomStarts'])
    teachers = [LocalTeacher() for _ in range(4)]
    selected = {(s['focusTick'],s['fly']):s for s in specs if s['category'] == CONTEXTS[3]}
    last_full, last_cargo, rows = [None]*4, [0]*4, []
    for tick in range(meta['frames']):
        observations = w.sensory()
        labels = np.asarray([t.label(f) for t,f in zip(teachers,w.agents)],np.float32)
        if (not np.array_equal(observations,arrays['observations'][tick])
                or not np.array_equal(labels,arrays['labels'][tick])
                or not np.array_equal(body_state(w),arrays['bodies'][tick])):
            raise ValueError('Original replayed sensory/body/label frame differs')
        for fly, (agent, teacher) in enumerate(zip(w.agents,teachers)):
            cargo = agent.cargo
            if cargo != last_cargo[fly]:
                last_full[fly] = None
            last_cargo[fly] = cargo
            visible = observations[fly,30:126].reshape(4,24).max(1) >= .002
            if cargo in (1,2) and visible[cargo] and agent.stations[cargo]['stock'] >= 2:
                last_full[fly] = tick
            if (tick,fly) in selected:
                if teacher.return_stage != cargo or cargo not in (1,2) or labels[fly,2] <= 1:
                    raise ValueError('Selected positive source return has no teacher return-memory state')
                rows.append(cue_row(selected[tick,fly],last_full[fly]))
        w.advance(commands(arrays['appliedActions'][tick]))
    if (len(rows) != len(selected) or w.deliveries != meta['productsNotVerification']
            or not np.array_equal(body_state(w),arrays['bodies'][-1])):
        raise ValueError('Incomplete original physical/teacher-memory replay')
    return rows


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    started = time.perf_counter()
    episodes, manifest = load_histories(args.dataset,replay=False)
    specs = operation_specs(episodes); rows = []
    for index,(arrays,meta) in enumerate(episodes):
        rows.extend(audit_episode(arrays,meta,[s for s in specs if s['episode'] == index]))
    report = {'analysisOnly':True,'isServiceEvidence':False,'optimizerUsed':False,'teacherControlsBodies':False,
              'originalFramesReplayed':True,'datasetManifestHash':file_hash(args.dataset/'manifest.json'),
              'behaviorParameterHash':manifest['parameterHash'],'seconds':time.perf_counter()-started,
              'sourceHash':file_hash(Path(__file__)),'trainingSelectionOnly':True,'rows':rows,
              'families':{k:{'selectedSupportedReturns':sum(r['kind']==k for r in rows),
                             'cueOutsideGradientHistory':sum(r['kind']==k and r['outsideDifferentiatedHistory'] for r in rows),
                             'maxCueAgeSeconds':max([r['cueAgeSeconds'] for r in rows if r['kind']==k],default=0)} for k in KINDS}}
    if any(file_hash(args.dataset/e['file']) != e['sha256'] for e in manifest['episodes']):
        raise ValueError('Original data changed during audit')
    atomic_json(args.out,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
