"""TRAINING-only selection of real event actors; no new input or controller.

Four windows per family contribute one recorded fly each (16 columns total).
Each selected column retains the exact cached full-history neural prefix and
every original sensory/label frame. Events never enter the brain's inputs.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .layout_demonstration_cache import load_dataset
from .layout_demonstration_curriculum import DATA_SOURCES
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_teacher import KINDS


def event_choices(spec, meta, arrays, burn, frames):
    category = spec['category']
    if category == 'uniform-trajectory':
        return [{'fly': fly, 'eventTick': None} for fly in range(4)]
    after = category.startswith('after-pickup-')
    event_name, stage_text = (category.removeprefix('after-') if after else category).rsplit('-', 1)
    stage = int(stage_text)
    choices = []
    for event in meta['interactionEvents']:
        if event['event'] != event_name:
            continue
        cargo = event['cargoAfter'] if event_name == 'pickup' else event['cargoBefore']
        if cargo != stage:
            continue
        tick = round(event['time']/meta['dt'])-1
        anchor = tick+frames if after else tick
        start = int(np.clip(anchor-burn-frames//2, 0, meta['frames']-burn-frames))
        if start != spec['start']:
            continue
        fly = event['fly']
        if type(fly) is not int or fly not in range(4) or not 0 <= tick < meta['frames']:
            raise ValueError('Invalid focused event identity')
        if (arrays['labels'][tick, fly, 2] <= 1.
                or arrays['bodies'][tick, fly, 5] != event['cargoBefore']
                or arrays['bodies'][tick+1, fly, 5] != event['cargoAfter']):
            raise ValueError('Focused event disagrees with actual body/label transition')
        choices.append({'fly': fly, 'eventTick': tick})
    if not choices:
        raise ValueError('Event window has no matching physical actor: '+category)
    return sorted(choices, key=lambda e: (e['eventTick'], e['fly']))


def resolve_focus_dataset(args, continuation):
    requested = getattr(args, 'event_focused_demonstrations', None)
    previous = None
    if continuation:
        record = continuation['record']
        path = Path(record['sourceRun'])/'manifest.json'
        if file_hash(path) != record['sourceFileSHA256']['manifest.json']:
            raise ValueError('Source manifest changed before sampler selection')
        previous = json.loads(path.read_text()).get('eventFocusedDataset')
    path = requested if requested is not None else previous
    resolved = Path(path).resolve() if path else None
    if resolved and continuation and not resolved.is_relative_to(Path(continuation['record']['sourceRun']).parent):
        raise ValueError('Focused data must be inside the experiment workspace')
    return resolved, {'previousDataset': previous, 'dataset': str(resolved) if resolved else None,
                      'explicitRevision': requested is not None,
                      'samplingPolicyChanged': bool(previous) != bool(resolved) or
                          bool(previous and str(resolved) != previous),
                      'rngStateRetainedButDrawInterpretationChanges': bool(continuation and not previous and resolved)}


class EventFocusedDemonstrations:
    def __init__(self, bank, dataset):
        self.bank = bank
        dataset = Path(dataset).resolve()
        episodes, manifest = load_dataset(dataset)
        engine = Path(__file__).parent
        if (file_hash(dataset/'manifest.json') != bank.manifest['datasetManifestHash']
                or [{'file': e['file'], 'sha256': e['sha256']} for e in manifest['episodes']]
                    != bank.manifest['datasetFiles']
                or set(manifest['sourceHashes']) != set(DATA_SOURCES)
                or any(file_hash(engine/n) != manifest['sourceHashes'][n] for n in DATA_SOURCES)):
            raise ValueError('Focused dataset/cache/source identity mismatch')
        burn, frames = bank.manifest['burn'], bank.manifest['gradientFrames']
        self.choices = []
        categories = Counter()
        for index, spec in enumerate(bank.manifest['windows']):
            arrays, meta = episodes[spec['episode']]
            if meta['seed'] != spec['seed'] or meta['kind'] != spec['kind']:
                raise ValueError('Focused episode identity mismatch')
            part = slice(spec['start'], spec['stop'])
            if (not np.array_equal(bank.x[index], arrays['observations'][part])
                    or not np.array_equal(bank.y[index], arrays['labels'][part])):
                raise ValueError('Focused cache differs from physical inputs/labels')
            self.choices.append(event_choices(spec, meta, arrays, burn, frames))
            categories[spec['category']] += 1
        self.record = {'dataset': str(dataset), 'datasetManifestHash': bank.manifest['datasetManifestHash'],
                       'windows': len(self.choices), 'categories': dict(categories),
                       'allCachedInputsAndTargetsMatched': True, 'eventBodyTransitionsVerified': True,
                       'actorsPerBatch': 16, 'windowsPerFamily': 4,
                       'prefixStateUnchanged': True, 'observationsAndLabelsUnchanged': True,
                       'teacherAtInference': False, 'isServiceEvidence': False,
                       'samplerSourceHash': file_hash(Path(__file__))}

    def sample(self, rng, device):
        selections, xs, ys, states = [], [], [], []
        for kind in KINDS:
            groups = self.bank.groups[kind]
            for _ in range(4):
                category = str(rng.choice(sorted(groups)))
                index = int(rng.choice(groups[category]))
                choice = self.choices[index][int(rng.integers(len(self.choices[index])))]
                fly = choice['fly']
                # Selecting an independent recurrent batch column preserves its
                # sensory encoding of the other bodies in the recorded world.
                xs.append(self.bank.x[index, :, fly:fly+1])
                ys.append(self.bank.y[index, :, fly:fly+1])
                states.append(self.bank.states[index, :, fly:fly+1])
                selections.append({'windowIndex': index, **self.bank.manifest['windows'][index], **choice})
        tensors = [torch.as_tensor(np.concatenate(parts, axis=1), device=device) for parts in (xs, ys, states)]
        return (*tensors, {'sampling': 'one-physical-event-actor-per-window', 'selections': selections,
                          'actors': 16, 'prefixStateUnchanged': True, 'noNewBrainInputs': True})
