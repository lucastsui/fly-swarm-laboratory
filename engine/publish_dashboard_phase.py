"""Publish a completed, retained experimental checkpoint to the local dashboard.

This does not promote a model as reliable. Validation is reported separately.
Both immutable revision history and the replaced latest pointer are retained.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .phase_viewer import EXPERIMENT, FIXED_HASH, INTERFACE, PHYSICS, inspect_candidate, read_phase, sha_file


def publish(args):
    candidate = args.candidate.resolve(strict=True)
    actual_file, actual_parameters = sha_file(candidate), inspect_candidate(candidate)
    if actual_file != args.file_sha256 or actual_parameters != args.parameter_hash:
        raise ValueError('Published source hashes do not match the downloaded candidate')
    now = datetime.now(timezone.utc)
    value = {'schema': EXPERIMENT, 'phase': args.phase, 'label': args.label,
             'candidate': str(candidate), 'fileSha256': actual_file, 'parameterHash': actual_parameters,
             'sensoryInterface': INTERFACE, 'physics': PHYSICS, 'fixedHash': FIXED_HASH,
             'validationStatus': args.validation_status, 'validationSummary': args.validation_summary,
             'trainingMethod': 'Recurrent supervised backpropagation on existing synaptic gains and neuronal excitability; fixed graph, local senses and motor decoder. No teacher or optimizer in this view.',
             'publishedAt': now.isoformat(), 'initialLayout': 'wide'}
    args.pointer.parent.mkdir(parents=True, exist_ok=True)
    history = args.pointer.parent / 'phase-history'
    history.mkdir(exist_ok=True)
    revision = history / (now.strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8] + '.json')
    revision.write_text(json.dumps(value, indent=2), encoding='utf-8')
    read_phase(revision)
    temporary = args.pointer.with_name(args.pointer.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_bytes(revision.read_bytes())
    temporary.replace(args.pointer)
    print(json.dumps({'published': True, 'phase': args.phase, 'parameterHash': actual_parameters,
                      'pointer': str(args.pointer), 'revision': str(revision)}), flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--pointer', type=Path, default=Path('.runtime/dashboard/latest-phase.json'))
    for name in ('phase', 'label', 'file-sha256', 'parameter-hash', 'validation-summary'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--validation-status', choices=('pending', 'incomplete', 'failed'), default='pending')
    publish(parser.parse_args())
