"""Original frozen physical evaluator under explicit deterministic GPU rules.

Keep original world/score/evaluator code unchanged. This is a NEW paired
evaluation protocol; do not compare it as if the numerical mode were unchanged
from legacy nondeterministic runs. Unsupported deterministic ops fail closed.
"""
import argparse
import json
import os
from pathlib import Path
import torch
from . import layout_recovery_eval
from .layout_correction_prefix import DATA_SOURCES
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def validate_workspace(value):
    if value not in (':4096:8', ':16:8'):
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG before starting the Python process')


def validate_repeatability(proof, candidate_file_hash):
    if (proof.get('schema') != 'frozen-forward-repeatability-v1'
            or proof.get('hashes', {}).get('candidate') != candidate_file_hash
            or proof.get('parametersUnchanged') is not True):
        raise ValueError('Matching original frozen repeatability proof required')
    rows = [r for r in proof['results'] if r['strictDeterministicAlgorithms']]
    if len(rows) != 1 or not rows[0].get('finished') or len(rows[0]['runs']) != 3:
        raise ValueError('Three completed strict-mode repeats required')
    runs = rows[0]['runs']
    if (len({r['outputHash'] for r in runs}) != 1 or len({r['finalStateHash'] for r in runs}) != 1
            or any(not r['versusFirst']['bitwiseEqual'] for r in runs[1:])):
        raise ValueError('Strict mode did not demonstrate repeatable inference; do not run this protocol')


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve all prior evaluations')
    validate_workspace(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    proof_path = args.repeatability_proof
    proof = json.loads(proof_path.read_text())
    candidate_hash = file_hash(args.candidate)
    validate_repeatability(proof, candidate_hash)
    if (proof['torch'] != torch.__version__ or proof['cuda'] != torch.version.cuda
            or proof['device'] != torch.cuda.get_device_name()
            or proof['workspaceConfig'] != os.environ['CUBLAS_WORKSPACE_CONFIG']
            or any(file_hash(Path(__file__).with_name(n)) != h for n,h in proof['sourceHashes'].items())):
        raise ValueError('Original repeatability environment or sources changed')
    # The original evaluator must receive only its original arguments.
    delattr(args, 'repeatability_proof')
    torch.use_deterministic_algorithms(True)
    names = set(DATA_SOURCES) | {'layout_recovery_eval.py', Path(__file__).name}
    sources = {n: file_hash(Path(__file__).with_name(n)) for n in names}
    try:
        layout_recovery_eval.main(args)
        if (not torch.are_deterministic_algorithms_enabled()
                or file_hash(args.candidate) != candidate_hash
                or any(file_hash(Path(__file__).with_name(n)) != h for n,h in sources.items())):
            raise ValueError('Evaluation numerical rules or source inputs changed')
        report = json.loads((args.out/'evaluation.json').read_text())
        atomic_json(args.out/'numerical-protocol.json', {
            'schema': 'strict-deterministic-frozen-service-v1', 'finished': True,
            'strictDeterministicAlgorithms': True, 'warnOnly': False,
            'workspaceConfig': os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(), 'sourceHashes': sources,
            'candidateFileHash': candidate_hash, 'parameterHash': report['checkpointHash'],
            'repeatabilityProofHash': file_hash(proof_path),
            'fixedHash': report['fixedHash'], 'evaluationHash': file_hash(args.out/'evaluation.json'),
            'manifestHash': file_hash(args.out/'manifest.json'),
            'originalPhysicsAndServiceCriteriaUnchanged': True,
            'legacyNumericalProtocolMatched': False, 'requiresMatchedDeterministicBaseline': True})
    except BaseException as error:
        if args.out.exists():
            atomic_json(args.out/'deterministic-failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'candidate', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--repeatability-proof', type=Path, required=True)
    p.add_argument('--cases', type=int, default=8)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--kinds', default='compact,wide,rotated,permuted')
    p.add_argument('--random-starts', action='store_true')
    p.add_argument('--move-at', type=int, default=0)
    p.add_argument('--original-senses', action='store_true')
    p.add_argument('--migrate', action='store_true')
    p.add_argument('--interaction-events', action='store_true')
    main(p.parse_args())
