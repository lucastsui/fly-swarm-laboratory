"""Read-only exact output Jacobians on original physical operation histories.

No optimizer, perturbation, inference aid, teacher action or saved candidate.
Context selection affects diagnostics only, never the brain input.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .layout_operation_sampling import OperationCache
from .layout_operation_prefix import CONTEXTS, KINDS
from .layout_decision_resume import AnchoredOperationCache
from .layout_operation_focus import focus_values
from .layout_recovery_train import recurrent_predictions
from .layout_recovery_brain import load_model
from .layout_recovery_demonstrations import file_hash
from .layout_recovery_protocol import atomic_json


def geometry(gradients, scales=(.003, .00001)):
    if len(gradients) < 2 or any(len(row) != 2 for row in gradients):
        raise ValueError('Require multiple two-block Jacobian rows')
    if len(scales) != 2 or any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Positive finite parameter-coordinate scales required')
    grams, blocks = [], {}
    for block, name in enumerate(('log_gains', 'tonic')):
        values = [row[block].detach().reshape(-1) for row in gradients]
        if any(v.shape != values[0].shape or not torch.isfinite(v).all() for v in values):
            raise ValueError('Invalid exact gradient')
        matrix = torch.stack(values).double()
        gram = (matrix @ matrix.T).cpu().numpy()
        del matrix
        norms = np.sqrt(np.maximum(0, np.diag(gram)))
        denom = norms[:, None]*norms[None, :]
        cosine = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        blocks[name] = {'rowNorms': norms.tolist(), 'cosines': cosine.tolist(),
                        'zeroRows': np.flatnonzero(norms == 0).tolist()}
        grams.append(gram*scales[block]**2)
    combined = sum(grams)
    norms = np.sqrt(np.maximum(0, np.diag(combined)))
    denom = norms[:, None]*norms[None, :]
    cosine = np.divide(combined, denom, out=np.zeros_like(combined), where=denom > 0)
    return {'parameterBlocks': blocks, 'coordinateScales': list(scales),
            'scaledOutputGram': combined.tolist(), 'scaledRowNorms': norms.tolist(),
            'scaledCosines': cosine.tolist(),
            'normalizedGramEigenvalues': np.linalg.eigvalsh(cosine).tolist()}


def probe(model, bank):
    before, fixed = model.checkpoint_hash(), model.fingerprint()
    if model.surrogate_training or before != bank.manifest['parameterHash']:
        raise ValueError('Exact derivative and exact current full-history prefix required')
    parameters = (model.log_gains, model.tonic)
    if any(p.grad is not None for p in parameters):
        raise ValueError('Probe must start without accumulated gradients')
    ids = [bank.groups[k, c][0] for k in KINDS for c in CONTEXTS]
    selections = [bank.manifest['windows'][i] for i in ids]
    x, y, state, motion = bank.batch(ids, model.tonic.device)
    prediction = recurrent_predictions(model, x, state, 64, model.weights(), gradient_start=0)
    grip, target = focus_values(prediction, y[64:], selections)
    indexes = [[i for i, s in enumerate(selections) if s['category'] == c] for c in CONTEXTS]
    outputs = [grip[i].mean() for i in indexes]
    outputs += [prediction[..., head].mean() for head in range(2)]
    names = list(CONTEXTS)+['mean-forward-output', 'mean-turn-output']
    gradients = []
    for i, output in enumerate(outputs):
        gradients.append([g.detach() for g in torch.autograd.grad(output, parameters, retain_graph=i < len(outputs)-1)])
        print(json.dumps({'exactJacobianRow': names[i], 'rowsComplete': i+1, 'parametersUpdated': False}), flush=True)
    result = geometry(gradients)
    result.update(rowNames=names, meanRawOutputs=[float(o.detach()) for o in outputs],
                  parentMotionMSE=(prediction[..., :2]-motion).square().mean((0, 1)).detach().cpu().tolist(),
                  windows=[{'windowIndex': i, **s, 'rawGrip': float(grip[j].detach()),
                            'originalTarget': float(target[j]), 'correct': bool((grip[j] > 1) == (target[j] > 1))}
                           for j, (i, s) in enumerate(zip(ids, selections))])
    if model.checkpoint_hash() != before or model.fingerprint() != fixed or any(p.grad is not None for p in parameters):
        raise AssertionError('Read-only Jacobian probe changed brain')
    return result


def main(args):
    if args.out.exists():
        raise FileExistsError('Preserve diagnostic')
    torch.set_num_threads(4)
    model = load_model(args.root, args.candidate).train()
    if model.checkpoint_hash() != args.expected_parameter_hash:
        raise ValueError('Wrong selected candidate')
    fresh = OperationCache(args.cache, args.dataset, args.expected_parameter_hash, model.fixed_hash, model.n)
    original = OperationCache(args.original_cache, args.dataset, args.original_parameter_hash, model.fixed_hash, model.n)
    if fresh.manifest['candidateFileHash'] != file_hash(args.candidate):
        raise ValueError('Exact original candidate prefix required')
    bank = AnchoredOperationCache(fresh, original)
    paths = [args.candidate, args.dataset/'manifest.json']
    paths += [folder/name for folder in (args.cache, args.original_cache) for name in ('cache.npz','manifest.json')]
    paths += [args.dataset/e['file'] for e in bank.manifest['datasetFiles']]
    hashes = {str(p.resolve()): file_hash(p) for p in paths}
    names = set(bank.manifest['sourceHashes']) | {'layout_context_credit_probe.py','layout_operation_sampling.py',
        'layout_decision_resume.py','layout_operation_focus.py','layout_recovery_train.py','layout_recovery_brain.py'}
    sources = {n: file_hash(Path(__file__).parent/n) for n in sorted(names)}
    result = probe(model, bank)
    if any(file_hash(Path(p)) != h for p,h in hashes.items()) or any(file_hash(Path(__file__).parent/n) != h for n,h in sources.items()):
        raise AssertionError('Source/checkpoint/data changed during diagnostic')
    result.update(parameterHash=model.checkpoint_hash(), fixedHash=model.fixed_hash,
                  prefixIsExactForCandidate=True, motionAnchorParameterHash=args.original_parameter_hash,
                  inputFileHashes=hashes, sourceHashes=sources, optimizerUsed=False, parametersUnchanged=True,
                  candidateSaved=False, physicalActionsExecuted=False, isServiceEvidence=False,
                  qualification='Six mean-output Jacobian rows on 16 selected training windows; not individual-window separability or service proof')
    atomic_json(args.out, result)
    print('CONTEXT_CREDIT_COMPLETE '+json.dumps({k:result[k] for k in ('parameterHash','rowNames','meanRawOutputs','scaledCosines','normalizedGramEigenvalues')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','candidate','cache','original-cache','dataset','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-parameter-hash', required=True)
    p.add_argument('--original-parameter-hash', required=True)
    main(p.parse_args())
