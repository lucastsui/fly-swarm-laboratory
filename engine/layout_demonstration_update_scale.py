"""Explicit bounded step-size experiments without resetting saved Adam.

The optimizer's base learning rates and moments are preserved in checkpoints.
Only its learning rates DURING an update are multiplied; forward computation,
labels, derivatives, clipping, physics and checkpoint ownership are unchanged.
"""
import contextlib
import json
import math
from pathlib import Path
from .layout_recovery_demonstrations import file_hash


def validate_scale(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not .1 <= value <= 4.:
        raise ValueError('Step scale must be finite and between 0.1 and 4')
    return float(value)


def resolve_update_scale(args, continuation):
    previous = 1.
    if continuation:
        record = continuation['record']
        path = Path(record['sourceRun'])/'manifest.json'
        if file_hash(path) != record['sourceFileSHA256']['manifest.json']:
            raise ValueError('Source manifest changed before update-scale selection')
        previous = validate_scale(json.loads(path.read_text()).get('effectiveStepScale', 1.))
    requested = getattr(args, 'revise_step_scale', None)
    current = previous if requested is None else validate_scale(requested)
    return current, {'previous': previous, 'effective': current, 'explicitRevision': requested is not None,
                     'changed': current != previous, 'baseAdamControlsPreserved': True,
                     'savedAdamMomentsPreserved': continuation is not None,
                     'forwardLabelsAndClippingUnchanged': True, 'isServiceEvidence': False}


@contextlib.contextmanager
def scaled_learning_rates(optimizer, scale):
    scale = validate_scale(scale)
    rates = [g['lr'] for g in optimizer.param_groups]
    try:
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate*scale
        yield
    finally:
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate
