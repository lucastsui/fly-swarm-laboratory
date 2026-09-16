"""Training-only prerequisite curriculum; no inference or body control code."""
import numpy as np


def select_microfit_scenes(metadata, targets):
    """One near-box scene per role, each with both grip-on and grip-off cargo."""
    selected = []
    for box in range(4):
        matches = [i for i, item in enumerate(metadata) if item['box'] == box and item['near']
                   and (targets[:, 4*i:4*i+4, 2] > 1.).any()
                   and (targets[:, 4*i:4*i+4, 2] <= 1.).any()]
        if not matches:
            raise ValueError('Calibration bank lacks a mixed-cargo scene for every box role')
        selected.append(matches[0])
    return selected


def microfit_passes(fit):
    """A necessary TRAINING-fit gate, never the separate physical release gate."""
    values = [fit.get('gripRecall'), fit.get('gripFalsePositiveRate'), *fit.get('parentMotionMSE', [])]
    if len(values) != 4 or any(v is None or not np.isfinite(v) for v in values):
        return False
    return bool(values[0] >= .9 and values[1] <= .1 and max(values[2:]) <= .05)


def discard_pending_warmup_replay(exchange):
    """Clear only in-memory pending replay; all raw evidence files stay intact."""
    with exchange.lock:
        paths = [path.name for path in exchange.queue]
        exchange.queue.clear()
    return {'reason': 'warmup complete; collect fresh canonical-version experience',
            'rawExperiencePreserved': True, 'discardedPendingIds': paths}
