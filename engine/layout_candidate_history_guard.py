"""Additional candidate-full-history guard; original margin budgets unchanged.

This frozen replay recomputes states with the temporary candidate from the
beginning of each recorded TRAINING trajectory. It keeps the parent's motion
targets and all observations/labels. It is not a physical service test.
"""
import copy
import numpy as np
import torch
from .layout_batched_prefix_cache import fill_states
from .layout_contextual_operation_probe import measure_bank
from .layout_margin_projected_train import margin_gate


def validate_histories(bank, episodes):
    windows = bank.manifest['windows']
    if not episodes or not windows:
        raise ValueError('Full original physical histories required')
    for i, w in enumerate(windows):
        if not 0 <= w['episode'] < len(episodes):
            raise ValueError('Invalid episode mapping')
        arrays, meta = episodes[w['episode']]
        sl, fly = slice(w['start'],w['stop']), w['fly']
        if (meta['seed'] != w['seed'] or meta['kind'] != w['kind']
                or not np.array_equal(bank.x[i], arrays['observations'][sl,fly:fly+1])
                or not np.array_equal(bank.y[i], arrays['labels'][sl,fly:fly+1])):
            raise ValueError('Original observations, labels or episode identity changed')


@torch.no_grad()
def candidate_history_guard(model, bank, episodes, original_measurement):
    validate_histories(bank, episodes)
    expected, fixed, mode = model.checkpoint_hash(), model.fingerprint(), model.training
    parameters = list(model.parameters())
    requires_grad = [p.requires_grad for p in parameters]
    if expected == bank.manifest['parameterHash'] or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Changed temporary candidate without accumulated gradients required')
    try:
        model.eval().requires_grad_(False)
        states = fill_states(model, episodes, bank.manifest['windows'], 32)
        candidate = copy.copy(bank)
        candidate.states = np.stack([states[i,:,w['fly']:w['fly']+1]
                                    for i,w in enumerate(bank.manifest['windows'])])
        if (candidate.states.shape != bank.states.shape or candidate.states.dtype != np.float32
                or not np.isfinite(candidate.states).all()):
            raise ValueError('Finite aligned full-history neuronal states required')
        del states
        measurement = measure_bank(model, candidate, list(range(len(bank.manifest['windows']))))
        return {'schema': 'additional-candidate-full-history-guard-v1',
                'candidateParameterHash': expected, 'parentParameterHash': bank.manifest['parameterHash'],
                'gate': margin_gate(original_measurement, measurement, 0.), 'measurement': measurement,
                'candidateFullHistoryPrefix': True, 'originalMotionTargetsRetained': True,
                'originalGuardThresholdsUnchanged': True, 'physicalActionsExecuted': False,
                'optimizerUsed': False, 'isServiceEvidence': False}
    finally:
        for p, original in zip(parameters, requires_grad):
            p.requires_grad_(original)
        model.train(mode)
        if model.checkpoint_hash() != expected or model.fingerprint() != fixed or any(p.grad is not None for p in model.parameters()):
            raise AssertionError('Frozen candidate history guard changed the model')
