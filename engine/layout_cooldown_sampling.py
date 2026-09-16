"""Explicit timing-aware TRAINING overlay; original prefix cache is immutable.

All original inputs, labels and physical event actors are checked first. Only
the separately versioned, physically replayed grip targets are substituted.
This is NOT used for learner-only network packets or any inference actions.
"""
from pathlib import Path
import numpy as np
import torch
from .layout_cooldown_dataset import load_revision
from .layout_demonstration_focus import EventFocusedDemonstrations
from .layout_recovery_demonstrations import file_hash


class CooldownFocusedDemonstrations:
    def __init__(self, bank, dataset, revision):
        self.bank = bank
        # This checks every cached original observation/label and event actor.
        self.focused = EventFocusedDemonstrations(bank, dataset)
        self.revised, self.revision = load_revision(revision, dataset)
        if self.revision['datasetManifestHash'] != bank.manifest['datasetManifestHash']:
            raise ValueError('Timing revision does not match original prefix data')
        self.manifest = bank.manifest
        self.record = {'sampling': 'timing-aware-one-physical-event-actor-per-window',
                       'originalPhysicalData': self.focused.record,
                       'labelVersion': self.revision['labelVersion'],
                       'labelRevisionManifestHash': file_hash(Path(revision)/'manifest.json'),
                       'allEpisodeReplayVerified': True, 'originalCacheUnmodified': True,
                       'teacherMotionLabelsPreserved': True, 'teacherAtInference': False,
                       'networkPacketLabelSemanticsChanged': False, 'isServiceEvidence': False,
                       'sourceHash': file_hash(Path(__file__))}

    def revised_window(self, index):
        s = self.bank.manifest['windows'][index]
        revised = self.revised[s['episode']][s['start']:s['stop']]
        if (revised.shape != self.bank.y[index].shape
                or not np.array_equal(revised[..., :2], self.bank.y[index, ..., :2])):
            raise ValueError('Physical window/motion target mismatch')
        return revised

    def sample(self, rng, device):
        x, original, state, meta = self.focused.sample(rng, device)
        parts = [self.revised_window(s['windowIndex'])[:, s['fly']:s['fly']+1]
                 for s in meta['selections']]
        target = torch.as_tensor(np.concatenate(parts, axis=1), device=device)
        if not torch.equal(target[..., :2], original[..., :2]):
            raise ValueError('Selected teacher movement changed')
        return x, target, state, {**meta, 'sampling': self.record['sampling'],
                                 'labelVersion': self.record['labelVersion'],
                                 'labelRevisionManifestHash': self.record['labelRevisionManifestHash'],
                                 'changedGripLabels': int((target[..., 2] != original[..., 2]).sum())}

    def batch(self, indexes, device):
        x, original, state = self.bank.batch(indexes, device)
        target = torch.as_tensor(np.concatenate([self.revised_window(i) for i in indexes], axis=1), device=device)
        if not torch.equal(target[..., :2], original[..., :2]):
            raise ValueError('Diagnostic teacher movement changed')
        return x, target, state

    def diagnostic_indexes(self):
        return self.bank.diagnostic_indexes()
