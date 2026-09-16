"""Training curriculum that cannot omit loaded-cargo interaction examples.

Each family contributes pickup, loaded transfer/delivery, post-pickup release,
and uniform exploration. Selection metadata is NEVER added to sensory inputs.
"""
import numpy as np
import torch
from .layout_cooldown_sampling import CooldownFocusedDemonstrations
from .layout_recovery_teacher import KINDS

VERSION = 'four-family-four-service-contexts-v1'


def service_buckets(groups):
    result = {}
    for kind in KINDS:
        categories = groups[kind]
        buckets = {
            'pickup': sorted(c for c in categories if c.startswith('pickup-')),
            'loaded-operation': sorted(c for c in categories if c.startswith(('transfer-', 'delivery-', 'return-'))),
            'after-pickup': sorted(c for c in categories if c.startswith('after-pickup-')),
            'exploration': ['uniform-trajectory'] if 'uniform-trajectory' in categories else [],
        }
        if any(not names or any(not categories[n] for n in names) for names in buckets.values()):
            raise ValueError('Every family must contain every service training context')
        result[kind] = buckets
    return result


class ServiceBalancedDemonstrations(CooldownFocusedDemonstrations):
    def __init__(self, bank, dataset, revision):
        super().__init__(bank, dataset, revision)
        self.groups = {k: {} for k in KINDS}
        self.choices = {}
        burn, frames = bank.manifest['burn'], bank.manifest['gradientFrames']
        for index, spec in enumerate(bank.manifest['windows']):
            choices = self.focused.choices[index]
            if spec['category'].startswith(('pickup-', 'transfer-', 'delivery-', 'return-')):
                # A window clipped near episode start can put its event before
                # the loss. It cannot count as a positive interaction example.
                target = self.revised_window(index)
                choices = [c for c in choices if burn <= c['eventTick']-spec['start'] < burn+frames
                           and target[c['eventTick']-spec['start'], c['fly'], 2] > 1.]
            if choices:
                self.choices[index] = choices
                self.groups[spec['kind']].setdefault(spec['category'], []).append(index)
        self.buckets = service_buckets(self.groups)
        self.record.update(sampling=VERSION, requiredContexts=list(self.buckets[KINDS[0]]),
                           eachFamilyAndContextInEveryBatch=True, interactionEventsInsideLossRequired=True,
                           eligibleWindows=len(self.choices), externalDecisionNetwork=False)

    def sample(self, rng, device):
        selections, inputs, targets, states = [], [], [], []
        for kind in KINDS:
            for context, categories in self.buckets[kind].items():
                category = str(rng.choice(categories))
                index = int(rng.choice(self.groups[kind][category]))
                choices = self.choices[index]
                choice = choices[int(rng.integers(len(choices)))]
                fly = choice['fly']
                inputs.append(self.bank.x[index, :, fly:fly+1])
                targets.append(self.revised_window(index)[:, fly:fly+1])
                states.append(self.bank.states[index, :, fly:fly+1])
                selections.append({'windowIndex': index, **self.bank.manifest['windows'][index],
                                   **choice, 'trainingContext': context})
        values = [torch.as_tensor(np.concatenate(parts, axis=1), device=device)
                  for parts in (inputs, targets, states)]
        return (*values, {'sampling': VERSION, 'selections': selections, 'actors': 16,
                          'prefixStateUnchanged': True, 'noNewBrainInputs': True,
                          'labelVersion': self.record['labelVersion'],
                          'labelRevisionManifestHash': self.record['labelRevisionManifestHash'],
                          'everyFamilyHasLoadedOperation': True})
