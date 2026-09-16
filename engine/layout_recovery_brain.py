"""Fixed extra sensory projection; all computation remains in existing cells."""
from pathlib import Path
import numpy as np
import torch
from .layout_excitability import ExcitableConnectome
from .layout_recovery_world import CHANNELS, INTERFACE


def status_projection(annotated):
    mapping = np.zeros(len(annotated), np.int64)
    mask = np.zeros(len(annotated), np.float32)
    for bearing in range(24):
        cells = np.flatnonzero((annotated >= 30) & (annotated < 126) & ((annotated-30) % 24 == bearing))
        if len(cells) < 7:
            raise ValueError('Insufficient existing retinal cells for status channels')
        for feature in range(7):
            mapping[cells[feature::7]] = 129 + 24*feature + bearing
        mask[cells] = 1.
    return mapping, mask


class RecoveryConnectome(ExcitableConnectome):
    def __init__(self, root, gains, tonic, surrogate=False, status_scale=.15):
        super().__init__(root, gains, tonic, annotated=True, surrogate=surrogate)
        if not 0 <= status_scale <= .5:
            raise ValueError('Invalid fixed status scale')
        indices, mask = status_projection(self.annotated_channels.cpu().numpy())
        device = self.tonic.device
        self.register_buffer('status_channels', torch.as_tensor(indices, device=device))
        self.register_buffer('status_mask', torch.as_tensor(mask, device=device))
        self.register_buffer('status_scale', torch.tensor(status_scale, device=device))
        self.input_channels = CHANNELS
        self.interface = INTERFACE
        self.fixed_hash = self.fingerprint()

    def sensory_drive(self, observations):
        return (super().sensory_drive(observations) + self.status_scale
                * observations.T[self.status_channels] * self.status_mask[:, None])


def load_model(root, candidate, surrogate=False, migrate=False, status_scale=.15):
    with np.load(candidate, allow_pickle=False) as data:
        interface = str(data['interface'])
        if interface != INTERFACE and not (migrate and interface == 'annotated-color-cargo-v1'):
            raise ValueError('Explicit migration required; wrong checkpoint interface')
        if interface == INTERFACE:
            status_scale = float(data['status_scale'])
        model = RecoveryConnectome(root, data['gains'].copy(), data['tonic'].copy(), surrogate, status_scale)
        if interface == INTERFACE and str(data['fixed_hash']) != model.fixed_hash:
            raise ValueError('Fixed graph / sensory / motor fingerprint mismatch')
    return model


def save_model(path, model):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve all prior candidates')
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, gains=model.log_gains.detach().cpu().numpy(),
                            tonic=model.tonic.detach().cpu().numpy(), interface=np.asarray(model.interface),
                            status_scale=model.status_scale.cpu().numpy(), fixed_hash=np.asarray(model.fixed_hash))
    temporary.replace(path)
