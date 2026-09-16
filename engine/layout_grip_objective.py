"""Training objectives for the EXISTING binary grip threshold, not a decoder.

Raw MN9 readout is 40 times its mean rate. The unchanged body threshold .025
therefore corresponds to raw readout 1.0. We need correct sides of that boundary,
not a regression to arbitrary amplitudes 2.2 and .2.
"""
import torch

VERSION = 'fixed-threshold-balanced-hinge-v1'
THRESHOLD = 1.


def check_margin(margin):
    if not 0 < margin <= .5:
        raise ValueError('Grip margin must be in (0, .5]')


def threshold_grip_loss(prediction, target, margin=.1):
    check_margin(margin)
    pred, label = prediction.reshape(-1), target.reshape(-1)
    positive = (label > THRESHOLD).to(pred.dtype)
    negative = 1.-positive
    pc, nc = positive.sum(), negative.sum()
    on = (torch.relu(THRESHOLD+margin-pred).square()*positive).sum()/pc.clamp_min(1)
    off = (torch.relu(pred-(THRESHOLD-margin)).square()*negative).sum()/nc.clamp_min(1)
    count = (pc > 0).to(pred.dtype)+(nc > 0).to(pred.dtype)
    return (on+off)/count.clamp_min(1)


def matched_grip_ranking(prediction, target, margin=.1):
    """Require positive-minus-negative separation within each SAME-view group."""
    check_margin(margin)
    if prediction.shape[-1] != 4 or target.shape != prediction.shape:
        raise ValueError('Expected intact four-cargo groups')
    positive = target > THRESHOLD
    pairs = (positive[..., :, None] & ~positive[..., None, :]).to(prediction.dtype)
    gap = prediction[..., :, None]-prediction[..., None, :]
    error = torch.relu(2*margin-gap).square()
    return (pairs*error).sum()/pairs.sum().clamp_min(1)
