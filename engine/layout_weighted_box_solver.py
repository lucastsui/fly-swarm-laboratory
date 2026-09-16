"""Weighted normalized output penalties; original box solver and bounds retained.

Weights scale TRAINING squared residuals, not movement outputs or acceptance
budgets. Returned predicted changes stay in unweighted neural-output units.
"""
import math
import torch
from .layout_box_solver import solve_box


@torch.no_grad()
def bounded_direction(gradients, residual, base, row_weights, caps=(.03, .0001), iterations=1024,
                      on_progress=lambda value: None):
    if len(base) != 2 or len(caps) != 2 or any(not math.isfinite(c) or c <= 0 for c in caps):
        raise ValueError('Two finite positive brain coordinate bounds required')
    if not gradients or len(residual) != len(gradients):
        raise ValueError('Aligned individual neural outputs required')
    if len(row_weights) != len(gradients) or any(not math.isfinite(w) or w <= 0 for w in row_weights):
        raise ValueError('One finite positive squared-error weight per output required')
    bounds = ((-2., 2.), (-.1, .1))
    matrices, indexes, lower, upper, dimensions = [], [], [], [], []
    for block, (parent, cap, limits) in enumerate(zip(base, caps, bounds)):
        if not parent.numel() or not torch.isfinite(parent).all() or not torch.all((parent >= limits[0]) & (parent <= limits[1])):
            raise ValueError('Original parent must lie within global parameter bounds')
        if any(len(row) != 2 or row[block].shape != parent.shape or not torch.isfinite(row[block]).all() for row in gradients):
            raise ValueError('Finite aligned brain Jacobian required')
        matrix = torch.stack([row[block].reshape(-1) for row in gradients])*cap
        ids = torch.nonzero(matrix.abs().amax(0) > 0).flatten()
        matrices.append(matrix[:, ids]); indexes.append(ids); dimensions.append(len(ids))
        lower.append(torch.clamp((limits[0]-parent.reshape(-1)[ids])/cap, min=-1.))
        upper.append(torch.clamp((limits[1]-parent.reshape(-1)[ids])/cap, max=1.))
    a = torch.cat(matrices, dim=1)
    norms = a.square().sum(1).sqrt()
    if not torch.isfinite(norms).all() or not torch.all(norms > 0):
        raise ValueError('Every requested output must have a nonzero finite gradient')
    a.div_(norms[:, None])
    b = torch.as_tensor(residual, dtype=a.dtype, device=a.device)/norms
    weights = torch.as_tensor(row_weights, dtype=a.dtype, device=a.device)
    square_roots = weights.sqrt()
    u, report = solve_box(a*square_roots[:, None], b*square_roots, torch.cat(lower), torch.cat(upper),
                          iterations=iterations, on_progress=on_progress)
    report['predictedWeightedNormalizedOutputChange'] = report['predictedNormalizedOutputChange']
    report['predictedNormalizedOutputChange'] = (a@u).cpu().tolist()
    report['squaredResidualWeights'] = list(row_weights)
    report['objectiveMeaning'] = 'Weighted row-normalized linear squared residual plus original ridge; NOT task success'
    delta = [torch.zeros_like(parent) for parent in base]
    for d, ids, value, cap in zip(delta, indexes, u.split(dimensions), caps):
        d.reshape(-1)[ids] = value*cap
    report.update(activeJacobianCoordinates=dimensions, totalParameterCoordinates=[p.numel() for p in base],
                  coordinateCaps=list(caps), requestedOutputChange=list(residual),
                  predictedBoundedOutputChange=((a@u)*norms).cpu().tolist(),
                  globalParameterBounds=[list(pair) for pair in bounds],
                  exactZeroGradientCoordinatesOmittedOnly=True)
    return delta, report
