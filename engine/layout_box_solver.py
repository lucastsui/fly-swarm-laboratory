"""Bound-constrained low-rank least squares for existing brain parameters.

This is a training optimizer, not an inference network. Solve for normalized
parameter changes jointly under their box bounds; do not clip a solved neural
direction afterward. Float32 matvecs, float64 small eigenvalue calculation.
"""
import math
import torch


@torch.no_grad()
def solve_box(a, b, lower, upper, damping=.001, iterations=1024, tolerance=1e-5,
              on_progress=lambda value: None):
    if a.ndim != 2 or not a.numel() or b.shape != (a.shape[0],):
        raise ValueError('Finite nonempty matrix and aligned target required')
    if lower.shape != (a.shape[1],) or upper.shape != lower.shape:
        raise ValueError('Aligned coordinate bounds required')
    if any(t.device != a.device or t.dtype != a.dtype or not torch.isfinite(t).all()
           for t in (a, b, lower, upper)) or a.dtype not in (torch.float32, torch.float64):
        raise ValueError('Finite same-device floating inputs required')
    if not torch.all(lower <= 0) or not torch.all(upper >= 0):
        raise ValueError('Feasible zero change must lie inside all bounds')
    if not math.isfinite(damping) or damping <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive finite damping/tolerance required')
    if not isinstance(iterations, int) or not 1 <= iterations <= 4096:
        raise ValueError('Bounded positive iteration count required')
    gram = (a @ a.T).double()
    lipschitz = float(torch.linalg.eigvalsh(gram).max()) + damping
    lipschitz *= 1.00001
    coefficients = torch.linalg.solve(gram + damping*torch.eye(a.shape[0], device=a.device, dtype=torch.float64), b.double())
    raw = a.T @ coefficients.to(a.dtype)
    clip = lambda v: torch.maximum(lower, torch.minimum(upper, v))
    objective = lambda u: .5*float((a@u-b).square().sum()) + .5*damping*float(u.square().sum())
    clipped = clip(raw)
    zero = torch.zeros_like(lower)
    zero_objective, clipped_objective = objective(zero), objective(clipped)
    u = clipped.clone() if clipped_objective < zero_objective else zero
    best, best_objective = u.clone(), min(zero_objective, clipped_objective)
    extrapolated = u.clone(); acceleration = 1.
    converged = False; stationarity = None
    for iteration in range(1, iterations+1):
        gradient = a.T @ (a@extrapolated-b) + damping*extrapolated
        new = clip(extrapolated-gradient/lipschitz)
        next_acceleration = (1+math.sqrt(1+4*acceleration**2))/2
        extrapolated = new + ((acceleration-1)/next_acceleration)*(new-u)
        u, acceleration = new, next_acceleration
        if iteration % 32 == 0 or iteration == iterations:
            value = objective(u)
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite bound-constrained iterate')
            if value < best_objective:
                best, best_objective = u.clone(), value
            g = a.T @ (a@best-b) + damping*best
            stationarity = float((best-clip(best-g/lipschitz)).abs().max())*lipschitz
            converged = stationarity <= tolerance
            on_progress({'boxSolverIterations': iteration, 'objective': best_objective,
                         'projectedGradientInfinityNorm': stationarity, 'converged': converged})
            if converged:
                break
    if not torch.isfinite(best).all() or not torch.all((best >= lower) & (best <= upper)):
        raise AssertionError('Solver returned an infeasible direction')
    return best, {'iterations': iteration, 'iterationLimit': iterations, 'converged': converged,
                  'projectedGradientInfinityNorm': stationarity, 'tolerance': tolerance,
                  'damping': damping, 'zeroObjective': zero_objective,
                  'clippedUnconstrainedObjective': clipped_objective, 'boundedObjective': best_objective,
                  'predictedNormalizedOutputChange': (a@best).cpu().tolist(),
                  'objectiveMeaning': 'Row-normalized linear surrogate plus ridge, NOT task success'}


@torch.no_grad()
def bounded_direction(gradients, residual, base, caps=(.03, .0001), iterations=1024,
                      on_progress=lambda value: None):
    if len(base) != 2 or len(caps) != 2 or any(not math.isfinite(c) or c <= 0 for c in caps):
        raise ValueError('Two finite positive brain coordinate bounds required')
    if not gradients or len(residual) != len(gradients):
        raise ValueError('Aligned individual neural outputs required')
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
    u, report = solve_box(a, b, torch.cat(lower), torch.cat(upper), iterations=iterations, on_progress=on_progress)
    delta = [torch.zeros_like(parent) for parent in base]
    for d, ids, value, cap in zip(delta, indexes, u.split(dimensions), caps):
        d.reshape(-1)[ids] = value*cap
    report.update(activeJacobianCoordinates=dimensions, totalParameterCoordinates=[p.numel() for p in base],
                  coordinateCaps=list(caps), requestedOutputChange=list(residual),
                  predictedBoundedOutputChange=((a@u)*norms).cpu().tolist(),
                  globalParameterBounds=[list(pair) for pair in bounds],
                  exactZeroGradientCoordinatesOmittedOnly=True)
    return delta, report
