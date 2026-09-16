"""Training-only family coverage; no inference controller or sensory selector."""
from .layout_recovery_curriculum import balanced_world, STRATIFIED_VERSION
from .layout_recovery_teacher import KINDS, training_world


def worker_world(rng, episode, slot, *, balanced=False, stratified=False):
    if stratified:
        if not balanced or slot not in range(len(KINDS)):
            raise ValueError('Stratification requires four balanced worker slots')
        # Keep each slot's family through independently timed episode resets.
        # Scenario, cargo and starting pose remain independently sampled.
        return balanced_world(rng, episode, fixed_kind=KINDS[slot])
    return (balanced_world if balanced else training_world)(rng, episode)


def validate_family_coverage(metadata, actors=None):
    worlds = metadata.get('worlds', [])
    if (len(worlds) != len(KINDS) or (actors is not None and actors != 4*len(KINDS))
            or {w.get('kind') for w in worlds} != set(KINDS)
            or any(w.get('familyStratified') is not True or w.get('curriculum') != STRATIFIED_VERSION
                   for w in worlds)):
        raise ValueError('Correction lacks the required four-family training coverage')
