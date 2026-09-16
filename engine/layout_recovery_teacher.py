"""Training-only local demonstration labels. Never imported by frozen service."""
import math
import numpy as np
from .haul_world import wrap
from .layout_world import color_sensory
from .layout_recovery_world import RecoveryWorld, available, accepts

LABEL_VERSION = 'local-status-return-grip-v1'
KINDS = ('compact', 'wide', 'rotated', 'permuted')
HARD_SEEDS = ((8400005, 'compact'), (8501003, 'wide'), (8501010, 'wide'), (8501011, 'wide'))


class LocalTeacher:
    """A label generator with memory of visibly blocked delivery boxes.

Memory is NOT fed to the brain. Only the recurrent connectome can remember
anything at execution. Geometry labels refer to currently visible targets.
"""
    def __init__(self):
        self.return_stage = 0

    def label(self, agent):
        retina = color_sensory(agent)[30:126].reshape(4, 24)
        visible = retina.max(1) >= .002
        target_index = None
        if self.return_stage != agent.cargo:
            self.return_stage = 0
        if agent.cargo:
            stage = agent.cargo
            if stage < 3 and visible[stage]:
                # Busy processing alone is temporary; full output needs a free
                # hand. Remember a full box until the material is returned, or
                # until the destination is seen to become available again.
                if agent.stations[stage]['stock'] >= 2:
                    self.return_stage = stage
                elif accepts(agent.stations[stage]):
                    self.return_stage = 0
            destination = stage-1 if self.return_stage else stage
            if visible[destination]:
                target_index = destination
        else:
            # Choose among visible supply only. Favor downstream work and do
            # not pick for a visibly full destination. This is a teacher label,
            # not an input encoder, and is absent from the deployed action path.
            candidates = [i for i in range(3) if visible[i] and available(agent.stations[i]) > 0
                          and not (i < 2 and visible[i+1] and agent.stations[i+1]['stock'] >= 2)]
            if candidates:
                target_index = max(candidates)
        interaction = .2  # raw readout / 40 = .005, below fixed grip threshold
        if target_index is not None:
            target = agent.stations[target_index]
            dx, dy = target['x']-agent.x, target['y']-agent.y
            distance = math.hypot(dx, dy)
            bearing = wrap(math.atan2(dy, dx)-agent.heading)
            proximity = float(np.clip((distance-.55)/.6, 0, 1))
            speed = proximity * max(0., math.cos(bearing))
            turn = float(np.clip(1.2*bearing, -1, 1))*proximity
            can_grip = (not agent.cargo or target_index == agent.cargo-1
                        or target_index == 3 or accepts(target))
            if distance < .8 and can_grip:
                interaction = 2.2
            return np.asarray([speed, turn, interaction], np.float32)
        near = agent.sensory()[26] > .4
        return np.asarray([.7 if near else .8, 1.2 if near else 0., interaction], np.float32)


def training_world(rng, index):
    """Mixture of old failures, normal starts and explicit recovery fixtures."""
    if index % 5 == 0:
        seed, kind = HARD_SEEDS[int(rng.integers(len(HARD_SEEDS)))]
    else:
        seed, kind = int(rng.integers(8600000, 8690000)), KINDS[index % 4]
    world = RecoveryWorld(seed=seed, kind=kind, random_starts=index % 3 == 1)
    scenario = 'normal'
    if index % 4 == 2:
        scenario = 'injected-loaded-stages-training-only'
        for i, agent in enumerate(world.agents):
            agent.cargo = i
        for station in world.stations[1:3]:
            station['stock'] = int(rng.integers(3))
    elif index % 4 == 3:
        scenario = 'injected-full-buffer-recovery-training-only'
        # Start close enough to observe a blocked box, with room to move.
        world.stations[2]['stock'] = 2
        target = world.stations[2]
        for i, agent in enumerate(world.agents):
            angle = i*math.pi/2
            agent.x = float(np.clip(target['x']+1.2*math.cos(angle), .23, 19.77))
            agent.y = float(np.clip(target['y']+1.2*math.sin(angle), .23, 13.77))
            agent.heading = math.atan2(target['y']-agent.y, target['x']-agent.x)
            agent.cargo = 2
            agent.trajectory.clear()
            agent.trajectory.append([agent.x, agent.y])
        world.refresh_landmarks()
    return world, {'seed': seed, 'kind': kind, 'scenario': scenario}
