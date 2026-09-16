"""Versioned experimental physics: deliberate returns, local status vision.

No controller runs here. The same three fixed motor outputs move, turn and
operate the grip. Returns require carrying the item back to its source box.
Each source has a separate unlimited return tray, NOT extra processing capacity.
The legacy world is untouched. This is engineered embodiment, not fly biology.
"""
import math
import numpy as np
from .haul_world import HaulWorld
from .layout_world import LayoutSwarmWorld, color_sensory
from .plastic_brain import DT

PHYSICS = 'source-return-trays-v1'
INTERFACE = 'annotated-color-cargo-status-v2'
CHANNELS = 297  # 129 old + 3 output + 2 accept + 2 processing retinal maps


def available(station):
    return station['stock'] + station.get('returned', 0)


def accepts(station):
    return station['timer'] == 0 and station['stock'] < 2


class ReturnFly(HaulWorld):
    def interact(self, motor):
        self.cooldown = max(0., self.cooldown - DT)
        if not motor['interact'] or self.cooldown > 0:
            return 0.
        self.cooldown = 1.
        self.interactions += 1
        distances = [math.hypot(self.x-s['x'], self.y-s['y']) for s in self.stations]
        index = int(np.argmin(distances))
        station = self.stations[index]
        if distances[index] >= .8:
            return 0.
        if self.cargo and index == self.cargo and (index == 3 or accepts(station)):
            self.transfers += 1
            if index == 3:
                self.deliveries += 1
                reward = 1.
                self.events.append('Finished product delivered')
            else:
                station['timer'] = 1. if index == 1 else 1.5
                reward = .2
                self.events.append('Material supplied to '+station['kind'])
            self.cargo = 0
            return reward
        if self.cargo and index == self.cargo-1:
            station['returned'] += 1
            self.cargo = 0
            self.returns += 1
            self.events.append('Material placed in source return tray')
        elif not self.cargo and index < 3 and available(station) > 0:
            key = 'returned' if station['returned'] > 0 else 'stock'
            station[key] -= 1
            self.cargo = index+1
            self.pickups += 1
            self.events.append('Picked up from '+station['kind'])
        return 0.


def recovery_sensory(agent):
    original = color_sensory(agent)
    colors = original[30:126].reshape(4, 24)
    # Every property is gated only by visibility of its own box. Never cargo,
    # selected target, global stock, or teacher state. Returned material stays
    # visible; pending processing is a separate channel from finished output.
    output = colors[:3] * np.asarray([min(available(s), 3)/3 for s in agent.stations[:3]])[:, None]
    capacity = colors[1:3] * np.asarray([accepts(s) for s in agent.stations[1:3]])[:, None]
    processing = colors[1:3] * np.asarray([s['timer'] > 0 for s in agent.stations[1:3]])[:, None]
    return np.r_[original, output.ravel(), capacity.ravel(), processing.ravel()].astype(np.float32)


class RecoveryWorld(LayoutSwarmWorld):
    physics = PHYSICS

    def __init__(self, *args, random_starts=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Promote existing body instances without rerunning initialization or
        # disturbing independent trajectories/shared station ownership.
        for station in self.stations:
            station['returned'] = 0
        for agent in self.agents:
            agent.__class__ = ReturnFly
            agent.returns = 0
        if random_starts:
            rng = np.random.default_rng(self.agents[0].rng.integers(1, 2**31))
            for i, agent in enumerate(self.agents):
                for _ in range(10000):
                    x, y = rng.uniform([.3, .3], [19.7, 13.7])
                    if all(math.hypot(x-b.x, y-b.y) > .48 for b in self.agents[:i]):
                        break
                else:
                    raise ValueError('Cannot place bodies')
                agent.x, agent.y = float(x), float(y)
                agent.trajectory.clear()
                agent.trajectory.append([agent.x, agent.y])
        self.refresh_landmarks()

    def sensory(self):
        self.refresh_landmarks()
        return np.asarray([recovery_sensory(a) for a in self.agents])

    @property
    def returns(self):
        return sum(a.returns for a in self.agents)

    def material_count(self):
        """Conservation count for finite-supply tests; one item per recipe."""
        return (sum(available(s) + int(s['timer'] > 0) for s in self.stations)
                + sum(bool(a.cargo) for a in self.agents) + self.deliveries)

    def snapshot(self, motors):
        value = super().snapshot(motors)
        value['physics'] = self.physics
        value['returns'] = self.returns
        value['returnTrayCapacity'] = 'unlimited; separate from processing buffer'
        return value
