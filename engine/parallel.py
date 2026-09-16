"""Independent assembly lines, with cooperating couriers inside each line."""
from collections import deque

import numpy as np


class ParallelFactories:
    def __init__(self, factory_class, environments, flies_per_factory, seed=41):
        self.worlds = [factory_class(flies_per_factory, seed=seed + i * 997)
                       for i in range(environments)]
        self.flies_per_factory = flies_per_factory
        self.flies = [fly for world in self.worlds for fly in world.flies]
        self.owners = {id(fly): world for world in self.worlds for fly in world.flies}
        self._point_dist = factory_class._point_dist
        self.steps = 0
        self.history = deque(maxlen=1400)
        self.recent_rewards = deque(maxlen=100)

    @property
    def products(self):
        return sum(world.products for world in self.worlds)

    @property
    def deliveries(self):
        return sum(world.deliveries for world in self.worlds)

    @property
    def total_reward(self):
        return sum(world.total_reward for world in self.worlds)

    def observation(self, fly):
        return self.owners[id(fly)].observation(fly)

    def advance(self, actions, before):
        count = self.flies_per_factory
        rewards = np.concatenate([
            world.advance(actions[i * count:(i + 1) * count], before[i * count:(i + 1) * count])
            for i, world in enumerate(self.worlds)
        ])
        self.steps += 1
        self.recent_rewards.append(float(rewards.sum()))
        if self.steps % 5 == 0:
            self.history.append({
                "step": self.steps,
                "transitions": self.steps * len(self.flies),
                "reward": self.total_reward,
                "rate": 100 * sum(self.recent_rewards) / (len(self.recent_rewards) * len(self.flies)),
                "products": self.products,
                "deliveries": self.deliveries,
                "meanReturn": self.total_reward / max(self.products, 1),
            })
        return rewards

    def checkpoint(self):
        # Primitives only: safe to load with torch weights_only=True.
        worlds = []
        for world in self.worlds:
            worlds.append({key: list(value) if isinstance(value, deque) else value
                           for key, value in vars(world).items() if key != "rng"})
            worlds[-1]["randomState"] = world.rng.getstate()
        return {"worlds": worlds, "steps": self.steps,
                "history": list(self.history), "recentRewards": list(self.recent_rewards)}

    def restore(self, saved):
        if len(saved["worlds"]) != len(self.worlds):
            raise ValueError("Checkpoint factory count does not match its settings.")
        for world, record in zip(self.worlds, saved["worlds"]):
            for key, value in record.items():
                if key == "randomState":
                    world.rng.setstate(value)
                elif key in ("part_output", "product_output", "events", "history"):
                    setattr(world, key, deque(value, maxlen=getattr(world, key).maxlen))
                else:
                    setattr(world, key, value)
        self.flies = [fly for world in self.worlds for fly in world.flies]
        self.owners = {id(fly): world for world in self.worlds for fly in world.flies}
        self.steps = saved["steps"]
        self.history.extend(saved["history"])
        self.recent_rewards.extend(saved["recentRewards"])
