"""Independent 2D assembly environment, without service or GPU startup side effects."""
from __future__ import annotations
import math
import random
import time
from collections import deque
from typing import Any
import numpy as np

class Factory:
    """A continuous 2D floor; couriers steer locally, never teleport between stations."""

    STATIONS = [
        {"name": "Raw supply", "x": 3.0, "y": 5.0},
        {"name": "Assembler 01", "x": 11.0, "y": 5.0},
        {"name": "Assembler 02", "x": 19.0, "y": 5.0},
        {"name": "Output depot", "x": 27.0, "y": 5.0},
    ]
    BLOCKERS = [(7.5, 2.1, 0.55), (7.5, 8.1, 0.55), (15.2, 2.8, 0.48), (22.7, 7.6, 0.55)]
    DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def __init__(self, fly_count: int = 8, seed: int | None = None):
        self.rng = random.Random(seed or 9)
        self.fly_count = fly_count
        self.stock = 28
        self.refill_timer = 0
        self.raw_input: list[list[int]] = []
        self.part_output: deque[dict[str, Any]] = deque()
        self.part_input: list[list[int]] = []
        self.product_output: deque[dict[str, Any]] = deque()
        self.stage1_timer = 0
        self.stage2_timer = 0
        self.flies = []
        self.spawn_flies(fly_count)
        self.deliveries = self.products = self.steps = self.episode = 0
        self.total_reward = 0.0
        self.events: deque[dict[str, Any]] = deque(maxlen=36)
        self.history: deque[dict[str, Any]] = deque(maxlen=1400)
        self.started = time.perf_counter()
        self.last_sample_step = -1
        self.avg_steps_per_second = 0.0

    def spawn_flies(self, count: int):
        self.fly_count = count
        self.flies = []
        for i in range(count):
            self.flies.append({
                "id": i,
                "x": 3.0 + self.rng.uniform(-0.65, 0.65),
                "y": 5.0 + self.rng.uniform(-1.25, 1.25),
                "cargo": 0,
                "action": "Looking for work",
                "reward": 0.0,
                "deliveries": 0,
                "activity": 0.0,
                "goal": None,
                "goal_kind": "",
                "stall": 0,
            })

    @staticmethod
    def _point_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def candidates(self):
        values = []
        if len(self.raw_input) + sum(f["cargo"] == 1 for f in self.flies) < 2 and self.stock > 0:
            values.append((1, 0, tuple((self.STATIONS[0][k] for k in ("x", "y"))), "Raw supply"))
        if self.part_output and len(self.part_input) + sum(f["cargo"] == 2 for f in self.flies) < 2:
            values.append((2, 1, (11.0, 5.0), "Assembler 01"))
        if self.product_output:
            values.append((3, 2, (19.0, 5.0), "Assembler 02"))
        return values

    def objective(self, fly: dict[str, Any]) -> tuple[tuple[float, float], str]:
        if fly["cargo"] == 1:
            return (11.0, 5.0), "Assembler 01"
        if fly["cargo"] == 2:
            return (19.0, 5.0), "Assembler 02"
        if fly["cargo"] == 3:
            return (27.0, 5.0), "Output depot"
        options = self.candidates()
        if not options:
            return (3.0, 5.0), "Raw supply"
        # Demand is a weak shared scent; distance and local crowding shape each courier's choice.
        def cost(option):
            _, priority, xy, _ = option
            distance = self._point_dist((fly["x"], fly["y"]), xy)
            crowd = sum(
                other["id"] != fly["id"] and other["cargo"] == 0 and
                other.get("goal") == option[3]
                for other in self.flies
            )
            return distance - priority * 1.7 + crowd * 1.15
        chosen = min(options, key=cost)
        return chosen[2], chosen[3]

    def _obstacle_sensors(self, x: float, y: float):
        signals = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            distance = 1.0
            for bx, by, radius in self.BLOCKERS:
                clearance = self._point_dist((x + dx * 1.1, y + dy * 1.1), (bx, by)) - radius
                distance = min(distance, max(0.0, clearance / 1.1))
            signals.append(1.0 - distance)
        return signals

    def observation(self, fly: dict[str, Any], position=None) -> np.ndarray:
        x, y = position or (fly["x"], fly["y"])
        target, name = self.objective(fly)
        fly["goal"], fly["goal_kind"], fly["goal_xy"] = name, name, target
        dx, dy = target[0] - x, target[1] - y
        distance = max(math.hypot(dx, dy), 1e-6)
        inventory_nearby = self._obstacle_sensors(x, y)
        nearby = [other for other in self.flies if other["id"] != fly["id"]]
        closest = min(nearby, key=lambda other: self._point_dist((x, y), (other["x"], other["y"])), default=None)
        if closest is None:
            crowd_dx, crowd_dy, crowd_distance = 0.0, 0.0, 1.0
        else:
            crowd_dx = (closest["x"] - x) / 31.0
            crowd_dy = (closest["y"] - y) / 11.0
            crowd_distance = min(1.0, self._point_dist((x, y), (closest["x"], closest["y"])) / 3.0)
        return np.asarray([
            dx / 31.0, dy / 11.0, min(1.0, distance / 33.0),
            float(fly["cargo"] == 0), float(fly["cargo"] == 1),
            float(fly["cargo"] == 2), float(fly["cargo"] == 3),
            self.stock / 28.0, len(self.raw_input) / 2.0,
            max(0.0, 1.0 - self.stage1_timer / 20.0), min(1.0, len(self.part_output) / 3.0),
            len(self.part_input) / 2.0,
            max(0.0, 1.0 - self.stage2_timer / 24.0), min(1.0, len(self.product_output) / 3.0),
            *inventory_nearby, crowd_dx, crowd_dy, crowd_distance, min(1.0, distance / 12.0)
        ], dtype=np.float32)

    def _can_move(self, x: float, y: float, fly_id: int) -> bool:
        if not (0.35 <= x <= 30.65 and 0.35 <= y <= 10.65):
            return False
        if any(self._point_dist((x, y), (bx, by)) < radius + 0.32 for bx, by, radius in self.BLOCKERS):
            return False
        if any(self._point_dist((x, y), (s["x"], s["y"])) < 0.92 for s in self.STATIONS[1:3]):
            return False
        return not any(f["id"] != fly_id and self._point_dist((x, y), (f["x"], f["y"])) < 0.35 for f in self.flies)

    def _reward_owners(self, rewards: np.ndarray, owners, amount: float):
        ids = sorted(set(int(i) for i in owners if 0 <= int(i) < self.fly_count))
        if ids:
            for index in ids:
                rewards[index] += amount / len(ids)

    def _tick_machines(self, rewards: np.ndarray):
        if self.stage1_timer > 0:
            self.stage1_timer -= 1
        if self.stage1_timer == 0 and len(self.raw_input) >= 2 and len(self.part_output) < 8:
            owners = self.raw_input.pop(0) + self.raw_input.pop(0)
            self.part_output.append({"owners": sorted(set(owners))})
            self.stage1_timer = 18
            self._reward_owners(rewards, owners, 0.75)
            self.event("Assembler 01 produced an intermediate part.", "part")
        if self.stage2_timer > 0:
            self.stage2_timer -= 1
        if self.stage2_timer == 0 and len(self.part_input) >= 2 and len(self.product_output) < 8:
            owners = self.part_input.pop(0) + self.part_input.pop(0)
            self.product_output.append({"owners": sorted(set(owners))})
            self.stage2_timer = 24
            self.products += 1
            self.episode += 1
            self._reward_owners(rewards, owners, 5.0)
            self.event("Final product completed · contributor reward credited upstream.", "product")

    def event(self, text: str, kind: str):
        self.events.append({"step": self.steps, "text": text, "kind": kind})

    def interact(self, fly: dict[str, Any], rewards: np.ndarray):
        x, y, cargo = fly["x"], fly["y"], fly["cargo"]
        if cargo == 0 and self._point_dist((x, y), (3.0, 5.0)) <= 1.0 and self.stock > 0:
            self.stock -= 1
            fly["cargo"] = 1
            fly["action"] = "Picked up raw material"
            rewards[fly["id"]] += 0.08
            self.event(f"Fly {fly['id'] + 1} collected raw material.", "pickup")
            return
        if cargo == 0 and self._point_dist((x, y), (11.0, 5.0)) <= 1.0 and self.part_output:
            item = self.part_output.popleft()
            fly["cargo"] = 2
            fly["payload"] = item["owners"]
            fly["action"] = "Picked up an intermediate part"
            rewards[fly["id"]] += 0.1
            self.event(f"Fly {fly['id'] + 1} collected an intermediate part.", "pickup")
            return
        if cargo == 0 and self._point_dist((x, y), (19.0, 5.0)) <= 1.0 and self.product_output:
            item = self.product_output.popleft()
            fly["cargo"] = 3
            fly["payload"] = item["owners"]
            fly["action"] = "Picked up a finished product"
            rewards[fly["id"]] += 0.1
            self.event(f"Fly {fly['id'] + 1} picked up the finished product.", "pickup")
            return
        station = {1: ((11.0, 5.0), self.raw_input, "Assembler 01"),
                   2: ((19.0, 5.0), self.part_input, "Assembler 02")}.get(cargo)
        if station and self._point_dist((x, y), station[0]) <= 1.0:
            buffer = station[1]
            if len(buffer) < 2:
                item_owners = list(fly.get("payload", [])) + [fly["id"]]
                buffer.append(item_owners)
                item_name = "raw material" if cargo == 1 else "intermediate part"
                fly["cargo"] = 0
                fly.pop("payload", None)
                fly["deliveries"] += 1
                self.deliveries += 1
                fly["action"] = f"Delivered {item_name} to {station[2]}"
                rewards[fly["id"]] += 0.6
                self.event(f"Fly {fly['id'] + 1} delivered {item_name} to {station[2]}.", "delivery")
                return
            rewards[fly["id"]] -= 0.03
            fly["action"] = "Assembler input full · waiting"
            return
        if cargo == 3 and self._point_dist((x, y), (27.0, 5.0)) <= 1.0:
            owners = list(fly.get("payload", [])) + [fly["id"]]
            self._reward_owners(rewards, owners, 5.0)
            fly["cargo"] = 0
            fly.pop("payload", None)
            fly["deliveries"] += 1
            self.deliveries += 1
            fly["action"] = "Delivered final product to depot"
            self.event(f"Fly {fly['id'] + 1} delivered a final product to output.", "delivery")
            return
        rewards[fly["id"]] -= 0.02
        fly["action"] = "Nothing to pick up here"

    def advance(self, actions: np.ndarray, previous_distance: np.ndarray):
        rewards = np.zeros(self.fly_count, dtype=np.float32)
        if self.stock == 0:
            self.refill_timer += 1
            if self.refill_timer >= 36:
                self.stock = 12
                self.refill_timer = 0
                self.event("Supply depot restocked raw material.", "restock")
        for fly, action in zip(self.flies, actions.tolist()):
            if action < 4:
                dx, dy = self.DIRECTIONS[action]
                nx, ny = fly["x"] + dx * 0.24, fly["y"] + dy * 0.24
                if self._can_move(nx, ny, fly["id"]):
                    fly["x"], fly["y"] = nx, ny
                    fly["action"] = ("Moving north", "Moving south", "Moving west", "Moving east")[action]
                    fly["stall"] = max(0, fly["stall"] - 1)
                else:
                    fly["action"] = "Avoiding an obstacle or another fly"
                    fly["stall"] += 1
                    rewards[fly["id"]] -= 0.025
            elif action == 4:
                self.interact(fly, rewards)
            else:
                fly["action"] = "Waiting for a useful load" if not fly["cargo"] else "Holding cargo"
                rewards[fly["id"]] -= 0.004
        for i, fly in enumerate(self.flies):
            target = fly.get("goal_xy", (fly["x"], fly["y"]))
            distance = self._point_dist((fly["x"], fly["y"]), target)
            rewards[i] += float(np.clip(previous_distance[i] - distance, -0.25, 0.25)) * 0.08
        self._tick_machines(rewards)
        self.steps += 1
        self.total_reward += float(rewards.sum())
        for i, value in enumerate(rewards):
            self.flies[i]["reward"] += float(value)
        now = time.perf_counter() - self.started
        instantaneous = self.steps / max(now, 1e-6)
        self.avg_steps_per_second = instantaneous if not self.avg_steps_per_second else self.avg_steps_per_second * .9 + instantaneous * .1
        if self.steps % 5 == 0:
            self.history.append({
                "step": self.steps, "reward": round(self.total_reward, 3),
                "rate": round(self.total_reward * 100 / max(self.steps, 1), 3),
                "products": self.products, "deliveries": self.deliveries,
                "meanReturn": round(self.total_reward / max(self.episode, 1), 3),
            })
        return rewards

    def stations(self):
        return [
            {**self.STATIONS[0], "input": 0, "output": self.stock, "progress": 1.0, "status": f"{self.stock} raw materials"},
            {**self.STATIONS[1], "input": len(self.raw_input), "output": len(self.part_output), "progress": max(0.0, 1.0 - self.stage1_timer / 18.0), "status": "Crafting" if self.stage1_timer else f"{len(self.raw_input)}/2 raw"},
            {**self.STATIONS[2], "input": len(self.part_input), "output": len(self.product_output), "progress": max(0.0, 1.0 - self.stage2_timer / 24.0), "status": "Crafting" if self.stage2_timer else f"{len(self.part_input)}/2 parts"},
            {**self.STATIONS[3], "input": 0, "output": self.products, "progress": 1.0, "status": f"{self.products} completed"},
        ]
