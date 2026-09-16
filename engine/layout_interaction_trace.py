"""Read-only event instrumentation, never changes a motor or simulation state."""
import math


def before_interaction(world):
    return [(a.interactions, a.pickups, a.transfers, a.returns, a.cargo) for a in world.agents]


def interaction_events(world, before, tick):
    result = []
    for index, (agent, old) in enumerate(zip(world.agents, before)):
        attempts, pickups, transfers, returns, cargo = old
        if agent.interactions == attempts:
            continue
        event = ('return' if agent.returns > returns else 'pickup' if agent.pickups > pickups
                 else 'delivery' if agent.transfers > transfers and cargo == 3
                 else 'transfer' if agent.transfers > transfers else 'rejected')
        distances = [math.hypot(agent.x-s['x'], agent.y-s['y']) for s in world.stations]
        nearest = min(range(4), key=distances.__getitem__)
        result.append({'time': tick*.05, 'fly': index, 'event': event,
                       'cargoBefore': cargo, 'cargoAfter': agent.cargo,
                       'nearestBox': nearest, 'boxDistance': distances[nearest],
                       'position': [agent.x, agent.y], 'speed': agent.speed, 'turn': agent.turn})
    return result
