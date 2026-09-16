"""Measure real cluster progress, then verify pause/resume and activity routing."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=40)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = "http://127.0.0.1:8768/api/engine/"
    def api(path, value=None, origin=None):
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        request = urllib.request.Request(base + path, data=None if value is None else json.dumps(value).encode(), headers=headers)
        with opener.open(request, timeout=20) as response:
            return json.load(response)
    first, began = api("state"), time.perf_counter()
    time.sleep(args.seconds)
    last, duration = api("state"), time.perf_counter() - began
    assert first["running"] and last["running"]
    assert len(last["cluster"]["workers"]) == 2
    workers = []
    for before, after in zip(first["cluster"]["workers"], last["cluster"]["workers"]):
        assert before["id"] == after["id"]
        accepted = after["acceptedUpdates"] - before["acceptedUpdates"]
        assert accepted > 0 and after["status"] == "Training" and not after["error"]
        workers.append({**after, "acceptedDuringTest": accepted,
                        "sampledTransitionsPerSecond": (after["sampledTransitions"] - before["sampledTransitions"]) / duration})
    assert last["updates"] - first["updates"] == sum(w["acceptedDuringTest"] for w in workers)
    assert last["cluster"]["parameterHash"] != first["cluster"]["parameterHash"]
    metrics = {"seconds": duration, "actionsPerSecond": (last["transitions"] - first["transitions"]) / duration,
               "updatesPerSecond": (last["updates"] - first["updates"]) / duration,
               "sampledTransitionsPerSecond": sum(w["sampledTransitionsPerSecond"] for w in workers),
               "workers": workers, "initialVersion": first["updates"], "finalVersion": last["updates"],
               "initialHash": first["cluster"]["parameterHash"], "finalHash": last["cluster"]["parameterHash"],
               "products": last["products"], "deliveries": last["deliveries"]}
    try:
        api("health", origin="https://untrusted.example")
        raise AssertionError("Untrusted origin was allowed")
    except urllib.error.HTTPError as exc:
        assert exc.code == 403
    api("health", origin="http://localhost:5173")
    activity = []
    for worker, environment, fly in (("spark1", 0, 0), ("spark2", 15, 3)):
        path = f"state?worker={worker}&environment={environment}&fly={fly}"
        selected = api(path)
        for _ in range(12):
            if not selected["viewPending"]:
                break
            time.sleep(.25)
            selected = api(path)
        assert not selected["viewPending"] and len(selected["brainView"]["activity"]) == 574
        assert selected["selectedFactory"]["index"] == environment and selected["selectedFly"] == fly
        assert selected["brain"]["neurons"] == 166700 and selected["brain"]["edges"] == 25582938
        activity.append(selected["brainView"]["activity"])
    assert activity[0] != activity[1]
    api("control", {"action": "pause"})
    try:
        time.sleep(2)
        paused = api("state")
        time.sleep(2)
        paused_again = api("state")
        assert not paused_again["running"] and paused["updates"] == paused_again["updates"]
        assert paused["transitions"] == paused_again["transitions"]
        assert all(w["actionsPerSecond"] == 0 for w in paused_again["cluster"]["workers"])
    finally:
        api("control", {"action": "start"})
    time.sleep(3)
    resumed = api("state")
    assert resumed["updates"] > paused_again["updates"]
    metrics["checks"] = {"bothContribute": True, "singleOptimizerVersion": True, "weightsChanged": True,
                         "fullGraphOnBothGPUs": True, "perFlyActivityRouting": True, "originGuard": True,
                         "pauseFreezesUpdatesAndFactories": True, "resumedAndLeftTraining": True}
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
