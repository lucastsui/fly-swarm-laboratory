"""Headless throughput sweep: four cooperating flies per independent factory."""
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from .brain import Brain
from .factory import Factory
from .training import TrainingRuntime

parser = argparse.ArgumentParser()
parser.add_argument('--graph', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--batches', type=int, nargs='+', default=[32,64,128,256,512])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
brain = Brain(args.graph, 22)
results = []
for count in args.batches:
    engine = TrainingRuntime(args.graph, args.output.parent, Factory)
    engine.brain = brain
    engine._load_weights(torch.load(args.checkpoint, map_location='cuda', weights_only=True))
    engine.optimizer = torch.optim.AdamW([
        {'params':brain.encoder.parameters(),'lr':1e-4},
        {'params':[brain.log_gain,brain.bias],'lr':1e-5},
        {'params':list(brain.readout.parameters())+list(brain.actor.parameters())+list(brain.critic.parameters()),'lr':1e-4},
    ], weight_decay=1e-5)
    engine.environment_count, engine.fly_count = count//4, 4
    engine._new_worlds()
    engine.publish_snapshot = lambda: None
    engine.last_checkpoint = time.perf_counter()+1e9
    engine.set_running(True)
    for _ in range(24): engine.tick()
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(24): engine.tick()
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-started)
    result = {'agents':count,'factories':count//4,'fliesPerFactory':4,
              'actionsPerSecond':count*24/statistics.median(timings),
              'sampledTransitionsPerSecond':max(32,count)/statistics.median(timings),
              'updatesPerSecond':1/statistics.median(timings),
              'cycleSeconds':timings,'memoryGiB':torch.cuda.max_memory_allocated()/2**30,
              'finite':all(bool(torch.isfinite(p.grad).all()) for p in brain.parameters() if p.grad is not None)}
    results.append(result)
    print(json.dumps(result),flush=True)
    args.output.write_text(json.dumps(results,indent=2))
    del engine
    brain.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
