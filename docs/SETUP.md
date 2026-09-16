# Setup and reproduction

Run commands from the repository root. Keep this checkout separate from an existing live deployment: viewers use ports 8769 and 8770. Starting a new checkout does not migrate a live factory or preserve the old process's recurrent state.

## Frontend

Node.js >=22.13.0 is required (Node 22 LTS is the CI target).

```sh
npm ci
npm test
npm run typecheck
npm run dev
```

Open http://localhost:5173. This starts the dashboard only. `npm run build` checks the production bundle; `npm start` serves the generated local Wrangler bundle. Optional Sites build integration is retained with empty resource bindings and no personal project ID. GitHub is not a GPU hosting service.

## Python/CUDA

Use Python 3.11 and an isolated environment:

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a compatible PyTorch build separately using the [official instructions](https://pytorch.org/get-started/locally/). The original Windows runtime used **torch 2.11.0+cu128**, CUDA 12.8 wheels and an RTX 5090 Laptop GPU. DGX Spark is ARM64/GB10 and needs a matching NVIDIA/PyTorch runtime, not copied Windows wheels. Numerical repeatability can depend on GPU architecture and sparse kernels.

For CPU fixture tests, install `torch==2.11.0` from the official CPU wheel index. For SSH tools also install `requirements-cluster.txt`.

```sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python scripts/test-smoke.py
```

Several full-model classes explicitly allocate CUDA tensors. CPU fixture tests do not establish full CPU inference support. Memory needs depend on batch size, recurrent horizon and optimizer state; a late bounded Spark training run reported roughly 13 GiB peak allocations. Benchmark your own hardware with small batches first.

## Prepared graph and models

The companion **`fly-swarm-runtime-v1.zip`** contains the prepared graph/interface, original-layout `candidate-200.npz`, experimental `fixed-transition-v2-c60.npz`, and a portable phase manifest. It is a separate deliverable/release asset, not committed to Git. Extract it into this repository, preserving `.runtime/`, then run:

```sh
python scripts/verify-artifacts.py
```

The script checks all listed files against `artifacts/manifest.json`. No optimizer state, credential, token or world snapshot is included. Models have different sensory interfaces; renaming a checkpoint does not make it compatible. C60 remains unvalidated. If the archive is unavailable, request it from the owner. Public data can recreate the initial graph/interface, **not learned weights**.

### Rebuild the initial graph

The fetcher downloads approximately 1.1 GB of public MaleCNS v1.0 tables to `../connectome-data`, verifies SHA-256 and refuses to overwrite mismatches. Preparation needs several GB of host RAM/disk space beyond runtimes.

```sh
python scripts/fetch-data.py
python -m engine.prepare ../connectome-data
python -m engine.plastic_brain --prepare .runtime/dopamine-haul
python -m engine.build_layout_senses --root .runtime/dopamine-haul --annotations ../connectome-data/annotations.feather
```

`plane.EmbodiedBrain` creates the fixed original projection; the last command adds the later annotated projection. Recompressed files can differ across library versions; compare array fingerprints before substituting prepared data. Browser anatomy is already committed. `scripts/prepare-morphology.py` downloads SWCs and rebuilds it; `scripts/verify-morphology.py` verifies against source skeletons. These are networked preparation jobs, not ordinary tests.

## Frozen demo

After extracting/verifying the archive, use a separate activated Python terminal:

```sh
# Original-layout reference, four colliding flies, port 8769:
python -m engine.run_demo --model reference --run
# OR last displayed experimental model, port 8770:
python -m engine.run_demo --model experimental --run
```

Run one viewer at a time unless deliberately using two GPU workloads. The page initially selects the experimental view; select **reference** for the reference service. No teacher, optimizer or online updates run. Without `--run`, the new viewer starts paused. The launcher refuses to start on an occupied port.

The experimental viewer starts a new seeded `wide` layout, not the paused live session's inventory, coordinates, recurrent state or elapsed time. Its exact last training layout is retained as `docs/results/current-layout.json` for evaluation; restoring a layout is not restoring a factory snapshot.

Windows: `./start-engine.ps1 -Model reference -Run` uses `.venv/Scripts/python.exe`. It neither attaches to nor restarts another service. Closing that terminal stops that viewer. Dashboard controls pause/resume inference, not training.

## Re-evaluate the reference

These are substantial GPU evaluations. They write fresh reports, not model weights. Do not run during throughput measurements or overwrite retained evidence.

```sh
python -m engine.service_training --root .runtime/dopamine-haul --candidate .runtime/service-training/sequence-lr0005/candidate-200.npz --out .runtime/recheck-finite --evaluate-only --cases 64 --batch 16 --eval-seed 5300000 --horizon 4800
python -m engine.evaluate_swarm --root .runtime/dopamine-haul --candidate .runtime/service-training/sequence-lr0005/candidate-200.npz --out .runtime/recheck-swarm.json --cases 16 --seconds 600 --seed 5700000 --swarm-only
```

These repeat historical seeds. Use independent predeclared seeds for a new holdout. Exact reproduction across GPU architectures is not guaranteed. Inspect each CLI's protocol/frozen flags before comparing another model family.

## Training and multiple machines

See [architecture](ARCHITECTURE.md). Neither npm, CI nor the demo launcher starts training. `service_training` is the original-interface recurrent trainer. `layout_recovery_train`/`layout_recovery_worker` implement the later shared learner and experience workers. `layout_demonstration_train` and other `layout_*` modules are research variants: read their `--help`, and do not assume candidates are interchangeable.

Set the same private `FLY_COORDINATOR_HOST` in the recovery learner/workers; its safe default is loopback. Generate fresh credentials with `python -m engine.layout_recovery_security .runtime/cluster-security` (requires OpenSSL). Transfer credentials securely, pin the certificate, keep its private key on the coordinator, and firewall port 8843. Tune batch/state sizes per device. More machines do not pool GPU memory or guarantee linear speedup.

Historical helper modules are not an automatic cluster provisioning system. `benchmark_distributed_view` requires explicit private deployment JSON; personal SSH hosts/directories were removed. See [security](../SECURITY.md).
