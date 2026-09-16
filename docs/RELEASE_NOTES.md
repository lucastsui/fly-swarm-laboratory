# v0.1.0 research snapshot

Initial standalone repository snapshot of Fly Swarm Laboratory, September 15, 2026.

Includes the dashboard, anatomical assets, full-graph engine, training/diagnostic modules, tests, retained results and setup documentation. Companion `fly-swarm-runtime-v1.zip` supplies the prepared graph/interface and two selected frozen models. Verify it with `artifacts/manifest.json` and `scripts/verify-artifacts.py` after extracting into the checkout.

- Original-layout reference: 62/64 finite factories cleared; four colliding flies produced 410 products over 16 ten-minute test worlds.
- Later arbitrary-layout experiments did not establish reliability. Last displayed C60 remains behaviorally unvalidated.
- Learning in the successful reference uses supervised recurrent backpropagation, not dopamine-only plasticity.
- Source snapshot and artifact export do not resume training or the paused original demo.
- Large raw datasets, historical optimizer states and all intermediate checkpoints are not part of this release.

This is a research snapshot, not a production release or a claim of biologically faithful whole-fly simulation. See README, security guidance, data attribution and retained evidence before using it.
