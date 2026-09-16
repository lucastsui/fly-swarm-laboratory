# Architecture and experiment families

| Family | Core modules | Learned elements | Interface |
|---|---|---|---|
| Initial reservoir/actor–critic | `brain`, `training`, `parallel`, `cluster_*` | Encoder/readout and controller parameters | Learned policy head; historical prototype |
| Embodied dopamine-inspired | `plane`, `plastic_brain`, `haul_*`, `conditioning_experiment` | Selected, then all existing synaptic gains | Fixed annotated sensory/output populations |
| Supervised connectome | `supervised_steering`, `supervised_joint`, `service_training`, `layout_*` | Existing synaptic gains; later tonic/excitability | Fixed per-experiment projection and motor readout |

The strong reference belongs to the third family. It is not a dopamine-only result or merely a trained decoder. Earlier systems remain for research provenance, not as the final embodied controller.

## Graph and dynamics

`prepare` retains annotations with a superclass, excluding explicit Glia status, and released edges between surviving endpoints. Weak edges and autapses remain. Rows are postsynaptic, columns presynaptic. Input-normalized contact counts and simplified transmitter signs are modeling choices, not a complete physiological efficacy measurement.

The embodied model takes four neural substeps per 0.05-second physical action. Supervised weights are `base_weight * exp(log_gain)`: bounded gains preserve routes and signs. The custom sparse derivative uses chunked edge-local products. Some later experiments use surrogate activation gradients, explicitly distinct from exact derivatives and biological plasticity.

Each fly has a distinct state column. Shared parameters do not imply shared momentary activation, body state, cargo or observations. Independent factories differ from multiple flies sharing inventory and collisions in one factory.

## Distributed computation

The early actor–critic cluster exchanges gradients; the plasticity cluster exchanges proposals. The later recovery system distributes frozen canonical checkpoints and accepts versioned recurrent experience windows for learner updates. These protocols are different.

This is data/experience parallelism, not partitioning one neural tick across GPUs or synchronous DDP. Workers hold full model copies. Staleness, synchronization and bandwidth matter. A locally optimized GPU can outperform network-distributed inference for four dashboard flies.

## Training, evaluation and display

- Training owns its optimizer, curriculum, output directory and checkpoint sequence.
- Frozen evaluation owns independent seeds and a report, recording teacher/noise/learning flags and model identity.
- `service_viewer` displays the reference with matching evidence.
- `phase_viewer` loads immutable hash-checked manifests. Publication is not reliability promotion. A parameter switch starts a new display world; a delivery does not.
- `publish_dashboard_phase` validates and archives manifests before atomically updating a pointer. It neither trains nor proves reliability.

## Anatomy and playback

The sampled SWC anatomy is separate from the numerical graph. `neural_telemetry` records actual sampled rates and presynaptic values entering the fourth recurrent substep, allowing weighted signals to be reconstructed for sampled endpoint pairs. Highlighted branches do not model spatial conduction or locate individual synapses.

`fast_frozen_policy` caches sparse structures and optionally uses CUDA graphs. Immutable timestamped snapshots let HTTP readers avoid blocking on the GPU lock. The frontend buffers/interpolates motion and updates trails. Speed changes pacing, not physics step size; missing samples are not biological silence.

## Generalization limits

Role/cargo ambiguity, recurrent saturation, interaction timing, pickup-return loops, sparse successful examples and conflicting gradients all appeared. Static or teacher-forced fit often did not transfer to self-driven trajectories. Some curricula improved subtasks but reduced products. The `layout_*` programs preserve these diagnostics, not a claim that arbitrary layouts were solved.
