# Larger MLP execution comparisons

The source-preserving whole-program residency choice found in the r12 finite
search also improves two larger MLPs on the NPU. These are fixed-plan transfer
experiments, not newly calibrated CP-SAT optimality proofs for the larger shapes.
The public beam frontier fix is evaluated separately.

Both programs contain two GEMMs sharing runtime weights, an explicit BF16 RNE
bridge after the first ReLU, and a final FP32 ReLU output. Inputs and independent
sequential FP32/RNE references come from `scripts/make-shared-mlp-workload.py`.
Original tolerances remain `atol=rtol=1e-4`. The NPU baseline uses phase-local
M16/N16/K16 tiling; the alternative retains the whole program on chip. Both
execute identical source and inputs. Neither uses precomputed outputs.

| Program | Source operations | Baseline configurations | Whole configurations | Baseline ms | Whole ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Batch 48, width 16, 2 layers | 100608 | 12 | 1 | 2.282932 | 0.219735 | 10.3895× |
| Batch 16, width 32, 2 layers | 132608 | 8 | 1 | 1.576507 | 0.213236 | 7.3932× |

Each entry is the median of three fresh-process medians, each with five warmups
and 31 timed calls, CPU affinity 0. The 12 processes execute 432 calls in total,
all passing numerical checks. Independent recomputation confirms bit-identical
references and saved outputs, compiled/manifest/source identities, and exclusive
phase sums equal to full wall time for every call. Cases run round-robin, with our compilation and
CPU-heavy validation completed first. Background applications and CPU governor
are unchanged. The timing boundary includes reset, all region transfers,
submission, wait, readback, control and output copy for a prepared call; initial
compilation, context creation and buffer allocation are outside it.

Individual process medians (ms):

- Batch 48 baseline: 2.282932, 2.269812, 2.286162; whole: 0.218346, 0.219735, 0.220006.
- Width 32 baseline: 1.554557, 1.576507, 1.618036; whole: 0.212735, 0.213735, 0.213236.

The larger gains accompany fewer physical configurations and corresponding
launch/transfer operations; these experiments do not isolate each contribution.
They compare NPU plans, not NPU against CPU or CP-SAT against another solver.

A four-layer, batch-16, width-16 MLP also compiles and passes three diagnostic
board calls using phase-local M16 execution. Its whole-program form exceeds the
current per-core program-memory capacity. A separate whole-tile layout-helper
outlining experiment fits the code, but has not been integrated into the checked
backend or run on hardware. No speedup is claimed for that experiment.

## Reproduction evidence

Evidence root:
`/home/shiran/project/orbit-artifact/results/taskflow/full-joint-v32-r1`.

- `larger-probe-r13`: compiled whole variants, four-layer phase variant, source,
  numerical producer receipts, and diagnostic results; saved diagnostic outputs
  match references exactly (`max_abs_error=0`).
- `larger-baselines-r13`: independently compiled phase-local baselines.
- `larger-comparison-r13`: frozen measurement protocol, producer, exact commands,
  per-process identity/sample/output/result records and summary.

All these executions use the frozen r12 compiler SHA256
`3616f1ae32734e74df37dfe9fb228b7f0b35a9f864a7455f2e321834839aef1c`
and its runtime execution contract. The newer beam algorithm does not change
these archived compiler or measurement identities. Reproduction requires the
pinned AIE/AIR environment and NPU; export alone is not hardware validation.
