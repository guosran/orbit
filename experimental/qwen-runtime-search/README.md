# Whole Qwen graph search

The r17 path optimizes the typed complete Qwen source graph through the existing
ORBIT C++ beam driver. See [the r17 report](../../docs/taskflow-qwen-whole-graph-r17.md)
for the domain, execution boundaries, calibration, and validation limits.

The source producer is `decode-operator-source.py`; it appends actual original
AIR decoder operators to the exact prefill graph from `qwen-node-runtime.py`.
`make-operator-catalogue.py` consumes complete training traces and registered
compiled wave-range covers. `program-operator-search.cpp` checks the source and
local cover product, invokes `searchProgramSpaceBeam`, and materializes and
checks the selected common ProgramGraph. `whole-replay.py` binds that selected
graph to the frozen complete runtime; `selected-decode-helper.py` dispatches the
actual chosen partition at each decode position over the same original BOs.

The full graph uses no CP-SAT/ILP solver and has no fixed wall-clock, evaluation,
or round cutoff. Beam convergence does not certify a global optimum. Training
and heldout requests remain separate; missing prices are unknown and actual
launch/transport costs are counted once at their measured boundary.

The earlier r16 control experiment below has only two global placement bits.
Its solver certificate and speedups apply to that small domain.

---

# Earlier r16 source-bound control experiment

This experiment registers the existing complete Triton-XDNA/AIR Qwen runtime
through `scripts/qwen-runtime-adapter.py`, then calls ORBIT's actual public C++
`searchProgramSpaceBeam` and CP `searchProgramSpace` APIs. It is an external
runtime adapter. It does not lower Qwen through native ProgramGraph or search
inside the opaque fused decoder.

The finite domain has two global placement decisions: RMSNorm and SwiGLU may
run on CPU or NPU. Matrix operations stay on NPU; weight residency, exact K,
bucket 1, capacity 2048, and 128 generated outputs stay fixed. The graph and
local model bind the exact prompt and source identity. Validate actual source,
model, and runtime artifact hashes with the source adapter before search/replay.
The C++ evaluator's matching source IDs are not a substitute for that check.

Build against an existing configured ORBIT build with the beam-search test:

```sh
python experimental/qwen-runtime-search/build-prototype.py \
  --build-dir /tmp/orbit-full-v32-build --output /tmp/qwen-local-search
```

Collect two fresh full-request training cohorts with the frozen runtime. Both
use static weights, exact K, bucket 1, and capacity 2048; `--ops matmul` places
the two small families on CPU, while `--ops all` places them on NPU. Keep
validation traces separate. For each prompt's registered graph:

```sh
python experimental/qwen-runtime-search/make-local-cost-model.py \
  --graph source.json --cpu training-cpu.jsonl --npu training-npu.jsonl \
  --output local-model.json
/tmp/qwen-local-search source.json local-model.json beam > beam.json
/tmp/qwen-local-search source.json local-model.json cp > cp.json
```

The producer matches the source-ordered timer calls individually. Each source
node gets its own CPU/NPU median, or a fixed pooled median. Each inclusive
fused decode dispatch is a fixed source node. The exclusive host residual is
`request - sum(prefill calls) - sum(decode dispatch steps)` and owns embedding,
KV handoff, inline work, control, and copies outside those dispatch steps.
Untimed KV nodes are accounted in that residual; their explicit zero is not a
measurement that KV work is free. Launch and transfer costs already belong to
synchronous call wall times, so graph edges receive no additional charge.

The CP result only proves the best model price in this four-assignment domain.
Check the actual result's proof flag; beam never proves a domain optimum.
Compilation, model collection, and source verification are outside reported
search time. Full-request validation must report numerical gate limits and
runtime variation separately. Historical baseline runs with bucket 128 and
nonresident weights are a speed comparison, not validation of this model.

For the three-prompt experiment, place the frozen manifests at
`source-p{5,128,512}.json` in a new evidence directory, then use
`run-search.py --root DIR --binary BINARY --cpu TRAIN_CPU --npu TRAIN_NPU`.
Run in the existing Qwen Python environment. The wrapper rehashes actual sources
through the frozen adapter, records those checks, produces models, runs both
real C++ algorithms, and writes `seal.json` before independent heldout collection.
Device strings in the results determine replay; internal choice indices may use
a different order from the source manifest's declared choices.

The current build and sealed search evidence is at
`/home/shiran/project/orbit-artifact/results/taskflow/qwen-source-search-r16`.
`seal.json` binds the three source manifests, local models, actual source-check
receipts, binary/build commands, training inputs, and both algorithms' results.
The first attempted fit rejected two different spellings of the same model
path (snapshot symlink versus resolved blob); the preserved attempt log explains
the failure. The final producer compares canonical resolved paths, and actual
model bytes are verified separately by the source adapter. Independent four-plan
heldout board validation and its numerical audit are separate evidence.

Run the synthetic CPU-only contracts after device timing ends:

```sh
python experimental/qwen-runtime-search/test-contracts.py \
  --binary /tmp/qwen-local-search --graph source.json
```

These fixtures test identity/trace/cost rejection and the actual C++ APIs; their
invented timing values are never Qwen performance results.
