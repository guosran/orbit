# Qwen runtime placement search: four-assignment checkpoint

This experiment runs the original complete Triton-XDNA/AIR Qwen2.5-3B-Instruct
Q4 runtime: 36 layers, prompt lengths 5/128/512, and 128 generated tokens.
ORBIT's actual C++ public beam search and CP search selected global RMSNorm and
SwiGLU placement. This is **not whole-operator-graph optimization**. The user
subsequently required that broader task; this experiment is its control evidence.

The two placement bits form four assignments. Matmul stays on NPU; static
weights, exact K, bucket 1, four CPU threads and context capacity 2048 are fixed.
Decode is a fixed full-model implementation here. These fixed improvements
were previously chosen manually and are not discoveries of this search.

## Search and independent board results

| Prompt | Beam search (ms) | CP search (ms) | Selected full request (ms) | Same-fixed-settings all-NPU-small-ops / selected |
|---:|---:|---:|---:|---:|
| 5 | 0.120 | 39.773 | 6754.020 | 1.0172x |
| 128 | 0.116 | 2.918 | 6838.217 | 1.0247x |
| 512 | 0.262 | 6.041 | 7887.075 | 1.0819x |

Both algorithms select CPU RMSNorm and CPU SwiGLU for all lengths. CP returns
`OPTIMAL` for the four-assignment model domain; beam returns `FEASIBLE`.
Times in the table are the search APIs' wall times, excluding process startup,
source verification, calibration and compilation. External search-process wall
times are 27–68 ms. The configured solver budget is 600 seconds.

Two fresh training processes each measure three requests per prompt. Local
costs bind individual source call sites. Inclusive operation timings own their
launch, synchronization and transfer work; the exclusive request residual owns
the remaining host work. No additional edge fee duplicates those timings.
Fixed-node and residual medians are pooled across the two training cohorts,
which is an additive-model assumption, not a hardware timing theorem.

After freezing the model and selections, twelve fresh validation processes
measure all four assignments at all three prompt lengths, three requests each.
The selected assignment has the lowest full-request median in each group.
Selected prediction errors are 0.885%, 0.238% and 0.604%. These observations do
not establish statistical separation or a hardware-global optimum.

Relative to the separately measured r15 original baseline, full-request ratios
are 1.177x, 1.116x and 1.224x. Relative to r15's manually tuned version, ratios
are 1.009x, 1.006x and 1.003x: the selected configuration is identical, so there
is no demonstrated additional optimization over that manual choice. Earlier
2.23–3.95x numbers were manual tuning's **first-token** ratios, not ORBIT results.

## Correctness and evidence

The independent audit checks source/plan/CLI bindings, all 36 layers through
144 prefill GEMM calls, 127 decode calls, and all 128 generated tokens per request.
All 48 capture comparisons pass the unchanged vendor first-32-token mutual-top5
gate, which stops at the first divergence. That gate is not full-logit
equivalence to the BF16 reference. Every CPU/CPU selected capture also matches
the previous same-configuration tuned and exact-K captures bit for bit across
all 128 logits rows and tokens. Captures cover the first measured sample;
remaining samples retain their token streams and timings.

Evidence directory:
`/home/shiran/project/orbit-artifact/results/taskflow/qwen-source-search-r16`.
`protocol.json`, `seal.json`, `summary.json`, `independent-audit.json`, raw
commands, logs, plans and execution receipts retain the experiment boundaries.
The initial model-path check failed on a snapshot symlink versus its resolved
blob; that attempt is preserved. The corrected check compares resolved paths.

Reproduction code is under `experimental/qwen-runtime-search/` and
`scripts/qwen-runtime-adapter.py`. Five adapter tests and seventeen synthetic
producer/search contract checks pass. Synthetic tests are not timing evidence.

The subsequent full-graph work must expose the operators inside decode,
register executable implementation covers, model real invocation boundaries,
and replay an actual graph selection. Expanding the two global bits alone does
not complete that work.
