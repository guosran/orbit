# Source-ready Qwen joint mapping, r19

This experiment extends the r18 executable catalogue with a dependency-driven
CPU/NPU executor, compiled CPU regions, heterogeneous projection fission and
actual decode arithmetic overlap. ORBIT searches the complete source-bound
Qwen graph. Compilation and numerical admission happen before a candidate is
priced; neither pending entries nor missing prices are executable choices.

The workload is the original Triton-XDNA Qwen2.5-3B-Instruct Q4_0 program:
36 layers, five prompt tokens and 128 generated tokens. Its typed source has
138759 operators and 157615 values. Each complete request executes all 581
prefill operators, attention and KV handoff, followed by 127 complete recurrent
decode invocations. The source identity is
`a80070e7f879e074ce803d4e67dd4fa1f2f131505cad1b3abeb6687dd0a831fe`.

## Final board comparison

The final ORBIT plan completes the full request in a median **6517.662 ms**:
**1.105x** versus the original supported-NPU mapping and **1.032x** versus the
prior manual mapping replayed in the same r19 runtime. The latter is a modest
3.07% latency reduction. Its TTFT is **0.993x** versus the prior manual mapping,
so there is no prior-mapping TTFT improvement.

Each row is one fresh process, one warmup and three measured complete requests,
with five input tokens, all 36 layers and 128 generated outputs. Values are
median [minimum, maximum] in milliseconds; ratios divide each baseline median
by the selected median. These are three-sample observations, without confidence
intervals or a statistical-significance claim.

| Plan | Full request ms | Full ratio | TTFT ms | TTFT ratio |
| --- | ---: | ---: | ---: | ---: |
| ORBIT final selected | 6517.662 [6515.682, 6643.322] | 1.000x | 445.559 [424.852, 454.025] | 1.000x |
| Original supported-NPU mapping | 7200.046 [7133.253, 7219.606] | 1.105x | 702.296 [669.080, 722.680] | 1.576x |
| Prior manual mapping, same r19 runtime | 6724.119 [6719.401, 6847.747] | 1.032x | 442.460 [436.773, 529.800] | 0.993x |

The original supported-NPU mapping retains mandatory CPU prefill work:
embedding, RoPE, attention and the prefill LM-head, plus host work. Attention
and the LM-head inside its fused decode execute on the NPU. All three runs use the same frozen
ready runtime, physical-dependency bridge, two-stage preparation, verifier cache
policy, original guards and numerical gate. Manual baselines use independently
checked prescribed graphs, with no beam or priority optimization. Their
measurements are fresh; earlier r18 timings are not used as these baselines.

All three original gates pass and produce the same output count and shapes;
they do **not** produce identical 128-token streams. The supported-NPU baseline
diverges from both the HF reference and prior-r18 capture at token index 2.
Only its first three logit rows have the same preceding context: maximum error
is 3.138671875 against HF and 0.3828125 against prior r18. Its accepted quality
gate is not a claim of full128 numerical equivalence.

The selected run agrees with all 128 prior-r18 tokens, with maximum same-context
logit error 0.125 and non-bitexact logits. The fresh prior-manual baseline is
bitexact to that prior-r18 capture, with zero logit error. Selected versus HF
passes the unchanged mutual-top5 gate over the available 32 outputs, diverges
at index 5, and has six same-context rows with maximum logit error
4.186279296875. Errors after autoregressive divergence are not attributed to a
local kernel; full128 HF BF16 equivalence is not established.

All final processes exit successfully with no sealed-file drift, exact source
coverage, phase-union conservation and persistent NPU ownership. Each measured
request ends with zero process swap. The selected completion reports
`ru_maxrss` of 13576130560 bytes; the largest request-sampled `VmHWM` is
13579919360 bytes. These are distinct counters. The model predicts 6480.799960 ms,
0.566% below its measured median; the controller component remains an
approximation.

Machine-readable samples, medians, ranges, ratios, quality bounds, search and
fit costs, protocol settings, and exact evidence identities are in
[`taskflow-qwen-joint-mapping-r19-results.json`](taskflow-qwen-joint-mapping-r19-results.json).
This report uses sealed source/binary identities; it does not label the r18
parent revision as the r19 implementation commit.

## Execution and cost boundaries

CPU work runs on the main thread and NPU calls on one persistent owner thread.
The owner spans preparation, prefill and the final decode dispatch. A prepared
program validates SSA, source dependencies, consumers and successor lists once;
individual requests reuse that immutable description and ready queues. Actual
values remain owned until their final consumer completes. The selected replay
also applies the checked physical predecessor edges, including alias hazards.

Invocation prices include their actual input staging, allocation, launch,
completion wait, synchronization and output copy. They cannot be added blindly
when independent CPU and NPU callbacks overlap. Calibration independently
recomputes callback interval unions and conserves both prefill wall time and
the complete autoregressive request. A heterogeneous projection's opaque local
price is its first callback start through assembly completion. Its internal
idle time is not charged a second time to the external controller model.

The fitted nonnegative controller model charges an intercept plus explicit
CPU/NPU callback counts outside inclusive invocation boundaries. This is a
measured approximation, with calibration residuals and heldout prediction error
reported separately. It does not turn host scheduling into device compute or
claim that a callback-count proxy is an exact per-operator launch model.

Private XRT invocations retain actual BO and context owners through device
completion, host visibility and output cloning. Successful completion is
checked explicitly. Failure handling drains known terminal states; ambiguous
device ownership retains its objects and terminates the run instead of freeing
potentially active buffers.

## Executable transformations

| Dimension | Actual registered execution |
| --- | --- |
| Placement | Source-bound original CPU/NPU operations and CPU/NPU output-column projection splits |
| Fusion and fission | Original multi-launch FFN implementations, arbitrary supported source-convex CPU regions, and disjoint exact-cover alternatives |
| Tiling and resource size | Compiled GEMM/AIR variants with actual physical core and local-buffer evidence |
| Replicas and spatial placement | Independent output shards, explicit source-output rectangles and real column placement |
| Temporal order | Named ready-order seeds followed by legal adjacent priority swaps, plus the two actual FFN gate/up orders |
| Dataflow | Admitted complete decoder contexts and compiled projection/FFN implementations |
| Residence | Shared resident FFN weights, private staged BOs, release after each call, and legal CPU input/output copy policies |
| Overlap | Independent CPU and NPU ready callbacks; pure next-position RoPE arithmetic alongside current-token NPU decode |

The CPU region lowerer consumes source operators and value boundaries, rather
than recognizing an FFN or QKV name. It checks source convexity, effect barriers,
external dependencies, output layouts and exported aliases. Inductor candidates
require actual-input specialization receipts and hashes of generated native
code. Eager grouping does not establish native kernel fusion. Cloning an
exported view is rejected rather than advertised as an equivalent ABI.

The fusion/fission frontier producer considers every legal adjacent pairwise
CPU-cover merge and every legal unordered two-way fission of each incumbent
CPU cover. It has no operator-count, wall-time or evaluation cutoff. New
structural templates remain pending until actual compilation, complete-program
measurement and numerical admission. Disjoint trials can share a calibration
cohort, including different templates. Local frontier closure and ORBIT's
mapping convergence are distinct from global optimality.

Decode prefetch only computes the original pure RoPE slab for the next known
position. It neither predicts a token nor changes KV, RMS or BO contents early.
The measured overlap excludes event-wait synchronization. The original
autoregressive dispatch/argmax loop and complete decoder remain the execution
boundary. Unsupported transformations cannot enter through field-only edits.

## Measurement admission and memory

Every formal cohort performs one warmup and three complete 128-token requests.
The original mutual-top5 comparator, finite-logit checks, full token/logit
capture, source exact-cover checks, owner checks and original memory guards
remain in force. Bitwise agreement, first token divergence and same-context
logit error are reported separately from the original numerical acceptance
gate. Errors after autoregressive divergence are not treated as local kernel
errors.

Preparation releases dead allocator pages and reads actual live CPU weight
storage pages outside request timing. Its two-stage protocol precedes the
unchanged original-weight and decoder-loaded memory guards. The original
loader can temporarily hold raw quantized weights, BF16 conversions and
concatenated projections; available RAM alone does not fault back previously
swapped live pages. Preparation records actual storage owners, alias unions,
page counts, faults and before/after memory. It changes no weight bits, BO
contents, operating-system swap settings, numerical rule or guard.

Earlier successful cohorts retain their actual preparation protocols. Failure
receipts are preserved and excluded from prices. A failed 12-layer reload FFN
cohort is split into smaller complete-model cohorts to obtain all local prices.
Reload retains both the CPU source weights and private staged BOs; shared
retention aliases those weights and release-after-call makes BO storage
temporary. Splitting training does not admit an oversized final combination:
the selected complete program must still pass its unchanged hardware guards.

Manifests seal runtime, source, configuration, compiled artifacts, tokenizer,
model and numerical evidence. Requested snapshot paths and resolved files are
both checked so a retargeted model symlink cannot bypass identity checks.
Recovery retains successful cohorts, protocol strata and the complete rejected
attempt chain. A seal mismatch stops execution before device work.

Integrity verification itself reads approximately 8.28 GiB of large files,
including approximately 5.75 GiB of HF reference-model shards that the NPU
request does not load. The final verifier still computes every actual SHA256,
then issues `POSIX_FADV_DONTNEED` on the same read-only descriptor for files of
at least 64 MiB. This limits verifier interference with the host page cache;
it changes no file contents, swap setting, request timer or guard. Receipts
record each successful kernel advice operation. The advice does not guarantee
physical eviction, and a successful short diagnostic alone does not establish
that verifier cache pressure caused earlier failures. Hardware and numerical
failures remain isolated while independent calibration cohorts continue; any
seal drift stops the campaign immediately.

## Search and reproducibility

The initial admitted training set contains 34 cohorts and 102 measured complete
requests, in addition to warmups. It produces 4876 implementations in 297
source-cover components with 24377 component choices. Every implementation has
a new r19 price; r18 contributes executable templates and source contracts only.
The implementation kinds are 834 original prefill, 648 GEMM, 216 FFN, 1000 CPU
regions, 144 heterogeneous projections, 1526 host/control and 508 decode entries.
The four complete decoder alternatives each couple all 127 recurrent steps.

The initial controller fit has zero intercept, 0.087753 ms per CPU callback and
0.130254 ms per NPU callback. Its training RMSE is 16.8571 ms, MAE 10.9688 ms and
maximum absolute residual 85.4091 ms. These are calibration residuals for the
external controller component, not full-request prediction error or speedup.
Original-only, decode-prefault and two-stage-prefault preparations retain
separate sample annotations and residual summaries.

The initial v1 catalogue passed price fitting but failed the actual common CLI:
the new heterogeneous templates omitted their host-output visibility field.
After that explicit binding repair, v2 exposed a second missing registration:
the catalogue-level ready policy and finite priority seeds. Versioned upgrade
producers bind both repairs to the original fit receipt and actual frozen
executor. They retain the original prices, runtime registrations and support
proof byte for byte. The usable v3 catalogue explicitly registers source order,
reverse source order, CPU first, NPU first and longest first. Failed v1/v2
materialization and preflight logs remain separate evidence.

Before the initial formal search, 100 assignments generated from the actual v3
domain passed independent fast/reference feasibility and unknown-cost checks.
The largest cost difference was 1.2733e-11 ms; fast evaluation took 88.972 ms
and reference evaluation 2458.032 ms for all 100. These deterministic probes
are separate from the subsequent audit of captured actual search evaluations.

The search has no wall-time, evaluation-count or round-count cutoff. It checks
source-cover replacements and ready-priority changes until their registered
legal local neighborhoods stop improving. The visited-assignment set stores
lossless fixed-width packed keys with exact byte equality; it does not replace
identity checks with collision-prone fingerprints. Reference evaluation remains
available to compare the optimized calendar with full physical checking.

Initialization uses the same complete joint evaluator. One coordinate pass
checks every choice of a dimension against the same current assignment, then
commits that dimension's best feasible priced state. It does not add independent
local minima or accept unknown costs as zero. The resulting assignment is only
an initial beam proposal: the ordinary uncapped beam and complete final
cover/priority neighborhoods still run. A nonseparable regression has a
coordinate-pass result of 8 followed by a beam result of 0, directly checking
that initialization is not the stopping rule. All initialization evaluations
are present in the actual checkpoint and trace; reports separate their count
and duration from subsequent search.

For later frontier rounds, cover warm start maps an actual previously checked
ORBIT result by stable implementation IDs and source/artifact/ABI equality into
the new cover groups. It never imports old choice indices or costs. The old
cover is freshly evaluated under each currently registered named priority;
learned priority ranks are searched again. New groups and choices remain in
the domain and require their own actual evaluation witnesses.

An earlier all-zero-only initialization run was explicitly interrupted for this
implementation revision after 1046.617 seconds, with approximately 1.91 million
retained evaluations. It was still improving and did not reach convergence.
Its exact kernel peak RSS was 9980125184 bytes, with sampled process swap zero.
Its partial trace, interruption reason, final observer seal and original
search-executable/source snapshots are retained under `search-round0`; its numbers are
not presented as a completed search or hidden from the experiment cost.

Manual baselines use explicit complete group-to-choice assignments and source
order. They pass the same physical graph checks and selected-runtime bridge;
the materialization path skips every beam and priority search. A manually
prescribed plan is never reported as an ORBIT discovery. Baseline and selected
heldout runs use the same final preparation protocol and measurement boundary.

## First complete incumbent, before frontier expansion

The coordinate-seeded initial search completed in 205.366 seconds including
process startup and final serialization; the CLI reported 202.866 seconds.
It made 192045 cost evaluations: 24085 during initialization, 167922 during beam
and final cover checking, and 38 actual ready-priority trials. Initialization
took 11.978 seconds and improved 217 coordinates. The exact kernel peak RSS
was 11046260736 bytes (10.29 GiB), including serialization. Sampled process swap
remained zero. All inputs were unchanged across the observed search.

Every one of the 24377 cover choices and five named priorities had an actual
feasible priced witness. The final 24080 single-group neighbors and 38
incomparable adjacent-priority trials had no improving move; the remaining 835
adjacent pairs were comparable in the checked DAG and could not change ready
execution. A separate 100 captured-evaluation audit checked the actual recorded
costs against full reference evaluation, with maximum difference 5.4570e-12 ms.

The selected graph has 874 physical invocations and 142 CPU covers: 106 eager
groups and 36 actual Inductor specializations. The independent compiler audit
proves native multi-operator fusion for 33 of those specializations. Layer 35
uses the reload/up-gate FFN alternative. The decoder uses coalesce1 and pure
RoPE prefetch. New standalone GEMM and heterogeneous projection variants were
evaluated but were not selected. This is an ORBIT result, not a manual plan.

On the board, the three measured complete-request times were 6498.938,
6754.323 and 6631.348 ms: median 6631.348 ms. TTFT was 416.533, 466.644 and
551.327 ms: median 466.644 ms. The model predicted 6548.763 ms, 1.25% below
the measured median. The peak RSS was 13696274432 bytes, with zero process
swap. Original gates, exact covers, ownership and phase conservation passed.
All three 128-token streams agree with the prior r18 capture; logits are not
bitwise identical, with same-context maximum absolute error 0.125. The original
HF BF16 mutual-top5 comparison passes over its available 32-token reference;
this does not establish 128-token BF16 equivalence.

The generic incumbent frontier produced 250 legal merge/fission moves and
2854 new CPU-region choices, packed into 28 complete-program cohorts. All 28
completed their own full128 warm1/repeat3 measurement and original admission
gates, with no failures or sealed dependency drift. Strict fitting verifies a
one-to-one mapping from all 2854 candidates to nonempty actual price samples.

The resulting first expansion has 62 actual training cohorts, 7730
implementations and 3854 CPU regions, with no missing local prices. Its exact
cover domain has 261 components, 121921 choices and a largest component of
2400 choices. This is approximately five times the initial choice count,
although the implementation count grows by only 59%. Merging cover components
creates additional combinations; implementation counts alone underestimate
search growth. No registered choice is discarded to fit the earlier CLI
resource allowance. The CLI raises its caller-side structural allowance to the
actual finite domain cardinality with checked integer accumulation; the generic
API default is unchanged. This changes model admission, not search stopping.

The strict expansion producer took 177.86 seconds and peaked at 1543368 KiB
RSS with zero reported swaps. This run used cProfile: domain construction took
7.378 seconds cumulative and exact-cover grouping 3.589 seconds cumulative,
including profiling overhead. These are producer measurements, separate from
the subsequent ORBIT process and from board inference latency.

Strict warm-start preparation exposed three changed CPU implementation IDs.
The fitter had selected newer copies of the same actual specialization proof;
generated-file paths participate in artifact identity. A versioned migration
verifies both actual proof files and generated hashes, producer, source, ABI,
execution and memory contracts before retaining the original canonical artifact
and ID. Only generated-file paths may differ. All 4876 parent implementation
IDs remain available; all 7730 fresh prices, samples, runtime support and 121921
choices remain unchanged. Code-hash, ABI and heap changes have explicit
rejection tests. The first expansion catalogue and failed warm-start preflight
remain separate evidence rather than being overwritten.

The expanded-domain search completed in 937.695 seconds, including final graph
serialization. It performed 1094999 actual evaluations: 121670 initialization,
851631 beam, 121660 final cover replacements and 38 ready-priority trials.
Initialization took 81.939 seconds; two beam rounds took 716.064 seconds. Exact
kernel peak RSS was 12848545792 bytes, with sampled swap zero. All search
inputs remained unchanged. Every registered cover choice and named priority
had a priced feasible witness; captured actual costs passed 100 independent
reference checks with maximum difference 4.5475e-12 ms. The final cover and
priority neighborhoods had no improving move. This establishes local closure
inside the measured catalogue, not global optimality.

Its selected graph contains 838 physical invocations, with prefix invocations
reduced from 364 to 328. Placement remains 436 CPU and 145 NPU source operators
in the prefix. Thirty-six CPU covers now include attention alongside QKV bias, split and
Q/K RoPE; compiled covers may still invoke several native kernels and batched
matrix multiplies. Grouping or compilation does not establish a
single fused native kernel for an entire cover. The prediction is 6503.747 ms;
this is a model prediction rather than a board measurement. At that round's
conclusion the new incumbent exposed 1368 further unpriced merge/fission
choices in 40 complete-program cohorts. Frontier closure had not yet been
established; the following expansion measured those choices.

The expanded incumbent also passed a separate complete-model board run. Its
three measured request times were 6515.729, 6951.194 and 6542.112 ms, with median
6542.112 ms. TTFT was 432.522, 525.007 and 437.217 ms, with median 437.217 ms.
Peak RSS was 13724909568 bytes and request-end process swap was zero. All
original gates, source covers, phase conservation and persistent-owner checks
passed. Its complete 128-token output agrees with the prior r18 capture;
same-context maximum logit error is 0.125 and logits are not bitwise identical.

## Second measured frontier expansion

All 40 second-expansion cohorts passed their original full128 warm1/repeat3,
numerical, memory, source-cover, ownership and seal checks. No new cohort failed.
The combined admission contains 102 actual training cohorts: 306 measured
complete requests, in addition to warmups. The metadata inventory binds every
one of the 1368 pending source/policy choices to actual successful cohort,
specification, original gate and completion receipts. It does not reread the
model or substitute for the independent numerical auditor.

Strict fitting completed with 9098 executable, priced implementations, including
5222 CPU regions. All 1368 new choices map to actual prices and three, six or
twelve complete sample observations; all 2854 choices from the preceding
frontier remain priced. There are no missing local prices or reused r18 prices.
The domain contains 261 components, comprising 218 variable and 43 fixed
components, with 538009 component choices. The largest component has 12385
choices over five source operators. These are component cardinalities, not
the number of complete-program assignments.

The actual strict-fit process exited successfully after 241.98 seconds and
peaked at 2102784 KiB RSS (about 2.01 GiB). Its cProfile total was 240.454 seconds;
domain construction consumed 11.855 seconds cumulative, including 7.395 seconds
for exact-cover grouping. Source CPU-region construction/checking covered 5222
regions in 11.955 seconds cumulative. Profiling overhead and file verification
belong to these producer measurements; they are neither search time nor NPU
inference latency.

A versioned strict identity migration checks 365 CPU entries whose generated
proof paths changed, retaining the original canonical artifacts and IDs only
when actual generated code, specialization, source, ABI and memory contracts
agree. All 7730 parent implementation IDs remain available, while all 9098 fresh
prices, samples and the new support evidence remain unchanged. The raw v1
catalogue and migrated v2 catalogue are both retained. The previous 838-invocation
incumbent still maps through the new 261 components and five named priorities;
100 actual C++ fast/reference preflight assignments passed before search.

Evidence under the r19 artifact root includes:

- `r2-actual-receipt-inventory-v1.json`, SHA256
  `d1045f6535ba1f3dde7cb25d355d5cb3394003ed2dc173da81666ac4a6086557`.
- `fit-frontier-r2-receipt.json`, SHA256
  `887807367717d570c9a840dca25cfc167eceac60dfa252e6fcd47e5ae0a91766`,
  containing each new choice's actual price/sample binding and all 102 training
  receipts, plus the completed time/cProfile evidence.
- Raw `catalogue-r19-frontier-r2-v1.json`, SHA256
  `016496a0856e3c9be91800d124ada51b01b8c656fd0e5adf21c8852ec281203a`.
- Migrated `catalogue-r19-frontier-r2-v2.json`, SHA256
  `88d7fd45630e6dad00cd84461fbdc4ae2ed615af16c09bd7527245b88796bbdd`.

The second expanded-domain search completed without a wall-time, evaluation
or round cutoff after 2566.998 seconds (42 minutes 46.998 seconds), including
final graph construction and serialization. It performed 4839791 actual
evaluations. The exact kernel peak RSS was 17922715648 bytes (about 16.69 GiB,
or 17.923 GB); observed process swap was zero. The sealed inputs remained
unchanged. This larger domain substantially increases search cost; the earlier
rounds and interrupted initial run remain part of the recorded experiment.

The actual selected graph has 839 physical invocations and 329 prefix
invocations. Prefix placement remains 436 CPU and 145 NPU source operators.
Its 143 CPU covers comprise 107 eager groups and 36 actual Inductor
specializations. The layer-zero five-operator cover is split into two- and
three-operator covers; the other 35 attention-containing covers retain five
operators each. Layer 35 uses reload/up-gate FFN execution, while the decoder
retains coalesce1 and pure RoPE prefetch. There are no newly selected standalone
GEMM or heterogeneous-half variants. This selection changes graph partitioning,
policies and order rather than moving source operators between CPU and NPU.

A separate complete-model board replay exited successfully without file drift.
Its three full-request times were 6523.213076, 6558.340211 and 6467.812054 ms:
median 6523.213076 ms. TTFT was 438.028976, 424.176876 and 420.679981 ms:
median 424.176876 ms. Peak RSS was 13618450432 bytes, and request-end swap was
zero. The original numerical and memory gates, source covers, persistent owner
and phase conservation checks all passed. All 128 generated tokens agree with
the prior r18 capture; logits are not bitwise identical, with same-context
maximum absolute error 0.125. The available 32-token HF BF16 reference passes
the unchanged mutual-top5 gate. Its first token divergence remains index 5;
the six comparable-context rows have maximum logit error 4.186279. This does
not establish full128 BF16 equivalence or a final baseline speedup.

The replay is bound by
`heldout-selected-frontier-r2-r19-v1-prepared/command-receipt.json`, SHA256
`0058dae26977af76a4e9093068092a52c468ac33447d85a71ce48597e8cef753`.
Its execution result is retained as
`heldout-selected-frontier-r2-r19-v1.execution-result.json`, with the independent
original-gate report retained alongside the execution evidence.

## Third measured frontier expansion

The next incumbent frontier exposed two unpriced layer-zero `qkv_split`
leaf-policy choices. All eight complete-program cohorts passed the original
full128 warm1/repeat3 protocol and independent numerical, source, ownership,
phase, memory and seal gates. The admission retains all 102 preceding actual
training backings and now contains 110 actual cohorts. No failed attempt is
used as a price sample.

Strict fitting maps each new choice to twelve actual samples. Their local
inclusive median prices are 0.082243 ms and 0.1496215 ms, respectively; they
are not whole-request times or speedups. The resulting catalogue has 9100
priced implementations and 5224 CPU regions, with no missing local prices or
old r18 prices reused. Its 261 components contain 553867 choices, with 218
variable components and 43 fixed components. The largest component now has
28243 choices over five source operators. Two new implementations can create
many exact-cover combinations; implementation growth alone does not measure
search-domain growth.

The actual strict-fit producer exited successfully after 253.73 seconds
(4 minutes 13.73 seconds), peaking at 2126348 KiB RSS (about 2.03 GiB).
cProfile recorded 252.212 seconds total; cumulative domain construction took
12.015 seconds, exact-cover grouping 6.463 seconds and checking 5224 CPU regions
10.982 seconds. These are producer measurements, separate from search and
board timing. `r3-actual-receipt-inventory-v1.json` binds the eight successful
cohorts and both new choices; `fit-frontier-r3-receipt.json` binds all 110
training receipts, actual sample prices and the completed profile. The latter's
SHA256 is `86b03f8a3a836dd3efe44c6e7d144c0a11e3a6ef47a666f20aa446f45dc9b7c5`.

The strict path-only identity migration checks 401 entries and preserves all
9098 parent implementation IDs, while retaining all 9100 freshly fitted prices
and the actual new support evidence. The 839-invocation prior incumbent passes
strict warm-start mapping under all five named priorities. An actual 100-case
C++ fast/reference preflight passed, with maximum cost difference
1.09139e-11 ms. The migrated catalogue is
`catalogue-r19-frontier-r3-v2.json`, SHA256
`e7c5c8d2bbbc749961bf2f757856ff17e97ee836de932e4da1c1ffc69f75c7c4`,
with registration identity
`0872c94c2a89cf230a4c3a0585e4f812c8772ab6da8bee638afeff15a82dd77b`.
The raw catalogue and migration receipts remain separate evidence.

## Final search, frontier closure and limits

The third expanded-domain formal search completed with unchanged sealed
inputs and no wall-time, evaluation or round cutoff. Observed process wall time
was 2563.525 seconds; the CLI reported 2559.710 seconds. It made **4982513**
actual evaluations: 553616 coordinate-initialization evaluations, 3875253 beam
evaluations, 553606 final cover-neighborhood checks and 38 legal priority
trials. The exact kernel peak RSS was **18064039936 bytes** (about 16.82 GiB).
Sampled process swap peaked at **14352384 bytes** (13.6875 MiB), not zero; this
sampled `/proc` counter is not a kernel lifetime swap high-water mark.

Every one of the 553867 registered cover choices and five named priority seeds
has an actual feasible priced witness. All 553606 final single-group cover
neighbors and 38 legal adjacent-priority trials were nonimproving; 800 other
adjacent pairs were comparable in the checked dependency graph and were
excluded by that legality proof. The independent audit checks 100 captured
actual evaluations against the full reference calendar, with maximum error
4.5475e-12 ms. This is actual measured-domain search coverage, not a claim that
every Cartesian combination was visited or that a global optimum was proved.

The checked final graph is
`c5d291cb9a5b74162395951b50fe3fd9b315a453ed45eec52e22da2a36c2b5c7`.
It contains 839 physical invocations: 569 CPU and 270 NPU across the whole
request, including all decode steps. The prefix has 329 invocations and 143 CPU
covers, with logical placement of 436 CPU and 145 NPU source operators. There
are 37 actual Inductor specializations and 106 eager groups. Compared with r2,
three CPU covers change backend or input-copy policy; source partitioning,
placement, L35 reload/up-gate FFN and coalesce1/RoPE-prefetch decode remain.
Compilation can use several native kernels and external matrix multiplies;
it is not proof that an entire cover is one native fused kernel. The strict
whole-cover multi-arithmetic single-kernel fusion criterion is not established
for any of the selected compiled covers.

The final generic frontier enumerates 146 CPU covers, 72 adjacent pairs and
285 legal merge/fission moves. It finds **zero unpriced candidates, pending
trials or further measurement cohorts**. It also preserves explicit rejections
for 71 nonconvex/effectful merges and 351 illegal fission children. This closes
all supported adjacent pairwise CPU-cover merges and legal unordered two-way
fissions of this incumbent, combined with the separately checked registered
cover/priority cost neighborhood. It does not exhaust every partition of every
program or prove Cartesian/global optimality.

The backend scope remains explicit. Only actual compiled/admitted GEMM tiling,
resource/replica/spatial, FFN order/residence and CPU region/copy variants enter
the search. Decoder alternatives are four validated complete-36-layer contexts
(coalescing crossed with pure host RoPE prefetch), not arbitrary internal
attention transformations. Invocation-inclusive timing accounts for transport,
launch and synchronization once; it does not establish unrestricted separate
DMA overlap or arbitrary SRAM-residence transformations. The fitted controller
has 330 observations, RMSE 11.997 ms and maximum absolute calibration residual
86.781 ms. Heldout measurements do not refit prices or change the selected plan.

Final selected and both fresh baseline replays, original numerical gates,
physical/source checks and the scoped generic frontier closure have completed.
The exact selected execution receipt SHA256 is
`2c414d80db37eaa05a641d83dd966eadb02a7e77d1a1c87ce5bb0e38dfa3265a`.
The original-NPU and prior-manual receipt SHA256 values are
`01d5d655b0a748daff98efb39771ad7c60c29dfe095daf74bb586199dd832fb4`
and `21bbc1d4b2fe662cc479d92a95e20dedd23b5ecf1f1991cfb5683b2fa14fc961`.
The full machine-readable result retains the source/catalogue/registration and
proof paths/hashes, fit profiles and all historical search costs, including the
explicitly interrupted initial search. No pending final board stage or global
optimality claim is hidden behind the local closure result.
