# Review guidance distilled from guosran's Coredac PRs

This document translates review feedback into checks for ORBIT's C++/MLIR changes. It is evidence and review guidance, not an instruction to copy Coredac's architecture or every historical naming preference. Current user requirements, ORBIT's invariants, and the local LLVM style take precedence.

## Scope and collection status

The requested repositories are the public repositories [coredac/neura](https://github.com/coredac/neura) and [coredac/amoeba](https://github.com/coredac/amoeba). The exact author is [guosran](https://github.com/guosran), GitHub numeric user ID **165251838**. Repository metadata and complete REST JSON response pages are cached under `/tmp/orbit-review-distillation-raw/` and preserved in `orbit-artifact/results/taskflow/review-cleanup-v31-r1/comments/`. JSON whitespace may be normalized; IDs, authors, dates and complete comment bodies are preserved.

**Collection is sealed for the API-visible snapshot.** Search discovery is complete: both responses have `incomplete_results=false`, and each result fits one requested 100-item page. The final receipt is `/tmp/orbit-review-distillation-raw/coverage.json`; full normalized events are in `normalized-comments.json`. All three endpoints were exhausted for every discovered PR: **105 response pages, `gaps=[]`**. The large inline-comment collections include Neura #173 (100+14), #254 (100+2), and #269 (100+12); their second pages are included in the review.

Search `author:guosran` returned 28 Neura and 6 Amoeba PRs, but three Neura results have actual `user.login=Copilot`. Strict author coverage is therefore **25 Neura + 6 Amoeba = 31 PRs**. Those three Copilot-authored PRs are supplementary and must not inflate the strict-author count. Discovery included open and closed PRs, including closed unmerged PRs; no merged-only filter was used.

The strict 31 PRs contain **1,077 events**: 74 conversation comments, 620 inline comments/replies, and 383 submitted reviews. The collector classifies these as 500 human-reviewer events, 523 author replies, and 54 bot events. Of the human-reviewer events, **378 have non-empty bodies**: 32 conversation comments, 341 inline comments, and 5 review summaries. Non-empty does not mean every event supplies a distinct substantive rule: short acknowledgments and repeated remarks remain in the raw record. The three supplementary Copilot PRs contain one author reply and no human-reviewer feedback.

| Repository | Strict guosran-authored PR inventory |
|---|---|
| neura | [173](https://github.com/coredac/neura/pull/173), [179](https://github.com/coredac/neura/pull/179), [195](https://github.com/coredac/neura/pull/195), [215](https://github.com/coredac/neura/pull/215), [220](https://github.com/coredac/neura/pull/220), [222](https://github.com/coredac/neura/pull/222), [243](https://github.com/coredac/neura/pull/243), [245](https://github.com/coredac/neura/pull/245), [254](https://github.com/coredac/neura/pull/254), [263](https://github.com/coredac/neura/pull/263), [269](https://github.com/coredac/neura/pull/269), [279](https://github.com/coredac/neura/pull/279), [281](https://github.com/coredac/neura/pull/281), [286](https://github.com/coredac/neura/pull/286), [293](https://github.com/coredac/neura/pull/293), [298](https://github.com/coredac/neura/pull/298), [307](https://github.com/coredac/neura/pull/307), [311](https://github.com/coredac/neura/pull/311), [313](https://github.com/coredac/neura/pull/313), [315](https://github.com/coredac/neura/pull/315), [340](https://github.com/coredac/neura/pull/340), [343](https://github.com/coredac/neura/pull/343), [344](https://github.com/coredac/neura/pull/344), [349](https://github.com/coredac/neura/pull/349), [356](https://github.com/coredac/neura/pull/356) |
| amoeba | [2](https://github.com/coredac/amoeba/pull/2), [3](https://github.com/coredac/amoeba/pull/3), [4](https://github.com/coredac/amoeba/pull/4), [10](https://github.com/coredac/amoeba/pull/10), [11](https://github.com/coredac/amoeba/pull/11), [12](https://github.com/coredac/amoeba/pull/12) |

The supplementary Copilot-authored PRs are [neura#230](https://github.com/coredac/neura/pull/230), [#231](https://github.com/coredac/neura/pull/231), and [#255](https://github.com/coredac/neura/pull/255). Their PR descriptions are not independent reviewer evidence.

For each PR, collect these three REST resources with `per_page=100`, incrementing `page` until the returned array has fewer than 100 entries:

- `/repos/{owner}/{repo}/issues/{number}/comments`: PR conversation / issue discussion.
- `/repos/{owner}/{repo}/pulls/{number}/comments`: inline review comments and replies.
- `/repos/{owner}/{repo}/pulls/{number}/reviews`: submitted review summaries and states, including empty approval bodies.

Keep repository, PR URL, comment/review URL, numeric ID, author login/type, creation/submission and update dates, complete body, and original API page. Classify author replies separately from human reviewers and bots; an approval with an empty body counts as a review event, not a substantive recommendation. Deleted comments, unpublished reviews, and inaccessible content cannot be recovered by these public reads. REST comment collection does not by itself certify GraphQL thread resolution state. No comments or messages were posted to GitHub.

## Rules with reviewer evidence

The repeated themes below come from human review comments. They are not a statistical ranking of every sentence in the collection. Each rule includes a concrete ORBIT review question; it does not claim that an associated ORBIT fix has already landed.

### R1. Preserve data, memory and resource ordering together

**Reviewer evidence.** ShangkunLi identified WAW ordering omitted from a purported RAW-oriented allocation rule in [neura#307](https://github.com/coredac/neura/pull/307#discussion_r3039150904), asked explicitly about WAR/WAW in [#311](https://github.com/coredac/neura/pull/311#discussion_r3200005664), and showed an actual context-order deadlock in [#311](https://github.com/coredac/neura/pull/311#discussion_r3227057571).

**Distillation.** A legal dataflow schedule must include all relevant dependencies and must remain acyclic after imposing resource/context order. A visually plausible placement is insufficient.

**ORBIT check.** Validate source-version RAW/WAR/WAW dependencies, copy availability, explicit scheduling predecessors, and chosen resource serialization. Negative tests should reverse a required memory edge or create a context/resource cycle and require rejection. Distinguish a critical-path scoring edge from an execution-order obligation.

**Full-collection follow-up.** [neura#254](https://github.com/coredac/neura/pull/254#discussion_r2744774640) distinguishes readiness/trigger memrefs from read/write access roles. [#254](https://github.com/coredac/neura/pull/254#discussion_r2750608925) shows that an input handle and a task's returned handle can denote the same physical storage. Track storage identity, logical version/readiness, and actual access separately; do not invent a second SRAM for an alias or discard an ordering edge because a returned handle is not an ordinary read.

### R2. Resource count, shape, replication and partitioning have different meanings

**Reviewer evidence.** In [neura#269](https://github.com/coredac/neura/pull/269#issuecomment-3957019369), ShangkunLi rejected dividing trip count by the number of CGRAs when those CGRAs merely form one larger mapping fabric. In [#307](https://github.com/coredac/neura/pull/307#discussion_r3013654035), the reviewer described count and shape as coupled upstream decisions that allocation must consume.

**Distillation.** More assigned resources do not prove replicated execution or a reduction in iteration count.

**ORBIT check.** Preserve selected `resourceCount`, `resourceShape`, placement, and replica/partition decisions through graph construction, timing and lowering. Apply work division only when source-domain partitioning and full output coverage are proved. Test a nontrivial count/shape input, not only default one-core plans.

### R3. Test consumption of public decisions and actual generated behavior

**Reviewer evidence.** [neura#307](https://github.com/coredac/neura/pull/307#issuecomment-4191912769) requests a task-graph input carrying count and shape, followed through allocation. [#222](https://github.com/coredac/neura/pull/222#discussion_r2660630490) notes that a claimed mapping did not run because its mapping pass was disabled. [#245](https://github.com/coredac/neura/pull/245#discussion_r2724087935) asks for the real end-to-end lowering process.

**Distillation.** Parser acceptance, labels, and intermediate plans cannot establish that a selected decision reaches execution.

**ORBIT check.** Use source → selected graph → implementation → independent check tests. Mutate a selected route, shape or implementation and verify the resulting graph/IR changes or rejects it. Keep CPU contracts, actual compiler success, numeric board validation, and measured speedup as separate claims.

**Amoeba follow-up.** [amoeba#2](https://github.com/coredac/amoeba/pull/2#discussion_r3969297139) gives a concrete branching-counter counterexample to a single-chain analysis. State and check the supported topology; reject unsupported input with a diagnostic rather than quietly treating it as the supported subset. Assertions can document internal invariants, but externally supplied malformed or unsupported IR still needs an ordinary failure path.

**Lowering follow-up.** [neura#173](https://github.com/coredac/neura/pull/173#discussion_r2455693037) requests checking that the source dialect is absent after lowering. [#173](https://github.com/coredac/neura/pull/173#discussion_r2474339638) asks for a case the custom lowering cannot handle. Check both the expected final IR and the deliberate unsupported/fallback path; a fallback must preserve semantics and be identified in the result.

### R4. Keep pass responsibilities and shared helpers explicit

**Reviewer evidence.** [neura#279](https://github.com/coredac/neura/pull/279#issuecomment-4148266950) asks to land allocation independently of optimization. [#307](https://github.com/coredac/neura/pull/307#discussion_r3039150904) requires allocation to work even without a streaming-fusion pass. [#279](https://github.com/coredac/neura/pull/279#discussion_r2954612983) asks for a utility usable with manually assigned resource counts. [#311](https://github.com/coredac/neura/pull/311#discussion_r3205688048) challenges overlapping passes.

**Distillation.** An optimizer chooses decisions; a consumer realizes those decisions; a checker validates them. Share semantic helpers instead of duplicating planner logic or requiring an unrelated preceding pass.

**ORBIT check.** Reuse the public ProgramGraph and established target abstractions. Exact replay must not silently re-optimize, repair, or replace selected copies/routes. A backend witness must state its supported domain and physical refinements.

**Amoeba follow-up.** [amoeba#2](https://github.com/coredac/amoeba/pull/2#discussion_r4006419425) distinguishes helpers shared across strategies from private strategy methods. [amoeba#10](https://github.com/coredac/amoeba/pull/10#discussion_r4071917225) asks to merge overlapping trip-count logic, while [#10](https://github.com/coredac/amoeba/pull/10#discussion_r4073062683) questions splitting one pass across unnecessary files. Reuse and modularity do not require a new public API or file for every small helper; organize code by actual ownership and reuse.

### R5. Separate profiling inputs, scheduling outputs and runtime meaning

**Reviewer evidence.** [neura#311](https://github.com/coredac/neura/pull/311#issuecomment-4402887333) distinguishes input duration from output placement/order and warns that an absolute clock start is misleading for asynchronous data-driven hardware. [#311](https://github.com/coredac/neura/pull/311#discussion_r3199762949) asks what duration means without profiling.

**Distillation.** A structural slot or ordinal is not a measured duration, and an input estimate is not a produced execution timestamp.

**ORBIT check.** Name phase/context indices, durations, invocation boundaries and measured wall time distinctly. Preserve provenance and units. The historical suggestion to use a structural default of one must **not** be imported as a physical timing estimate: uncalibrated physical costs remain unknown. Keep modeled lower bounds, feasible estimates, completed calls and cycle measurements distinct.

**Failure-state follow-up.** [neura#254](https://github.com/coredac/neura/pull/254#discussion_r2764070281) warns against returning `(0,0)` when placement fails, since it is a real legal location. Use an explicit failure/unknown result rather than a plausible concrete decision. The same principle applies to missing timing, capacity, route, and implementation evidence.

### R6. State hardware conflict rules precisely

**Reviewer evidence.** [neura#286](https://github.com/coredac/neura/pull/286#issuecomment-4077099220) explains when routing and computation may share a register-cluster read. [#293](https://github.com/coredac/neura/pull/293#discussion_r2966150148) and [#293](https://github.com/coredac/neura/pull/293#discussion_r2971793531) distinguish read/write occupancy and legitimate concurrent reads/writes.

**Distillation.** A generic occupied flag or temporal-memory bound is not a substitute for the hardware's actual shared-port rules.

**ORBIT check.** Independently verify DMA channels/BDs/locks, buffer address ranges, physical core ownership, transfer routes and resource capacities. Logical copy validity does not automatically prove physical buffer reuse; conversely, legal internal phase-pool reuse must not be charged as simultaneous whole-call residency.

**Architecture follow-up.** [neura#215](https://github.com/coredac/neura/pull/215#discussion_r2635683665) replaces a fixed hop bound with one derived from architecture dimensions. [#173](https://github.com/coredac/neura/pull/173#discussion_r2474245440) asks about hardware limits on output count. Read supported limits from the selected architecture/capability contract and reject violations; do not generalize a validated single machine's constants into universal hardware rules.

### R7. Include compile/search scalability in algorithm review

**Reviewer evidence.** [neura#293](https://github.com/coredac/neura/pull/293#discussion_r2958130545) asks whether repeated occupation scans increase compilation time and stresses removing recorded state on retry. [#269](https://github.com/coredac/neura/pull/269#issuecomment-3957397844) requests pass timing on a real ResNet workload.

**Distillation.** Measure algorithmic cost and preserve rollback invariants; a faster heuristic decision is not enough if validating or materializing it explodes.

**ORBIT check.** Record model construction, solve, independent checking and end-to-end wall time separately, with RSS and timeout status. Cache only with complete source/architecture/model identities. Validate contracted ownership before applying a complexity shortcut. Never call a preprocessing allocation failure “solver memory use” or a timeout “infeasible”.

### R8. Explain heuristic priorities and verify regressions broadly

**Reviewer evidence.** [neura#222](https://github.com/coredac/neura/pull/222#discussion_r2655515698) questions ranking degree ahead of critical operations. [#279](https://github.com/coredac/neura/pull/279#discussion_r2954556425) requests the parameter rationale. [#195](https://github.com/coredac/neura/pull/195#issuecomment-3578249948) asks why only one mapping result changed when shared logic should affect others.

**Distillation.** Document what a priority/weight optimizes, check global effects, and compare the real baseline in the same domain.

**ORBIT check.** Preserve deterministic ordering and report improved, unchanged and regressed cases. Explain CP-SAT/ILP objective, feasibility domain, remaining bound/gap and timeout. A finite-domain optimum proves that stated model/domain only; extending solver time cannot introduce an absent backend implementation.

**Final-page evidence.** [neura#173](https://github.com/coredac/neura/pull/173#discussion_r2485217848) explicitly asks to avoid nondeterministic symbols in checks. Use stable semantic assertions or FileCheck captures; do not loosen a correctness assertion into a wildcard merely to accommodate inconsistent results.

### R9. Exercise the resource-constrained feature, not only its easy default

**Reviewer evidence.** [neura#311](https://github.com/coredac/neura/pull/311#issuecomment-4436433298) recommends 1×1/1×2 arrays so tests actually exercise temporal reuse instead of assigning every task context zero.

**Distillation.** Choose workloads and architectures that trigger the mechanism being tested.

**ORBIT check.** Include capacity boundaries, dependent and independent tasks, nontrivial buffer reuse, multiple invocations, and conflicting resource assignments. Keep one legal case and one focused mutation for each claimed invariant. Hardware speedup tests still require a fair paired boundary; a constrained synthetic contract does not establish physical performance.

### R10. Use clear types, names and explanations; keep format changes controlled

**Reviewer evidence.** [neura#293](https://github.com/coredac/neura/pull/293#discussion_r2984626104) requests explicit types where the right-hand side is unclear. [#293](https://github.com/coredac/neura/pull/293#discussion_r2992603625) recommends a narrower operation type. [#293](https://github.com/coredac/neura/pull/293#discussion_r2985410370) proposes LLVM's changed-line formatting workflow. [#254](https://github.com/coredac/neura/pull/254#issuecomment-3857839870) asks for descriptive, punctuated comments.

**Distillation.** Types and names should reveal the contract; comments should explain why a condition, sentinel, interval or fallback is needed. Apply the repository's LLVM formatting consistently.

**ORBIT check.** Prefer typed MLIR operations and explicit types when deduction hides semantics. Explain non-obvious half-open bounds, missing costs and supported subsets. Avoid combining unrelated format churn with a semantic fix. Historical Coredac requests for snake_case or `this->` are local preferences, not grounds to override current ORBIT/LLVM naming conventions.

**Amoeba follow-up.** [amoeba#2](https://github.com/coredac/amoeba/pull/2#discussion_r4015078827) explicitly rejects copying AI conversation context into comments. Comments must explain the final code for a new maintainer, without references to the drafting conversation or unexplained claims about “historical behavior”. [#2](https://github.com/coredac/amoeba/pull/2#discussion_r4015091103) also asks for a meaningful distinction between orientation and rotation; use terminology that corresponds to a precise supported decision.

### R11. Keep claims, comments and PR scope synchronized with implementation

**Reviewer evidence.** [neura#293](https://github.com/coredac/neura/pull/293#issuecomment-4122748886) rejects a proposed “long-term” merge solution because destination placement is not known at that stage. [#307](https://github.com/coredac/neura/pull/307#discussion_r3039078019) corrects language implying candidate task pairs are already fused. [#195](https://github.com/coredac/neura/pull/195#discussion_r2632082088) asks to separate parameter experiments from the current fix.

**Distillation.** Describe what the current code proves and realizes. Keep speculative work and unrelated experiments separate.

**ORBIT check.** Receipts must identify logical selections, implemented physical refinements, unsupported dimensions and unpriced work. Do not equate a selected logical group with a physical configuration, a hardware-capable library with a registered search candidate, or model-optimal with hardware-optimal.

### R12. Make review iteration and stacked changes traceable

**Reviewer evidence.** [neura#173](https://github.com/coredac/neura/pull/173#issuecomment-3459740422) requests a reply to each review concern. [#179](https://github.com/coredac/neura/pull/179#issuecomment-3456808695) asks to land the underlying change first. [amoeba#10](https://github.com/coredac/amoeba/pull/10#issuecomment-5716533599) raises stale-base and non-main merge-target confusion.

**Distillation.** Keep the reviewed branch/base, concrete changes and validation easy to follow.

**ORBIT check.** Maintain a rule → change → test/evidence mapping. Preserve frozen tools/raw experiments and identify a new compiler revision when behavior changes. Updating this document does not authorize posting review replies, changing another repository, merging, or pushing.

## Applying these rules to ORBIT

For each new fix, record the rule ID, exact affected code, meaningful positive/negative test, and result. A source link establishes the origin of guidance; it does not establish that ORBIT already satisfies it. Avoid bulk edits justified only by style preferences, and never weaken a checker to make a selected plan appear implemented.

The root agent maintains the separate rule → implementation → test mapping; this guidance document does not certify those fixes. Collection coverage is sealed as described above. The guidance's exact comment references can be checked against the raw cache by running `python3 /tmp/orbit-review-distillation-raw/build-guidelines-evidence-index.py`; its `guidelines-evidence-index.json` preserves each cited author's identity, dates, complete original body and source page.
