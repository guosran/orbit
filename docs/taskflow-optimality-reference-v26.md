# Finite-domain optimality references v26

The oracle suite in `test/TaskflowGeneric/fixtures/oracle-v26` supplies explicit
finite domains for small GEMM, fork/join, shared-input and capacity cases. All
8 benchmark instances prove the minimum of the declared public domain under the
frozen analytical evaluator in less than 44 seconds (64–1536 callbacks). These
are model costs, not hardware latency. Partition, scalar implementation and
other fixed axes are visible in every space artifact. Full NPU abstract-space
optimality is not claimed.

Run `test/TaskflowGeneric/optimality_reference_v26.py --tool TOOL --output NEW_DIR`
to generate CP references and same-domain beam quality results. The beam adapter
uses existing `selectSearchBeam` on legal single-dimension neighbors; it does
not replace or represent the unmodified legacy default search. It cannot claim
optimality. A reference requires domain exhaustion and known feasibility/costs.

Public spaces export every choice/table, source fingerprint and evaluator-config
identity. Compare compiler SHA as well as domain/evaluator IDs. Configuration
identity alone does not freeze executable behavior. Each run checks the frozen
compiler SHA before and after execution and records complete traces. The driver
checks exhaustive coverage, minimum cost and pointwise beam/oracle agreement.

The `check-taskflow-public-search` target includes solver ON/OFF contracts,
beam checks and v26 identity/null-domain regressions. Compound placement makes
resource/memory change atomically, avoiding an artificial invalid-neighbor trap.

The full report, proof times, gap tables and independent solver-OFF enumeration
are in sibling `orbit-artifact/taskflow/docs/taskflow-optimality-reference-v26.md`
and `orbit-artifact/results/taskflow/optimality-reference-v26-r1`.
