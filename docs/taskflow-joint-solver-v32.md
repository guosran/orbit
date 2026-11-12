# Joint MILP warm start and exact resource reduction (v32)

The SCIP backend now submits the independently checked baseline as a complete primal start, including all optional assignment, route, schedule, maximum and resource-comparison auxiliary variables. `SCIPcheckSol` checks it before `SCIPaddSolFree` stores it. Failure is reported explicitly and retains the checked baseline; no baseline variable is fixed and the finite choice domain is unchanged.

For capacity-one resources, each unordered positive-duration interval pair now needs one conditional ordering variable and two inequalities. Present intervals must satisfy `end_i <= start_j` or `end_j <= start_i`; absent intervals relax the disjunction. Half-open endpoints may touch. Capacities above one retain the exact integer start-anchor occupancy formulation; zero-duration intervals consume no capacity.

A guarded valid inequality also strengthens the makespan bound. It applies only when every represented group has one or two inputs and outputs, configuration input/output limits are both two, and every positive-duration legal transfer option uses a common unary resource. Thus only two one-input/one-output groups can share a configuration; any group saturating either port limit must be alone. Let `D` be a group's minimum total transfer duration, `C` its serial compute duration, `S` setup time, `Cmin` the minimum compute duration of any one-input/one-output group, and `Omax` their maximum possible output duration. Such a unit group has weight `D + (S + max(0,Cmin-Omax))/2`; other groups have weight `D+C+S`. Twice the sum of selected weights, plus twice submit/complete time, lower-bounds twice the serial whole-program makespan. After the last input in a two-unit configuration, at least `Cmin` computation remains and only the other group's output can fill at most `Omax` of that transfer gap. All guards and safe-integer checks must hold; otherwise this optional inequality is omitted. No complete plans are pre-priced.

The typed contract passes **51 checks**, including independent calendar enumeration for capacity one/two, route choices, endpoint touching, zero duration, and a real source with asymmetric 4 ns/100 ns compute durations and long input transfers. The asymmetric case has independent optimum 2005 ns and catches the invalid substitution of maximum compute time for `Cmin`. Unknown-price/incomplete domains retain their previous conservative proof scope. LLVM formatting checks pass.

## Same-domain diagnostic

A short CPU-only verification reuses the exact v31 `fork2-rows2`, K=8 source and declared synthetic costs. Both backends used one worker, a 15-second whole-pipeline wall limit and 100,000-variable cap. The domain and model identities match:

- Domain: `a61f5de19b2cf9cf0a7edb7aaad57008d1d29e3acc1ed93d58557785a0fb26a9`.
- Model: `46d06bf2a719825c22baf48bfb5fca9aea5eaf2d509e1425fdee298fd3688170`.
- Frozen tool: `/tmp/orbit-milp-v32-final-r1/taskflow-opt`, SHA256 `917ce31602f436204658bf4971bc9f566ccc14c9625c9529d2b29cf96f7aabf8`.

| Backend | Status | Objective / bound (ns) | Gap | Variables / constraints | Model build | Solve |
|---|---|---|---|---|---|---|
| SCIP v32 | OPTIMAL | 102 / 102 | 0 | 47,590 / 97,770 | 109.1 ms | 1,385.3 ms |
| CP-SAT | FEASIBLE | 102 / 40 | 60.78% | 3,558 / 8,070 | 9.0 ms | 13,488.9 ms |

SCIP's external wall time was 1.69 seconds and peak RSS 500,272 KiB. Both emitted plans independently replayed successfully. The SCIP message confirms accepted primal start and the guarded unary-port inequality. The old v31 SCIP formulation reached its variable cap on this same domain; the new unary formulation fits the cap.

This is a limited diagnostic, not a broad solver ranking. The MILP now has a valid bound specialized to a checked resource/port class; CP-SAT has its existing hint and reserves 10% of remaining time for checking, so the internal search allocations differ despite equal pipeline limits. SCIP optimality remains subject to numerical tolerances followed by independent integer schedule checking. Costs are explicitly synthetic, and the result proves the declared finite model optimum rather than hardware optimality or NPU speedup.

Raw diagnostic evidence is in `/tmp/orbit-milp-v32-final-r1`, including both solver receipts, external time/RSS logs, replay output and the frozen tool. Typed test logs are `/tmp/milp-v32-test-r3.log`. Previous v31 raw results are preserved unchanged.
