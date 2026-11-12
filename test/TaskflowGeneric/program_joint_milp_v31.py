#!/usr/bin/env python3
"""Compare direct solvers with an independent tiny scheduling oracle.

Costs are synthetic integer nanoseconds. These contracts establish encoding
equivalence and coverage semantics, not NPU performance or hardware optimality.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import program_joint_v23 as fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", type=Path)
    parser.add_argument("--scip-off", action="store_true")
    parser.add_argument("--cp-sat-off", action="store_true")
    args = parser.parse_args()
    modes = [("cp-sat", args.cp_sat_off), ("scip", args.scip_off)]
    checks = 0
    rows = []
    with tempfile.TemporaryDirectory(prefix="orbit-joint-milp-") as temporary:
        root = Path(temporary)
        source = root / "shared.mlir"
        source.write_text(fixture.source())

        def run(label, config, solver, succeeds=True):
            folder = root / label
            path = root / (label + ".json")
            path.write_text(json.dumps(config, indent=2) + "\n")
            command = [str(args.tool.resolve()), str(source),
                       "--joint-solver=" + solver,
                       "--target-config=" + str(path),
                       "--joint-program-search=" + str(folder),
                       "--joint-seconds=10"]
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=30)
            assert (result.returncode == 0) == succeeds, result.stderr
            return json.loads((folder / "solution.json").read_text())

        # The oracle enumerates resource orders independently of either solver.
        # Shared input/DMA and payload capacity change the optimum, so a model
        # that drops communication or aggregate memory fails these cases.
        for capacity, independent_dma in [(8, False), (8, True),
                                          (12, False), (12, True)]:
            config = fixture.config(capacity, independent_dma)
            optimum, orders = fixture.oracle(capacity, independent_dma)
            domain_ids = []
            for solver, unavailable in modes:
                result = run(f"c{capacity}-{independent_dma}-{solver}",
                             config, solver)
                assert result["has_plan"] and result["solver"] == solver
                assert result["status"] == ("UNAVAILABLE" if unavailable
                                             else "OPTIMAL"), result
                if not unavailable:
                    assert result["makespan_ns"] == optimum, result
                    assert abs(result["best_bound_ns"] - optimum) < 1e-6
                    assert result["relative_gap"] == 0
                    assert result["full_domain_optimal"]
                assert result["pipeline_total_ms"] >= result["solve_ms"]
                assert result["pipeline_total_ms"] >= result["preprocessing_ms"]
                domain_ids.append(result["domain_id"])
                checks += 8
                rows.append(dict(capacity=capacity,
                                 independent_dma=independent_dma, solver=solver,
                                 optimum_ns=optimum, enumerated_orders=orders,
                                 status=result["status"]))
            assert domain_ids[0] == domain_ids[1]
            checks += 1

        # A supported alternative with no price leaves a known incumbent, but
        # prevents an optimum claim over the original intended choice domain.
        unknown = fixture.config(12, True)
        links = unknown["program_joint"]["architecture"]["links"]
        extra = copy.deepcopy(links[0])
        extra.update(id="unpriced", bandwidth_bytes_per_ms=0)
        links.append(extra)
        for solver, unavailable in modes:
            result = run("unknown-alternative-" + solver, unknown, solver)
            assert result["has_plan"] and not result["costs_complete"]
            assert result["status"] == ("UNAVAILABLE" if unavailable
                                         else "FEASIBLE"), result
            assert not result["full_domain_optimal"]
            assert result["optimization_scope"] == "known-cost-generated-subset"
            checks += 5

        # Missing every route price is unknown coverage, not a free schedule or
        # a proof that no physical implementation exists.
        unknown = fixture.config(12, True)
        for link in unknown["program_joint"]["architecture"]["links"]:
            link["bandwidth_bytes_per_ms"] = 0
        for solver, unavailable in modes:
            result = run("all-route-prices-missing-" + solver,
                         unknown, solver, succeeds=False)
            assert not result["has_plan"] and not result["costs_complete"]
            assert result["status"] == ("UNAVAILABLE" if unavailable
                                         else "UNKNOWN"), result
            assert not result["full_domain_optimal"]
            checks += 4

    print(json.dumps(dict(passed=True, checks=checks, cases=rows,
                          hardware_executed=False)))


if __name__ == "__main__":
    main()
