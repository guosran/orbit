#!/usr/bin/env python3
"""CPU-only external ABI fixture: checks general priority search and replay ledger.

This binds an actual fixture file hash for importer coverage. It is not an NPU
execution or an arithmetic/backend-equivalence proof.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile


def main():
    binary = pathlib.Path(sys.argv[1]).resolve()
    binding = pathlib.Path(__file__).resolve()
    source_id = "0" * 64
    source = {
        "schema": "orbit.program-operator-source.v1",
        "source_id": source_id,
        "numerical_contract": "CPU-only structural scheduling fixture",
        "bindings": [{"id": "fixture", "file": str(binding),
                      "sha256": hashlib.sha256(binding.read_bytes()).hexdigest(),
                      "symbol": "fixture"}],
        "values": [{"id": name, "dtype": "f32", "shape": [4],
                    "role": role, "layout": "row_major"}
                   for name, role in [("x", "input"), ("a", "output"),
                                      ("b", "temporary"), ("c", "output")]],
        "operators": [dict(id=name, kind="fixture", binding="fixture",
                           source_ordinal=i, inputs=inputs, outputs=[output],
                           dependencies=[], attributes={})
                      for i, (name, inputs, output) in enumerate(
                          [("A", ["x"], "a"), ("B", ["x"], "b"),
                           ("C", ["b"], "c")])],
    }
    implementations = []
    for i, (device, inputs, outputs, duration) in enumerate(
            [("cpu", [0], [1], 2), ("cpu", [0], [2], 1),
             ("npu", [2], [3], 10)]):
        implementations.append(dict(
            id=f"impl{i}", source_id=source_id, artifact_identity="fixture",
            entrypoint=f"fixture{i}", device=device, resource=device,
            memory="runtime-abi", source_operators=[i], inputs=inputs,
            outputs=outputs, wall_ms=duration, executable=True,
            launch_count=int(device == "npu"),
            cost_boundary="inclusive-region-wall",
            transport_ownership="implementation-inclusive",
            visibility_policy="host-materialized",
            execution_variant=dict(
                runtime_context_identity="fixture", dataflow="fixture",
                residence_policy="fixture", replicas=1,
                work_distribution="single-invocation",
                resource_demands={device: 1},
                execution_policy="source-dependency-ready",
                source_ordering_identity=source_id,
                async_runtime_proof_identity="CPU-only-contract-fixture")))
    catalogue = dict(
        schema="orbit.program-operator-implementations.v1", source_id=source_id,
        execution_registration_identity="fixture-registration",
        execution_policy="source-dependency-ready",
        # One seed deliberately cannot discover B-before-A by template search.
        ready_priority_policies=["source-order"],
        fixed_application_invocation={"wall_ms": 0},
        implementations=implementations,
        choice_groups=[dict(id=f"g{i}", choices=[dict(id="only", implementations=[f"impl{i}"])])
                       for i in range(3)],
        architecture=dict(
            domains=[dict(id="system", kind="system", parent="")],
            resources=[dict(id=d, kind="compute", device=d, capacity=1,
                            domain="system") for d in ["cpu", "npu"]],
            memories=[dict(id="runtime-abi", kind="system-bo", domain="system",
                           capacity_bytes=4096, accessible_resources=["cpu", "npu"])],
            links=[]))
    with tempfile.TemporaryDirectory(prefix="orbit-ready-cli-") as temporary:
        root = pathlib.Path(temporary)
        src, cat, out = [root / f"{name}.json" for name in ["source", "catalogue", "result"]]
        src.write_text(json.dumps(source))
        cat.write_text(json.dumps(catalogue))
        result = subprocess.run([str(binary), str(src), str(cat), str(out)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(out.read_text())
        joint = report["joint_alternating_search"]
        admission = report["registered_domain_admission"]
        assert admission["registered_dimensions"] == 1
        assert admission["registered_dimension_choices"] == 1
        assert admission["caller_model_variable_quota"] >= admission["registered_dimension_choices"]
        assert admission["generic_api_default_changed"] is False
        assert admission["registered_choices_dropped"] is False
        initialization = report["coordinate_initializer"]
        assert initialization["enabled"] and initialization["passes"] == 1
        assert initialization["attempted"] and initialization["completed"]
        assert initialization["has_priced_seed"] and initialization["seed_cost_ms"] == 13
        assert initialization["search_stopping_rule"] is False
        checkpoint_rows = [json.loads(line) for line in pathlib.Path(str(out) + ".checkpoint.jsonl").read_text().splitlines()]
        assert sum(row["evaluation_phase"] == "coordinate-initialization" for row in checkpoint_rows) == initialization["evaluations"]
        assert report["beam_trace"]["rows"] == len(checkpoint_rows)
        assert sum(report["actual_evaluation_breakdown"].values()) == report["actual_cost_evaluations"]
        assert report["full_model_predicted_ms"] == 11, report
        assert report["selected_graph_independently_checked"]
        assert joint["temporal_improvements"] == 1, joint
        assert joint["temporal_evaluations"] > 0, joint
        assert "no improving adjacent" in joint["stop_reason"]
        rows = [json.loads(line) for line in pathlib.Path(joint["action_trace"]).read_text().splitlines()]
        assert any(row["kind"] == "accepted-priority" for row in rows)
        assert report["selected_cover_prediction"]["estimated_ms"] == 11
        assert report["selected_cover_prediction"]["inclusive_invocation_work_ms"] == 13
        # A manual baseline must preserve the specified source-order priority;
        # running even a singleton-domain search would optimize it from 13 to 11.
        manual = dict(schema="orbit.program-manual-choices.v1", source_id=source_id,
                      execution_registration_identity="fixture-registration",
                      purpose="manual-baseline", ready_priority_policy="source-order",
                      choices=[dict(group_id=f"g{i}", choice_id="only") for i in range(3)])
        manual_path, manual_out = root / "manual.json", root / "manual-result.json"
        manual_path.write_text(json.dumps(manual))
        completed = subprocess.run([str(binary), str(src), str(cat), str(manual_out),
                                    "--materialize-choices", str(manual_path)],
                                   capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        baseline = json.loads(manual_out.read_text())
        assert baseline["mode"] == "manual-materialization"
        assert baseline["search_performed"] is False
        assert baseline["full_model_predicted_ms"] == 13, baseline
        assert baseline["selected_graph_independently_checked"]
        assert not pathlib.Path(str(manual_out) + ".checkpoint.jsonl").exists()
        assert not pathlib.Path(str(manual_out) + ".beam-trace.jsonl").exists()
        bad_cases = [dict(manual, source_id="wrong"),
                     dict(manual, execution_registration_identity="wrong"),
                     dict(manual, choices=manual["choices"][:-1]),
                     dict(manual, choices=[manual["choices"][0]] * 3),
                     dict(manual, ready_priority_policy="unregistered"),
                     dict(manual, choices=[dict(group_id="g0", choice_id="unknown"),
                                           *manual["choices"][1:]])]
        for i, bad in enumerate(bad_cases):
            manual_path.write_text(json.dumps(bad))
            failed = subprocess.run([str(binary), str(src), str(cat),
                                     str(root / f"bad-manual-{i}.json"),
                                     "--materialize-choices", str(manual_path)],
                                    capture_output=True, text=True)
            assert failed.returncode != 0, bad
        # Warm starts map exact implementation/source contracts, so neither old
        # group order/IDs nor a prior cost can select the new candidate blindly.
        warm_catalogue = json.loads(json.dumps(catalogue))
        warm_catalogue["execution_registration_identity"] = "fresh-fixture-registration"
        warm_catalogue["ready_priority_policies"] = ["source-order", "reverse-source-order",
                                                     "cpu-first", "npu-first", "longest-first"]
        warm_catalogue["implementations"][2]["wall_ms"] = 20
        warm_catalogue["choice_groups"] = list(reversed(warm_catalogue["choice_groups"]))
        for i, group in enumerate(warm_catalogue["choice_groups"]):
            group["id"] = f"fresh-component-{i}"
            group["choices"][0]["id"] = f"fresh-choice-{i}"
        warm_cat_path, warm_out = root / "warm-cat.json", root / "warm-result.json"
        warm_cat_path.write_text(json.dumps(warm_catalogue))
        warmed = subprocess.run([str(binary), str(src), str(warm_cat_path), str(warm_out),
                                  "--warm-start", str(out)], capture_output=True, text=True)
        assert warmed.returncode == 0, warmed.stdout + warmed.stderr
        warm_report = json.loads(warm_out.read_text())
        assert warm_report["full_model_predicted_ms"] == 21, warm_report
        assert warm_report["cover_warm_start"]["historical_costs_reused"] is False
        assert warm_report["cover_warm_start"]["old_assignment_indices_used"] is False
        assert warm_report["cover_warm_start"]["prior_ready_priority_ignored"] is True
        assert warm_report["cover_warm_start"]["initial_state_proposals"] == 5
        assert {row["group_id"] for row in warm_report["cover_warm_start"]["mapped_choices"]} == {
            f"fresh-component-{i}" for i in range(3)}
        merged = json.loads(json.dumps(warm_catalogue))
        merged["choice_groups"] = [dict(id="merged-new-component", choices=[
            dict(id="merged-actual-cover", implementations=["impl0", "impl1"])]),
            dict(id="other-new-component", choices=[dict(id="only", implementations=["impl2"])])]
        merged_path, merged_out = root / "merged-cat.json", root / "merged-result.json"
        merged_path.write_text(json.dumps(merged))
        merged_result = subprocess.run([str(binary), str(src), str(merged_path), str(merged_out),
                                         "4", "--warm-start", str(out)], capture_output=True, text=True)
        assert merged_result.returncode == 0, merged_result.stdout + merged_result.stderr
        assert json.loads(merged_out.read_text())["full_model_predicted_ms"] == 21
        changed_artifact = json.loads(json.dumps(report))
        changed_artifact["selected_program_graph"]["external_implementation_selection"]["regions"][0][
            "artifact_identity"] = "different-unregistered-artifact"
        missing_id = json.loads(json.dumps(report))
        missing_id["selected_program_graph"]["external_implementation_selection"]["regions"][0][
            "id"] = "missing-old-implementation"
        for i, bad in enumerate((dict(report, source_id="wrong"),
                                  dict(report, selected_graph_independently_checked=False),
                                  baseline, changed_artifact, missing_id)):
            prior_path = root / f"bad-prior-{i}.json"
            prior_path.write_text(json.dumps(bad))
            failed = subprocess.run([str(binary), str(src), str(warm_cat_path),
                                     str(root / f"bad-warm-result-{i}.json"),
                                     "--warm-start", str(prior_path)], capture_output=True, text=True)
            assert failed.returncode != 0, bad
        for impl in catalogue["implementations"]:
            impl["controller_features"] = {"cpu_callbacks": int(impl["device"] == "cpu"),
                                           "npu_callbacks": int(impl["device"] == "npu")}
        catalogue["invocation_control_model"] = dict(
            schema="orbit.program-invocation-control-model.v1",
            identity="CPU-fixture-controller", source_id=source_id,
            intercept_ms=1, coefficients={"cpu_callbacks": 3, "npu_callbacks": 2},
            boundary="outside-inclusive-invocation-wall", fit={"scope": "synthetic CPU fixture"})
        cat.write_text(json.dumps(catalogue))
        controlled = root / "controlled.json"
        result = subprocess.run([str(binary), str(src), str(cat), str(controlled)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(controlled.read_text())
        assert report["full_model_predicted_ms"] == 20, report
        assert report["cost_components"]["physical_callback_makespan_ms"] == 11
        assert report["cost_components"]["invocation_control_ms"] == 9
        assert report["cost_components"]["controller_is_approximate"]
        # Every full objective is unknown when application overhead is missing,
        # despite a physically feasible, fully priced callback schedule. This
        # exercises hasBest=true/cost=null rather than just no feasible result.
        unknown_app = json.loads(json.dumps(catalogue))
        unknown_app["fixed_application_invocation"] = {}
        unknown_app_path = root / "unknown-application.json"
        unknown_app_path.write_text(json.dumps(unknown_app))
        unknown_app_out = root / "unknown-application-result.json"
        completed = subprocess.run([str(binary), str(src), str(unknown_app_path),
                                    str(unknown_app_out)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        unknown_report = json.loads(unknown_app_out.read_text())
        assert unknown_report["report"]["has_best"] is True
        assert unknown_report["full_model_predicted_ms"] is None
        assert unknown_report["coordinate_initializer"]["attempted"] is True
        assert unknown_report["coordinate_initializer"]["completed"] is False
        assert unknown_report["coordinate_initializer"]["passes"] == 0
        assert unknown_report["coordinate_initializer"]["seed_cost_ms"] is None
        assert unknown_report["joint_alternating_search"]["enabled"] is False
        assert unknown_report["joint_alternating_search"]["passes"] == 0
        assert "no priced" in unknown_report["joint_alternating_search"]["stop_reason"]
        assert unknown_report["joint_alternating_search"]["action_trace"] is None
        assert unknown_report["cost_components"]["other_fixed_application_ms"] is None
        unknown_local = json.loads(json.dumps(catalogue))
        for implementation in unknown_local["implementations"]:
            implementation["wall_ms"] = None
        unknown_local_path = root / "unknown-local-prices.json"
        unknown_local_path.write_text(json.dumps(unknown_local))
        unknown_local_out = root / "unknown-local-prices-result.json"
        completed = subprocess.run([str(binary), str(src), str(unknown_local_path),
                                    str(unknown_local_out)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        unknown_report = json.loads(unknown_local_out.read_text())
        assert unknown_report["full_model_predicted_ms"] is None
        assert unknown_report["coordinate_initializer"]["attempted"] is True
        assert unknown_report["coordinate_initializer"]["completed"] is False
        assert unknown_report["coordinate_initializer"]["passes"] == 0
        assert unknown_report["joint_alternating_search"]["enabled"] is False
        assert unknown_report.get("cost_components") is None
        catalogue["invocation_control_model"]["coefficients"]["cpu_callbacks"] = None
        cat.write_text(json.dumps(catalogue))
        invalid = subprocess.run([str(binary), str(src), str(cat), str(root / "invalid.json")],
                                 capture_output=True, text=True)
        assert invalid.returncode != 0, "unknown controller price was silently accepted"
    print(json.dumps({"status": "passed", "general_priority_improvement_ms": [13, 11]}))


if __name__ == "__main__":
    main()
