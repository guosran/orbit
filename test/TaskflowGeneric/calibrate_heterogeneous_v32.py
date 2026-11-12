#!/usr/bin/env python3
"""Offline synthetic receipt/NNLS tests; no compiler, ELF or device executes."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("local_calibration", ROOT / "scripts/calibrate-heterogeneous.py")
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


class CalibrationContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="local-phase-calibration-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = [self.case("train-1", "train", 1), self.case("train-3", "train", 3),
                        self.case("heldout-2", "holdout", 2), self.case("heldout-4", "holdout", 4)]
        self.index = dict(schema="orbit.heterogeneous.calibration-index.v32",
                          runtime_identity=CAL.RUNTIME, execution_contract_identity="synthetic-contract",
                          ranking_min_relative_gap=.05,
                          feature_sets=dict(application={p: ["call"] for p in CAL.APP_PHASES},
                                            cpu=dict(body=["call", "source_operations"]),
                                            npu={p: ["call"] for p in CAL.NPU_PHASES}),
                          cases=self.entries)
        self.index_path = self.root / "index.json"
        CAL.save(self.index_path, self.index)

    def ref(self, path, value):
        CAL.save(path, value)
        return dict(path=str(path), sha256=CAL.sha(path))

    def case(self, name, split, work):
        directory = self.root / name
        directory.mkdir()
        source = directory / "source.mlir"
        library = directory / "synthetic-library"
        source.write_text("synthetic source identity only\n")
        library.write_text("synthetic artifact, never executed\n")
        region = dict(index=0, device="cpu", source="source.mlir", child_graph_id="child-" + name)
        manifest = dict(schema="orbit.heterogeneous.static.v32", id="plan-" + name,
                        graph_id="parent-" + name, execution_contract_identity="synthetic-contract",
                        execution_contract=dict(runtime_identity=CAL.RUNTIME, executor_sha256="synthetic-executor"),
                        regions=[region])
        manifest_path = directory / "manifest.json"
        CAL.save(manifest_path, manifest)
        artifact = dict(index=0, device="cpu", child_graph_id=region["child_graph_id"],
                        source_sha256=CAL.sha(source), library=str(library), library_sha256=CAL.sha(library))
        compiled = dict(schema="orbit.heterogeneous.compiled.v32", runtime_identity=CAL.RUNTIME,
                        executor_sha256="synthetic-executor", plan_id=manifest["id"],
                        manifest=str(manifest_path), manifest_sha256=CAL.sha(manifest_path), regions=[artifact])
        compiled_path = directory / "compiled.json"
        CAL.save(compiled_path, compiled)
        features = dict(schema="orbit.heterogeneous.edge-features.v32", parent_graph_id=manifest["graph_id"],
                        application=dict(call=1, regions=1, input_bytes=16, output_bytes=16),
                        regions=[dict(region=0, device="cpu", source_family="synthetic-local-family",
                                      implementations=["scalar"], child_graph_id=region["child_graph_id"],
                                      features=dict(call=1, source_operations=work))])
        identity = dict(runtime=CAL.RUNTIME, native_wait2=True, executor_sha256="synthetic-executor",
                        compiled=str(compiled_path), compiled_sha256=CAL.sha(compiled_path), plan_id=manifest["id"])
        body = 10 + 10 * work
        samples = [dict(warmup=warm, wall_ns=body + 3, reset_ns=1, control_ns=1, copy_ns=1,
                        regions=[dict(region=0, body_ns=body)]) for warm in (True, False, False)]
        return dict(id=name, split=split, ranking_group="same-source-shape", **{
            key: self.ref(directory / (key + ".json"), value) for key, value in
            dict(features=features, identity=identity, samples=samples,
                 result=dict(status="passed", calls=3, checks=[dict(Y=True)] * 3)).items()})

    def test_training_only_freeze_and_out_of_range_unknown(self):
        forbidden = {Path(entry[k]["path"]) for entry in self.entries if entry["split"] == "holdout"
                     for k in ("features", "identity", "samples", "result")}
        original = CAL.read

        def train_only_read(path):
            self.assertNotIn(Path(path), forbidden, "fit opened holdout evidence")
            return original(path)

        model_dir = self.root / "model"
        with patch.object(CAL, "read", side_effect=train_only_read):
            CAL.fit(self.index_path, model_dir)
        report = CAL.validate(self.index_path, model_dir, self.root / "validation")
        self.assertEqual(report["known"], 1)
        self.assertFalse(report["coverage_complete"])
        self.assertAlmostEqual(report["rows"][0]["prediction_ns"], 33)
        self.assertIsNone(report["rows"][1]["prediction_ns"])
        self.assertEqual(report["ranking"]["eligible"], 1)
        self.assertEqual(report["ranking"]["correct"], 0)
        model = CAL.read(model_dir / "model.json")
        self.assertEqual(len(model["regions"]), 1)
        self.assertNotIn("plans", model)
        self.assertTrue(all(c["value"] >= 0 for c in model["regions"][0]["phases"]["body"]["coefficients_ns"].values()))

    def mutate_receipt(self, entry, key, mutate):
        value = CAL.read(entry[key]["path"])
        mutate(value)
        entry[key] = self.ref(Path(entry[key]["path"]), value)

    def test_foreign_parent_or_child_graph_rejected(self):
        for field in ("parent_graph_id", "child_graph_id"):
            entry = copy.deepcopy(self.entries[0])
            def corrupt(value):
                if field == "parent_graph_id":
                    value[field] = "foreign"
                else:
                    value["regions"][0][field] = "foreign"
            self.mutate_receipt(entry, "features", corrupt)
            with self.assertRaisesRegex(ValueError, "foreign"):
                CAL.load_case(entry, self.index_path, self.index, {})
            # Restore the shared file for the next independently sealed case.
            self.setUp_restore_features()

    def setUp_restore_features(self):
        self.entries[0] = self.case_rebuild("train-1", "train", 1)

    def case_rebuild(self, name, split, work):
        import shutil
        shutil.rmtree(self.root / name)
        return self.case(name, split, work)

    def test_missing_or_changed_evidence_is_rejected(self):
        entry = copy.deepcopy(self.entries[0])
        Path(entry["samples"]["path"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            CAL.load_case(entry, self.index_path, self.index, {})

    def test_overlapping_or_missing_runtime_phases_rejected(self):
        for extra in (False, True):
            entry = copy.deepcopy(self.entries[0])
            def corrupt(value):
                if extra:
                    value[0]["regions"][0]["wait_ns"] = 1
                else:
                    value[0]["wall_ns"] += 1
            self.mutate_receipt(entry, "samples", corrupt)
            with self.assertRaises(ValueError):
                CAL.load_case(entry, self.index_path, self.index, {})
            self.setUp_restore_features()

    def test_numeric_failure_is_rejected(self):
        entry = copy.deepcopy(self.entries[0])
        self.mutate_receipt(entry, "result", lambda r: r["checks"][1].update(Y=False))
        with self.assertRaisesRegex(ValueError, "numeric"):
            CAL.load_case(entry, self.index_path, self.index, {})

    def test_alias_overlap_between_splits_is_rejected(self):
        self.index["cases"][2]["samples"] = copy.deepcopy(self.entries[0]["samples"])
        path = Path(self.index["cases"][2]["samples"]["path"])
        self.index["cases"][2]["samples"]["path"] = str(path.parent / "." / path.name)
        CAL.save(self.index_path, self.index)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            CAL.load_index(self.index_path)

    def test_training_artifact_mutation_after_freeze_rejected(self):
        model_dir = self.root / "model"
        CAL.fit(self.index_path, model_dir)
        compiled = CAL.read(CAL.read(self.entries[0]["identity"]["path"])["compiled"])
        Path(compiled["regions"][0]["library"]).write_text("changed artifact")
        with self.assertRaisesRegex(ValueError, "training evidence changed"):
            CAL.validate(self.index_path, model_dir, self.root / "validation")

    def test_multi_configuration_receipts_and_runtime_counts_are_bound(self):
        entry = copy.deepcopy(self.entries[0])
        identity = CAL.read(entry["identity"]["path"])
        compiled_path = Path(identity["compiled"])
        compiled = CAL.read(compiled_path)
        manifest_path = Path(compiled["manifest"])
        manifest = CAL.read(manifest_path)
        directory = manifest_path.parent
        air = directory / "program.air.mlir"
        air.write_text("synthetic NPU selection, never compiled")
        graph = manifest["regions"][0]["child_graph_id"]
        selections, configurations = [], []
        for i in range(2):
            selected = directory / ("selected-" + str(i) + ".mlir")
            selected.write_text("synthetic selected AIR " + str(i))
            physical = dict(graph_id=graph, runtime_contract_sha256="contract-" + str(i))
            CAL.save(str(selected) + ".physical.json", physical)
            proof = directory / ("proof-" + str(i) + ".json")
            CAL.save(proof, dict(program_graph_id=graph, runtime_contract_sha256=physical["runtime_contract_sha256"],
                                 generic_module_sha256="realized-" + str(i)))
            elf = directory / ("selected-" + str(i) + ".elf")
            elf.write_text("synthetic ELF " + str(i))
            selections.append(dict(air=selected.name))
            configurations.append(dict(air_sha256=CAL.sha(selected),
                physical_receipt_sha256=CAL.sha(str(selected) + ".physical.json"),
                runtime_proof=str(proof), runtime_proof_sha256=CAL.sha(proof),
                runtime_contract_sha256=physical["runtime_contract_sha256"],
                realized_runtime_sha256="realized-" + str(i), elf=str(elf), elf_sha256=CAL.sha(elf)))
        CAL.save(str(air) + ".physical.json", dict(graph_id=graph, configurations=selections,
                                                   physical_invocation_count=2))
        artifact = compiled["regions"][0]
        artifact.update(device="npu", configurations=configurations,
                        physical_receipt_sha256=CAL.sha(str(air) + ".physical.json"))
        manifest["regions"][0].update(device="npu", air=air.name, physical_invocation_count=2)
        CAL.save(manifest_path, manifest)
        compiled["manifest_sha256"] = CAL.sha(manifest_path)
        CAL.save(compiled_path, compiled)
        identity["compiled_sha256"] = CAL.sha(compiled_path)
        entry["identity"] = self.ref(Path(entry["identity"]["path"]), identity)
        features = CAL.read(entry["features"]["path"])
        features["application"]["configuration_count"] = 2
        features["regions"][0]["device"] = "npu"
        features["regions"][0]["features"]["configuration_count"] = 2
        entry["features"] = self.ref(Path(entry["features"]["path"]), features)
        samples = [dict(warmup=w, wall_ns=19, reset_ns=1, copy_ns=1, control_ns=1,
                        regions=[dict(region=0, sync_to_ns=1, submit_ns=3, wait_ns=5,
                                      sync_from_ns=7, configuration_count=2)]) for w in (True, False, False)]
        entry["samples"] = self.ref(Path(entry["samples"]["path"]), samples)
        loaded = CAL.load_case(entry, self.index_path, self.index, {})
        self.assertEqual(loaded["regions"][0]["wait"], 5)
        samples[1]["regions"][0]["configuration_count"] = 1
        entry["samples"] = self.ref(Path(entry["samples"]["path"]), samples)
        with self.assertRaisesRegex(ValueError, "runtime invocation count"):
            CAL.load_case(entry, self.index_path, self.index, {})

    def test_process_receipt_scope_does_not_mislabel_supervisor_as_python(self):
        entry = copy.deepcopy(self.entries[0])
        process = dict(schema="orbit.heterogeneous.process.v32", pid=37,
                       started_ns=1, ended_ns=2, returncode=0,
                       pid_scope="supervisor", identity_sha256=entry["identity"]["sha256"])
        entry["process"] = self.ref(self.root / "process.json", process)
        loaded = CAL.load_case(entry, self.index_path, self.index, {})
        self.assertIsNone(loaded["process_token"])
        process["pid_scope"] = "measurement-python"
        entry["process"] = self.ref(self.root / "process.json", process)
        self.assertEqual(CAL.load_case(entry, self.index_path, self.index, {})["process_token"], (37, 1))
        process["identity_sha256"] = "foreign"
        entry["process"] = self.ref(self.root / "process.json", process)
        with self.assertRaisesRegex(ValueError, "foreign process"):
            CAL.load_case(entry, self.index_path, self.index, {})

    def aggregate_rows(self, left=(10, 11, 12), right=(30, 31, 32), predictions=(11, 31)):
        rows = []
        for index, measurements in enumerate((left, right)):
            for process, measured in enumerate(measurements):
                rows.append(dict(ranking_group="same-program", witness=dict(runtime_contract="contract",
                    parent_graph_id="same-parent", region_graph_ids=["child-" + str(index)]),
                    prediction_ns=predictions[index], measured_ns=measured,
                    process_token=(100 + process, 1000 + process), process_status="synthetic"))
        return rows

    def test_three_process_typed_witness_gate_and_confident_top_one(self):
        result = CAL.aggregate_validation(self.aggregate_rows())
        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["group_accuracy"]["status"], "passed")
        self.assertEqual(result["top_one"][0]["status"], "passed")
        self.assertFalse(result["cp_speedup_claim"])
        # Same parent with a different ordered child graph must remain a
        # different implementation choice, rather than merging measurements.
        self.assertNotEqual(result["groups"][0]["witness_id"], result["groups"][1]["witness_id"])

    def test_overlap_ties_unknown_and_repeated_processes_do_not_pass_ranking(self):
        result = CAL.aggregate_validation(self.aggregate_rows((10, 20, 30), (11, 21, 31), (20, 21)))
        self.assertEqual(result["group_accuracy"]["status"], "passed")
        self.assertEqual(result["top_one"][0]["status"], "unknown")
        rows = self.aggregate_rows()
        for row in rows:
            row["process_token"] = (100, 1000)
        self.assertEqual(CAL.aggregate_validation(rows)["group_accuracy"]["status"], "diagnostic_unknown")
        rows = self.aggregate_rows(predictions=(11, None))
        result = CAL.aggregate_validation(rows)
        self.assertEqual(result["group_accuracy"]["total"], 2)
        self.assertEqual(result["group_accuracy"]["known"], 1)
        self.assertFalse(result["group_accuracy"]["coverage_complete"])
        self.assertEqual(result["top_one"][0]["status"], "unknown")

    def test_model_mutation_after_freeze_rejected(self):
        model_dir = self.root / "model"
        CAL.fit(self.index_path, model_dir)
        model = CAL.read(model_dir / "model.json")
        model["application"]["reset"]["coefficients_ns"]["call"]["value"] += 1
        CAL.save(model_dir / "model.json", model)
        with self.assertRaisesRegex(ValueError, "frozen model/input changed"):
            CAL.validate(self.index_path, model_dir, self.root / "validation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
