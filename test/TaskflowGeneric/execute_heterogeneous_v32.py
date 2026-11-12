#!/usr/bin/env python3
"""CPU-only typed runner contracts; fake XRT is instance-local, never installed.

Run with the runtime's NumPy/ml_dtypes Python. No device/runtime module is
imported: fake runs exercise only sequencing, ownership and fail-closed checks.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "typed_executor_under_test", ROOT / "scripts/execute-heterogeneous.py")
EXECUTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXECUTOR)


class ManifestContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="typed-executor-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        tool = self.root / "synthetic-tool-identity"
        tool.write_text("CPU contract fixture, never executed\n")
        files = {role: dict(path=str(tool), sha256=EXECUTOR.sha(tool))
                 for role in ("clang", "mlir-opt", "mlir-translate", "aircc", "peano")}
        self.manifest = dict(
            schema="orbit.heterogeneous.static.v32", id="synthetic-plan",
            execution_contract_identity="synthetic-contract-fixture",
            execution_contract=dict(runtime_identity=EXECUTOR.RUNTIME,
                                    executor_sha256=EXECUTOR.SOURCE_SHA, files=files),
            storage=[dict(index=0, name="A", role="input", dtype="f32",
                          shape=[4], bytes=16, realization="numpy-host")],
            regions=[dict(index=0, device="cpu", child_graph_id="child",
                          source="source.mlir", function="test",
                          abi=[dict(argument=0, storage=0, dtype="f32", shape=[4])])])
        self.path = self.root / "manifest.json"

    def save(self):
        self.path.write_text(json.dumps(self.manifest))
        return self.path

    def test_valid_typed_manifest_and_byte_mismatch(self):
        EXECUTOR.validate_manifest(self.save())
        self.manifest["storage"][0]["bytes"] = 8
        with self.assertRaises(ValueError):
            EXECUTOR.validate_manifest(self.save())

    def test_negative_storage_index_is_not_a_python_alias(self):
        # Python storage[-1] would otherwise pass geometry checks and alias A.
        self.manifest["regions"][0]["abi"][0]["storage"] = -1
        with self.assertRaises(ValueError):
            EXECUTOR.validate_manifest(self.save())

    def test_context_capacity_counts_all_npu_regions_before_device_creation(self):
        region = self.manifest["regions"][0]
        region.update(device="npu", physical_invocation_count=8)
        self.manifest["regions"].append(dict(region, index=1))
        contract = self.manifest["execution_contract"]
        contract["max_prepared_npu_contexts"] = 16
        EXECUTOR.validate_manifest(self.save())
        self.manifest["regions"][1]["physical_invocation_count"] = 9
        with self.assertRaisesRegex(ValueError, "exceeds prepared NPU context"):
            EXECUTOR.validate_manifest(self.save())
        for invalid in (0, -1, True, 16.0):
            contract["max_prepared_npu_contexts"] = invalid
            with self.assertRaisesRegex(ValueError, "positive integer"):
                EXECUTOR.validate_manifest(self.save())

    def test_foreign_runtime_and_executor_are_rejected(self):
        for key in ("runtime_identity", "executor_sha256"):
            old = self.manifest["execution_contract"][key]
            self.manifest["execution_contract"][key] = "foreign"
            with self.assertRaises(ValueError):
                EXECUTOR.validate_manifest(self.save())
            self.manifest["execution_contract"][key] = old

    def compiled(self):
        manifest = self.save()
        compiled = dict(schema="orbit.heterogeneous.compiled.v32",
                        manifest=str(manifest), manifest_sha256=EXECUTOR.sha(manifest),
                        executor_sha256=EXECUTOR.SOURCE_SHA,
                        runtime_identity=EXECUTOR.RUNTIME,
                        plan_id="synthetic-plan", regions=[])
        return compiled

    def test_physical_abi_overrides_public_abi_and_checks_scratch_geometry(self):
        self.manifest["storage"].append(dict(
            index=1, name="T", role="temporary", dtype="bf16", shape=[8],
            bytes=16, realization="numpy-host"))
        region = self.manifest["regions"][0]
        region["device"] = "npu"
        region["npu_abi"] = [region["abi"][0], dict(
            argument=1, storage=1, dtype="bf16", shape=[8])]
        EXECUTOR.validate_manifest(self.save())
        region["npu_abi"][1]["shape"] = [4]
        with self.assertRaises(ValueError):
            EXECUTOR.validate_manifest(self.save())

    def test_order_only_flags_are_typed_readonly_contracts(self):
        binding = self.manifest["regions"][0]["abi"][0]
        binding.update(role="input", order_only="false")
        with self.assertRaises(ValueError):
            EXECUTOR.validate_manifest(self.save())
        binding.update(role="output", order_only=True)
        with self.assertRaises(ValueError):
            EXECUTOR.validate_manifest(self.save())

    def test_compiled_foreign_runtime_is_rejected_independently_of_manifest(self):
        # An empty CPU-only program avoids loading a library, and isolates the
        # compiled receipt check from region coverage or source checks.
        self.manifest["regions"] = []
        compiled = self.compiled()
        compiled["runtime_identity"] = "foreign-runtime"
        path = self.root / "compiled.json"
        path.write_text(json.dumps(compiled))
        with self.assertRaises(ValueError):
            EXECUTOR.Prepared(path)

    @unittest.skipUnless(shutil.which("cc"), "CPU C compiler needed for ABI integration")
    def test_real_cpu_library_uses_bound_descriptors_and_returns_detached_output(self):
        # Exercise Prepared's real ctypes memref ABI without importing XRT.
        # This tiny synthetic library is not a source correctness oracle.
        self.manifest["storage"].append(dict(
            index=1, name="C", role="output", dtype="f32", shape=[4],
            bytes=16, realization="numpy-host"))
        self.manifest["regions"][0]["abi"].append(dict(
            argument=1, storage=1, dtype="f32", shape=[4]))
        source = self.root / "source.mlir"
        source.write_text("synthetic source identity, no compiler claim\n")
        csource = self.root / "fixture.c"
        csource.write_text("""
#include <stdint.h>
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} MemRef;
void _mlir_ciface_test(MemRef *a, MemRef *c) {
  for (int64_t i = 0; i < a->sizes[0]; ++i)
    c->aligned[c->offset + i * c->strides[0]] =
      2 * a->aligned[a->offset + i * a->strides[0]];
}
""")
        library = self.root / "fixture.so"
        subprocess.run([shutil.which("cc"), "-shared", "-fPIC", str(csource),
                        "-o", str(library)], check=True, capture_output=True,
                       timeout=30)
        compiled = self.compiled()
        compiled["regions"] = [dict(index=0, device="cpu", child_graph_id="child",
                                    source_sha256=EXECUTOR.sha(source),
                                    library=str(library),
                                    library_sha256=EXECUTOR.sha(library))]
        path = self.root / "compiled.json"
        path.write_text(json.dumps(compiled))
        prepared = EXECUTOR.Prepared(path)
        try:
            source_array = np.arange(4, dtype=np.float32)
            outputs, sample = prepared.one(dict(A=source_array))
            np.testing.assert_array_equal(outputs["C"], source_array * 2)
            self.assertEqual(len(sample["regions"]), 1)
            self.assertEqual(prepared.runners, [])
            self.assertFalse(hasattr(prepared, "xrt"))
            prepared.arrays[1].fill(-9)
            np.testing.assert_array_equal(outputs["C"], source_array * 2)
        finally:
            prepared.close()

    def test_truncated_compiled_region_list_is_rejected_before_preparation(self):
        # Missing a call changes the program, even when source/manifest hashes
        # match. The constructor must not silently accept zip's shorter input.
        path = self.root / "compiled.json"
        path.write_text(json.dumps(self.compiled()))
        with self.assertRaises(ValueError):
            EXECUTOR.Prepared(path)


class ConfigurationReceiptContracts(unittest.TestCase):
    def setUp(self):
        import hashlib
        self.directory = tempfile.TemporaryDirectory(prefix="configuration-receipt-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.air = self.root / "parent.air.mlir"
        self.air.write_text("synthetic parent, never compiled")
        abi = dict(buffers=[dict(name="A", role="input", dtype="f32", shape=[4])])
        self.physical = dict(graph_id="child", abi=abi, abi_storage_indices=[0],
                             physical_invocation_count=2, configurations=[],
                             configuration_coverage=dict(program_graph_id="child", physical_invocation_count=2,
                                 local_copies_crossing_boundaries=0, additional_host_transfer_bytes=0,
                                 runtime_requires_serial_wait2=True, covered_event_count=2,
                                 configurations=[]))
        self.artifact = dict(configurations=[])
        self.region = dict(child_graph_id="child", air=str(self.air), physical_invocation_count=2)
        for ordinal in range(2):
            air = self.root / ("configuration-" + str(ordinal) + ".mlir")
            air.write_text("synthetic selected AIR " + str(ordinal))
            contract = dict(program_graph_id="child", ordinal=ordinal)
            EXECUTOR.save(str(air) + ".runtime-contract.json", contract)
            digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                               ensure_ascii=False).encode()).hexdigest()
            selection = dict(air=air.name, begin_event=ordinal, end_event_exclusive=ordinal + 1,
                             runtime_contract_sha256=digest)
            self.physical["configurations"].append(selection)
            self.physical["configuration_coverage"]["configurations"].append(dict(
                index=ordinal, begin_event=ordinal, end_event_exclusive=ordinal + 1))
            physical = dict(graph_id="child", abi=abi, abi_storage_indices=[0],
                            selection=dict(begin_event=ordinal, end_event_exclusive=ordinal + 1),
                            runtime_contract_sha256=digest)
            EXECUTOR.save(str(air) + ".physical.json", physical)
            elf = self.root / ("configuration-" + str(ordinal) + ".elf")
            elf.write_text("synthetic ELF identity " + str(ordinal))
            proof = self.root / ("proof-" + str(ordinal) + ".json")
            EXECUTOR.save(proof, dict(program_graph_id="child", runtime_contract_sha256=digest,
                                      generic_module_sha256="synthetic-realized-" + str(ordinal)))
            self.artifact["configurations"].append(dict(
                elf=str(elf), elf_sha256=EXECUTOR.sha(elf), air_sha256=EXECUTOR.sha(air),
                physical_receipt_sha256=EXECUTOR.sha(str(air) + ".physical.json"),
                runtime_contract_sha256=digest, runtime_proof=str(proof), runtime_proof_sha256=EXECUTOR.sha(proof),
                realized_runtime_sha256="synthetic-realized-" + str(ordinal)))
        self.seal_parent()

    def seal_parent(self):
        EXECUTOR.save(str(self.air) + ".physical.json", self.physical)
        self.artifact["physical_receipt_sha256"] = EXECUTOR.sha(str(self.air) + ".physical.json")

    def validate(self):
        return EXECUTOR.validate_npu_configurations(self.root, self.region, self.artifact)

    def test_exact_ordered_configuration_sequence_validates_without_device(self):
        self.assertEqual(len(self.validate()), 2)

    def test_shortened_or_reordered_compiled_sequence_is_rejected(self):
        self.artifact["configurations"].reverse()
        with self.assertRaisesRegex(ValueError, "order/source/ABI"):
            self.validate()
        self.artifact["configurations"].reverse()
        self.artifact["configurations"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.validate()

    def test_physical_coverage_gap_is_rejected_even_with_matching_parent_hash(self):
        self.physical["configurations"][1]["begin_event"] = 2
        self.seal_parent()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.validate()

    def test_changed_elf_or_proof_is_rejected(self):
        Path(self.artifact["configurations"][1]["elf"]).write_text("changed ELF")
        with self.assertRaisesRegex(ValueError, "ELF/runtime proof"):
            self.validate()


class InvocationOwnership(unittest.TestCase):
    def prepared(self, fail_wait=False, with_order_only=False):
        prepared = EXECUTOR.Prepared.__new__(EXECUTOR.Prepared)
        prepared.np = np
        prepared.manifest = dict(storage=[
            dict(index=0, name="A", role="input", dtype="f32"),
            dict(index=1, name="T", role="temporary", dtype="f32"),
            dict(index=2, name="C", role="output", dtype="f32")])
        prepared.arrays = {index: np.empty(4, np.float32) for index in range(3)}
        events = []

        class BO:
            def __init__(self, name):
                self.name = name

            def sync(self, direction):
                events.append(("sync", self.name, direction))

        a, temporary, c = [BO(name) for name in ("A", "T", "C")]

        class Run:
            def __init__(self, kernel):
                self.kernel = kernel
                events.append(("construct", kernel))

            def set_arg(self, ordinal, bo):
                events.append(("bind", self.kernel, ordinal, bo.name))

            def start(self):
                events.append(("start", self.kernel))

            def wait2(self):
                events.append(("wait2", self.kernel))
                if fail_wait:
                    raise RuntimeError("synthetic native completion failure")
                if self.kernel == "first":
                    np.add(prepared.arrays[0], 1, out=prepared.arrays[1])
                else:
                    np.add(prepared.arrays[1], 3, out=prepared.arrays[2])

        prepared.xrt = types.SimpleNamespace(
            run=Run, xclBOSyncDirection=types.SimpleNamespace(
                XCL_BO_SYNC_BO_TO_DEVICE="to", XCL_BO_SYNC_BO_FROM_DEVICE="from"))

        def cpu_body():
            events.append(("cpu",))
            prepared.arrays[1] *= 2

        # Output ownership is per region ABI, even for program-private T.
        first_buffers = [a, temporary]
        first_abi = [dict(argument=0, storage=0, role="input", order_only=False),
                     dict(argument=1, storage=1, role="output", order_only=False)]
        last_abi = [dict(argument=0, storage=1, role="input", order_only=False),
                    dict(argument=1, storage=2, role="output", order_only=False)]
        if with_order_only:
            prepared.manifest["storage"].append(dict(
                index=3, name="O", role="temporary", dtype="f32"))
            prepared.arrays[3] = np.empty(4, np.float32)
            first_buffers.append(BO("O"))
            first_abi.append(dict(argument=2, storage=3, role="input", order_only=True))
        prepared.steps = [
            (dict(index=0, device="npu", abi=first_abi, npu_abi=first_abi),
             (["first"], first_buffers, [a, temporary], [temporary])),
            (dict(index=1, device="cpu"), cpu_body),
            (dict(index=2, device="npu", abi=last_abi, npu_abi=last_abi),
             (["last"], [temporary, c], [temporary, c], [c]))]
        return prepared, events

    def test_exact_serial_calls_syncs_private_intermediate_and_detached_output(self):
        prepared, events = self.prepared()
        inputs = dict(A=np.arange(4, dtype=np.float32))
        outputs, sample = prepared.one(inputs)
        np.testing.assert_array_equal(outputs["C"], (inputs["A"] + 1) * 2 + 3)
        self.assertEqual(events, [
            ("sync", "A", "to"), ("sync", "T", "to"),
            ("construct", "first"), ("bind", "first", 0, "A"),
            ("bind", "first", 1, "T"), ("start", "first"),
            ("wait2", "first"), ("sync", "T", "from"), ("cpu",),
            ("sync", "T", "to"), ("sync", "C", "to"),
            ("construct", "last"), ("bind", "last", 0, "T"),
            ("bind", "last", 1, "C"), ("start", "last"),
            ("wait2", "last"), ("sync", "C", "from")])
        self.assertEqual([r["region"] for r in sample["regions"]], [0, 1, 2])
        owned = sum(v for row in sample["regions"] for key, v in row.items()
                    if key.endswith("_ns"))
        self.assertEqual(sample["wall_ns"], owned + sample["reset_ns"] +
                         sample["control_ns"] + sample["copy_ns"])
        self.assertGreaterEqual(sample["control_ns"], 0)
        prepared.arrays[2].fill(-99)
        self.assertTrue(np.all(outputs["C"] >= 5))
        outputs2, _ = prepared.one(dict(A=inputs["A"] + 10))
        np.testing.assert_array_equal(outputs2["C"], (inputs["A"] + 11) * 2 + 3)

    def test_order_only_physical_argument_is_bound_without_payload_sync(self):
        prepared, events = self.prepared(with_order_only=True)
        inputs = dict(A=np.arange(4, dtype=np.float32))
        outputs, _ = prepared.one(inputs)
        np.testing.assert_array_equal(outputs["C"], (inputs["A"] + 1) * 2 + 3)
        self.assertIn(("bind", "first", 2, "O"), events)
        self.assertFalse(any(event[0] == "sync" and event[1] == "O"
                             for event in events))

    def multi_configuration(self, fail_second=False):
        prepared = EXECUTOR.Prepared.__new__(EXECUTOR.Prepared)
        prepared.np = np
        prepared.manifest = dict(storage=[
            dict(index=0, name="A", role="input", dtype="f32"),
            dict(index=1, name="T", role="temporary", dtype="f32"),
            dict(index=2, name="C", role="output", dtype="f32")])
        prepared.arrays = {i: np.empty(4, np.float32) for i in range(3)}
        device = {i: np.empty(4, np.float32) for i in range(3)}
        events = []

        class BO:
            def __init__(self, index):
                self.index = index

            def sync(self, direction):
                events.append(("sync", self.index, direction))
                source, target = ((prepared.arrays, device) if direction == "to"
                                  else (device, prepared.arrays))
                np.copyto(target[self.index], source[self.index])

        class Run:
            def __init__(self, kernel):
                self.kernel = kernel
                events.append(("construct", kernel))

            def set_arg(self, ordinal, bo):
                events.append(("bind", self.kernel, ordinal, bo.index))

            def start(self):
                events.append(("start", self.kernel))

            def wait2(self):
                events.append(("wait2", self.kernel))
                if self.kernel == "produce":
                    np.add(device[0], 7, out=device[1])
                elif self.kernel == "consume":
                    if fail_second:
                        raise RuntimeError("second configuration failed")
                    # Host T is still poisoned. An intermediate TO sync would
                    # overwrite device T and make this full output nonfinite.
                    np.multiply(device[1], 3, out=device[2])
                else:
                    raise AssertionError("later configuration must not run")

        prepared.xrt = types.SimpleNamespace(run=Run, xclBOSyncDirection=types.SimpleNamespace(
            XCL_BO_SYNC_BO_TO_DEVICE="to", XCL_BO_SYNC_BO_FROM_DEVICE="from"))
        buffers = [BO(i) for i in range(3)]
        kernels = ["produce", "consume", "forbidden"] if fail_second else ["produce", "consume"]
        prepared.steps = [(dict(index=0, device="npu"),
                           (kernels, buffers, buffers, [buffers[2]]))]
        return prepared, events

    def test_multi_configuration_preserves_device_scratch_and_exclusive_wall(self):
        prepared, events = self.multi_configuration()
        source = np.arange(4, dtype=np.float32)
        outputs, sample = prepared.one(dict(A=source))
        np.testing.assert_array_equal(outputs["C"], (source + 7) * 3)
        self.assertTrue(np.isnan(prepared.arrays[1]).all())
        waits = [i for i, event in enumerate(events) if event[0] == "wait2"]
        self.assertEqual(len(waits), 2)
        self.assertFalse(any(e[0] == "sync" for e in events[waits[0] + 1:waits[1]]))
        self.assertEqual(sum(e[0] == "sync" and e[-1] == "to" for e in events), 3)
        self.assertEqual(events[-1], ("sync", 2, "from"))
        row = sample["regions"][0]
        self.assertEqual(row["configuration_count"], 2)
        self.assertEqual(sample["wall_ns"], sample["reset_ns"] + sample["control_ns"] +
                         sample["copy_ns"] + sum(v for k, v in row.items() if k.endswith("_ns")))

    def test_multi_configuration_failure_stops_before_later_call_and_readback(self):
        prepared, events = self.multi_configuration(fail_second=True)
        with self.assertRaisesRegex(RuntimeError, "second configuration failed"):
            prepared.one(dict(A=np.arange(4, dtype=np.float32)))
        self.assertEqual(events[-1], ("wait2", "consume"))
        self.assertFalse(any(event[0] == "construct" and event[1] == "forbidden" for event in events))
        self.assertFalse(any(event[0] == "sync" and event[-1] == "from" for event in events))

    def test_input_validation_precedes_every_region_call(self):
        prepared, events = self.prepared()
        with self.assertRaises(ValueError):
            prepared.one(dict(A=np.arange(4, dtype=np.float64)))
        self.assertEqual(events, [])

    def test_native_wait_failure_propagates_without_polling_or_readback(self):
        prepared, events = self.prepared(fail_wait=True)
        with self.assertRaisesRegex(RuntimeError, "completion failure"):
            prepared.one(dict(A=np.arange(4, dtype=np.float32)))
        self.assertEqual(events[-1], ("wait2", "first"))
        self.assertFalse(any(event[0] == "cpu" or event[-1] == "from"
                             for event in events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
