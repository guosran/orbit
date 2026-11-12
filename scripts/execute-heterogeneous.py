#!/usr/bin/env python3
"""Compile and measure compiler-selected typed CPU/NPU regions.

This runner does not partition, plan, rank, or implement source arithmetic.
CPU arithmetic comes from the emitted source body; NPU calls use the emitted
AIR and exact generated external kernel objects. Measurements use native
wait2, with no polling fallback or installed-runtime monkeypatches.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


RUNTIME = "orbit.serial-prepared-typed-native-wait2.v32"
SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(base, name):
    path = Path(name)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def command(argv, directory, name, timeout=300):
    argv = list(map(str, argv))
    save(directory / (name + ".command.json"), argv)
    begin = time.perf_counter_ns()
    with (directory / (name + ".log")).open("w") as log:
        result = subprocess.run(argv, cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    save(directory / (name + ".result.json"), dict(
        returncode=result.returncode, wall_ns=time.perf_counter_ns() - begin))
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {directory / (name + '.log')}")


def validate_manifest(path):
    path = Path(path).resolve()
    manifest = read(path)
    if manifest.get("schema") != "orbit.heterogeneous.static.v32":
        raise ValueError("expected compiler-issued typed v32 manifest")
    contract = manifest.get("execution_contract", {})
    if (contract.get("runtime_identity") != RUNTIME or
            contract.get("executor_sha256") != SOURCE_SHA or
            manifest.get("execution_contract_identity") in (None, "unbound")):
        raise ValueError("runtime source/contract differs from emitted manifest")
    for role, entry in contract.get("files", {}).items():
        if not Path(entry["path"]).is_absolute() or sha(entry["path"]) != entry["sha256"]:
            raise ValueError("execution-contract file changed: " + role)
    if not all(role in contract.get("files", {})
               for role in ("clang", "mlir-opt", "mlir-translate", "aircc", "peano")):
        raise ValueError("compiler identities missing from execution contract")
    storage = manifest["storage"]
    if [s["index"] for s in storage] != list(range(len(storage))):
        raise ValueError("storage indices must be dense")
    for item in storage:
        if item["dtype"] not in ("bf16", "f32"):
            raise ValueError("unsupported typed ABI")
        size = 2 if item["dtype"] == "bf16" else 4
        for extent in item["shape"]:
            if type(extent) is not int or extent <= 0:
                raise ValueError("invalid storage geometry")
            size *= extent
        if size != item["bytes"]:
            raise ValueError("typed byte size mismatch")
    for index, region in enumerate(manifest["regions"]):
        if region["index"] != index or region["device"] not in ("cpu", "npu"):
            raise ValueError("invalid region order or device")
        for argument, binding in enumerate(region.get("npu_abi", region["abi"])):
            if (type(binding.get("storage")) is not int or
                    not 0 <= binding["storage"] < len(storage)):
                raise ValueError("region storage index is outside declared backing")
            if (type(binding.get("order_only", False)) is not bool or
                    (binding.get("order_only", False) and binding.get("role") != "input")):
                raise ValueError("order_only requires a boolean on an input ABI binding")
            item = storage[binding["storage"]]
            if (binding["argument"] != argument or binding["shape"] != item["shape"] or
                    binding["dtype"] != item["dtype"]):
                raise ValueError("region ABI differs from declared backing")
    limit = contract.get("max_prepared_npu_contexts")
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("max_prepared_npu_contexts must be a positive integer")
        # Every selected configuration stays resident in this prepared runtime.
        # Count across regions: checking each region alone misses shared device
        # exhaustion. A streaming-context executor would need its own contract
        # and measured setup/teardown costs, not a silent fallback here.
        count = 0
        for region in manifest["regions"]:
            if region["device"] == "npu":
                invocations = region.get("physical_invocation_count", 1)
                if type(invocations) is not int or invocations <= 0:
                    raise ValueError("invalid physical invocation count")
                count += invocations
        if count > limit:
            raise ValueError("selected program exceeds prepared NPU context capacity")
    return path, manifest


def compile_npu_configuration(air, region, directory, tools):
    from aie.utils.compile.utils import cxx_core_compile_command
    physical = read(str(air) + ".physical.json")
    if physical["graph_id"] != region["child_graph_id"]:
        raise ValueError("AIR physical receipt differs from selected child graph")
    objects = []
    for kernel in physical.get("matrix_realization", {}).get("kernels", []):
        cpp = air.parent / (kernel["symbol"] + ".cc")
        object_file = directory / kernel["object"]
        args = cxx_core_compile_command(str(cpp), "aie2p", str(object_file))
        command(args, directory, kernel["symbol"])
        objects.append(dict(source=str(cpp), source_sha256=sha(cpp),
                            object=str(object_file), object_sha256=sha(object_file)))
    local_air = directory / "program.air.mlir"
    shutil.copy2(air, local_air)
    elf = directory / "program.elf"
    required = ("taskflow-opt", "air-opt", "aie-opt", "aiecc")
    if any(role not in tools for role in required):
        raise ValueError("selected runtime realization compiler identities are missing")
    runtime_contract = Path(str(air) + ".runtime-contract.json")
    contract_value = read(runtime_contract)
    # The C++ emitter binds its checked source/selected graph to this
    # canonical integer/string-only contract; no plan is reconstructed.
    contract_hash = hashlib.sha256(json.dumps(contract_value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if (contract_hash != physical.get("runtime_contract_sha256") or
            contract_value.get("program_graph_id") != region["child_graph_id"]):
        raise ValueError("runtime transfer contract differs from selected physical graph")
    # AIR's ELF mode materializes the explicit configuration wrapper.
    # This intermediate ELF is never submitted; only the subsequently
    # checked and realized runtime is compiled into program.elf.
    command([tools["aircc"], "--device", "npu2", "--no-xchesscc", "--no-xbridge",
             "--output-format", "elf", "--elf-name", directory / "unrealized.elf",
             "--peano=" + str(Path(tools["peano"]).parents[1]),
             "--stack-size=4096", "--omit-ping-pong-transform=all", local_air],
            directory, "aircc-extract")
    extracted = directory / "air_project/npu.program.air.mlir"
    generic = directory / "extracted.generic.mlir"
    command([tools["air-opt"], extracted, "--mlir-print-op-generic", "-o", generic],
            directory, "normalize-extracted")
    realized = directory / "runtime.generic.mlir"
    proof_path = directory / "runtime-proof.json"
    command([tools["taskflow-opt"], generic,
             "--realize-program-runtime=" + str(runtime_contract),
             "--air-output=" + str(realized),
             "--implementation-output=" + str(proof_path)], directory, "realize-runtime")
    proof = read(proof_path)
    if (proof.get("program_graph_id") != region["child_graph_id"] or
            proof.get("runtime_contract_sha256") != contract_hash or
            proof.get("generic_module_sha256") != sha(realized)):
        raise ValueError("realized runtime proof differs from selected transfer contract")
    canonical = directory / "runtime.mlir"
    command([tools["aie-opt"], realized, "-o", canonical], directory, "verify-runtime")
    command([tools["aiecc"], "--tmpdir=" + str(directory / "aie_project"),
             "--get-full-elf", "--expand-load-pdis", "--full-elf-name=" + str(elf),
             "--get-scratchpad-parameters",
             "--peano=" + str(Path(tools["peano"]).parents[1]),
             "-O", "3", "-j", "0", "--unified", canonical], directory, "aiecc")
    hardware = read(directory / "aie_project/full_elf_config.json")
    kernel = next(k for k in hardware["xrt-kernels"] if k["name"] == "main")
    result = dict(elf=str(elf), elf_sha256=sha(elf),
                kernel="main:" + str(kernel["instance"][0]["id"]),
                air_sha256=sha(air), physical_receipt_sha256=sha(str(air) + ".physical.json"),
                runtime_contract_sha256=contract_hash,
                runtime_proof=str(proof_path), runtime_proof_sha256=sha(proof_path),
                realized_runtime_sha256=sha(realized), objects=objects)
    return result


def compile_manifest(path, output):
    path, manifest = validate_manifest(path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = manifest["execution_contract"]["files"]
    tools = {role: entry["path"] for role, entry in files.items()}
    base = path.parent
    compiled = []
    for region in manifest["regions"]:
        directory = output / ("region-" + str(region["index"]))
        directory.mkdir()
        source = resolve(base, region["source"])
        item = dict(index=region["index"], device=region["device"],
                    source_sha256=sha(source), child_graph_id=region["child_graph_id"])
        if region["device"] == "cpu":
            cpu = resolve(base, region["cpu_source"])
            command([tools["mlir-opt"], cpu, "--lower-affine", "--convert-scf-to-cf",
                     "--convert-math-to-llvm", "--convert-arith-to-llvm",
                     "--finalize-memref-to-llvm", "--convert-func-to-llvm",
                     "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
                     "-o", directory / "llvm.mlir"], directory, "lower")
            command([tools["mlir-translate"], "--mlir-to-llvmir", directory / "llvm.mlir",
                     "-o", directory / "cpu.ll"], directory, "translate")
            library = directory / "cpu.so"
            command([tools["clang"], "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                     directory / "cpu.ll", "-lm", "-lgcc_s", "-o", library],
                    directory, "cpu-compile")
            item.update(library=str(library), library_sha256=sha(library),
                        cpu_source_sha256=sha(cpu))
        else:
            air = resolve(base, region["air"])
            physical = read(str(air) + ".physical.json")
            configurations = physical.get("configurations")
            if configurations is None:
                item.update(compile_npu_configuration(air, region, directory, tools))
            else:
                # Follow the compiler-proved exact event partition in order.
                if (not configurations or physical.get("graph_id") != region["child_graph_id"] or
                        physical.get("physical_invocation_count") != len(configurations)):
                    raise ValueError("invalid selected configuration sequence")
                artifacts = []
                for index, configuration in enumerate(configurations):
                    child_directory = directory / ("configuration-" + str(index))
                    child_directory.mkdir()
                    child_air = resolve(air.parent, configuration["air"])
                    artifacts.append(compile_npu_configuration(child_air, region, child_directory, tools))
                item.update(configurations=artifacts, physical_receipt_sha256=sha(str(air) + ".physical.json"))
        compiled.append(item)
    result = dict(schema="orbit.heterogeneous.compiled.v32", manifest=str(path),
                  manifest_sha256=sha(path), plan_id=manifest["id"], runtime_identity=RUNTIME,
                  executor_sha256=SOURCE_SHA, regions=compiled)
    save(output / "compiled.json", result)
    return output / "compiled.json"


def memref_type(rank):
    class Descriptor(ctypes.Structure):
        _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64), ("sizes", ctypes.c_int64 * rank),
                    ("strides", ctypes.c_int64 * rank)]
    return Descriptor


def validate_npu_configurations(base, region, artifact):
    """Bind every submitted ELF to its ordered, checked physical selection.

    This runs before opening a device. A shortened compiled list must not turn
    a complete regional program into a silently truncated one.
    """
    air = resolve(base, region["air"])
    physical_path = Path(str(air) + ".physical.json")
    physical = read(physical_path)
    graph = region["child_graph_id"]
    if (sha(physical_path) != artifact.get("physical_receipt_sha256") or
            physical.get("graph_id") != graph):
        raise ValueError("compiled physical sequence differs from selected graph")
    expected = physical.get("configurations")
    actual = artifact.get("configurations")
    if expected is None:
        if actual is not None or physical.get("physical_invocation_count", 1) != 1:
            raise ValueError("single configuration receipt has a foreign sequence")
        expected, actual = [dict(air=air)], [artifact]
    elif (type(expected) is not list or not expected or
          type(actual) is not list or len(actual) != len(expected) or
          type(physical.get("physical_invocation_count")) is not int or
          physical["physical_invocation_count"] != len(expected)):
        raise ValueError("compiled configuration coverage differs from physical sequence")
    if region.get("physical_invocation_count", len(expected)) != len(expected):
        raise ValueError("manifest configuration count differs from physical sequence")
    coverage = physical.get("configuration_coverage")
    if len(expected) > 1:
        if (not coverage or coverage.get("program_graph_id") != graph or
                coverage.get("physical_invocation_count") != len(expected) or
                coverage.get("local_copies_crossing_boundaries") != 0 or
                coverage.get("additional_host_transfer_bytes") != 0 or
                coverage.get("runtime_requires_serial_wait2") is not True or
                len(coverage.get("configurations", [])) != len(expected)):
            raise ValueError("physical sequence lacks a complete serial coverage proof")
    cursor = 0
    for ordinal, (selection, compiled) in enumerate(zip(expected, actual)):
        selected_air = resolve(air.parent, selection["air"])
        selected_physical_path = Path(str(selected_air) + ".physical.json")
        selected_physical = read(selected_physical_path)
        if (sha(selected_air) != compiled.get("air_sha256") or
                sha(selected_physical_path) != compiled.get("physical_receipt_sha256") or
                selected_physical.get("graph_id") != graph or
                selected_physical.get("abi_storage_indices") != physical.get("abi_storage_indices") or
                selected_physical.get("abi") != physical.get("abi")):
            raise ValueError("configuration order/source/ABI differs from physical selection")
        if len(expected) > 1:
            begin, end = selection.get("begin_event"), selection.get("end_event_exclusive")
            interval = selected_physical.get("selection", {})
            proved = coverage["configurations"][ordinal]
            if (type(begin) is not int or type(end) is not int or begin != cursor or end <= begin or
                    interval.get("begin_event") != begin or interval.get("end_event_exclusive") != end or
                    proved.get("index") != ordinal or proved.get("begin_event") != begin or
                    proved.get("end_event_exclusive") != end):
                raise ValueError("physical configurations are missing, reordered or overlapping")
            cursor = end
        contract = read(str(selected_air) + ".runtime-contract.json")
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        proof = read(compiled["runtime_proof"])
        if (contract.get("program_graph_id") != graph or
                digest != selected_physical.get("runtime_contract_sha256") or
                digest != compiled.get("runtime_contract_sha256") or
                (len(expected) > 1 and digest != selection.get("runtime_contract_sha256")) or
                sha(compiled["runtime_proof"]) != compiled.get("runtime_proof_sha256") or
                proof.get("program_graph_id") != graph or
                proof.get("runtime_contract_sha256") != digest or
                proof.get("generic_module_sha256") != compiled.get("realized_runtime_sha256") or
                sha(compiled["elf"]) != compiled.get("elf_sha256")):
            raise ValueError("configuration ELF/runtime proof differs from selected contract")
    if len(expected) > 1 and cursor != coverage.get("covered_event_count"):
        raise ValueError("physical configuration sequence omits selected events")
    return actual


class Prepared:
    def __init__(self, compiled):
        import numpy as np
        from ml_dtypes import bfloat16
        self.np = np
        self.path = Path(compiled).resolve()
        self.compiled = read(self.path)
        self.manifest_path, self.manifest = validate_manifest(self.compiled["manifest"])
        if (self.compiled.get("schema") != "orbit.heterogeneous.compiled.v32" or
                self.compiled.get("runtime_identity") != RUNTIME or
                self.compiled["executor_sha256"] != SOURCE_SHA or
                sha(self.manifest_path) != self.compiled["manifest_sha256"] or
                self.manifest["id"] != self.compiled["plan_id"]):
            raise ValueError("compiled artifact differs from selected manifest/runtime")
        if len(self.compiled["regions"]) != len(self.manifest["regions"]):
            raise ValueError("compiled region coverage differs from selected manifest")
        for region, artifact in zip(self.manifest["regions"], self.compiled["regions"]):
            if region["device"] == "npu":
                validate_npu_configurations(self.manifest_path.parent, region, artifact)
        self.arrays, self.bos, self.steps, self.runners, self.owners = {}, {}, [], [], []
        used = {b["storage"] for region in self.manifest["regions"] for b in region.get("npu_abi", region["abi"])}
        if any(self.manifest["storage"][index]["realization"] == "xrt-mapped-bo" for index in used):
            import pyxrt
            self.xrt = pyxrt
            self.device = pyxrt.device(0)
            self.owners.append(self.device)
        for storage in self.manifest["storage"]:
            if storage["index"] not in used:
                continue
            dtype = bfloat16 if storage["dtype"] == "bf16" else np.float32
            if storage["realization"] == "xrt-mapped-bo":
                bo = self.xrt.ext.bo(self.device, storage["bytes"])
                mapped = bo.map()
                array = np.frombuffer(mapped, dtype=dtype).reshape(storage["shape"])
                self.bos[storage["index"]] = bo
                self.owners.extend([bo, mapped])
            else:
                array = np.empty(storage["shape"], dtype=dtype)
            self.arrays[storage["index"]] = array
        for region, artifact in zip(self.manifest["regions"], self.compiled["regions"]):
            if (region["index"] != artifact["index"] or region["device"] != artifact["device"] or
                    region["child_graph_id"] != artifact["child_graph_id"] or
                    sha(resolve(self.manifest_path.parent, region["source"])) != artifact["source_sha256"]):
                raise ValueError("compiled source/graph binding changed")
            if region["device"] == "cpu":
                if sha(artifact["library"]) != artifact["library_sha256"]:
                    raise ValueError("CPU library changed")
                library = ctypes.CDLL(artifact["library"])
                function = getattr(library, "_mlir_ciface_" + region["function"])
                descriptors, kinds = [], []
                for binding in region["abi"]:
                    array = self.arrays[binding["storage"]]
                    kind = memref_type(array.ndim)
                    kinds.append(kind)
                    descriptors.append(kind(array.ctypes.data, array.ctypes.data, 0,
                        (ctypes.c_int64 * array.ndim)(*array.shape),
                        (ctypes.c_int64 * array.ndim)(*(s // array.itemsize for s in array.strides))))
                function.argtypes = [ctypes.POINTER(kind) for kind in kinds]
                function.restype = None
                arguments = [ctypes.byref(d) for d in descriptors]
                self.owners.extend([library, descriptors])
                self.steps.append((region, lambda function=function, arguments=arguments: function(*arguments)))
            else:
                from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner
                kernels = []
                for configuration in artifact.get("configurations", [artifact]):
                    if sha(configuration["elf"]) != configuration["elf_sha256"]:
                        raise ValueError("NPU ELF changed")
                    runner = MultiLaunchRunner(configuration["elf"], configuration["kernel"])
                    self.runners.append(runner)
                    kernels.append(runner.kernel)
                if not kernels:
                    raise ValueError("empty NPU configuration sequence")
                actual_abi = region.get("npu_abi", region["abi"])
                buffers = [self.bos[b["storage"]] for b in actual_abi]
                outputs = [bo for bo, b in zip(buffers, actual_abi) if b["role"] == "output"]
                synced = [bo for bo, b in zip(buffers, actual_abi) if not b.get("order_only", False)]
                self.steps.append((region, (kernels, buffers, synced, outputs)))
        self.pointers = {str(index): dict(address=int(array.ctypes.data), bytes=array.nbytes)
                         for index, array in self.arrays.items()}

    def one(self, inputs):
        np = self.np
        for index, array in self.arrays.items():
            if self.manifest["storage"][index]["role"] != "input":
                array.fill(np.nan)
        # All shape/type checks and poisoning precede the measured boundary.
        bindings = []
        for storage in self.manifest["storage"]:
            if storage["role"] != "input":
                continue
            array = self.arrays[storage["index"]]
            source = inputs[storage["name"]]
            if storage["dtype"] == "bf16":
                if source.dtype != np.uint16:
                    raise ValueError("BF16 archives must contain explicit uint16 payload bits")
                source = source.view(array.dtype)
            if source.shape != array.shape or source.dtype != array.dtype:
                raise ValueError("input differs from typed ABI")
            if not np.isfinite(source.astype(np.float32)).all():
                raise ValueError("nonfinite input outside numerical contract")
            bindings.append((array, source))
        spans = []
        begin = time.perf_counter_ns()
        for array, source in bindings:
            np.copyto(array, source, casting="no")
        reset_end = time.perf_counter_ns()
        for region, step in self.steps:
            start = time.perf_counter_ns()
            if region["device"] == "cpu":
                step()
                finish = time.perf_counter_ns()
                spans.append(dict(region=region["index"], body_ns=finish - start))
            else:
                kernels, buffers, synced, outputs = step
                for bo in synced:
                    bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                visible = time.perf_counter_ns()
                submit_ns = wait_ns = 0
                # Device scratch survives between configurations. Syncing a
                # host BO inside this sequence would overwrite that state.
                for kernel in kernels:
                    launch_begin = time.perf_counter_ns()
                    run = self.xrt.run(kernel)
                    for index, bo in enumerate(buffers):
                        run.set_arg(index, bo)
                    run.start()
                    submitted = time.perf_counter_ns()
                    run.wait2()
                    complete = time.perf_counter_ns()
                    submit_ns += submitted - launch_begin
                    wait_ns += complete - submitted
                complete = time.perf_counter_ns()
                for bo in outputs:
                    bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                finish = time.perf_counter_ns()
                spans.append(dict(region=region["index"], sync_to_ns=visible - start,
                                  submit_ns=submit_ns, wait_ns=wait_ns,
                                  configuration_count=len(kernels),
                                  sync_from_ns=finish - complete))
        copy_begin = time.perf_counter_ns()
        outputs = {s["name"]: self.arrays[s["index"]].copy()
                   for s in self.manifest["storage"] if s["role"] == "output"}
        end = time.perf_counter_ns()
        owned = sum(sum(v for k, v in row.items() if k.endswith("_ns")) for row in spans)
        sample = dict(wall_ns=end - begin, reset_ns=reset_end - begin,
                      copy_ns=end - copy_begin, control_ns=(end - begin) - owned -
                      (reset_end - begin) - (end - copy_begin), regions=spans)
        return outputs, sample

    def close(self):
        for runner in self.runners:
            runner.unload()
        self.runners.clear()
        self.steps.clear()
        self.arrays.clear()
        self.bos.clear()
        self.owners.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.compile_only:
        if not args.manifest:
            parser.error("--compile-only requires --manifest")
        compile_manifest(args.manifest, args.output / "compiled")
        return
    if not args.compiled or not args.inputs or not args.reference:
        parser.error("measurement requires compiled artifacts, inputs and independent reference")
    import numpy as np
    item = Prepared(args.compiled)
    captured, samples = [], []
    inputs, expected = dict(np.load(args.inputs)), dict(np.load(args.reference))
    try:
        for repeat in range(args.warmups + args.repeats):
            outputs, sample = item.one(inputs)
            captured.append(outputs)
            samples.append(dict(repeat=repeat, warmup=repeat < args.warmups, **sample))
        checks = []
        for outputs in captured:
            if set(outputs) != set(expected):
                raise ValueError("reference ABI differs from compiler outputs")
            checks.append({name: bool(np.isfinite(value).all() and
                np.allclose(value, expected[name], atol=args.atol, rtol=args.rtol))
                for name, value in outputs.items()})
        np.savez(args.output / "outputs.npz", **{
            name: np.stack([output[name] for output in captured]) for name in captured[0]})
        save(args.output / "samples.json", samples)
        save(args.output / "identity.json", dict(runtime=RUNTIME, executor_sha256=SOURCE_SHA,
             compiled=str(args.compiled.resolve()), compiled_sha256=sha(args.compiled),
             plan_id=item.manifest["id"], pointers=item.pointers, native_wait2=True,
             boundary="prepared reset + exact regions (sync/submit/wait/readback) + output copy"))
        passed = all(all(row.values()) for row in checks)
        result = dict(status="passed" if passed else "numeric_failed", checks=checks,
                      calls=len(samples), median_ms=float(np.median([
                          sample["wall_ns"] for sample in samples if not sample["warmup"]])) / 1e6)
        save(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
        if not passed:
            raise SystemExit(1)
    finally:
        item.close()


if __name__ == "__main__":
    main()
