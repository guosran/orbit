#!/usr/bin/env python3
"""Build actual full-Qwen spatial, L1-retention and DMA-dataflow variants.

The vendor file is immutable. Only three explicitly recognized module-level
parameters change in an owned copy; every function/class body stays identical.
These variants require their own xclbin/context: unlike r17 wave splitting,
different core placements or L1 programs cannot share an unchanged CDO.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


PARAMETERS = {"ATTN_PCOL", "PROJ_RC_CACHE", "COALESCE"}
OBJECTS = ("proj_qmm.o", "rms_residual.o", "glu.o", "rope.o",
           "attn_qk.ll", "attn_kv.ll")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def transform(source, values):
    if set(values) != PARAMETERS:
        raise ValueError("Only the three registered decode parameters may change")
    if values["ATTN_PCOL"] not in (4, 5):
        raise ValueError("Attention needs one complete, available two-CU column")
    if any(type(values[k]) is not int or values[k] not in (0, 1)
           for k in ("PROJ_RC_CACHE", "COALESCE")):
        raise ValueError("Cache and coalescing choices must be binary integers")
    original = ast.parse(source)
    tree = ast.parse(source)
    changed = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in PARAMETERS:
            continue
        if target.id in changed:
            raise ValueError("Ambiguous duplicate source parameter")
        changed[target.id] = ast.dump(statement.value, include_attributes=False)
        statement.value = ast.Constant(values[target.id])
    if set(changed) != PARAMETERS:
        raise ValueError("Original builder no longer exposes the registered parameters")
    # Check the complete module after restoring only the allowed expressions.
    restored = ast.parse(ast.unparse(tree))
    originals = {s.targets[0].id: s.value for s in original.body
                 if isinstance(s, ast.Assign) and len(s.targets) == 1
                 and isinstance(s.targets[0], ast.Name)
                 and s.targets[0].id in PARAMETERS}
    for statement in restored.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in originals):
            statement.value = originals[statement.targets[0].id]
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError("A variant changed more than its registered parameters")
    return ast.unparse(tree) + "\n", changed


def rotate_columns(module, rotation):
    """Move the entire routed graph, preserving each herd's logical lane order.

    Rotations by two columns preserve the original two-column projection
    herds. Moving attention alone can create a packet/circuit port conflict;
    this bijection moves its producers, consumers and memtiles together.
    Actual routing still has to pass aircc and hardware validation.
    """
    from air.ir import IntegerAttr
    if rotation not in (0, 2, 4, 6):
        raise ValueError("Column rotation must preserve two-column herd blocks")
    changes = []

    def walk(operation):
        yield operation
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from walk(child.operation)

    with module.context:
        for operation in walk(module.operation):
            for key in ("air.memtile_col", "x_loc"):
                if key not in operation.attributes:
                    continue
                attribute = IntegerAttr(operation.attributes[key])
                old = attribute.value
                new = (old + rotation) % 8
                if not 0 <= old < 8:
                    raise ValueError("Source uses a column outside the NPU2 array")
                if key == "x_loc":
                    if operation.name != "air.herd":
                        raise ValueError("Unknown column-bearing operation")
                    sizes = operation.opview.sizes
                    extent = IntegerAttr(sizes[0].owner.attributes["value"]).value
                    if [new + i for i in range(extent)] != [
                            (old + i + rotation) % 8 for i in range(extent)]:
                        raise ValueError("Rotation splits a herd across the array boundary")
                operation.attributes[key] = IntegerAttr.get(attribute.type, new)
                changes.append(dict(operation=operation.name, attribute=key,
                                    before=old, after=new))
    return changes


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve(strict=True)
    before = record(source)
    parameters = dict(ATTN_PCOL=args.attention_column,
                      PROJ_RC_CACHE=args.reduction_cache,
                      COALESCE=args.coalesce)
    text, old_expressions = transform(source.read_text(), parameters)
    owned = output / "owned-builder.py"
    owned.write_text(text)
    environment = dict(DECODE_MODEL="qwen2.5-3b", UNIFIED="1", LM_HEAD="0",
                       NLAYERS="1", DECODE_GOLDEN="1",
                       DECODE_GOLDEN_L=str(args.length), DECODE_DYNSEQ="0",
                       DECODE_RT_ROUNDS="0", AIR_TARGET_DEVICE="npu2")
    # Clear unrelated debug/bisect options: these are source-bound complete
    # model variants, never a reduced decoder wearing the same model name.
    for key in ("DECODE_UNI_DEC", "UNI_LM", "DECODE_PAIR_ROWS"):
        os.environ.pop(key, None)
    os.environ.update(environment)
    sys.path.insert(0, str(source.parent))
    module = type(sys)("orbit_owned_decode_variant")
    # Preserve the original builder's relative imports/resource lookup while
    # compiling the actual saved owned text and recording both source hashes.
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(text, str(owned), "exec"), module.__dict__)
    if ((module.UNI_DEC, module.UNI_LM, module.W_DUAL_CHAN,
         module.MODEL_NAME) != (36, 25, 0, "qwen2.5-3b")):
        raise ValueError("Variant changed the complete model or packed-weight ABI")
    request = dict(schema="orbit.qwen-decode-variant-build.v1",
                   source=before, owned_source=record(owned),
                   producer=record(__file__), parameters=parameters,
                   column_rotation=args.column_rotation,
                   original_expressions=old_expressions, environment=environment,
                   length=args.length, layers=36, vocab_waves=25,
                   attention_cores=[[(column + args.column_rotation) % 8, row]
                                    for column, low, high in module.ATTN_CU_LOC
                                    for row in (low, high)],
                   l1_reduction_cache_bytes_per_projection_core=(
                       2 * module.RCACHE_LEN if args.reduction_cache else 0),
                   projection_cores=[[(c + args.column_rotation) % 8, y + 2] for c in module.PCOL
                                     for y in range(module.NCY)],
                   kernel_inputs={name: record(args.kernel_dir / name)
                                  for name in OBJECTS},
                   weight_layout=dict(dual_channel=module.W_DUAL_CHAN,
                                      vocab_i2=module.VOCAB_I2),
                   execution_contract="Own xclbin/context; original full-model BO ABI and numerical operations")
    request["variant_id"] = hashlib.sha256(canonical(dict(
        source=before, parameters=parameters, column_rotation=args.column_rotation,
        kernel_inputs=request["kernel_inputs"]))).hexdigest()
    save(output / "request.json", request)
    for name in OBJECTS:
        (output / name).symlink_to((args.kernel_dir / name).resolve(strict=True))
    os.chdir(output)
    ir = module.build_module()
    placement_changes = rotate_columns(ir, args.column_rotation)
    (output / "source.air.mlir").write_text(str(ir))
    save(output / "emitted.json", dict(request=request["variant_id"],
                                       placement_changes=placement_changes,
                                       source_air=record(output / "source.air.mlir")))
    if args.emit_only:
        return
    from air.backend.xrt import XRTBackend
    start = time.monotonic()
    backend = XRTBackend(omit_while_true_loop=False, output_format="xclbin",
                         kernel_name="MLIR_AIE", instance_name="",
                         stack_size=module.STACK_SIZE,
                         use_lock_race_condition_fix_v2=True,
                         coalesce_shim_dma=bool(module.COALESCE))
    artifact = backend.compile(ir, output_binary_name="decode", insts="decode.insts.bin")
    if record(source) != before:
        raise ValueError("Original vendor source changed during compilation")
    helper_spec = importlib.util.spec_from_file_location(
        "orbit_decode_regions_metadata", Path(__file__).with_name("decode-regions.py"))
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    save(output / "compiled.json", dict(
        schema="orbit.qwen-decode-variant-compiled.v1", variant_id=request["variant_id"],
        request=record(output / "request.json"), source=before,
        length=args.length, parameters=parameters,
        column_rotation=args.column_rotation,
        source_air=record(output / "source.air.mlir"),
        abi=helper.public_abi(str(ir)), binary=record(artifact.output_binary),
        instructions=record(artifact.insts),
        device_files=helper.device_files(output / "air_project"),
        compile_seconds=time.monotonic() - start))


def package(args):
    builds = [json.loads((p / "compiled.json").read_text()) for p in args.build]
    if len(builds) != 2 or len({b["length"] for b in builds}) != 2:
        raise ValueError("Instruction specialization needs two distinct context builds")
    first = builds[0]
    for b in builds:
        if any(b[key] != first[key] for key in
               ("variant_id", "source", "parameters", "column_rotation", "abi", "device_files")):
            raise ValueError("Template pair changes program, resources, ABI or CDO")
        if (b["length"] + 15) // 16 != (first["length"] + 15) // 16:
            raise ValueError("Template pair changes actual KV capacity")
        for key in ("binary", "instructions", "request", "source_air"):
            if record(b[key]["path"]) != b[key]:
                raise ValueError("Compiled artifact changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for b in builds:
        prefix = output / ("decode_L" + str(b["length"]))
        Path(str(prefix) + ".xclbin").symlink_to(b["binary"]["path"])
        Path(str(prefix) + ".insts.bin").symlink_to(b["instructions"]["path"])
    save(output / "variant.json", dict(
        schema="orbit.qwen-decode-variant-template.v1", variant_id=first["variant_id"],
        parameters=first["parameters"], abi=first["abi"],
        column_rotation=first["column_rotation"],
        capacity=16 * ((first["length"] + 15) // 16), layers=36, vocab_waves=25,
        device_files=first["device_files"],
        builds=[record(p / "compiled.json") for p in args.build],
        runtime="Original FusedDecoder over this variant directory, with own xclbin/context"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--kernel-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--attention-column", type=int, choices=(4, 5), default=4)
    p.add_argument("--reduction-cache", type=int, choices=(0, 1), default=1)
    p.add_argument("--coalesce", type=int, choices=(0, 1), default=1)
    p.add_argument("--column-rotation", type=int, choices=(0, 2, 4, 6), default=0)
    p.add_argument("--emit-only", action="store_true")
    p = commands.add_parser("package")
    p.add_argument("--build", type=Path, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (build if args.command == "build" else package)(args)


if __name__ == "__main__":
    main()
