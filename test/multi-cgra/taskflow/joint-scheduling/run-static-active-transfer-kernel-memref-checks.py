#!/usr/bin/env python3
"""Check authenticated memref.load/store roots inside Neura kernels.

The positive case proves the imported GCN active boxes for function arguments
1 through 12. A small standalone module exercises direct KernelOp memref
loads/stores and checks that a valid state route succeeds while mismatched
Taskflow roots, a capacity-inconsistent memref.cast, and an out-of-range index
remain unknown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


DEFAULT_FUNCTION = (
    "_Z8gcn_funciPA256_KiS1_S1_S1_PA16_S_S3_S3_S3_PA256_iPA256_S4_"
    "PA256_A16_iSA_SA_Pi"
)
EXPECTED_CAPACITIES = {
    1: [256, 256],
    2: [256, 256],
    3: [256, 256],
    4: [256, 256],
    5: [256, 16],
    6: [16, 16],
    7: [16, 16],
    8: [16, 16],
    9: [4, 256],
    10: [4, 256, 256],
    11: [3, 256, 16],
    12: [12, 256, 16],
}
EXPECTED_ACTIVE = {
    1: [5, 5],
    2: [5, 5],
    3: [5, 5],
    4: [5, 5],
    5: [5, 16],
    6: [16, 16],
    7: [16, 16],
    8: [16, 16],
    9: [4, 5],
    10: [4, 5, 5],
    11: [3, 5, 16],
    12: [12, 5, 16],
}

FIXTURE = r'''module {
  func.func @kernel_memref_valid(
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %read_done, %write_done = taskflow.task @Direct
        will_reads(%data : memref<?xi32>)
        will_writes(%data : memref<?xi32>)
        [original_read_memrefs(%data : memref<?xi32>),
         original_write_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>, memref<?xi32>)
          -> (memref<?xi32>, memref<?xi32>) {
      ^bb0(%read: memref<?xi32>, %write: memref<?xi32>):
        neura.kernel inputs(%read, %write : memref<?xi32>, memref<?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%kernel_read: memref<?xi32>, %kernel_write: memref<?xi32>):
            %c0 = arith.constant 0 : index
            %value = memref.load %kernel_read[%c0] : memref<?xi32>
            memref.store %value, %kernel_write[%c0] : memref<?xi32>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
                         done_writes(%write : memref<?xi32>)
    }
    return
  }

  func.func @kernel_memref_wrong_state(
      %actual: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      },
      %declared: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %done = taskflow.task @WrongState
        will_reads(%actual : memref<?xi32>)
        [original_read_memrefs(%declared : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>):
        neura.kernel inputs(%read : memref<?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%kernel_read: memref<?xi32>):
            %c0 = arith.constant 0 : index
            %value = memref.load %kernel_read[%c0] : memref<?xi32>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }

  func.func @kernel_memref_bad_cast(
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %done = taskflow.task @BadCast
        will_reads(%data : memref<?xi32>)
        [original_read_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>):
        neura.kernel inputs(%read : memref<?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%kernel_read: memref<?xi32>):
            %view = memref.cast %kernel_read : memref<?xi32> to memref<5xi32>
            %c0 = arith.constant 0 : index
            %value = memref.load %view[%c0] : memref<5xi32>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }

  func.func @kernel_memref_out_of_bounds(
      %data: memref<?xi32> {
        amoeba.logical_transfer_shape = array<i64: 4>, amoeba.noalias
      }) attributes {
        amoeba.input0_caller_evidence_path = "fixture-caller.mlir",
        amoeba.input0_caller_function = "main",
        amoeba.input0_caller_noalias_proof_mode = "prepared-external-caller",
        amoeba.input0_caller_noalias_proven,
        amoeba.input0_caller_prepared_input_path = "fixture-prepared.mlir"
      } {
    %done = taskflow.task @OutOfBounds
        will_reads(%data : memref<?xi32>)
        [original_read_memrefs(%data : memref<?xi32>)]
        : (memref<?xi32>) -> (memref<?xi32>) {
      ^bb0(%read: memref<?xi32>):
        neura.kernel inputs(%read : memref<?xi32>)
            attributes {accelerator = "neura", dataflow_mode = "predicate"} {
          ^bb0(%kernel_read: memref<?xi32>):
            %c4 = arith.constant 4 : index
            %value = memref.load %kernel_read[%c4] : memref<?xi32>
            neura.yield {yield_type = "void"}
        }
        taskflow.yield done_reads(%read : memref<?xi32>)
    }
    return
  }
}'''


def fail(message: str) -> None:
    raise SystemExit(message)


def run_proof(optimizer: Path, architecture: Path, module: Path,
              function: str, arguments: str, output: Path) -> dict:
    report = output.with_suffix(".json")
    command = [
        str(optimizer),
        str(module),
        "--verify-each",
        f"--architecture-spec={architecture}",
        "--prove-static-active-transfer-shapes="
        f"function={function} arguments={arguments} report-output={report}",
        "-o",
        str(output),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode:
        fail(f"active-transfer proof failed to run for {function}:\n"
             f"{result.stdout[-5000:]}")
    return json.loads(report.read_text())


def check_gcn(proof_path: Path) -> None:
    document = json.loads(proof_path.read_text())
    rows = document.get("proofs")
    if not isinstance(rows, list) or len(rows) != 12:
        fail("real GCN report does not contain exactly arguments 1 through 12")
    by_argument = {row.get("argument"): row for row in rows}
    if set(by_argument) != set(EXPECTED_ACTIVE):
        fail("real GCN proof arguments are not exactly 1 through 12")
    for argument, active in EXPECTED_ACTIVE.items():
        row = by_argument[argument]
        if (row.get("status") != "proven" or
                row.get("capacity_shape") != EXPECTED_CAPACITIES[argument] or
                row.get("active_shape") != active or row.get("reasons")):
            fail(f"GCN argument {argument} active box/capacity mismatch: {row}")
        accesses = row.get("accesses")
        if not isinstance(accesses, list) or not accesses:
            fail(f"GCN argument {argument} has no access evidence")
        for access in accesses:
            if (not access.get("proven") or
                    access.get("operation") not in ("memref.load", "memref.store") or
                    not isinstance(access.get("input_index"), int) or
                    not all(axis.get("proven") for axis in access.get("axes", []))):
                fail(f"GCN argument {argument} has an unauthenticated direct access")


def check_synthetic(optimizer: Path, architecture: Path,
                    output_root: Path) -> None:
    fixture = output_root / "kernel-memref-negative-cases.mlir"
    fixture.write_text(FIXTURE)
    positive = run_proof(
        optimizer, architecture, fixture, "kernel_memref_valid", "0",
        output_root / "kernel-memref-positive.mlir")
    row = positive["proofs"][0]
    if (row.get("status") != "proven" or row.get("active_shape") != [1] or
            row.get("capacity_shape") != [4] or row.get("reasons")):
        fail(f"valid direct kernel memref load/store route did not prove: {row}")
    accesses = row.get("accesses", [])
    if ({access.get("operation") for access in accesses} !=
            {"memref.load", "memref.store"} or
            any(not access.get("proven") for access in accesses)):
        fail("positive direct kernel route did not authenticate both load and store")

    negative_expectations = {
        "kernel_memref_wrong_state": "does not authenticate its state",
        "kernel_memref_bad_cast": "static extents disagree with logical transfer capacity",
        "kernel_memref_out_of_bounds": "outside the original capacity",
    }
    for function, expected_reason in negative_expectations.items():
        report = run_proof(
            optimizer, architecture, fixture, function, "0",
            output_root / f"{function}.mlir")
        row = report["proofs"][0]
        reasons = list(row.get("reasons", []))
        reasons.extend(access.get("reason", "") for access in row.get("accesses", []))
        if (row.get("status") != "unknown" or "active_shape" in row or
                not any(expected_reason in reason for reason in reasons)):
            fail(f"negative {function} unexpectedly proved or lost its guard: {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--gcn-input", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--function", default=DEFAULT_FUNCTION)
    args = parser.parse_args()

    optimizer = args.optimizer.resolve()
    source = args.gcn_input.resolve()
    architecture = args.architecture.resolve()
    if not optimizer.is_file() or not source.is_file() or not architecture.is_file():
        fail("optimizer, GCN input, and architecture must all exist")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    run_proof(
        optimizer, architecture, source, args.function,
        ",".join(str(argument) for argument in EXPECTED_ACTIVE),
        output_root / "gcn-active-transfer.mlir")
    # run_proof writes the report adjacent to the output MLIR.
    check_gcn(output_root / "gcn-active-transfer.json")
    check_synthetic(optimizer, architecture, output_root)
    print("active-transfer kernel memref checks passed: real GCN args 1-12 "
          "and direct load/store route negatives")


if __name__ == "__main__":
    main()
