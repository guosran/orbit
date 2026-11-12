#!/usr/bin/env python3
"""Exercise finite input-feed loops and direct circuit runtime realization.

Inputs are actual extracted AIE and its sealed selected-graph contract. No
hardware is accessed. Every negative must reach a specific runtime guard;
parser failures and generic nonzero exits do not count as rejection evidence.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


SSA = r"%[A-Za-z0-9_.$-]+"
LOOP_GUARD = (
    "matrix runtime: unsupported runtime operation, loop, wrapper or descriptor"
)
ROUTE_GUARD = "matrix runtime: shim has no unique direct DMA route to checked core"
CHAIN_GUARD = "matrix runtime: routed core DMA chain is unsupported or absent"


def replace_once(text, old, new):
    """Do not silently pass a negative whose mutation missed its target."""
    if old == new or text.count(old) != 1:
        raise ValueError("mutation requires one unambiguous original operation")
    return text.replace(old, new, 1)


def loop_cases(text):
    """Extract SSA IDs from the actual runtime, independently of printer IDs."""
    runtime = text.index('"aie.runtime_sequence"')
    loop = text.index('"scf.for"', runtime)
    terminator = '}) : (index, index, index) -> ()'
    end = text.index(terminator, loop) + len(terminator)
    body = text[loop:end]
    bounds = re.search(r'"scf.for"\((' + SSA + r'), (' + SSA
                       + r'), (' + SSA + r')\)', body)
    if not bounds:
        raise ValueError("fixture needs a static three-operand feed loop")
    lower, upper, _step = bounds.groups()
    induction = re.search(r'\^bb0\((' + SSA + r'): index\)', body).group(1)
    handle = re.search(r'(' + SSA + r') = "aiex.dma_configure_task_for"',
                       body).group(1)
    constant_pattern = (
        re.escape(upper) + r' = "arith.constant"\(\) <\{value = '
        r'([0-9]+) : index\}> : \(\) -> index'
    )
    constant = re.search(constant_pattern, text[runtime:])
    if not constant:
        raise ValueError("fixture upper bound is not a positive index constant")
    original = constant.group(0)

    def bound_replacement(replacement):
        # Restrict replacement to this sequence's SSA scope.
        return text[:runtime] + replace_once(text[runtime:], original, replacement)

    def body_replacement(replacement):
        return text[:loop] + replacement + text[end:]

    dynamic = (f'{upper} = "arith.addi"({lower}, {lower}) : '
               '(index, index) -> index')
    overflow = re.sub(r'value = [0-9]+',
                      'value = 9223372036854775807', original)
    zero = re.sub(r'value = [0-9]+', 'value = 0', original)
    yield_op = '"scf.yield"() : () -> ()'
    sideeffect = '"air.wait_all"() : () -> ()\n        ' + yield_op
    extra_task = (f'"aiex.dma_start_task"({handle}) : (index) -> ()\n        '
                  + yield_op)
    bd_line = next(line for line in body.splitlines() if '"aie.dma_bd"' in line)
    # A pure operation in the BD region makes the enclosing loop IV live. This
    # is SSA-valid, but explicitly outside the invariant static-feed contract.
    dependent_bd = (f'        %orbit_test_iv = "arith.addi"({induction}, '
                    f'{induction}) : (index, index) -> index\n' + bd_line)
    return [
        ("unsafe-repeated-static-feed", text,
         "matrix runtime: multiple host output transfers in one configuration "
         "are unsupported"),
        ("dynamic-bound", bound_replacement(dynamic), LOOP_GUARD),
        ("overflow-trip-count", bound_replacement(overflow), LOOP_GUARD),
        ("zero-trip-loop", bound_replacement(zero), LOOP_GUARD),
        ("extra-sideeffect", body_replacement(
            replace_once(body, yield_op, sideeffect)), LOOP_GUARD),
        ("iv-dependent-bd-region", body_replacement(
            replace_once(body, bd_line, dependent_bd)), LOOP_GUARD),
        ("multiple-tasks", body_replacement(
            replace_once(body, yield_op, extra_task)), LOOP_GUARD),
    ]


def circuit_cases(text):
    """Resolve the input shim tile before mutating its actual direct route."""
    shim = next(line for line in text.splitlines()
                if '"aie.shim_dma_allocation"' in line
                and 'channel_dir = 1 :' in line)
    tile = re.search(r'"aie.shim_dma_allocation"\((' + SSA + r')\)', shim).group(1)
    flow = next(line for line in text.splitlines()
                if f'"aie.flow"({tile}, ' in line)
    channel = re.search(r'dest_channel = ([0-9]+)', flow).group(1)
    wrong = replace_once(flow, f'dest_channel = {channel}',
                         f'dest_channel = {1 - int(channel)}')
    return [
        ("positive-circuit", text, None),
        ("wrong-core-channel", replace_once(text, flow, wrong), CHAIN_GUARD),
        ("missing-circuit-route", replace_once(text, flow, ""), ROUTE_GUARD),
        ("duplicate-circuit-route", replace_once(text, flow, flow + "\n" + flow),
         ROUTE_GUARD),
    ]


def run_case(args, name, text, expected, contract):
    path = args.output / (name + ".mlir")
    path.write_text(text)
    if args.helper:
        command = [args.tool, str(contract), str(path),
                   str(args.output / (name + "-out"))]
    else:
        command = [args.tool, str(path),
                   "--realize-program-runtime=" + str(contract),
                   "--air-output=" + str(args.output / (name + ".output.mlir")),
                   "--implementation-output=" + str(args.output / (name + ".proof.json"))]
    result = subprocess.run(command, capture_output=True, text=True)
    diagnostic = result.stdout + result.stderr
    (args.output / (name + ".log")).write_text(diagnostic)
    parser_failure = any(marker in diagnostic for marker in
                         ("error:", "expected operation", "undeclared SSA"))
    passed = (result.returncode == 0 if expected is None else
              result.returncode != 0 and expected in diagnostic
              and not parser_failure)
    return {"case": name, "command": command, "exit_code": result.returncode,
            "expected_runtime_diagnostic": expected, "parser_failure": parser_failure,
            "pass": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--circuit-input", type=Path)
    parser.add_argument("--circuit-contract", type=Path)
    parser.add_argument("--helper", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.circuit_input) != bool(args.circuit_contract):
        parser.error("circuit input and contract must be supplied together")
    if args.output:
        args.output.mkdir(exist_ok=False)
    else:
        args.output = Path(tempfile.mkdtemp(prefix="orbit-matrix-runtime-static-"))
    cases = [(name, text, diagnostic, args.contract)
             for name, text, diagnostic in loop_cases(args.input.read_text())]
    if args.circuit_input:
        cases += [(name, text, diagnostic, args.circuit_contract)
                  for name, text, diagnostic in circuit_cases(
                      args.circuit_input.read_text())]
    rows = [run_case(args, *case) for case in cases]
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Evidence: {args.output}")
    for row in rows:
        print(f"{row['case']}: {'PASS' if row['pass'] else 'FAIL'}")
    return 0 if all(row["pass"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
