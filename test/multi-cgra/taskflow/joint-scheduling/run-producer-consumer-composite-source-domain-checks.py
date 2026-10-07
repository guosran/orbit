#!/usr/bin/env python3
"""Check producer-consumer fusion source proof and fail-closed mutations."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile


FUNCTION = "source_partition_lineage"
PRODUCER = "Task"
CONSUMER = "Consumer"
FUSED = "Task.fuse.Consumer"
SCHEMA = "orbit-joint-neighborhood-typed-actions-v1"


def fail(message: str) -> None:
    raise SystemExit(message)


def run(command: list[str], expect_success: bool) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if (result.returncode == 0) != expect_success:
        fail(
            f"unexpected exit status {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def mask_quoted_strings(text: str) -> str:
    masked = list(text)
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            masked[index] = "\0" if char != "\n" else "\n"
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            masked[index] = "\0"
            quoted = True
    return "".join(masked)


def task_span(text: str, task_name: str) -> tuple[int, int]:
    # Ignore quoted source witnesses when locating the actual region braces.
    plain = mask_quoted_strings(text)
    start = plain.find(f"taskflow.task @{task_name}")
    if start >= 0:
        signature = re.search(r":\s*\(", plain[start:])
        if not signature:
            fail(f"missing type signature for {task_name}")
        body = plain.find("{", start + signature.start())
        if body < 0:
            fail(f"missing body for {task_name}")
        depth = 0
        for position in range(body, len(plain)):
            if plain[position] == "{":
                depth += 1
            elif plain[position] == "}":
                depth -= 1
                if depth == 0:
                    return text.rfind("\n", 0, start) + 1, position + 1

    # Captured canonical candidates use generic taskflow.task syntax.  Locate
    # the operation header by its explicit task_name attribute and balance the
    # actual region braces while ignoring serialized source-proof strings.
    generic = re.search(
        rf'(?m)^\s*(?:%[A-Za-z0-9_.$-]+(?::[0-9]+)?\s*=\s*)?'
        rf'"taskflow\.task"[^\n]*\btask_name\s*=\s*"{re.escape(task_name)}"',
        text,
    )
    if generic:
        line_start = text.rfind("\n", 0, generic.start()) + 1
        region = plain.find("({", generic.end())
        if region < 0:
            fail(f"missing generic task body for {task_name}")
        body = region + 1
        depth = 0
        for position in range(body, len(plain)):
            if plain[position] == "{":
                depth += 1
            elif plain[position] == "}":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    if end < len(text) and text[end] == ")":
                        end += 1
                    return line_start, end
    fail(f"unterminated task body for {task_name}")


def replace_once_unquoted(text: str, old: str, new: str, label: str) -> str:
    plain = mask_quoted_strings(text)
    if plain.count(old) != 1:
        fail(f"{label}: expected exactly one unquoted occurrence of {old!r}")
    start = plain.index(old)
    return text[:start] + new + text[start + len(old) :]


def substitute_once_unquoted(
    text: str, pattern: str, replacement: str, label: str
) -> str:
    plain = mask_quoted_strings(text)
    matches = list(re.finditer(pattern, plain, flags=re.MULTILINE))
    if len(matches) != 1:
        fail(f"{label}: expected exactly one unquoted regex match, got {len(matches)}")
    match = matches[0]
    original_segment = text[match.start() : match.end()]
    original_match = re.match(pattern, original_segment, flags=re.MULTILINE)
    if not original_match or original_match.end() != len(original_segment):
        fail(f"{label}: matched text differs from its unquoted source span")
    return (
        text[: match.start()]
        + original_match.expand(replacement)
        + text[match.end() :]
    )


def contains_unquoted(text: str, expected: str) -> bool:
    return expected in mask_quoted_strings(text)


def extract_graph_facts(
    optimizer: str, module: pathlib.Path, function: str, output: pathlib.Path
) -> dict[str, object]:
    pipeline = (
        "builtin.module(extract-joint-task-graph-facts{"
        f"function={function} output={output} fact-only=true"
        "})"
    )
    run(
        [optimizer, str(module), f"--pass-pipeline={pipeline}", "-o", "/dev/null"],
        expect_success=True,
    )
    return json.loads(output.read_text())


def substitute_first_unquoted(
    text: str, pattern: str, replacement: str, label: str
) -> str:
    plain = mask_quoted_strings(text)
    match = re.search(pattern, plain, flags=re.MULTILINE)
    if not match:
        fail(f"{label}: expected an unquoted regex match")
    return text[: match.start()] + match.expand(replacement) + text[match.end() :]


def string_attribute_equals(text: str, attribute: str, expected: str) -> bool:
    plain = mask_quoted_strings(text)
    matches = list(re.finditer(re.escape(attribute) + r"\s*=\s*", plain))
    if len(matches) != 1:
        return False
    value_start = matches[0].end()
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    if value_start >= len(text) or text[value_start] != '"':
        return False
    end = value_start + 1
    escaped = False
    while end < len(text):
        if escaped:
            escaped = False
        elif text[end] == "\\":
            escaped = True
        elif text[end] == '"':
            return text[value_start + 1 : end] == expected
        end += 1
    return False


def replace_string_attribute_once(
    text: str, attribute: str, value: str, label: str
) -> str:
    plain = mask_quoted_strings(text)
    matches = list(
        re.finditer(re.escape(attribute) + r"\s*=\s*", plain)
    )
    if len(matches) != 1:
        fail(f"{label}: expected exactly one unquoted attribute, got {len(matches)}")
    value_start = matches[0].end()
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    if value_start >= len(text) or text[value_start] != '"':
        fail(f"{label}: attribute value is not a quoted string")
    end = value_start + 1
    escaped = False
    while end < len(text):
        if escaped:
            escaped = False
        elif text[end] == "\\":
            escaped = True
        elif text[end] == '"':
            return text[: value_start + 1] + value + text[end:]
        end += 1
    fail(f"{label}: unterminated attribute string")


def mutate_task(text: str, task_name: str, transform, label: str) -> str:
    begin, end = task_span(text, task_name)
    before = text[begin:end]
    after = transform(before)
    if after == before:
        fail(f"{label}: mutation made no change")
    return text[:begin] + after + text[end:]


def parse_neura_definitions(text: str) -> dict[str, tuple[str, list[str]]]:
    definitions: dict[str, tuple[str, list[str]]] = {}
    for line in text.splitlines():
        match = re.match(
            r'^\s*(%[A-Za-z0-9_.$-]+)\s*=\s*"neura\.([A-Za-z0-9_]+)"\(([^)]*)\)',
            line,
        )
        if not match:
            continue
        result, operation, operands = match.groups()
        definitions[result] = (
            operation,
            re.findall(r"%[A-Za-z0-9_.$-]+", operands),
        )
    return definitions


def producer_store_data_mov(module: str) -> tuple[int, str, str, str]:
    begin, end = task_span(module, PRODUCER)
    task = module[begin:end]
    lines = task.splitlines()
    store_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith('"neura.store_indexed"')
        ),
        None,
    )
    if store_index is None:
        fail("producer fixture lost its indexed store")
    store_match = re.search(
        r'"neura\.store_indexed"\((%[A-Za-z0-9_.$-]+)', lines[store_index]
    )
    if not store_match:
        fail("cannot read producer indexed-store value")
    stored_value = store_match.group(1)
    definition_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(
                rf'^\s*{re.escape(stored_value)}\s*=\s*"neura\.data_mov"\(',
                line,
            )
        ),
        None,
    )
    if definition_index is None:
        fail("producer store value is not an explicit DataMov wrapper")
    definition = lines[definition_index]
    type_match = re.search(r"\s:\s(.*)$", definition)
    if not type_match:
        fail("cannot read producer DataMov type signature")
    return store_index, stored_value, type_match.group(1), lines[store_index]


def add_forwarded_move_attribute(module: str, move_value: str) -> str:
    begin, end = task_span(module, PRODUCER)
    task = module[begin:end]
    lines = task.splitlines()
    definition_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(
                rf'^\s*{re.escape(move_value)}\s*=\s*"neura\.data_mov"\(',
                line,
            )
        ),
        None,
    )
    if definition_index is None:
        fail("cannot find forwarded producer DataMov for attribute tamper")
    line = lines[definition_index]
    operands_end = line.find(")")
    if operands_end < 0 or line[operands_end + 1 :].lstrip().startswith("{"):
        fail("forwarded producer DataMov already has or lacks its attribute slot")
    lines[definition_index] = (
        line[: operands_end + 1]
        + ' {test_nonidentity_route = "keep"}'
        + line[operands_end + 1 :]
    )
    return module[:begin] + "\n".join(lines) + module[end:]


def add_type_changing_forwarded_moves(module: str) -> str:
    begin, end = task_span(module, PRODUCER)
    task = module[begin:end]
    lines = task.splitlines()
    store_index, stored_value, type_signature, store_line = producer_store_data_mov(
        module
    )
    store_match = re.search(
        r'("neura\.store_indexed"\()(%[A-Za-z0-9_.$-]+)(,)', store_line
    )
    if not store_match:
        fail("cannot rewrite producer indexed-store value for type tamper")
    source_type = type_signature.split(" -> ", 1)[0][1:-1]
    result_type = type_signature.split(" -> ", 1)[1]
    if source_type != result_type or "!neura.data<i32," not in result_type:
        fail(f"fixture no longer has the expected i32 DataMov: {type_signature}")
    alternate_type = result_type.replace("!neura.data<i32,", "!neura.data<i64,", 1)
    indent = re.match(r"^\s*", store_line).group(0)
    first = "%test_type_changing_data_mov"
    second = "%test_type_restoring_data_mov"
    lines.insert(
        store_index,
        f'{indent}{first} = "neura.data_mov"({stored_value}) : '
        f'({source_type}) -> {alternate_type}',
    )
    lines.insert(
        store_index + 1,
        f'{indent}{second} = "neura.data_mov"({first}) : '
        f'({alternate_type}) -> {result_type}',
    )
    store_index += 2
    lines[store_index] = (
        store_line[: store_match.start(2)]
        + second
        + store_line[store_match.end(2) :]
    )
    return module[:begin] + "\n".join(lines) + module[end:]


def add_second_kernel_block(module: str) -> str:
    begin, end = task_span(module, PRODUCER)
    task = module[begin:end]
    lines = task.splitlines()
    kernel_start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(
                r'^\s*(?:%[A-Za-z0-9_.$-]+\s*=\s*)?"neura\.kernel"', line
            )
        ),
        None,
    )
    if kernel_start is None:
        fail("producer fixture lost its Neura kernel")
    kernel_text = "\n".join(lines[kernel_start:])
    kernel_plain = mask_quoted_strings(kernel_text)
    marker = kernel_plain.find("({")
    if marker < 0:
        fail("cannot locate producer kernel region")
    body_open = marker + 1
    depth = 0
    body_close = None
    for position in range(body_open, len(kernel_plain)):
        if kernel_plain[position] == "{":
            depth += 1
        elif kernel_plain[position] == "}":
            depth -= 1
            if depth == 0:
                body_close = position
                break
    if body_close is None:
        fail("unterminated producer kernel region")
    if not any('"neura.data_mov"' in line for line in lines[kernel_start:]):
        fail("producer kernel has no DataMov to exercise the multi-block guard")
    kernel_body_lines = lines[kernel_start:]
    block_line = next(
        (line for line in kernel_body_lines if line.lstrip().startswith("^bb0")),
        None,
    )
    yield_line = next(
        (
            line
            for line in kernel_body_lines
            if line.lstrip().startswith('"neura.yield"')
        ),
        None,
    )
    if block_line is None or yield_line is None:
        fail("producer kernel is missing its single-block terminator")
    block_indent = re.match(r"^\s*", block_line).group(0)
    operation_indent = block_indent + "  "
    prefix = kernel_text[:body_close]
    suffix = kernel_text[body_close:]
    extra_block = (
        f"\n{block_indent}^bb1(%test_other_block_input: !neura.data<i32, i1>):\n"
        f'{operation_indent}%test_other_block_move = "neura.data_mov"('
        "%test_other_block_input) : (!neura.data<i32, i1>) -> "
        "!neura.data<i32, i1>\n"
        f'{operation_indent}{yield_line.lstrip()}\n'
    )
    changed_kernel = prefix + extra_block + suffix
    rewritten = "\n".join(lines[:kernel_start]) + changed_kernel
    return module[:begin] + rewritten + module[end:]


def assert_one_transparent_forwarding_move(
    fused_task: str, producer_result: str, label: str
) -> None:
    multiplies = [
        line
        for line in fused_task.splitlines()
        if '"neura.mul"(' in line
    ]
    if len(multiplies) != 1:
        fail(f"{label}: expected one fused consumer multiply, got {len(multiplies)}")
    operands_match = re.search(r'"neura\.mul"\(([^)]*)\)', multiplies[0])
    if not operands_match:
        fail(f"{label}: cannot parse fused consumer multiply operands")
    operands = re.findall(r"%[A-Za-z0-9_.$-]+", operands_match.group(1))
    definitions = parse_neura_definitions(fused_task)
    if not operands:
        fail(f"{label}: fused consumer multiply has no SSA input")
    value = operands[0]
    moves: list[str] = []
    while value in definitions and definitions[value][0] == "data_mov":
        move_operands = definitions[value][1]
        if len(move_operands) != 1:
            fail(f"{label}: malformed DataMov on forwarded path")
        moves.append(value)
        value = move_operands[0]
    if len(moves) != 1 or value != producer_result:
        fail(
            f"{label}: expected consumer DataMov -> producer value "
            f"{producer_result}, got moves={moves}, root={value}"
        )


def typed_fusion_action(
    mode: str = "producer-consumer-forwarded",
) -> dict[str, object]:
    return {
        "family": "fusion",
        "label": f"fuse:{PRODUCER}:{CONSUMER}:{mode}",
        "primitives": [
            {
                "kind": "fusion",
                "firstTask": PRODUCER,
                "secondTask": CONSUMER,
                "axis": 0,
                "factor": 1,
                "mode": mode,
            }
        ],
        "shapeTask": "",
        "shapeRows": 0,
        "shapeCols": 0,
        "canonicalReset": False,
    }


def main() -> None:
    if len(sys.argv) != 3:
        fail(
            "usage: run-producer-consumer-composite-source-domain-checks.py "
            "FIXTURE OPTIMIZER"
        )
    fixture = pathlib.Path(sys.argv[1]).resolve()
    optimizer = sys.argv[2]

    with tempfile.TemporaryDirectory(prefix="pc-composite-source-domain-") as temp:
        directory = pathlib.Path(temp)
        canonical = directory / "canonical.mlir"
        source_pipeline = (
            "builtin.module(func.func(construct-hyperblock-from-task),"
            "assign-accelerator,classify-task-and-counter,"
            "convert-taskflow-to-neura,func.func(lower-affine-to-neura),"
            "insert-data-mov,bind-source-iteration-domain)"
        )
        run(
            [
                optimizer,
                str(fixture),
                f"--pass-pipeline={source_pipeline}",
                "-o",
                str(canonical),
            ],
            expect_success=True,
        )

        actions = directory / "forwarded-fusion.actions.json"
        actions.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "canonicalInput": str(canonical),
                    "candidateInput": str(canonical),
                    "function": FUNCTION,
                    "stage": "full-joint",
                    "maxPartitionFactor": 4,
                    "actions": [typed_fusion_action()],
                },
                indent=2,
            )
            + "\n"
        )
        replay_dir = directory / "forwarded-fusion-replay"
        pipeline = (
            "builtin.module(replay-joint-neighborhood-actions{"
            f"action-file={actions} canonical-input={canonical} "
            f"candidate-input={canonical} function={FUNCTION} "
            f"stage=full-joint max-partition-factor=4 output-dir={replay_dir}"
            "})"
        )
        run(
            [
                optimizer,
                str(canonical),
                f"--pass-pipeline={pipeline}",
                "-o",
                "/dev/null",
            ],
            expect_success=True,
        )
        generic_candidate = replay_dir / "candidate.mlir"
        candidate = directory / "forwarded-candidate-readable.mlir"
        run(
            [
                optimizer,
                str(generic_candidate),
                "--verify-each",
                "-o",
                str(candidate),
            ],
            expect_success=True,
        )
        facts = json.loads((replay_dir / "source-facts.json").read_text())
        if facts.get("status") != "complete" or facts.get("actions_applied") != 1:
            fail(f"unexpected complete replay facts: {facts}")
        if facts.get("source_iteration_domain_verified") is not True:
            fail(f"forwarded fusion did not pass source proof: {facts}")
        positive = candidate.read_text()
        fused_start, fused_end = task_span(positive, FUSED)
        fused = positive[fused_start:fused_end]
        if (
            not string_attribute_equals(fused, "amoeba.neura.fusion.mode", "forwarded")
            or not contains_unquoted(
                fused, "amoeba.neura.fusion.eliminated_loads = 1 : i64"
            )
            or not contains_unquoted(
                fused, "amoeba.neura.fusion.eliminated_stores = 1 : i64"
            )
        ):
            fail("forwarded fusion did not record exactly one removed load/store")
        load_pattern = (
            r'^\s*(?:%[A-Za-z0-9_.$-]+\s*=\s*)?'
            r'(?:"(?:memref\.load|neura\.load_indexed)"'
            r'|(?:memref\.load|neura\.load_indexed))(?=\s|\()'
        )
        store_pattern = (
            r'^\s*(?:"(?:memref\.store|neura\.store_indexed)"'
            r'|(?:memref\.store|neura\.store_indexed))(?=\s|\()'
        )
        memory_loads = re.findall(load_pattern, fused, flags=re.MULTILINE)
        memory_stores = re.findall(store_pattern, fused, flags=re.MULTILINE)
        if len(memory_loads) != 1 or len(memory_stores) != 1:
            fail("forwarded fusion did not remove the private intermediate memory access pair")
        def binary_results(neura_name: str, arith_name: str):
            found = []
            for line in fused.splitlines():
                match = re.match(
                    r"^\s*(%[A-Za-z0-9_.$-]+)\s*=\s*(.*?)\s*$", line
                )
                if not match:
                    continue
                result, expression = match.groups()
                if f"\"neura.{neura_name}\"(" in expression:
                    operands = re.search(r"\(([^)]*)\)", expression)
                    if operands:
                        found.append((result, operands.group(1)))
                elif re.match(rf"{re.escape(arith_name)}\s+", expression):
                    operands = re.match(
                        rf"{re.escape(arith_name)}\s+(.+?)\s*:\s*.+$",
                        expression,
                    )
                    if operands:
                        found.append((result, operands.group(1)))
            return found

        adds = binary_results("add", "arith.addi")
        multiplies = binary_results("mul", "arith.muli")
        if len(adds) != 1 or len(multiplies) != 1:
            fail("forwarded fusion lost the fixture's unique add/multiply chain")
        add_result, _ = adds[0]
        assert_one_transparent_forwarding_move(fused, add_result, "forwarded fusion")
        run(
            [
                optimizer,
                str(candidate),
                "--verify-source-iteration-domain-partitions="
                f"parent-module={canonical} function={FUNCTION}",
                "-o",
                "/dev/null",
            ],
            expect_success=True,
        )

        retained_actions = directory / "retained-fusion.actions.json"
        retained_actions.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "canonicalInput": str(canonical),
                    "candidateInput": str(canonical),
                    "function": FUNCTION,
                    "stage": "full-joint",
                    "maxPartitionFactor": 4,
                    "actions": [
                        typed_fusion_action("producer-consumer-retained")
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        retained_replay_dir = directory / "retained-fusion-replay"
        retained_pipeline = (
            "builtin.module(replay-joint-neighborhood-actions{"
            f"action-file={retained_actions} canonical-input={canonical} "
            f"candidate-input={canonical} function={FUNCTION} "
            f"stage=full-joint max-partition-factor=4 "
            f"output-dir={retained_replay_dir}"
            "})"
        )
        run(
            [
                optimizer,
                str(canonical),
                f"--pass-pipeline={retained_pipeline}",
                "-o",
                "/dev/null",
            ],
            expect_success=True,
        )
        retained_facts = json.loads(
            (retained_replay_dir / "source-facts.json").read_text()
        )
        if (
            retained_facts.get("status") != "complete"
            or retained_facts.get("actions_applied") != 1
            or retained_facts.get("source_iteration_domain_verified") is not True
        ):
            fail(f"retained load-sharing replay did not pass source proof: {retained_facts}")
        retained_generic = retained_replay_dir / "candidate.mlir"
        retained_candidate = directory / "retained-candidate-readable.mlir"
        run(
            [
                optimizer,
                str(retained_generic),
                "--verify-each",
                "-o",
                str(retained_candidate),
            ],
            expect_success=True,
        )
        retained_text = retained_candidate.read_text()
        retained_start, retained_end = task_span(retained_text, FUSED)
        retained_task = retained_text[retained_start:retained_end]
        if (
            not string_attribute_equals(
                retained_task, "amoeba.neura.fusion.mode", "retained"
            )
            or not contains_unquoted(
                retained_task, "amoeba.neura.fusion.eliminated_loads = 1 : i64"
            )
            or not contains_unquoted(
                retained_task, "amoeba.neura.fusion.eliminated_stores = 0 : i64"
            )
        ):
            fail("retained load sharing did not record one load and zero stores removed")
        retained_add_results = [
            result
            for result, _ in re.findall(
                r'^\s*(%[A-Za-z0-9_.$-]+)\s*=\s*"neura\.add"\(([^)]*)\)',
                retained_task,
                flags=re.MULTILINE,
            )
        ]
        if len(retained_add_results) != 1:
            fail("retained fusion lost the fixture's unique producer add")
        assert_one_transparent_forwarding_move(
            retained_task, retained_add_results[0], "retained fusion"
        )
        retained_loads = re.findall(
            load_pattern, retained_task, flags=re.MULTILINE
        )
        retained_stores = re.findall(
            store_pattern, retained_task, flags=re.MULTILINE
        )
        if len(retained_loads) != 1 or len(retained_stores) != 2:
            fail("retained load sharing must keep the producer and consumer stores")
        run(
            [
                optimizer,
                str(retained_candidate),
                "--verify-source-iteration-domain-partitions="
                f"parent-module={canonical} function={FUNCTION}",
                "-o",
                "/dev/null",
            ],
            expect_success=True,
        )
        retained_graph_facts = extract_graph_facts(
            optimizer,
            retained_candidate,
            FUNCTION,
            directory / "retained-graph-facts.json",
        )
        retained_record = next(
            (
                task
                for task in retained_graph_facts.get("tasks", [])
                if task.get("task_name") == FUSED
            ),
            None,
        )
        expected_retained_fusion = {
            "mode": "retained",
            "eliminated_loads": 1,
            "eliminated_stores": 0,
        }
        if (
            retained_record is None
            or retained_record.get("neura_fusion") != expected_retained_fusion
        ):
            fail(
                "fact extraction did not preserve valid retained 1/0 metadata: "
                f"{retained_record}"
            )
        structural_nodes = retained_graph_facts.get("structural_key", {}).get(
            "nodes", []
        )
        if not any(
            node.get("neura_fusion") == expected_retained_fusion
            for node in structural_nodes
        ):
            fail(
                "graph structural key dropped the retained elimination counts: "
                f"{structural_nodes}"
            )

        def expect_fusion_rejected(label: str, mutant: str, reason: str) -> None:
            path = directory / f"{label}.mlir"
            path.write_text(mutant)
            result = run(
                [
                    optimizer,
                    str(path),
                    "--materialize-neura-joint-rewrite="
                    "first-task-name=Task second-task-name=Consumer "
                    "fusion-mode=producer-consumer-retained",
                    "-o",
                    "/dev/null",
                ],
                expect_success=False,
            )
            diagnostics = result.stdout + result.stderr
            if reason not in diagnostics:
                fail(f"{label}: expected fail-closed reason {reason!r}:\n{diagnostics}")

        canonical_text = canonical.read_text()
        _, stored_value, _, _ = producer_store_data_mov(canonical_text)
        expect_fusion_rejected(
            "annotated-forwarded-datamov",
            add_forwarded_move_attribute(canonical_text, stored_value),
            "non-identity producer DataMov chain",
        )
        expect_fusion_rejected(
            "type-changing-forwarded-datamovs",
            add_type_changing_forwarded_moves(canonical_text),
            "non-identity producer DataMov chain",
        )
        expect_fusion_rejected(
            "multi-block-producer-kernel",
            add_second_kernel_block(canonical_text),
            "requires one stateless kernel per task",
        )

        tampered_counts = replace_once_unquoted(
            retained_text,
            "amoeba.neura.fusion.eliminated_loads = 1 : i64",
            "amoeba.neura.fusion.eliminated_loads = 0 : i64",
            "retained load count",
        )
        tampered_path = directory / "retained-load-count-tamper.mlir"
        tampered_path.write_text(tampered_counts)
        tampered_result = run(
            [
                optimizer,
                str(tampered_path),
                "--verify-source-iteration-domain-partitions="
                f"parent-module={canonical} function={FUNCTION}",
                "-o",
                "/dev/null",
            ],
            expect_success=False,
        )
        if "source iteration-domain partition is unproven" not in (
            tampered_result.stdout + tampered_result.stderr
        ):
            fail("composite source proof did not reject forged retained load counts")

        forged_store_count = directory / "retained-store-count-tamper.mlir"
        forged_store_count.write_text(
            replace_once_unquoted(
                retained_text,
                "amoeba.neura.fusion.eliminated_stores = 0 : i64",
                "amoeba.neura.fusion.eliminated_stores = 1 : i64",
                "retained store count",
            )
        )
        facts_output = directory / "forged-retained-store-count-facts.json"
        facts_pipeline = (
            "builtin.module(extract-joint-task-graph-facts{"
            f"function={FUNCTION} output={facts_output} fact-only=true"
            "})"
        )
        forged_store_result = run(
            [
                optimizer,
                str(forged_store_count),
                f"--pass-pipeline={facts_pipeline}",
                "-o",
                "/dev/null",
            ],
            expect_success=False,
        )
        if "incomplete or invalid amoeba.neura.fusion metadata" not in (
            forged_store_result.stdout + forged_store_result.stderr
        ):
            fail("fact extraction did not reject a forged retained store count")

        public = run(
            [
                optimizer,
                str(canonical),
                "--materialize-neura-joint-rewrite=first-task-name=PublicProducer "
                "second-task-name=PublicConsumer "
                "fusion-mode=producer-consumer-forwarded",
                "-o",
                "/dev/null",
            ],
            expect_success=False,
        )
        if "requires private intermediate storage" not in (
            public.stdout + public.stderr
        ):
            fail("forwarded materializer did not reject caller-visible intermediate storage")

        for mode in (
            "producer-consumer-retained",
            "producer-consumer-forwarded",
        ):
            repeated = run(
                [
                    optimizer,
                    str(canonical),
                    "--materialize-neura-joint-rewrite="
                    "first-task-name=RepeatedProducer "
                    "second-task-name=RepeatedConsumer "
                    f"fusion-mode={mode}",
                    "-o",
                    "/dev/null",
                ],
                expect_success=False,
            )
            if "not injective over every Taskflow firing axis" not in (
                repeated.stdout + repeated.stderr
            ):
                fail(
                    f"{mode} did not reject the repeated-address intermediate "
                    f"for the injectivity reason:\n{repeated.stderr}"
                )

        def reject(label: str, mutant: str) -> None:
            path = directory / f"{label}.mlir"
            path.write_text(mutant)
            result = run(
                [
                    optimizer,
                    str(path),
                    "--verify-source-iteration-domain-partitions="
                    f"parent-module={canonical} function={FUNCTION}",
                    "-o",
                    "/dev/null",
                ],
                expect_success=False,
            )
            diagnostics = result.stdout + result.stderr
            if "source iteration-domain partition is unproven" not in diagnostics:
                fail(f"{label}: failed outside source-proof diagnostic\n{diagnostics}")

        reject(
            "forged-parent-name",
            mutate_task(
                positive,
                FUSED,
                lambda task: replace_once_unquoted(
                    task,
                    f"taskflow.task @{FUSED}",
                    "taskflow.task @Task.fuse.Foreign",
                    "fused parent name",
                ),
                "parent name",
            ),
        )

        def mutate_forwarded_body(task: str) -> str:
            return substitute_once_unquoted(
                task, r"\barith\.addi\b", "arith.subi", "body mutation"
            )

        reject(
            "changed-fused-body",
            mutate_task(positive, FUSED, mutate_forwarded_body, "body"),
        )

        def remove_noalias(text: str) -> str:
            return substitute_first_unquoted(
                text,
                r'(%[A-Za-z0-9_.$-]+\s*:\s*memref<4xi32>\s*)\{\s*amoeba\.noalias\s*\}',
                r"\1",
                "alias mutation",
            )

        reject("changed-alias-contract", remove_noalias(positive))

        def forge_source_parent(task: str) -> str:
            return replace_string_attribute_once(
                task,
                "amoeba.source_iteration_source_control_binding",
                "forged-parent-origin",
                "source-origin mutation",
            )

        reject(
            "forged-source-parent-origin",
            mutate_task(positive, FUSED, forge_source_parent, "source parent"),
        )

        def mutate_counter(task: str) -> str:
            return substitute_once_unquoted(
                task,
                r'(?m)(^\s*%[A-Za-z0-9_.$-]+\s*=\s*taskflow\.counter[^\n]*?counter_id\s*=\s*)0(\s*: i32)',
                r"\g<1>7\g<2>",
                "counter mutation",
            )

        reject(
            "changed-fused-counter",
            mutate_task(positive, FUSED, mutate_counter, "counter"),
        )

        def bypass_task_completion(task: str) -> str:
            return substitute_once_unquoted(
                task,
                r"(will_reads\()(%[A-Za-z0-9_.$-]+)(\s*:\s*memref<4xi32>\))",
                r"\1%arg0\3",
                "completion mutation",
            )

        reject(
            "bypassed-fused-completion",
            mutate_task(positive, "Completion", bypass_task_completion,
                        "completion edge"),
        )

    print("retained and forwarded producer-consumer source-domain positive and tamper checks passed")


if __name__ == "__main__":
    main()
