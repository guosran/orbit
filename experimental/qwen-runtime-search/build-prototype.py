#!/usr/bin/env python3
"""Build the adapter using an existing ORBIT beam-search test's flags/libraries."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--build-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = Path(__file__).with_name("qwen-local-search.cpp").resolve()
output = args.output.resolve()
object_path = str(output) + ".o"
commands = subprocess.check_output(
    ["ninja", "-C", str(args.build_dir), "-t", "commands", "taskflow-program-beam-search-test"],
    text=True,
).splitlines()
compile_command = next(
    command for command in commands
    if " -c " in command and command.endswith("/test/TaskflowGeneric/program-beam-search.cpp")
)
compile_args = shlex.split(compile_command)
old_object = compile_args[compile_args.index("-o") + 1]
compile_args[compile_args.index("-c") + 1] = str(source)
compile_args[compile_args.index("-o") + 1] = object_path
for flag in ["-MT", "-MF"]:
    if flag in compile_args:
        compile_args[compile_args.index(flag) + 1] = object_path + (".d" if flag == "-MF" else "")
link_args = shlex.split(commands[-1].removeprefix(": && ").removesuffix(" && :"))
link_args[link_args.index(old_object)] = object_path
link_args[link_args.index("-o") + 1] = str(output)
for command in [compile_args, link_args]:
    subprocess.run(command, cwd=args.build_dir, check=True)
Path(str(output) + ".build.json").write_text(
    json.dumps({"compile": compile_args, "link": link_args}, indent=2) + "\n"
)
