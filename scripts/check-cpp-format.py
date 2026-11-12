#!/usr/bin/env python3
"""Check ORBIT-owned C++ sources against the repository's LLVM format.

Run with Python 3 and clang-format on PATH, or pass --clang-format /path/to/tool.
Third-party sources and generated build files are outside these source roots.
"""

import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang-format", default="clang-format")
    args = parser.parse_args()
    formatter = shutil.which(args.clang_format)
    if formatter is None:
        parser.error("clang-format was not found; pass --clang-format with its path")
    root = Path(__file__).resolve().parents[1]
    sources = sorted(
        path
        for scope in ("include", "lib", "tools", "test")
        for path in (root / scope).rglob("*")
        if path.is_file() and path.suffix in (".h", ".cpp", ".inc")
    )
    result = subprocess.run(
        [formatter, "--style=file", "--dry-run", "--Werror", *map(str, sources)],
        cwd=root,
        check=False,
    )
    if result.returncode == 0:
        print(f"LLVM format: {len(sources)} ORBIT-owned C++ files passed")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
