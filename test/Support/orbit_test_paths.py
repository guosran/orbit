"""Standard-library paths for compiler-only regression tests.

ORBIT_BUILD_ROOT contains core/, air/, and amoeba/ build directories. Generated
regression output defaults beside those builds, never inside compiler sources.
"""
import os
import shutil
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2]


def build_root():
    return Path(os.environ.get(
        'ORBIT_BUILD_ROOT',
        str(CORE_ROOT.parent / 'orbit-artifact' / 'builds' / 'taskflow'))).expanduser().resolve()


def taskflow_tool(flavor='core'):
    return build_root() / flavor / 'tools' / 'taskflow-opt' / 'taskflow-opt'


def amoeba_tool():
    return build_root() / 'amoeba' / 'tools' / 'mlir-amoeba-opt' / 'mlir-amoeba-opt'


def air_opt():
    explicit = os.environ.get('AIR_OPT')
    if explicit:
        return Path(explicit)
    found = shutil.which('air-opt')
    if found:
        return Path(found)
    sibling = Path(sys.executable).parent / 'air-opt'
    return sibling if sibling.is_file() else Path('air-opt')
