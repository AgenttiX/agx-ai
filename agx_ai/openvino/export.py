"""Skipping up-to-date conversions and replacing the old ones atomically.

This module uses only the Python standard library, so that it can be tested without OpenVINO.
"""

from collections.abc import Callable
import json
from pathlib import Path
import shutil
from typing import Any

MARKER = "export.json"


def is_up_to_date(output: Path, info: dict[str, Any]) -> bool:
    """Whether the output directory contains a conversion with the same information (the marker file)."""
    marker = output / MARKER
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == info
    except json.JSONDecodeError:
        return False


def convert_once(output: Path, info: dict[str, Any], convert: Callable[[Path], None]) -> bool:
    """Run the conversion to the output directory unless it already contains a conversion with the same information.

    The information should contain everything that affects the result,
    e.g. the model revisions, the settings and the library versions.
    The conversion is written to a temporary directory, which replaces the old output only when it is complete,
    so that OVMS never sees a partial model.
    Returns whether the conversion was run.
    """
    if is_up_to_date(output, info):
        print(f"{output} is up to date: {json.dumps(info)}", flush=True)
        return False

    tmp = output.with_name(output.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print(f"Converting to {output}: {json.dumps(info)}", flush=True)
    convert(tmp)
    (tmp / MARKER).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(output, ignore_errors=True)
    tmp.rename(output)
    print(f"Converted to {output}", flush=True)
    return True
