"""UTF-8 JSON output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from people_flow.errors import OutputError


def write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write readable JSON with a trailing newline and clear errors."""

    resolved = path.expanduser().resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise OutputError(f"Unable to write JSON output: {resolved}") from exc
    return resolved
