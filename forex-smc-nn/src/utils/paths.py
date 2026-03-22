from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Project root: parent of `src/` (where configs, data, models live)."""
    return Path(__file__).resolve().parent.parent.parent
