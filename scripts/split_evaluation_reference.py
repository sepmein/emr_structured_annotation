"""Compatibility entry point; implementation lives in emr_annotation.training_data.split_reference."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_annotation.training_data import split_reference as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    # Preserve module identity, private symbols, and monkeypatch behavior.
    sys.modules[__name__] = _implementation
