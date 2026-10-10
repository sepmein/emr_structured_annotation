"""Compatibility entry point; implementation lives in emr_annotation.annotation_analysis.double_annotations."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_annotation.annotation_analysis import double_annotations as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    # Preserve module identity, private symbols, and monkeypatch behavior.
    sys.modules[__name__] = _implementation
