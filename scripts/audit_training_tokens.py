"""Compatibility entry point; implementation lives in emr_annotation.training_data.token_audit."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emr_annotation.training_data import token_audit as _implementation

# Preserve module identity and all public/private symbols.
sys.modules[__name__] = _implementation
