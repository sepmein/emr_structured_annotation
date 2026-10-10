"""Compatibility module for the original script-style Label Studio entry."""
import sys
from label_studio_backend import service as _implementation

# Preserve private symbols and monkeypatch identity for legacy callers.
sys.modules[__name__] = _implementation
