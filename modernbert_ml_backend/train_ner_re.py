"""Compatibility module and CLI for the original training entry."""
import sys
from training import trainer as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    # Preserve private symbols and monkeypatch identity for legacy callers.
    sys.modules[__name__] = _implementation
