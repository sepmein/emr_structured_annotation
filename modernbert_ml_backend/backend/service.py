"""Compatibility alias for the renamed Label Studio service implementation."""
import sys
from label_studio_backend import service as _implementation

sys.modules[__name__] = _implementation
