"""Compatibility exports for the router-based DocMasker module.

The only FastAPI application now lives at the repository root in ``main.py``.
"""

from .router import get_masking_service, router

__all__ = ["get_masking_service", "router"]
