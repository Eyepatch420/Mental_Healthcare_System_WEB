"""API package for stress prediction service."""

from .main import app  # expose FastAPI app at package import

__all__ = ["app"]
