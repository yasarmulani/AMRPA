"""Injection utilities."""

from .injector import (
    inject_amrpa,
    get_amrpa_wrappers,
    reset_amrpa_history,
    get_amrpa_metrics
)
from .detector import ArchitectureDetector

__all__ = [
    "inject_amrpa",
    "get_amrpa_wrappers",
    "reset_amrpa_history",
    "get_amrpa_metrics",
    "ArchitectureDetector",
]
