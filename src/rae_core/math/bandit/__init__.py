"""
Multi-Armed Bandit system for Math Level Selection

This package implements online learning for the Math Layer Controller using
Multi-Armed Bandits with UCB algorithm and safety guardrails.
"""

from .arm import Arm, create_default_arms
from .bandit import BanditConfig, MultiArmedBandit
from .monitor import BanditMonitor, MonitorAlert

__all__ = [
    "Arm",
    "create_default_arms",
    "BanditConfig",
    "MultiArmedBandit",
    "BanditMonitor",
    "MonitorAlert",
]
