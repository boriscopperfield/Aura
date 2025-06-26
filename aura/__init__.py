"""
AURA - AI-Native Meta-OS

An AI-native meta-operating system that fundamentally reimagines how humans
interact with computational systems. Rather than forcing users to decompose
their intentions into primitive operations, AURA accepts high-level, ambiguous
goals and autonomously orchestrates complex workflows to achieve them.
"""

__version__ = "4.0.0"
__author__ = "AURA Team"

from aura.config import settings
from aura.utils.logging import logger
from aura.core.events import event_bus
from aura.core.transaction import transaction_manager
from aura.core.planner import get_planner_service
from aura.memory.models import MemoryNode, Query