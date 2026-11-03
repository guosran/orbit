"""Backend-independent static heterogeneous execution planning."""
from .mapper import map_program
from .contract import PlanningError
__all__ = ['map_program', 'PlanningError']
