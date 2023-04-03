"""CHSVI: A point-based algorithm for multi-agent control

Basic usage:
1. Generate BaseCPOMDP class object (or one if its child class) representing
a coordinator's POMDP model.
2. Use one of the presolve methods to initialize an upper bound and lower bound.
One can skip this step if one wishes to use default bounds.
3. Initialize a CHSVI class object with the presolve solutions if available.
4. Run CHSVI.Solve

See rundectiger.py for an example
"""

__author__ = "Dengwang Tang"

# coordinator's POMDP models
from .cpomdp import BaseCPOMDP
from .models.condindep import CondIndepCPOMDP
from .models.delaysharing import DelaySharingCPOMDP

# CHSVI algorithm class
from .core import CHSVI, UpperBound, LowerBound

# presolve methods
from . import presolve