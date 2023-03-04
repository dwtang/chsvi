"""Implementing HSVI Algorithm (Smith and Simmons 2006) in Python

Added some of my own tweaks to make the algorithm a bit faster
"""

__author__ = "Dengwang Tang"

from .pomdpparser import POMDPParser
from .pomdpmodel import POMDP
from .hsvi import HSVI, HeuristicSearchValueIteration