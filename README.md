# CHSVI

Coordinator's Heuristic Search Value Iteration: an anytime algorithm for multi-agent control problems

Based on *D. Tang, A. Nayyar, R. Jain, A Novel Point-based Algorithm for Multi-agent Control Using the Common Information Approach* [link to paper](https://arxiv.org/abs/2304.04346)

Dependency: 
- numpy, scipy
- [Gurobi Optimization Studio](https://www.gurobi.com) and its associated python package gurobipy

The most basic input format is a [BaseCPOMDP](chsvi/cpomdp.py) class, a model for a general coordinator's POMDP. For convenience, we implemented two subclasses:
- [Delayed Sharing Information Model](chsvi/models/delaysharing.py)
- [Conditional Independent State Model](chsvi/models/condindep.py)

For example usage of the above models, see [rundectiger.py](rundectiger.py) and [runmulticast.py](runmulticast.py) respectively.

Released under MIT license

```
Copyright (c) 2023 Dengwang Tang <dwtang@umich.edu>
```
