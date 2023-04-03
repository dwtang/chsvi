"""Running CHSVI Algorithm on DecTiger with delayed information sharing
described in the paper
"""

import numpy as np
from numpy.random import default_rng
import sys
import time

from chsvi.models.delaysharing import DelaySharingCPOMDP
from chsvi.core import CHSVI
from chsvi.presolve import FullInfoHSVI


def DecTigerCPOMDP(S, d, discount=0.9):
    """DecTiger with delayed information sharing

    Args:
        S: number of doors
        d: delay
        discount: discount factor

    Returns:
        A DelaySharingCPOMDP model
    """
    A1, A2 = S+1, S+1 
    Z1, Z2 = S, S
    PT = np.zeros((S, A1, A2, S))
    PZ1 = np.zeros((A1, A2, S, Z1))
    PZ2 = np.zeros((A1, A2, S, Z2))
    r = np.zeros((S, A1, A2))
    b0 = np.ones(S) / S # uniform random at start

    # state transition
    PT[...] = 1/S # uniform by default
    PT[:, -1, -1, :] = np.eye(S) # except that if both agents are listening
    # no change to the state

    # observation distribution
    PZ1[...] = 1/S # uniform by default
    PZ1[-1, -1, :, :] = (0.7 * np.eye(S) + 0.15) / (0.7 + 0.15 * S)
    # except that if both agents are listening

    # create the observation matrix PZ: A1 x A2 x S x Z1 x Z2
    PZ2 = PZ1.copy()
    PZ = PZ1.reshape((A1, A2, S, Z1, 1)) * PZ2.reshape((A1, A2, S, 1, Z2))

    # create instantaneous reward
    ipad = 10 / (S-1)
    tiger = -100
    listeningcost = 1
    for i in range(S):
        r[i, 0:S, 0:S] = 2 * ipad 
        r[i, i, 0:S] = tiger
        r[i, 0:S, i] = tiger
        r[i, i, i] = tiger / 2
        r[i, S, 0:S] = ipad - listeningcost
        r[i, 0:S, S] = ipad - listeningcost
        r[i, S, i] = tiger - listeningcost
        r[i, i, S] = tiger - listeningcost
        r[i, S, S] = -2 * listeningcost

    return DelaySharingCPOMDP(PT, PZ, r, d, discount, b0)

def main():
    S, d, discount = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    # generate a BasePOMDP model
    Tiger = DecTigerCPOMDP(S, d, discount)
    # time keeping (optional)
    t0 = time.time()
    # Presolve a full information POMDP to initialize the upper bound 
    # (optional but recommended)
    presolveres = FullInfoHSVI(Tiger, targetgap=0.01)
    # Initialize the CHSVI solver with the presolve solution 
    Solver = CHSVI(Tiger, t0=t0, presolveres=presolveres)
    # Solve the coordinator's POMDP
    Solver.Solve(targetgap=0.01, timeout=86400)

if __name__ == '__main__':
    main()
