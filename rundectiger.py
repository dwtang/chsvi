"""Testing Coordinator's HSVI on DecTiger with delayed information sharing

"""

import numpy as np
import sys
from numpy.random import default_rng

from chsvi.models.delaysharing import DelaySharingCPOMDP
from chsvi.models.condindep import CondIndepCPOMDP
from chsvi.core import CoordinatorsHeuristicSearchValueIteration
from chsvi.io import WriteCPOMDP

def DecTigerCPOMDP(S=2, d=2):
    A1, A2 = S+1, S+1 
    Z1, Z2 = S, S
    PT = np.zeros((S, A1, A2, S))
    PZ1 = np.zeros((A1, A2, S, Z1))
    PZ2 = np.zeros((A1, A2, S, Z2))
    r = np.zeros((S, A1, A2))
    discount = 0.95
    b0 = np.ones(S) / S # uniform random at start

    # state transition
    PT[...] = 1/S # uniform by default
    PT[:, -1, -1, :] = np.eye(S) # except that if both agents are listening
    # no change to the state

    # observation distribution
    PZ1[...] = 1/S # uniform by default
    PZ1[-1, -1, :, :] = (0.7 * np.eye(S) + 0.15) / (0.7 + 0.15 * S)
    # except that if both agents are listening then one can get the correct
    # door with prob 0.85, and wrong result with prob 0.15

    PZ2 = PZ1.copy()
    PZ = PZ1.reshape((A1, A2, S, Z1, 1)) * PZ2.reshape((A1, A2, S, 1, Z2))

    # reward
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

    # print(r[0])
    # print(r[1])

    return DelaySharingCPOMDP(PT, PZ, r, d, discount, b0)


def main():
    Model = DecTigerCPOMDP(S=2, d=1)
    if len(sys.argv) > 1:
        timeout = float(sys.argv[1])
    else:
        timeout = np.inf
    CoordinatorsHeuristicSearchValueIteration(
        Model, presolvetime=30, timeout=timeout
    ) # presolvecalllimit=161


if __name__ == '__main__':
    main()