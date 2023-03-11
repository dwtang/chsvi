"""Testing Coordinator's HSVI on DecTiger with delayed information sharing

"""

import numpy as np
import sys
from numpy.random import default_rng

from chsvi.models.delaysharing import DelaySharingCPOMDP
from chsvi.core import CHSVI
import chsvi.presolve


def DecTigerCPOMDP(S=2, d=2, discount=0.95):
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
    tiger1 = DecTigerCPOMDP(S=2, d=1, discount=0.9)
    tiger2 = DecTigerCPOMDP(S=2, d=2, discount=0.9)
    
    Smat = np.zeros((tiger1.S, tiger2.S), dtype=bool)
    for s2 in range(tiger2.S):
        state, midx2 = np.unravel_index(s2, shape=tiger2.Sdims)
        m0, m1 = tiger2.Allmpair[midx2]
        midx1 = tiger1.Allmpairlookup[(m0[1:], m1[1:])]
        s1 = np.ravel_multi_index((state, midx1), tiger1.Sdims)
        Smat[s1, s2] = 1

    presolveres = chsvi.presolve.FullInfoHSVI(
        tiger1, timeout=8, calllimit=np.inf
    )
    Solver1 = CHSVI(tiger1, presolveres=presolveres)
    Solver1.Solve()

    Solver2 = CHSVI(tiger2, presolveres=(Solver1.VU, Smat))
    Solver2.Solve()


def main1():
    tiger1 = DecTigerCPOMDP(S=2, d=1, discount=0.95)
    tiger1.negate()

    presolveres = chsvi.presolve.FullInfoHSVI(
        tiger1, timeout=8, calllimit=2000
    )
    Solver1 = CHSVI(tiger1, presolveres=presolveres)
    Solver1.Solve()

if __name__ == '__main__':
    main()
