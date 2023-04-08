"""Testing Coordinator's HSVI on a multiagent broadcast channel model

"""

import numpy as np
import sys
from numpy.random import default_rng
import itertools

from chsvi.models.condindep import CondIndepCPOMDP
from chsvi.core import CHSVI
from chsvi.io import WriteCPOMDP
import chsvi.presolve

def MulticastCPOMDP(C, p):
    S = (C[0] + 1, C[1] + 1) # state: number of packets in buffer
    A = (2, 2) # action: 0: do not transmit, 1: transmit
    PT0 = np.zeros((S[0], *A, S[0]))
    PT1 = np.zeros((S[0], *A, S[0]))
    
    def Pidle(k):
        Pidle = (1.0 - p[k]) * np.eye(S[k]) + np.diag(np.full(S[k] - 1, p[k]), k=1)
        Pidle[-1, -1] = 1.0
        return Pidle

    def Psend(k):
        Psend = p[k] * np.eye(S[k]) + np.diag(np.full(S[k] - 1, 1.0 - p[k]), k=-1)
        Psend[0, 0] = 1.0
        return Psend

    PT0[:, 0, 0, :] = Pidle(0)
    PT0[:, 0, 1, :] = Pidle(0)
    PT0[:, 1, 0, :] = Psend(0)
    PT0[:, 1, 1, :] = Pidle(0)

    PT1[:, 0, 0, :] = Pidle(1)
    PT1[:, 0, 1, :] = Psend(1)
    PT1[:, 1, 0, :] = Pidle(1)
    PT1[:, 1, 1, :] = Pidle(1)

    r0s = -np.arange(S[0]) 
    # every packet staying in the buffer costs 1 per time unit
    r0s[-1] -= 2 * p[0] 
    # if a new packet arrives when buffer is full we impose a penalty of 2
    r1s = -np.arange(S[1])
    r1s[-1] -= 2 * p[1]

    rs = r0s[:, np.newaxis] + r1s[np.newaxis, :]
    r = np.zeros((*S, *A))
    for a0 in range(A[0]):
        for a1 in range(A[1]):
            r[:, :, a0, a1] = rs - 0.5 * (a0 + a1) # sending costs power

    PT = [PT0, PT1]
    discount = 0.9
    b0 = [np.zeros(S[k]) for k in range(2)]
    for k in range(2):
        b0[k][0] = 1.0 # initially buffers are empty
    return CondIndepCPOMDP(S, A, PT, r, discount, b0)

def main():
    C = (int(sys.argv[1]), int(sys.argv[2])) # buffer sizes
    p = (float(sys.argv[3]), float(sys.argv[4])) # packet arrival probability
    Model = MulticastCPOMDP(C, p)
    # filename = "MultiCast_{0}_{1}".format(C[0], C[1])
    # WriteCPOMDP(Model, filename)
    # return
    presolveres = chsvi.presolve.FullInfoMDP(Model)
    Solver = CHSVI(Model, presolveres=presolveres)
    Solver.Solve(timeout=86400, targetgap=0.01)


if __name__ == '__main__':
    main()