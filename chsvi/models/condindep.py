"""Control Sharing Model

Each agent has a controlled Markov Chain. Actions are public.
See A. Mahajan, “Optimal decentralized control of coupled subsystems
with control sharing,” IEEE Trans. Automat. Contr., vol. 58, no. 9, pp.
2377–2382, 2013.

"""

import numpy as np
import scipy.sparse as spsp
import itertools

from chsvi.cpomdp import BaseCPOMDP
from chsvi.hsvi import POMDP

class CondIndepCPOMDP(BaseCPOMDP):
    """Conditionally independent coordinator's POMDP Model
    """

    def __init__(self, S, A, PT, r, discount, b0):
        """
        Input:
            S: tuple of 2 integers, denoting the size of state spaces
            A: tuple of 2 integers, denoting the size of action spaces
            PT: list of numpy matrices Si x (Atuple) x Si
            r: S tuple x A tuple numpy array
            b0: list of Si vector
        """
        I = len(S)
        Sbar = np.prod(S)
        O = np.prod(A)
        P = np.empty((Sbar, *A), dtype=object)
        b0bar = np.zeros(Sbar)
        for s in range(Sbar):
            stuple = np.unravel_index(s, shape=S)
            b0bar[s] = np.prod([b0[i][stuple[i]] for i in range(I)])
            for a in itertools.product(*[range(Ai) for Ai in A]):
                TZ = spsp.lil_matrix((O, Sbar))
                o = np.ravel_multi_index(a, A)
                for s1 in range(Sbar):
                    s1tuple = np.unravel_index(s1, shape=S)
                    TZ[o, s1] = np.prod([
                        PT[i][(stuple[i], *a, s1tuple[i])]
                        for i in range(I)
                    ])
                P[(s, *a)] = TZ

        Mmap = np.zeros((I, *S), dtype=np.int32)
        Mmap[0, :, :] = np.arange(S[0])[:, np.newaxis]
        Mmap[1, :, :] = np.arange(S[1])[np.newaxis, :]
        Mmap = Mmap.reshape(I, Sbar)
        rbar = r.reshape((Sbar, *A))

        super().__init__(P, Mmap, rbar, discount, b0bar)
        self.original_params = {
            "S": S,
            "PT": PT,
            "r": r,
            "b0": b0
        }


    def relaxedPOMDP(self):
        """Returns the full information POMDP

        Output:
            0: a POMDP instance
            1: S x Sbar 0-1 matrix indicating the mapping from the state
               here to the state in the relaxed POMDP
               The value function for this CPOMDP at b can be upper bounded by
               the value function of relaxed POMDP evaluated at the belief
               btilde = (this matrix) @ b

        """
        PTOG = self.original_params["PT"]
        S = self.original_params["S"]
        PT = np.zeros((self.S, self.AT, self.S))
        for s_idx in range(self.S):
            stuple = np.unravel_index(s_idx, shape=S)
            for a_idx in range(self.AT):
                a = np.unravel_index(a_idx, shape=self.A)
                for s1_idx in range(self.S):
                    s1tuple = np.unravel_index(s1_idx, shape=S)
                    PT[s_idx, a_idx, s1_idx] = np.prod([
                        PTOG[i][(stuple[i], *a, s1tuple[i])]
                        for i in range(self.I)
                    ])

        PZ = np.zeros((self.AT, self.S, self.S))
        for a_idx in range(self.AT):
            PZ[a_idx, :, :] = np.eye(self.S)

        MDP = POMDP(P=(PT, PZ), 
            r=self.original_params["r"].reshape((self.S, -1)), 
            discount=self.discount
        )
        return MDP, None
