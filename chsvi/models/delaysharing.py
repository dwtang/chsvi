"""Delayed Sharing POMDP Class

"""

import numpy as np
import scipy.sparse as spsp
import itertools

from chsvi.cpomdp import BaseCPOMDP
from chsvi.hsvi import POMDP

class DelaySharingCPOMDP(BaseCPOMDP):
    """Delayed Sharing Coordinator's POMDP

    Private observations and actions are shared with all other agents with a
    delay of d

    Property:
        ghint: list of tuples (g1, g2), g1 is an M1 array representing 
        prescription for player 1

    """
    def __init__(self, PT, PZ, r, d, discount, b0=None, ghint=None):
        """Initialize POMDP Model

        Input:
        PT: S x A0 x ... x A(I-1) x S' array
        PZ: A0 x ... x A(I-1) x S' x Z1 x Z2 array
        r: S x A0 x ... x A(I-1) numpy array
        d: Delay, positive integer
        discount: number in (0, 1)
        b0: None or S numpy array
        """
        S = PT.shape[0]
        I = len(PT.shape) - 2
        A = PT.shape[1:I+1]
        Z = PZ.shape[-I:]
        if b0 is None:
            b0 = np.ones(S) / S

        # create augmented state, private state, and common observation 
        # representations
        
        Odims = tuple(Z[i] * A[i] for i in range(I)) 
        # the private observation and action d steps ago
        O = np.prod(Odims) + 1 # adding a symbol for "none"

        Midims = []
        for i in range(I):
            Midims.append([(Z[i] * A[i],) * k for k in range(1, d+1)])
        
        # generate all possible private states
        Allm = [None] * I
        for i in range(I):
            Allm[i] = [[(-1,) * d]]
            for dims in Midims[i]:
                khistlist = list(itertools.product(*(range(k) for k in dims)))
                prefix = (-1,) * (d - len(khistlist[0]))
                Allm[i].append([(*prefix, *m) for m in khistlist])

        # print(Allm[0])

        # generate all possible joint private states
        Allmpair = []
        for k in range(d+1):
            Allmpair.extend(itertools.product(*(Allm[i][k] for i in range(I))))

        # print("Allmpair:")
        # print("\n".join([str(it) for it in Allmpair]))

        # Now we can flatten Allm
        for i in range(I):
            Allm[i] = list(itertools.chain(*Allm[i]))
        
        MT = len(Allmpair)
        M = tuple(len(Allmi) for Allmi in Allm)
        Sbar = S * MT
        
        # create the extended r, b0
        rbar = np.zeros((S, MT, *A))
        b0bar = np.zeros((S, MT))
        rbar[...] = np.expand_dims(r, axis=1)
        b0bar[:, 0] = b0 # initially the private information is Allmpair[0]
        rbar = rbar.reshape((Sbar, *A))
        b0bar = b0bar.flatten()

        Allapair = list(itertools.product(*(range(a) for a in A)))
        Allzpair = list(itertools.product(*(range(z) for z in Z)))
        Allmpairlookup = {Allmpair[m]: m for m in range(MT)}
        Allmlookup = [
            {Allm[i][mi]: mi for mi in range(M[i])} for i in range(I)
        ]

        # create Mmap, nextMmap, nextOmap
        Mmap = np.zeros((I, S, MT), dtype=np.int32)
        nextMmap = np.empty((np.prod(M), *A, *Z), dtype=np.int32)
        nextOmap = np.empty((np.prod(M),), dtype=np.int32)
        for m_idx in range(MT):
            m = Allmpair[m_idx]
            for i in range(I):
                Mmap[i, :, m_idx] = Allmlookup[i][m[i]]
            o = tuple(m[i][0] for i in range(I))
            if o[0] == -1:
                nextOmap[m_idx] = 0 # first common observation is "none"
            else:
                nextOmap[m_idx] = np.ravel_multi_index(o, Odims) + 1
            for a in Allapair:
                for z in Allzpair:
                    za = tuple(z[i] * A[i] + a[i] for i in range(I))
                    next_m = tuple(m[i][1:] + (za[i],) for i in range(I))
                    next_m_idx = Allmpairlookup[next_m]
                    nextMmap[(m_idx, *a, *z)] = next_m_idx

        Mmap = Mmap.reshape((I, Sbar))

        # create P matrix
        Pbar = np.full((S, MT, *A), None)
        for s, m, a in itertools.product(range(S), range(MT), Allapair):
            TZ = spsp.lil_matrix((O, Sbar))
            next_o = nextOmap[m]
            for next_s, z in itertools.product(range(S), Allzpair):
                pr = PT[(s, *a, next_s)] * PZ[(*a, next_s, *z)]
                if pr > 0:
                    next_m = nextMmap[(m, *a, *z)]
                    TZ[next_o, next_s * MT + next_m] = pr
            Pbar[(s, m, *a)] = TZ
        
        Pbar = Pbar.reshape((Sbar, *A))
        super().__init__(Pbar, Mmap, rbar, discount, b0bar)
        self.original_params = {
            "S": S,
            "A": A,
            "Z": Z,
            "PT": PT,
            "PZ": PZ,
            "r": r,
            "d": d,
            "b0": b0,
        }
        self.Sdims = (S, MT)
        self.Odims = Odims
        self.Allm = Allm
        self.Allmpair = Allmpair
        self.ghint = ghint


    def get_sbar_idx(self, smtuple):
        return np.ravel_multi_index(smtuple, self.Sdims)

    def get_sbar_tuple(self, sbar):
        s, m_idx = np.unravel_index(sbar, self.Sdims)
        res = (s,) + self.Allmpair[m_idx]
        return res

    def get_m_tuple(self, m, i):
        return self.Allm[i][m]

    def get_o_tuple(self, o):
        if o == 0:
            return (-1, -1)
        else:
            return np.unravel_index(o-1, self.Odims)

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
        S = self.original_params["S"]
        A = self.AT # an action pair is considered an action
        O = np.prod(self.original_params["Z"]) # observations are common now
        PT = self.original_params["PT"].reshape((S, A, S))
        PZ = self.original_params["PZ"].reshape((A, S, O))
        r = self.original_params["r"].reshape((S, A))
        discount = self.discount
        b0 = self.original_params["b0"]

        # generate the 0-1 S x Sbar state aggregation matrix 
        Amat = np.zeros((S, *self.Sdims), dtype=bool)
        Amat[...] = np.eye(S, dtype=bool)[:, :, np.newaxis]
        Amat = Amat.reshape((S, -1))

        return POMDP((PT, PZ), r, discount, b0), Amat