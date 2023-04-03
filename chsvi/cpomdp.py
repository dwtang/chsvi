"""Coordinator's POMDP Model

Input to the CHSVI algorithm.

To use this class, please either use the provided models in models folder
or inherit the base class and implement the new relaxedPOMDP method
"""

import numpy as np
import scipy.sparse as spsp


class BaseCPOMDP():
    """General Coordinator's POMDP Class

    Properties:
    S: number of augmented states
    I: number of agents
    A: tuple of number of actions for agents 
    AT: number of action vectors 
    SA: tuple (S, *A)
    M: tuple of number of private states agents
    O: number of common observations
    Mmap: I x S numpy array describing the mapping from augmented
                states to private states
    Mmat: list of I matrices, Mmat[i] = M[i] x S 0-1 csr matrix encoding Mmap[i]
    discount: number in (0, 1) discount factor
    Vmax, Vmin: crude upper and lower bounds for values
    
    _P: (S * AT) x (O * S') compressed sparse row (CSR) matrix
        (In the delay information sharing model, or in general, given
        (s, a), the distribution of (o, s') is quite sparse)
    _r: (S * AT) numpy array 

    Methods:
    P, Q, SAIndex, SAi_Index, setbelief, PObg, Qbg

    """
    def __init__(self, P, Mmap, r, discount, b0=None):
        """Initialize BaseCPOMDP
        Input:
            P: S x (A tuple) array of O x S' sparse array 
                P(s, *a, o, s') = Pr(s', o|s, a)
            Mmap: I x S array, the mapping from augmented state to private info
                Mmap[i, s] = m_s^i
                private info should be encoded into contiguous non-negative integers
                starting from 0, i.e. M^i = {0, 1, ..., M-1}
            r: S x (A tuple) array, instantaneous reward
            discount: scalar
            b0: S array or None
        """
        assert(P.shape == r.shape)
        self.S = P.shape[0]
        self.A = r.shape[1:]
        if b0 is not None:
            assert(b0.shape == (self.S,))
        
        self.AT = np.prod(self.A)
        self.I = len(self.A)
        self.SA = (self.S, *self.A)
        self.set_Mmap(Mmap)
        
        self._P = my_reshape(P)
        self.O = self._P.shape[1] // self.S
        self._r = r.flatten()
        self.discount = discount
        self.b0 = b0
        self.Vmin = np.min(self._r) / (1 - self.discount)
        self.Vmax = np.max(self._r) / (1 - self.discount)

    def set_Mmap(self, Mmap):
        self.M = tuple(np.max(Mmap, axis=1) + 1)
        self.Mmap = Mmap
        self.Mmat = [onehot(Mmap[i], self.M[i]) for i in range(self.I)]
        self.MmatT = [mat.T.toarray() for mat in self.Mmat]

    def P(self, Y):
        """Compute sum_o sum_{s'}P(s', o|s, a) Y(o. s') 
        Input:
            Y: S' x O array
        Output:
            w: S*A numpy array
        """
        return self._P @ Y.T.flatten()

    def Q(self, Y):
        """Compute r(s, a) + discount * sum_o sum_{s'}P(s', o|s, a) Y(o. s') 
        Input:
            Y: S' x O array
        Output:
            q: S*A numpy array
        """
        return self._r + self.discount * self.P(Y)
        
    def SAIndex(self, g):
        multi_idx = [
            (s, *(g[i][self.Mmap[i][s]] for i in range(len(g))))
            for s in range(self.S)
        ]
        return np.ravel_multi_index(tuple(zip(*multi_idx)), self.SA[:len(g)+1])

    def SAi_Index(self, g, i):
        multi_idx = [
            (s, g[self.Mmap[i][s]]) for s in range(self.S)
        ]
        return np.ravel_multi_index(tuple(zip(*multi_idx)), (self.S, self.A[i]))


    def setbelief(self, b, g):
        """Set current belief and prescription
        Input:
            b: S vector of belief
            g: tuples of integer vectors indicating prescriptions,
               g[i][m] = action agent i takes at private state m 
               g must be of length I, i.e. contains prescription for all agents

        Set:
            self.bgr: scalar, instantaneous reward
            self.bgP: O x S' numpy array, representing P(s', o|b, g)
        """

        lin_idx = self.SAIndex(g)
        self.bgr = b @ self._r[lin_idx]
        self.bgP = (b @ self._P[lin_idx]).reshape((self.O, self.S))
        self.bgPO = np.sum(self.bgP, axis=-1)

    def nextbelief(self, o):
        if self.bgPO[o] == 0:
            return None
        return self.bgP[o, :] / self.bgPO[o]

    def PObg(self, alp):
        """Compute v(o, l) = sum_{s'} P(s', o|b, g) alp(s', l)
        Input:
            alp: S' x V numpy array
        Output:
            v: O x V numpy array 
        """
        return self.bgP @ alp

    def Qbg(self, y):
        return self.bgr + self.discount * np.sum(y)

    def negate(self):
        self._r = -self.r
        self.Vmin = np.min(self._r) / (1 - self.discount)
        self.Vmax = np.max(self._r) / (1 - self.discount)

    def relaxedPOMDP(self):
        """Returns the full information POMDP

        Output: (Model, Smat)
            Model: a POMDP instance
            Smat: S x Sbar 0-1 matrix indicating the mapping from the state
               here to the state in the relaxed POMDP
               The value function for this CPOMDP at b can be upper bounded by
               the value function of relaxed POMDP evaluated at the belief
               btilde = (this matrix) @ b

        """
        raise NotImplementedError("Please implement relaxedPOMDP for BaseCPOMDP.")


def my_reshape(P):
    """reshape a numpy array of sparse array into a csr array"""
    P = P.flatten()
    P = [TZ.reshape(1, -1) for TZ in P]
    return spsp.csr_matrix(spsp.vstack(P))


def onehot(Mmap, M):
    mat = np.zeros((M, Mmap.size), dtype=np.int32)
    for s in range(Mmap.size):
        mat[Mmap[s], s] = 1
    return spsp.csr_matrix(mat)