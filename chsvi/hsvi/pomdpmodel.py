"""POMDP Model Class
"""

import numpy as np

class POMDP():
    def __init__(self, P, r, discount, b0=None):
        """Initialize POMDP Model
        P: S x A x O x S' numpy array or tuple 
           (S x A x S' numpy array, A x S' x O numpy array)
        r: S x A or S x A x O x S' numpy array
        discount: number in (0, 1)
        b0: None or S numpy array

        Creates:
        self.P: S x A x O x S' numpy array
        self.r: S x A numpy array
        self.discount: number in (0, 1) discount factor
        self.Vmax, self.Vmin: cude upper and lower bounds for values
        
        self.b: S numpy array, current belief
        self.bPO: A x O matrix of P(o|self.b, a)
        """
        if isinstance(P, tuple):
            self.S, self.A, self.O = P[0].shape[0], P[0].shape[1], P[1].shape[2]
            assert(P[1].shape[0] == self.A)
            assert(P[1].shape[1] == self.S)
            self.P = np.zeros((self.S, self.A, self.S, self.O))
            for s in range(self.S):
                self.P[s] = P[0][s, :, :, np.newaxis] * P[1] 
                # A x S' x 1 mults A x S' x O so that numpy broadcasting applies
            self.P = np.swapaxes(self.P, 2, 3) 
            # switch from S x A x S' x O to S x A x O x S'
        else:
            self.S, self.A, self.O = P.shape[0], P.shape[1], P.shape[2]
            self.P = P
        self.PO = np.sum(self.P, axis=-1) # S x A x O
        if len(r.shape) == 2:
            self.r = r
        elif len(r.shape) == 4:
            self.r = np.sum(self.P * r, axis=(2,3)) # convert to expected reward 
        else:
            raise ValueError("Dimension of r should be 2 or 4")

        self.discount = discount
        self.Vmin = np.min(self.r) / (1 - self.discount)
        self.Vmax = np.max(self.r) / (1 - self.discount)
        if b0 is None:
            self.setbelief(1 / self.S * np.ones(self.S))
        else:
            assert(b0.shape == (self.S,))
            assert(np.sum(b0) == 1.0)
            self.setbelief(b0)

    @property
    def PT(self):
        return np.sum(self.P, axis=2)

    def Pmult(self, v):
        """Compute w(s, a, o) = sum_{s'} P(s', o|s, a) v(s', :)
        Input:
            v: S' x whatever
        Output:
            w: S x A x O x whatever 
        """
        return self.P @ v

    def rPlusGammaTimes(self, v):
        """Compute w(s, a) = r(s,a) + discount * v 
        Input:
            v: S x A matrix
        Output:
            w: S x A matrix
        """
        return self.r + self.discount * v
    

    def setbelief(self, b):
        self.b = b
        self.br = self.b @ self.r # A vector
        self.bPOS = np.tensordot(self.b, self.P, axes=1) # A x O x S
        self.bPO = np.sum(self.bPOS, axis=-1) # A x O

    def nextbelief(self, a, o):
        if self.bPO[a, o] == 0:
            return None
        return self.bPOS[a, o, :] / self.bPO[a, o]


    def Q(self, vbprime):
        """Compute the expected total reward for each action given the future
        value function
        Input:
            vbprime: A x O matrix
        Output:
            q: A vector
        """
        return self.br + self.discount * np.sum(self.bPO * vbprime, axis=-1)

    def tao(self, b, a, o):
        """Belief update function
        Input:
            b: S vector
        Output:
            b1: S vector
        """
        bo = b @ self.P[:, a, o, :] # S vector, not normalized
        bo /= np.sum(bo)

        return bo

    def negate(self):
        self.r = -self.r
        self.Vmin = np.min(self.r) / (1 - self.discount)
        self.Vmax = np.max(self.r) / (1 - self.discount)