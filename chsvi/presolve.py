import numpy as np
import scipy.sparse as spsp
import itertools

from chsvi.hsvi import POMDP, HeuristicSearchValueIteration
from chsvi.cpomdp import BaseCPOMDP


def FullInfoHSVI(Model, timeout=10, calllimit=np.inf):
    """Find full information POMDP upper bound for the value functions

    Input: 
        Model: a BaseCPOMDP instance with a relaxedPOMDP method that
        returns a POMDP model (if the relaxed POMDP is actually an
        MDP please use MDPUpperBound instead)
    """

    # Create a POMDP model assuming information are shared without delay
    info = "Presolving a full information POMDP with HSVI for upper bound initialization for "
    if timeout != np.inf:
        info += "{0} seconds".format(timeout)
        if calllimit != np.inf:
            info += "or {0} calls at root, whichever comes first".format(calllimit)
    elif calllimit != np.inf:
        info += "{0} calls at root".format(calllimit)
    
    print(info)

    FIModel, Amat = Model.relaxedPOMDP() # a pomdp
    res = HeuristicSearchValueIteration(FIModel, 
        timeout=timeout, calllimit=calllimit, ret="UB")

    res["A"] = Amat
    res["ymin"] = Amat.T @ res["Vmin"]

    return res # dict with keys "A", "BT", "vBar", "vBarVerts", "ymin"

def FullInfoMDP(Model):
    FIModel, Amat = Model.relaxedPOMDP() # could be a pomdp or mdp
    # value iteration
    Vub = ValueIteration(FIModel)
    Vlb = ValueIteration(FIModel, maximizing=False)
    
    res = {
        "A": Amat,
        "BT": np.zeros((0, FIModel.S)),
        "vBar": np.zeros((0)),
        "vBarVerts": Vub,
        "ymin": Amat.T @ Vlb
    }
    return res


def ValueIteration(MDP, maximizing=True):
    PT = MDP.PT
    if maximizing:
        V = MDP.Vmax * np.ones(MDP.S)
        optimize = lambda x : np.max(x, axis=-1)
    else:
        V = MDP.Vmax * np.ones(MDP.S)
        optimize = lambda x : np.min(x, axis=-1)
    while True:
        V1 = optimize(MDP.rPlusGammaTimes(PT @ V))
        if np.sum(np.abs(V1 - V)) < 1e-5:
            break
        V = V1
    return V



def FixedPrescriptionBound(Model, preset=None):
    """Find Fixed Prescription Lower Bound from a fixed, hand picked set
    of prescriptions (this can be bad... but much better than Vmin)

    Input: 
        Model: a DelaySharingCPOMDP instance with hint prescriptions given
        in the property DelaySharingCPOMDP.ghint
        ghint[j] the j th suggested prescription pair
        preset: a set of prescriptions to use, preset[j] is a tuple of I
        prescriptions, where preset[j][i] is an Model.M[i] vector
    """
    if preset is None:
        fixedactions = [
            [np.full(Model.M[i], ai, dtype=np.int32) for ai in range(Model.A[i])]
            for i in range(Model.I)
        ]
        preset = list(itertools.product(*fixedactions))

    num_pres = len(preset)


    shapes = [
        (np.prod((Model.S,) + Model.A[0:i]), Model.A[i], num_pres) 
        for i in range(Model.I)
    ]
    shapes.append((Model.S, num_pres))
    alp = [
        Model.Vmin * np.ones(shapes[i])
        for i in range(Model.I + 1)
    ]

    for j in range(num_pres):
        lin_idx = Model.SAIndex(preset[j])
        alpj = Model.Vmin * np.ones(Model.S)
        while True:
            V = np.array([alpj] * Model.O).flatten() # (O x S) order
            alpj1 = Model._r[lin_idx] + Model.discount * (Model._P[lin_idx] @ V)
            # Model._P is (S*A) x (O*S)
            gap = np.sum(np.abs(alpj1 - alpj))
            # print(gap)
            if gap < 1e-5:
                break
            alpj = alpj1

        alp[-1][:, j] = alpj
        alp[-2][:, :, j] = (
            Model._r + Model.discount * (Model._P @ V)
        ).reshape(alp[-2].shape[:-1])
        for i in range(Model.I - 1, 0, -1): # I-1, I-2, ..., 1
            alpstarij = np.swapaxes(
                alp[i][:, :, j].reshape((Model.S, -1, Model.A[i]))
                # S x (A0...Ai-1) x Ai
            , 1, 2).reshape((Model.S * Model.A[i], -1))
            # S x Ai x (A0...Ai-1) reshape to (S*Ai) x (A0...Ai-1)
            lin_idx_next = Model.SAi_Index(preset[j][i], i)
            alp[i-1][:, :, j] = alpstarij[lin_idx_next].reshape(alp[i-1].shape[:-1]) 
            # S x (A0...Ai-1) -> SA0...Ai-1
            
    return alp


class MoreInfoCPOMDP(BaseCPOMDP):
    def __init__(self, CPOMDP, Omap):
        assert(Omap.size == CPOMDP.S)
        newO = np.max(Omap) + 1

        _P = spsp.coo_matrix(CPOMDP._P)
        newP = [None for no in range(newO)]
        # Masks = [spsp.lil_matrix(_P.shape, dtype=bool) for no in range(newO)]
        # for no in range(newO):
        #     for o in range(CPOMDP.O):
        #         Masks[no][:, o * CPOMDP.S + partition[no]] = 1

        newcol = [Omap[c % CPOMDP.S] * _P.shape[1] + c for c in _P.col]
        _newP = spsp.coo_matrix((_P.data, (_P.row, newcol)), shape=_P.shape)

        self.S = CPOMDP.S
        self.I = CPOMDP.I
        self.A = CPOMDP.A
        self.AT = CPOMDP.AT
        self.SA = CPOMDP.SA
        self.M = CPOMDP.M
        self.Mmap = CPOMDP.Mmap
        self.Mmat = CPOMDP.Mmat
        self.discount = CPOMDP.discount
        self.Vmax = CPOMDP.Vmax
        self.Vmin = CPOMDP.Vmin
        self._r = CPOMDP._r
        self.b0 = CPOMDP.b0
        
        self._P = spsp.csr_matrix(_newP)
        self.O = newO * CPOMDP.O
