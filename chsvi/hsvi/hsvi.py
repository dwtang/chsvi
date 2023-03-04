"""POMDP Solver
"""

import numpy as np
from numpy.random import default_rng
import time
import gurobipy as grb

class OptimizationError(Exception):
    """Raised when gurobi got some internal error that 
    can happen, although rarely
    """
    pass

def FastInformedBound(M, maximizing=True):
    """Fast Informed Bound for Maximizing or Minimizing POMDP
    Input:
        M: a pomdp model 
        maximizing: bool
    Output:
        alp: S x A representing A alpha-vectors
        if maximizing, the convex function induced by these alpha-vectors 
        (i.e. f(b) = max(b @ alp)) form an upper bound for the solution.
        if minimizing, the concave function induced by these alpha-vectors 
        (i.e. f(b) = min(b @ alp)) form a lower bound for the solution 
    """
    tol = min(1.0, M.Vmax - M.Vmin) * 1e-6
    if maximizing:
        alp = M.Vmax * np.ones((M.S, M.A))
        optimize = lambda x : np.max(x, axis=-1)
    else:
        alp = M.Vmin * np.ones((M.S, M.A))
        optimize = lambda x : np.min(x, axis=-1)

    while True:
        tmp = optimize(M.Pmult(alp))  
        # M.Pmult(alp): S x A x O x A', tmp: S x A x O
        alp1 = M.rPlusGammaTimes(np.sum(tmp, axis=-1))
        if np.max(np.abs(alp1 - alp)) < tol:
            break
        alp = alp1
    return alp

def FixedActionBound(M):
    """Fixed Action Lower Bound for Maximizing POMDP
    Input:
        M: a pomdp model 
    Output:
        alp: S x A representing A alpha-vectors. The convex function induced 
        by these alpha-vectors (i.e. f(b) = max(b @ alp)) form a lower bound
        for the solution.
    """
    tol = min(1.0, M.Vmax - M.Vmin) * 1e-6
    alp = M.Vmin * np.ones((M.S, M.A))
    PT = M.PT

    while True:
        alp1 = M.rPlusGammaTimes(np.sum(PT * alp.T, axis=-1))
        if np.max(np.abs(alp1 - alp)) < tol:
            break
        alp = alp1
    return alp

class UpperBound():
    def __init__(self, M, t0=time.time()):
        self.M = M
        self.t0 = t0
        initalp = FastInformedBound(M) # S x A 
        rng = default_rng(777)
        N = self.M.S * 2
        # finding the vertices of the convex hull formed by alpha vectors 
        # can be expensive especially for large state space and action space
        # do some random sampling in the belief simplex instead
        # row vectors of beliefs
        self.BT = rng.dirichlet(np.ones(self.M.S), size=N)
        # vector of upper bounds
        self.vBar = np.max(self.BT @ initalp, axis=-1) 
        # bound at vertices
        self.vBarVerts = np.max(initalp, axis=-1) 

        # set up an LP model
        self.setuplp()

        # stats
        self.prevnumpoints = self.numpoints
        self.timespentonlp = 0.0
        self.timereorg = 0.0
        self.timeupdate = 0.0
        self.numlps = 0

    def setuplp(self):
        minalp = FastInformedBound(self.M, maximizing=False)
        self.Vmin = np.min(minalp, axis=-1)
        self._lp = grb.Model("upper bound")
        self._y = self._lp.addMVar(shape=(self.M.S,), vtype=grb.GRB.CONTINUOUS,
                                   lb=-np.inf, name="y")
        self._lp.addConstr(self._y >= self.Vmin)
        self._constr_ub = self._lp.addConstr(self._y <= self.vBarVerts)
        self._constr = self._lp.addConstr(self.BT @ self._y <= self.vBar)

    def update(self, b):
        self.M.setbelief(b)
        vbprime = np.zeros((self.M.A, self.M.O))
        tb0 = time.time()
        for a in range(self.M.A):
            for o in range(self.M.O):
                next_b = self.M.nextbelief(a, o)
                if next_b is not None:
                    vbprime[a, o] = self.access_lp(next_b)
        qb = self.M.Q(vbprime)
        ab = np.argmax(qb)
        vb = qb[ab]
        self.timeupdate += time.time() - tb0

        # Add point (b, vb) to the mix
        self.BT = np.concatenate((self.BT, [b]))
        self.vBar = np.append(self.vBar, vb)
        self.__reset_constr()
        if self.numpoints > 1.2 * self.prevnumpoints + 100:
            t00 = time.time()
            self.reorganize()
            self.timereorg += time.time() - t00
        return vb, ab

    def __getitem__(self, b):
        # res = self.access_sawtooth(b)
        res = self.access_lp(b)
        return res

    def access_sawtooth(self, b):
        # At this time and age, believe it or not
        # This function is actually slower than Gurobi LP
        vb = np.inf
        for i in range(self.BT.shape[0]):
            btilde = self.BT[i]
            idx = btilde > 0
            coef = np.min(b[idx] / btilde[idx])
            vb1 = coef * self.vBar[i] + (b - coef * btilde) @ self.vBarVerts
            if vb1 < vb:
                vb = vb1
        return vb

    def access_lp(self, b):
        try:
            t00 = time.time()
            self._lp.setObjective(b @ self._y, grb.GRB.MAXIMIZE)
            self._lp.optimize()
            res = self._lp.ObjVal
            self.timespentonlp += time.time() - t00
            self.numlps += 1
        except (AttributeError, grb.GurobiError) as e:
            raise OptimizationError(e + " LP status: {0}".format(
                self._lp.Status
            ))
        else: # no error
            return res

    def reorganize(self):
        # print("[{0:.3f}s] Prev numpoints {1}, current {2}".format(time.time() - self.t0, self.prevnumpoints, self.numpoints))
        # print("[{0:.3f}s] Reorganizing UpperBound...".format(time.time() - self.t0))
        for s in range(self.M.S):
            self._lp.setObjective(self._y[s], grb.GRB.MAXIMIZE)
            self._lp.optimize()
            self.vBarVerts[s] = self._lp.ObjVal 
            self.__reset_constr_verts()
        
        keepdix = [True] * self.BT.shape[0]
        for i in range(self.BT.shape[0]):
            keepdix[i] = False
            self._lp.remove(self._constr)
            self._constr = self._lp.addConstr(
                self.BT[keepdix] @ self._y <= self.vBar[keepdix]
            )
            self._lp.setObjective(self.BT[i] @ self._y, grb.GRB.MAXIMIZE)
            self._lp.optimize()
            # print("Belief: {0}, Value: {1}, New Value: {2}".format(self.BT[i], self.vBar[i], self._lp.ObjVal))
            if self.vBar[i] < self._lp.ObjVal - 1e-6:
                # This means that without BT[i], the upper bound here would be 
                # higher so it is better to keep BT[i] in
                keepdix[i] = True

        self.BT = self.BT[keepdix]
        self.vBar = self.vBar[keepdix]
        self.__reset_constr()
        # print("[{0:.3f}s] Reducing numpoints to {1}".format(time.time() - self.t0, self.numpoints))
        self.prevnumpoints = self.numpoints

    def __reset_constr(self):
        self._lp.remove(self._constr)
        self._constr = self._lp.addConstr(self.BT @ self._y <= self.vBar)

    def __reset_constr_verts(self):
        self._lp.remove(self._constr_ub)
        self._constr_ub = self._lp.addConstr(self._y <= self.vBarVerts)

    @property
    def numpoints(self):
        return self.BT.shape[0]

    def output(self):
        self.reorganize()
        return {"BT": self.BT, "vBar": self.vBar,
                "vBarVerts": self.vBarVerts, "Vmin": self.Vmin}


class LowerBound():
    def __init__(self, M, t0=time.time()):
        self.M = M
        self.t0 = t0
        # self.alp = M.Vmin * np.ones((self.M.S, 1))
        self.alp = FixedActionBound(self.M)
        # self.alp is a S x V matrix whose column vectors are alpha vectors
        self.alpao = self.M.Pmult(self.alp) 
        # self.alpao is S x A x O x V
        self.prevnumalp = self.alp.shape[1]
        self.timereorg = 0.0
        self.timebackup = 0.0

    def update(self, b):
        """Point-based backup
        """
        tb0 = time.time()
        bTalphaao = np.tensordot(b, self.alpao, axes=1) # A x O x V
        alpidx = np.argmax(bTalphaao, axis=-1) # A x O indices in range(V)
        Vb1 = np.zeros((self.M.S, self.M.A))
        for a in range(self.M.A):
            Vb1[:, a] = sum(
                self.alpao[:, a, o, alpidx[a, o]] for o in range(self.M.O)
            )
        alphaab = self.M.rPlusGammaTimes(Vb1) # S x A
        astar = np.argmax(b @ alphaab)
        newalp = alphaab[:, astar]

        self.alp = np.c_[self.alp, newalp]
        newalpao = self.M.Pmult(newalp) # S x A x O
        self.alpao = np.concatenate(
            (self.alpao, np.expand_dims(newalpao, axis=-1)),
            axis=-1
        )
        self.timebackup += time.time() - tb0

        if self.numalp > 2 * self.prevnumalp + 50:
            t00 = time.time()
            self.reorganize()
            self.timereorg += time.time() - t00
        return b @ newalp

    def __getitem__(self, b):
        return np.max(b @ self.alp)

    def reorganize(self):
        """Only alpha vectors that are point-wise dominated are pruned
        """
        # print("[{0:.3f}s] Prev numalp {1}, current {2}".format(time.time() - self.t0, self.prevnumalp, self.numalp))
        # print("[{0:.3f}s] Reorganizing LowerBound...".format(time.time() - self.t0))    
        order = np.argsort(self.alp[0, :])
        self.alp = self.alp[:, order]
        self.alpao = self.alpao[:, :, :, order]
        keepdix = [True] * self.alp.shape[1]
        for v in range(self.alp.shape[1]):
            for v1 in range(v + 1, self.alp.shape[1]):
                if np.all(self.alp[:, v] <= self.alp[:, v1]):
                    keepdix[v] = False
                    break

        self.alp = self.alp[:, keepdix]
        self.alpao = self.alpao[:, :, :, keepdix]
        # print("[{0:.3f}s] Reducing numalp to {1}".format(time.time() - self.t0, self.numalp))
        self.prevnumalp = self.alp.shape[1]
    
    @property
    def numalp(self):
        return self.alp.shape[1]

    def output(self):
        self.reorganize()
        return self.alp


class HSVI():
    def __init__(self, M, epsilon=None, t0=time.time()):
        self.M = M
        self.VU = UpperBound(self.M, t0)
        self.VL = LowerBound(self.M, t0)
        self.b0 = self.M.b
        self.anytime = (epsilon is None)
        self.epsilon = 0.95 * (
            self.VU[self.b0] - self.VL[self.b0]
        ) if self.anytime else epsilon
        # self.epsilon = 1
        self.pathcount = 0
        self.nodecount = 0
        self.t0 = t0
        self.timeheuristic = 0.0
        # self.ee = 1.0 / (np.log(self.M.O) / np.log(self.M.discount) - 1)
        self.ee = 0 # a hyperparameter dictating next gap selection
        # print("EE param:", self.ee)
        # the size of the next gap is proportional to (P(o|b, a*))^ee
        # between (-1, 0]. 0 is HSVI as it was presented in the paper
        # ee < 0 means that we allow larger gaps for smaller 
        
    def Solve(self, timeout=np.inf, calllimit=np.inf):
        gap = np.inf
        while gap > self.epsilon:
            try:
                self.Explore(self.b0, self.epsilon)
                self.pathcount += 1
                vu, vl = self.VU[self.b0], self.VL[self.b0]
                gap = vu - vl
                if self.anytime:
                    self.epsilon = 0.95 * gap
                print(
                    ("[{0:.3f}s] {1} calls, LB: {2:.3f}, UB: {3:.3f}," + 
                    " gap: {4:.3f}").format(
                    time.time() - self.t0, self.pathcount, vl, vu, gap)
                )
                print(
                    "[{0:.3f}s] {1} LB alpha vectors, {2} UB points".format(
                        time.time() - self.t0, self.VL.numalp, self.VU.numpoints
                    )
                )
                if time.time() - self.t0 > timeout or self.pathcount >= calllimit:
                    print("[{0:.3f}s] Timeout! Terminating Algorithm".format(
                        time.time() - self.t0
                    ))
                    break
            except OptimizationError as e:
                print(
                    "[{0:.3f}s] OptimizationError: {1}, ".format(
                    time.time() - self.t0, e), "Start Exploring Over"
                )
            except KeyboardInterrupt:
                print("User KeyboardInterrupt. Terminating Algorithm")
                break
        
        self.printstats()

        
    def printstats(self):
        vl, vu = self.LB(), self.UB()
        t1 = time.time()
        print("Value lower bound: {0:.6f}, upper bound: {1:.6f}, gap: {2:.6f}".format(
            vl, vu, vu - vl))
        print("Explore path count: {0}".format(self.pathcount))
        print("Explore node count: {0}".format(self.nodecount))
        print("Algorithm Time: {0:.3f}s".format(t1 - self.t0))
        print("Time spent on Upper Bound Access: {0:.3f}s".format(
            self.VU.timespentonlp)
        )
        print("Number of Upper Bound Accesses: {0}".format(self.VU.numlps))
        print("Time spent on Upper Bound Update: {0:.3f}s".format(
            self.VU.timeupdate)
        )
        print("Time spent on Upper Bound Reorg: {0:.3f}s".format(
            self.VU.timereorg)
        )
        print("Time spent on Lower Bound Backup: {0:.3f}s".format(
            self.VL.timebackup)
        )
        print("Time spent on Lower Bound Reorg: {0:.3f}s".format(
            self.VL.timereorg)
        )
        print("Time spent on Search Heuristic: {0:.3f}s".format(
            self.timeheuristic)
        )
        

    def Explore(self, b, eps):
        self.nodecount += 1
        
        vub, astar = self.VU.update(b)
        vlb = self.VL.update(b)
        gap = vub - vlb

        tb0 = time.time()

        # Displaying Information
        # depth = round(np.log(eps / self.epsilon) / np.log(self.M.discount))
        # print("[{0:.3f}s] Exploring belief {1}, gap: {2:.4f}, target: {3:.4f}".format(time.time() - self.t0, _print_belief(b), gap, eps))
        
        if gap <= eps:
            return
        
        self.M.setbelief(b)
        beliefs = [self.M.nextbelief(astar, o) for o in range(self.M.O)]
        excessvec = np.array([0 if (b1 is None) else self.VU[b1] - self.VL[b1]
            for b1 in beliefs]) - eps / self.M.discount

        score = self.M.bPO[astar] * excessvec
        ostar = np.argmax(score)

        self.timeheuristic += time.time() - tb0

        if score[ostar] <= 0:
            # It is possible (although rare) that after the above updates,
            # none of the excess gap is positive (since the upper bound 
            # update at b can result in tighter upper bounds at b')
            self.VU.update(b)
            self.VL.update(b)
            return
        
        # next_eps = eps / self.M.discount
        next_eps = (eps / self.M.discount) * (self.M.bPO[astar, ostar] ** self.ee) / np.sum(self.M.bPO[astar] ** (self.ee + 1))

        self.Explore(beliefs[ostar], next_eps)
        # print("[{0:.3f}s] Post return update at t={1}".format(time.time() - self.t0, depth))
        self.VU.update(b)
        self.VL.update(b)

    def UB(self, b=None):
        if b is None:
            b = self.b0
        return self.VU[b]

    def LB(self, b=None):
        if b is None:
            b = self.b0
        return self.VL[b]


def _print_belief(b):
    pos = b > 1e-6
    if np.sum(pos) <= 6:
        s = "["
        s += ", ".join(
            "{0}: {1:.3f}".format(j, b[j]) 
            for j in range(b.size) if pos[j]
        )
        s += "]"
        return s
    else:
        return "[...]"


def HeuristicSearchValueIteration(M, epsilon=None, timeout=np.inf,
                                  calllimit=np.inf, ret="LB"):
    """Heuristic Search Value Iteration Algorithm

    Input:
        M, a pomdp model that should have the following fields:
            M.S: number of states
            M.A: number of actions
            M.O: number of observations
            M.discount: discount factor
            M.b: initial/current belief
            M.Vmax: upper bound for the value of any policy
            M.Vmin: lower bound for the value of any policy
        it should also implement the following methods:
            M.PT: S x A x S' transition kernel
            M.Pmult(v)
            M.rPlusGammaTimes(v)
            M.setbelief(b)
            M.nextbelief(a, o)
            M.Q(vbprime)

        epsilon: desired gap (set to None for anytime HSVI)
        timeout: time limit in seconds
    """
    t0 = time.time()
    Solver = HSVI(M, epsilon, t0)
    Solver.Solve(timeout, calllimit)
    if ret == "UB":
        return Solver.VU.output()
    else:
        return Solver.VL.output()