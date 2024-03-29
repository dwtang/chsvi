"""Coordinator's POMDP Solver

"""

import numpy as np
from numpy.random import default_rng
import time

import chsvi.presolve
from chsvi.optmodel import UBOptimizationModel, OptimizationError


class UpperBound():
    """CPOMDP Upper Bound Class
    
    Using three OptimizationModel object to deal with optimization problems
    """
    def __init__(self, Model, presolveres=None):
        self.Model = Model

        # presolve step
        if presolveres is None:
            presolveres = {
                "Smat": None,
                "LHS": np.concatenate((np.eye(Model.S), -np.eye(Model.S)), axis=0),
                "RHS": np.concatenate(
                    (np.full((Model.S, *Model.A), Model.Vmax),
                     -np.full((Model.S, *Model.A), Model.Vmin)),
                    axis=0
                )
            }
        # return a point-based upper bound on the for certain
        # aggregated belief points (i.e. the belief on aggregated state)
       
        # set up the LP and BLP models
        self._p = [
            UBOptimizationModel(self.Model, j, presolveres)
            for j in range(self.Model.I + 1)
        ] 

        # stats
        self.timeupdate = 0.0


    def update(self, b, g):
        """Update at belief
        (b, g): the tuple as a whole represents a belief
        b: S vector representing belief
        if at stage 0, g = ()
        if at stage 1, g = (g0)
        if at stage 2, g = (g0, g1)
            
        """
        tb0 = time.time()
        stage = len(g)
        lin_idx = self.Model.SAIndex(g)
        self.Model.setbelief(b, g)
        blp = self._p[stage]

        if stage < self.Model.I:
            yg = blp.y.reshape((-1, self.Model.A[stage]))[lin_idx, :] 
            # yg is an S x Ai array representing 
            # y(s, g0(m_s^0), ... , gi-1(m_s^i-1), ai)
            bMmat = b[:, np.newaxis] * self.Model.MmatT[stage] # S x Mi
            objfunc = ((bMmat @ blp.g) * yg).sum()
            vb = blp.solve(objfunc)
            gb = blp.gsol()
        else: # stage == I
            vO = np.zeros(self.Model.O, dtype=np.float64)
            for o in range(self.Model.O):
                if self.Model.bgPO[o] > 0:
                    vO[o] = blp.solve(self.Model.bgP[o] @ blp.y)
            vb = self.Model.Qbg(vO)
            gb = None
        
        self.timeupdate += time.time() - tb0
        
        # update constraints
        prevblp = self._p[stage-1]
        prevblp.addLEConstr(b @ prevblp.y[lin_idx], vb)

        return vb, gb

    def __getitem__(self, b):
        return self._p[-1].solve(b @ self._p[-1].y)

    @property
    def numpoints(self):
        return [p.numconstr for p in self._p]

    def transfer(self, Model, Smat=None):
        """Transfer to new model with larger state space

        Model: new CPOMDP model that has fewer common info than the old one
        Smat: S_old x Model.S 0-1 mapping matrix
        """
        self.Model = Model
        if Smat is not None:  
            for blp in self._p:
                blp.transfer(Model, Smat)
        self.timeupdate = 0


class LowerBound():
    """Lower Bound Class for CHSVI

    Properties:
    alp: list of I+1 numpy arrays, alp[i] is the alpha-vector for the
        belief (on S x A0 ...x Ai). alp[-1] is the alpha-vector for belief at
        stage 0 (i.e. the regular belief on S)
        for i < I, alp[i] is an (S * A0 * ... * Ai-1) x Ai x V array
        alp[I] is an S x V array
        At stage i, alp[i-1] is updated using alp[i]
    """

    def __init__(self, Model, presolvealp=None):
        self.Model = Model
        # shapes = [
        #     (np.prod((self.Model.S,) + self.Model.A[0:i]), self.Model.A[i], 1) 
        #     for i in range(self.Model.I)
        # ]
        # shapes.append((self.Model.S, 1))
        # self.alp = [
        #     self.Model.Vmin * np.ones(shapes[i])
        #     for i in range(self.Model.I + 1)
        # ]
        self.timereorg = 0.0
        if presolvealp is None:
            self.alp = chsvi.presolve.FixedPrescriptionBound(self.Model)
        else:
            self.alp = [alp.copy() for alp in presolvealp]
        self.numalp = [alp.shape[-1] for alp in self.alp]
        self.prevnumalp = [0 for alp in self.alp]
        for i in range(-1, self.Model.I):
            self.reorganize(i)

        self.timebackup = 0.0

    def update(self, b, g):
        """Point-based backup
        """
        tb0 = time.time()
        stage = len(g)
        lin_idx = self.Model.SAIndex(g)

        if stage < self.Model.I:
            # need to write highly vectorized code to ensure speed
            # at the sacrifice of readability...
            # print("alp:", self.alp[stage])
            balp = (b[:, np.newaxis, np.newaxis] * 
                self.alp[stage][lin_idx][:, :, 0:self.numalp[stage]])
            # print("balp:", balp) 
            # balp is an S x Ai x V array (V is the number of alpha-vectors)
            # balp[s, ai, alpha] = b(s)alpha(s, ai)
            mbalp = (self.Model.Mmat[stage] @ balp.reshape((self.Model.S, -1))
                ).reshape((self.Model.M[stage], self.Model.A[stage], -1))
            # print("mbalp", mbalp)
            # mbalp is an Mi x Ai x V array
            # mbalp[mi, ai, alpha] = sum_{s:m_s^i = mi} b(s)alpha(s, ai)
            Jalp = np.sum(np.max(mbalp, axis=1), axis=0) # V
            # print("Jalp:", Jalp)
            # Jalp[alpha] = max_{gi} sum_{s} b(s)alpha(s, gi(m_s^i))
            # ... = sum_{mi} max_{ai} sum_{s:m_s^i = mi} b(s)alpha(s, ai)
            idx = np.argmax(Jalp)
            vb = Jalp[idx]
            gnext = np.argmax(mbalp[:, :, idx], axis=1) # Argmax of Mi x Ai

            alpstar = self.alp[stage][..., idx] # (S*A0...Ai-1) x Ai
            alpstar1 = np.swapaxes(
                alpstar.reshape((self.Model.S, -1, self.Model.A[stage]))
                # S x (A0...Ai-1) x Ai
            , 1, 2).reshape((self.Model.S * self.Model.A[stage], -1))
            # S x Ai x (A0...Ai-1) reshape to (S*Ai) x (A0...Ai-1)
            lin_idx_next = self.Model.SAi_Index(gnext, stage)
            newalp = alpstar1[lin_idx_next] # S x (A0...Ai-1)

        else: # stage == self.Model.I
            self.Model.setbelief(b, g)
            alpo_idx = np.argmax(
                self.Model.PObg(self.alp[stage][:, 0:self.numalp[stage]]),
                axis=1
            )
            # alpo_idx[o] = index of best alpha-vector under observation o
            newalp = self.Model.Q(self.alp[stage][:, alpo_idx])
            vb = b @ newalp[lin_idx]

        newalp = newalp.reshape(self.alp[stage-1].shape[:-1])
        self.add(newalp, stage-1)
        self.timebackup += time.time() - tb0

        if self.numalp[stage-1] > 2 * self.prevnumalp[stage-1] + 50:
            self.reorganize(stage-1)
        return vb

    def __getitem__(self, b):
        return np.max(b @ self.alp[-1][:, 0:self.numalp[-1]])

    def add(self, newalp, i):
        # print("add to {i}".format(i=i))
        if self.numalp[i] == self.alp[i].shape[-1]:
            zeroarray = np.zeros(self.alp[i].shape)
            self.alp[i] = np.concatenate((self.alp[i], zeroarray), axis=-1)
        self.alp[i][..., self.numalp[i]] = newalp
        self.numalp[i] += 1

    def reorganize(self, i):
        """Only alpha vectors that are point-wise dominated are pruned
        """
        t00 = time.time()
        # print("[{0:.3f}s] Prev numalp {1}, current {2}".format(time.time() - self.t0, self.prevnumalp, self.numalp))
        # print("[{0:.3f}s] Reorganizing LowerBound...".format(time.time() - self.t0))    
        
        if i == -1:
            order = np.argsort(self.alp[i][0, 0:self.numalp[i]])
        else:
            order = np.argsort(self.alp[i][0, 0, 0:self.numalp[i]])
        self.alp[i] = self.alp[i][..., order]
        keepdix = [True] * self.numalp[i]
        for v in range(self.numalp[i]):
            for v1 in range(v + 1, self.numalp[i]):
                if np.all(self.alp[i][..., v] <= self.alp[i][..., v1]):
                    keepdix[v] = False
                    break

        self.alp[i] = self.alp[i][..., keepdix]
        self.numalp[i] = self.alp[i].shape[-1]
        # print("[{0:.3f}s] Reducing numalp to {1}".format(time.time() - self.t0, self.numalp))
        self.prevnumalp[i] = self.numalp[i]
        self.timereorg += time.time() - t00
    
    @property
    def numalptotal(self):
        return sum(self.numalp)


class CHSVI():
    """CHSVI Solver"""
    def __init__(self, Model, zeta=0.85, t0=None, presolveres=None, presolvealp=None):
        """Args:
            Model: a BaseCPOMDP object
            zeta: float, hyperparameter in (0, 1)
            t0: float, starting time (in unix time)
            presolveres: (Upperbound, Smat) or dictionary
                upper bound presolve solution used for upper bound initialization
            presolvealp: same format as LowerBound.alp
                alpha-vectors for lower bound initialization
        """
        if t0 is None:
            self.t0 = time.time()
        else:
            self.t0 = t0
        self.Model = Model
        if isinstance(presolveres, tuple): # (UpperBound, Smat)
            presolveres[0].transfer(Model, presolveres[1])
            self.VU = presolveres[0]
        else: # dictionary
            self.VU = UpperBound(Model, presolveres=presolveres)
        self.VL = LowerBound(Model, presolvealp=presolvealp)
        self.b0 = self.Model.b0
        self.zeta = zeta  # hyperparameter
        self.pathcount = 0
        self.nodecount = 0
        self.prevnodecount = 0
        self.timeheuristic = 0.0
        print("[{0:.3f}s] Initialization complete!".format(time.time() - self.t0))
        

    def Solve(self, timeout=np.inf, targetgap=0.0):
        """Run the CHSVI algorithm after initialization

        Args:
            timeout: time limit in seconds
            targetgap: float, terminates algorithm when gap is smaller than targetgap
        """
        epsilon = self.zeta * (self.VU[self.b0] - self.VL[self.b0])
        while True:
            try:
                self.Explore(self.b0, (), epsilon)
                self.pathcount += 1
                self.prevnodecount = self.nodecount
                vu, vl = self.VU[self.b0], self.VL[self.b0]
                gap = vu - vl
                epsilon = self.zeta * gap
                print(
                    ("[{0:.3f}s] {1} calls at root, {2} total calls, " + 
                    "LB: {3:.6f}, UB: {4:.6f}, gap: {5:.6f}").format(
                    time.time() - self.t0, self.pathcount, self.nodecount,
                    vl, vu, gap)
                )
                print(
                    "[{0:.3f}s] {1} LB alpha vectors, {2} UB points".format(
                        time.time() - self.t0, self.VL.numalp, self.VU.numpoints
                    )
                )
                if gap <= targetgap:
                    print("[{0:.3f}s] Target gap met. Terminating Algorithm".format(
                        time.time() - self.t0
                    ))
                    break
                if time.time() - self.t0 > timeout:
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
        print("Time spent on Upper Bound Update: {0:.3f}s".format(
            self.VU.timeupdate)
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
        

    def Explore(self, b, g, eps):
        self.nodecount += 1
        
        vub, g1 = self.VU.update(b, g)
        vlb = self.VL.update(b, g)
        gap = vub - vlb

        tb0 = time.time()

        # self._debug_print(b, g, g1, eps, vub, vlb, True)
        if gap <= eps:
            return
        
        next_b, next_g, next_eps = self.SearchHeuristics(b, g, g1, eps)
        if next_b is not None:
            self.Explore(next_b, next_g, next_eps)
        vub1, _ = self.VU.update(b, g)
        vlb1 = self.VL.update(b, g)
        
        # self._debug_print(b, g, g1, eps, vub1, vlb1, False)

    def _debug_print(self, b, g, g1, eps, vu, vl, down):
        """For debug purpose"""
        if down:
            prefix = "*down* depth: {0} belief: ".format(self.nodecount - self.prevnodecount - 1)
        else:
            prefix = "*up* "
        print(prefix + self._belief_and_prescription(b, g, g1))
        print("bounds: [{lb:.6f}, {ub:.6f}] target: {eps:.6f}".format(ub=vu, lb=vl, eps=eps))

    def _belief_and_prescription(self, b, g, g1):
        """For debug purpose"""
        line = []
        if g1 is None:
            for s in range(self.Model.S):
                if b[s] > 0:
                    a = tuple(g[i][self.Model.Mmap[i][s]] for i in range(len(g)))
                    line.append("({s}, {a}): {pr:.3f}".format(s=s, a=a, pr=b[s]))
        else:
            for s in range(self.Model.S):
                if b[s] > 0:
                    a = tuple(g[i][self.Model.Mmap[i][s]] for i in range(len(g)))
                    a1 = g1[self.Model.Mmap[len(g)][s]]
                    line.append("({s}, {a}): ({pr:.3f}, {a1})".format(s=s, a=a, a1=a1, pr=b[s]))
        return "[" + ", ".join(line) + "]"

    def SearchHeuristics(self, b, g, g1, eps):
        if len(g) < self.Model.I:
            return b, g + (g1,), eps
        else:
            t00 = time.time()
            next_eps = eps / self.Model.discount
            self.Model.setbelief(b, g)
            beliefs = [self.Model.nextbelief(o) for o in range(self.Model.O)]
            gapvec = np.array(
                [0 if (b1 is None) else self.VU[b1] - self.VL[b1]
                for b1 in beliefs]
            )
            score = self.Model.bgPO * (gapvec - next_eps)
            ostar = np.argmax(score)
            # print("*down* obs #", ostar)
            self.timeheuristic += time.time() - t00
            if score[ostar] <= 0:
                return None, None, None
            else:
                return beliefs[ostar], (), next_eps


    def UB(self, b=None):
        if b is None:
            b = self.b0
        return self.VU[b]

    def LB(self, b=None):
        if b is None:
            b = self.b0
        return self.VL[b]

