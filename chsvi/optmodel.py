"""Optimization Models

All calls to Gurobi optimization studio is wrapped in the classes defined in this
file. To replace Gurobi with something else, only the classes in this file needs
to be reimplemented, no other files need to change.

"""
import numpy as np
import gurobipy as grb

class OptimizationError(Exception):
    """Raised when gurobi got some internal error that cause the 
    optimization process to fail
    """
    pass

class OptimizationModel():
    """A generic gurobi model wrapper with constraint management

    Every once in a while, the model removes redundant constraints
    """
    def __init__(self, name):
        self.grbmodel = grb.Model(name)
        self.grbmodel.setAttr("ModelSense", grb.GRB.MAXIMIZE)
        self.constrbank = {}
        self.constrcounter = 0
        self.prevnumconstr = 0

    def addLEConstr(self, expr, rhs):
        """Add a constraint expr <= rhs"""
        handle = self.grbmodel.addConstr(expr <= rhs)
        self.constrbank[self.constrcounter] = (expr, rhs, handle)
        self.constrcounter += 1

        if self.numconstr >= 1.5 * self.prevnumconstr + 500:
            self.reorganize()

    def solve(self, expr, uberr=True):
        self.grbmodel.setObjective(expr)
        self.grbmodel.optimize()
        if (not uberr) and (
            self.grbmodel.Status == grb.GRB.UNBOUNDED or
            self.grbmodel.Status == grb.GRB.INF_OR_UNBD 
        ):
            return np.inf
        if self.grbmodel.Status == grb.GRB.OPTIMAL:
            return self.grbmodel.ObjVal
        elif self.grbmodel.Status == grb.GRB.INTERRUPTED: # user interrupt
            raise KeyboardInterrupt
        else:
            raise OptimizationError(self.grbmodel.Status)

    def reorganize(self):
        """Remove redundant constraints"""
        # print("OptimizationModel {0} reorganizing...".format(self.grbmodel.ModelName))
        allconstrs = sorted(self.constrbank.keys()) # earlier constraints more loose?
        for i in allconstrs:
            expr, rhs, handle = self.constrbank[i]
            self.grbmodel.remove(handle)
            new_rhs = self.solve(expr, uberr=False) # find maximum if constraint isn't there
            if new_rhs <= rhs: # constraint is redundant!
                del self.constrbank[i]
            else: # constraint is important add it back
                handle = self.grbmodel.addConstr(expr <= rhs)
                self.constrbank[i] = (expr, rhs, handle)
        self.prevnumconstr = self.numconstr


    @property
    def numconstr(self):
        return len(self.constrbank)


class UBOptimizationModel(OptimizationModel):
    """Optimization Models used by stage 0, 1, 2 of upper bound update

    This class make use of the CPOMDP model only through S, A, M. No
    observation or transition probability are used here. 

    Model: a CPOMDP model
    stage: integer
    res: a dictionary with field "Smat", "LHS", "RHS"
    privcompress: private information compression map is a Mi numpy array
    """
    def __init__(self, Model, stage, res):
        super().__init__(
            "upper bound {0}".format(stage)
        )
        S = Model.S
        if res["Smat"] is None:
            S0 = S
        else:
            S0 = res["Smat"].shape[0]
        self.stage = stage
        self.isblp = stage < Model.I
        if self.isblp: # blinear optimization for stage 0 to I-1
            Acumprod = np.prod(Model.A[0:stage+1])
            # prescription 
            self.g = self.grbmodel.addMVar(
                shape=(Model.M[stage], Model.A[stage]),
                vtype=grb.GRB.BINARY, name="g"
            )
            # distribution sums up to 1
            self.gconstr = []
            for m in range(Model.M[stage]):
                self.gconstr.append(self.grbmodel.addConstr(
                    self.g[m].sum() == 1
                ))

            self.grbmodel.setParam("NonConvex", 2) # invoke bilinear solver
            self.grbmodel.setParam("TimeLimit", 10) # in seconds
        else: # stage == I
            Acumprod = 1
        
        # the main variable for all problems is called y
        self.y = self.grbmodel.addMVar(
            shape=(S * Acumprod,),
            vtype=grb.GRB.CONTINUOUS,
            lb=-np.inf, name="y"
        )
        # auxiliary variable z(s) = max y(s, m1, m2) that can be used to
        # represent upper bound on aggregated state belief value function
        if res["Smat"] is None:
            zmax = self.y.reshape((S0, Acumprod))
            zmin = self.y.reshape((S0, Acumprod))
        else:
            zmax = self.grbmodel.addMVar(
                shape=(S0, Acumprod), vtype=grb.GRB.CONTINUOUS,
                lb=-np.inf, name="zmax"
            )
            zmin = self.grbmodel.addMVar(
                shape=(S0, Acumprod), vtype=grb.GRB.CONTINUOUS,
                lb=-np.inf, name="zmin"
            )
            
        numpoints = res["LHS"].shape[0]
        Qmax = res["RHS"].reshape((numpoints, Acumprod, -1))
        Qmaxstage = np.max(Qmax, axis=-1)

        if res["Smat"] is not None:
            self.grbmodel.addConstr(
                self.y.reshape((S, -1)) <= 
                res["Smat"].T @ zmax
            )
            self.grbmodel.addConstr(
                self.y.reshape((S, -1)) >= 
                res["Smat"].T @ zmin
            )

        for i in range(numpoints):
            bplus = res["LHS"][i] * (res["LHS"][i] > 0)
            bminus = res["LHS"][i] * (res["LHS"][i] < 0)
            for a in range(Acumprod):
                # bz = [(res["LHS"][i, s] * zmax[s, a] 
                #       if res["LHS"][i, s] > 0 else 
                #       res["LHS"][i, s] * zmin[s, a]) for s in range(S0)]
                self.addLEConstr(bplus @ zmax[:, a] + bminus @ zmin[:, a], Qmaxstage[i, a])

    
    def gsol(self):
        """Transform from solution probability vector to prescription vector
        """
        if self.isblp:
            return np.argmax(self.g.X, axis=1)
        else:
            return None

    def transfer(self, Model, Smat):
        """Transfer to a new model with larger state space

        Model: new CPOMDP model that has fewer common info than the old one
        Smat: S_old x Model.S 0-1 mapping matrix
        """
        # TODO: get rid of this function?

        self.reorganize()
        y_old = self.y

        if self.isblp:
            self.grbmodel.remove(self.gconstr)
            self.grbmodel.remove(self.g)

            # prescription 
            self.g = self.grbmodel.addMVar(
                shape=(Model.M[self.stage], Model.A[self.stage]),
                vtype=grb.GRB.BINARY, name="g"
            )
            # distribution sums up to 1
            self.gconstr = []
            for m in range(Model.M[self.stage]):
                self.gconstr.append(self.grbmodel.addConstr(
                    self.g[m].sum() == 1
                ))

        S_old = Smat.shape[0]
        Acumprod = y_old.size // S_old
        # the main variable for all problems is called y
        self.y = self.grbmodel.addMVar(
            shape=(Model.S * Acumprod,),
            vtype=grb.GRB.CONTINUOUS,
            lb=-np.inf, name="y"
        )
        # now y_old is the new auxiliary variable z
        self.grbmodel.addConstr(
            self.y.reshape((Model.S, -1)) >= Model.Vmin
        ) # this is a temporary fix...
        self.grbmodel.addConstr(
            self.y.reshape((Model.S, -1)) <= 
            Smat.T @ y_old.reshape((S_old, -1))
        )