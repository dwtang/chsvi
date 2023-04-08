"""Writing a CPOMDP model or a transformation of it to a file

"""
import itertools
import numpy as np
import scipy.sparse as spsp


def WriteMultiStepCPOMDP(Model, filename):
    """Construct the coordinator's POMDP model for a given CPOMDP 
    to .pomdp file following Cassandra's format

    Warning: even a small CPOMDP can result in a gigantic POMDP due to the size
    of prescription space. Apply this function only on very small instances (i.e.
    Ai**Mi ~ 10^3)

    The .pomdp file format and the HSVI solver written by Trey Smith both assume
    that P(s' o|s, a) = T(s, a, s')Z(a, s', o), i.e. given a and s', o is 
    conditionally independent of s. Unfortunately, this is not satisfied by 
    DelaySharingCPOMDP and many other classes, as o is a function of the private
    state, which is a part of the current state but not part of the next state. 
    Therefore, in the transformation, we will define (o, s) as the new state
    
    Input: 
        Model: a BaseCPOMDP instance
    """
    # generate all prescriptions
    pres = [
        list(itertools.product(range(Model.A[i]), repeat=Model.M[i]))
        for i in range(Model.I)
    ]
    newA = max(len(presi) for presi in pres)

    # print("\n".join(str(it) for it in pres[0]))
    
    # generate hyper states
    statedims = [(Model.O, Model.S,) + Model.A[0:i] for i in range(Model.I + 1)]
    allstatesbystage = [
        list(itertools.product(*[range(k) for k in dims]))
        for dims in statedims
    ]
    allstates = list(itertools.chain(*allstatesbystage))
    # print("\n".join(str(it) for it in allstates))
    newS = len(allstates)
    statelookup = {allstates[i]: i for i in range(newS)}

    # new discount factor since 1 stage is streached into I+1 stages
    newdiscount = Model.discount ** (1.0/(Model.I+1))

    # newO is still the old O but just define it to make code nicer
    newO = Model.O

    # initial belief: can be set to any distribution with marginal b0
    # where O, S components are independent
    newb0 = np.zeros(newS)
    for s in range(Model.S):
        i = statelookup[(0, s)]
        newb0[i] = Model.b0[s]

    content = []
    line = "discount: {0:.10f}".format(newdiscount)
    content.append(line)
    line = "values: reward"
    content.append(line)
    line = "states: {0}".format(newS)
    content.append(line)
    line = "actions: {0}".format(newA)
    content.append(line)
    line = "observations: {0}".format(newO)
    content.append(line)
    line = "start: " + " ".join(
        "{0:.6f}".format(newb0[j]) for j in range(newS)
    )
    content.append(line)

    # print("\n".join(content))
    # return

    # observation kernel
    for s in range(newS):
        o = allstates[s][0]
        line = "O: * : {end_s} : {obs} {pr:.10f}".format(
            end_s=s, obs=o, pr=1.0
        )
        content.append(line)

    # state transition
    for i in range(Model.I):
        for j in range(newA):
            actual_j = min(j, newA-1)
            print("working on prescription #{0} of agent #{1}".format(j, i))
            # (_o, _s, a0, ... , ai-1) -> (_o, _s, a0, ... , ai-1)
            for stuple in allstatesbystage[i]:
                s = statelookup[stuple]
                inner_s = stuple[1]
                mi = Model.Mmap[i][inner_s]
                ai = pres[i][actual_j][mi]
                next_stuple = stuple + (ai,)
                next_s = statelookup[next_stuple]
                line = "T: {a} : {start_s} : {end_s} {pr:.10f}".format(
                    a=j, start_s=s, end_s=next_s, pr=1.0)
                content.append(line)

    # the last stage state transition and rewards
    for stuple in allstatesbystage[-1]:
        s = statelookup[stuple]
        saidx = np.ravel_multi_index(stuple[1:], Model.SA)
        rsa = Model._r[saidx] / (newdiscount ** 2) 
        # since the rewards at new models first arrive at time index 2, need to scale
        # up the reward in the transformed model by 1/newdiscount^2
        line = "R: * : {start_s} : * : * {v:.10f}".format(start_s=s, v=rsa)
        content.append(line)

        Psa = spsp.coo_matrix(Model._P[saidx]) # O*S
        for j in range(len(Psa.col)):
            next_s, pr = Psa.col[j], Psa.data[j]
            line = "T: * : {start_s} : {end_s} {pr:.10f}".format(
                    start_s=s, end_s=next_s, pr=pr)
            content.append(line)
    

    content = "\n".join(content)
    with open(filename + ".pomdp", "w") as f:
        f.write(content)

    print("Model written to {0}.pomdp".format(filename))


def WriteCPOMDP(Model, filename):
    """Construct the coordinator's POMDP model for a given CPOMDP 
    to .pomdp file following Cassandra's format

    Warning: even a small CPOMDP can result in a gigantic POMDP due to the size
    of prescription space. Apply this function only on very small instances (i.e.
    prod(Ai**Mi) ~ 10^3)

    The .pomdp file format and the HSVI solver written by Trey Smith both assume
    that P(s' o|s, a) = T(s, a, s')Z(a, s', o), i.e. given a and s', o is 
    conditionally independent of s. Unfortunately, this is not satisfied by 
    DelaySharingCPOMDP and many other classes, as o is a function of the private
    state, which is a part of the current state but not part of the next state. 
    Therefore, in the transformation, we will define (o, s') as the new state
    
    Input: 
        Model: a BaseCPOMDP instance
    """
    # generate all prescriptions
    pres = [
        itertools.product(range(Model.A[i]), repeat=Model.M[i])
        for i in range(Model.I)
    ]
    prespairs = list(itertools.product(*pres))
    newA = len(prespairs)
    
    # generate hyper states
    allstates = list(itertools.product(range(Model.O), range(Model.S)))
    # print("\n".join(str(it) for it in allstates))
    newS = len(allstates)

    # initial belief: can be set to any distribution with marginal b0
    # where O, S components are independent
    newb0 = np.zeros((Model.O, Model.S))
    for s in range(Model.S):
        newb0[0, s] = Model.b0[s]
    newb0 = newb0.flatten()

    content = []
    line = "discount: {0:.10f}".format(Model.discount)
    content.append(line)
    line = "values: reward"
    content.append(line)
    line = "states: {0}".format(newS)
    content.append(line)
    line = "actions: {0}".format(newA)
    content.append(line)
    line = "observations: {0}".format(Model.O)
    content.append(line)
    line = "start: " + " ".join(
        "{0:.6f}".format(newb0[j]) for j in range(newS)
    )
    content.append(line)

    # observation kernel
    for s in range(newS):
        o = allstates[s][0]
        line = "O: * : {end_s} : {obs} {pr:.10f}".format(
            end_s=s, obs=o, pr=1.0
        )
        content.append(line)

    # state transition and rewards
    for j in range(newA):
        print("working on prescription #{0} of {1}".format(j, newA))
        Saidx = Model.SAIndex(prespairs[j])
        PSj = Model._P[Saidx] # O*S
        for news in range(newS):
            o, s = np.unravel_index(news, (Model.O, Model.S))
            rsa = Model._r[Saidx][s]
        
            line = "R: {a} : {start_s} : * : * {v:.10f}".format(
                a=j ,start_s=news, v=rsa
            )
            content.append(line)

            Psj = spsp.coo_matrix(PSj[s])
            for l in range(len(Psj.col)):
                next_s, pr = Psj.col[l], Psj.data[l]
                line = "T: {a} : {start_s} : {end_s} {pr:.10f}".format(
                        a=j, start_s=news, end_s=next_s, pr=pr)
                content.append(line)
    

    content = "\n".join(content)
    with open(filename + ".pomdp", "w") as f:
        f.write(content)

    print("Model written to {0}.pomdp".format(filename))