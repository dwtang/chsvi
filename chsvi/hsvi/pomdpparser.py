# Mostly followed Maxwell Forbes's parser code in https://github.com/mbforbes/py-pomdp
# This is a simple one I setup for debugging
# Specifying states, actions, or observations by name is not supported
import numpy as np
# import scipy.sparse as spsp
from chsvi.hsvi.pomdpmodel import POMDP

class POMDPParser():
    def __init__(self, filename):
        with open(filename) as fh:
            self.contents = [
                x.rstrip() for x in fh 
                if not (x.startswith('#') or x.isspace())
            ]

        self.S, self.A, self.O = 0, 0, 0
        self.PT, self.PZ, self.r, self.b0 = None, None, None, None
        i = 0
        while i < len(self.contents):
            line = self.contents[i]
            # print("Parsing: " + line)
            if line.startswith("discount:"):
                self.discount = float(line.split()[1])
                i += 1
            elif line.startswith("states:"):
                i, self.S = self.__get_sao(i)
            elif line.startswith("actions:"):
                i, self.A = self.__get_sao(i)
            elif line.startswith("observations:"):
                i, self.O = self.__get_sao(i)
            elif line.startswith("values:"):
                if line.split()[1] != "reward":
                    raise NotImplementedError("Only support reward")
                i += 1
            elif line.startswith("start:"):
                i = self.__get_start_dist(i)
            elif line.startswith("T:"):
                i = self.__get_transition_kernel(i)
            elif line.startswith("O:"):
                i = self.__get_observation_kernel(i)
            elif line.startswith("R:"):
                i = self.__get_rewards(i)
            else:
                raise Exception("Cannot parse line " + line)

            if (self.PT is None) and self.S > 0 and self.A > 0 and self.O > 0:
                self.PT = np.zeros((self.S, self.A, self.S))
                self.PZ = np.zeros((self.A, self.S, self.O))
                self.r = np.zeros((self.S, self.A, self.O, self.S))
    
        print("Parsing Complete!")

    def generatePOMDP(self):
        return POMDP((self.PT, self.PZ), self.r, self.discount, self.b0)

    def __get_sao(self, i):
        pieces = self.contents[i].split()
        if not pieces[1].isnumeric():
            raise NotImplementedError("Please specify number of " + pieces[0][:-1])
        return i + 1, int(pieces[1])

    def __get_start_dist(self, i):
        pieces = [x for x in self.contents[i].split() if (x.find(':') == -1)]
        if len(pieces) == 0:
            probs = self.contents[i+1].split()
            next_i = i + 2
        else:
            probs = pieces
            next_i = i + 1
        assert(len(probs) == self.S)
        self.b0 = np.array([float(x) for x in probs])
        assert(self.b0.sum() == 1.0)
        return next_i


    def __get_transition_kernel(self, i):
        pieces = [x for x in self.contents[i].split() if (x.find(':') == -1)]
        action = _idx(pieces[0])

        if len(pieces) == 4:
            # case 1: T: <action> : <start-state> : <next-state> %f
            start_state = _idx(pieces[1])
            next_state = _idx(pieces[2])
            prob = float(pieces[3])
            self.PT[start_state, action, next_state] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: T: <action> : <start-state> : <next-state>
            # %f
            start_state = _idx(pieces[1])
            next_state = _idx(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.PT[start_state, action, next_state] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: T: <action> : <start-state>
            # %f %f ... %f
            start_state = _idx(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert(len(probs) == self.S)
            for next_state in range(self.S):
                prob = float(probs[next_state])
                self.PT[start_state, action, next_state] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: T: <action>
                # identity
                self.PT[:, action, :] = 0.0
                for start_state in range(self.S):
                    self.PT[start_state, action, start_state] = 1.0
                return i + 2
            elif next_line == "uniform":
                # case 5: T: <action>
                # uniform
                prob = 1.0 / self.S
                self.PT[:, action, :] = prob
                return i + 2
            else:
                # case 6: T: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for start_state in range(self.S):
                    probs = next_line.split()
                    assert(len(probs) == self.S)
                    for next_state in range(self.S):
                        prob = float(probs[k])
                        self.PT[start_state, action, next_state] = prob
                    next_line = self.contents[i+2+j]
                return i + 1 + self.S
        else:
            raise Exception("Cannot parse line " + line)

    def __get_observation_kernel(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = _idx(pieces[0])

        if len(pieces) == 4:
            # case 1: O: <action> : <next-state> : <obs> %f
            next_state = _idx(pieces[1])
            obs = _idx(pieces[2])
            prob = float(pieces[3])
            self.PZ[action, next_state, obs] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: O: <action> : <next-state> : <obs>
            # %f
            next_state = _idx(pieces[1])
            obs = _idx(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.PZ[action, next_state, obs] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: O: <action> : <next-state>
            # %f %f ... %f
            next_state = _idx(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert(len(probs) == self.O)
            for obs in range(self.O):
                prob = float(probs[obs])
                self.PZ[action, next_state, obs] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: O: <action>
                # identity
                assert(self.S == self.O)
                self.PZ[action, :, :] = 0.0
                for next_state in range(self.S):
                    self.PZ[action, next_state, next_state] = 1.0
                return i + 2
            elif next_line == "uniform":
                # case 5: O: <action>
                # uniform
                prob = 1.0 / self.O
                self.PZ[action, :, :] = prob
                return i + 2
            else:
                # case 6: O: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for next_state in range(self.S):
                    probs = next_line.split()
                    assert(len(probs) == self.O)
                    for obs in range(self.O):
                        prob = float(probs[k])
                        self.PZ[action, next_state, obs] = prob
                    next_line = self.contents[i+2+j]
                return i + 1 + self.S
        else:
            raise Exception("Cannot parse line: " + line)

    def __get_rewards(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = _idx(pieces[0])
        start_state = _idx(pieces[1])

        if len(pieces) >= 4:
            # case 1: R: <action> : <start-state> : <next-state> : <obs> %f
            next_state = _idx(pieces[2])
            obs = _idx(pieces[3])
            # if next_state != slice(None):
            #     raise NotImplementedError("Reward Depends on next state in line " + line)
            # if obs != slice(None):
            #     raise NotImplementedError("Reward Depends on next observation in line" + line)
            reward = float(self.contents[i+1]) if len(pieces) == 4 else float(pieces[-1])
            self.r[start_state, action, obs, next_state] = reward
            return i + 1 + (len(pieces) == 4)
        elif len(pieces) >= 2:
            # case 3: R: <action> : <start-state> : <end-state>
            # %f %f ... %f
            # case 4: R: <action> : <start-state>
            # %f matrix
            raise NotImplementedError("Developer hasn't got this case yet: " + line)
        else:
            raise Exception("Cannot parse line: " + line)


def _idx(s):
    return slice(None) if (s == "*") else int(s)