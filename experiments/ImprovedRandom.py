
def addCount(dict, s, v):
    if (s not in dict.keys()):
        dict[s] = v
    else:
        dict[s] += v

def addP(P, counts, keys):
    ll = len(keys)
    for ns in P.keys():
        if (ns not in keys):
            P[ns] = P[ns] * (counts - 1) / (counts - 1 + ll)
        else:
            P[ns] = (P[ns] * (counts - 1) + 1) / (counts - 1 + ll)
    for k in keys:
        if (k not in P.keys()):
            P[k] = 1 / (counts - 1 + ll)

class IRandom:
    def __init__(self,env):
        self.env=env

        self.action_space=env.action_space

    def name(self):
        return "Random"

    def reset(self,initialstate):
        self.stateCounter = {}
        self.stateActionCounter = {}
        self.stateActionStateCounter = {}

        self.stateCounter[initialstate] = 1
        self.stateCounter['*'] = 1
        self.stateActionCounter[initialstate] = {}
        self.stateActionCounter['*'] = {}
        self.stateActionStateCounter[initialstate] = {}
        self.stateActionStateCounter['*'] = {}


    def play(self,state):
        if (state not in self.stateCounter.keys()):
            return self.action_space.sample()
        for a in range(self.action_space.n):
            if (a not in  self.stateActionStateCounter[state].keys()):
                return a
            if (self.stateActionCounter[state]=={}) and (a not in self.stateActionStateCounter['*'].keys()):
                return a
        return self.env.action_space.sample()


    def update(self, state, action, reward, nextstate):
        isnew_state = (self.stateCounter[state] == 1)
        isGnew_nextstate = (nextstate not in self.stateCounter.keys())
        isLnew_nextstate = (nextstate not in self.stateCounter.keys()) or (
                    action not in self.stateActionStateCounter[state].keys()) or (
                                       nextstate not in self.stateActionStateCounter[state][action].keys())
        ls = [state]
        if (isnew_state):
            ls.append('*')
        lGns = [nextstate]
        if (isGnew_nextstate):
            lGns.append('*')
        lLns = [nextstate]
        if (isLnew_nextstate):
            lLns.append('*')

        if (isGnew_nextstate):  # We reach a new state
            self.stateActionCounter[nextstate] = {}
            self.stateActionStateCounter[nextstate] = {}
            #self.TransitionEstimate[nextstate] = {}
            #self.SSquareTransitionEstimate[nextstate] = {}
            #self.MeanRewardEstimate[nextstate] = {}
            #self.SSquareRewardEstimate[nextstate] = {}

        for ns in lGns:
            addCount(self.stateCounter, ns, 1)
        for s in ls:
            addCount(self.stateActionCounter[s], action, 1)

        for s in ls:
            if (action not in self.stateActionStateCounter[s].keys()):  # action never played in state s
                #self.MeanRewardEstimate[s][action] = reward
                self.stateActionStateCounter[s][action] = {'*': 1, nextstate: 1}
                #self.TransitionEstimate[s][action] = {'*': 0.5, nextstate: 0.5}
                #self.SSquareRewardEstimate[s][action] = 0.
                #self.SSquareTransitionEstimate[s][action] = {'*': 0., nextstate: 0.}
            else:
                for ns in lLns:
                    addCount(self.stateActionStateCounter[s][action], ns, 1)
                #nMeanreward = (self.MeanRewardEstimate[s][action] * (self.stateActionCounter[s][action] - 1) + reward) /      self.stateActionCounter[s][action]
                #self.SSquareRewardEstimate[s][action] += (reward - self.MeanRewardEstimate[s][action]) * (                    reward - nMeanreward)
                #self.MeanRewardEstimate[s][action] = nMeanreward
                #addP(self.TransitionEstimate[s][action], self.stateActionCounter[s][action], lLns)