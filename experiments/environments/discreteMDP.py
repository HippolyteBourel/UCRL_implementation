import numpy as np


import sys
from six import StringIO
from gym import Env, spaces
from gym.utils import seeding
from gym import utils

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Circle
import string



def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

def clip(x,range):
    return max(min(x,range[1]),range[0])

class DiscreteMDP(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - R: reward distributions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, done), ...]
      R[s][a] == distribution(mean,param)
    (**) list or array of length nS


    """

    metadata = {'render.modes': ['text', 'ansi', 'pylab']}

    def __init__(self, nS, nA, P, R, isd,nameActions=[]):
        self.P = P
        self.R = R
        self.isd = isd
        self.lastaction=None # for rendering
        self.nS = nS
        self.nA = nA

        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        if(len(nameActions)==0):
            self.nameActions = list(string.ascii_uppercase)[0:min(self.nA,26)]


        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.initializedRender=False
        self.seed()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d= transitions[i]
        r =  clip(rewarddis.rvs(), self.reward_range)
        self.s = s
        self.lastaction=a
        return (s, r, d, "")

    def getTransition(self,s,a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]]=c[0]
        return transition
    
    # nb_iter is the number of reward samples used to cimpute the mean of the reward for the given pair of state-action.
    def getReward(self, s, a, nb_iter = 1):
        rewarddis = self.R[s][a]
        r =  np.mean([clip(rewarddis.rvs(), self.reward_range) for _ in range(nb_iter)])
        return r


    def initRender(self):
        self.numFigure = plt.gcf().number
        plt.figure(self.numFigure)
        scale = self.nS
        G = nx.MultiDiGraph(action=0, rw=0.)
        for s in self.states:
            for a in self.actions:
                for ssl in self.P[s][a]: #ssl = (p(s),s, 'done')
                    G.add_edge(s, ssl[1], action=a, weight=ssl[0], rw=self.R[s][a].mean())

        pos = nx.spring_layout(G)
        for x in self.states:
            pos[x] = [pos[x][0] * scale, pos[x][1] * scale]
        self.G = G
        self.pos = pos

        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                  'tab:olive', 'tab:cyan']
        plt.clf()
        ax = plt.gca()

        nx.draw_networkx_nodes(G, pos, node_size=400,
                               node_color=['tab:gray' if s != self.s else 'tab:orange' for s in self.G.nodes()])
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

        for n in G:
            c = Circle(pos[n], radius=0.2, alpha=0.)
            ax.add_patch(c)
            G.node[n]['patch'] = c
        counts = np.zeros((self.nS, self.nS))
        countsR = np.zeros((self.nS, self.nS))
        seen = {}
        for u, v, d in G.edges(data=True):
            # print(u,v,d)
            n1 = G.node[u]['patch']
            n2 = G.node[v]['patch']
            rad = 0.1
            if (u, v) in seen:
                rad = seen.get((u, v))
                rad = (rad + np.sign(rad) * 0.1) * -1
            alpha = d['weight']
            color = colors[d['action']]
            if alpha > 0:
                counts[u][v] = counts[u][v] + 1
                if (u != v):
                    e = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                                        arrowstyle='-|>',
                                        connectionstyle='arc3,rad=%s' % rad,
                                        mutation_scale=15.0 + scale,
                                        lw=2,
                                        alpha=alpha,
                                        color=color)
                    seen[(u, v)] = rad
                    ax.add_patch(e)
                    if (d['rw'] > 0):
                        countsR[u][v] = countsR[u][v] + 1
                        nx.draw_networkx_edge_labels([u, v, d], pos,
                                                     edge_labels=dict([((u, v), str(np.ceil(d['rw'] * 100) / 100))]),
                                                     label_pos=0.5 + 0.1 * countsR[u][v], font_color=color, alpha=alpha,
                                                     font_size=8)

                else:
                    n1c = [n1.center[0] + 0.1 * (2 * counts[u][v] + scale),
                           n1.center[1] + 0.1 * (2 * counts[u][v] + scale)]
                    e1 = FancyArrowPatch(n1.center, n1c,
                                         arrowstyle='-|>',
                                         connectionstyle='arc3,rad=1.',
                                         mutation_scale=15.0 + scale,
                                         lw=2,
                                         alpha=alpha,
                                         color=color)
                    e2 = FancyArrowPatch(n1c, n1.center,
                                         arrowstyle='-|>',
                                         connectionstyle='arc3,rad=1.',
                                         mutation_scale=15.0 + scale,
                                         lw=2,
                                         alpha=alpha,
                                         color=color
                                         )
                    ax.add_patch(e1)
                    ax.add_patch(e2)
                    if (d['rw'] > 0):
                        countsR[u][v] = countsR[u][v] + 1
                        pos[u] = [pos[u][0] + 0.1 * (2 * countsR[u][v] + scale),
                                  pos[u][1] + 0.1 * (2 * countsR[u][v] + scale)]
                        nx.draw_networkx_edge_labels([u, v, d], pos,
                                                     edge_labels=dict(
                                                         [((u, v), str(np.ceil(d['rw'] * 100) / 100))]),
                                                     label_pos=0.5, font_color=color,
                                                     alpha=alpha, font_size=8)
                        pos[u] = [pos[u][0] - 0.1 * (2 * countsR[u][v] + scale),
                                  pos[u][1] - 0.1 * (2 * countsR[u][v] + scale)]

        ax.autoscale()
        plt.axis('equal')
        plt.axis('off')
        plt.savefig('MDP-discrete.png')
        plt.show(block=False)
        plt.pause(0.5)



    def render(self,mode='pylab'):
        if (mode=="text"):
            #Print the MDp in text mode.
            # Red  = current state
            # Blue = all states accessible from current state (by playing some action)
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            desc = [str(s)  for s in self.states]


            desc[self.s] = utils.colorize(desc[self.s], "red", highlight=True)
            for a in self.actions:
                for ssl in self.P[self.s][a]:
                    if (ssl[0]>0):
                        desc[ssl[1]] = utils.colorize(desc[ssl[1]], "blue", highlight=True)

            if self.lastaction is not None:
                outfile.write("  ({})\t".format(self.nameActions[self.lastaction % 26]))
            else:
                outfile.write("\n")
            outfile.write("".join(''.join(line) for line in desc) + "\n")

            if mode != 'text':
                return outfile
        else:
            # Print the MDP in an image MDP.png, MDP.pdf
            # Node colors : orange = current state, gray = other states
            # Edge colors : the color indicates the corresponding action (e.g. blue= action 0, red = action 1, etc)
            # Edge transparency: indicates the probability with which we transit to that state.
            # Edge label: A label indicates a positive reward, with mean value given by the labal (color of the label = action)
            # Print also the MDP only shoinwg the rewards in MDPonlytherewards.pdg, MDPonlytherewards.pdf
            # Node colors : orange = current state, gray = other states
            # Edge colors : the color indicates the corresponding action (e.g. blue= action 0, red = action 1, etc)
            # Edge transparency: indicates the value of the mean reward.
            # Edge label: A label indicates a positive reward, with mean value given by the labal (color of the label = action)

            if (not self.initializedRender):
                self.initRender()
                self.initializedRender=True
            G = self.G
            pos = self.pos


            plt.figure(self.numFigure)
            nx.draw_networkx_nodes(G, pos, node_size=400,
                                   node_color=['tab:gray' if s != self.s else 'tab:orange' for s in self.G.nodes()])
            plt.show(block=False)
            plt.pause(0.01)


import scipy.stats as stat

class RandomMDP(DiscreteMDP):
    def __init__(self, nbStates,nbActions, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5):
        self.nS = nbStates
        self.nA = nbActions
        self.states = range(0,nbStates)
        self.actions = range(0,nbActions)


        self.startdistribution = np.zeros((self.nS))
        self.rewards = {}
        self.transitions = {}
        self.P = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            self.rewards[s]={}
            for a in self.actions:
                self.P[s][a]=[]
                self.transitions[s][a]={}
                self.rewards[s][a]=stat.norm(loc=self.sparserand(p=maxProportionSupportReward, min=minNonZeroReward),scale=rewardStd)
                transitionssa = np.zeros((self.nS))
                for s2 in self.states:
                    transitionssa[s2] = self.sparserand(maxProportionSupportTransition)
                mass = sum(transitionssa)
                if (mass > 0):
                    transitionssa = transitionssa / sum(transitionssa)
                    transitionssa = self.reshapeDistribution(transitionssa, minNonZeroProbability)
                else:
                    transitionssa[np.random.randint(self.nS)] = 1.
                li= self.P[s][a]
                [li.append((transitionssa[s], s, False)) for s in self.states if transitionssa[s]>0]
                self.transitions[s][a]= {ss:transitionssa[ss] for ss in self.states}

            self.startdistribution[s] = self.sparserand(maxProportionSupportStart)
        mass = sum(self.startdistribution)
        if (mass > 0):
            self.startdistribution = self.startdistribution / sum(self.startdistribution)
            self.startdistribution = self.reshapeDistribution(self.startdistribution, minNonZeroProbability)
        else:
            self.startdistribution[np.random.randint(self.nS)] = 1.

        checkRewards = sum([sum([self.rewards[s][a].mean() for a in self.actions]) for s in self.states])
        if(checkRewards==0):
            s = np.random.randint(0,self.nS)
            a = np.random.randint(0,self.nA)
            self.rewards[s][a]=stat.norm(loc=minNonZeroReward + np.random.rand() * (1. - minNonZeroReward),scale=rewardStd)
        #self.isd = {s:self.startdistribution[s] for s in self.states}

        super(RandomMDP, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)

    def sparserand(self,p=0.5, min=0., max=1.):
        u = np.random.rand()
        if (u <= p):
            return min + np.random.rand() * (max - min)
        return 0.

    def reshapeDistribution(self,distribution, p):
        mdistribution = [0 if x < p else x for x in distribution]
        missingmass = 1. - sum(mdistribution)
        if (missingmass == 1):
            mdistribution[np.random.randint(0, len(distribution))] = p
            missingmass = missingmass - p
        while (missingmass > 0):
            i = np.random.randint(0, len(distribution))
            if (mdistribution[i] >= p):
                newp = min(1., mdistribution[i] + missingmass)
                missingmass = missingmass + (mdistribution[i] - newp)
                mdistribution[i] = newp
        return mdistribution


class RiverSwim(DiscreteMDP):
    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):#, ergodic=False): # TODO ergordic option
        self.nS = nbStates
        self.nA = 2
        self.states = range(0,nbStates)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            # GOING RIGHT
            self.transitions[s][0]={}
            self.P[s][0]= [] #0=right", 1=left
            li = self.P[s][0]
            prr=0.
            if (s<self.nS-1):
                li.append((rightProbaright, s+1, False))
                self.transitions[s][0][s+1]=rightProbaright
                prr=rightProbaright
            prl = 0.
            if (s>0):
                li.append((rightProbaLeft, s-1, False))
                self.transitions[s][0][s-1]=rightProbaLeft
                prl=rightProbaLeft
            li.append((1.-prr-prl, s, False))
            self.transitions[s][0][s ] = 1.-prr-prl

            # GOING LEFT
            #if ergodic:
            #    pll = 0.95
            #else:
            #    pll = 1
            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1]={}
            li = self.P[s][1]
            if (s > 0):
                li.append((1., s - 1, False))
                self.transitions[s][1][s-1]=1.
            else:
                li.append((1., s, False))
                self.transitions[s][1][s]=1.

            self.rewards[s]={}
            if (s==self.nS-1):
                self.rewards[s][0] = stat.norm(loc=rewardR,scale=0.)
            else:
                self.rewards[s][0] = stat.norm(loc=0., scale=0.)
            if (s==0):
                self.rewards[s][1] = stat.norm(loc=rewardL,scale=0.)
            else:
                self.rewards[s][1] = stat.norm(loc=0., scale=0.)
                
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)


        super(RiverSwim, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)


    # def render(self, mode="text"):
    #     if (mode=="text"):
    #         outfile = StringIO() if mode == 'ansi' else sys.stdout
    #
    #         desc = [str(s)  for s in self.states]
    #
    #         desc[self.s] = utils.colorize(desc[self.s], "red", highlight=True)
    #         if self.lastaction is not None:
    #             outfile.write("  ({})\t".format(["R", "L"][self.lastaction]))
    #         else:
    #             outfile.write("\n")
    #         outfile.write("".join(''.join(line) for line in desc) + "\n")
    #
    #         if mode != 'text':
    #             return outfile
    #     else:
    #         super(RiverSwim,self).render(mode)
    
    
    
    
    
    
class ThreeState(DiscreteMDP):
    def __init__(self, delta = 0.005, fixed_reward = True):
        self.nS = 3
        self.nA = 2
        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a randomly generated MDP
        
        # for s in...
        s = 0
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((delta, 1, False))
        self.transitions[s][0][1] = delta
        self.P[s][0].append((1.- delta, 2, False))
        self.transitions[s][0][2] = 1. - delta
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = stat.norm(loc=0.,scale=0.)
            self.rewards[s][1] = stat.norm(loc=0.,scale=0.)
        else:
            self.rewards[s][0] = stat.norm(loc=0.,scale=0.)
            self.rewards[s][1] = stat.norm(loc=0.,scale=0.)
        
        s = 1
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((1., 0, False))
        self.transitions[s][0][0] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.P[s][1] = []  # 0=right", 1=left
        self.transitions[s][1] = self.transitions[s][0]
        self.P[s][1] = self.P[s][0]
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = stat.norm(loc=1./3.,scale=0.)
            self.rewards[s][1] = stat.norm(loc=1./3.,scale=0.)
        else:
            self.rewards[s][0] = stat.bernoulli(1./3.)
            self.rewards[s][1] = stat.bernoulli(1./3.)
        
        s = 2
        self.P[s]={}
        self.transitions[s]={}
        # Action 0
        self.transitions[s][0]={}
        self.P[s][0]= [] #0=right", 1=left
        self.P[s][0].append((1., 2, False))
        self.transitions[s][0][2] = 1.
        # Action 1 which just the same for s = 0 and s = 1
        self.transitions[s][1]={}
        self.P[s][1]= [] #0=right", 1=left
        self.P[s][1].append((delta, 1, False))
        self.transitions[s][1][1] = delta
        self.P[s][1].append((1.- delta, 0, False))
        self.transitions[s][1][0] = 1. - delta
        # reward
        self.rewards[s]={}
        if fixed_reward:
            self.rewards[s][0] = stat.norm(loc=2./3.,scale=0.)
            self.rewards[s][1] = stat.norm(loc=2./3.,scale=0.)
        else:
            self.rewards[s][0] = stat.bernoulli(2./3.)
            self.rewards[s][1] = stat.bernoulli(2./3.)
         
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)
        super(ThreeState, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)
        
        
        
class RiverSwimErgo(DiscreteMDP):
    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):#, ergodic=False): # TODO ergordic option
        self.nS = nbStates
        self.nA = 2
        self.states = range(0,nbStates)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            # GOING RIGHT
            self.transitions[s][0]={}
            self.P[s][0]= [] #0=right", 1=left
            li = self.P[s][0]
            prr=0.
            if (s<self.nS-1):
                li.append((rightProbaright, s+1, False))
                self.transitions[s][0][s+1]=rightProbaright
                prr=rightProbaright
            prl = 0.
            if (s>0):
                li.append((rightProbaLeft, s-1, False))
                self.transitions[s][0][s-1]=rightProbaLeft
                prl=rightProbaLeft
            li.append((1.-prr-prl, s, False))
            self.transitions[s][0][s ] = 1.-prr-prl

            # GOING LEFT
            #if ergodic:
            #    pll = 0.95
            #else:
            #    pll = 1
            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1]={}
            li = self.P[s][1]
            if (s > 0):
                if s!=self.nS-1:
                    li.append((0.1, s + 1, False))
                    self.transitions[s][1][s+1]=0.1
                    li.append((0.9, s - 1, False))
                    self.transitions[s][1][s-1]=0.9
                else:
                    li.append((1, s - 1, False))
                    self.transitions[s][1][s-1]=1
            else:
                li.append((0.9, s, False))
                self.transitions[s][1][s]=0.9
                li.append((0.1, s + 1, False))
                self.transitions[s][1][s+1]=0.1

            self.rewards[s]={}
            if (s==self.nS-1):
                self.rewards[s][0] = stat.norm(loc=rewardR,scale=0.)
            else:
                self.rewards[s][0] = stat.norm(loc=0., scale=0.)
            if (s==0):
                self.rewards[s][1] = stat.norm(loc=rewardL,scale=0.)
            else:
                self.rewards[s][1] = stat.norm(loc=0., scale=0.)
                
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)


        super(RiverSwimErgo, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)







class RiverSwim_biClass(DiscreteMDP):
    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.7, rightProbaLeft2=0.1):#, ergodic=False): # TODO ergordic option
        self.nS = nbStates
        self.nA = 2
        self.states = range(0,nbStates)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            # GOING RIGHT
            self.transitions[s][0]={}
            self.P[s][0]= [] #0=right", 1=left
            li = self.P[s][0]
            prr=0.
            if (s<(self.nS//2)):
                li.append((rightProbaright2, s+1, False))
                self.transitions[s][0][s+1]=rightProbaright2
                prr=rightProbaright2
            elif (s<self.nS-1):
                li.append((rightProbaright, s+1, False))
                self.transitions[s][0][s+1]=rightProbaright
                prr=rightProbaright
            prl = 0.
            if (s>(self.nS//2)):
                li.append((rightProbaLeft2, s-1, False))
                self.transitions[s][0][s-1]=rightProbaLeft2
                prl=rightProbaLeft2
            elif (s>0):
                li.append((rightProbaLeft, s-1, False))
                self.transitions[s][0][s-1]=rightProbaLeft
                prl=rightProbaLeft
            li.append((1.-prr-prl, s, False))
            self.transitions[s][0][s ] = 1.-prr-prl

            # GOING LEFT
            #if ergodic:
            #    pll = 0.95
            #else:
            #    pll = 1
            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1]={}
            li = self.P[s][1]
            if (s > 0):
                li.append((1., s - 1, False))
                self.transitions[s][1][s-1]=1.
            else:
                li.append((1., s, False))
                self.transitions[s][1][s]=1.

            self.rewards[s]={}
            if (s==self.nS-1):
                self.rewards[s][0] = stat.norm(loc=rewardR,scale=0.)
            else:
                self.rewards[s][0] = stat.norm(loc=0., scale=0.)
            if (s==0):
                self.rewards[s][1] = stat.norm(loc=rewardL,scale=0.)
            else:
                self.rewards[s][1] = stat.norm(loc=0., scale=0.)
                
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)


        super(RiverSwim_biClass, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)
        
        
        
        
        
        
        
class RiverSwim_shuffle(DiscreteMDP):
    def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.7, rightProbaLeft2=0.1):#, ergodic=False): # TODO ergordic option
        self.nS = nbStates
        self.nA = 2
        self.states = range(0,nbStates)
        self.actions = range(0,self.nA)
        self.nameActions = ["R", "L"]


        self.startdistribution = np.zeros((self.nS))
        self.startdistribution[0] =1.
        self.rewards = {}
        self.P = {}
        self.transitions = {}
        # Initialize a randomly generated MDP
        for s in self.states:
            self.P[s]={}
            self.transitions[s]={}
            # GOING RIGHT
            self.transitions[s][0]={}
            self.P[s][0]= [] #0=right", 1=left
            li = self.P[s][0]
            prr=0.
            if (s<self.nS-1):
                if s%2 == 0:
                    li.append((rightProbaright, s+1, False))
                    self.transitions[s][0][s+1]=rightProbaright
                    prr=rightProbaright
                else:
                    li.append((rightProbaright2, s+1, False))
                    self.transitions[s][0][s+1]=rightProbaright2
                    prr=rightProbaright2
            prl = 0.
            if (s>0):
                if s%2 == 0:
                    li.append((rightProbaLeft, s-1, False))
                    self.transitions[s][0][s-1]=rightProbaLeft
                    prl=rightProbaLeft
                else:
                    li.append((rightProbaLeft2, s-1, False))
                    self.transitions[s][0][s-1]=rightProbaLeft2
                    prl=rightProbaLeft2
            li.append((1.-prr-prl, s, False))
            self.transitions[s][0][s ] = 1.-prr-prl

            # GOING LEFT
            #if ergodic:
            #    pll = 0.95
            #else:
            #    pll = 1
            self.P[s][1] = []  # 0=right", 1=left
            self.transitions[s][1]={}
            li = self.P[s][1]
            if (s > 0):
                li.append((1., s - 1, False))
                self.transitions[s][1][s-1]=1.
            else:
                li.append((1., s, False))
                self.transitions[s][1][s]=1.

            self.rewards[s]={}
            if (s==self.nS-1):
                self.rewards[s][0] = stat.norm(loc=rewardR,scale=0.)
            else:
                self.rewards[s][0] = stat.norm(loc=0., scale=0.)
            if (s==0):
                self.rewards[s][1] = stat.norm(loc=rewardL,scale=0.)
            else:
                self.rewards[s][1] = stat.norm(loc=0., scale=0.)
                
        #print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)


        super(RiverSwim_shuffle, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)