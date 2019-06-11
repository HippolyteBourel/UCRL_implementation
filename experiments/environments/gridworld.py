import numpy as np
import sys
from six import StringIO, b

import scipy.stats as stat
import matplotlib.pyplot as plt

from gym import utils
from gym.envs.toy_text import discrete
import environments.discreteMDP

from gym import Env, spaces
import string


#maps = {"random":randomGridWorld, "4-room":fourRoom}

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


def randomGridWorld(sizeX, sizeY,density,lengthofwalks):
    maze = np.ones((sizeX, sizeY))
    s = [np.random.randint(sizeX), np.random.randint(sizeY)]
    for i in range((int)(density * sizeX * sizeY)):
        p = np.exp(-np.log( 2) / lengthofwalks)  # proba p de continuer le mur courant, 1-p de prendre créer un nouveau mur : pour p^k = 1/2 then p =exp(-ln(2)/k)
        b = np.random.binomial(1, p)
        if (b == 0):
            next = np.random.randint(4)
            if (next == 0):     s = [(s[0] + 1) % sizeX, s[1]]
            if (next == 1):     s = [(s[0] - 1) % sizeX, s[1]]
            if (next == 2):     s = [s[0], (s[1] + 1) % sizeY]
            if (next == 3):     s = [s[0], (s[1] - 1) % sizeY]
        else:
            s = [np.random.randint(sizeX), np.random.randint(sizeY)]
        maze[s[0]][s[1]] = 0.
    return maze

def fourRoom(X,Y):
    Y2 = (int) (Y/2)
    X2 = (int) (X/2)
    maze = np.ones((X,Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y-1] = 0.
        maze[x][Y2] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X-1][y] = 0.
        maze[X2][y] = 0.
        maze[X2][(int) (Y2/2)] = 1.
        maze[X2][(int) (3*Y2/2)] = 1.
        maze[(int) (X2/2)][Y2] = 1.
        maze[(int) (3*X2/2)][Y2] = 1.
    return maze

def twoRoom(X,Y):
    X2 = (int) (X/2)
    maze = np.ones((X,Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y-1] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X-1][y] = 0.
        maze[X2][y] = 0.
    maze[X2][ (int) (Y/2)] = 1.
    return maze

class GridWorld_withWall(environments.discreteMDP.DiscreteMDP):
    """


    """

    metadata = {'render.modes': ['text', 'ansi', 'pylab'], 'maps': ['random','2-rrom', '4-room']}

    def __init__(self, sizeX,sizeY, map_name="random", slippery=0.1,nbGoals=1,rewardStd=0.,density=0.2, lengthofwalks=5, initialSingleStateDistribution=False):
        # initialSingleStateDistribution: If set to True, the initial distribution is a Dirac at one state (this state is uniformly chosen amongts valid non-goal states)
        # If set to False, then the initial distribution is set to be uniform over all valid non-goal states.

        #desc = maps[map_name]
        self.sizeX, self.sizeY = sizeX, sizeY
        self.reward_range = (0, 1)
        self.rewardStd=rewardStd

        self.nA = 4
        self.nS = sizeX * sizeY
        self.nameActions= ["Up", "Down", "Left", "Right"]

        self.initializedRender=False

        #stochastic transitions
        slip=min(slippery,1./3.)
        self.massmap = [[slip, 1.-3*slip, slip, 0., slip],  # up : up down left right stay
                   [slip, 0., slip, 1.-3*slip, slip],  # down
                   [1.-3*slip, slip, 0., slip, slip],  # left
                   [0., slip, 1.-3*slip, slip, slip]]  # right

        if (map_name=="2-room"):
            self.maze=twoRoom(sizeX, sizeY)
        elif (map_name=="4-room"):
            self.maze = fourRoom(sizeX, sizeY)
        else:
            self.maze = randomGridWorld(sizeX, sizeY, density, lengthofwalks)


        self.goalstates=self.makeGoalStates(nbGoals)
        if (initialSingleStateDistribution):
            isd=self.makeInitialSingleStateDistribution(self.maze)
        else:
            isd=self.makeInitialDistribution(self.maze)
        P = self.makeTransition(isd)
        R = self.makeRewards()

        super(GridWorld, self).__init__(self.nS, self.nA, P, R, isd,nameActions=self.nameActions)

    def to_s(self,rowcol):
            return rowcol[0] * self.sizeY + rowcol[1]

    def from_s(self,s):
            return s//self.sizeY, s%self.sizeY

    def makeGoalStates(self, nb):
        goalstates = []
        for g in range(nb):
            s = [0, 0]#[np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
            while (self.maze[s[0]][s[1]] == 0):
                s = [self.sizeX - 2, self.sizeY - 2]#[np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
            goalstates.append(self.to_s(s))
            self.maze[s[0]][s[1]] = 2.
        return goalstates


    def makeInitialSingleStateDistribution(self,maze):

        xy = [1,1]#[np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
        while (self.maze[xy[0]][xy[1]] != 1):
            xy = [np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
        isd = np.array(maze == -1.).astype('float64').ravel()
        isd[self.to_s(xy)]=1.
        return isd

    def makeInitialDistribution(self,maze):
         isd = np.array(maze == 1.).astype('float64').ravel()
         isd /= isd.sum()
         return isd

    def makeTransition(self,initialstatedistribution):
            X = self.sizeX
            Y = self.sizeY
            P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
            nbempty=0

            for s in range(self.nS):
                x,y = self.from_s(s)
                if (self.maze[x][y] == 2.):
                    rew = 1.
                    for a in range(self.nA):
                        li = P[s][a]
                        for ns in range(self.nS):
                            #rw = stat.norm(loc=1.,scale=self.rewardStd)
                            #li.append((initialstatedistribution[ns],ns,rw,False))
                            if(initialstatedistribution[ns] > 0):
                                li.append((initialstatedistribution[ns],ns,False))
                else:
                    us = [(x - 1) % X, y % Y]
                    ds = [(x + 1) % X, y % Y]
                    ls = [x % X, (y - 1) % Y]
                    rs = [x % X, (y + 1) % Y]
                    ss=[x,y]
                    if (self.maze[us[0]][us[1]] <= 0 or self.maze[x][y] <= 0): us = ss
                    if (self.maze[ds[0]][ds[1]] <= 0 or self.maze[x][y] <= 0): ds = ss
                    if (self.maze[ls[0]][ls[1]] <= 0 or self.maze[x][y] <= 0): ls = ss
                    if (self.maze[rs[0]][rs[1]] <= 0 or self.maze[x][y] <= 0): rs = ss
                    #rew = stat.norm(loc=0., scale=self.rewardStd)
                    for a in range(self.nA):
                        li = P[s][a]
                        #li.append((1.0, newstate, rew, done))
                        #li.append((1.0, newstate, done))
                        li.append((self.massmap[a][0],self.to_s(ls),False))
                        li.append((self.massmap[a][1],self.to_s(us),False))
                        li.append((self.massmap[a][2],self.to_s(rs),False))
                        li.append((self.massmap[a][3],self.to_s(ds),False))
                        li.append((self.massmap[a][4],self.to_s(ss),False))

            return P

    def makeRewards(self):
        if (self.rewardStd>0):
            R = {s: {a: stat.norm(loc=0.,scale=self.rewardStd) for a in range(self.nA)} for s in range(self.nS)}
        else:
            R ={s: {a: stat.rv_discrete(name='custm', values=(
            [0.], [1.])) for a in range(self.nA)} for s in range(self.nS)}


        for s in range(self.nS):
            x, y = self.from_s(s)
            if (self.maze[x][y] == 2.):
                for a in range(self.nA):
                    if (self.rewardStd > 0):
                        R[s][a] = stat.norm(loc=1.,scale=self.rewardStd)
                    else:
                        R[s][a] = stat.rv_discrete(name='custm', values=([1.], [1.]))
        return R

    def getTransition(self,s,a):
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            transition[c[1]]=c[0]
        return transition

    def render(self, mode='text'):
        if (mode== 'pylab'):
            if (not self.initializedRender):
                self.initRender()
                self.initializedRender = True

            plt.figure(self.numFigure)
            row, col = self.from_s(self.s)
            v = self.maze[row][col]
            self.maze[row][col] = 1.5
            plt.imshow(self.maze, cmap='hot', interpolation='nearest')
            self.maze[row][col] = v
            plt.show(block=False)
            plt.pause(0.01)
        elif (mode=='text') or (mode == 'ansi'):
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            symbols = {0.:'X', 1.:'.',2.:'G'}
            desc = [[symbols[c] for c in line] for line in self.maze]
            row, col = self.from_s(self.s)
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(self.nameActions[self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc) + "\n")

            if mode != 'text':
                return outfile
        else:
            super(GridWorld, self).render(mode)



    def initRender(self):
        self.numFigure = plt.gcf().number
        plt.figure(self.numFigure)
        plt.imshow(self.maze, cmap='hot', interpolation='nearest')
        plt.savefig('MDP-gridworld.png')
        plt.show(block=False)
        plt.pause(0.5)







# Upgrade of the previous class, walls are no longer visible as unaccessible states for the learner (they're no longer exisiting for the learner).
class GridWorld(environments.discreteMDP.DiscreteMDP):

    metadata = {'render.modes': ['text', 'ansi', 'pylab'], 'maps': ['random','2-rrom', '4-room']}

    def __init__(self, sizeX,sizeY, map_name="random", slippery=0.1,nbGoals=1,rewardStd=0.,density=0.2, lengthofwalks=5, initialSingleStateDistribution=False):
        # initialSingleStateDistribution: If set to True, the initial distribution is a Dirac at one state (this state is uniformly chosen amongts valid non-goal states)
        # If set to False, then the initial distribution is set to be uniform over all valid non-goal states.
        
        #desc = maps[map_name]
        self.sizeX, self.sizeY = sizeX, sizeY
        self.reward_range = (0, 1)
        self.rewardStd=rewardStd

        self.nA = 4
        self.nS_all = sizeX * sizeY
        self.nameActions= ["Up", "Down", "Left", "Right"]

        self.initializedRender=False

        #stochastic transitions
        slip=min(slippery,1./3.)
        self.massmap = [[slip, 1.-3*slip, slip, 0., slip],  # up : up down left right stay
                   [slip, 0., slip, 1.-3*slip, slip],  # down
                   [1.-3*slip, slip, 0., slip, slip],  # left
                   [0., slip, 1.-3*slip, slip, slip]]  # right

        if (map_name=="2-room"):
            self.maze=twoRoom(sizeX, sizeY)
        elif (map_name=="4-room"):
            self.maze = fourRoom(sizeX, sizeY)
        else:
            self.maze = randomGridWorld(sizeX, sizeY, density, lengthofwalks)

        self.mapping = []
        for x in range(sizeX):
            for y in range(sizeY):
                if self.maze[x, y] >= 1:
                    self.mapping.append(self.to_s((x, y)))
        
        self.nS = len(self.mapping)

        self.goalstates=self.makeGoalStates(nbGoals)
        if (initialSingleStateDistribution):
            isd=self.makeInitialSingleStateDistribution(self.maze)
        else:
            isd=self.makeInitialDistribution(self.maze)
        P = self.makeTransition(isd)
        R = self.makeRewards()

        self.P = P
        self.R = R
        self.isd = isd
        self.lastaction=None # for rendering

        self.states = range(0,self.nS)
        self.actions = range(0,self.nA)
        self.nameActions = list(string.ascii_uppercase)[0:min(self.nA,26)]


        self.reward_range = (0, 1)
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.initializedRender=False
        self.seed()
        self.reset()

    def to_s(self,rowcol):
            return rowcol[0] * self.sizeY + rowcol[1]

    def from_s(self,s):
            return s//self.sizeY, s%self.sizeY

    def step(self, a):
        transitions = self.P[self.s][a]
        rewarddis = self.R[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, d= transitions[i]
        r =  clip(rewarddis.rvs(), self.reward_range)
        self.s = s
        self.lastaction=a
        s = self.mapping.index(s)
        return (s, r, d, "")

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction=None
        return self.mapping.index(self.s)

    def makeGoalStates(self, nb):
        goalstates = []
        for g in range(nb):
            s = [0, 0]#[np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
            while (self.maze[s[0]][s[1]] == 0):
                s = [self.sizeX - 2, self.sizeY - 2]#[np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
            goalstates.append(self.to_s(s))
            self.maze[s[0]][s[1]] = 2.
        return goalstates


    def makeInitialSingleStateDistribution(self,maze):

        xy = [1,1]#[np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
        while (self.maze[xy[0]][xy[1]] != 1):
            xy = [np.random.randint(self.sizeX), np.random.randint(self.sizeY)]
        isd = np.array(maze == -1.).astype('float64').ravel()
        isd[self.to_s(xy)]=1.
        return isd

    def makeInitialDistribution(self,maze):
         isd = np.array(maze == 1.).astype('float64').ravel()
         isd /= isd.sum()
         return isd

    def makeTransition(self,initialstatedistribution):
            X = self.sizeX
            Y = self.sizeY
            P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS_all)}
            nbempty=0

            for s in range(self.nS_all):
                x,y = self.from_s(s)
                if (self.maze[x][y] == 2.):
                    rew = 1.
                    for a in range(self.nA):
                        li = P[s][a]
                        for ns in range(self.nS_all):
                            #rw = stat.norm(loc=1.,scale=self.rewardStd)
                            #li.append((initialstatedistribution[ns],ns,rw,False))
                            if(initialstatedistribution[ns] > 0):
                                li.append((initialstatedistribution[ns],ns,False))
                else:
                    us = [(x - 1) % X, y % Y]
                    ds = [(x + 1) % X, y % Y]
                    ls = [x % X, (y - 1) % Y]
                    rs = [x % X, (y + 1) % Y]
                    ss=[x,y]
                    if (self.maze[us[0]][us[1]] <= 0 or self.maze[x][y] <= 0): us = ss
                    if (self.maze[ds[0]][ds[1]] <= 0 or self.maze[x][y] <= 0): ds = ss
                    if (self.maze[ls[0]][ls[1]] <= 0 or self.maze[x][y] <= 0): ls = ss
                    if (self.maze[rs[0]][rs[1]] <= 0 or self.maze[x][y] <= 0): rs = ss
                    #rew = stat.norm(loc=0., scale=self.rewardStd)
                    for a in range(self.nA):
                        li = P[s][a]
                        #li.append((1.0, newstate, rew, done))
                        #li.append((1.0, newstate, done))
                        li.append((self.massmap[a][0],self.to_s(ls),False))
                        li.append((self.massmap[a][1],self.to_s(us),False))
                        li.append((self.massmap[a][2],self.to_s(rs),False))
                        li.append((self.massmap[a][3],self.to_s(ds),False))
                        li.append((self.massmap[a][4],self.to_s(ss),False))

            return P

    def makeRewards(self):
        if (self.rewardStd>0):
            R = {s: {a: stat.norm(loc=0.,scale=self.rewardStd) for a in range(self.nA)} for s in range(self.nS_all)}
        else:
            R ={s: {a: stat.rv_discrete(name='custm', values=(
            [0.], [1.])) for a in range(self.nA)} for s in range(self.nS_all)}


        for s in range(self.nS_all):
            x, y = self.from_s(s)
            if (self.maze[x][y] == 2.):
                for a in range(self.nA):
                    if (self.rewardStd > 0):
                        R[s][a] = stat.norm(loc=1.,scale=self.rewardStd)
                    else:
                        R[s][a] = stat.rv_discrete(name='custm', values=([1.], [1.]))
        return R

    def getTransition(self,s,a):
        s = self.mapping[s]
        transition = np.zeros(self.nS)
        for c in self.P[s][a]:
            if c[1] in self.mapping:
                c1 = self.mapping.index(c[1])
                transition[c1]=c[0]
        return transition

    def render(self, mode='text'):
        if (mode== 'pylab'):
            if (not self.initializedRender):
                self.initRender()
                self.initializedRender = True

            plt.figure(self.numFigure)
            row, col = self.from_s(self.s)
            v = self.maze[row][col]
            self.maze[row][col] = 1.5
            plt.imshow(self.maze, cmap='hot', interpolation='nearest')
            self.maze[row][col] = v
            plt.show(block=False)
            plt.pause(0.01)
        elif (mode=='text') or (mode == 'ansi'):
            outfile = StringIO() if mode == 'ansi' else sys.stdout

            symbols = {0.:'X', 1.:'.',2.:'G'}
            desc = [[symbols[c] for c in line] for line in self.maze]
            row, col = self.from_s(self.s)
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(self.nameActions[self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc) + "\n")

            if mode != 'text':
                return outfile
        else:
            super(GridWorld, self).render(mode)



    def initRender(self):
        self.numFigure = plt.gcf().number
        plt.figure(self.numFigure)
        plt.imshow(self.maze, cmap='hot', interpolation='nearest')
        plt.savefig('MDP-gridworld.png')
        plt.show(block=False)
        plt.pause(0.5)