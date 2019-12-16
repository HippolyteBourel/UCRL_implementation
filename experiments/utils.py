from environments import gridworld, discreteMDP
import pylab as pl
import gym
import pickle
from gym.envs.registration import  register
import numpy as np

def buildGridworld(sizeX=10,sizeY=10,map_name="4-room",rewardStd=0., initialSingleStateDistribution=False,max_steps=np.infty,reward_threshold=np.infty):
    register(
        id='Gridworld'+map_name+'-v0',
        entry_point='environments.gridworld:GridWorld',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'sizeX': sizeX,'sizeY':sizeY,'map_name':map_name,'rewardStd':rewardStd, 'initialSingleStateDistribution':initialSingleStateDistribution}
    )
    g = gym.make('Gridworld'+map_name+'-v0')
    return g, g.env.nS, 4

def buildRandomMDP(nbStates=5,nbActions=4, max_steps=np.infty,reward_threshold=np.infty, maxProportionSupportTransition=0.5, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.2, minNonZeroReward=0.3, rewardStd=0.5):
    register(
        id='RandomMDP-v0',
        entry_point='environments.discreteMDP:RandomMDP',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'nbActions': nbActions, 'maxProportionSupportTransition': maxProportionSupportTransition, 'maxProportionSupportReward': maxProportionSupportReward,
                'maxProportionSupportStart': maxProportionSupportStart, 'minNonZeroProbability':minNonZeroProbability, 'minNonZeroReward':minNonZeroReward, 'rewardStd':rewardStd }
    )

    return gym.make('RandomMDP-v0'), nbStates, nbActions



def buildRiverSwim(nbStates=5, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):
    register(
        id='RiverSwim-v0',
        entry_point='environments.discreteMDP:RiverSwim',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, }
    )

    return gym.make('RiverSwim-v0'), nbStates,2

def buildRiverSwimErgo(nbStates=5, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):
    register(
        id='RiverSwimErgo-v' + str(nbStates),
        entry_point='environments.discreteMDP:RiverSwimErgo',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, }
    )

    return gym.make('RiverSwimErgo-v' + str(nbStates)), nbStates,2


def buildRiverSwim_biClass(nbStates=5, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.5, rightProbaLeft2=0.05):
    register(
        id='RiverSwim_biClass-v' + str(nbStates),
        entry_point='environments.discreteMDP:RiverSwim_biClass',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, 'rightProbaright2': rightProbaright2, 'rightProbaLeft2': rightProbaLeft2, }
    )

    return gym.make('RiverSwim_biClass-v' + str(nbStates)), nbStates,2

def buildRiverSwim_shuffle(nbStates=5, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.5, rightProbaLeft2=0.05):
    register(
        id='RiverSwim_shuffle-v' + str(nbStates),
        entry_point='environments.discreteMDP:RiverSwim_shuffle',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'nbStates': nbStates, 'rightProbaright': rightProbaright, 'rightProbaLeft': rightProbaLeft,
                'rewardL': rewardL, 'rewardR':rewardR, 'rightProbaright2': rightProbaright2, 'rightProbaLeft2': rightProbaLeft2, }
    )

    return gym.make('RiverSwim_shuffle-v' + str(nbStates)), nbStates,2


def buildThreeState(delta = 0.005, max_steps=np.infty,reward_threshold=np.infty, fixed_reward = True):
    register(
        id='ThreeState-v0',
        entry_point='environments.discreteMDP:ThreeState',
        max_episode_steps=max_steps,
        reward_threshold=reward_threshold,
        kwargs={'delta': delta, 'fixed_reward': fixed_reward, }
    )

    return gym.make('ThreeState-v0'), 3, 2

def cumulativeRewards(env,learner,nbReplicates,timeHorizon,rendermode='pylab', reverse = False):
    cumRewards = []
    if(rendermode==''):
        for i_episode in range(nbReplicates):
            observation = env.reset()
            learner.reset(observation)
            cumreward = 0.
            cumrewards = []
            print("New initialization of ", learner.name())
            print("Initial state:" + str(observation))
            for t in range(timeHorizon):
                state = observation
                action = learner.play(state)  # Get action
                if reverse:
                    observation, reward, done, info = env.step((action + 1) % 2)
                else:
                    observation, reward, done, info = env.step(action)
                learner.update(state, action, reward, observation)  # Update learners
                cumreward += reward
                cumrewards.append(cumreward)

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
            cumRewards.append(cumrewards)
            print("Cumreward:" + str(cumreward))
    else:
        for i_episode in range(nbReplicates):
            observation = env.reset()
            learner.reset(observation)
            cumreward=0.
            cumrewards=[]
            print("New initialization of ", learner.name())
            print("Initial state:"+str(observation))
            for t in range(timeHorizon):
                state=observation
                env.render(rendermode)
                action = learner.play(state) #Get action
                observation, reward, done, info = env.step(action)
                #print("S:"+str(state)+" A:"+str(action) + " R:"+str(reward)+" S:"+str(observation) +" done:"+str(done) +"\n")
                learner.update(state, action, reward, observation)  # Update learners
                cumreward+=reward
                cumrewards.append(cumreward)

                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
            cumRewards.append(cumrewards)
            print("Cumreward:"+str(cumreward))
    return cumRewards

def plotCumulativeRegret(name,cumulativerewards_, timeHorizon,nbFigure=1):
    #print(len(cumulativerewards_[0]), cumulativerewards_[0])
    avg_cum_r = np.mean(cumulativerewards_, axis=0)
    #print(len(avg_cum_r), avg_cum_r)
    std_cum_r = np.std(cumulativerewards_, axis=0)
    nbFigure = pl.gcf().number
    pl.figure(nbFigure)
    pl.plot(avg_cum_r)
    step=timeHorizon//10
    pl.errorbar(np.arange(0,timeHorizon,step), avg_cum_r[0:timeHorizon:step], std_cum_r[0:timeHorizon:step], linestyle='None', capsize=10)
    pl.xlabel("Time steps", fontsize=10)
    pl.ylabel("Cumulative regret", fontsize=10)
    #pl.show()
    #pl.savefig("../experiments/Figure-" + name + "-cumulativerewards_rewards" + '.png')
    pl.savefig("results/Figure-" + name + "-cumulativerewards_rewards" + '.pdf')


# To plot the regret over timepsteps of n different learner,names is the list of their names, cumulativerewards_ is the list of their cumulative
# reward + the one of the optimal policy used to compute the regret. Some optimal policy are given in the Optimal file in the learner folder.
def plotCumulativeRegrets(names,cumulativerewards_, timeHorizon, testName = "riverSwim", semilog = False, ysemilog = False):
    #print(len(cumulativerewards_[0]), len(cumulativerewards_))#[0])#print(len(cumulativerewards_[0]), cumulativerewards_[0])
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    textfile = "results/Regret-"
    colors= ['black', 'blue','gray', 'green', 'red']#['black', 'purple', 'blue','cyan','yellow', 'orange', 'red', 'chocolate']
    avgcum_rs= [ np.mean(cumulativerewards_[-1], axis=0) - np.mean(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_) - 1)]
    std_cum_rs= [ 1.96 * np.std(cumulativerewards_[i], axis=0) / np.sqrt(len(cumulativerewards_[i])) for  i in range(len(cumulativerewards_) - 1)]
    for i in range(len(cumulativerewards_) - 1):
        pl.plot(avgcum_rs[i], label=names[i],color=colors[i%len(colors)])
        step=timeHorizon//10
        pl.errorbar(np.arange(0,timeHorizon,step), avgcum_rs[i][0:timeHorizon:step], std_cum_rs[i][0:timeHorizon:step], color=colors[i%len(colors)], linestyle='None', capsize=10)
        textfile+=names[i]+"-"
        print(names[i], ' has regret ', avgcum_rs[i][-1], ' after ', len(avgcum_rs[i]), ' time steps with variance ', std_cum_rs[i][-1])
        #pl.show()
    pl.legend()
    pl.xlabel("Time steps", fontsize=13, fontname = "Arial")
    pl.ylabel("Regret", fontsize=13, fontname = "Arial")
    pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))
    if semilog:
        pl.xscale('log')
        textfile += "_xsemilog"
    else:
        pl.xlim(0, timeHorizon)
    if ysemilog:
        pl.yscale('log')
        textfile += "_ysemilog"
        pl.ylim(100)
    else:
        pl.ylim(0)
    #pl.savefig(textfile + testName + '.png')
    pl.savefig(textfile + testName + '.pdf')

# This function computes nC and C parameters of C_UCRL_C for a given MDP, nC being the number of classes in this MDP and C its the matrix which
# associate each pair of state-action (s, a) to its class (represented by a natural c in [0, nC - 1]).
def computeC(env): # TODO
    nC = 0
    C = np.zeros([S, A])
    return nC, C

def plotSpan(names,cumulativerewards_, timeHorizon, testName = "riverSwim"):
    #print(len(cumulativerewards_[0]), cumulativerewards_[0])
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    textfile = "results/Span-"
    colors= ['black', 'purple', 'blue','cyan','yellow', 'orange', 'red']
    avgcum_rs= [ np.mean(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_))]
    #std_cum_rs= [ np.std(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_))]
    #std_cum_rs= [ 1.96 * np.std(cumulativerewards_[i], axis=0) / np.sqrt(len(cumulativerewards_[i])) for  i in range(len(cumulativerewards_) - 1)]
    for i in range(len(cumulativerewards_)):
        pl.plot(avgcum_rs[i], label=names[i],color=colors[i%len(colors)])
        step=timeHorizon//10
        #pl.errorbar(np.arange(0,timeHorizon,step), avgcum_rs[i][0:timeHorizon:step], std_cum_rs[i][0:timeHorizon:step], color=colors[i%len(colors)], linestyle='None', capsize=10)
        textfile+=names[i]+"-"
        #pl.show()
    pl.legend()
    pl.xlabel("Time steps", fontsize=10)
    pl.ylabel("Regret", fontsize=10)
    #pl.savefig(textfile + testName + '.png')
    pl.savefig(textfile + testName + '.pdf')
    
def SpanList(env,learner,nbReplicates,timeHorizon,rendermode='pylab', reverse = False):
    spanList = []
    for i_episode in range(nbReplicates):
        observation = env.reset()
        learner.reset(observation)

        print("New initialization of ", learner.name())
        print("Initial state:" + str(observation))
        spanlist = []
        for t in range(timeHorizon):
            state = observation
            action = learner.play(state)  # Get action
            if reverse:
                observation, reward, done, info = env.step((action + 1) % 2)
            else:
                observation, reward, done, info = env.step(action)
            learner.update(state, action, reward, observation)  # Update learners
            span = learner.span[-1]
            spanlist.append(span)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        print("Span : ", span)
        spanList.append(spanlist)
    return spanList

def f(N, nS, nA, delta = 0.05):
    temp = 2 * (1 + 1 / N) * np.log(np.sqrt(N + 1) * (2**nS - 2) / (delta / (nS * nA)))
    return np.sqrt(temp / N)

def cumulativeRewards_testN(env,learner,nbReplicates,timeHorizon, Delta, C, nC, rendermode='pylab', reverse = False, indic_cluster = True):
    cumRewards = []
    indics = []
    print("Delta = ", Delta)
    for i_episode in range(nbReplicates):
        observation = env.reset()
        learner.reset(observation)
        cumreward = 0.
        cumrewards = []
        indic = []
        nS = env.env.nS
        nA = env.env.nA
        N = np.zeros((nS, nA))
        print("New initialization of ", learner.name())
        print("Initial state:" + str(observation))
        for t in range(timeHorizon):
            state = observation
            action = learner.play(state)  # Get action
            if reverse:
                observation, reward, done, info = env.step((action + 1) % 2)
            else:
                observation, reward, done, info = env.step(action)
            learner.update(state, action, reward, observation)  # Update learners
            cumreward += reward
            cumrewards.append(cumreward)
            N[state, action] += 1
            if t%1000 == 0 and t != 0:
                test = np.zeros((nS, nA))
                test_c = np.zeros(nC)
                #test_C = True
                for s in range(nS):
                    for a in range(nA):
                        #test_C = test_C and (C[s, a] == learner.C[s, a])   On a pas forcément la même numerotation..
                        test[s, a] = (f(N[s, a], nS, nA) <= Delta)
                        test_c[C[s, a]] = (test_c[C[s, a]] or test[s, a])
                if not (False in test):
                    print("Test f(N) < Delta valid for all pairs (s, a) at timestep : ", t)
                if not (False in test_c):
                    print("Test f(N) < Delta valid for all c at timestep : ", t)
                #if nC == learner.nC:#test_C:
                #    print("Clustering could be the true one at timestep : ", t)
                #    print("True one : ", C, "\n Current one : ", learner.C)
                if indic_cluster:
                    indic.append(cluster_indicator(C, learner.C))
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        cumRewards.append(cumrewards)
        indics.append(indic)
        #print("At the end, N = ", learner.Nk)
        #print("At the end beta_r = ", learner.r_distances)
        print("Cumreward:" + str(cumreward))
    if indic_cluster:
        return cumRewards, indics
    #return cumRewards

def test_N(Delta, nS, nA):
    N = 1
    while f(N, nS, nA) > Delta:
        N += 1
    print("Minimum N for f(N) < Delta is :", N)
    
def cluster_indicator(C, CC):
    nS = len(C)
    nA = len(C[0])
    res = 0
    count = 0
    for s in range(nS - 1):
        for a in range(nA - 1):
            for ss in range(s + 1, nS):
                for aa in range(a + 1, nA):
                    count += 1
                    if (C[s, a] != C[ss, aa]) and (CC[s, a] == CC[ss, aa]):
                        res += 1
                        #return 1
    return (res / count) * 100#0

# Norme 1 of the difference between 2 vectors of same size.
def diffNorme1(v1, v2):
	res = 0
	for i in range(len(v1)):
		res += abs(v1[i] - v2[i])
	return res

def cluster_error(C, CC, p_estimate, nA, nS, N):
    nS = len(C)
    nA = len(C[0])
    res = 0
    count = 0
    for s in range(nS - 1):
        for a in range(nA - 1):
            for ss in range(s + 1, nS):
                for aa in range(a + 1, nA):
                    if (C[s, a] != C[ss, aa]) and (CC[s, a] == CC[ss, aa]):
                        res += (diffNorme1(compute_p_estimate_c(CC[s, a], p_estimate, CC, nA, nS, N), compute_p_estimate_c(C[s, a], p_estimate, C, nA, nS, N)) +
                                diffNorme1(compute_p_estimate_c(CC[ss, aa], p_estimate, CC, nA, nS, N), compute_p_estimate_c(C[ss, aa], p_estimate, C, nA, nS, N)))
                        #count += 1
                        #return 1
    return (res )#/ count) * 100#0

def cluster_error2(C, CC, p_estimate, nA, nS, N):
    nS = len(C)
    nA = len(C[0])
    res = 0
    count = 0
    for s in range(nS - 1):
        for a in range(nA - 1):
            for ss in range(s + 1, nS):
                for aa in range(a + 1, nA):
                    if (C[s, a] != C[ss, aa]) and (CC[s, a] == CC[ss, aa]):
                        res += (diffNorme1(compute_p_estimate_c(CC[s, a], p_estimate, CC, nA, nS, N), compute_p_estimate_c(CC[s, a], p_estimate, CC, nA, nS, N, (s, a))) +
                                diffNorme1(compute_p_estimate_c(CC[ss, aa], p_estimate, CC, nA, nS, N), compute_p_estimate_c(CC[ss, aa], p_estimate, CC, nA, nS, N, (ss, aa))))
                        count += 1
                        #print("qC = ", compute_p_estimate_c(CC[s, a], p_estimate, CC, nA, nS, N), "\and qC\sa = ", compute_p_estimate_c(CC[s, a], p_estimate, CC, nA, nS, N, (s, a)))
                        #print("qC = ", compute_p_estimate_c(CC[ss, aa], p_estimate, CC, nA, nS, N), "\and qC\sa = ", compute_p_estimate_c(CC[ss, aa], p_estimate, CC, nA, nS, N, (ss, aa)))
                        #return 1
    return (res )#/ count) * 100#0


def error_biggest_cluster(C, CC, nA, nS):
    nC = max([max(C[c]) for c in range(len(C))]) + 1
    nCC = max([max(CC[c]) for c in range(len(CC))]) + 1
    mat = np.zeros((nCC, nC))
    res = 0
    for s in range(nS):
        for a in range(nA):
            mat[CC[s, a], C[s, a]] += 1
    mc = [np.argmax(mat[cc]) for cc in range(nCC)]
    for s in range(nS):
        for a in range(nA):
            if C[s, a] != mc[CC[s, a]]:
                res += 1
    return res / (nA * nS)#, res2 / (nA * nS)

def error_biggest_cluster2(C, CC, nA, nS, p_estimate, N):
    nC = max([max(C[c]) for c in range(len(C))]) + 1
    nCC = max([max(CC[c]) for c in range(len(CC))]) + 1
    mat = np.zeros((nCC, nC))
    res = 0
    for s in range(nS):
        for a in range(nA):
            mat[CC[s, a], C[s, a]] += 1
    mc = [np.argmax(mat[cc]) for cc in range(nCC)]
    for s in range(nS):
        for a in range(nA):
            if C[s, a] != mc[CC[s, a]]:
                res += diffNorme1(compute_p_estimate_c(CC[s, a], p_estimate, CC, nA, nS, N), compute_p_estimate_c(CC[s, a], p_estimate, CC, nA, nS, N, (s, a)))
    return res / (nA * nS)
        

def plotIndics(names,cumulativerewards_, timeHorizon, testName = "riverSwim", semilog = False):
    #print(len(cumulativerewards_[0]), len(cumulativerewards_))#[0])
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    textfile = "results/IndicCluster-"
    colors= ['black', 'blue', 'purple','cyan','yellow', 'orange', 'red']
    avgcum_rs= [ np.mean(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_))]
    #std_cum_rs= [ np.std(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_))]
    std_cum_rs= [ 1.96 * np.std(cumulativerewards_[i], axis=0) / np.sqrt(len(cumulativerewards_[i])) for  i in range(len(cumulativerewards_))]
    for i in range(len(cumulativerewards_)):
        pl.plot(avgcum_rs[i], label=names[i],color=colors[i%len(colors)])
        step=(timeHorizon//10)//1000
        pl.errorbar(np.arange(0,(timeHorizon//1000),step), avgcum_rs[i][0:(timeHorizon//1000):step], std_cum_rs[i][0:(timeHorizon//1000):step], color=colors[i%len(colors)], linestyle='None', capsize=10)
        textfile+=names[i]+"-"
        #pl.show()
    pl.legend()
    pl.xlabel("Time (x1000)", fontsize=11, fontname = "Arial")
    pl.ylabel("Mis-clustering ratio", fontsize=11, fontname = "Arial")
    pl.xlim(0, timeHorizon // 1000)
    if semilog:
        pl.yscale('log')
        textfile += "_ysemilog_"
    #pl.savefig(textfile + testName + '.png')
    pl.savefig(textfile + testName + '.pdf')
    
    
def plotErrors(names,cumulativerewards_, timeHorizon, testName = "riverSwim", semilog = False):
    #print(len(cumulativerewards_[0]), cumulativerewards_[0])
    nbFigure = pl.gcf().number+1
    pl.figure(nbFigure)
    textfile = "results/ErrorCluster-"
    colors= ['black', 'blue', 'purple','cyan','yellow', 'orange', 'red']
    avgcum_rs= [ np.mean(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_))]
    #std_cum_rs= [ np.std(cumulativerewards_[i], axis=0) for  i in range(len(cumulativerewards_))]
    std_cum_rs= [ 1.96 * np.std(cumulativerewards_[i], axis=0) / np.sqrt(len(cumulativerewards_[i])) for  i in range(len(cumulativerewards_))]
    for i in range(len(cumulativerewards_)):
        pl.plot(avgcum_rs[i], label=names[i],color=colors[i%len(colors)])
        step=(timeHorizon//10)//1000
        pl.errorbar(np.arange(0,(timeHorizon//1000),step), avgcum_rs[i][0:(timeHorizon//1000):step], std_cum_rs[i][0:(timeHorizon//1000):step], color=colors[i%len(colors)], linestyle='None', capsize=10)
        textfile+=names[i]+"-"
        #pl.show()
    pl.legend()
    pl.xlabel("Time (x1000)", fontsize=11, fontname = "Arial")
    pl.ylabel("Mis-clustering bias", fontsize=11, fontname = "Arial")
    pl.xlim(0, timeHorizon // 1000)
    if semilog:
        pl.yscale('log')
        textfile += "_ysemilog_"
    #pl.savefig(textfile + testName + '.png')
    pl.savefig(textfile + testName + '.pdf')
    
    
def compute_p_estimate_c(c, p_estimate, C, nA, nS, N, nsa = (-1, -1)):
    res = np.zeros(nS)
    count = 0
    for s in range(nS):
        for a in range(nA):
            if C[s, a] == c and (s, a) != nsa:
                count += N[s, a]
                res += N[s, a] * p_estimate[s, a]
    return res / count
    
    
def cumulativeRewards_testC(env,learner,nbReplicates,timeHorizon, Delta, C, nC, rendermode='pylab', version = 1):
    cumRewards = []
    indics = []
    if version == 5:
        errors = []
    print("Delta = ", Delta)
    for i_episode in range(nbReplicates):
        observation = env.reset()
        learner.reset(observation)
        cumreward = 0.
        cumrewards = []
        indic = []
        if version == 5:
            error = []
        nS = env.env.nS
        nA = env.env.nA
        N = np.zeros((nS, nA))
        print("New initialization of ", learner.name())
        print("Initial state:" + str(observation))
        for t in range(timeHorizon):
            state = observation
            action = learner.play(state)  # Get action
            observation, reward, done, info = env.step(action)
            learner.update(state, action, reward, observation)  # Update learners
            cumreward += reward
            cumrewards.append(cumreward)
            N[state, action] += 1
            if t%1000 == 0 and t != 0:
                #indic.append(cluster_indicator(C, learner.C))
                if version == 1:
                    indic.append(cluster_error(C, learner.C, learner.p_estimate, nA, nS, N))
                if version == 2:
                    indic.append(cluster_error2(C, learner.C, learner.p_estimate, nA, nS, N))
                if version == 3:
                    indic.append(error_biggest_cluster(C, learner.C, nA, nS))
                if version == 4:
                    indic.append(error_biggest_cluster2(C, learner.C, nA, nS, learner.p_estimate, N))
                if version == 5:
                    indic.append(error_biggest_cluster(C, learner.C, nA, nS))
                    error.append(error_biggest_cluster2(C, learner.C, nA, nS, learner.p_estimate, N))
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        cumRewards.append(cumrewards)
        indics.append(indic)
        if version == 5:
            errors.append(error)
        #print("At the end, N = ", learner.Nk)
        #print("At the end beta_r = ", learner.r_distances)
        print("Cumreward:" + str(cumreward))
    if version == 5:
        return errors, indics
    return cumRewards, indics
    #return cumRewards
    