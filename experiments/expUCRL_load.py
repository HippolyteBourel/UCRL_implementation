from learners.UCRL import *
from learners.Random import *
from learners.Optimal import *
from learners.KL_UCRL import *
from learners.C_UCRL_C import *
from learners.C_UCRL import *
from learners.C_UCRL_old import *
from learners.UCRL2_L import *
#from learners.KL_UCRL_L import *
#from learners.UCRL2_MSC import *
from learners.C_UCRL_C_MSC import *
from learners.C_UCRL_MSC import *
from learners.UCRL_Thompson import *
from learners.UCRL2_local import *
from learners.UCRL2_local2 import *
from learners.UCRL2_L_sqrtSC import *
from learners.C_UCRL_C_sqrtSC import *
from learners.C_UCRL_sqrtSC import *
from learners.UCRL2_peeling import *
from learners.UCRL2_Bernstein import *
from learners.C_UCRL_C_sigma import *
from learners.SCAL import *
from learners.UCRL_Lplus import *
from learners.UCRL3 import *
from learners.UCRL3_old import *
from environments import equivalence
#from learners.ImprovedMDPLearner2 import *
from environments import gridworld, discreteMDP
from utils import *

def run_exp(rendermode='', testName = "riverSwim"):
    timeHorizon=100000
    nbReplicates=10
    
    if testName == "random_grid":
        env, nbS, nbA = buildGridworld(sizeX=8,sizeY=5,map_name="random",rewardStd=0.01, initialSingleStateDistribution=True)
    elif testName == "2-room":
        env, nbS, nbA = buildGridworld(sizeX=9, sizeY=11, map_name="2-room", rewardStd=0.0, initialSingleStateDistribution=True)
    elif testName == "4-room":
        env, nbS, nbA = buildGridworld(sizeX=7, sizeY=7, map_name="4-room", rewardStd=0.0, initialSingleStateDistribution=True)
    elif testName == "random":
        env, nbS, nbA = buildRandomMDP(nbStates=6,nbActions=3, maxProportionSupportTransition=0.25, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.5)
    elif testName == "three-state":
        ns_river = 1
        env, nbS, nbA = buildThreeState(delta = 0.005)
    elif testName == "three-state-bernoulli":
        ns_river = 1
        env, nbS, nbA = buildThreeState(delta = 0.005, fixed_reward = False)
    elif testName == "riverSwimErgo":
        ns_river = 6
        env, nbS, nbA = buildRiverSwimErgo(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
    elif testName == "riverSwimErgo25":
        ns_river = 25
        env, nbS, nbA = buildRiverSwimErgo(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
    elif testName == "riverSwimErgo50":
        ns_river = 50
        env, nbS, nbA = buildRiverSwimErgo(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
    elif testName == "riverSwim25_shuffle":
        ns_river = 25
        env, nbS, nbA = buildRiverSwim_shuffle(nbStates=ns_river, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.4, rightProbaLeft2=0.05)
    elif testName == "riverSwim25_biClass":
        ns_river = 25
        env, nbS, nbA = buildRiverSwim_biClass(nbStates=ns_river, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.4, rightProbaLeft2=0.05)
    else:
        if testName == "riverSwim25":
            ns_river = 25
        else:
            ns_river = 6
        env, nbS, nbA = buildRiverSwim(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
    
    if testName == "random_grid" or testName == "2-room" or testName == "4-room":
        equivalence.displayGridworldEquivalenceClasses(env.env, 0.)
        equivalence.displayGridworldAggregationClasses(env.env)
    C, nC = equivalence.compute_C_nC(env.env)
    
    profile_mapping = equivalence.compute_sigma(env.env)
    sizeSupport = equivalence.compute_sizeSupport(env.env)
    
    print("*********************************************")
    
    cumRewards = []
    names = []
    
    cumRewards1 = []
    cumRewards2 = []
    cumRewards3 = []
    cumRewards4 = []
    cumRewards5 = []
    cumRewards6 = []
    cumRewards7 = []
    cumRewards8 = []
    cumRewards9 = []
    cumRewards10 = []
    cumRewards11 = []
    cumRewards12 = []
    
    for i in range(2):
        sup = '_' + str(i)
    
        #learner1 = SCAL_L( nbS,nbA, delta=0.05, c = 10)
        #names.append(learner1.name())
        #cumRewards1 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner1.name() + "_" + str(timeHorizon) + sup), 'rb'))
    
        #learner2 = C_UCRL_C_sigma_plus( nbS,nbA, delta=0.05, C = C, nC = nC, profile_mapping = profile_mapping)
        #names.append(learner2.name())
        #cumRewards2 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner2.name() + "_" + str(timeHorizon) + sup), 'rb'))
        
        #learner3 = C_UCRL_C14_Nrwd_Lplus_local3( nbS,nbA, delta=0.05, C = C, nC = nC)
        #names.append(learner3.name())
        #cumRewards3 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner3.name() + "_" + str(timeHorizon) + sup), 'rb'))
        
        learner4 = UCRL3( nbS,nbA, delta=0.05)#, C = C, nC = nC, sizeSupport = sizeSupport)
        names.append("UCRL3")#learner4.name())
        cumRewards4 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner4.name() + "_" + str(timeHorizon) + sup), 'rb'))
        
        learner5 = UCRL2_boost( nbS,nbA, delta=0.05)#, C = C, nC = nC)
        names.append(learner5.name())
        cumRewards5 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner5.name() + "_" + str(timeHorizon) + sup), 'rb'))
        
        learner6 = UCRL2_L_boost( nbS,nbA, delta=0.05)#, c = 5)
        names.append("UCRL2-L")#learner6.name())
        cumRewards6 += pickle.load(open(("results/cumRewards_" + testName + "_" + "UCRL2_L" + "_" + str(timeHorizon) + sup), 'rb'))
    
        #learner7 = UCRL3_old( nbS,nbA, delta=0.05)#, C = C, nC = nC)
        #names.append(learner7.name())
        #cumRewards7 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner7.name() + "_" + str(timeHorizon) + sup), 'rb'))
    
        #learner8 = C_UCRL_C14_Lplus_local3( nbS,nbA, delta=0.05, C = C, nC = nC)
        #names.append(learner8.name())
        #cumRewards8 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner8.name() + "_" + str(timeHorizon) + sup), 'rb'))
        
        #learner9 = UCRL2_boost( nbS,nbA, delta=0.05)#, c = 2)
        #names.append(learner9.name())
        #cumRewards9 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner9.name() + "_" + str(timeHorizon) + sup), 'rb'))
        
        learner10 = UCRL2_Bernstein( nbS,nbA, delta=0.05)#, c = 5)
        names.append("UCRL2-B")#learner10.name())
        cumRewards10 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner10.name() + "_" + str(timeHorizon) + sup), 'rb'))
    
        learner11 = KL_UCRL( nbS,nbA, delta=0.05)#, c = 5)
        names.append(learner11.name())
        cumRewards11 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner11.name() + "_" + str(timeHorizon) + sup), 'rb'))
        
        #learner12 = SCAL2_L( nbS,nbA, delta=0.05, c = 2)
        #names.append(learner12.name())
        #cumRewards12 += pickle.load(open(("results/cumRewards_" + testName + "_" + learner12.name() + "_" + str(timeHorizon) + sup), 'rb'))
    
    
    if testName == "4-room":
        opti_learner = Opti_77_4room(env.env)
    elif testName == "2-room":
        opti_learner = Opti_911_2room(env.env)
    elif testName == "random_grid" or testName == "2-room" or testName == "4-room":
        print("Computing an estimate of the optimal policy (for regret)...")
        opti_learner = Opti_learner(env.env, nbS, nbA)
        print("Done, the estimation of the optimal policy : ")
        print(opti_learner.policy)
    else:
        opti_learner = Opti_swimmer(env)
    
    cumReward_opti = cumulativeRewards(env,opti_learner,1,min((100000, 5 * timeHorizon)),rendermode)
    gain =  cumReward_opti[0][-1] / (min((100000, 5 * timeHorizon)))
    opti_reward = [[t * gain for t in range(timeHorizon)]]
    
    if len(cumRewards1) != 0:
        cumRewards.append(cumRewards1)
    if len(cumRewards2) != 0:
        cumRewards.append(cumRewards2)
    if len(cumRewards3) != 0:
        cumRewards.append(cumRewards3)
    if len(cumRewards4) != 0:
        cumRewards.append(cumRewards4)
    if len(cumRewards5) != 0:
        cumRewards.append(cumRewards5)
    if len(cumRewards6) != 0:
        cumRewards.append(cumRewards6)
    if len(cumRewards7) != 0:
        cumRewards.append(cumRewards7)
    if len(cumRewards8) != 0:
        cumRewards.append(cumRewards8)
    if len(cumRewards9) != 0:
        cumRewards.append(cumRewards9)
    if len(cumRewards10) != 0:
        cumRewards.append(cumRewards10)
    if len(cumRewards11) != 0:
        cumRewards.append(cumRewards11)
    if len(cumRewards12) != 0:
        cumRewards.append(cumRewards12)
    
    cumRewards.append(opti_reward)
    
    #names = ["C-UCRL(classes, profiles)", "C-UCRL", "UCRL2-L"]
    
    plotCumulativeRegrets(names, cumRewards, timeHorizon, testName, ysemilog = True)
    plotCumulativeRegrets(names, cumRewards, timeHorizon, testName)#, semilog = True)
    
    print("*********************************************")

#run_exp(rendermode='pylab')    #Pylab rendering
#run_exp(rendermode='text')    #Text rendering
run_exp(rendermode='', testName = 'riverSwim')#'4-room')#, testName = 'three-state')#-bernoulli')#, testName = '4-room')        #No rendering

