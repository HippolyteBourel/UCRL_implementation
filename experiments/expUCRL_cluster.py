from learners.UCRL import *
from learners.Random import *
from learners.Optimal import *
from learners.KL_UCRL import *
from learners.C_UCRL_C import *
from learners.C_UCRL import *
from learners.C_UCRL_old import *
from learners.UCRL2_L import *
#from learners.KL_UCRL_L import *
from learners.UCRL2_MSC import *
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
from environments import equivalence
#from learners.ImprovedMDPLearner2 import *
from utils import *

def run_exp(rendermode='', testName_List = ["riverSwim"], sup = ""):
    timeHorizon=1000000
    nbReplicates=10
    
    indics = []
    CumRewards = []
    
    print("*********************************************")
    
    for testName in testName_List:
        if testName == "random_grid":
            env, nbS, nbA = buildGridworld(sizeX=8,sizeY=5,map_name="random",rewardStd=0.01, initialSingleStateDistribution=True)
        elif testName == "2-room":
            env, nbS, nbA = buildGridworld(sizeX=5, sizeY=10, map_name="2-room", rewardStd=0.01, initialSingleStateDistribution=True)
        elif testName == "4-room":
            env, nbS, nbA = buildGridworld(sizeX=7, sizeY=7, map_name="4-room", rewardStd=0.01, initialSingleStateDistribution=True)
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
        else:
            if testName == "riverSwim25":
                ns_river = 25
            else:
                ns_river = 6
            env, nbS, nbA = buildRiverSwim(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
    
        if testName != "riverSwim" and testName != "three-state" and testName != "three-state-bernoulli" and testName != "riverSwim25" and testName != "riverSwimErgo" and testName != "riverSwimErgo25"  and testName != "riverSwimErgo50":
            equivalence.displayGridworldEquivalenceClasses(env.env, 0.)
            equivalence.displayGridworldAggregationClasses(env.env)
        C, nC = equivalence.compute_C_nC(env.env)
        
        profile_mapping = equivalence.compute_sigma(env.env)
        
        Delta = equivalence.compute_Delta(env.env, profile_mapping)
        
        test_N(Delta, env.env.nS, env.env.nA)
        
        print('****** Starting test on ', testName, ' ******')
        
        cumRewards = []
        names = []
    
        learner = C_UCRL_div4C( nbS,nbA, delta=0.05)#, c = 15)#, C = C, nC = nC)
        cumRewards, indic = cumulativeRewards_testC(env,learner,nbReplicates,timeHorizon,Delta,C,nC,rendermode, version = 5) # many tests are covered by this function, the version = 5 is to use the lasts tests (the used in the ACML submission)
        indics.append(indic)
        CumRewards.append(cumRewards)
        pickle.dump(indic, open(("results/indic_" + testName + "_" + learner.name() + "_" + str(timeHorizon)) + sup, 'wb'))
        pickle.dump(cumRewards, open(("results/error_" + testName + "_" + learner.name() + "_" + str(timeHorizon)) + sup, 'wb'))
           
    plotIndics(testName_List, indics, timeHorizon, learner.name())#, semilog = True)
    plotErrors(testName_List, CumRewards, timeHorizon, learner.name())
    
    
    print("*********************************************")


#run_exp(rendermode='pylab')    #Pylab rendering
#run_exp(rendermode='text')    #Text rendering
run_exp(rendermode='', testName_List = ['riverSwimErgo25', 'riverSwimErgo50'], sup = '_5')#, 'riverSwimErgo25'])#, 'riverSwimErgo50'])

