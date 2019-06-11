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
from learners.UCRL_Lplus import *
from environments import equivalence
#from learners.ImprovedMDPLearner2 import *
from utils import *

def run_exp(rendermode='', testName = "riverSwim", sup = ''):
    timeHorizon=100000
    nbReplicates=60
    
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
        env, nbS, nbA = buildThreeState(delta = 0.00, fixed_reward = False)
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
    
    #if testName == "random_grid" or testName == "2-room" or testName == "4-room":
    #    equivalence.displayGridworldEquivalenceClasses(env.env, 0.)
    #    equivalence.displayGridworldAggregationClasses(env.env)
    C, nC = equivalence.compute_C_nC(env.env)
    
    profile_mapping = equivalence.compute_sigma(env.env)
    sizeSupport = equivalence.compute_sizeSupport(env.env)
    
    print("sup = ", sup)
    
    print("*********************************************")
    
    cumRewards = []
    names = []
    
    #learner1 = UCRL2_L_MSC( nbS,nbA, delta=0.05)#, c = 10)
    #names.append(learner1.name())
    #cumRewards1 = cumulativeRewards(env,learner1,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    #cumRewards.append(cumRewards1)
    #pickle.dump(cumRewards1, open(("results/cumRewards_" + testName + "_" + learner1.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    #learner2 = C_UCRL_C_sigma_plus( nbS,nbA, delta=0.05, C = C, nC = nC, profile_mapping = profile_mapping)
    #names.append(learner2.name())
    #cumRewards2 = cumulativeRewards(env,learner2,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    #cumRewards.append(cumRewards2)
    #pickle.dump(cumRewards2, open(("results/cumRewards_" + testName + "_" + learner2.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    learner3 = UCRL2_L_local2( nbS,nbA, delta=0.05)#, C = C, nC = nC)
    names.append(learner3.name())
    cumRewards3 = cumulativeRewards(env,learner3,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    cumRewards.append(cumRewards3)
    pickle.dump(cumRewards3, open(("results/cumRewards_" + testName + "_" + learner3.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    #learner4 = C_UCRL_C9_fixed( nbS,nbA, delta=0.05, C = C, nC = nC)
    #names.append(learner4.name())
    #cumRewards4 = cumulativeRewards(env,learner4,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    #cumRewards.append(cumRewards4)
    #pickle.dump(cumRewards4, open(("results/cumRewards_" + testName + "_" + learner4.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    #learner5 = C_UCRL_C14_Nrwd( nbS,nbA, delta=0.05, C = C, nC = nC)#, sizeSupport = sizeSupport)#, T = timeHorizon)
    #names.append(learner5.name())
    #cumRewards5 = cumulativeRewards(env,learner5,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    #cumRewards.append(cumRewards5)
    #pickle.dump(cumRewards5, open(("results/cumRewards_" + testName + "_" + learner5.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    #learner6 = KL_UCRL( nbS,nbA, delta=0.05)#, c = 5)
    #names.append(learner6.name())
    #cumRewards6 = cumulativeRewards(env,learner6,nbReplicates,timeHorizon,rendermode)
    #cumRewards.append(cumRewards6)
    #pickle.dump(cumRewards6, open(("results/cumRewards_" + testName + "_" + learner6.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    #learner7 = UCRL2_boost( nbS,nbA, delta=0.05)
    #names.append(learner7.name())
    #cumRewards7 = cumulativeRewards(env,learner7,nbReplicates,timeHorizon,rendermode)
    #cumRewards.append(cumRewards7)
    #pickle.dump(cumRewards7, open(("results/cumRewards_" + testName + "_" + learner7.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    #learner8 = SCAL_L( nbS,nbA, delta=0.05, c = 5)
    #names.append(learner8.name())
    #cumRewards8 = cumulativeRewards(env,learner8,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    #cumRewards.append(cumRewards8)
    #pickle.dump(cumRewards8, open(("results/cumRewards_" + testName + "_" + learner8.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    learner9 = SCAL_L( nbS,nbA, delta=0.05, c = 2)
    names.append(learner9.name())
    cumRewards9 = cumulativeRewards(env,learner9,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    cumRewards.append(cumRewards9)
    pickle.dump(cumRewards9, open(("results/cumRewards_" + testName + "_" + learner9.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    learner10 = SCAL_L( nbS,nbA, delta=0.05, c = 5)
    names.append(learner10.name())
    cumRewards10 = cumulativeRewards(env,learner10,nbReplicates,timeHorizon,rendermode)
    cumRewards.append(cumRewards10)
    pickle.dump(cumRewards10, open(("results/cumRewards_" + testName + "_" + learner10.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    learner11 = SCAL_L( nbS,nbA, delta=0.05, c = 10)
    names.append(learner11.name())
    cumRewards11 = cumulativeRewards(env,learner11,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    cumRewards.append(cumRewards11)
    pickle.dump(cumRewards11, open(("results/cumRewards_" + testName + "_" + learner11.name() + "_" + str(timeHorizon) + sup), 'wb'))
    
    learner12 = SCAL_L( nbS,nbA, delta=0.05, c = 20)
    names.append(learner12.name())
    cumRewards12 = cumulativeRewards(env,learner12,nbReplicates,timeHorizon,rendermode)#, reverse = True)
    cumRewards.append(cumRewards12)
    pickle.dump(cumRewards12, open(("results/cumRewards_" + testName + "_" + learner12.name() + "_" + str(timeHorizon)+ sup), 'wb'))
    
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
    
    cumRewards.append(opti_reward)
    
    plotCumulativeRegrets(names, cumRewards, timeHorizon, testName)
    
    
    print("*********************************************")

#run_exp(rendermode='pylab')    #Pylab rendering
#run_exp(rendermode='text')    #Text rendering
run_exp(rendermode='', testName = 'three-state-bernoulli', sup = '_0') #, testName = 'riverSwimErgo50')#, testName = 'three-state')#-bernoulli')#, testName = '4-room')        #No rendering
