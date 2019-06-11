# UCRL implementation

This implementation, and related work, is the result of my internship under the supervision of Odalric-Ambrym Maillard and Sadegh Talebi, in the Sequel team of Inria Lille (from february to june 2019). The work on C-UCRL has essentially been done by Mahsa Asadi and my supervisors, I join later for experimental parts and improvements of C-UCRL(C).

The code in itself is in the "experiments" folder, see the first section below for an explanation about how to run an experiment using this implementation.

Most of the related work currently done using this implementation is presented in my internship report, available in this project. (Notice that currently this report is unfortunately incomplete, proof of regret of UCRL3 for example is not entirely wrote, I will update it as soon as the work is done).

For an exhaustive description of the learning algorithms implemented here, see the second section below. For a cleaner introduction of the most relevant of these algorithms see my report. A lot of thing were implemented and tested, I kept the failing tries in the code so be carefull about the learner you're choosing if you want to use the code.

Finally this work as been done for research purpose, none of the algorithm is implemented in an optimal way it was not the objective. Some mistakes should still be hided in the implementation, don't hesitate to contact me (hippolyte.bourel@ens-rennes.fr) if you have a question or a remark.

## How to run an experiment?

The full implementation is done in python, to run an experiment the only thing you need to do is basically to open the \experiments folder in a terminal and to run the following command:
```
python3 expUCRL.py
```

Then you'll finf results (plots and eventually binaries) in the \experiments\results folder. Plot are presented average on the number of replicate chosed and with 95% confidence intervals. All names of results' files are procedurally generated (with name of environment, time horizon and names of learners) so they're quite explicite (but when you just change the number of replicate you may delete the previous results be carefull). 

### Modify the experiment

If you want to choose the learners used, the environment used, the time horizon, the number of runs of if you want to save the result in binary to plot later with additional results then you need to modify the experiments\expUCRL.py file in the following way:

To choose your learners it simple you have list of commented (or not) repetitive 5 lines in the code starting looking like this:
```{python}
learner3 = UCRL2_L_local2( nbS,nbA, delta=0.05)#, C = C, nC = nC)
names.append(learner3.name())
cumRewards3 = cumulativeRewards(env,learner3,nbReplicates,timeHorizon,rendermode)
cumRewards.append(cumRewards3)
pickle.dump(cumRewards3, open(("results/cumRewards_" + testName + "_" + learner3.name() + "_" + str(timeHorizon) + sup), 'wb'))
```
To add a learner into you current experiment uncomment its 4 first lines.
By modifying the first line (and choosing the good inputs, as explained in following section) you choose a learner in the implemented set (folder \experiments\learners).
Don't pay attention to the 3 next lines if you don't want to go deep in the code (but 2nd line add the name of learner in a list, it is used by the ploting function; 3rd line: run the experiment for aforementionned learner in the chosen environement, time horizon and number of replicates; 4th line add results in a list used by the ploting function.
If you uncomment the fifth line you save in a binary the results for this learner (all, so pay attention its quickly really heavy (GigaOctet size for big experiements... it is a seriously non optimal part of my implementation but it's the easiest way to save results that are long to run, or to parrallelize experiments and then use the \experiments\expUCRL_load.py file to plot them, additionally this file (experiments\expUCRL_load.py) a similar structure than expUCRL.py so I will not describe it).

 To modify the time horizon and number of replicates (to average results) modify these to lines at the beginning of the definition of the 'run_exp(..)' function:
 ```{python}
    timeHorizon=100000
    nbReplicates=60
```

Finally to choose the environment modify the 'testName' in the call of the function in last line of the file:
 ```{python}
run_exp(rendermode='', testName = 'three-state-bernoulli', sup = '_0')
```
The list of available environments is proposed in following subsection, the 'sup' input parameter add a the given string at the end of binary files name (when you use pickle to save), it usefulle when you parallelize experiments (which is by the way the fastest way to run big experiments in personnal computer like I did, python run only one process in you computer, so you can optimize the computation time by running one python process in each core (times 2 if multi-threaded) that's not the optimal way to di this, and my change this by using multiprocesing package later).

If you modify the "rendermode" input (which I strongly advise against, the best one is '') you can add some dynamic display of the environment, but it is just too slow to be used in practice (two others inputs are 'pylab' and 'text').

### Available environments

For a proper introduction of our environment (value of reward, shape...) look I advice to have a look to my report. All environment are defined in the files contained in the \experiments\environments folder (plus an additional file named equivalence.py containing function usefull to test the C-UCRL families of learner (as for example the function computing equivalences classes for a given environment, usefull when you need this knowledge for your learner, you may notice that this function is by default runned in the run_exp function from \environments\expUCRL.py function si you don't really need to pay attention to this).

The testing base is built on gym package from OpenAI. The currently availabe environment are:

'riverSwim' (or in practice any input that is not mentionned later, because it is the default choice) it is a 6-states communicating riverSwim environment.

'riverSwim25' it is a 25-states communicating riverSwim environment.

'riverSwim25_biclass' it is 25-state communicating riverSwim environment with transition probability for the middle states cut in two subset (the first half of 'middle states' have a 0.6 proba to right will performing the going right action, the second half is as usual). (used to test C-UCRL learner, based on quivalence classes notion)

'riverSwim25_shuffle' similar to the previous one, but instead of changing transition probabilities in the midddle it alternates. (used to test C-UCRL learner, based on quivalence classes notion)

'riverSwimErgo50' it is a 50-states ergodic riverSwim environment.

'riverSwimErgo25'it is a 25-states ergodic riverSwim environment.

'riverSwimErgo'it is a 6-states ergodic riverSwim environment.

'three-state-bernoulli' the three state environment built as shown in my report, similar to the one proposed in SCAL's paper from Fruit et al., to modify its delta input modify the input of the buildThreeState(...) function in the \experiments\expUCRL.py file. An additionnal comment about this environment is that the single transitions starting from states 0 and 1 are doubled because there are two actions possibles in each states, but the result is the same in both cases and doesn't not affect the learner or the results, this particularity comes from the fact that currently learners needs enronments with same number of actions in each states, this should not be too difficult to modify for a learner but I don't really plan to take the time to do this for all of them... I will maybe propose later a second \experiments folder cleaned with this updated but only containing the more important learners).

'three-state' similar to the previous one but with fixed rewards (instead of Bernouilli initially), it is just a simplification of the previous environment, not usefull in practice because the previous one is alread really simple.

'random' supposed to build a random gridworld, I never used it so it probably doesn't work. (don't use it or modify it before)

'random_grid' same as the previous one but with an higher probability to work... (don't use it or modify before)

'4-room' a 7x7 4-room gridworld (20 states without walls), important to no that the learner does not consider walls as states so the environment is communicating.

'2-room' a 9x11 2-room gridworld (55 states without walls) with same remark as previous one.

Additionnally I plan to add in short time a discrete mountain-car environment, I'll update this document when done.

## Exhaustive list of implemented learners

A lot of learner are implemented, but a lot of them are tries (eventually failed ones) or really small modification of other learners, there is a lot of information about relevant learners in my report and additionnally I tried to put a comment above the definition of each learner in the code. I cannot detail everything here, I just propose a short description of each algorithm, if something is not clear, or wrong the easiest is to contact me. Combination between some algorithms may be relevant (for example a big merging of C-UCRL, UCRL3, Modified Stopping Criterion and SCAL, because it is probably the best thing we can currently implement but obviously I didn't had, and will not have, the time to implement all of these combinations). The following list is subject to change (I may add new stuff).

All learners are implemented as classes (with a play method that give the policy at each step and an update function to update after each step), input depends on the learner (some need additional inputs), globally the structure of an update of a learner is defined in a method 'new_episode', and what is called distances are the confidence bounds. The notations in the code are based on the Jacksh et al. paper from 2010 initially but evolve depending on the paper on which the learner is based on.

I used a lot heritance between classes, so it's no longer really possible to modify the more simple learners without destroying everything.

If you want to implement a new learner (to compare it to some of implemented ones), the easiest is to have a look to the \learner\UCRL.py file, UCRL2 is implemented in this file, and is clearly the simpler implemented algorithm (\learner\Random.py is also an option even if it not really a learner...) . This give the necessary structure for a learner (necessary functions are play, update and name in practice) then it should not be difficult to add a new learner based on this structure.

I decompose the list in subsection depending on the file in which learners are defined (ordered as much I can in the historical order of implementation (so supposed to be coherent considering the heritance problems)). Names at beginning of each paragraphs are names of the class in the implementation.

### \learners\UCRL.py

UCRL2 is our first implementation of the vanilla UCRL2 algorithm from Jacksh et al. 2010. Not optimal but some algorithms are based on it (class heritance) so I strongly advice to keep et as it is.

UCRL2_boost is a second implementation if the vanilla UCRL2 algorithm there are small modifications that may improve sligthly the computationnal time, most of following classes inherit from this one.

UCRL2_bis inherit from UCRL2_boost, it's simple (but inefficient) modification of UCRL2 using random policy for unknown states (basically, it's change absolutely nothing because it's already random...). No need to take care of this one.

### \learners\Random.py

Random is just a random player, playing randomly at each time step and learning nothing, it gives the simpler structure for a learner.

### \learners\Optimal.py

Opti_swimmer is always playing the "0" action, it the oracle for all riverSwims environments (where the optimal policy is to always go ti the right, which means in our implementation to play action 0) it is also the oracle for three-states environments.

Opti_learner takes the environment as input and then performs a policy iteration on it, it practice I never used it (too slow...) so it may not work.

Opti_77_4room is the oracle for a 7x7 4-room environment, it takes the environment as input (that give it the wrapper to avoid walls from its policy).

Opti_911_2room same as previous one but for a 9x11 2-room.

### \learners\UCRL3.py

Currently this file does not exist (the implementation of UCRL3 used for experiments of the paper is in the \learners\UCRL2_Lplus.py file under it "work in prgress" name. I'll add it soon and additionally I'll implement UCRL3 with modified stopping criterion which is supposed to be (as I know) the best optimistic algorithm in practice without additional knowledge.

### \learners\PSRL.py

There is nothing in this file, I plan to add at some point PSRL (from Osband et al.), because with all refinements of UCRL we have I'm no longer sure that PSRL still outperforms UCRL(3 with modified stopping criterion) (for infinite time horizon) that's a point that deserves to be clarified.

### \learners\KL_UCRL.py

KL_UCRL is our implementation of the vanilla KL-UCRL algorithm form Filippi et al. 2011. As explained in the report (and pseudo-code is provided) it is impossible to implement KL-UCRL exactly as introduced in its paper (it is mandatory to catch some definition error...). This implementation is partially based on the one of Sadegh Talebi (similar error catching globally). As an "experimental validation" of our choices the code of Sadegh obtains same results on riverSwims environments (which are surprinsingly an order better than the one presented in Filippi's paper, we suspect a wrong rescaling in this paper.

### \learners\C_UCRL_C.py (this one changed a lot recently, so lot of heritance from what's coming next)

If you don't know the C-UCRL algorithm it is necessary to have look to a paper introducing it (my report does) in order to understand following subsections. This file contains a lot of tries around the C-UCRL(C) algorithm (C-UCRL with classes known and profile mapping unknwown), some of these iterations are discussed in my report bu globally most of our tries are not discussed anywhere (this is the result of a collaborative work with Sadegh Talebi), obviously the one discussed in the paper seems to be the most relevant tries, but ideas in the others are also interesting for some of the other explaining why we kept (all of) them.

C_UCRL_C is the naive implementation of C-UCRL(C), to make it short: it doesn't work, for the explanation have a look to my report.

C_UCRL_C2 the idea here is to randomly (with a decreasing probability over the number of sample for a given pair s, a) replace the estimated profile by a l-shift permutation (with l increasing each time we use the shift), over the time it test all l-shifts. The idea behing was to add some forced exploration in the profile mapping to make the naive implementation works. In practice it does not work.

C_UCRL_C2_sqrtSC it the same algorithm but replacing the classical doubling criterion by what we'll call the sqrt stopping criterion which basically just increase the number of episode by replacing the criterion:
nu(t, s, a) > N(t, s, a) (doubling criterion)
by the followig modification:
nu(t, s, a) > sqrt(N(t, s, a))
as criterion to end an episode. Obviously it increase the computation time (a lot), to much compared to the gain, so all variant with "sqrtSC" I recommand to ignore it.

C_UCRL_C2bis_sqrtSC same as the previous one but the probability to use the shift lower depending on the time instead of the number of sample, does not work in practice, but there is no reason for this one to "never learn compared to previous one" which were just wrong in general.

C_UCRL_C2bis same as the previous one without this sqrt stopping criterion.

C_UCRL_C3 the idea here is to exclude from the classes the unsampled state-action pairs (implemented having the idea of excluding from classes all states-actions pairs sampled less than N0 times with arbitrary N0 not necesseraly equal to 1 for further tests). This idea is basic but necessary, we should obviously not give aggregated estimate to unknown state-action pairs, it does not make any sense (that should be add to C-UCRL with clustering). Even if this should be used all the time this class in itself has no interest.

C_UCRL_C4 it is the combination of C_UCRL_C2bis and C_UCRL_C3, no interest.

C_UCRL_C5_fixed: here we touch to an interesting idea, this idea it to cut the given classes into subclasses (in order to control, into subclasses, the bias brought by estimated profile mapping on the confidence bounds). Here the criterion used to defined these subclasses depends on some given alpha: all pairs (s, a) in a subclass are such that:
N(tk, s0, a0) / alpha < N(tk, s, a) < N(tk, s0, a0) x alpha
the objective is to have similar number of samples for all pairs in a subset (and so similar error on estimates) which should allow to prevent contamination of well estimated pairs and alllow good optimism in unknown pairs. This idea has some results in experiments, but its incomplete: have a look to following classes that are more interesting.

C_UCRL_C5_increasing same as before but with alpha increasing over the time (to have all element in same class at the end), precisely alpha = 1 + log(t). Again it is an incomplete idea.

C_UCRL_C6 the idea here is to keep this notion of subclasses, but instead of this alpha idea, a subclass contains all pairs with same experimental support on transition probabilities. It is again an incomplete idea but dealing with the support is interesting as elements outside from the support are the one on wich the unknown profile mapping brought most of it bias (because we cannot estimate the profile mapping outside from the experimental support, as it is an argsort).

C_UCRL_C7_fixed the idea here is to combine C_UCRL_C5_fixed with a notion of optimism on the profile mapping: we build the set of plausible profile mapping using the element-wise confidence interval (see the report) and then at each step of the Extended Value Iteration we choose the most optimistic profile mapping among this set. First this cost a lot computaitonnally, second it does not work well the bernstein bounds we used for this one are not tight enough to have a restricted enough set of profile mapping, with confidence bounds introduce in later work on UCRL3 it may be better, so it should be interesting to come back to this idea at some point. Additionally we stop doing these subclasses when the set of sigma is restricted to one element (which never happend is there are two equal transition probability...).

C_UCRL_C8_fixed incomplete modification of the previous one, can be ignored.

C_UCRL_C9_fixed it is C_UCRL_C5_fixed but using confidence bounds considering the bias brought be estimated sigma on ordered aggregated estimates (these bounds are introduced in the report), these bounds are worse than the aggregated one (used in C-UCRL(C, sigma) and wrongly used in other C-UCRL algorithm). This algorithm is less performing in practice but is closer than something that can be proved to have an upper bound on the regret, no need to take care of it (following classes more interesting).

C_UCRL_C10 this algorithm is the first one introducing some alternative notion to this idea of subclasses. The idea is inspired by the "new"/true bounds we're now using (that are the real confidence bounds on the ordered distribution probabilities, there is still a bias that we are not able to estimate as explained in my report) the idea is th following: instead of subclasses we compute an aggregate estimate based on a subclass but witha different subclass for all pairs. Precisely for all pairs we build the subclasse of all pairs in same class with higher number of samples and then we use this subclass to compute aggregated estimate for the given pair. This allow to have tighter (or at least same) confidence bounds for all pairs (because of properties of used almost unbiased bounds), it was not the case in last classes. This algorithm is not that much performing in practice, but the new idea to use aggregated estimate seems may be the promising one.

C_UCRL_C11 is globally the same algorithm as before but adding optimism on the profile mapping only on elements that are ouside from the experimental support of the transition. this improves experimental result and may be a good compromise to deal with bias we're ignoring so far.

C_UCRL_C12 is a combination of C_UCRLC5_fixed, the bounds we're no using (ones that partially take into account the bias brougth by estimated profile mapping), and the same optimism over profile mapping that the one used in C_UCRL_C11. This algorithm was the best one in practice (over ones using the same bounds), and it should be possible to control the bias on this algorithm in order to have a regret upper bound (but so far we're not able to control this bias, so we continue to explore other strategies on which we can prove things, algorithms based on UCRL3 performs better but it probably come from improvment coming from UCRL3 not from C_UCRL_C things). So this class is interesting, even if we can't prove anything as long as we can't controle the bias.

C_UCRL_C13 is similar to the previous class but we add the support of the transition probability as input and use it to perform like C-UCRL(C, sigma) when sigma is known with high probability (see the report for clarification of this). Not interesting because need additional input.

C_UCRL_C14 is globally the last big idea of this subsection (and presented in the report) the idea is to perform element-wise aggregation (in opposition to the global aggregation performed so far) when profile mapping is known with high probability for the considered element (this allow aggregation while profile mapping is known with high probability, even without support known or equal transition). We're still using the previous bound that control partially the bias (but we should modify it and go back to the bounds of C-UCRL(C, sigma) as now, when we perform aggregation, with high proability there is no bias). Gloablly the conclusion for this algorithm is that aggregation of transition probability performed like this almost not change anything in practice, in fact it the observation that motivates UCRL3 which come in next classes.

C_UCRL_C14_Nrwd is the same algorithm as before but taking of aggregation for rewards, this class as no interest in practice but was used to observe the real gain brought be our new idea of element-wise aggregation, the final result was explained in previous paragraph: no improvement (because outside of the support which is small in practice, we cannot perform element-wise aggregation because profile is never known with high probability for these element, so for most of transition there is no aggregation which motivate UCRL3).

C_UCRL_C14_Nrwd_Lplus_local3 is the same algorithm as the previous one, but based on UCRL3, we should maybe stop to consider the bounds that partially deals with the bias, but in any case this algorithm performs quite well in practice (it does really improve UCRL3 in our environments even without the reward) and additionnally here there is no problem of bias so we can almost re-use the entire proof of upper bound on regret of C-UCRL(C, sigma) to prove a bound for this algorithm (which will be in general between UCRL2 and C-UCRL(C, sigma) and practice closer than C-UCRL(C, sigma)), but this proof still have to be wrote.

C_UCRL_C14_Lplus_local3 is the same as previous but allowing aggregation of the reward, so obviously it performs better in practice and same remarks about proofs could be done.

### \learners\C_UCRL.py

### \learners\C_UCRL_old.py

### \learners\UCRL2_L.py

### \learners\UCRL2_MSC.py

### \learners\C_UCRL_C_MSC.py

### \learners\C_UCRL_MSC.py

### \learners\UCRL_Thompson.py

### \learners\UCRL2_local.py

### \learners\UCRL2_local2.py

### \learners\UCRL_L_sqrtSC.py

### \learners\C_UCRL_C_sqrtSC.py

### \learners\C_UCRL_sqrtSC.py

### \learners\UCRL2_peeling.py

### \learners\UCRL2_Bernstein.py

### \learners\C_UCRL_C_sigma.py

### \learners\SCAL.py

### \learners\UCRL_Lplus.py


