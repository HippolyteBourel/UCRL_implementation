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

I decompose the list in subsection depending on the file in which learners are defined (ordered as much I can in the historical order of implementation (so supposed to be coherent considering the heritance problems)).

### \learners\UCRL.py

### \learners\Random.py

### \learners\Optimal.py

### \learners\KL_UCRL.py

### \learners\C_UCRL_C.py (this one changed a lot recently, so lot of heritance from what's coming next)

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


