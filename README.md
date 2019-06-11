# UCRL implementation

This implementation, and related work, is the result of my internship under the supervision of Odalric-Ambrym Maillard and Sadegh Talebi, in the Sequel team of Inria Lille (from february to june 2019). The work on C-UCRL has essentially been done by Mahsa Asadi and my supervisors, I join later for experimental part and improvements of C-UCRL(C).

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

If you modify the "rendermode" input (which I strongly advise against) you can add some dynamic display of the environment, but it is just too slow to be used in practice.

### Available environments

Not done

## Exhaustive list of implemented learners

To do
