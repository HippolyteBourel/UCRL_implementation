import numpy as np
import pylab as pl
import copy as cp


def mapping(g,s,a):
    #s: int, a:int
    #_,t = g.transition(g.stateOfIndex(s),g.actionOfIndex(a))
    t = g.getTransition(s,a)
    st=sorted(t)
    r = list(range(0,len(t)))
    indexmap = np.zeros((len(t)))
    reverseindexmap = np.zeros((len(t)))
    for i in range(0,len(t)):
        # print(st[i],",",i, ", " , t[i])
        k = 0
        j = r[k]
        while (st[j] != t[i] and k< len(r)):
            k+=1
            j= r[k]
        indexmap[i]=j
        reverseindexmap[j]=i
        r.pop(k)

    return st,indexmap,reverseindexmap  #st[indexmap[j]]=t[j]  st[j]=t[reverseindexmap[j]]


def compare(g,s1,a1,s2,a2):
    #_,t1 = g.transition(g.stateOfIndex(s1),g.actionOfIndex(a1))
    #_,t2 = g.transition(g.stateOfIndex(s2),g.actionOfIndex(a2))
    t1 = g.getTransition(s1,a1)
    t2 = g.getTransition(s2, a2)
    r1 = g.getReward(s1, a1)
    r2 = g.getReward(s2, a2)
    st1=sorted(t1)
    st2=sorted(t2)
    err = 0
    for i in range(g.nS):
        err += abs(st1[i]-st2[i])
    err += abs(r1 - r2)
    return err

def equivalenceClass(g,s,a,eps):
    equiva = []
    for s2 in range(g.nS):
            for a2 in range(g.nA):
                if (compare(g,s,a,s2,a2)<=eps):
                    equiva.append([s2,a2])
    return equiva

def equivalenceClasses(g,eps):
    eqclasses = []
    stateactionpairs = []
    sasize =0
    nbeqclasses =0
    indexEqClass = np.zeros((g.nS,g.nA))
    for s in range(g.nS):
            for a in range(g.nA):
                stateactionpairs.append([s,a])
                sasize+=1
    # print(stateactionpairs)
    while(sasize>0):
        s,a =  stateactionpairs.pop()
        sasize-=1
        eqC = equivalenceClass(g,s,a,eps)
        eqclasses.append(eqC)
        nbeqclasses+=1
        indexEqClass[s][a] = nbeqclasses-1
        for e in eqC:
            s, a = e
            #print(e)
            if e in stateactionpairs:#(stateactionpairs.count(e)>0):
                s,a=e
                indexEqClass[s][a] = nbeqclasses - 1
                stateactionpairs.remove(e)
                sasize-=1
    return eqclasses,indexEqClass



def plotGridWorldEquivClasses(g, eqclasses, folder=".", numFigure=1):
    nbFigure = pl.gcf().number + 1
    pl.figure(nbFigure)
    actions = g.nameActions
    equiv0 = np.zeros((g.sizeX,g.sizeY))
    equiv1 = np.zeros((g.sizeX,g.sizeY))
    equiv2 = np.zeros((g.sizeX,g.sizeY))
    equiv3 = np.zeros((g.sizeX,g.sizeY))
    numq=0
    eqClasses = sorted(eqclasses,key=lambda x: len(x))
    for eq in eqClasses:
        numq+=1
        for e in eq:
            x,y=g.from_s(e[0])
            if(g.maze[x][y]>0):
                if(e[1]==0):
                    equiv0[x][y]=numq
                if(e[1]==1):
                    equiv1[x][y] = numq
                if(e[1]==2):
                    equiv2[x][y] = numq
                if(e[1]==3):
                    equiv3[x][y] = numq
    f, axarr = pl.subplots(2, 2)
    axarr[0, 0].imshow(equiv0, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[0, 0].set_title(actions[0])
    axarr[0, 1].imshow(equiv1, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[0, 1].set_title(actions[1])
    axarr[1, 0].imshow(equiv2, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[1, 0].set_title(actions[2])
    axarr[1, 1].imshow(equiv3, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr[1, 1].set_title(actions[3])
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    pl.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    pl.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    pl.savefig('Classes.png')



def displayGridworldEquivalenceClasses(g, eps):
    eqClasses,indexEqClass=equivalenceClasses(g,eps)
    plotGridWorldEquivClasses(g, eqClasses)
 
def compareAggreg(g, c0, c1, eqClasses):
    res = True
    for a in range(g.nA):
        for c in eqClasses:
            sum0 = sum([g.getTransition(c0[0], a)[s] for s in c])
            sum1 = sum([g.getTransition(c1[0], a)[s] for s in c])
            if sum0 != sum1:
                res = False
                break
    return res

# Computing bisimulation classes (see Givan et al. 2003).
# As for the other classe definition we are not taking into account the reward because of the definition of our gridworlds:
# the only state having a different reward (called gaol state in the code) also have differents transition probability (for all actions
# transition to the initial state) so its not necessary to take into account the reward (it non information in our specific case).
def aggregationClasses(g):
    eqClasses = [[s] for s in range(g.nS)]
    not_converged = True
    new_eqClasses = []
    while not_converged:
        not_converged = False
        while len(eqClasses) > 0:
            if len(eqClasses) == 1:
                new_eqClasses.append(eqClasses.pop(0))
            else:
                not_modif = True
                for i in range(1, len(eqClasses)):
                    if compareAggreg(g, eqClasses[0], eqClasses[i], eqClasses + new_eqClasses):
                        eqClasses[0] += eqClasss.pop(i)
                        new_eqClasses.append(eqClasses.pop(0))
                        not_converged = True
                        not_modif = False
                        break
                if not_modif:
                    new_eqClasses.append(eqClasses.pop(0))
        eqClasses = cp.deepcopy(new_eqClasses)
        new_eqClasses = []
    return eqClasses


def plotGridworldClasses(g, eqClasses):
    pl.figure()
    actions = g.nameActions
    equiv = np.zeros((g.sizeX,g.sizeY))
    numq = 0
    eqClasses = sorted(eqClasses,key=lambda x: len(x))
    for eq in eqClasses:
        numq+=1
        for s in eq:
            x, y = g.from_s(s)
            if(g.maze[x][y] > 0):
                equiv[x][y] = numq
    f, axarr = pl.subplots(1, 1)
    axarr.imshow(equiv, cmap='hot', interpolation='nearest',vmin=0, vmax=numq)
    axarr.set_title("Bisimilarity equivalence")
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    #pl.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    #pl.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    pl.savefig('aggregationClasses.png')
    
def displayGridworldAggregationClasses(g):
    eqClasses = aggregationClasses(g)
    plotGridworldClasses(g, eqClasses)
    
def compute_C_nC(g):
    print("#############\nWARNING: Be sure that the environment your using has discrete rewards when using the function environments.equivalence.compute_C_nC\n#############")
    eqClasses, _ = equivalenceClasses(g,0.)
    C = np.zeros((g.nS,g.nA), dtype = int)
    numq=0
    eqClasses = sorted(eqClasses,key=lambda x: len(x))
    for eq in eqClasses:
        for c in eq:
            C[c[0], c[1]] = numq
        numq+=1
    return C, numq

# Compute the true profile mapping for a given environment
def compute_sigma(g):
    sigma = np.zeros((g.nS, g.nA, g.nS))
    for s in range(g.nS):
        for a in range(g.nA):
            li = list(np.argsort(g.getTransition(s, a)))
            #li.reverse()
            sigma[s, a] = np.array(li)
    return sigma

# Compute a gap delta used in some tests on C_UCRL(function of the environment and the profile mapping)
def compute_Delta(g, sigma):
    Delta = float("infinity")
    for s in range(g.nS - 1):
        for a in range(g.nA):
            p = g.getTransition(s, a)
            for ss in range(s + 1, g.nS):
                for aa in range(g.nA):
                    pp = g.getTransition(ss, aa)
                    temp = 0
                    for next_s in range(g.nS):
                        temp += abs(p[int(sigma[s, a, next_s])] - pp[int(sigma[ss, aa, next_s])])
                    if temp != 0:
                        Delta = min((Delta, temp))
    return Delta

# To compute the size of the support of each transition function in the environment (for example it is used in the class C_UCRL_C13 (in the file
# learner.C_UCRL_C))
def compute_sizeSupport(g):
    sizeSupport = np.zeros((g.nS, g.nA))
    for s in range(g.nS):
        for a in range(g.nA):
            p = g.getTransition(s, a)
            for ss in range(g.nS):
                if p[ss] > 0:
                    sizeSupport[s, a] += 1
    return sizeSupport