import scipy as sp
import numpy as np
import math
import random
from learners.Value_iter import *

# to implement the one used (from numpy) are not sufficient (no controle on the variance)
#def sampleDirichletMat(alpha):
#	# to do idem precedent
#	return 0

# to implement the one used (from numpy) are not sufficient (no controle on the variance)
#def sampleNormalGammaMat(params):
#	# to do idem github (aucune fonction amtlab utilisee)
#	return 0

#Implementation of the PSRL algorithm from Ian et al. 2013 ((More) Efficient Reinforcement Learning via Posterior Sampling), it is important to
# notice that the regret of this algorithm is controlled on for episodic MDP, or our current test base (the 22 May 2019) only perform experiments
# in infinite-horizon MDP. But even in infinite horizon PSRL seriously outperform UCRL2 and so deserves to be compared.
class PSRL:# TODO
	def __init__(self,nS, nA, delta):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []

	def name(self):
		return "PSRL"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for s in range(self.nS):
			for a in range(self.nA):
				self.Nk[s, a] += self.vk[s, a]
	
	# Auxiliary function to update R the accumulated reward.
	def updateR(self):
		self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]
	
	# Auxiliary function to update P the transitions count.
	def updateP(self):
		self.Pk[self.observations[0][-2], self.observations[1][-1], self.observations[0][-1]] += 1


	def sampling(self):
		for s in range(self.nS):
			for a in range(self.nA):
				# gamma random
				#self.r_sample[s, a] = np.random.gamma(1, scale = self.Rk[s, a]/max(1, self.Nk[s, a]))
				self.r_sample[s, a] = self.Rk[s, a]/max(1, self.Nk[s, a])
				# Posterior sampling
				temp = [(self.Pk[s, a, i] + 1)/max(1, self.Nk[s, a]) for i in range(self.nS)] # to verify (+1 ?)
				self.p_sample[s, a] = np.random.dirichlet(temp)

	# Value iteration ran based on the sampled rewards and transition (self.r_sample and self.p_sample)
	def VI(self, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results
		u1 = np.zeros(self.nS)
		niter = 0
		while True:
			niter += 1
			for s in range(self.nS):
				for a in range(self.nA):
					temp = self.r_sample[s, a] + sum([u * p for (u, p) in zip(u0, self.p_sample[s, a])])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in VI")
				break
		self.u = u0

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		self.sampling() # compute the the r_sample and p_sample used in the value iteration
		self.VI() # compute the new policy based on the sampled transitions and rewards

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.policy = np.zeros((self.nS), dtype = 'int')
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.p_sample = np.zeros((self.nS, self.nA, self.nS))
		self.r_sample = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.new_episode()
		self.span = [0]

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		action = self.policy[state]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		self.vk[state, action] += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.updateR()
		self.t += 1