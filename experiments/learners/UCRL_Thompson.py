import numpy as np
import random as rd
import copy as cp
from learners.UCRL2_L import *

# UCRL2 variation implying posterior (or Thompson) sampling, introduced by Agrawal and Jia 2017.
# All notations are thus from the paper except delta which replace rho (because of the inheritance from UCRL2).
class UCRL_Thompson(UCRL2_L):
	def __init__(self,nS, nA, delta):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))
		# The following constants are used in the algorithm for the posterior sampling, the value are chosen
		# coherents with the B apendix of the paper.
		self.psi = int(10 * nS * np.log(nS * nA / delta)) # 10 factor is an arbitrary constant
		self.omega = 613 * np.log(2 / delta)
		self.kappa = self.omega / 6
		self.eta = 12 * self.omega * nS**2
		# The list of sampled transition prob
		self.Q = np.zeros((self.psi, self.nS, self.nA, self.nS))
		# The list of sampled rewards
		self.beta = 1# to define (number of samples for the reward)
		self.U = np.zeros((self.beta, self.nS, self.nA))
		

	def name(self):
		return "UCRL_Thompson"

	# Auxiliary used to sample the reward. Currently test with Laplace bound (as UCRL2_L) instead of sampling for sack of simplicity.
	def computeU(self, r_estimate):
		pass

	# Sampling function, updating self.Q and self.U the lists of samples.
	def sampling(self, p_estimate, r_estimate):
		# Sampling rewards (update self.U)
		self.computeU(r_estimate)
		# Sampling transitions (update self.Q)
		log4S = np.log(4 * self.nS)
		for s in range(self.nS):
			for a in range(self.nA):
				if self.Nk[s, a] >= self.eta:
					# Posterior sampling
					temp = [(self.Nk[s, a, i] + self.omega) / self.kappa for i in range(self.nS)]
					for j in range(self.psi):
						self.Q[j, s, a] = np.random.dirichlet(temp)
				else:
					# OPtimistic sampling
					p_minus = np.zeros(self.nS)
					for next_s in range(self.nS):
						temp1 = (3 * p_estimate[s, a, next_s] * log4S) / max((1, self.Nk[s, a]))
						temp2 = (3 * log4S) / max((1, self.Nk[s, a]))
						delta = min((np.sqrt(temp1) + temp2, p_estimate[s, a, next_s]))
						p_minus[next_s] = p_estimate[s, a, next_s] - delta
					z = rd.randint(0, self.nS - 1)
					sum_p = sum(p_minus)
					p_minus[z] += (1 - sum_p)
					for j in range(self.psi):
						self.Q[j, s, a] = p_minus # Far from being the most optimal way to implement it..
						
		
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, s, a, indice):
		argmax = np.argmax([self.Q[j, s, a, indice] for j in range(self.psi)])
		max_p = cp.deepcopy(self.Q[argmax, s, a])
		return max_p
	
	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, epsilon = 0.01, max_iter = 500):
		u0 = np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		indice = 0
		itera = 0
		while True:
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(s, a, indice)
					temp = r_estimate[s, a] + self.r_distances[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			elif itera > max_iter:
				print("No convergence in EVI at time : ", self.t)
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				indice = np.argmax(u0)
				itera += 1

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.vk = np.zeros((self.nS, self.nA))
		self.computeN()
		Rk = self.computeR()
		Pk = self.computeP()
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate[s, a] = Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = Pk[s, a, next_s] / div
		self.distances()
		self.sampling(p_estimate, r_estimate)
		self.EVI(r_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.new_episode()

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
		self.t += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		