import numpy as np
import random as rd
import copy as cp
from learners.UCRL2_L import *

# Upgraded UCRL2 with modified stopping criterion from Maillard and Asadi 2018 and Laplace bound as UCRL2_L.
# Epsilon is the one used in the stopping criterion (function play).
########################
# Doesn't work: use URCL2_L.UCRL2_L_MSC instead
########################
class UCRL2_MSC(UCRL2_L):
	def __init__(self,nS, nA, delta, epsilon = 0.01):
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
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.epsilon = epsilon
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
	
	def name(self):
		return "UCRL2-L MSC"
	
	# bH function from the paper, it's the Laplace confidence bound used for the reward of C_UCRL and the stopping criterion.
	def bH(self, n, delta):
		temp = (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / delta)
		return np.sqrt(temp / (2 * n))
	
	# Auxiliary function to compute P the current transitions count.
	def computeP(self):
		P = np.zeros((self.nS, self.nA, self.nS))
		for t in range(len(self.observations[1])):
			P[self.observations[0][t], self.observations[1][t], self.observations[0][t+1]] += 1
		self.Pk = P

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.vk = np.zeros((self.nS, self.nA))
		self.computeN()
		Rk = self.computeR()
		self.computeP()
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate[s, a] = Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.p_estimate = p_estimate
		self.distances()
		self.EVI(r_estimate, p_estimate)
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		if self.t > 2:
			action  = self.policy[state]
			s = self.observations[0][-2]
			a = self.observations[1][-1]
			self.Pk[s, a, state] += 1
			n = max([1, self.Nk[s, a] + self.vk[s, a]])
			d = self.delta / (self.nS**2 * self.nA)
			temp1 = abs(self.p_estimate[s, a, state] - self.Pk[s, a, state] / n)
			temp2 = (1 + self.epsilon) * self.bH(n, d)
			if (temp1  > temp2) or ( self.vk[state, action] >= max([1, self.Nk[state, action]])): # Stoppping criterion
				self.new_episode()
		action  = self.policy[state]
		return action
		