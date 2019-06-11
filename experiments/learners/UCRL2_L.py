from learners.UCRL import *
from learners.UCRL2_local import *
from learners.UCRL2_local2 import *
import scipy as sp
import numpy as np

# This is a really slight modification of UCRL2: we use the Laplace confidence bounds (cf: C_UCRL_C) instead of the originals ones.
class UCRL2_L(UCRL2):
	def name(self):
		return "UCRL2_L"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
				

# This is a really slight modification of UCRL2: we use the Laplace confidence bounds (cf: C_UCRL_C) instead of the originals ones.
# Heritance from a sligthly more efficient implem of UCRL2 (both version kept to be sure to not break any heritanc edependancy).
class UCRL2_L_boost(UCRL2_boost):
	def name(self):
		return "UCRL2-L"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
				
# This is a really slight modification of UCRL2: we use the Laplace confidence bounds (cf: C_UCRL_C) instead of the originals ones.
# Heritance from a sligthly more efficient implem of UCRL2 (both version kept to be sure to not break any heritanc edependancy).
class UCRL2_L_MSC(UCRL2_boost):
	def __init__(self,nS, nA, delta, epsilon = 0.):
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
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.epsilon = epsilon
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
	
	def name(self):
		return "UCRL2-L MSC"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
	
	# bH function from the paper, it's the Laplace confidence bound used for the reward of C_UCRL and the stopping criterion.
	def bH(self, n, delta):
		temp = (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / delta)
		return np.sqrt(temp / (2 * n))
	
	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		r_estimate = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				r_estimate[s, a] = self.Rk[s, a] / div
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		self.p_estimate = p_estimate
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
				#print(temp1 > temp2)
				self.new_episode()
		action  = self.policy[state]
		return action
	
class UCRL2_L_local(UCRL2_local):
	def name(self):
		return "UCRL2_L_local"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
					

class UCRL2_L_local2(UCRL2_local2):
	def name(self):
		return "UCRL2-L local"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
					

class UCRL2_L_local_MSC(UCRL2_L_local):
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
		self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.epsilon = epsilon
	
	# bH function from the paper, it's the Laplace confidence bound used for the reward of C_UCRL and the stopping criterion.
	def bH(self, n, delta):
		temp = (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / delta)
		return np.sqrt(temp / (2 * n))
	
	def name(self):
		return "UCRL2_L_local_MSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		if self.t > 2:
			action  = self.policy[state]
			s = self.observations[0][-2]
			a = self.observations[1][-1]
			self.Pk[s, a, state] += 1
			n = max([1, self.Nk[s, a] + self.vk[s, a]])
			d = self.delta / (2 * self.nS * self.nA)
			temp1 = abs(self.p_estimate[s, a, state] - self.Pk[s, a, state] / n)
			temp2 = self.p_distances[s, a, state]#(1 + self.epsilon) * self.bH(n, d)
			if (temp1  > temp2) or ( self.vk[state, action] >= max([1, self.Nk[state, action]])): # Stoppping criterion
				self.new_episode()
		action  = self.policy[state]
		return action
	
	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		self.vk[state, action] += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		#self.updateP()
		self.updateR()
		self.t += 1

class UCRL2_L_local2_MSC(UCRL2_L_local2):
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
		self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.epsilon = epsilon
	
	# bH function from the paper, it's the Laplace confidence bound used for the reward of C_UCRL and the stopping criterion.
	def bH(self, n, delta):
		temp = (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / delta)
		return np.sqrt(temp / (2 * n))
	
	def name(self):
		return "UCRL2_L_local2_MSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		if self.t > 2:
			action  = self.policy[state]
			s = self.observations[0][-2]
			a = self.observations[1][-1]
			self.Pk[s, a, state] += 1
			n = max([1, self.Nk[s, a] + self.vk[s, a]])
			d = self.delta / (2 * self.nS * self.nA)
			temp1 = abs(self.p_estimate[s, a, state] - self.Pk[s, a, state] / n)
			temp2 = self.p_distances[s, a, state]#(1 + self.epsilon) * self.bH(n, d)
			if (temp1  > temp2) or ( self.vk[state, action] >= max([1, self.Nk[state, action]])): # Stoppping criterion
				#print(temp1 > temp2)
				self.new_episode()
		action  = self.policy[state]
		return action
	
	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		self.vk[state, action] += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		#self.updateP()
		self.updateR()
		self.t += 1
		


class UCRL3_L(UCRL3):
	def name(self):
		return "UCRL3_L"
	
	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)