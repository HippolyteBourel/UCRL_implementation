import scipy as sp
import numpy as np

# UCRL3 is what we defined as a local UCRL (with element-wise confidence bounds), and a near optimistic optimisation in the inner
# maximization of the Extended Value Iteration. (it the same algorithm as \UCR_Lplus\UCRL2_Lplus_local3 without inheritance)
# Here is the first version of UCRL3 with optimism only on the experimental support plus the best element, and so it is know to have
# a linear regret while performing the regret analysis, this is due to the worst case scenario: all elements can be reached with low
# probability (wich is a non-realistic but whatever). The simpliest solution is to have the support known but UCRL3_old(K) is
# currently not implemented.
class UCRL3_old:
	def __init__(self, nS, nA, delta):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		#self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.span = []
		self.p_distances = np.zeros((nS, nA, nS, 2))
	
	def name(self):
		return "UCRL3(old)"
	
	###### Computation of confidences intervams (named distances in implementation) ######
	# Bounds are estimated using 16 steps of dichotomic search

	# Auxiliary function to compute upper confidence bounds
	def beta_minus(self, p, n, d):
		g = (1 / 2 - p) / np.log(1 / p - 1)
		return np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)

	# Auxiliary function to compute lower confidence bounds
	def beta_plus(self, p, n, d):
		if p >= 0.5:
			g = p * (1 - p)
		else:
			g = (1 / 2 - p) / np.log(1 / p - 1)
		return np.sqrt((2 * g * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)

	# Compute the upper confidence bound, for given estimate and number of sample
	def bound_upper(self, p_est, n, d):
		p_tilde = p_est + np.sqrt((2 * (1/4) * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
		up = p_tilde
		down = p_est
		for i in range(16):
			temp = (up + down) / 2
			if (temp - self.beta_minus(temp, n, d)) <= p_est:
				down = temp
			else:
				up = temp
		return (up + down) / 2 - p_est
		
	# Compute the lower confidence bound, for given estimate and number of sample
	def bound_lower(self, p_est, n, d):
		p_tilde = p_est - np.sqrt((2 * (1/4) * (1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / n)
		down = p_tilde
		up = p_est
		for i in range(16):
			temp = (up + down) / 2
			if (temp + self.beta_plus(temp, n, d)) >= p_est:
				up = temp
			else:
				down = temp
		return p_est - (up + down) / 2

	# Tigther than Laplace bounds confidence interval, depending on the estimate value, we have for each transition of each pair 2 bounds
	# The lower (index 0) one and the upper (index 1) one.
	def distances(self, p_estimate, r_estimate):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				self.r_distances[s, a] = self.bound_upper(r_estimate[s, a], n, d)
				for next_s in range(self.nS):
					self.p_distances[s, a, next_s, 0] = self.bound_lower(p_estimate[s, a, next_s], n, d)
					self.p_distances[s, a, next_s, 1] = self.bound_upper(p_estimate[s, a, next_s], n, d)


	###### Functions used to initialize an episode ######
	
	# Inner maximization of the Extended Value Iteration
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8)):
		max_p = np.zeros(self.nS)
		delta = 1.
		for next_s in range(self.nS):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
			delta += - max_p[next_s]
		l = 0
		while (delta > epsilon) and (l <= self.nS - 1):
			idx = self.nS - 1 - l
			idx = sorted_indices[idx]
			if (l == 0) or self.supports[s, a, idx]:
				new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
				max_p[idx] += new_delta
				delta += - new_delta
			l += 1
		return max_p
	
	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)
		u1 = np.zeros(self.nS)
		sorted_indices = np.argsort(u0)
		itera = 0
		while True:
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
						u1[s] = temp
						self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				self.u = u1
				self.span.append(max(u1) - min(u1))
				break
			elif itera > max_iter:
				print("No convergence in the EVI at time : ", self.t)
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
				itera += 1

	def computeSupports(self, p_estimate):
		supports = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				for next_s in range(self.nS):
					supports[s, a, next_s] = p_estimate[s, a, next_s] > 0
		return supports
	

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
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances(p_estimate, r_estimate)
		self.supports = self.computeSupports(p_estimate)
		if self.t > 1:
			self.EVI(r_estimate, p_estimate)
	
	
	###### Steps and updates functions ######
	
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
	
	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.new_episode()
		self.u = np.zeros(self.nS)
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
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

# UCRL3 with modified stopping criterion based on hypothesis testing
class UCRL3_old_MSC(UCRL3_old):
	def name(self):
		return "UCRL3(old)-MSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		if self.t > 2:
			action  = self.policy[state]
			s = self.observations[0][-2]
			a = self.observations[1][-1]
			#Pk = self.Pk[s, a, state]
			n = max([1, self.Nk[s, a] + self.vk[s, a]])
			#d = self.delta / (2 * self.nS * self.nA)
			temp1 = self.p_estimate[s, a, state] - self.Pk[s, a, state] / n
			temp0 = - self.p_distances[s, a, state, 0]#(1 + self.epsilon) * self.bH(n, d)
			temp2 = self.p_distances[s, a, state, 1]
			if (temp1  > temp2) or (temp0 > temp1) or ( self.vk[state, action] >= max([1, self.Nk[state, action]])): # Stoppping criterion
				self.new_episode()
		action  = self.policy[state]
		return action