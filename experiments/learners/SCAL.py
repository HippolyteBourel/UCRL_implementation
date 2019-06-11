from learners.UCRL2_Bernstein import *
from learners.UCRL2_L import *
import numpy as np
import random as rd
import copy as cp

# SCAL algorithm introduced by Fruit et al. 2018 (Vanilla version with Berstein's bounds).
# WARNING: If UCRL2_Bernstein inherit form UCRL2_local (which seems better) instead of UCRL_local2 your not running SCAL as introduced by R. Fruit et al.
# This implementation is more inspired from the code (github RonanFR) related to the paper than the paper itself.
# A remark is that in the implementation of the authors a gamma is used (and it is also introduced in the paper as the contraction factor < 1...)
# but it is fixed equal to 1 in their implementation, so currently fo sake of simplicity of the code we simply ignore it, for the compleness of our tests
# we would probably add it later.
# Another remark is that a difference between this implementation and the other is that we are not taking care of the rmax value and the complication
# following it because in our test base rmax is always fixed equal to 1 (it clearly allows to simplify the implementation).
class SCAL(UCRL2_Bernstein2):
	def __init__(self,nS, nA, delta, c):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS, 2), dtype=int) # Notice that for SCAL the policy is stochastic
		self.policy_prob = np.zeros(self.nS) # for a given state s: policy_prob[s] is the porbability to choose the action policy[s, 0] and because
		# the stochastic policy built with SCAL has only 2 possible actions by state, this probability value is sufficient to run the stochastic policy. 
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.c = c
		self.span = []

	def name(self):
		return "SCAL (c = " + str(self.c) + ")"
	
	# To checke nd in ScOpt
	def checkEnd(self, x, y):
		temp = [abs(x[i] - y[i]) for i in range(len(x))]
		return max(temp) - min(temp)

	# ScOpt is used in SCAL instead of the EVI in UCRL, it allows the control on the span performed by SCAL.
	# Called ScEVI in the implementation related to the paper.
	def ScOpt(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results MAYBE: add a recenter like in RonanFR?
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		counter = 0
		while True:
			counter += 1
			#print("Time t = ", self.t, " iter c = ", counter, " u0 : ", u0)
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s, 0]])):
						u1[s] = temp
						#print(min((1, r_estimate[s, a] + self.r_distances[s, a])))
						self.policy[s, 0] = a
						self.policy[s, 1] = a # Not really usefull
			max_v = max(u1)
			min_v = min(u1)
			th = min_v + self.c
			u2 = np.array([min((th, u1[s])) for s in range(self.nS)]) # span truncation
			u1_mina = np.zeros(self.nS)
			if (self.checkEnd(u0, u2) < epsilon) or (counter > max_iter):
				for s in range(self.nS):
					if u1[s] <= th:
						self.policy_prob[s] = 1.
					else: # in this case we have to compute the pessimistic action
						for a in range(self.nA):
							min_p = self.max_proba(p_estimate, sorted_indices, s, a, reverse = True)
							temp = max((0, r_estimate[s, a] - self.r_distances[s, a])) - sum([u * p for (u, p) in zip(u0, min_p)]) #0 - sum([u * p for (u, p) in zip(u0, min_p)])
							if (a == 0) or (temp + action_noise[a] < u1_mina[s] + action_noise[self.policy[s, 1]]):
								u1_mina[s] = temp
								self.policy[s, 1] = a
								self.policy[s, 1] = (self.policy[s, 0] + 1) % 2
						if u1_mina[s] > th:
							# I choose to continue to run the algo, instead of raising an error, by choosing the min action as with the operator N in RonanFr code
							self.policy_prob[s] = 0.
							print("The policy does not exist at time : ", self.t)
						else: # Same filtering as RonanFR code
							w1 = w2 = 0.
							if (abs(u1_mina[s] - u1[s])) < 1e-8: # if equals choosing depending on the noise
								if (u1_mina[s] + action_noise[self.policy[s, 1]]) > (u0[s] + action_noise[self.policy[s, 0]]):
									w1 = 1.
								else:
									w2 = 1.
							elif (abs(u1_mina[s] - th)) < 1e-8:
								w1 =  1.
							elif (abs(u1[s] - th)) < 1e-8:
								w2 = 1.
							else:
								w1 = (u1[s] - th) / (u1[s] - u1_mina[s])
								w2 = (th - u1_mina[s]) / (u1[s] - u1_mina[s])
							if not (abs(th - (w1 * u1_mina[s] + w2 * u1[s])) < 1e-8) or (w1 < 0) or (w2 < 0) or not (abs(w1 + w2 - 1) < 1e-8):
								print("Error encountered in ScOpt at time t = ", self.t)
							self.policy_prob[s] = w2
				break
			else:
				u0 = cp.deepcopy(u2)
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
		self.u = u2
		self.span.append(max(u2) - min(u2))
				

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
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances()
		if self.t > 1:
			self.ScOpt(r_estimate, p_estimate)

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		r = rd.random()
		if r < self.policy_prob[state]:
			action = self.policy[state, 0]
		else:
			action = self.policy[state, 1]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			r = rd.random()
			if r < self.policy_prob[state]:
				action = self.policy[state, 0]
			else:
				action = self.policy[state, 1]
		return action






class SCAL2(UCRL2_Bernstein):
	def __init__(self,nS, nA, delta, c):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS, 2), dtype=int) # Notice that for SCAL the policy is stochastic
		self.policy_prob = np.zeros(self.nS) # for a given state s: policy_prob[s] is the porbability to choose the action policy[s, 0] and because
		# the stochastic policy built with SCAL has only 2 possible actions by state, this probability value is sufficient to run the stochastic policy. 
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.c = c
		self.span = []

	def name(self):
		return "SCAL2 (c = " + str(self.c) + ")"
	
	# To checke nd in ScOpt
	def checkEnd(self, x, y):
		temp = [abs(x[i] - y[i]) for i in range(len(x))]
		return max(temp) - min(temp)

	# ScOpt is used in SCAL instead of the EVI in UCRL, it allows the control on the span performed by SCAL.
	# Called ScEVI in the implementation related to the paper.
	def ScOpt(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results MAYBE: add a recenter like in RonanFR?
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		counter = 0
		while True:
			counter += 1
			#print("Time t = ", self.t, " iter c = ", counter, " u0 : ", u0)
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s, 0]])):
						u1[s] = temp
						#print(min((1, r_estimate[s, a] + self.r_distances[s, a])))
						self.policy[s, 0] = a
						self.policy[s, 1] = a # Not really usefull
			max_v = max(u1)
			min_v = min(u1)
			th = min_v + self.c
			u2 = np.array([min((th, u1[s])) for s in range(self.nS)]) # span truncation
			u1_mina = np.zeros(self.nS)
			if (self.checkEnd(u0, u2) < epsilon) or (counter > max_iter):
				for s in range(self.nS):
					if u1[s] <= th:
						self.policy_prob[s] = 1.
					else: # in this case we have to compute the pessimistic action
						for a in range(self.nA):
							min_p = self.max_proba(p_estimate, sorted_indices, s, a, reverse = True)
							temp = max((0, r_estimate[s, a] - self.r_distances[s, a])) - sum([u * p for (u, p) in zip(u0, min_p)]) #0 - sum([u * p for (u, p) in zip(u0, min_p)])
							if (a == 0) or (temp + action_noise[a] < u1_mina[s] + action_noise[self.policy[s, 1]]):
								u1_mina[s] = temp
								self.policy[s, 1] = a
								self.policy[s, 1] = (self.policy[s, 0] + 1) % 2
						if u1_mina[s] > th:
							# I choose to continue to run the algo, instead of raising an error, by choosing the min action as with the operator N in RonanFr code
							self.policy_prob[s] = 0.
							print("The policy does not exist at time : ", self.t)
						else: # Same filtering as RonanFR code
							w1 = w2 = 0.
							if (abs(u1_mina[s] - u1[s])) < 1e-8: # if equals choosing depending on the noise
								if (u1_mina[s] + action_noise[self.policy[s, 1]]) > (u0[s] + action_noise[self.policy[s, 0]]):
									w1 = 1.
								else:
									w2 = 1.
							elif (abs(u1_mina[s] - th)) < 1e-8:
								w1 =  1.
							elif (abs(u1[s] - th)) < 1e-8:
								w2 = 1.
							else:
								w1 = (u1[s] - th) / (u1[s] - u1_mina[s])
								w2 = (th - u1_mina[s]) / (u1[s] - u1_mina[s])
							if not (abs(th - (w1 * u1_mina[s] + w2 * u1[s])) < 1e-8) or (w1 < 0) or (w2 < 0) or not (abs(w1 + w2 - 1) < 1e-8):
								print("Error encountered in ScOpt at time t = ", self.t)
							self.policy_prob[s] = w2
				break
			else:
				u0 = cp.deepcopy(u2)
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
		self.u = u2
		self.span.append(max(u2) - min(u2))
				

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
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances()
		if self.t > 1:
			self.ScOpt(r_estimate, p_estimate)

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		r = rd.random()
		if r < self.policy_prob[state]:
			action = self.policy[state, 0]
		else:
			action = self.policy[state, 1]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			r = rd.random()
			if r < self.policy_prob[state]:
				action = self.policy[state, 0]
			else:
				action = self.policy[state, 1]
		return action



class SCAL2_L(UCRL2_L_local):#UCRL2_Bernstein2):
	def __init__(self,nS, nA, delta, c):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS, 2), dtype=int) # Notice that for SCAL the policy is stochastic
		self.policy_prob = np.zeros(self.nS) # for a given state s: policy_prob[s] is the porbability to choose the action policy[s, 0] and because
		# the stochastic policy built with SCAL has only 2 possible actions by state, this probability value is sufficient to run the stochastic policy. 
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.c = c
		self.span = []

	def name(self):
		return "SCAL2_L (c = " + str(self.c) + ")"
	
	# To checke nd in ScOpt
	def checkEnd(self, x, y):
		temp = [abs(x[i] - y[i]) for i in range(len(x))]
		return max(temp) - min(temp)

	# ScOpt is used in SCAL instead of the EVI in UCRL, it allows the control on the span performed by SCAL.
	# Called ScEVI in the implementation related to the paper.
	def ScOpt(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results MAYBE: add a recenter like in RonanFR?
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		counter = 0
		while True:
			counter += 1
			#print("Time t = ", self.t, " iter c = ", counter, " u0 : ", u0)
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s, 0]])):
						u1[s] = temp
						#print(min((1, r_estimate[s, a] + self.r_distances[s, a])))
						self.policy[s, 0] = a
						self.policy[s, 1] = a # Not really usefull
			max_v = max(u1)
			min_v = min(u1)
			th = min_v + self.c
			u2 = np.array([min((th, u1[s])) for s in range(self.nS)]) # span truncation
			u1_mina = np.zeros(self.nS)
			if (self.checkEnd(u0, u2) < epsilon) or (counter > max_iter):
				for s in range(self.nS):
					if u1[s] <= th:
						self.policy_prob[s] = 1.
					else: # in this case we have to compute the pessimistic action
						for a in range(self.nA):
							min_p = self.max_proba(p_estimate, sorted_indices, s, a, reverse = True)
							temp = max((0, r_estimate[s, a] - self.r_distances[s, a])) - sum([u * p for (u, p) in zip(u0, min_p)]) #0 - sum([u * p for (u, p) in zip(u0, min_p)])
							if (a == 0) or (temp + action_noise[a] < u1_mina[s] + action_noise[self.policy[s, 1]]):
								u1_mina[s] = temp
								self.policy[s, 1] = a
								self.policy[s, 1] = (self.policy[s, 0] + 1) % 2
						if u1_mina[s] > th:
							# I choose to continue to run the algo, instead of raising an error, by choosing the min action as with the operator N in RonanFr code
							self.policy_prob[s] = 0.
							print("The policy does not exist at time : ", self.t)
						else: # Same filtering as RonanFR code
							w1 = w2 = 0.
							if (abs(u1_mina[s] - u1[s])) < 1e-8: # if equals choosing depending on the noise
								if (u1_mina[s] + action_noise[self.policy[s, 1]]) > (u0[s] + action_noise[self.policy[s, 0]]):
									w1 = 1.
								else:
									w2 = 1.
							elif (abs(u1_mina[s] - th)) < 1e-8:
								w1 =  1.
							elif (abs(u1[s] - th)) < 1e-8:
								w2 = 1.
							else:
								w1 = (u1[s] - th) / (u1[s] - u1_mina[s])
								w2 = (th - u1_mina[s]) / (u1[s] - u1_mina[s])
							if not (abs(th - (w1 * u1_mina[s] + w2 * u1[s])) < 1e-8) or (w1 < 0) or (w2 < 0) or not (abs(w1 + w2 - 1) < 1e-8):
								print("Error encountered in ScOpt at time t = ", self.t)
							self.policy_prob[s] = w2
				break
			else:
				u0 = cp.deepcopy(u2)
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
		self.u = u2
		self.span.append(max(u2) - min(u2))
				

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
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances()
		if self.t > 1:
			self.ScOpt(r_estimate, p_estimate)

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		r = rd.random()
		if r < self.policy_prob[state]:
			action = self.policy[state, 0]
		else:
			action = self.policy[state, 1]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			r = rd.random()
			if r < self.policy_prob[state]:
				action = self.policy[state, 0]
			else:
				action = self.policy[state, 1]
		return action


class SCAL_L(UCRL2_L_local2):#UCRL2_Bernstein2):
	def __init__(self,nS, nA, delta, c):
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nS, 2), dtype=int) # Notice that for SCAL the policy is stochastic
		self.policy_prob = np.zeros(self.nS) # for a given state s: policy_prob[s] is the porbability to choose the action policy[s, 0] and because
		# the stochastic policy built with SCAL has only 2 possible actions by state, this probability value is sufficient to run the stochastic policy. 
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA, self.nS))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.r_estimate = np.zeros((self.nS, self.nA))
		self.p_estimate = np.zeros((self.nS, self.nA, self.nS))
		self.c = c
		self.span = []

	def name(self):
		return "SCAL-L (c = " + str(self.c) + ")"
	
	# To checke nd in ScOpt
	def checkEnd(self, x, y):
		temp = [abs(x[i] - y[i]) for i in range(len(x))]
		return max(temp) - min(temp)

	# ScOpt is used in SCAL instead of the EVI in UCRL, it allows the control on the span performed by SCAL.
	# Called ScEVI in the implementation related to the paper.
	def ScOpt(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results MAYBE: add a recenter like in RonanFR?
		u1 = np.zeros(self.nS)
		sorted_indices = np.arange(self.nS)
		counter = 0
		while True:
			counter += 1
			#print("Time t = ", self.t, " iter c = ", counter, " u0 : ", u0)
			for s in range(self.nS):
				for a in range(self.nA):
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum([u * p for (u, p) in zip(u0, max_p)])
					if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s, 0]])):
						u1[s] = temp
						#print(min((1, r_estimate[s, a] + self.r_distances[s, a])))
						self.policy[s, 0] = a
						self.policy[s, 1] = a # Not really usefull
			max_v = max(u1)
			min_v = min(u1)
			th = min_v + self.c
			u2 = np.array([min((th, u1[s])) for s in range(self.nS)]) # span truncation
			u1_mina = np.zeros(self.nS)
			if (self.checkEnd(u0, u2) < epsilon) or (counter > max_iter):
				for s in range(self.nS):
					if u1[s] <= th:
						self.policy_prob[s] = 1.
					else: # in this case we have to compute the pessimistic action
						for a in range(self.nA):
							min_p = self.max_proba(p_estimate, sorted_indices, s, a, reverse = True)
							temp = max((0, r_estimate[s, a] - self.r_distances[s, a])) - sum([u * p for (u, p) in zip(u0, min_p)]) #0 - sum([u * p for (u, p) in zip(u0, min_p)])
							if (a == 0) or (temp + action_noise[a] < u1_mina[s] + action_noise[self.policy[s, 1]]):
								u1_mina[s] = temp
								self.policy[s, 1] = a
								self.policy[s, 1] = (self.policy[s, 0] + 1) % 2
						if u1_mina[s] > th:
							# I choose to continue to run the algo, instead of raising an error, by choosing the min action as with the operator N in RonanFr code
							self.policy_prob[s] = 0.
							print("The policy does not exist at time : ", self.t)
						else: # Same filtering as RonanFR code
							w1 = w2 = 0.
							if (abs(u1_mina[s] - u1[s])) < 1e-8: # if equals choosing depending on the noise
								if (u1_mina[s] + action_noise[self.policy[s, 1]]) > (u0[s] + action_noise[self.policy[s, 0]]):
									w1 = 1.
								else:
									w2 = 1.
							elif (abs(u1_mina[s] - th)) < 1e-8:
								w1 =  1.
							elif (abs(u1[s] - th)) < 1e-8:
								w2 = 1.
							else:
								w1 = (u1[s] - th) / (u1[s] - u1_mina[s])
								w2 = (th - u1_mina[s]) / (u1[s] - u1_mina[s])
							if not (abs(th - (w1 * u1_mina[s] + w2 * u1[s])) < 1e-8) or (w1 < 0) or (w2 < 0) or not (abs(w1 + w2 - 1) < 1e-8):
								print("Error encountered in ScOpt at time t = ", self.t)
							self.policy_prob[s] = w2
				break
			else:
				u0 = cp.deepcopy(u2)
				u1 = np.zeros(self.nS)
				sorted_indices = np.argsort(u0)
		self.u = u2
		self.span.append(max(u2) - min(u2))
				

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
		self.p_estimate = p_estimate
		self.r_estimate = r_estimate
		self.distances()
		if self.t > 1:
			self.ScOpt(r_estimate, p_estimate)

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		r = rd.random()
		if r < self.policy_prob[state]:
			action = self.policy[state, 0]
		else:
			action = self.policy[state, 1]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			r = rd.random()
			if r < self.policy_prob[state]:
				action = self.policy[state, 0]
			else:
				action = self.policy[state, 1]
		return action






	# return sp{v1 - v2}
	#def spanDiff(self, v1, v2):
	#	n = len(v1)
	#	diff = [v1[i] - v2[i] for i in range(n)]
	#	return max(diff) - min(diff)
	
	#def ScOpt(self, p_estimate, r_estimate, ref_state, v0_init, epsilon = 0.01):
	#	v0 = v0_init
	#	e = np.exp(1)
	#	temp = value_operator(v0)
	#	v1 = [(temp[i] - temp[ref_state] * e) for i in range(len(temp))]
	#	n = 0
	#	const = self.spanDiff(v1, v0)
	#	test = self.spanDiff(v1, v0) + (2 * self.gamma**n) / (1 - self.gamma) * const
	#	while test > epsilon:
	#		n += 1
	#		v0 = cp.deepcopy(v1)
	#		temp = value_operator(v0)
	#		v1 = [(temp[i] - temp[ref_state] * e) for i in range(len(temp))]
	#		test = self.spanDiff(v1, v0) + (2 * self.gamma**n) / (1 - self.gamma) * const
	#	# Then as explained in Appendix C of the paper we compute the stochastic policy
	#	for s in rage(self.nS):
	#		maxL, index_maxL = some_aux()# TODO
	#		minL, index_minL = some_aux()# TODO
	#		if maxL <= minL + self.c:
	#			self.policy[s, 0] = index_maxL
	#			self.policy_prob[s] = 1.
	#		else:
	#			self.policy[s] = np.array([index_maxL, index_minL], dtype = int)
	#			temp = self.c / (maxL - minL)
	#			self.policy_prob[s] = np.array([1. - temp, temp])
	