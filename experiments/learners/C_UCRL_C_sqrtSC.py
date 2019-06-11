from learners.C_UCRL_C import *
import scipy as sp
import numpy as np
import copy as cp

# Completed but not tested.

# C_UCRL_C is the C_UCRL(C) algorithm introduced by Maillard and Asadi 2018.
# It extends the UCRL2 class, see this one for commentary about its definition, here only the modifications will be discribed.
# Inputs:
#	nC number of equivalence classes in the MDP
#	C equivalence classes in the MDP, reprensented by a nS x nA matrix C with for each pair (s, a),
#		C[s, a] = c with c natural in  [0, nC - 1] the class of the pair.
class C_UCRL_C_sqrtSC(C_UCRL_C):
	def name(self):
		return "C_UCRL_C_sqrtSC"
	
	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self,state):
		action = self.policy[state]
		if self.vk[self.C[state, action]] >= np.sqrt(max([1, self.Nk_c[self.C[state, action]]])): # Stoppping criterion
			self.new_episode()
			action  = self.policy[state]
		return action