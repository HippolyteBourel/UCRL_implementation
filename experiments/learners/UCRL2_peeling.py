from learners.UCRL import *
import scipy as sp
import numpy as np

# Used to compute the peeling bound.
def llnp(x):
	return np.log(np.log(max(x, np.exp(1))))

# This is a really slight modification of UCRL2: we use the peeling confidence bounds (cf: UNiform PAC, Dann et al. 2017) instead of the originals ones.
class UCRL2_peeling(UCRL2_boost):
	def name(self):
		return "UCRL2_peeling"
	


	def distances(self):
		d = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				#self.r_distances[s, a] = np.sqrt(((1 + 1 / n) * np.log(2 * np.sqrt(n + 1) / d)) / (2 * n))
				#self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.nS * self.nA * self.t / self.delta))
				#								/ (2 * max([1, self.Nk[s, a]])))
				self.r_distances[s, a] = np.sqrt((4 / n) * (2 * llnp(n) + np.log((3 * (2**(self.nS) - 2)) / self.delta)))
				self.p_distances[s, a] = np.sqrt((4 / n) * (2 * llnp(n) + np.log((3 * (2**(self.nS) - 2)) / self.delta)))