import numpy as np

from k_means_cluster import k_means_cluster
from ..my_math import lin_alg as mp
from ..my_math import common_functions as cf


np.random.seed(3)


class gmm():
    def __init__(self, num_mixtures, data) -> None:
        self.num_mixtures = num_mixtures
        self.data = data
        
    def initialise_params(self, method='kmc'):
        if method == 'kmc':
            kmc = kmc.k_means_cluster(self.data, num_clusters=self.num_mixtures)
            self.means = kmc.centres
            self.sigmas = kmc.sigmas
            self.weights = kmc.bins / kmc.num_examples
    def calc_responsibilty(self, x):
        gammas = np.array([self.weights[i] * cf.gaussian_density(x, self.means[i], self.sigmas[i]) for i in range(self.num_mixtures)])
        return gammas / np.sum(gammas)

    def update_params(self):
        all_gammas = np.array([self.calc_responsibilty(ex) for ex in self.data])
        num_effective_samples = np.sum(all_gammas, axis=0)
        self.weights = num_effective_samples / np.mean(num_effective_samples)
        self.means =  np.array([np.sum(all_gammas[:, i] * self.data) / num_effective_samples[i] for i in range(self.num_mixtures)]) 
        for i in range(self.num_mixtures):
            xu = self.data - self.means[i]
            self.sigmas[i] = (all_gammas[:, i] * xu).dot(xu.T) / num_effective_samples[i]



    # def assign_modes(self):

    # def expecto_maximus(self):
