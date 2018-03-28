import numpy as np


def kl_bernoulli(p, q): 
    """
    Compute kl-divergence for Bernoulli distributions
    """
    result =  p * np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    return result

def kl_exponential(p, q): 
    """
    Compute kl-divergence for Exponential distributions
    """
    result =  (p/q) - 1 - np.log(p/q)
    return result

class KLUCBPolicy :
    """
    KL-UCB algorithm
    """
    def __init__(self, K, kl_distance = kl_bernoulli):
        self.K = K
        self.kl_distance = kl_distance
        self.reset()

    def reset(self):
        self.N = np.zeros(self.K)
        self.S = np.zeros(self.K)

    def get_klucb_upper(self, k, t, precision = 1e-6, max_iterations = 50):
        """
        Compute the upper confidence bound for each arm with bisection method
        """
        upperbound = np.log(t)/self.N[k]
        reward=self.S[k]/self.N[k]

        u = upperbound
        l = reward
        n = 0
        
        while n < max_iterations and u - l > precision:
            n += 1
            q = (l + u)/2
            if self.kl_distance(reward, q) > upperbound:
                u = q
            else:
                l = q

        return (l+u)/2

    def select_next_arm(self):
        n = np.sum(self.N)
        indices = np.zeros(self.K)
        for k in range(self.K):
            if(self.N[k]==0):
                return k

            #KL-UCB index
            indices[k] = self.get_klucb_upper(k, n)

        target_arm = np.argmax(indices)
        return target_arm

    def update_state(self, k, r):
        self.N[k] += 1
        self.S[k] += r
