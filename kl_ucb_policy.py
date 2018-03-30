import numpy as np


def kl_bernoulli(p, q):
    """
    Compute kl-divergence for Bernoulli distributions
    """
    result =  p * np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    return result

def dkl_bernoulli(p, q):
    result = (q-p)/(q*(1.0-q))
    return result

def kl_exponential(p, q):
    """
    Compute kl-divergence for Exponential distributions
    """
    result =  (p/q) - 1 - np.log(p/q)
    return result

def klucb_upper_newton(kl_distance, N, S, k, t, precision = 1e-6, max_iterations = 50):
    """
    Compute the upper confidence bound for each arm using Newton's iterations method
    """
    delta = 0.1
    logtdt = np.log(t)/N[k]
    p = max(S[k]/N[k], delta)
    if(p>=1):
        return 1

    converged = False
    q = p + delta

    for n in range(max_iterations):
        f  = logtdt - kl_distance(p, q)
        df = - dkl_bernoulli(p, q) #TODO : cas bernoulli et exponentiel

        if(f*f < precision):
            converged = True
            break

    q = min(1 - delta , max(q - f / df, p + delta))

    if(not converged):
        print("KL-UCB algorithm: Newton iteration did not converge!", "p=", p, "logtdt=", logtdt)

    return q

def klucb_upper_bisection(kl_distance, N, S, k, t, precision = 1e-6, max_iterations = 50):
    """
    Compute the upper confidence bound for each arm with bisection method
    """
    upperbound = np.log(t)/N[k]
    reward=S[k]/N[k]

    u = upperbound
    l = reward
    n = 0

    while n < max_iterations and u - l > precision:
        n += 1
        q = (l + u)/2
        if kl_distance(reward, q) > upperbound:
            u = q
        else:
            l = q

    return (l+u)/2

class KLUCBPolicy :
    """
    KL-UCB algorithm
    """
    def __init__(self, K, kl_distance = kl_bernoulli, klucb_upper = klucb_upper_bisection):
        self.K = K
        self.kl_distance = kl_distance
        self.klucb_upper = klucb_upper
        self.reset()

    def reset(self):
        self.N = np.zeros(self.K)
        self.S = np.zeros(self.K)

    def select_next_arm(self):
        t = np.sum(self.N)
        indices = np.zeros(self.K)
        for k in range(self.K):
            if(self.N[k]==0):
                return k

            #KL-UCB index
            indices[k] = self.klucb_upper(self.kl_distance, self.N, self.S, k, t)

        target_arm = np.argmax(indices)
        return target_arm

    def update_state(self, k, r):
        self.N[k] += 1
        self.S[k] += r
