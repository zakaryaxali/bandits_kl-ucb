import numpy as np

class KLUCBPolicy :
    def __init__(self, K, delta):
        self.K = K
        self.delta = delta
        self.EPS = 10**(-12)
        self.reset()

    def kl_distance(self, p, q): #calculate kl-divergence
        result =  p * np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
        return result

    def dkl(self, p, q):
        result = (q-p)/(q*(1.0-q))
        return result

    def reset(self):
        self.N = np.zeros(self.K)
        self.S = np.zeros(self.K)

    def get_klucb_upper(self, k, n):
        logndn = np.log(n)/self.N[k]
        arg1=self.S[k]/self.N[k]
        #print("arg1 =", arg1)
        p = max(arg1, self.delta)
        if(p>=1):
            return 1

        converged = False
        q = p + self.delta

        for t in range(20):
            f  = logndn - self.kl_distance(p, q)
            df = - self.dkl(p, q)

            if(f*f < self.EPS):
                converged = True
                break

        q = min(1 - self.delta , max(q - f / df, p + self.delta))

        if(not converged):
            print("KL-UCB algorithm: Newton iteration did not converge!", "p=", p, "logndn=", logndn)

        return q

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
