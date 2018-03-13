import numpy as np
#import math

class KLUCBPolicy :
    def __init__(self, K, DELTA):
        self.K = K
        self.DELTA = DELTA
        self.EPS = 10**(-12)
        self.reset()

    def kl(self, p, q): #calculate kl-divergence
        result =  p * np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
        return result

    def dkl(self, p, q):
        result = (q-p)/(q*(1.0-q))
        return result

    def reset(self):
        self.N = np.zeros(self.K)
        self.S = np.zeros(self.K)

    def getKLUCBUpper(self, k, n):
        logndn = np.log(n)/self.N[k]
        arg1=self.S[k]/self.N[k]
        #print("arg1 =", arg1)
        p = max(arg1, self.DELTA)
        if(p>=1):
            return 1

        converged = False
        q = p + self.DELTA

        for t in range(20):
            f  = logndn - self.kl(p, q)
            df = - self.dkl(p, q)

            if(f*f < self.EPS):
                converged = True
                break

        q = min(1 - self.DELTA , max(q - f / df, p + self.DELTA))

        if(not converged):
            print("WARNNG:Newton iteration in KL-UCB algorithm did not converge! p=" + str(p) + " logndn=" + str(logndn))

        return q

    def selectNextArm(self):
        n = np.sum(self.N)
        indices = np.zeros(self.K)
        for k in range(self.K):
            if(self.N[k]==0):
                return k

            #KL-UCB index
            indices[k] = self.getKLUCBUpper(k, n)

        targetArm = np.argmax(indices)
        return targetArm

    def updateState(self, k, r):
        self.N[k] += 1
        self.S[k] += r
