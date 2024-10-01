import numpy as np

class L1L2TVModel(object):
    def __init__(self):
        self.alpha1  = np.array(0)
        self.alpha2  = np.array(0)
        self.lambdaa = np.array(0)
        self.beta    = np.array(0)
        self.gamma1  = np.array(0)
        self.gamma2  = np.array(0)

    def __str__(self):
        return "Model Parameters:\n - alpha1 = "+str(self.alpha1)+"\n - alpha2 = "+str(self.alpha2)+"\n - lambda = "+str(self.lambdaa)+"\n - beta   = "+str(self.beta)+"\n - gamma1 = "+str(self.gamma1)+"\n - gamma2 = "+str(self.gamma2)