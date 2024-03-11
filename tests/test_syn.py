import numpy as np


class Neuron:
    def __init__(self):
        self.V = np.zeros(2)

    def getV(self):
        return self.V
    def addIsyn(self, Isyn):
        print(Isyn)

class Synapse:
    def __init__(self, post):
        self.postsyn = post

        self.Erev = np.zeros(2) - 1
        self.gmax = 1
        self.R = 1
    def add_Isyn2Post(self):
        Vpost = self.postsyn.getV()

        gsyn = self.gmax * self.R

        gsyn = np.reshape(gsyn, (-1, 1))

        Vdiff = self.Erev - np.reshape(Vpost, (1, -1))
        Itmp = gsyn * Vdiff
        Isyn = np.sum(Itmp, axis=0)

        self.postsyn.addIsyn(Isyn)
        return
##############################################
n = Neuron()
s = Synapse(n)

s.add_Isyn2Post()