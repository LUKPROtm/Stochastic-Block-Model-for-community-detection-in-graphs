##### Degree unknown k SBM aka MFM mixture or finite mixture #############################################
import random
import numpy as np
import math
import scipy
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import poisson
from collections import Counter



class StochasticBlockModel():
    """
    Standard stochastic block model (generative)
    """
    
    def __init__(self, n, p, W):
        self.n = n
        self.p = p
        self.W = W
        self.k = len(self.W[0])
        self.X = []
        for i in range(self.n):
            self.X.append(random.choices([i for i in range(self.k)], p)[0])
            
        self.Y = np.zeros((n,n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (random.random() < self.W[self.X[i]][self.X[j]]):
                    self.Y[i][j] = 1
                    self.Y[j][i] = 1  


                    
                
    def __repr__(self):
        temp = "Adj matrix\n"
        for row in self.Y:
            temp += str(row) + "\n"
        temp += "\nClasses:\n"
        temp += str(self.X)
        return temp
    
    def indicator(self, event):
        return 1 if event is True else 0
    
    def e(self,a, k,h):
        edge = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.X[j] == h and self.X[i] == k and j != i:
                    if self.Y[i][j] == a:
                        edge += 1              
        return (edge)/(1 + self.indicator(h == k))
    
    

class Collapsed_Gibbs_Sampler():
    
    def __init__(self, Y, gamma = 1, a = 1, b = 1 , k_init = 5, prnt = False):
        self.Y = Y                                  # adjacency matrix
        self.n = len(self.Y[0])                     # number of nodes
        self.gamma = gamma                          # hyperparameter gamma
        self.a = a                                  # hyperparameter a
        self.b = b                                  # hyperparameter b
        self.prnt = prnt                            # se true fa dei print, per debug
        
        self.Vn_dict = {}                           # dict per store valori di vn
        self.W = {}                                 # matrice della probabilità di edges tra classi
        self.mod = {r : 0 for r in range(k_init)}   # numero di nodi all'interno di una classe
        self.X = np.zeros(self.n, dtype = "int64")  # classi
        self.classes = [0]                          # lista che contiene le classi attive
        
        self.counter = k_init                       # il counter serve a sapere come chiamare una nuova classe creata
        
        for i in range(self.n):
            # in realtà le probabilità dovrebbero venire da una dirichlet, non credo cambi nulla
            self.X[i] = random.choices([i for i in range(k_init)])[0]
            self.mod[self.X[i]] += 1
        self.classes = [r for r in self.mod if self.mod[r] > 0]
        self.k = len(self.classes)                  # numero di classi
        
        # self.W = {}
        # for k in range(self.k):
        #     self.W[self.classes[k]] = {}
        #     for h in range(k, self.k):
        #         self.W[self.classes[k]][self.classes[h]] = np.random.default_rng().beta( self.a,self.b)
        
        
    def sample(self, tmax):
        # gibbs sampler
        for t in range(tmax):
            self.step1()                    # update W conditional on X and Y
            # print()
            for i in range(self.n):
                self.step2(i)               # update Xi conditional on X-i, W, Y
        self.step1()                        # questo è utile solo a scopo di debug
        return self.X
    
    
    
    def step1(self):
        # update W conditional on X and Y
        self.W = {}
        for k in range(self.k):
            self.W[self.classes[k]] = {}
            for h in range(k, self.k):
                A1, A0 = self.Abar(self.classes[k],self.classes[h])  
                self.W[self.classes[k]][self.classes[h]] = np.random.default_rng().beta(A1 + self.a, A0 + self.b)
                


    
    def Abar(self, k, h):
        # la prima cosa che ritorna è il numero di edges i,j tale che X[i] = k, X[j] = h e Yij = 1, la seconda 
        # tale che ... e  Yij = 0
        # senza correzione da il doppio quando k = h
        edge = 0 
        total = 0
        for i in range(self.n):
            if self.X[i] == k:
                for j in range(self.n):
                    if self.X[j] == h and j != i:
                        total += 1
                        edge += self.Y[i][j]
        if k == h:
            return edge/2, (total - edge)/2
        return edge, total - edge    
    

    
    def step2(self, i):
        # update Xi conditional on X-i, W, Y
        past = self.X[i] 
        self.mod[past] -= 1

        
        log_probas = np.zeros(self.k + 1)
        # proba existing tables
        for n_class in range(self.k):
            r = self.classes[n_class]
            log_probas[n_class] = self.log_proba_existing_table(i, r)
             
        # proba new table
        log_probas[self.k] = self.log_proba_new_table(i)
        norm_probas = self.normalize_proba(log_probas)
        
        self.X[i] = np.random.choice(self.classes + [self.counter], 1, p= norm_probas)[0]
        

        
        if self.mod[past] == 0 and self.X[i] == self.counter:
            self.X[i] = past
            self.mod[past] += 1
            return
        
        
        if self.X[i] < self.counter:
            self.mod[self.X[i]] += 1
            if self.X[i] != past:
                if self.prnt:
                    print("CH ", i, "[", self.X[i], "]", end = " - ", sep = "")
            
        else:
            if self.prnt:
                print("NEW ", i, "[", self.X[i], "]", end = " - ", sep = "")
            self.mod[self.counter] = 1
            self.k += 1
            self.add_W_temp()
            self.classes.append(self.counter)
            self.counter += 1

        
        if self.mod[past] == 0:
            if self.prnt:
                print("DELETE ", i, "[", past, "]", end = " - ", sep = "")
            self.k -= 1
            self.mod.pop(past)
            self.classes.remove(past)
                

            
    
    def log_proba_existing_table(self, i, r):
        if self.mod[r] == 0:
            return -np.inf
        temp = 0
        temp += np.log(self.mod[r] + self.gamma)
        for j in range(self.n):
            if j != i:
                k1, k2 = self.get_W_indices(r, self.X[j])
                if self.Y[i][j] == 1:
                    temp += np.log(self.W[k1][k2])
                else:
                    temp += np.log(1 - self.W[k1][k2])
        # if i == 0:
        #     print("log(c+gamma)", f"{np.log(self.mod[r] + self.gamma):.3f}", "llik ", temp - np.log(self.mod[r] + self.gamma), "tot ", temp)
        return temp
    
    def log_proba_new_table(self, i):
        temp = 0
        t = sum(1 for value in self.mod.values() if value > 0)
        temp += self.Vn(t + 1)
        temp -= self.Vn(t)
        # if i == 0:
        #     print(f"{temp:.3f}", end = " ")
        temp += np.log(self.gamma)
        temp += self.m(i)
        # if i == 0:
        #     print("m", f"{self.m(i):.3f}", "v", self.Vn(t + 1) - self.Vn(t), "tot: ", temp)
        if self.counter > 10000:
            print("a")
            print(self.Vn(t + 1) - self.Vn(t))
            print(np.log(self.gamma))
            print(self.m(i))
        return temp
    
        
    def Vn(self, t):
        # log
        if t in self.Vn_dict.keys():
            return self.Vn_dict[t]
        else:
            kmax = self.n + 1
            temp = np.zeros(kmax - 1)
            for k in range(1, kmax):
                if k < t:
                    temp[k -1] = -np.inf
                else: 
                    #k t
                    if t!= 0:
                        for x in range(t):
                            temp[k - 1] += np.log(k-x)
                            
                    #gamma k n        
                    for x in range(self.n):
                        temp[k - 1] -= np.log(self.gamma * k + x)
                     
                    # for poisson(1) shifted      
                    temp[k - 1] += np.log(scipy.stats.poisson.pmf(k - 1, mu = 1) )
            
            res = np.logaddexp.reduce(temp)
            self.Vn_dict[t] = res
            return res
    


        
    def m(self, i):
        # giusta
        temp = 0
        for r in self.classes:
            if self.mod[r] == 0 and self.X[i] == r:
                pass
            else:
                temp -= np.log(scipy.special.beta(self.a,self.b))
                a1 = 0
                for j in range(self.n):
                    if j != i and self.X[j] == r:
                        a1 += self.Y[i][j]
                temp += np.log(scipy.special.beta(a1 + self.a, self.mod[r] - a1 + self.b))
                # print("a", a1, self.mod[r] - a1, i)
        return temp
    

                    
    def add_W_temp(self):
        for classe in self.classes:
            self.W[classe][self.counter] = np.random.default_rng().beta(self.a,self.b)
        self.W[self.counter] = {self.counter : np.random.default_rng().beta(self.a, self.b)} 
            
        
    def logsumexp(self, x):
        c = x.max()
        return c + np.log(np.sum(np.exp(x - c)))               
                    
    
    
    def get_W_indices(self,k,h):
        return (k,h) if k <= h else (h,k)

    def normalize_proba(self, x):
        return np.exp(x - self.logsumexp(x))
    
    def update(self):
        # per debug
        self.mod = Counter(self.X)
        self.k = len(self.mod)
        self.classes = list(self.mod.keys())
        self.classes.sort()
        self.step1()
        


###############################################################################


def errore(a, b):
    return  sum([a[i] != b[i] for i in range(len(a)) ])/len(a)

        
             

###############################################################################
  
    
# k = 3
# n = 70
# p = [1/k for i in range(k)]
# W = [[0.8 if i==j else 0.1 for i in range(k)]for j in range(k)]
# model = StochasticBlockModel(n, p, W)

# gibbs = Collapsed_Gibbs_Sampler(model.Y, gamma = 1, prnt = False, k_init = 2)

# prediction = gibbs.sample(2000)
# print(k, gibbs.k)
# print(normalized_mutual_info_score(model.X, prediction))         
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    