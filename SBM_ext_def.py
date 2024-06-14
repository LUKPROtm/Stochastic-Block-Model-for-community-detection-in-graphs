######################################## SBM ################################################

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score

 
class StochasticBlockModel_General():
    """
    Class that generates SBMs
    """
    def __init__(self, n, N, Classes, Alphabet, eta):
        self.n = n
        self.N = N
        self.Classes = Classes
        self.Alphabet = Alphabet
        Alphabet_list = []
        for i in self.Alphabet:
            Alphabet_list += self.Alphabet[i]
        self.eta = eta  
        self.X = []
        for i in range(self.n):
            self.X.append(random.choices(list(Classes.keys()),list(Classes.values()))[0])
        self.X = np.array(self.X)
        self.Y = [[9 for i in range(self.n)] for j in range(self.n)]
        for i,j in self.N:
            self.Y
            a = random.choices(Alphabet_list, [self.eta_value(self.X[i], self.X[j], a) for a in Alphabet_list])[0] 
            self.Y[i][j] = a[0]
            self.Y[j][i] = a[1]
          
            
    def eta_value(self, k, h, a):
        (a1, a2) = a
        if k!=h or a1 == a2:
            if (k,h) in self.eta.keys():
                return self.eta[(k,h)][a]
            elif (h,k) in self.eta.keys():
                return self.eta[(h,k)][(a2, a1) ]
            else:
                raise Exception("missing value")
        else:
            if a in self.eta[(k,k)].keys():
                return self.eta[(k,k)][a]/2
            else: 
                return self.eta[(k,k)][(a2, a1) ]/2
  
    
    def __repr__(self):
        temp = "Adj matrix\n"
        for row in self.Y:
            temp += str(row) + "\n"
        temp += "\nClasses:\n"
        temp += str(self.X)
        return temp

    
                            

class Gibbs_sampler_General():
    """
    The Gibbs sampler. 
    Parameters:
        N: set of observed pairs
        Y: adjacency matrix
        k: number of classes
        Alphabet: a dict in the form {"A0": [(0,0), (1,1)], "A10": [], "A11": []}
        T: parameter of the dirichlet prior, it will be multiplied by k
        
    How to use it:
        initialize it and then use the function sample_improved(M0) to sample with 2*M0 iterations
    
    """
    def __init__(self, N, Y, k, Alphabet, T = 100):
        
        self.N = N
        self.Y = Y
        self.n = len(self.Y[0])
        self.k = k
        self.Alphabet = Alphabet
        self.r0 = len(Alphabet["A0"])
        self.r1 = len(Alphabet["A10"])
        self.A = Alphabet["A0"] + Alphabet["A10"] + Alphabet["A11"]
        self.Ap = Alphabet["A0"] + Alphabet["A10"] 
        self.r = self.r0 + 2* self.r1
        self.T = T
        
        self.X = np.random.choice(self.k, self.n)
        
        self.p = np.random.default_rng().dirichlet([100* self.k for i in range(self.k)], 1)[0]
        
        eta_matrix = {(k,h): np.random.default_rng().dirichlet([1 for i in range(self.r0 + self.r1 if k == h else self.r)], 1) for k in range(self.k) for h in range(k, self.k)}
        self.eta = eta_matrix
        for (k,h) in self.eta:
            if k != h:
                self.eta[(k,h)] = {self.A[i]: eta_matrix[(k,h)][0][i] for i in range(self.r) }
            else:
                self.eta[(k,h)] = {self.Ap[i]: eta_matrix[(k,h)][0][i] for i in range(self.r0 + self.r1) }
                

    def sample_improved(self, M0):
        schedule_T = np.linspace(10*self.n, self.T *self.k, M0).astype(int)
        schedule_w = np.linspace(1/n, 1, M0)
        for t in range(M0):
            self.step1_improved(schedule_T[t], schedule_w[t])
            for i in range(self.n):
                self.step2(i)
        for t in range(M0):
            self.step1()
            for i in range(self.n):
                self.step2(i)
        return self.X
    

    def sample(self, tmax):
        # if you don't want to use the method for improve convergence
        for t in range(tmax):
            self.step1()
            for i in range(self.n):
                self.step2(i)
        return self.X
    

    def step1_improved(self, T0, w0):
        if T0 is None:
            T0 = self.T * self.k
        m = [sum([self.X[i] == k for i in range(self.n)]) for k in range(self.k) ]    
        self.p = np.random.default_rng().dirichlet([T0 + m[classe] for classe in range(self.k)], 1)[0]        

        eta_matrix = {(k,h): np.random.default_rng().dirichlet([max(1, w0*( 1 + self.e(a,k,h))) for a in (self.Ap if k == h else self.A)], 1) for k in range(self.k) for h in range(k, self.k)}
        
        for (k,h) in self.eta:
            if k != h:
                self.eta[(k,h)] = {self.A[i]: eta_matrix[(k,h)][0][i] for i in range(self.r) }
            else:
                self.eta[(k,h)] = {self.Ap[i]: eta_matrix[(k,h)][0][i] for i in range(self.r0 + self.r1) }
      
    
    def step1(self):
        m = [sum([self.X[i] == k for i in range(self.n)]) for k in range(self.k) ]    
        self.p = np.random.default_rng().dirichlet([self.T * self.k + m[classe] for classe in range(self.k)], 1)[0]        

        eta_matrix = {(k,h): np.random.default_rng().dirichlet([1 + self.e(a,k,h) for a in (self.Ap if k == h else self.A)], 1) for k in range(self.k) for h in range(k, self.k)}
        
        for (k,h) in self.eta:
            if k != h:
                self.eta[(k,h)] = {self.A[i]: eta_matrix[(k,h)][0][i] for i in range(self.r) }
            else:
                self.eta[(k,h)] = {self.Ap[i]: eta_matrix[(k,h)][0][i] for i in range(self.r0 + self.r1) }
   
    
    def step2(self, i):
        log_probas = np.zeros(self.k)
        for classe in range(self.k):
            log_probas[classe] = self.step2_inner(i, classe) + np.log(self.p[classe])
        norm_probas = self.normalize_proba(log_probas)
        
        self.X[i] = np.random.choice(self.k, 1, p=norm_probas)[0]


    def step2_inner(self, i, classe):
        log_proba = 0
        for j in range(self.n):
            if self.seen(i,j):        
                kh , a12  = self.find_eta_indices(classe,self.X[j] , (self.Y[i][j], self.Y[j][i] ))
                log_proba += np.log(self.eta[kh][a12])
        return log_proba
     

    def seen(self, i,j):
        return (i,j) in self.N or (j,i) in self.N
      

    def e(self,a, k,h):
        edge = 0           
        a1, a2 = a
        for (i,j) in self.N:
            if self.X[i] == k and self.X[j] == h:
                if self.Y[i][j] == a1 and self.Y[j][i] == a2:
                    edge += 1
            if self.X[i] == h and self.X[j] == k:
                if self.Y[i][j] == a2 and self.Y[j][i] == a1:
                    edge += 1
        if k == h and a1 != a2:
            edge = edge /2
        return edge
 
        
    def find_eta_indices(self, k, h, a):
        a1, a2 = a
        if k < h:
            return (k,h), (a1, a2)
        elif k == h:
            if (a1, a2) in self.eta[(k,h)].keys():
                return (k,h), (a1, a2)
            else:
                return (k, h), (a2, a1)
        else:
           return (h,k),(a2, a1)
            
    
    def Iy(self):
        """
        Another metric
        """
        temp = 0
        for (i,j) in self.N: 
            kh, a12 = self.find_eta_indices(self.X[i],self.X[j],(self.Y[i][j], self.Y[j][i] ) )
            temp += np.log(self.eta[kh][a12])
        return -1/len(self.N) * temp


    def Hx(self):
        """
        Another metric
        """
        temp = 0 
        full_probas = []
        for i in range(self.n):
            probas = []
            z = 0
            for classe in range(self.k):
                probas.append(self.step2_inner(i, classe) * self.p[classe])
                z += probas[classe]
            for classe in range(self.k):
                probas[classe] /= z
            full_probas.append(probas)
                
        for i in range(self.n):
            for j in range(self.n):
                pi_ij = 0
                for classe in range(self.k):
                    pi_ij += full_probas[i][classe]*full_probas[j][classe] 
                temp += pi_ij * (1- pi_ij)
        return temp * 4/(self.n * (self.n - 1))
    
        
    def logsumexp(self, x):
        c = x.max()
        return c + np.log(np.sum(np.exp(x - c)))


    def normalize_proba(self, x):
        return np.exp(x - self.logsumexp(x))
  
        


############################################################################


# we first build the stochastic block model...
# number of nodes
n = 30
# set of observed pairs
N = {(a,b) for a in range(n) for b in range(a + 1, n) } - {(a,a) for a in range(n)}
# writing the alphabet separating between A0, A10, A11
Alphabet = {"A0": [(0,0), (1,1)], "A10": [], "A11": []}
# set of classes: is a dict in the form { class1: prior_proba_class1, ...}
Classes = {0: 0.5, 1: 0.3, 2: 0.2}
# eta is defined as a dict of dicts, as {(c, d): {a1: proba1, a2 : proba2, ...}, ... }
eta = {(0,0):     {(1,1): 1, (0,0) : 0},        # classes: c = 0, d = 0.   undirected edge is (1,1), no edge is (0,0)      
       (0, 1):       {(1,1): 0, (0,0) : 1},     # classes: c = 0, d = 1
       (1, 1):          {(1,1): 1, (0,0) : 0},  # classes: c = 1, d = 1
       (2, 2):      {(1,1): 1, (0,0) : 0},
       (2,0):      {(1,1): 0.1, (0,0) : 1},
       (2, 1):        {(1,1): 0.1, (0,0) : 1}
       }
# creation of the model
model = StochasticBlockModel_General(n, N, Classes, Alphabet, eta)


# now we can make inference on it
k = len(Classes)
# initialization
gibbs = Gibbs_sampler_General(N, model.Y, k, Alphabet, T = 100)
# prediction
prediction = gibbs.sample_improved(500)


# results: NMI
print("NMI: ", normalized_mutual_info_score(model.X, prediction))
