################################## Degree corrected SBM ########################################################
import random
import numpy as np
import math
from scipy.stats import poisson
import time
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score



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
    
    
    
    
    

class Multigraph_SBM_noDC():
    """
    Multigraph SBM, without degree correction (generative)    
    """
    
    def __init__(self, n, p, W):
        self.n = n
        self.p = p
        self.W = W
        self.k = len(self.W[0])
        self.X = []
        for i in range(self.n):
            self.X.append(random.choices([i for i in range(self.k)], p)[0])
        
        self.X = np.array(self.X)
        self.Y = np.zeros((self.n, self.n), dtype = "int64")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.Y[i][j] = int(poisson.rvs(self.W[self.X[i]][self.X[j]], size = 1)[0])
                self.Y[j][i] = self.Y[i][j]
        
        for i in range(self.n):
            self.Y[i][i] = 2 * int(poisson.rvs(self.W[self.X[i]][self.X[i]], size = 1)[0])
                            
    
    def __repr__(self):
        temp = "Adj matrix\n"
        for row in self.Y:
            temp += str(row) + "\n"
        temp += "\nClasses:\n"
        temp += str(self.X)
        return temp    
    
    
    
class Multigraph_SBM_DC():
    """
     Multigraph SBM, with degree correction (generative) 
    """
    
    def __init__(self, n, p, W, theta):
        self.n = n
        self.p = p
        self.W = W
        self.k = len(self.W[0])
        self.theta = theta
        self.X = []
        for i in range(self.n):
            self.X.append(random.choices([i for i in range(self.k)], p)[0])
        
        self.X = np.array(self.X)
        self.Y = np.zeros((self.n, self.n), dtype = "int64")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.Y[i][j] = int(poisson.rvs( self.theta[i] * self.theta[j] * self.W[self.X[i]][self.X[j]], size = 1)[0])
                self.Y[j][i] = self.Y[i][j]
        
        for i in range(self.n):
            self.Y[i][i] = 2 * int(poisson.rvs(self.theta[i] ** 2 * self.W[self.X[i]][self.X[i]], size = 1)[0])
                            
    
    def __repr__(self):
        temp = "Adj matrix\n"
        for row in self.Y:
            temp += str(row) + "\n"
        temp += "\nClasses:\n"
        temp += str(self.X)
        return temp
    


class Synthetic_data_article():
    def __init__(self, n, exp_degrees, k, lambd):
        self.exp_degrees = exp_degrees
        self.k = k
        self.n = n
        self.X = np.random.choice([i for i in range(self.k)], size = self.n)
        self.deg = np.random.choice(exp_degrees, size = self.n)
        self.kappa = np.zeros(self.k)
        for i in range(self.n):
            self.kappa[self.X[i]] += self.deg[i]
        self.theta = np.zeros(self.n)
        for i in range(self.n):
            self.theta[i] = self.deg[i]/self.kappa[self.X[i]]
        w_planted = np.zeros((self.k, self.k))
        for r in range(k):
            w_planted[r][r] = self.kappa[r]
        w_random = np.zeros((self.k, self.k))
        m = np.sum(self.kappa)/2
        for r in range(self.k):
            for s in range(self.k):
                w_random[r][s] = self.kappa[r]*self.kappa[s]/(2*m)
        self.w = lambd * w_planted + (1- lambd) * w_random
        # edges = np.zeros((k,k))
        # for r in range(self.k):
        #     for s in range(r, self.k):
        #         if s == r:
        #             edges[r][s] = int(poisson.rvs(self.w[r][s]/2, size = 1)[0])
        #         else:
        #             edges[r][s] = int(poisson.rvs(self.w[r][s]/2, size = 1)[0])
        #             edges[s][r] = edges[r][s]
        self.Y = np.zeros((n,n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.Y[i][j] = int(poisson.rvs( self.theta[i] * self.theta[j] * self.w[self.X[i]][self.X[j]], size = 1)[0])
                self.Y[j][i] = self.Y[i][j]
        
        for i in range(self.n):
            self.Y[i][i] = 2 * int(poisson.rvs(self.theta[i] **2 * self.w[self.X[i]][self.X[i]], size = 1)[0])
                     
        

class Separated_groups():
    def __init__(self, n, exp_degrees, k, lambd):
        self.exp_degrees = exp_degrees
        self.k = k
        self.n = n
        self.X = np.random.choice([i for i in range(self.k)], size = self.n)
        self.deg = np.random.choice(exp_degrees, size = self.n)
        self.kappa = np.zeros(self.k)
        for i in range(self.n):
            self.kappa[self.X[i]] += self.deg[i]
        self.theta = np.zeros(self.n)
        for i in range(self.n):
            self.theta[i] = self.deg[i]/self.kappa[self.X[i]]
        w_planted = np.zeros((self.k, self.k))
        for r in range(k):
            w_planted[r][r] = self.kappa[r]
        w_random = np.zeros((self.k, self.k))
        m = np.sum(self.kappa)/2
        for r in range(self.k):
            for s in range(self.k):
                w_random[r][s] = self.kappa[r]*self.kappa[s]/(2*m)
        self.w = lambd * w_planted + (1- lambd) * w_random
        self.Y = np.zeros((n,n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.Y[i][j] = int(poisson.rvs( self.theta[i] * self.theta[j] * self.w[self.X[i]][self.X[j]], size = 1)[0])
                self.Y[j][i] = self.Y[i][j]
        
        for i in range(self.n):
            self.Y[i][i] = 2 * int(poisson.rvs(self.theta[i] **2 * self.w[self.X[i]][self.X[i]], size = 1)[0])
                           

class Core_Periphery():
    def __init__(self, n, exp_degrees, lambd):
        self.exp_degrees = exp_degrees
        self.k = 2
        self.n = n
        self.X = np.random.choice([i for i in range(self.k)], size = self.n)
        
        self.deg = np.zeros(self.n)
        for i in range(self.n):
            if self.X[i] == 0:
                self.deg[i] = np.random.choice(exp_degrees[0])
            else:
                self.deg[i] = np.random.choice(exp_degrees[1])
        
        self.kappa = np.zeros(self.k)
        for i in range(self.n):
            self.kappa[self.X[i]] += self.deg[i]
        self.theta = np.zeros(self.n)
        for i in range(self.n):
            self.theta[i] = self.deg[i]/self.kappa[self.X[i]]
            
        w_planted = np.zeros((2,2))
        w_planted[0][0] = self.kappa[0] - self.kappa[1]
        w_planted[0][1] = self.kappa[1]
        w_planted[1][0] = self.kappa[1]
        w_planted[1][1] = 0 
            
            
        w_random = np.zeros((self.k, self.k))
        m = np.sum(self.kappa)/2
        for r in range(self.k):
            for s in range(self.k):
                w_random[r][s] = self.kappa[r]*self.kappa[s]/(2*m)
        self.w = lambd * w_planted + (1- lambd) * w_random
        self.Y = np.zeros((n,n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.Y[i][j] = int(poisson.rvs( self.theta[i] * self.theta[j] * self.w[self.X[i]][self.X[j]], size = 1)[0])
                self.Y[j][i] = self.Y[i][j]
        
        for i in range(self.n):
            self.Y[i][i] = 2 * int(poisson.rvs(self.theta[i] **2 * self.w[self.X[i]][self.X[i]], size = 1)[0])
                           

class Hierarchical():
    def __init__(self, n, exp_degrees, A, lambd):
        self.exp_degrees = exp_degrees
        self.k = 3
        self.A = A
        self.n = n
        self.X = np.random.choice([i for i in range(self.k)], size = self.n)
        
        self.deg = np.zeros(self.n)
        for i in range(self.n):
            if self.X[i] == 0:
                self.deg[i] = np.random.choice(exp_degrees[0])
            elif self.X[i] == 1:
                self.deg[i] = np.random.choice(exp_degrees[1])
            else: 
                self.deg[i] = np.random.choice(exp_degrees[2])
        
        self.kappa = np.zeros(self.k)
        for i in range(self.n):
            self.kappa[self.X[i]] += self.deg[i]
        self.theta = np.zeros(self.n)
        for i in range(self.n):
            self.theta[i] = self.deg[i]/self.kappa[self.X[i]]
            
        w_planted = np.zeros((self.k,self.k))
        w_planted[0][0] = self.kappa[0] - self.A
        w_planted[0][1] = A
        w_planted[1][0] = A
        w_planted[1][1] = self.kappa[1] - self.A 
        w_planted[2][2] = self.kappa[2]
       
            
        w_random = np.zeros((self.k, self.k))
        m = np.sum(self.kappa)/2
        for r in range(self.k):
            for s in range(self.k):
                w_random[r][s] = self.kappa[r]*self.kappa[s]/(2*m)
        self.w = lambd * w_planted + (1- lambd) * w_random
        self.Y = np.zeros((n,n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                self.Y[i][j] = int(poisson.rvs( self.theta[i] * self.theta[j] * self.w[self.X[i]][self.X[j]], size = 1)[0])
                self.Y[j][i] = self.Y[i][j]
        
        for i in range(self.n):
            self.Y[i][i] = 2 * int(poisson.rvs(self.theta[i] **2 * self.w[self.X[i]][self.X[i]], size = 1)[0])
                           



class GreedyAlgorithm():
    """
    Greedy algorithm for DC-SBM
    
    parameters:
        Y: adjacency matrix
        K: # of classes
        
    output:
        predicted class assignment X
        
    how to use it:
        instance the class, and after that use the function infere(n_samples), where
        n_sample is the number of random initialization (and returns the best result)
        
    """
    
    def __init__(self, Y, K):
        # Y: adj matrix     n: # of nodes       K: # of classes
        self.Y = Y
        self.n = len(self.Y[0])
        self.K = K
        
        
    def initialize(self):
        #randomly set the X
        self.X = np.random.choice(self.K, self.n)
        
        # degrees
        self.degree = np.sum(self.Y, axis = 0)
        
        # self-edges
        self.u = self.Y.diagonal()//2
        
        self.update()
        
        
    def update(self):
        # update m      numero di edges tra classe r e classe s
        self.m = np.zeros((self.K, self.K))
        for i in range(self.n):
            for j in range(self.n):
                self.m[self.X[i]][self.X[j]] += self.Y[i][j]
        
        # update kappa (stubs)
        self.kappa = np.zeros(self.K)
        for r in range(self.K):
            for s in range(self.K):
                self.kappa[r] += self.m[r][s]
        
        # update kit[i][t]     numero di edges che partono da i e che vanno alla classe t (esclusi self-edges)
        # diciamo che si potrebbe ottimizzare perchè poi certi updates diventerebbero inutili
        self.kit = np.zeros((self.n, self.K), dtype = "int64")
        for i in range(self.n):
            for j in range(self.n):
                if j!= i:
                    self.kit[i][self.X[j]] += self.Y[i][j]
                
        
        

    def infere(self, n_samples):
        max_score = -np.inf
        max_config = None
        for t in range(n_samples):
            print(t)
            self.initialize()
            config, score = self.run()
            if score > max_score:
                max_score = score
                max_config = config.copy()
        return max_config, max_score
    
    def run(self):
        #setti lo score di partenza
        max_score = self.score()
        Backup_max_config = self.X.copy()
        #cicli finchè lo score rimane invariato
        while True:
            # max_config è la configuarazione con maggior score alla fine del ciclo
            max_config = None
            # available nodes
            nodes = {i for i in range(self.n)}
            # cicli perchè devi cambiare tutti i nodi
            for t in range(self.n):
                max_change = - np.inf
                # node è il nodo che verrà cambiato in new_class
                node = None
                new_class = None
                # ciclo per decidere quale nodo e quale mossa dà il maggior cambio in loglikelihood
                for i in nodes:
                    for s in range(self.K):
                        change = self.change_in_loglikel(i, self.X[i], s)
                        if change > max_change:
                            node = i
                            new_class = s
                            max_change = change                          
                # fai la mossa
                self.move(node, new_class)
                #lo score della configurazione (ricorda che l'update l'ho tolto dall'interno dello score, e l'ho messo in move)
                score = self.score()
                if  score > max_score:
                    max_config = self.X.copy()
                    Backup_max_config = max_config.copy()
                    max_score = score 
                # rendi il nodo non available
                nodes.remove(node)
            
            if max_config is None:
                return Backup_max_config.copy(), max_score
            else:
                self.X = max_config.copy()
                self.update()
    
    def move(self, node, new_class):
        self.X[node] = new_class
        self.update()
        
        
    def score(self):
        temp = 0 
        for r in range(self.K):
            for s in range(self.K):
                if self.m[r][s] == 0:
                    temp += 0
                else:
                    temp += self.m[r][s] * np.log(self.m[r][s]/(self.kappa[r] * self.kappa[s]))
        return temp
    
        
    def change_in_loglikel(self, i, r, s):
        if r == s:
            # I impose that r != s
            return -np.inf
        temp = 0
        for t in range(self.K):
            if t != r and t != s:
                temp += self.a(self.m[r][t] - self.kit[i][t]) -self.a(self.m[r][t])  
                temp += self.a(self.m[s][t] + self.kit[i][t]) -self.a(self.m[s][t])   
        temp += self.a(self.m[r][s] + self.kit[i][r] - self.kit[i][s]) - self.a(self.m[r][s])
        temp += self.b(self.m[r][r] - 2 * (self.kit[i][r] + self.u[i])) - self.b(self.m[r][r])
        temp += self.b(self.m[s][s] + 2 * (self.kit[i][s] + self.u[i])) - self.b(self.m[s][s])
        temp += - self.a(self.kappa[r]- self.degree[i]) + self.a(self.kappa[r])
        temp += - self.a(self.kappa[s] + self.degree[i]) + self.a(self.kappa[s])
        return temp
      

    def b(self,x):
        return 0 if x == 0 else x * np.log(x) 

    def a(self, x):
        return 2 * self.b(x)
        
    

def corrette_k_3(true,pred):
    n = len(pred)
    pos = {0,1,2}
    for a in range(3):
        for b in pos.difference({a}):
            for c in pos.difference({a,b}):
                temp = 0
                sol = [a,b,c]
                for i in range(n):
                    if true[i] == sol[pred[i]]:
                        temp += 1
                print(temp, temp/n)
                    
                    

###############################################################################################

# k = 4
# n = 100
# p = [1/k for i in range(k)]
# W = [[0.2 if i==j else 0.01 for i in range(k)]for j in range(k)] #poisson
# # model = Multigraph_SBM_noDC(n, p, W)
# # a = GreedyAlgorithm(model.Y, k)   
# # a.initialize() 
# # pred, score = a.infere(1)
# # print(normalized_mutual_info_score(model.X, pred))

# theta = 3 * np.random.random(n)
# model = Multigraph_SBM_DC(n, p, W, theta)
# a = GreedyAlgorithm(model.Y, k)   
# a.initialize() 
# pred, score = a.infere(10)
# print(normalized_mutual_info_score(model.X, pred))