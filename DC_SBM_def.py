################################## Degree corrected SBM ########################################################
import random
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import normalized_mutual_info_score



class StochasticBlockModel():
    """
    Standard stochastic block model (generative)
    n = number of nodes
    p = prior probabilities on the classes
    W = what is called eta
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
    n = number of nodes
    p = prior probabilities on the classes
    W = what is called eta
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
     n = number of nodes
     p = prior probabilities on the classes
     W = what is called eta 
     theta as in the definition
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
    
  

class Separated_groups():
    """
    Generative model for the separated groups model
    n as usual
    exp_degrees is a list of expected degrees, and they will taken at random for each node
    k as usual
    lambd is the lambda in the thesis
    
    NB: for the scale free network I used a modification of this code
    """
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
    """
    Generative model for the Core peryphery model
    n as usual
    exp_degrees is a list of lists of expected degrees. In this case the expected degrees are class dependent
    k as usual
    lambd is the lambda in the thesis
    """
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
    """
    Generative model for the hierarchical model
    n as usual
    exp_degrees is a list of lists of expected degrees. In this case the expected degrees are class dependent (even though on the thesis I used the same for all)
    k as usual
    lambd is the lambda in the thesis
    """
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
        instance the class, and after that use the function infer(n_samples), where
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
        # update m      number of edges between class r and class s
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
        self.kit = np.zeros((self.n, self.K), dtype = "int64")
        for i in range(self.n):
            for j in range(self.n):
                if j!= i:
                    self.kit[i][self.X[j]] += self.Y[i][j]
                
        
    def infer(self, n_samples):
        max_score = -np.inf
        max_config = None
        for t in range(n_samples):
            print(t)
            self.initialize()
            config, score = self.run()
            if score > max_score:
                max_score = score
                max_config = config.copy()
        return max_config
    
    
    def run(self):
        #set the initial score
        max_score = self.score()
        Backup_max_config = self.X.copy()
        #cycle until the score does not change
        while True:
            # max_config is the configuration with the highest score in the loop
            max_config = None
            # available nodes
            nodes = {i for i in range(self.n)}
            # to change all the nodes
            for t in range(self.n):
                max_change = - np.inf
                # node is the node that will be changed in new_class
                node = None
                new_class = None
                # loop to decide the best node and move to use
                for i in nodes:
                    for s in range(self.K):
                        change = self.change_in_loglikel(i, self.X[i], s)
                        if change > max_change:
                            node = i
                            new_class = s
                            max_change = change                          
                # do the move
                self.move(node, new_class)
                #score of the configuration
                score = self.score()
                if  score > max_score:
                    max_config = self.X.copy()
                    Backup_max_config = max_config.copy()
                    max_score = score 
                # make the node unavailable
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
        
    

                    
                    

###############################################################################################

# build the graph
k = 4
n = 100
p = [1/k for i in range(k)]
exp_degrees = [10,20,30]
model = Separated_groups(n, exp_degrees, k, 0.8)

# Infer
a = GreedyAlgorithm(model.Y, k)    
prediction = a.infer(10)

# results
print("NMI: ", normalized_mutual_info_score(model.X, prediction))
