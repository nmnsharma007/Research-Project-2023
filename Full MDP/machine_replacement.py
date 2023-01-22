import numpy as np
import math

class Machine():
    def __init__(self,P,C,A,S,N):
        self.P = P # probability transition matrix
        self.C = C # cost vector
        self.A = A # number of actions
        self.S = S # number of states
        self.lookup = np.empty((2,N)) # lookup table
        self.N = N # horizon length

    def train(self):
        Jk = np.array([[0],[0]],dtype=np.float32)
        for i in reversed(range(N)):
            J_nextk = np.empty((self.A,self.S,1))
            for a in range(self.A):
                J_nextk[a] = self.C[a] + np.matmul(P[a],Jk)
            J_newk = Jk.copy()
            for s in range(self.S):
                J_newk[s] = math.inf
                for a in range(self.A):
                    if J_newk[s] > J_nextk[a][s]:
                        J_newk[s] = J_nextk[a][s] # update cost to a lower value
                        self.lookup[s][i] = a # selecting action with lower cost
            Jk = J_newk.copy()
        print(Jk)

theta = 0.1
P = np.array([
    [[1 - theta,theta],[0,1]],
    [[1,0],[1,0]]]
)

c = 4
R = 6
C = np.array([[[0],[c]],[[R],[R]]])

A = 2 # number of actions
S = 2 # number of states
N = 100000 # horizon length

machine = Machine(P,C,A,S,N)
machine.train()

print(machine.lookup)