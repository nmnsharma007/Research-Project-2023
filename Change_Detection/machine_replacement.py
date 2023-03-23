import numpy as np


def sigma(pi,y,B,P,S,u=1):
    unit = np.ones((S,1))
    return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

def T(pi,y,B,P,S,u=1):
    numerator = np.matmul(B[y],np.matmul(P[u].T,pi))
    denominator = sigma(pi,y,B,P,S)
    return numerator / denominator


class Machine():
    def __init__(self,P,C,A,S,B,N):
        self.P = P                                      # probability transition matrix
        self.C = C                                      # cost matrix
        self.A = A                                      # number of actions
        self.S = S                                      # number of states
        self.B = B                                      # observation probability matrix
        self.N = N                                      # length of the horizon
        self.lookup = np.empty((11,N), dtype=np.int32)  # lookup table
        self.Jk = np.zeros((11,1), dtype=np.float32)    # expected cummulative cost

    def train(self):
        for i in reversed(range(self.N)):
            # 11 belief states are possible : 0.0, 0.1, 0.2, ... 0.9, 1.0
            for j in range(11):
                # for belief_state = j*0.1
                curr_state = np.array([[j*0.1],[1-(j*0.1)]])

                # for action 0
                J_nextK0 = np.matmul(self.C[0].T, curr_state).item() + self.Jk[0][0]

                # for action 1
                y0 = round(T(curr_state, 0, self.B, self.P, self.S)[0][0], 1)   # observation 0
                sigma0 = sigma(curr_state, 0, self.B, self.P, self.S)
                y1 = round(T(curr_state, 1, self.B, self.P, self.S)[0][0], 1)   # observation 1
                sigma1 = sigma(curr_state, 1, self.B, self.P, self.S)
                J_nextK1 = np.matmul(self.C[1].T, curr_state).item() + self.Jk[int(y0*10)]*sigma0 + self.Jk[int(y1*10)]*sigma1

                # selecting the action with minimum cost and accordingly changing the lookup table
                if J_nextK0 < J_nextK1:
                    self.Jk[j][0] = J_nextK0
                    self.lookup[j][i] = 0
                else:
                    self.Jk[j][0] = J_nextK1
                    self.lookup[j][i] = 1

'''
pi = np.array([[0.5],[0.5]])
p = 0.3
q = 0.6
theta = 0.4
B = np.array([[[1-q,0],[0,p]],[[q,0],[0,1-p]]],dtype=np.float32)
P = np.array([[[0,1],[0,1]],[[1,0],[theta,1-theta]]],dtype=np.float32)
c = 40
R = 1
C = C = np.array([[[0],[c]],[[R],[R]]])
S = 2
A = 2
N = 4

# print(B.shape, P.shape, pi.shape, T(pi, 0, B, P, S).shape)

m = Machine(P,C,A,S,B,N)
m.train()
print(m.lookup)
'''