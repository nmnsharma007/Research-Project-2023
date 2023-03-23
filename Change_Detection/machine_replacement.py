import numpy as np


def sigma(pi,y,B,P,S,u=1):
    unit = np.ones((S,1))
    return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

def T(pi,y,B,P,S,u=1):
    numerator = np.matmul(B[y],np.matmul(P[u].T,pi))
    denominator = sigma(pi,y,B,P,S)
    return numerator / denominator


class Machine():
    def __init__(self,P,C,A,S,B):
        self.P = P                                      # probability transition matrix
        self.C = C                                      # cost matrix
        self.A = A                                      # number of actions
        self.S = S                                      # number of states
        self.B = B                                      # observation probability matrix
        self.policy = np.zeros((11,1),dtype=np.int32)   # stationary policy vector
        self.Vn = np.zeros((11,1))    # expected cumulative cost

    def train(self):
        while True:
            V_next = np.zeros((11,1))
            # 11 belief states are possible : 0.0, 0.1, 0.2, ... 0.9, 1.0
            for j in range(11):
                # for belief_state = j*0.1
                curr_state = np.array([[1-j*0.1],[j*0.1]])

                # for action 0
                y00 = round(T(curr_state, 0, self.B, self.P, self.S)[1][0], 1)   # observation 0
                sigma0 = sigma(curr_state, 0, self.B, self.P, self.S)
                y10 = round(T(curr_state, 1, self.B, self.P, self.S)[1][0], 1)   # observation 1
                sigma1 = sigma(curr_state, 1, self.B, self.P, self.S)
                V_next_0 = np.matmul(self.C[0].T, curr_state).item() + self.Vn[int(y00*10)]*sigma0 + self.Vn[int(y10*10)]*sigma1
                
                # for action 1
                y10 = round(T(curr_state, 0, self.B, self.P, self.S)[1][0], 1)   # observation 0
                sigma0 = sigma(curr_state, 0, self.B, self.P, self.S)
                y11 = round(T(curr_state, 1, self.B, self.P, self.S)[1][0], 1)   # observation 1
                sigma1 = sigma(curr_state, 1, self.B, self.P, self.S)
                V_next_1 = np.matmul(self.C[1].T, curr_state).item() + self.Vn[int(y10*10)]*sigma0 + self.Vn[int(y11*10)]*sigma1

                V_next[j][0] = min(V_next_0,V_next_1)
                if V_next_0 < V_next_1:
                    self.policy[j][0] = 0
                else:
                    self.policy[j][0] = 1   

            max_diff = 0
            for i in range(11):
                max_diff = max(max_diff,abs(V_next[i][0] - self.Vn[i][0]))
            
            if max_diff <= 0.01:
                break
                
            
            self.Vn = np.copy(V_next)

                # # selecting the action with minimum cost and accordingly changing the lookup table
                # if J_nextK0 < J_nextK1:
                #     self.Jk[j][0] = J_nextK0
                #     self.lookup[j][i] = 0
                # else:
                #     self.Jk[j][0] = J_nextK1
                #     self.lookup[j][i] = 1

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