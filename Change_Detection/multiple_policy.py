import numpy as np


def sigma(pi,y,B,P,S,u):
    unit = np.ones((S,1))
    return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

def T(pi,y,B,P,S,u):
    numerator = np.matmul(B[y],np.matmul(P[u].T,pi))
    denominator = sigma(pi,y,B,P,S,u)
    return numerator / denominator


class ValueMachine():
    def __init__(self,P,C,A,S,B,mu):
        self.P = P                                      # probability transition matrix
        self.C = C                                      # cost matrix
        self.A = A                                      # number of actions
        self.S = S                                      # number of states
        self.B = B                                      # observation probability matrix
        self.policy = np.zeros((11,1),dtype=np.int32)   # stationary policy vector
        self.Vn = np.zeros((11,1))                      # expected cumulative cost
        self.mu = mu                                    # policy

    def train(self):
        while True:
            V_next = np.zeros((11,1))
            # 11 belief states are possible : 0.0, 0.1, 0.2, ... 0.9, 1.0
            for j in range(11):
                # for belief_state = j*0.1
                curr_state = np.array([[1-j*0.1],[j*0.1]])

                action = 1 if j >= int(self.mu*10) else 0

                # for action 0
                y00 = round(T(curr_state, 0, self.B, self.P, self.S, action)[1][0], 1)   # observation 0
                sigma0 = sigma(curr_state, 0, self.B, self.P, self.S,action)
                y10 = round(T(curr_state, 1, self.B, self.P, self.S, action)[1][0], 1)   # observation 1
                sigma1 = sigma(curr_state, 1, self.B, self.P, self.S,action)
                V_next_0 = np.matmul(self.C[action].T, curr_state).item() + self.Vn[int(y00*10)]*sigma0 + self.Vn[int(y10*10)]*sigma1

                V_next[j][0] = V_next_0

            max_diff = 0
            for i in range(11):
                max_diff = max(max_diff,abs(V_next[i][0] - self.Vn[i][0]))
            
            if max_diff <= 0.0001:
                break
                
            
            self.Vn = np.copy(V_next)