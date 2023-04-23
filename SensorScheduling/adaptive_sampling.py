import numpy as np


def sigma(pi,y,f,A,S,D,u=1):
    unit = np.ones((S,1))
    # print(np.matmul(f[y],np.matmul(np.linalg.matrix_power(A.T,D[u]),pi)),f"action: {u}")
    return np.matmul(unit.T,np.matmul(f[y],np.matmul(np.linalg.matrix_power(A.T,D[u]),pi)))[0][0]

def T(pi,y,f,A,S,D,u=1):
    numerator = np.matmul(f[y],np.matmul(np.linalg.matrix_power(A.T,D[u]),pi))
    denominator = sigma(pi,y,f,A,S,D,u)
    # print(numerator, denominator)
    return numerator / denominator

def C(A,u,c,m,D):
    total = np.zeros((2,1))
    add_term = np.zeros((2,2))
    for i in range(D[u]):
        add_term = np.add(np.linalg.matrix_power(A,i),add_term)
    total = np.add(m[u],np.matmul(add_term,c[u]))
    # print(total.shape)
    return total

class Machine():
    def __init__(self,A,U,S,f,m,c,D,Y):
        self.A = A                                      # probability transition matrix
        self.m = m                                      
        self.c = c
        self.D = D
        self.Y = Y
        self.U = U                                      # number of actions
        self.S = S                                      # number of states
        self.f = f                                      # observation probability matrix
        self.policy = np.zeros((1001,1),dtype=np.int32)   # stationary policy vector
        self.Vn = np.zeros((1001,1))                      # expected cumulative cost

    def train(self):
        while True:
            V_next = np.zeros((1001,1))
            # 11 belief states are possible : 0.0, 0.1, 0.2, ... 0.9, 1.0
            for j in range(1001):
                # for belief_state = j*0.1
                curr_state = np.array([[j*0.001],[1-j*0.001]])

                # for action 0
                vnext = []
                for a in range(self.U):

                    # y10 = round(T(curr_state, 1, self.f, self.A, self.S)[1][0], 3)   # observation 1
                    # sigma1 = sigma(curr_state, 1, self.f, self.A, self.S)
                    # y20 = round(T(curr_state, 2, self.f, self.A, self.S)[1][0], 3)   # observation 1
                    # sigma2 = sigma(curr_state, 2, self.f, self.A, self.S)
                    if a == 0 or j == 1000:
                        vnext.append(np.matmul(self.c[a].T, curr_state).item())
                    else:
                        # print(f"State: {j}")
                        ysigma = 0
                        for y in range(self.Y):
                            v = round(T(curr_state, y, self.f, self.A, self.S, self.D, a)[0][0], 3)   # observation y
                            sig = sigma(curr_state, y, self.f, self.A, self.S, self.D, a)
                            ysigma += self.Vn[int(v*1000)].item()*sig
                        vnext.append(np.matmul(C(self.A,a,self.c,self.m,self.D).T, curr_state).item() + ysigma)
                    
                    # V_next_0 = np.matmul(self.C[0].T, curr_state).item()
                
                # for action 1
                # y10 = round(T(curr_state, 0, self.B, self.P, self.S)[1][0], 1)   # observation 0
                # sigma0 = sigma(curr_state, 0, self.B, self.P, self.S)
                # y11 = round(T(curr_state, 1, self.B, self.P, self.S)[1][0], 1)   # observation 1
                # sigma1 = sigma(curr_state, 1, self.B, self.P, self.S)
                # V_next_1 = np.matmul(self.C[1].T, curr_state).item() + self.Vn[int(y10*10)]*sigma0 + self.Vn[int(y11*10)]*sigma1

                V_next[j][0] = min(vnext)
                print(vnext)
                self.policy[j][0] = np.argmin(vnext)
                # if V_next_0 < V_next_1:
                #     self.policy[j][0] = 0
                # else:
                #     self.policy[j][0] = 1   

            max_diff = 0
            for i in range(1001):
                max_diff = max(max_diff,abs(V_next[i][0] - self.Vn[i][0]))
            
            if max_diff <= 0.00001:
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
