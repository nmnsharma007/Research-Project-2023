import numpy as np

class Machine():
    def __init__(self,P,C,U,S,B,N,Y):
        self.P = P                                      # probability transition matrix
        self.C = C                                      # cost matrix
        self.U = U                                      # number of actions
        self.S = S                                      # number of states
        self.B = B                                      # observation probability matrix
        self.N = N                                      # length of the horizon
        self.Y = Y                                      # number of observations
        self.lookup = [[] for x in range(self.N)]          # lookup table

    def train(self):
        T = np.zeros((1,2,1))
        # print(T.shape)
        for k in reversed(range(self.N)):
            T_k = [] # vectors for current iteration
            for u in range(self.U):
                T_k_u = []
                T_k_u_y = []
                for y in range(self.Y):
                    T_fixed_action = np.empty((0,2,1),dtype=np.float32)
                    for i in range(T.shape[0]):
                        new_vector = (self.C[u] / self.Y) + np.matmul(np.matmul(self.P[u],self.B[u].T),T[i])
                        # print(new_vector.shape)
                        T_fixed_action = np.vstack((T_fixed_action,[new_vector]))
                        # print(T_k_u_y.shape)
                    T_k_u_y.append(T_fixed_action)
                T_k_u_y = np.array(T_k_u_y)
                # print(T_k_u_y.shape)
                for i in range(T_k_u_y[0].shape[0]):
                    for j in range(T_k_u_y[1].shape[0]):
                        result = T_k_u_y[0][i] + T_k_u_y[1][j]
                        T_k_u.append(result)
                T_k_u = np.array(T_k_u)
                # print(T_k_u.shape)
                self.lookup[k].append(T_k_u)
                T_k.extend(T_k_u)
                # T_k = np.array(T_k)
                # print(T_k.shape)
                # break
            T = T_k
            T = np.array(T)
            print(T.shape)
            # break




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
U = 2
Y = 2
N = 4

# print(B.shape, P.shape, pi.shape, T(pi, 0, B, P, S).shape)

m = Machine(P,C,U,S,B,N,Y)
m.train()
# print(m.lookup)