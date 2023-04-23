import numpy as np
import random


def T(P,X,Y,U,B,c,a,pi):
    R_a_pi = np.zeros((X,X))
    # compute the social learning filter
    for i in range(X):
        curr_element = 0.0
        for y in range(Y):
            coefficient = 1 # assume initial value of 1
            lhs = np.matmul(c[a].T,np.matmul(B[y],np.matmul(P.T,pi))).item()
            for u in range(U):
                if u != a:
                    rhs = np.matmul(c[u].T,np.matmul(B[y],np.matmul(P.T,pi))).item()
                    coefficient *= 1.0 * (lhs <= rhs)
            curr_element += coefficient * B[y][i][i]
        # print(R_a_pi.shape,  curr_element)
        R_a_pi[i][i] = curr_element
    numerator = np.matmul(R_a_pi,np.matmul(P.T,pi))
    unit = np.ones((X,1))
    denominator = np.matmul(unit.T,np.matmul(R_a_pi,np.matmul(P.T,pi))).item()
    return numerator / denominator

            
    # numerator = 

class SLAgent():
    def __init__(self,P,U,S,B,C,Y):
        self.P = P                                      # probability transition matrix
        self.C = C
        self.U = U                                      # number of actions
        self.S = S                                      # number of states
        self.B = B                                      # observation probability matrix
        self.Y = Y                                      # number of observation states
    

    def start(self):
        pi_k = np.array([[0.3],[0.7]])
        true_state = 0
        while True:
            print(1)
            # step 1: private observation
            r = random.random()
            yk = 0
            prob = 0
            for (i,b) in enumerate(self.B):
                prob += b[true_state][true_state]
                if r <= prob:
                    yk = i
                    break

            # step 2: private belief
            numerator = np.matmul(self.B[yk], np.matmul(self.P.T, pi_k))
            n_k = numerator / np.matmul(np.ones((self.S,1)).T, numerator).item()
            
            # step 3: myopic action
            ak = np.argmin(np.array([np.matmul(self.c[a].T, n_k).T] for a in range(self.U)))

            # ste 4: social learning filter
            pi_k1 = T(self.P,self.S,self.Y,self.U,self.B,self.C,ak,pi_k)

            max_diff = 0
            for i in range(self.S):
                max_diff = max(max_diff,abs(pi_k[i][0] - pi_k1[i][0]))
            
            if max_diff <= 0.00001:
                print(pi_k)
                break

            pi_k = np.copy(pi_k1)




P = np.array([[1,0],[0,1]])
# A = np.array([[0.9,0.4],[0.1,0.6]])
S = 2
U = 3
Y = 2
p = 0.3
q = 0.6
B = np.array([[[1-q,0],[0,p]],[[q,0],[0,1-p]]],dtype=np.float32)
C = np.array([[[2],[1]],[[1],[2]],[[1],[1]]]) 

machine = SLAgent(P,U,S,B,C,Y)
machine.start()