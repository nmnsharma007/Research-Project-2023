import numpy as np


def sigma(pi,B,y,P,S,u=1):
    unit = np.ones((S,1))
    return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

def T(pi,y,B,P,S):
    numerator = np.matmul(B,np.matmul(P.T,pi))
    denominator = sigma(pi,B,y,P,S)
    return round(numerator / denominator,1)

pi = np.array([[0.5],[0.5]])
p = 0.3
q = 0.6
theta = 0.4
B = np.array([[[1-q,0],[0,p]],[[q,0],[0,1-p]]],dtype=np.float32)
P = np.array([[[0,1],[0,1]],[[1,0],[theta,1-theta]]],dtype=np.float32)
S = 2
N = 4

# 0 -> bad state, 1 -> good state
# 0 -> replace the machine, 1 -> keep the machine
# 0 -> good quality product, 1 -> bad quality product

machine = machine()
real_cur_state = 0
estimated_cur_state = 0
cost = 0.0
for i in range(machine.N):
    action = machine.lookup[estimated_cur_state][i]
    cost += machine.C[action][real_cur_state][0]

    # if replacement
    if action == 0:
        real_cur_state = 1
        estimated_cur_state = 0
    else: # keep the machine
        real_next_state = real_cur_state
        if real_cur_state == 0:
            pass # machine in bad state remains in bad state
        else:
            bernoulli = np.random.binomial(1,theta)
            if bernoulli == 1:
                real_next_state = 0 # if success, state changes to bad state
        
        product_quality = 0
        if real_cur_state == 0: # if machine was in bad state
            product_quality = np.random.binomial(1,1-q)
        else:
            product_quality = np.random.binomial(1,p)

    estimated_cur_state = T(pi,product_quality,B,P,S)
