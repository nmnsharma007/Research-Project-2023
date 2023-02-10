import numpy as np
from machine_replacement import Machine
import matplotlib.pyplot as plt

def sigma(pi,B,y,P,S,u=1):
    unit = np.ones((S,1))
    # print(np.matmul(B[y],np.matmul(P[u].T,pi))
    return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

def T(pi,y,B,P,S,u=1):
    numerator = np.matmul(B[y],np.matmul(P[u].T,pi))
    denominator = sigma(pi,B,y,P,S)
    return numerator / denominator

# print(pi.shape)
p = 0.3
q = 0.6
theta = 0.4
B = np.array([[[1-q,0],[0,p]],[[q,0],[0,1-p]]],dtype=np.float32)
P = np.array([[[0,1],[0,1]],[[1,0],[theta,1-theta]]],dtype=np.float32)
S = 2
N = 4
A = 2
c = 4
R = 3
epochs = 1000
C = np.array([[[R],[R]],[[c],[0]]]) # cost matrix for action 0 and action 1 respectively
# 0 -> bad state, 1 -> good state
# 0 -> replace the machine, 1 -> keep the machine
# 0 -> good quality product, 1 -> bad quality product

machine = Machine(P,C,A,S,B,N)
cost = 0.0
# print(machine.lookup.shape)
machine.train()
print(machine.lookup)
for e in range(epochs):
    np.random.seed(e)
    real_cur_state = 0
    pi = np.array([[1.0],[0.0]])
    estimated_cur_state = pi[0][0]
    for i in range(machine.N):
        action = machine.lookup[int(estimated_cur_state*10)][i]
        # print(action)
        cost += machine.C[action][real_cur_state][0]
        # print(f"Current estimated state: {estimated_cur_state}")
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
                product_quality = np.random.binomial(1,q)
            else:
                product_quality = np.random.binomial(1,1-p)

            pi = np.array([[estimated_cur_state],[1-estimated_cur_state]])
            estimated_cur_state = round(T(pi,product_quality,B,P,S)[0][0],1)
        # print(T(pi,product_quality,B,P,S).shape)

print(f"Simulated cost: {cost / epochs}")
print(f"Actual Value function: {machine.Jk[1]}")

# state_space = np.linspace(0,1,11)
# plt.plot(state_space,machine.Jk[:,])
# plt.show()