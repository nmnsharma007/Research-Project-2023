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
# state 0 --> changed state , state 1 --> unchanged state
# action 0 --> stop, action 1 --> continue
u = 0.6     # probability that changed state gives out observation 1 
v = 0.3     # probability that unchanged state gives out observation 1
theta = 0.4 # probability from unchanged to unchanged state
B = np.array([[[u,0],[0,v]],[[1-u,0],[0,1-v]]],dtype=np.float32)
P = np.array([[[1,0],[1 - theta,theta]],[[1,0],[1 - theta,theta]]],dtype=np.float32)
S = 2
A = 2
d = 3
# epochs = 1000
C = np.array([[[0],[1]],[[d],[0]]]) # cost matrix for action 0 and action 1 respectively

machine = Machine(P,C,A,S,B)
cost = 0.0
# print(machine.lookup.shape)
machine.train()
print(machine.Vn)
print(machine.policy)

# print(machine.lookup)
# for e in range(epochs):
#     np.random.seed(e)
#     real_cur_state = 0
#     pi = np.array([[0.0],[1.0]])
#     estimated_cur_state = pi[0][0]
#     for i in range(machine.N):
#         action = machine.lookup[int(estimated_cur_state*10)][i]
#         # print(action)
#         cost += machine.C[action][real_cur_state][0]
#         # print(f"Current estimated state: {estimated_cur_state}")
#         # if replacement
#         if action == 0:
#             real_cur_state = 1
#             estimated_cur_state = 0
#         else: # keep the machine
#             if real_cur_state == 0:
#                 pass # machine in bad state remains in bad state
#             else:
#                 bernoulli = np.random.binomial(1,theta)
#                 if bernoulli == 1:
#                     real_cur_state = 0 # if success, state changes to bad state
            
#             product_quality = 0
#             if real_cur_state == 0: # if machine was in bad state
#                 product_quality = np.random.binomial(1,q)
#             else:
#                 product_quality = np.random.binomial(1,1-p)

#             pi = np.array([[estimated_cur_state],[1-estimated_cur_state]])
#             estimated_cur_state = round(T(pi,product_quality,B,P,S)[0][0],1)
#         # print(T(pi,product_quality,B,P,S).shape)

# print(f"Simulated cost: {cost / epochs}")
# print(f"Actual Value function: {machine.Jk[1]}")

# state_space = np.linspace(0,1,11)
# plt.plot(state_space,machine.Jk[:,])
# plt.show()