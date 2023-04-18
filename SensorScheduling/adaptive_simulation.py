import numpy as np
from adaptive_sampling import Machine
import matplotlib.pyplot as plt


f = np.array([[[0.3,0],[0,0]],[[0.7,0],[0,0.2]],[[0.0,0],[0,0.8]]],dtype=np.float32)
A = np.array([[1,0],[0.1,0.9]])
X = 2
U = 5
Y = 3
d = 0.0235
m = np.array([[[0],[0.1647]],[[0],[0.1647]],[[0],[0.1647]],[[0],[0.1647]],[[0],[0.1647]]])
c = np.zeros((U,2,1)) # cost matrix for action 0 and action 1 respectively
D = np.array([0,1,3,5,10])
for u in range(U):
    c[u] = np.array([[0],[1]])

machine = Machine(A,U,X,f,m,c,D,Y)
machine.train()

# print(machine.policy)

# import numpy as np
# from machine_replacement import Machine
# import matplotlib.pyplot as plt

# def sigma(pi,B,y,P,S,u=1):
#     unit = np.ones((S,1))
#     # print(np.matmul(B[y],np.matmul(P[u].T,pi))
#     return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

# def T(pi,y,B,P,S,u=1):
#     numerator = np.matmul(B[y],np.matmul(P[u].T,pi))
#     denominator = sigma(pi,B,y,P,S)
#     return numerator / denominator

# # print(pi.shape)
# # state 0 --> changed state , state 1 --> unchanged state
# # action 0 --> stop, action 1 --> continue
# u = 0.6     # probability that changed state gives out observation 1 
# v = 0.3     # probability that unchanged state gives out observation 1
# theta = 0.4 # probability from unchanged to unchanged state
# B = np.array([[[u,0],[0,v]],[[1-u,0],[0,1-v]]],dtype=np.float32)
# P = np.array([[[1,0],[1 - theta,theta]],[[1,0],[1 - theta,theta]]],dtype=np.float32)
# S = 2
# A = 2
# d = 3
# # epochs = 1000
# C = np.array([[[0],[1]],[[d],[0]]]) # cost matrix for action 0 and action 1 respectively

# machine = Machine(P,C,A,S,B)
# cost = 0.0
# # print(machine.lookup.shape)
# machine.train()
# print(machine.Vn)
print(machine.policy)
x = np.linspace(0.0,1.0,1001)
y = machine.policy[:,0]
y = y.astype(np.int32)
plt.plot(x,y)
plt.xlabel('belief state')
plt.ylabel('action')
plt.show()

# # print(machine.lookup)
# # for e in range(epochs):
# #     np.random.seed(e)
# #     real_cur_state = 0
# #     pi = np.array([[0.0],[1.0]])
# #     estimated_cur_state = pi[0][0]
# #     for i in range(machine.N):
# #         action = machine.lookup[int(estimated_cur_state*10)][i]
# #         # print(action)
# #         cost += machine.C[action][real_cur_state][0]
# #         # print(f"Current estimated state: {estimated_cur_state}")
# #         # if replacement
# #         if action == 0:
# #             real_cur_state = 1
# #             estimated_cur_state = 0
# #         else: # keep the machine
# #             if real_cur_state == 0:
# #                 pass # machine in bad state remains in bad state
# #             else:
# #                 bernoulli = np.random.binomial(1,theta)
# #                 if bernoulli == 1:
# #                     real_cur_state = 0 # if success, state changes to bad state
            
# #             product_quality = 0
# #             if real_cur_state == 0: # if machine was in bad state
# #                 product_quality = np.random.binomial(1,q)
# #             else:
# #                 product_quality = np.random.binomial(1,1-p)

# #             pi = np.array([[estimated_cur_state],[1-estimated_cur_state]])
# #             estimated_cur_state = round(T(pi,product_quality,B,P,S)[0][0],1)
# #         # print(T(pi,product_quality,B,P,S).shape)

# # print(f"Simulated cost: {cost / epochs}")
# # print(f"Actual Value function: {machine.Jk[1]}")

# # state_space = np.linspace(0,1,11)
# # plt.plot(state_space,machine.Jk[:,])
# # plt.show()