
# simulation of machine replacement problem (FULL MDP)

import numpy as np
import random
from machine_replacement import Machine
import matplotlib.pyplot as plt


A = 2   # action 0(keep) and action 1(replace)
S = 2   # state 0(operational) and state 1(faulty)
N = 4

p = 0.1 # probability that machine will go to faulty state
c = 4
R = 3

C = np.array([[[0],[c]],[[R],[R]]]) # cost matrix for action 0 and action 1 respectively
P = [[[1-p, p], [0, 1]], [[1, 0], [1, 0]]] # transition probability matrix for each action

# Initializing the machine with above parameters
machine = Machine(P, C, A, S, N)
machine.train() # training the machine using backward DP

initial_state = 0

epochs = 50
total_cost = [] # total cost for each epoch
avg_cost = 0    #acg cost of all the epochs

print(machine.lookup)

for k in range(epochs):
    cost = 0
    np.random.seed(k)

    current_state = initial_state
    for i in range(machine.N):
        action = machine.lookup[current_state][i]
        cost += machine.C[action][current_state][0]
        
        q = np.random.random()

        prob_sum = 0
        next_state = None
        for j in range(len(P[action][current_state])):
            prob_sum += P[action][current_state][j]
            if q <= prob_sum:
                next_state = j
                break
        
        current_state = next_state

    total_cost.append(cost)
    avg_cost += cost/epochs

print('Avg Cost: ', avg_cost)


# Plotting (expected total cost with starting state as 0) vs epochs
y1 = total_cost
y2 = np.full(epochs, machine.Jk[initial_state], dtype=int)
x = np.linspace(0,epochs-1,epochs,dtype=int)

plt.plot(x, y1, label='simulated')
plt.plot(x, y2, label='expected')

plt.xlabel('epoch')
plt.ylabel('cost')

plt.legend()
plt.show()
