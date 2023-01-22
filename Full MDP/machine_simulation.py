# simulation of machine replacement problem (FULL MDP)
import numpy as np
import random
from machine_replacement import Machine
import matplotlib.pyplot as plt


A = 2   # action 0(keep) and action 1(replace)
S = 2   # state 0(operational) and state 1(faulty)

p = 0.3 # probability that machine will go to faulty state
C = [[0, 10], [20, 20]]
P = [[[1-p, p], [0, 1]], [[1, 0], [1, 0]]]

machine = Machine(P, C, A, S)
machine.train()

initial_state = 0
current_state = initial_state

total_cost = []
avg_cost = 0

iterations = 30
for k in range(iterations):
    cost = 0
    for i in range(machine.N):
        action = machine.lookup[current_state][i]
        cost += C[action][current_state]
        
        q = random.random()

        prob_sum = 0
        for j in len(P[action][current_state]):
            prob_sum += P[action][current_state][j]
            if q < prob_sum:
                next_state = j
                break

    total_cost.append(cost)
    avg_cost = cost/machine.N


x1 = total_cost
x2 = np.full((1,30), avg_cost, dtype=int)
y = np.linspace(0,29,30,dtype=int)

plt.plot(x1, y, label='simulated')
plt.plot(x2, y, label='expected')

plt.xlabel('epoch')
plt.ylabel('cost')

plt.legeng()
plt.show()







    



