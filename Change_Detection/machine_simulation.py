import numpy as np
from machine_replacement import Machine
import matplotlib.pyplot as plt
import scipy.stats as st

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

simulated_costs = []
actual_values = []
theta_values = np.linspace(0.1,0.9,10)

for theta in theta_values:
    u = 0.6     # probability that changed state (state 0) gives out observation 0 
    v = 0.3     # probability that unchanged state (state 1) gives out observation 0
    # theta = 0.7 # probability from unchanged to unchanged state
    B = np.array([[[u,0],[0,v]],[[1-u,0],[0,1-v]]],dtype=np.float32)
    P = np.array([[[1,0],[1 - theta,theta]],[[1,0],[1 - theta,theta]]],dtype=np.float32)
    S = 2
    A = 2
    d = 3
    epochs = 10000
    C = np.array([[[0],[1]],[[d],[0]]]) # cost matrix for action 0 and action 1 respectively

    machine = Machine(P,C,A,S,B)
    cost = 0.0
    cpi_cost = 0.0
    machine.train()

    values = []
    for e in range(epochs):
        np.random.seed(e)
        real_cur_state = 1 # start from unchanged state
        pi_0 = np.array([[0.0],[1.0]])
        estimated_cur_state = pi_0[1][0]
        tau = -1 # time when agent detects change
        tau_0 = -1 # time when environment changes state
        current_time = 0 
        while True:

            if tau != -1 and tau_0 != -1:
                # print(f"TAU: {tau} and TAU_0: {tau_0}")
                break
            action = machine.policy[int(estimated_cur_state*10)][0]
            pi = np.array([[1-estimated_cur_state],[estimated_cur_state]]) # current belief
            cpi_cost += np.matmul(machine.C[action].T,pi).item()
            
            # if change detected
            q = np.random.binomial(1,P[action][real_cur_state][0])
            if q == 1:
                real_cur_state = 0
                if tau_0 == -1:
                    tau_0 = current_time+1
            else:
                real_cur_state = 1
            
            if action == 0:
                if tau == -1:
                    tau = current_time+1
            else: # change not detected
                y = 0 # observation
                if real_cur_state == 0: # if machine was in bad state
                    y = np.random.binomial(1,1-u)
                else:
                    y = np.random.binomial(1,1-v)
                estimated_cur_state = round(T(pi,y,B,P,S,action)[1][0],1)

            current_time += 1
            
        cost += d * max(tau - tau_0,0) + 1.0 * (tau < tau_0)
        values.append(d * max(tau - tau_0,0) + 1.0 * (tau < tau_0))

    # print(machine.policy)
    # print(machine.Vn)
    # print(f"Simulated cost: {cost / epochs}")
    # print(f"Actual Value function: {machine.Vn[10]}")
    # print(cpi_cost/epochs)

    simulated_costs.append(cost / epochs)
    actual_values.append(machine.Vn[10])

    print(st.norm.interval(confidence=0.95, loc=np.mean(values), scale=st.sem(values)))

# print(f"Simulated cost: {simulated_costs}")
# print(f"Actual cost: {actual_values}")

plt.plot(theta_values,simulated_costs, label='simulated')
plt.plot(theta_values,actual_values, label='actual')
plt.ylim((0,10))
plt.xlabel('Prob from unchanged state to unchanged state')
plt.ylabel('cost')
plt.legend()
plt.show()