import numpy as np
from machine_replacement import Machine
from multiple_policy import ValueMachine
import matplotlib.pyplot as plt

def sigma(pi,B,y,P,S,u):
    unit = np.ones((S,1))
    # print(np.matmul(B[y],np.matmul(P[u].T,pi))
    return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

def T(pi,y,B,P,S,u):
    numerator = np.matmul(B[y],np.matmul(P[u].T,pi))
    denominator = sigma(pi,B,y,P,S,u)
    return numerator / denominator

# print(pi.shape)
# state 0 --> changed state , state 1 --> unchanged state
# action 0 --> stop, action 1 --> continue
u = 0.6     # probability that changed state (state 0) gives out observation 0 
v = 0.3     # probability that unchanged state (state 1) gives out observation 0
theta = 0.6 # probability from unchanged to unchanged state
B = np.array([[[u,0],[0,v]],[[1-u,0],[0,1-v]]],dtype=np.float32)
P = np.array([[[1,0],[1 - theta,theta]],[[1,0],[1 - theta,theta]]],dtype=np.float32)
S = 2
A = 2
d = 3
epochs = 1000
C = np.array([[[0],[1]],[[d],[0]]]) # cost matrix for action 0 and action 1 respectively

machine = Machine(P,C,A,S,B)
# machine.train()
mu_values = np.linspace(0.1,1,10)
simulated_costs = []
print(mu_values)

analytical_costs = []

for mu in mu_values:
    # analytical costs
    vMachine = ValueMachine(P,C,A,S,B,mu)
    vMachine.train()
    analytical_costs.append(vMachine.Vn[10])

    # simulated costs
    cost = 0.0
    for e in range(epochs):
        np.random.seed(e)
        real_cur_state = 1 # start from unchanged state
        pi_0 = np.array([[0.0],[1.0]])
        estimated_cur_state = pi_0[1][0]
        tau = -1 # time when agent detects change
        tau_0 = -1 # time when environment changes state
        current_time = 0
        # print("HERE")
        while True:

            if tau != -1 and tau_0 != -1:
                # print(f"TAU: {tau} and TAU_0: {tau_0}")
                break
            action = 1 if estimated_cur_state >= mu else 0
            pi = np.array([[1-estimated_cur_state],[estimated_cur_state]]) # current belief
            # cost += np.matmul(machine.C[action].T,pi).item()
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
    # print(f"Cost: {cost / epochs}")
    simulated_costs.append(cost / epochs)

print(f"Simulated cost: {simulated_costs}")
print(f"Analytical cost: {analytical_costs}")
plt.plot(mu_values,simulated_costs, label='simulated')
plt.plot(mu_values,analytical_costs, label='analytical')
plt.legend()
plt.show()
