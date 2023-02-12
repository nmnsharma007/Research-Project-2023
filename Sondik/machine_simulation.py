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

def get_action(lookup, state, U):
    # checking action 0 lines
    mn = np.full(U, float('inf'))
    
    for a in range(len(lookup)):
        for l in lookup[a]:
            mn[a] = min(mn[a], np.matmul(l.T, state).item())

    return np.argmin(mn, axis=0)

p = 0.3     # probability that good state produces good quality product
q = 0.6     # probability that bad state produces bad quality product
theta = 0.4 # probabilirt from good to bad

B = np.array([[[1-q,0],[0,p]],[[q,0],[0,1-p]]],dtype=np.float32)
P = np.array([[[0,1],[0,1]],[[1,0],[theta,1-theta]]],dtype=np.float32)
S = 2
N = 4
U = 2   # number of actions
c = 4
R = 3
Y = 2 
epochs = 1000
C = np.array([[[R],[R]],[[c],[0]]]) # cost matrix for action 0 and action 1 respectively

# 0 -> bad state, 1 -> good state
# 0 -> replace the machine, 1 -> keep the machine
# 0 -> good quality product, 1 -> bad quality product

machine = Machine(P,C,U,S,B,N,Y)
cost = 0.0
machine.train()

for e in range(epochs):
    np.random.seed(e)
    real_cur_state = 0
    pi = np.array([[1.0],[0.0]])
    estimated_cur_state = pi

    for i in range(machine.N):
        # choose action
        action = get_action(machine.lookup[i], estimated_cur_state)
        
        cost += machine.C[action][real_cur_state][0]

        if action == 0:             # if replacement
            real_cur_state = 1
            estimated_cur_state = np.array([[0.0],[1.0]])
        else:                       # keep the machine
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

            estimated_cur_state = T(estimated_cur_state,product_quality,B,P,S)
            real_cur_state = real_next_state

print(f"Simulated cost: {cost / epochs}")
print(f"Actual Value function: {machine.Jk[1]}")

state_space = np.linspace(0,1,11)
plt.plot(state_space,machine.Jk[:,])
plt.show()
