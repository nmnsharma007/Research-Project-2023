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
    
    for a in range(U):
        for l in lookup[a]:
            mn[a] = min(mn[a], np.matmul(l.T, state).item())

    return np.argmin(mn, axis=0)

def get_value(lookup, state, U):
    # checking action 0 lines
    mn = np.full(U, float('inf'))
    
    for a in range(U):
        for l in lookup[a]:
            mn[a] = min(mn[a], np.matmul(l.T, state).item())

    return min(mn)

p = 0.3     # probability that good state produces good quality product
q = 0.6     # probability that bad state produces bad quality product
theta = 0.4 # probability from good to bad

B = np.array([[[1-q,0],[0,p]],[[q,0],[0,1-p]]],dtype=np.float32)
P = np.array([[[0,1],[0,1]],[[1,0],[theta,1-theta]]],dtype=np.float32)
S = 2
N = 4
U = 2   # number of actions
c = 3
R = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
replace_cost = 6
Y = 2 
epochs = 10
C = np.array([[[replace_cost],[replace_cost]],[[c],[0]]]) # cost matrix for action 0 and action 1 respectively

# 0 -> bad state, 1 -> good state
# 0 -> replace the machine, 1 -> keep the machine
# 0 -> good quality product, 1 -> bad quality product

machine = Machine(P,C,U,S,B,N,Y,R)
cost = 0.0
machine.train()


# print(machine.lookup[0])
plt.xlim(0,1)
# plt.ylim(0,20)
colors=['red', 'blue']
for (i, a) in enumerate(machine.lookup[0]):
    for j in range(len(a)):
        plt.axline((0,a[j][1][0]),slope=a[j][0][0] - a[j][1][0], label='action '+str(i), color=colors[i])

handles, labels = plt.gca().get_legend_handles_labels()
temp = {k:v for k,v in zip(labels, handles)}

plt.legend(temp.values(), temp.keys(), loc='best')
# plt.legend()
plt.show()

for act in range(1):
    cost = 0.0
    for e in range(epochs):
        np.random.seed(e)
        real_cur_state = 0
        estimated_cur_state = np.array([[1.0],[0.0]])

        for i in range(machine.N):
            # choose action
            action = get_action(machine.lookup[i], estimated_cur_state, machine.U)
            # action = act
            
            cost += machine.C[action][real_cur_state][0]
            # cost += np.matmul(C[action].T, estimated_cur_state).item()

            if action == 0:             # if replacement
                real_cur_state = 1
                estimated_cur_state = np.array([[0.0],[1.0]])
            else:                       # keep the machine
                if real_cur_state == 0:
                    pass # machine in bad state remains in bad state
                else:
                    bernoulli = np.random.binomial(1,theta)
                    if bernoulli == 1:
                        real_cur_state = 0 # if success, state changes to bad state
                
                product_quality = 0
                if real_cur_state == 0: # if machine was in bad state
                    product_quality = np.random.binomial(1,q)
                else:
                    product_quality = np.random.binomial(1,1-p)

                estimated_cur_state = T(estimated_cur_state,product_quality,B,P,S)

    print(f"Simulated cost: {cost / epochs}")
    print(f"Expected cost: {get_value(machine.lookup[0], np.array([[1.0],[0.0]]), machine.U)}")

    # state_space = np.linspace(0,1,11)
    # plt.plot(state_space,machine.Jk[:,])
    # plt.show()
    # '''