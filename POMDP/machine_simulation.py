import numpy as np


def sigma(pi,B,y,P,S,u=1):
    unit = np.ones((S,1))
    return np.matmul(unit.T,np.matmul(B[y],np.matmul(P[u].T,pi)))[0][0]

def T(pi,y,B,P,S):
    numerator = np.matmul(B,np.matmul(P.T,pi))
    denominator = sigma(pi,B,y,P,S)
    return numerator / denominator

pi = np.array([[0.5],[0.5]])
p = 0.3
q = 0.6
theta = 0.4
B = np.array([[[1-q,0.0],[0.0,p]],[[q,0.0],[0.0,1-p]]])
P = np.array([[[0,1],[0,1]],[[1,0],[theta,1-theta]]],dtype=np.float32)
S = 2
