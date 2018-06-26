import numpy as np
from sklearn.utils.extmath import randomized_svd

M=[[1,0,1,0,0],
   [1,1,0,0,0],
   [0,1,0,0,0],
   [0,1,1,0,0],
   [0,0,0,1,0],
   [0,0,1,1,0],
   [0,0,0,1,0],
   [0,0,0,1,1]]

M=np.asarray(M)
U, Sigma, VT = randomized_svd(M, n_components=5, n_iter=1, random_state=None)
print(U)
print(Sigma)
print(VT)
