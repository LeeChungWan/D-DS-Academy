import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import elice_utils
from scipy.spatial.distance import cosine

from sklearn.decomposition import randomized_svd

A=[[1,0,1,0,0],
   [1,1,0,0,0],
   [0,1,0,0,0],
   [0,1,1,0,0],
   [0,0,0,1,0],
   [0,0,1,1,0],
   [0,0,0,1,0],
   [0,0,0,1,1]]

name_word=["romeo","juliet","happy","dagger","live","die","free","new-hampshire"]
name_doc=["D1", "D2", "D3", "D4", "D5"]

A=np.asarray(A)

#Computing SVD
U, Sigma, VT = randomized_svd(A, n_components=2, n_iter=1, random_state=None)

#Sigma is a diagonal matrix, but randomized_svd returns a vector so we have to change it back to a diagonal matrix
Sigma=np.diag(Sigma)

#Compute the vectors for each word and each document
Word=np.dot(U,Sigma)
Doc=np.dot(Sigma,VT)

#Scatter plot the resulting vectors
fig, ax = plt.subplots()
ax.scatter(Word.T[0], Word.T[1])
ax.scatter(Doc[0], Doc[1])


for i, txt in enumerate(name_word):
    if(txt=="free"):
        ax.annotate(txt, (Word.T[0][i]-0.06, Word.T[1][i]+0.01))
    else:
        ax.annotate(txt, (Word.T[0][i]+0.01, Word.T[1][i]+0.01))

for i, txt in enumerate(name_doc):
        ax.annotate(txt, (Doc[0][i]+0.01, Doc[1][i]+0.01))

plt.savefig('demo.png')
elice_utils.send_image('demo.png')

#Compute the vector for the query
query=(Word[3]+Word[5])/2


#Compute the cosine distance between that query and each document
results=[]

for i in range(len(Doc.T)):
    results.append((name_doc[i], cosine(Doc.T[i], query)))


#Sort the results in descending order
results.sort(key=lambda X: X[1])
print("Closest documents to the given query are: ")
print("Results = ", results)
