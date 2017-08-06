import numpy as np, pandas as pd, os
import matplotlib
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import skfuzzy as fuzz
from sklearn import datasets
################################################################################
dataset = []

dataset_name = 'arrhythmia'

with open(dataset_name + '.data') as f:
		conteudo = f.readlines()
for x in conteudo:
	dataset.append(x.strip().split(','))
dataset = np.asarray(dataset).astype(np.float)

target = []
for x in dataset:
	target.append(x[4])
target = np.asarray(target).astype(np.int)
dataset =  np.delete(dataset, -1, axis=1)


x = pd.DataFrame(dataset, columns=['Atributo Um', 'Atributo Dois', 'Atributo Tres', 'Atributo Quatro'])
y = pd.DataFrame(target, columns=['Classe'])

scaler = StandardScaler()
X_std = scaler.fit_transform(x)
lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(X_std)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
a= pd.DataFrame(dtm_lsa, columns = ["componente_um","componente_dois"])
a['targets']=y
fig1, axes1 = plt.subplots(3, 1, figsize=(8, 8))
alldata = np.vstack((a['componente_um'], a['componente_dois']))
fpcs = []

colors = ['#932107','#089cc1','#c26ddf','#d8fa11','#fdf08d','#d0b79b','#c2f95d','#663990','#fd37b6','#a79189','#06ef13','#606f3a','#70a083','#91501c','#de9559','#a5ae77','#38269c','#77a7c9'] 
numero_centroides = [2,4,16]
for count, ax in enumerate(axes1.reshape(-1), 1):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, numero_centroides[count-1], 2, error=0.005, maxiter=9999999, init=None)
    fpcs.append(fpc)
    cluster_membership = np.argmax(u, axis=0)
    for j in range(numero_centroides[count-1]):
        ax.plot(a['componente_um'][cluster_membership == j],
                a['componente_dois'][cluster_membership == j], '.', color=colors[j])
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')
    ax.set_title(str(numero_centroides[count-1])+' centroides')

fig1.tight_layout()
plt.show()