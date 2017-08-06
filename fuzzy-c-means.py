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
escalador = StandardScaler()
X_std = escalador.fit_transform(x)
lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(X_std)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
a= pd.DataFrame(dtm_lsa, columns = ["eixo_x","eixo_y"])
fig1, axes1 = plt.subplots(1, 3, figsize=(20, 5))
dados_completos = np.vstack((a['eixo_x'], a['eixo_y']))
cores = ['#932107','#089cc1','#c26ddf','#d8fa11','#fdf08d','#d0b79b','#c2f95d','#663990','#fd37b6','#a79189','#06ef13','#606f3a','#70a083','#91501c','#de9559','#a5ae77','#38269c','#77a7c9'] 
numero_centroides = [2,3,16]
for count, ax in enumerate(axes1.reshape(-1), 1):
    fuzzy_response = fuzz.cluster.cmeans(
        dados_completos, numero_centroides[count-1], 2, error=0.005, maxiter=9999999, init=None)
    membro_do_cluster = np.argmax(fuzzy_response[1], axis=0)
    for j in range(numero_centroides[count-1]):
        ax.plot(a['eixo_x'][membro_do_cluster == j],
                a['eixo_y'][membro_do_cluster == j], '.', color=cores[j])
    for pt in fuzzy_response[0]:
        ax.plot(pt[0], pt[1], 'rs')
    ax.set_title(str(numero_centroides[count-1])+' centroides')
fig1.tight_layout()
plt.show()