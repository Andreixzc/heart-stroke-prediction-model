import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
import graphviz
base = pd.read_csv('Presos_MissForestv2.csv', sep=';')

coluna_129 = base.iloc[:, 129]  # Selecionar a coluna pelo índice

print(coluna_129)


x_base = base.iloc[:, 0:129].values # X_base contem os atributos normais da base de dados.
y_base = base.iloc[:, 129].values # Y_base contem os atributos de classe.

# print(x_base)


from sklearn.model_selection import train_test_split
#Chamamos a funcao train_test_split passando a base de dados sem o atributo de classe, e o atributo de classe
#Essa funcao retorna os dados dividos para treino e para o teste.(A proporção é definida pelo parametro.)
X_treino, X_teste, y_treino, y_teste = train_test_split(x_base, y_base, test_size = 0.20, random_state = 0)
print(X_treino[0])
#Salvando arquivos separados em um arquivo pkl.
import pickle
with open('presos.pkl', 'wb') as f:
  pickle.dump([X_treino, X_teste, y_treino, y_teste], f)



