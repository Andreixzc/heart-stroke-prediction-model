import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
# import graphviz
base = pd.read_csv('Datasets/healthcare-dataset-stroke-data2.csv', na_values=["Unknown", "N/A"])
# base = base.drop('id',axis=1) #Removendo coluna redundante (ID) da base.
# print(base)

np.unique(base['stroke'], return_counts=True)
unique_values, value_counts = np.unique(base['stroke'], return_counts=True)
print(unique_values) # Printa os valores que a colua 'stroke' pode ter.
print(value_counts) # Printa a quantidade de cada valor de cada, da coluna 'stroke';
sns.countplot(x = base['stroke']) # Plota o gráfico mostrando a distribuição de registros com stroke = 0 e stroke = 1;
plt.savefig('figuras/proporcaoSimNao.png') # Salva o grafico.

x_base = base.iloc[:, 0:10].values # X_base contem os atributos normais da base de dados.
y_base = base.iloc[:, 10].values # Y_base contem os atributos de classe.



#Transformação de variaveis utilizando o labelEnconder.
from sklearn.preprocessing import LabelEncoder
label_encoder_gender = LabelEncoder()
label_encoder_married = LabelEncoder()
label_encoder_residence = LabelEncoder()
x_base[:, 0] = label_encoder_gender.fit_transform(x_base[:, 0])
x_base[:, 4] = label_encoder_married.fit_transform(x_base[:, 4])
x_base[:, 5] = label_encoder_residence.fit_transform(x_base[:, 5])
x_base[:, 6] = label_encoder_residence.fit_transform(x_base[:, 6])
x_base[:, 9] = label_encoder_residence.fit_transform(x_base[:, 9])


#Inputando dados ausentes
imputer = SimpleImputer(strategy='mean')  # ou strategy='median' ou strategy='most_frequent'
x_base = imputer.fit_transform(x_base)

#Verificando atributos com alta correlação.
from scipy.stats import pearsonr
for i in range(7): 
  for j in range(7): 
    corr, _ = pearsonr(x_base[:, i], x_base[:, j])
    if corr >= 0.6 and i != j:
      print("flag:", i, j) #Printa as colunas que possuirem alta correlação.

# print(x_base[0])
# print(y_base[1])

#Balanceia a base de dados duplicando aleatoriamete as instancias minoritarias da base de dados
# from imblearn.over_sampling import RandomOverSampler #Não pode fazer o balanceamento no conjunto inteiro de dados.
# ros = RandomOverSampler(random_state=0)
# x_base, y_base = ros.fit_resample(x_base, y_base)
# np.unique(y_base, return_counts=True)
# sns.countplot(x = y_base)
# plt.savefig('figuras/proporcaoSimNaoBalanceada.png') # Salva o grafico.


from sklearn.model_selection import train_test_split
#Chamamos a funcao train_test_split passando a base de dados sem o atributo de classe, e o atributo de classe
#Essa funcao retorna os dados dividos para treino e para o teste.(A proporção é definida pelo parametro.)
X_treino, X_teste, y_treino, y_teste = train_test_split(x_base, y_base, test_size = 0.20, random_state = 0)
print("len do treino antes do resample:")
print(np.unique(y_treino, return_counts=True))


from imblearn.over_sampling import RandomOverSampler #Não pode fazer o balanceamento no conjunto inteiro de dados.
ros = RandomOverSampler(random_state=0)
X_treino, y_treino = ros.fit_resample(x_base, y_base)
np.unique(y_base, return_counts=True)
sns.countplot(x = y_base)

plt.savefig('figuras/proporcaoSimNaoBalanceada.png') # Salva o grafico.

#Salvando arquivos separados em um arquivo pkl.
import pickle
with open('breast.pkl', 'wb') as f:
  pickle.dump([X_treino, X_teste, y_treino, y_teste], f)

print("len do treino depois do resample:")
print(np.unique(y_treino, return_counts=True))

print("len do teste")
print(np.unique(y_teste, return_counts=True))


