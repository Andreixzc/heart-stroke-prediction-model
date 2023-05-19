import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

base = pd.read_csv('Datasets/healthcare-dataset-stroke-data2.csv', na_values=["Unknown", "N/A"])

# Remover coluna redundante (ID) da base
# base = base.drop('id', axis=1)

# Verificar os valores únicos e as contagens da coluna 'stroke'
unique_values, value_counts = np.unique(base['stroke'], return_counts=True)
print(unique_values)  # Printa os valores que a coluna 'stroke' pode ter.
print(value_counts)  # Printa a quantidade de cada valor de cada, da coluna 'stroke'.

sns.countplot(x=base['stroke'])  # Plota o gráfico mostrando a distribuição de registros com stroke = 0 e stroke = 1;
plt.savefig('figuras/proporcaoSimNao.png')  # Salva o gráfico.

x_base = base.iloc[:, 0:10].values  # x_base contém os atributos normais da base de dados.
y_base = base.iloc[:, 10].values  # y_base contém os atributos de classe.

# Transformação de variáveis utilizando o LabelEncoder.
label_encoder_gender = LabelEncoder()
label_encoder_married = LabelEncoder()
label_encoder_residence = LabelEncoder()
x_base[:, 0] = label_encoder_gender.fit_transform(x_base[:, 0])
x_base[:, 4] = label_encoder_married.fit_transform(x_base[:, 4])
x_base[:, 5] = label_encoder_residence.fit_transform(x_base[:, 5])
x_base[:, 6] = label_encoder_residence.fit_transform(x_base[:, 6])
x_base[:, 9] = label_encoder_residence.fit_transform(x_base[:, 9])

# Inputando dados ausentes
imputer = SimpleImputer(strategy='mean')  # ou strategy='median' ou strategy='most_frequent'
x_base = imputer.fit_transform(x_base)

# Normalizar as variáveis numéricas
scaler = MinMaxScaler()
x_base[:, [1, 8, 9]] = scaler.fit_transform(x_base[:, [1, 8, 9]])

# Verificando atributos com alta correlação.
from scipy.stats import pearsonr

for i in range(7):
    for j in range(7):
        corr, _ = pearsonr(x_base[:, i], x_base[:, j])
        if corr >= 0.6 and i != j:
            print("flag:", i, j)  # Printa as colunas que possuem alta correlação.

# Balanceia a base de dados duplicando aleatoriamente as instâncias minoritárias da base de dados
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
x_base, y_base = ros.fit_resample(x_base, y_base)

sns.countplot(x=y_base)
plt.savefig('figuras/proporcaoSimNaoBalanceada.png')  # Salva o gráfico.

# Dividir os dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(x_base, y_base, test_size=0.20, random_state=0)

# Salvando arquivos separados em um arquivo pkl.
with open('BasePreProcessada/breast_rede.pkl', 'wb') as f:
    pickle.dump([X_treino, X_teste, y_treino, y_teste], f)
