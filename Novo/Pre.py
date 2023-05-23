import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import pdist, squareform
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
base = pd.read_csv('Datasets/strokeMod.csv', na_values=["Unknown", "N/A"])
#Retirada manual de registro unico com sexo marcado como 'outro'

# print(np.unique(base['stroke'], return_counts=True))


# label_values, label_count = np.unique(base['stroke'],return_counts=True)
# # Calcula a contagem de valores ausentes por instância
# count_missing = base.isnull().sum(axis=1)
# # Imprime a quantidade de instâncias com dados ausentes
# print("Quantidade de instâncias com dados ausentes:", count_missing.sum())


# Funcao que mostra a quantidade de instancias com atributos ausentes:

def remove_instancias_semelhantes(dataset):
    # Calcula a matriz de distâncias entre as instâncias
    dist_matrix = pdist(dataset.values, metric='euclidean')
    # Converte a matriz de distâncias em uma matriz quadrada
    dist_matrix_square = squareform(dist_matrix)
    # Define um limite de distância para considerar instâncias como semelhantes
    limite_distancia = 0.9  # Ajuste o valor conforme necessário
    # Obtém os índices das instâncias semelhantes
    indices_semelhantes = []
    n_instancias = len(dataset)
    for i in range(n_instancias):
        for j in range(i+1, n_instancias):
            if dist_matrix_square[i, j] <= limite_distancia:
                indices_semelhantes.append(j)
    # Remove as instâncias semelhantes do dataset
    dataset_filtrado = dataset.drop(indices_semelhantes)
    # Obtém a quantidade de instâncias removidas
    quantidade_removidas = len(indices_semelhantes)
    return dataset_filtrado, quantidade_removidas


def check_missing_attributes(df):
    # Contagem de instâncias sem "smoking_status" e "bmi"
    count_both_missing = df[df['smoking_status'].isnull() & df['bmi'].isnull()].shape[0]
    # Contagem de instâncias sem "smoking_status"
    count_smoking_status_missing = df[df['smoking_status'].isnull()].shape[0]
    # Contagem de instâncias sem "bmi"
    count_bmi_missing = df[df['bmi'].isnull()].shape[0]
    print("Instâncias que não possuem smoking_status e bmi:", count_both_missing)
    print("Instâncias que não possuem smoking_status:", count_smoking_status_missing)
    print("Instâncias que não possuem bmi:", count_bmi_missing)
    print("Instâncias totais: ",len(df))
    print("Proporção de instâncias positivas/negativas:")
    pos, total = np.unique(df['stroke'], return_counts=True)
    print(total)

def identificar_colunas_correlacionadas(dataset, limite_correlacao):
    # Obtém todas as colunas, exceto a última (atributo de classe)
    colunas_entrada = dataset.columns[:-1]

    # Calcula a matriz de correlação
    matriz_correlacao = dataset[colunas_entrada].corr()

    # Obtém as colunas com alta correlação em relação ao atributo de classificação
    colunas_correlacionadas = matriz_correlacao.iloc[-1][abs(matriz_correlacao.iloc[-1]) > limite_correlacao].index.tolist()

    return colunas_correlacionadas

def identify_redundant_features(df):
    target_column = df.columns[-1]  # Última coluna é o atributo de classe

    # Calcula a matriz de correlação
    corr_matrix = df.corr()

    # Filtra as features com correlação maior que um limite pré-definido
    correlation_threshold = 0.8  # Define o limite de correlação (ajuste conforme necessário)
    redundant_features = []
    for feature in df.columns[:-1]:  # Exclui a coluna do atributo de classe
        if abs(corr_matrix.loc[feature, target_column]) > correlation_threshold:
            redundant_features.append(feature)

    return redundant_features

# check_missing_attributes(base) #chamando a funcao

# print("Antes de tratar os dados ausentes:")
# check_missing_attributes(base)


# Tratando dados ausentes na coluna 'bmi', utilziando o simple imputer.
imputer = SimpleImputer(strategy='mean')
base['bmi'] = imputer.fit_transform(base[['bmi']])
# check_missing_attributes(base) #Comparar antes e depois ?

#Deletando instancias que não possuem informação sobre 'smoking_status'.
base = base.dropna(subset=['smoking_status'])

# print("Depois de tratar os dados ausentes:")
# check_missing_attributes(base)


colunas_categoricas = base[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
column_trans = make_column_transformer((OneHotEncoder(), ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']), remainder='passthrough')
df_codificado = column_trans.fit_transform(base)

# print(df_codificado)

colunas_transformadas = column_trans.get_feature_names_out()
# print(colunas_transformadas)

# Criar um novo DataFrame com o DataFrame codificado e os nomes das colunas
df_codificado_com_nomes = pd.DataFrame(df_codificado, columns=colunas_transformadas)

# Imprimir o DataFrame codificado com os nomes das colunas
# print(df_codificado_com_nomes)

df_novo = pd.DataFrame(df_codificado) #Convertendo a tabela codificada em um data frame.

colunas = df_codificado_com_nomes.columns.tolist()
df_novo, quantidade_removidas = remove_instancias_semelhantes(df_novo)
print("Quantidade de instâncias removidas:", quantidade_removidas)

print(identificar_colunas_correlacionadas(df_novo,0.8))
print(identify_redundant_features(df_novo))
print(len(df_novo.columns))

# Exibe o dataset filtrado
# print(dataset_filtrado)

x_base = df_novo.iloc[:, 0:19].values # X_base contem os atributos normais da base de dados.
y_base = df_novo.iloc[:, 19].values # Y_base contem os labels

# print(np.unique(y_base,return_counts=True))
# print(np.unique(y_base,return_counts=True)) # dados balanceados


# print(np.unique(y_base, return_counts=True))

# x_train, x_test, y_train, y_test = train_test_split(x_base, y_base)
# import pickle
# with open('raw.pkl', 'wb') as f:
#   pickle.dump([x_train,x_test,y_train,y_test], f)

# def reportSample(x_resampled,y_resampled,name):
#     print(name)
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.metrics import classification_report, fbeta_score,roc_auc_score
#     rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#     rf_classifier.fit(x_resampled,y_resampled)
#     from sklearn.metrics import accuracy_score
#     previsoes = rf_classifier.predict(x_test)
#     report = classification_report(y_test, previsoes)
#     probabilidades = rf_classifier.predict_proba(x_test)[:, 1]
#     auc = roc_auc_score(y_test, probabilidades)
#     print(report)
#     print("AUC = ",auc)

#RandomOverSampler
# from imblearn.over_sampling import RandomOverSampler
# over_sampler = RandomOverSampler(sampling_strategy=0.5)
# x_resampled, y_resampled = over_sampler.fit_resample(x_train, y_train)
# reportSample(x_resampled,y_resampled,"Random over sampler")



# NearMiss
# from imblearn.under_sampling import NearMiss
# nearmiss = NearMiss(version=2)
# print("Antes do resample:")
# print(np.unique(y_train,return_counts=True))
# x_resampled, y_resampled = nearmiss.fit_resample(x_train, y_train)
# print("Depois do resample:")
# print(np.unique(y_resampled,return_counts=True))
# reportSample(x_resampled,y_resampled,"NearMiss underSample")


# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# x_resampled,y_resampled = sm.fit_resample(x_train,y_train)
# reportSample(x_resampled,y_resampled,"Smote over sampling")


# import pickle
# with open('base.pkl', 'wb') as f:
#   pickle.dump([x_resampled,x_test,y_resampled,y_test], f)




# with open('comOverSmoteTeste.pkl', 'wb') as f:
#   pickle.dump([test_features, test_target], f)
# import pickle
# with open('comUnder.pkl', 'wb') as f:
#   pickle.dump([X_treino, X_teste, y_treino, y_teste], f)




