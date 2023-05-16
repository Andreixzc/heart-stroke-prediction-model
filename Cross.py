from sklearn import metrics
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
from scipy import stats
import copy
import numpy as np
import pickle
import scipy.stats as st

with open('ModelosTreinados/rede_neural.pkl', 'rb') as f:
    rede_neural = pickle.load(f)

with open('ModelosTreinados/randomForest.pkl', 'rb') as f:
    random_forest = pickle.load(f)

with open('ModelosTreinados/arvore.pkl', 'rb') as f:
    arvore_carregada = pickle.load(f)
    
with open('BasePreProcessada/breast.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))

# Cross-validate e calcular métricas para a rede neural
k = 5 #Numero de folds
metrics_rede_neural = cross_validate(rede_neural, X_treino, y_treino, cv=k, scoring=['precision', 'recall', 'f1'])

precision_rede_neural = metrics_rede_neural['test_precision']
recall_rede_neural = metrics_rede_neural['test_recall']
f1_rede_neural = metrics_rede_neural['test_f1']

# Cross-validate e calcular métricas para o random forest
metrics_random_forest = cross_validate(random_forest, X_treino, y_treino, cv=k, scoring=['precision', 'recall', 'f1'])

precision_random_forest = metrics_random_forest['test_precision']
recall_random_forest = metrics_random_forest['test_recall']
f1_random_forest = metrics_random_forest['test_f1']

# Cross-validate e calcular métricas para a árvore de decisão
metrics_arvore_decisao = cross_validate(arvore_carregada, X_treino, y_treino, cv=k, scoring=['precision', 'recall', 'f1'])

precision_arvore_decisao = metrics_arvore_decisao['test_precision']
recall_arvore_decisao = metrics_arvore_decisao['test_recall']
f1_arvore_decisao = metrics_arvore_decisao['test_f1']

# Imprimir os resultados
print("Resultados para a rede neural:")
print("Precision:", precision_rede_neural)
print("Recall:", recall_rede_neural)
print("F1-Score:", f1_rede_neural)

print("Resultados para o random forest:")
print("Precision:", precision_random_forest)
print("Recall:", recall_random_forest)
print("F1-Score:", f1_random_forest)

print("Resultados para a árvore de decisão:")
print("Precision:", precision_arvore_decisao)
print("Recall:", recall_arvore_decisao)
print("F1-Score:", f1_arvore_decisao)

#Media:
print("Médias rede neural:")
print("Média de precision para a rede neural:",np.mean(precision_rede_neural))
print("Média de recall para a rede neural:",np.mean(recall_rede_neural))
print("Média de F1-score para a rede neural:",np.mean(f1_rede_neural))

print("Médias random forest:")
print("Média de precision para a random forest:",np.mean(precision_random_forest))
print("Média de recall para a random forest:",np.mean(recall_random_forest))
print("Média de F1-score para a random forest:",np.mean(f1_random_forest))

print("Médias decision Tree:")
print("Média de precision para a decision Tree:",np.mean(precision_arvore_decisao))
print("Média de recall para a decision Tree:",np.mean(recall_arvore_decisao))
print("Média de F1-score para a decision Tree:",np.mean(f1_arvore_decisao))


#Calcular intervalos de confiança
print("Intervalo de confiança para a rede neural:")
print("Intervalo de confiança para Precision:", interval_confidence(precision_rede_neural))
print("Intervalo de confiança para Recall:", interval_confidence(recall_rede_neural))
print("Intervalo de confiança para F1-Score:", interval_confidence(f1_rede_neural))

print("Intervalo de confiança para o random forest:")
print("Intervalo de confiança para Precision:", interval_confidence(precision_random_forest))
print("Intervalo de confiança para Recall:", interval_confidence(recall_random_forest))
print("Intervalo de confiança para F1-Score:", interval_confidence(f1_random_forest))

print("Intervalo de confiança para a árvore de decisão:")
print("Intervalo de confiança para Precision:", interval_confidence(precision_arvore_decisao))
print("Intervalo de confiança para Recall:", interval_confidence(recall_arvore_decisao))
print("Intervalo de confiança para F1-Score:", interval_confidence(f1_arvore_decisao))
