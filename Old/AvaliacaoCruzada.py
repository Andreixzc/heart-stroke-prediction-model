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
    rede_neural= pickle.load(f)

with open('ModelosTreinados/randomForest.pkl', 'rb') as f:
    random_forest = pickle.load(f)

with open('ModelosTreinados/arvore.pkl', 'rb') as f:
    arvore_carregada = pickle.load(f)

with open('BasePreProcessada/breast.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))


recall_rede_neural = cross_validate(rede_neural, X_treino, y_treino, cv=5, scoring=['accuracy', 'recall'])
recall_random = cross_val_score(random_forest, X_treino, y_treino, cv=5, scoring='recall')
recall_arvore_decisao = cross_val_score(arvore_carregada, X_treino, y_treino, cv=5, scoring='recall')
print(recall_rede_neural)



# Imprimir os resultados
print("Recall na rede neural:")
print(recall_rede_neural)

print("Recall random forest:")
print(recall_random)

print("Recall Decision Tree:")
print(recall_arvore_decisao)

print("Intervalo de confianca recall rede neural:")
print(interval_confidence(recall_rede_neural))

print("Intervalo de confianca recall random forest:")
print(interval_confidence(recall_random))

print("Intervalo de confianca recall decision tree:")
print(interval_confidence(recall_arvore_decisao))

