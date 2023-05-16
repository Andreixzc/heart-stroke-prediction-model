import pickle
from sklearn.model_selection import cross_val_score
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.stats as st
import numpy as np
from scipy.stats import ttest_ind

with open('ModelosTreinados/randomForest.pkl', 'rb') as f:
    random_forest = pickle.load(f)

with open('ModelosTreinados/arvore.pkl', 'rb') as f:
    arvore_carregada = pickle.load(f)

with open('BasePreProcessada/breast.pkl', 'rb') as f:
    # dataset = pickle.load(f)
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# F1-scores do Random Forest
k = 2
f1_scores_rf = cross_val_score(random_forest, X_treino, y_treino, cv=k, scoring='f1_macro')
mean_rf = np.mean(f1_scores_rf)
std_rf = np.std(f1_scores_rf)
n_rf = len(f1_scores_rf)

# F1-scores da Árvore de Decisão
f1_scores_arvore = cross_val_score(arvore_carregada, X_treino, y_treino, cv=k, scoring='f1_macro')
mean_arvore = np.mean(f1_scores_arvore)
std_arvore = np.std(f1_scores_arvore)
n_arvore = len(f1_scores_arvore)

print("F1-scores do Random Forest:", f1_scores_rf)
print("F1-scores da Árvore de Decisão:", f1_scores_arvore)

#Function for calculating confidence interval from cross-validation
def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))


print("Intervalo de confianca do f1_score Random Forest:",interval_confidence(f1_scores_rf))
print("Intervalo de confianca da Árvore de decisão:",interval_confidence(f1_scores_arvore))


# t_statistic, p_value = ttest_ind(f1_scores_rf, f1_scores_arvore)

# print("Média F1-scores Random Forest:", mean_rf)
# print("Desvio padrão F1-scores Random Forest:", std_rf)

# print("Média F1-scores Árvore de Decisão:", mean_arvore)
# print("Desvio padrão F1-scores Árvore de Decisão:", std_arvore)

# print("Valor de p do teste t:", p_value)




