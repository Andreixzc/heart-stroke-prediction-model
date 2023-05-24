import sys

# Redirecionar a saída para um arquivo de texto
sys.stdout = open('output.txt', 'w')
import pickle
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import scipy.stats as st
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import NearMiss

with open('raw1.pkl', 'rb') as f:
    dataset = pickle.load(f)
def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))

def calcular_media_colunas(matriz):
    # Calcular a média das colunas da matriz
    media_colunas = np.mean(matriz, axis=0)
    return media_colunas.tolist()


def k_fold_cross_validation_com_grid_e_under(k, model, dataset):
    X_train = dataset[0]  # Características de treino
    X_test = dataset[1]  # Características de teste
    y_train = dataset[2]  # Rótulos de treino
    y_test = dataset[3]  # Rótulos de teste
    precision_values = []  # Lista vazia para armazenar os valores de precision
    recall_values = []  # Lista vazia para armazenar os valores de recall
    fscore_values = []  # Lista vazia para armazenar os valores de fscore

    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados

    scores = []  # Lista para armazenar as métricas de cada fold

    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampling NearMiss nos dados de treino do fold
        nm = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nm.fit_resample(X_train_fold, y_train_fold)

        # GridSearch no fold
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados
        best_params = grid_search.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros
        model.set_params(**best_params)
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_values.append(precision)  # Adicionar precision à lista
        recall_values.append(recall)  # Adicionar recall à lista
        fscore_values.append(fscore)  # Adicionar fscore à lista

        # Relatório de classificação do fold
        # print(f"Fold {len(scores)}:")
        # print("Melhores parametros para o fold:",len(scores))
        # print(best_params)
        # print(classification_report(y_val_fold, y_pred))
        # print("------------------------------")

    # Avaliação final no conjunto de teste
    y_pred_test = model.predict(X_test)
    final_score = model.score(X_test, y_test)
    scores.append(final_score)

    # Métricas de classificação do conjunto de teste
    f1_test = f1_score(y_test, y_pred_test, average=None)
    precision_test = precision_score(y_test, y_pred_test, average=None)
    recall_test = recall_score(y_test, y_pred_test, average=None)
    f1_scores.append(f1_test)
    precision_scores.append(precision_test)
    recall_scores.append(recall_test)

    # Cálculo da média do F1-score, precisão e recall para cada classe superior e inferior
    mean_f1_scores = [sum(scores) / len(scores) for scores in zip(*f1_scores)]
    mean_precision_scores = [sum(scores) / len(scores) for scores in zip(*precision_scores)]
    mean_recall_scores = [sum(scores) / len(scores) for scores in zip(*recall_scores)]
    

    # # Exibição das médias e intervalos de confiança
    # print("Intervalo de confiança do Fscore: ",calcular_media_colunas(fscore_values))
    # print("Intervalo de confiança do recall: ",calcular_media_colunas(recall_values))
    # print("Intervalo de confiança da precision: ",calcular_media_colunas(precision_values))
    # print("--------------------------------------------------------------------")
    # print("Media recall superior e inferior:",mean_recall_scores)
    # print("Media fscore superior e inferior:",mean_f1_scores)
    # print("Media precision superior e inferior:",mean_precision_scores)
     # Relatório de classificação do conjunto de teste
    print("Relatorio Final do conjunto inteiro de dados (sem k-fold):")
    print(classification_report(y_test, y_pred_test))
    print("------------------------------")










def k_fold_cross_validation_com_grid_e_over(k, model, dataset):
    X_train = dataset[0]  # Características de treino
    X_test = dataset[1]  # Características de teste
    y_train = dataset[2]  # Rótulos de treino
    y_test = dataset[3]  # Rótulos de teste
    precision_values = []  # Lista vazia para armazenar os valores de precision
    recall_values = []  # Lista vazia para armazenar os valores de recall
    fscore_values = []  # Lista vazia para armazenar os valores de fscore

    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados

    scores = []  # Lista para armazenar as métricas de cada fold

    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Oversampler nos dados de treino do fold
        ros = RandomOverSampler(random_state=0)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_fold, y_train_fold)

        # GridSearch no fold
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3)
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Melhores hiperparâmetros encontrados
        best_params = grid_search.best_params_

        # Treinamento do modelo com os melhores hiperparâmetros
        model.set_params(**best_params)
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_values.append(precision)  # Adicionar precision à lista
        recall_values.append(recall)  # Adicionar recall à lista
        fscore_values.append(fscore)  # Adicionar fscore à lista

         # Relatório de classificação do fold
        # print(f"Fold {len(scores)}:")
        # print("Melhores parametros para o fold:",len(scores))
        # print(best_params)
        # print(classification_report(y_val_fold, y_pred))
        # print("------------------------------")

    # Avaliação final no conjunto de teste
    y_pred_test = model.predict(X_test)
    final_score = model.score(X_test, y_test)
    scores.append(final_score)

    # Métricas de classificação do conjunto de teste
    f1_test = f1_score(y_test, y_pred_test, average=None)
    precision_test = precision_score(y_test, y_pred_test, average=None)
    recall_test = recall_score(y_test, y_pred_test, average=None)
    f1_scores.append(f1_test)
    precision_scores.append(precision_test)
    recall_scores.append(recall_test)

    # Cálculo da média do F1-score, precisão e recall para cada classe superior e inferior
    mean_f1_scores = [sum(scores) / len(scores) for scores in zip(*f1_scores)]
    mean_precision_scores = [sum(scores) / len(scores) for scores in zip(*precision_scores)]
    mean_recall_scores = [sum(scores) / len(scores) for scores in zip(*recall_scores)]
    

    # # Exibição das médias e intervalos de confiança
    # print("Intervalo de confiança do Fscore: ",calcular_media_colunas(fscore_values))
    # print("Intervalo de confiança do recall: ",calcular_media_colunas(recall_values))
    # print("Intervalo de confiança da precision: ",calcular_media_colunas(precision_values))
    # print("--------------------------------------------------------------------")
    # print("Media recall superior e inferior:",mean_recall_scores)
    # print("Media fscore superior e inferior:",mean_f1_scores)
    # print("Media precision superior e inferior:",mean_precision_scores)

    # Relatório de classificação do conjunto de teste
    print("Relatorio Final do conjunto inteiro de dados (sem k-fold):")
    print(classification_report(y_test, y_pred_test))
    print("------------------------------")

rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

print("Rf com over:")
k_fold_cross_validation_com_grid_e_over(5,rf,dataset)
print("Rf com under:")
k_fold_cross_validation_com_grid_e_under(5,rf,dataset)

print("DT com over:")
k_fold_cross_validation_com_grid_e_over(5,dt,dataset)
print("DT com under:")
k_fold_cross_validation_com_grid_e_under(5,dt,dataset)
# Fechar o arquivo de texto
sys.stdout.close()

# Restaurar a saída padrão
sys.stdout = sys.__stdout__