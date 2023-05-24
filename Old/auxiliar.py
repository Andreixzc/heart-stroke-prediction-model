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



def k_fold_cross_validation_com_grid_e_under_completo(k, model, dataset):
    X_train = np.concatenate((dataset[0], dataset[1]), axis=0)  # Características de treino
    y_train = np.concatenate((dataset[2], dataset[3]), axis=0)  # Rótulos de treino
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]
    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados
    scores = []  # Lista para armazenar as métricas de cada fold
    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampler nos dados de treino do fold
        nearmiss = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train_fold,y_train_fold)

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
        #Positivo e negativo.
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])

        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])

        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        # Relatório de classificação do fold
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:", len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("--------------------------------------------")
   
    # # Exibição das médias e intervalos de confiança
    print(f1_score_superior)
    print(f1_score_inferior)
    print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))


def k_fold_cross_validation_com_grid_e_over_completo(k, model, dataset):
    X_train = np.concatenate((dataset[0], dataset[1]), axis=0)  # Características de treino
    y_train = np.concatenate((dataset[2], dataset[3]), axis=0)  # Rótulos de treino
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]
    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados
    scores = []  # Lista para armazenar as métricas de cada fold
    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }

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
        #Positivo e negativo.
        precision, recall, fscore, support = precision_recall_fscore_support(y_val_fold, y_pred, average=None)
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])

        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])

        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        # Relatório de classificação do fold
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:", len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("--------------------------------------------")
   
    # # Exibição das médias e intervalos de confiança
    print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))

def k_fold_cross_validation_com_grid_e_over_parcial(k, model, dataset):
    X_train = dataset[0]  # Características de treino
    X_test = dataset[1]  # Características de teste
    y_train = dataset[2]  # Rótulos de treino
    y_test = dataset[3]  # Rótulos de teste
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]

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
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])
        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])
        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:",len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("------------------------------")

    # Avaliação final no conjunto de teste
    y_pred_test = model.predict(X_test)
    final_score = model.score(X_test, y_test)
    scores.append(final_score)
    # Exibição das médias e intervalos de confiança
    # # Exibição das médias e intervalos de confiança
    print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))
    # Relatório de classificação do conjunto de teste
    print("Relatorio Final do conjunto inteiro de dados (sem k-fold):")
    print(classification_report(y_test, y_pred_test))
    print("------------------------------")

def k_fold_cross_validation_com_grid_e_under_parcial(k, model, dataset):
    X_train = dataset[0]  # Características de treino
    X_test = dataset[1]  # Características de teste
    y_train = dataset[2]  # Rótulos de treino
    y_test = dataset[3]  # Rótulos de teste
    f1_score_superior = []
    f1_score_inferior = []
    recall_superior = []
    recall_inferior = []
    precisao_superior = []
    precisao_inferior = []
    precision_valid = [0,0]
    recall_valid = [0,0]
    fscore_valid = [0,0]

    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados

    scores = []  # Lista para armazenar as métricas de cada fold

    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }



    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampler nos dados de treino do fold
        nearmiss = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nearmiss.fit_resample(X_train_fold,y_train_fold)

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
        precision_valid = np.add(precision_valid,precision)
        recall_valid = np.add(recall_valid,recall)
        fscore_valid = np.add(fscore_valid,fscore) 
        f1_score_superior.append(fscore[0])
        f1_score_inferior.append(fscore[1])
        recall_superior.append(recall[0])
        recall_inferior.append(recall[1])
        precisao_superior.append(precision[0])
        precisao_inferior.append(precision[1])
        print(f"Fold {len(scores)}:")
        print("Melhores parametros para o fold:",len(scores))
        print(best_params)
        print(classification_report(y_val_fold, y_pred))
        print("------------------------------")

    # Avaliação final no conjunto de teste
    y_pred_test = model.predict(X_test)
    final_score = model.score(X_test, y_test)
    scores.append(final_score)
    # Exibição das médias e intervalos de confiança
    # # Exibição das médias e intervalos de confiança
    print(recall_inferior)
    # print("Intervalo de confiança do Fscore superior: ", interval_confidence(f1_score_superior))
    # print("Intervalo de confiança do Fscore inferior: ", interval_confidence(f1_score_inferior))
    print("Intervalo de confiança do Recall superior: ", interval_confidence(recall_superior))
    print("Intervalo de confiança do Recall inferior: ", interval_confidence(recall_inferior))
    print("Intervalo de confiança da Precisao superior: ", interval_confidence(precisao_superior))
    print("Intervalo de confiança da Precisao inferior: ", interval_confidence(precisao_inferior))
    print("----------------------------------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: " ,  np.round(fscore_valid/k,2))
    print("Recall médio na validação das classes Inferior e Superior: " ,  np.round(recall_valid/k,2))
    print("Precisão média na validação das classes Inferior e Superior: " , np.round(precision_valid/k,2))
    # Relatório de classificação do conjunto de teste
    print("Relatorio Final do conjunto inteiro de dados (sem k-fold):")
    print(classification_report(y_test, y_pred_test))
    print("------------------------------")


rf = RandomForestClassifier()
dt = DecisionTreeClassifier()

# print("Rf com over:")
# k_fold_cross_validation_com_grid_e_over(5,rf,dataset)
# print("Rf com under:")
# k_fold_cross_validation_com_grid_e_under(5,rf,dataset)

print("DT")
k_fold_cross_validation_com_grid_e_under_parcial(5,dt,dataset)
k_fold_cross_validation_com_grid_e_over_completo(5,dt,dataset)
print("rf")
k_fold_cross_validation_com_grid_e_under_parcial(5,rf,dataset)
k_fold_cross_validation_com_grid_e_over_completo(5,rf,dataset)


# print(interval_confidence(recall))
