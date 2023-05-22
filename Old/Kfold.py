from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
from imblearn.under_sampling import NearMiss
from sklearn.base import clone
from scipy import stats
import copy
import numpy as np
import pickle
import scipy.stats as st


with open('ModelosTreinados/randomForest.pkl', 'rb') as f:
    random_forest = pickle.load(f)

with open('ModelosTreinados/arvore.pkl', 'rb') as f:
    arvore_carregada = pickle.load(f)

with open('raw.pkl', 'rb') as f:
    dataset = pickle.load(f)


def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))

def calcular_media_colunas(matriz):
    # Calcular a média das colunas da matriz
    media_colunas = np.mean(matriz, axis=0)
    return media_colunas.tolist()

def k_fold1(k, modelo, dataset):
    # x, x_test, y, y_test
    melhor = 0
    x = dataset[0]
    x_test = dataset[1]
    y = dataset[2]
    y_test = dataset[3]
    kf = KFold(n_splits=k, shuffle=True)
    fold = 1
    labels = ['0', '1']
    labels2 = [0, 1]
    precision_valid = [0, 0]
    recall_valid = [0, 0]
    fscore_valid = [0, 0]
    
    precision_values = []  # Lista vazia para armazenar os valores de precision
    recall_values = []  # Lista vazia para armazenar os valores de recall
    fscore_values = []  # Lista vazia para armazenar os valores de fscore
    for train_index, valid_index in kf.split(x):
        #[tuple[ndarray, ndarray]]
        x_train = x[train_index]
        y_train = y[train_index]

        x_valid = x[valid_index]
        y_valid = y[valid_index]

        modelo.fit(x_train, y_train)
        predito = modelo.predict(x_valid)


        precision, recall, fscore, support = precision_recall_fscore_support(y_valid, predito, average=None)
        precision_values.append(precision)  # Adicionar precision à lista
        recall_values.append(recall)  # Adicionar recall à lista
        fscore_values.append(fscore)  # Adicionar fscore à lista

        precision_valid = np.add(precision_valid, precision)# variavel que incrementa todos os precision scores
        recall_valid = np.add(recall_valid, recall)# variavel que incrementa todos os recall's
        fscore_valid = np.add(fscore_valid, fscore)# variavel que incrementa todos os fscore's

        acuracia = np.mean(y_valid == predito)
        if acuracia > melhor:
            melhor = acuracia
            best_model = copy.deepcopy(modelo)

        print('FOLD: ' + str(fold))
        print(metrics.classification_report(y_valid, predito, target_names=labels))

        cm = metrics.confusion_matrix(y_valid, predito, labels=labels2)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels2)
        disp.plot()
        plt.savefig(f'MatrizesKfold/confusion_matrix_fold_{fold}.png')  # Salvar a matriz de confusão como imagem
        plt.close()  # Fechar a figura para liberar memória
        
        fold += 1
        print('---------------------------------------------------------')
    p = best_model.predict(x_test)
    print(classification_report(y_test, p))
    print("Precisão média na validação das classes Inferior e Superior: ", np.round(precision_valid / k, 2))
    print("----------------------------------------------------------------")
    print("Recall médio na validação das classes Inferior e Superior: ", np.round(recall_valid / k, 2))
    print("----------------------------------------------------------------")
    print("F1 score médio na validação das classes Inferior e Superior: ", np.round(fscore_valid / k, 2))
    
    print("Intervalo de confiança do Fscore: ",calcular_media_colunas(fscore_values))
    print("Intervalo de confiança do recall: ",calcular_media_colunas(recall_values))
    print("Intervalo de confiança da precision: ",calcular_media_colunas(precision_values))

def k_fold_cross_validation(k, model, dataset):
    X_train = dataset[0]  # Características de treino
    X_test = dataset[1]  # Características de teste
    y_train = dataset[2]  # Rótulos de treino
    y_test = dataset[3]  # Rótulos de teste

    skf = StratifiedKFold(n_splits=k, shuffle=True)  # Divisão em k folds estratificados
    skf = KFold(n_splits=k, shuffle=True)

    scores = []  # Lista para armazenar as métricas de cada fold

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]  # Dados de treino e validação do fold
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]  # Rótulos de treino e validação do fold

        # Undersampling NearMiss nos dados de treino do fold
        nm = NearMiss(version=2)
        X_train_resampled, y_train_resampled = nm.fit_resample(X_train_fold, y_train_fold)
        print(np.unique(y_train_resampled,return_counts=True))

        # Treinamento do modelo
        model.fit(X_train_resampled, y_train_resampled)

        # Avaliação do modelo no conjunto de validação do fold
        y_pred = model.predict(X_val_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)

        # Relatório de classificação do fold
        print(f"Fold {len(scores)}:")
        print(classification_report(y_val_fold, y_pred))
        print("------------------------------")

    # Avaliação final no conjunto de teste
    y_pred_test = model.predict(X_test)
    final_score = model.score(X_test, y_test)
    scores.append(final_score)

    # Relatório de classificação do conjunto de teste
    print("Final Test Set:")
    print(classification_report(y_test, y_pred_test))
    print("------------------------------")

    return scores

rf = RandomForestClassifier(n_estimators=100, random_state=42)
dt = DecisionTreeClassifier()
print("Decision Tree")
print(k_fold_cross_validation(5,dt,dataset))
# print("Randomn forest")
# k_fold(5,random_forest,dataset)'
#passar o modelo destreinado




