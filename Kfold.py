from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
from scipy import stats
import copy
import numpy as np
import pickle
import scipy.stats as st
with open('ModelosTreinados/rede_neural2.pkl', 'rb') as f:
    rede_neural= pickle.load(f)

with open('ModelosTreinados/randomForest.pkl', 'rb') as f:
    random_forest = pickle.load(f)

with open('ModelosTreinados/arvore.pkl', 'rb') as f:
    arvore_carregada = pickle.load(f)

with open('BasePreProcessada/breast.pkl', 'rb') as f:
    dataset = pickle.load(f)

with open('BasePreProcessada/breast_rede.pkl', 'rb') as f:
    dataset_rede = pickle.load(f)    
    # X_treino, X_teste, y_treino, y_teste = pickle.load(f)

with open('ModelosTreinados/tensor.pkl', 'rb') as f:
    tensor = pickle.load(f)

def interval_confidence(values):
    return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))

def calcular_media_colunas(matriz):
    # Calcular a média das colunas da matriz
    media_colunas = np.mean(matriz, axis=0)
    return media_colunas.tolist()

def k_fold(k, modelo, dataset):
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

print("Decision Tree")
k_fold(5, arvore_carregada, dataset)
print("Randomn forest")
k_fold(5,random_forest,dataset)
print("Rede Neural")
k_fold(5,rede_neural,dataset_rede)



