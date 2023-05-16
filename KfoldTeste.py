from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pickle

with open('ModelosTreinados/arvore.pkl', 'rb') as f:
    arvore_carregada = pickle.load(f)

with open('BasePreProcessada/breast.pkl', 'rb') as f:
    dataset = pickle.load(f)
    # X_treino, X_teste, y_treino, y_teste = pickle.load(f)


with open('ModelosTreinados/arvore.pkl', 'rb') as f:
    arvore_carregada = pickle.load(f)

with open('BasePreProcessada/breast.pkl', 'rb') as f:
    dataset = pickle.load(f)
    # X_treino, X_teste, y_treino, y_teste = pickle.load(f)

def interval_confidence(values):
    return stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=stats.sem(values))

def k_fold(k, modelo, data):
    X_treino = data[0]
    X_teste = data[1]
    y_treino = data[2]
    y_teste = data[3]
    kf = KFold(n_splits=k, shuffle=True)
    matrizes_confusao = []
    resultados = cross_val_predict(modelo, X_teste, y_teste, cv=k)
    media = accuracy_score(y_teste, resultados)
    desvio_padrao = np.std(resultados)

    for fold, (train_index, test_index) in enumerate(kf.split(X_teste)):
        X_train, X_test = X_teste[train_index], X_teste[test_index]
        y_train, y_test = y_teste[train_index], y_teste[test_index]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        matrizes_confusao.append(cm)
        
        # Exibir o número da matriz de confusão
        print(f"Matriz de Confusão - Fold {fold+1}")
        print(classification_report(y_test, y_pred))
        print("-----------------------")

        # Plotar e salvar a matriz de confusão em uma imagem
        plt.figure() 
        disp = ConfusionMatrix(modelo, classes=['0', '1'])
        disp.fit(X_train, y_train)
        disp.score(X_test, y_test)
        disp.show(outpath=f"MatrizesKfold/matriz_confusao_{fold+1}.png")

    # Imprimindo a média, desvio padrão e intervalo de confiança
    print("Média:", media)
    print("Desvio Padrão:", desvio_padrao)
    print("Intervalo de Confiança:", interval_confidence(resultados))
    print("Cross_val_predict:", resultados)
    confidence_interval = stats.norm.interval(0.95, loc=media, scale=desvio_padrao / np.sqrt(len(resultados)))
    print("Intervalo de confiança usando stats.norm:", confidence_interval)

k_fold(3, arvore_carregada, dataset)







# k = 2
# kf = KFold(n_splits=k, shuffle=True)
# matrizes_confusao = []
# resultados = cross_val_score(arvore_carregada, X_teste, y_teste, cv=k, scoring='accuracy')
# media = resultados.mean()
# desvio_padrao = resultados.std()

# for fold, (train_index, test_index) in enumerate(kf.split(X_teste)):
#     X_treino, X_test = X_teste[train_index], X_teste[test_index]
#     y_treino, y_test = y_teste[train_index], y_teste[test_index]
#     #x_train?
    
#     arvore_carregada.fit(X_treino, y_treino)
#     y_pred = arvore_carregada.predict(X_test)
    
    
#     cm = confusion_matrix(y_test, y_pred)
#     print(cm)
#     matrizes_confusao.append(cm)
    
#     # Exibir o número da matriz de confusão
#     print(f"Matriz de Confusão - Fold {fold+1}")
#     print(classification_report(y_test, y_pred))
#     print("-----------------------")

#     # Plotar e salvar a matriz de confusão em uma imagem
#     plt.figure() 
#     disp = ConfusionMatrix(arvore_carregada, classes=['0', '1'])
#     disp.fit(X_treino, y_treino)
#     disp.score(X_test, y_test)
#     disp.show(outpath=f"MatrizesKfold/matriz_confusao_{fold+1}.png")

# n = len(resultados)
# def interval_confidence(values):
#     return st.t.interval(confidence=0.95, df=len(values)-1, loc=np.mean(values), scale=st.sem(values))

# # Imprimindo a média, desvio padrão e intervalo de confiança
# print("Média:", media)
# print("Desvio Padrão:", desvio_padrao)
# print("Intervalo de Confiança:", interval_confidence(resultados))
# confidence_interval = stats.norm.interval(0.95, loc=media, scale=desvio_padrao / len(resultados) ** 0.5)
# print(confidence_interval)
# print(resultados)
# print("Accuracy: %0.2f (+/- %0.2f)" % (resultados.mean(), resultados.std() * 2))





