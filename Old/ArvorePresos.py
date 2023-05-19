from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

with open('BasePreProcessada/presos.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    }
    #retorna um gridSearch
    grid = GridSearchCV(
        estimator = DecisionTreeClassifier(),
        param_grid = params,
        cv = 2,
        n_jobs = 5,
        verbose = 1,
    )
    grid.fit(X_treino, y_treino)#Procura os melhores parametros
    params_filtrados = grid.best_params_


    arvore = DecisionTreeClassifier(max_depth = 3, criterion = params_filtrados.get('criterion'),max_features= params_filtrados.get("max_features"))
    arvore.fit(X_treino,y_treino)#Treinamos a arvore
    previsoes = arvore.predict(X_teste)#Utilizamos a arvore treinada para tentar prever novas entradas de dados
    print(accuracy_score(y_teste,previsoes))#printando a acuracia da arvore, usando como parametro os rotulos do input utilizado na classe predict()



# Salvando a árvore em um pkl
with open('presosArvore.pkl', 'wb') as f:
    pickle.dump(arvore, f)
