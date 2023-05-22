from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree
import matplotlib.pyplot as plt
import pickle

with open('base.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)


    print(X_treino)
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

# Plotando matriz de confusão
cm = ConfusionMatrix(arvore)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)
# cm.poof()
plt.savefig('figuras/matriz_confusao.png')
print(classification_report(y_teste, previsoes))


# # Plotando a árvore
# previsores = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_merried', 'ever_merried', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
# figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
# tree.plot_tree(arvore, feature_names=previsores, filled= True)
# plt.savefig('figuras/matriz_confusao.pngtree.png')

# Salvando a árvore em um pkl
with open('ModelosTreinados/arvore.pkl', 'wb') as f:
    pickle.dump(arvore, f)
