
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

with open('comUnder.pkl', 'rb') as f:
  X_treino, X_teste, y_treino, y_teste = pickle.load(f)

params = {
    'criterion':  ['gini', 'entropy'],
    'n_estimators':  [1, 2, 4, 6, 8, 10, 20],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
}

florest = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=params,
    cv=2,
    n_jobs=5,
    verbose=1,
)

florest.fit(X_treino, y_treino)
params = florest.best_params_
print(params)

florest = RandomForestClassifier(n_estimators= params.get("n_estimators"), max_features= params.get("max_features"), criterion= params.get("criterion"), random_state = 0)
florest.fit(X_treino, y_treino)

previsoes = florest.predict(X_teste)
previsoes
    
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_teste,previsoes)

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(florest)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)

print(classification_report(y_teste, previsoes))

print(florest.feature_importances_) 
with open('ModelosTreinados/randomForest.pkl', 'wb') as f:
    pickle.dump(florest, f)
