
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

with open('raw.pkl', 'rb') as f:
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

import pandas as pd
print(classification_report(y_teste, previsoes))


# report = classification_report(y_teste, previsoes, output_dict=True)
# report_df = pd.DataFrame(report).transpose()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Configurações para a plotagem do heatmap
# fig, ax = plt.subplots(figsize=(10, 7))
# heatmap = ax.imshow(report_df.values, cmap='Blues')

# # Adiciona os valores dentro das células do heatmap
# for i in range(len(report_df.columns)):
#     for j in range(len(report_df.index)):
#         ax.text(i, j, format(report_df.values[j, i], '.2f'), ha="center", va="center")

# # Configura os rótulos dos eixos x e y
# ax.set_xticks(np.arange(len(report_df.columns)))
# ax.set_yticks(np.arange(len(report_df.index)))
# ax.set_xticklabels(report_df.columns)
# ax.set_yticklabels(report_df.index)

# # Rotaciona os rótulos do eixo x para melhor visualização
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# # Adiciona uma barra de cores
# plt.colorbar(heatmap)

# # Salva a imagem
# plt.savefig('classification_report.png')





print(florest.feature_importances_) 
with open('ModelosTreinados/randomForest.pkl', 'wb') as f:
    pickle.dump(florest, f)
