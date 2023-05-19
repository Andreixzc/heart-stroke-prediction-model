import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Carregar os dados
with open('BasePreProcessada/breast_rede.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# Criar a rede neural com parâmetros ajustados
rede_neural = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000, random_state=42)
rede_neural.fit(X_treino,y_treino)
predict = rede_neural.predict(X_teste)
print(predict)
# accuracy = accuracy_score(y_teste, predict)
# print("Acurácia:", accuracy)
with open('ModelosTreinados/rede_neural2.pkl', 'wb') as f:
    pickle.dump(rede_neural, f)


