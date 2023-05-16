import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier

# Carregar os dados
with open('BasePreProcessada/breast.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# Criar a rede neural com parâmetros ajustados
rede_neural = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000, random_state=42)


with open('ModelosTreinados/rede_neural.pkl', 'wb') as f:
    pickle.dump(rede_neural, f)


