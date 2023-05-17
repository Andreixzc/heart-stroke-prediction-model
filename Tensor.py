import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Carregar os dados
with open('BasePreProcessada/breast.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# Pré-processamento dos dados (padronização)
scaler = StandardScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

# Definir a arquitetura da rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_treino.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_treino, y_treino, epochs=10, batch_size=32, validation_split=0.2)

# Fazer previsões usando o modelo treinado
y_pred_probs = model.predict(X_teste)
y_pred = (y_pred_probs > 0.5).astype(int)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_teste, y_pred)
print("Acurácia:", accuracy)

with open('ModelosTreinados/tensor.pkl', 'wb') as f:
    pickle.dump(model, f)
