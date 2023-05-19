import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
with open('BasePreProcessada/breast.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# Criar o classificador de Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo com os dados de treinamento
random_forest.fit(X_treino, y_treino)

# Fazer previsões usando o modelo treinado
y_pred = random_forest.predict(X_teste)

# Avaliar a acurácia do modelo
accuracy = accuracy_score(y_teste, y_pred)
print("Acurácia:", accuracy)
