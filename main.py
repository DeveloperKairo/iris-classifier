import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("--- Início da Classificação de Flores Iris com KNN ---")

iris = load_iris()
print("\nChaves do dataset Iris:", iris.keys())

df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['species_name'] = iris.target_names[iris.target]
df_iris['target'] = iris.target

print("\nPrimeiras 5 linhas do Dataset Iris:")
print(df_iris.head())

print("\nInformações gerais do DataFrame:")
df_iris.info()

print("\nEstatísticas descritivas das características numéricas:")
print(df_iris.describe())

print("\nGerando pairplot para visualização das relações entre características...")
sns.pairplot(df_iris, hue='species_name')
plt.suptitle('Relação entre as Características das Flores Iris por Espécie', y=1.02)
plt.show()
print("Pairplot gerado. Observe como as espécies se agrupam")

X = df_iris.drop(['species_name', 'target'], axis=1)
y = df_iris['target']

print(f"\nFormato de X (Características): {X.shape}")
print(f"Formato de y (Alvo): {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(f"\nFormato do Conjunto de Treinamento (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Formato do Conjunto de Teste (X_test, y_test): {X_test.shape}, {y_test.shape}")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPrimeiras 5 linhas de X_train_scaled (após escalonamento):")
print(X_train_scaled[:5])

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

print("\nModelo KNN treinado com sucesso com K=5!")

y_pred = knn.predict(X_test_scaled)

print("\nPrevisões realizadas no conjunto de teste (primeiras 10):")
print(y_pred[:10])
print("Valores reais do conjunto de teste (primeiras 10):")
print(y_test.values[:10])

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo KNN (K=5): {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão (K=5):")
print(conf_matrix)

class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nRelatório de Classificação (K=5):")
print(class_report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão do Modelo KNN (K=5)')
plt.show()
print("Matriz de Confusão para K=5 gerada.")

k_values = list(range(1, 21))
cv_scores = [] 

print("\nIniciando otimização do valor de K com validação cruzada...")
for k in k_values:
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_cv, X_train_scaled, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k_index = np.argmax(cv_scores)
optimal_k = k_values[optimal_k_index]

print(f"\nAcurácias médias de validação cruzada para diferentes K: {cv_scores}")
print(f"O valor ótimo de K encontrado é: {optimal_k}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='-')
plt.title('Acurácia Média da Validação Cruzada vs. Valor de K')
plt.xlabel('Número de Vizinhos (K)')
plt.ylabel('Acurácia Média')
plt.xticks(k_values) 
plt.grid(True)
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'K Ótimo = {optimal_k}')
plt.legend()
plt.show()
print("Gráfico de Acurácia vs. K gerado")

knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)

knn_optimal.fit(X_train_scaled, y_train)

y_pred_optimal = knn_optimal.predict(X_test_scaled)

accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f"\nAcurácia do modelo KNN com K ótimo ({optimal_k}): {accuracy_optimal:.4f}")

conf_matrix_optimal = confusion_matrix(y_test, y_pred_optimal)
print(f"\nMatriz de Confusão do Modelo Otimizado:")
print(conf_matrix_optimal)

class_report_optimal = classification_report(y_test, y_pred_optimal, target_names=iris.target_names)
print("\nRelatório de Classificação do Modelo Otimizado:")
print(class_report_optimal)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_optimal, annot=True, fmt='d', cmap='Greens', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title(f'Matriz de Confusão do Modelo KNN (K={optimal_k})')
plt.show()
print(f"Matriz de Confusão para K={optimal_k} gerada.")

print("\n--- Classificação de Flores Iris com KNN Concluída ---")