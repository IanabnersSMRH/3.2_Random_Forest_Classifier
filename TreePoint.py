from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  # <--- Para dividir los datos
from sklearn.metrics import accuracy_score

# 1. Generar dataset sintético para clasificación binaria
X_data, y_data = make_classification(n_samples=300, n_features=5, n_classes=2, random_state=42)

# 2. Dividir el dataset: 80% para entrenamiento y 20% para prueba (test)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 3. Árbol de decisión individual (entrenado con train, evaluado con test)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# 4. Bosque Aleatorio / Random Forest (entrenado con train, evaluado con test)
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 5. Resultados de precisión sobre datos no vistos (Test)
print("=== ÁRBOLES VS RANDOM FOREST (DATOS DE PRUEBA) ===")
print(f"Precisión Árbol Individual: {accuracy_score(y_test, dt_pred)*100:.2f}%")
print(f"Precisión Random Forest: {accuracy_score(y_test, rf_pred)*100:.2f}%")