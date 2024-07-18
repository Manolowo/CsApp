import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar el dataset
cs_clasification = pd.read_csv('C:/Users/diego/OneDrive/Escritorio/CsApp/data-center/data.csv', sep=';')

# Definir columnas de armas
weapon_columns = ['PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']

# Crear carpeta para almacenar los modelos si no existe
model_folder = 'C:/Users/diego/OneDrive/Escritorio/CsApp/models/decisiontree2'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

output_dir = 'C:/Users/diego/OneDrive/Escritorio/CsApp/py/static/graficos'
os.makedirs(output_dir, exist_ok=True)

# Convertir la columna 'RoundWinner' en numérica
cs_clasification['RoundWinner'] = pd.to_numeric(cs_clasification['RoundWinner'], errors='coerce')
cs_clasification['RoundWinner'] = cs_clasification['RoundWinner'].astype(int)

for column in weapon_columns:
    cs_clasification[column] = cs_clasification[column].apply(lambda x: 1 if x > 0 else 0)
    print(cs_clasification.groupby(column)['RoundWinner'].value_counts())

cs_clasification['RoundWinner'] = cs_clasification['RoundWinner'].astype(int)

X = cs_clasification[weapon_columns]

y = cs_clasification['RoundWinner']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisiones
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Guardar el modelo entrenado en un archivo .pkl en la carpeta 'models'
model_filename = os.path.join(model_folder, f'model_RoundWinner.pkl')
joblib.dump(decision_tree_model, model_filename)

# Realizar predicciones en el conjunto de prueba
y_pred = decision_tree_model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Gráfico de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
plt.title(f'Matriz de Confusión para RoundWinner')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'confusion_RoundWinner.png'))
plt.close()