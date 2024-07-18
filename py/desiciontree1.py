import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar los datos
cs_clasification = pd.read_csv('C:/Users/diego/OneDrive/Escritorio/CsApp/data-center/data.csv', sep=';')

# Definir columnas de armas
weapon_columns = ['PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']

# Crear directorios si no existen
model_folder = 'C:/Users/diego/OneDrive/Escritorio/CsApp/models/decisiontree1'
os.makedirs(model_folder, exist_ok=True)

output_dir = 'C:/Users/diego/OneDrive/Escritorio/CsApp/py/static/graficos'
os.makedirs(output_dir, exist_ok=True)

# Iterar sobre cada columna de arma
for column in weapon_columns:
    # Convertir la columna objetivo en binaria
    cs_clasification[column] = cs_clasification[column].apply(lambda x: 1 if x > 0 else 0)

    # Seleccionar características y objetivo para esta arma
    X = cs_clasification[['TeamStartingEquipmentValue']]
    y = cs_clasification[column]

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión logística
    logistic_model = LogisticRegression(max_iter=1000)  # Aumentar el número máximo de iteraciones si es necesario
    logistic_model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    model_filename = os.path.join(model_folder, f'model_{column}.pkl')
    joblib.dump(logistic_model, model_filename)

    # Realizar predicciones en el conjunto de prueba
    y_pred = logistic_model.predict(X_test)

    # Evaluar el rendimiento del modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    # Imprimir la exactitud y el reporte de clasificación
    print(f'Exactitud para {column}: {accuracy:.2f}')
    print(f'Reporte de Clasificación para {column}:\n{classification_rep}')

    # Gráfico de Confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
    plt.title(f'Matriz de Confusión para {column}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_{column}.png'))
    plt.close()
    
    # Gráfico de Histograma de Predicciones
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred, kde=False, bins=2, color='blue')
    plt.title(f'Histograma de Predicciones para {column}')
    plt.xlabel('Predicción (0 = No, 1 = Sí)')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histogram_{column}.png'))
    plt.close()