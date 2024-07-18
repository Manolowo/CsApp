import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ruta de los datos y modelos
data_path = 'data-center/data.csv'
model_folder1 = 'models/decisiontree1'
model_folder2 = 'models/decisiontree2'
output_dir = 'static/graficos'

# Cargar datos
cs_clasification = pd.read_csv(data_path, sep=';')

# Definir columnas de armas
weapon_columns = ['PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']

# Crear directorio para guardar gráficos si no existe
os.makedirs(output_dir, exist_ok=True)

# Título de la aplicación
st.title("Predicción de Clasificación de Armas")

# Mostrar dataset
st.subheader("Dataset")
st.write(cs_clasification.head())

# Función para cargar y evaluar modelos
def load_and_evaluate_models(model_folder, cs_clasification):
    model_results = {}
    for column in weapon_columns:
        # Convertir la columna objetivo en binaria
        cs_clasification[column] = cs_clasification[column].apply(lambda x: 1 if x > 0 else 0)

        # Seleccionar características y objetivo para esta arma
        X = cs_clasification[['TeamStartingEquipmentValue']]
        y = cs_clasification[column]

        # Cargar el modelo
        model_filename = os.path.join(model_folder, f'model_{column}.pkl')
        if not os.path.exists(model_filename):
            continue
        model = joblib.load(model_filename)

        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(X)

        # Evaluar el rendimiento del modelo
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        classification_rep = classification_report(y, y_pred, output_dict=True)

        model_results[column] = {
            'accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'classification_report': classification_rep
        }

        # Gráfico de Confusión
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
        plt.title(f'Matriz de Confusión para {column}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_{column}.png')
        plt.close()

    return model_results

# Evaluar modelos y mostrar resultados
st.subheader("Resultados de DecisionTree1")
results1 = load_and_evaluate_models(model_folder1, cs_clasification)
for weapon, result in results1.items():
    st.write(f"**{weapon}**")
    st.write(f"Accuracy: {result['accuracy']:.2f}")
    st.write(f"Classification Report:")
    st.write(pd.DataFrame(result['classification_report']).transpose())
    st.image(f'{output_dir}/confusion_{weapon}.png')

st.subheader("Resultados de DecisionTree2")
results2 = load_and_evaluate_models(model_folder2, cs_clasification)
for weapon, result in results2.items():
    st.write(f"**{weapon}**")
    st.write(f"Accuracy: {result['accuracy']:.2f}")
    st.write(f"Classification Report:")
    st.write(pd.DataFrame(result['classification_report']).transpose())
    st.image(f'{output_dir}/confusion_{weapon}.png')
