import streamlit as st
import requests

# URL de la aplicación Flask
FLASK_URL = 'http://127.0.0.1:5000'

# Página principal
st.title('Interfaz de Aplicación Flask')

# Botón para cargar la página principal de Flask
if st.button('Cargar Página Principal'):
    try:
        response = requests.get(FLASK_URL)
        if response.status_code == 200:
            st.write(response.text)
        else:
            st.write(f'Error al cargar la página principal. Status Code: {response.status_code}')
    except requests.exceptions.RequestException as e:
        st.write(f'Error de conexión: {e}')

# Realizar predicciones
st.sidebar.header('Realizar Predicción')
team_starting_value = st.sidebar.number_input('Team Starting Value', min_value=0.0, step=0.1)

if st.sidebar.button('Predecir'):
    payload = {'team_starting_value': team_starting_value}
    try:
        response = requests.post(f'{FLASK_URL}/predict', data=payload)
        if response.status_code == 200:
            st.write(response.text)
        else:
            st.write(f'Error al realizar la predicción. Status Code: {response.status_code}')
    except requests.exceptions.RequestException as e:
        st.write(f'Error de conexión: {e}')
