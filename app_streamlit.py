import streamlit as st
import requests

# URL de la aplicación Flask
FLASK_URL = 'http://localhost:5000'

# Página principal
st.title('Interfaz de Aplicación Flask')

# Cargar la página principal de Flask
if st.button('Cargar Página Principal'):
    response = requests.get(FLASK_URL)
    if response.status_code == 200:
        st.write(response.text)
    else:
        st.write('Error al cargar la página principal.')

# Realizar predicciones
st.sidebar.header('Realizar Predicción')
team_starting_value = st.sidebar.number_input('Team Starting Value', min_value=0.0, step=0.1)

if st.sidebar.button('Predecir'):
    payload = {'team_starting_value': team_starting_value}
    response = requests.post(f'{FLASK_URL}/predict', data=payload)
    if response.status_code == 200:
        st.write(response.text)
    else:
        st.write('Error al realizar la predicción.')

# Predicción de victoria
st.sidebar.header('Predicción de Victoria')
weapons = ['PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']
predictions = {weapon: st.sidebar.checkbox(weapon, value=False) for weapon in weapons}

if st.sidebar.button('Predecir Victoria'):
    payload = {weapon: int(predictions[weapon]) for weapon in weapons}
    response = requests.post(f'{FLASK_URL}/predict_victoria', data=payload)
    if response.status_code == 200:
        st.write(response.text)
    else:
        st.write('Error al realizar la predicción de victoria.')
