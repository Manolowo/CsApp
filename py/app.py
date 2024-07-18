from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__, static_url_path='/assets', static_folder='static')

MODEL_FOLDER1 = 'models/decisiontree1/'
MODEL_FOLDER2 = 'models/decisiontree2/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    team_starting_value = float(request.form['team_starting_value'])

    predictions = {}
    for weapon in ['PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']:
        model_filename = f'{MODEL_FOLDER1}model_{weapon}.pkl'
        logistic_model = joblib.load(model_filename)

        input_data = [[team_starting_value]]
        prediction = logistic_model.predict(input_data)
        predictions[weapon] = prediction[0]

    print(predictions)

    equip_pred = [predictions[weapon] for weapon in ['PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']]
    equip_pred_array = np.array(equip_pred).reshape(1, -1)

    decision_tree_model = joblib.load(f'{MODEL_FOLDER2}model_RoundWinner.pkl')
    round_winner_prediction = decision_tree_model.predict(equip_pred_array)

    return render_template('result.html', predictions=predictions, round_winner=round_winner_prediction[0])

@app.route('/predict_victoria', methods=['POST'])
def predict_victoria():
    # Extraer los valores de los checkboxes
    weapons = ['PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']
    equip_pred = [int(request.form.get(weapon, 0)) for weapon in weapons]
    
    # Crear un array numpy a partir de las predicciones
    equip_pred_array = np.array(equip_pred).reshape(1, -1)

    # Cargar el modelo de árbol de decisiones y hacer la predicción
    decision_tree_model = joblib.load(f'{MODEL_FOLDER2}model_RoundWinner.pkl')
    round_winner_prediction = decision_tree_model.predict(equip_pred_array)
    
    return render_template('result2.html', round_winner=round_winner_prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
