from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('modelo_paris.pkl')

@app.route('/')
def home():
    # Servir la página con el formulario
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recolectar valores del formulario
        squareMeters = float(request.form.get('squareMeters', 0))
        garage = float(request.form.get('garage', 0))
        basement = float(request.form.get('basement', 0))
        attic = float(request.form.get('attic', 0))
        floors = float(request.form.get('floors', 0))
        cityCode = float(request.form.get('cityCode', 0))

        # Crear DataFrame con los datos recolectados
        data = {
            'squareMeters': [squareMeters],
            'garage': [garage],
            'basement': [basement],
            'attic': [attic],
            'floors': [floors],
            'cityCode': [cityCode],
        }
        data_df = pd.DataFrame(data)

        # Realizar predicciones
        prediction = model.predict(data_df)

        # Devolver la predicción como JSON
        return jsonify({'prediction': prediction[0].item()})
    except Exception as e:
        # Devuelve un mensaje de error más útil
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
