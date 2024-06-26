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
        squareMeters = float(request.form.get('squareMeters'))
        basement = float(request.form.get('basement'))
        attic = float(request.form.get('attic'))
        cityCode = float(request.form.get('cityCode'))

        # Crear DataFrame con los datos recolectados
        data = {
            'squareMeters': [squareMeters],
            'basement': [basement],
            'attic': [attic],
            'cityCode': [cityCode],
        }
        data_df = pd.DataFrame(data)

        # Realizar predicciones
        prediction = model.predict(data_df)

        # Devolver la predicción como JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        # Devuelve un mensaje de error más útil
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
