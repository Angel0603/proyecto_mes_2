from flask import Flask, request, render_template, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar el modelo entrenado (Asegúrate de que el modelo está guardado en formato adecuado para Keras)
model = load_model('modelo_paris.h5')

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
        prediction = model.predict(data_df)[0][0]  # Ajustar según la forma de salida del modelo

        # Devolver la predicción como JSON
        return jsonify({'prediction': prediction})
    except Exception as e:
        # Devuelve un mensaje de error más útil
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
