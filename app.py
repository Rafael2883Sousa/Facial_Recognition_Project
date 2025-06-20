
import os
from io import BytesIO
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)
model = load_model("modelo/modelo.h5")
img_size = (160, 160)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    img = load_img(BytesIO(file.read()), target_size=img_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
 
    pred = model.predict(img_array)[0][0]
    resultado = "É o Indivíduo" if pred > 0.6 else "Não é o Indivíduo"

    return jsonify({'resultado': resultado, 'confiança': f"{pred:.2f}"})


if __name__ == '__main__':
    app.run(debug=True)
