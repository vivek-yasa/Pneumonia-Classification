from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from flask import Flask, render_template, request, redirect, Response
from PIL import Image
from io import BytesIO

# Keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Function to check the file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path1 = 'Xception.h5'  # Load .h5 Model
model = load_model(model_path1, compile=False)

def model_predict1(image, model):
    print("Predicted")
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    result = np.argmax(model.predict(image))
    print(result)

    if result == 0:
        return "NORMAL!", "result.html"
    elif result == 1:
        return "PNEUMONIA!", "result.html"

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']  # Fetch input
        if file and allowed_file(file.filename):
            print("@@ Input posted = ", file.filename)

            # Load image from file without saving to static folder
            image = Image.open(file.stream)
            image = image.resize((128, 128))  # Resize image to the required size
            pred, output_page = model_predict1(image, model)

            return render_template(output_page, pred_output=pred)

    return redirect(request.url)

# Uncomment this to run the app locally
# if __name__ == '__main__':
#     app.run(debug=False)

