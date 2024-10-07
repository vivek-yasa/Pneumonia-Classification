# Required Libraries
from flask import Flask, render_template, request, redirect
import os
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# Allowed file types for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model at the start of the application
model_path = os.getenv("MODEL_PATH", "Desktop/Pneumonia_1/xception.h5")
model = load_model(model_path, compile=False)

# Prediction function
def model_predict(image, model):
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(image))
    return ("NORMAL!", "result.html") if result == 0 else ("PNEUMONIA!", "result.html")

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            image = Image.open(file.stream).resize((128, 128))
            pred, output_page = model_predict(image, model)
            return render_template(output_page, pred_output=pred)
    return redirect(request.url)

# Run in debug mode during development; otherwise, configure for deployment
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)  # Uncomment for testing locally
