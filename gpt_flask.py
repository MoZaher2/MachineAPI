import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image
from flask_cors import CORS  # Import CORS extension

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Firebase
cred = credentials.Certificate("D:/Omar/gym-api-zaher-firebase-adminsdk-9y6h3-4f96a1dbed.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load your AI model
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Define a function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            index = np.argmax(prediction)
            class_name = class_names[index][0]
            confidence_score = prediction[0][index]
            return jsonify({'Muscle_Level': class_name, "Score": str(confidence_score)})

if __name__ == '__main__':
    app.run(debug=True)
