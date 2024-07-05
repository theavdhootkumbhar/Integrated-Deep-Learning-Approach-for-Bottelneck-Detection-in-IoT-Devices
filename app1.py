import os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

app = Flask(__name__)

# Load your pre-trained model
model = load_model("lstm_model_checkpoint.h5")

# Load the trained hybrid model
model_path = "lstm_model_checkpoint.h5"
try:
    hybrid_model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    hybrid_model = None

def preprocess_user_file(file_path):
    try:
        user_data = pd.read_csv(file_path)
        for col in user_data.columns[:-1]:
            user_data[col] = (user_data[col] - user_data[col].mean()) / user_data[col].std()
        user_data_array = user_data.values
        user_data_reshaped = user_data_array.reshape(user_data_array.shape[0], user_data_array.shape[1], 1)
        return user_data_reshaped, None
    except Exception as e:
        print(f"Error preprocessing file: {e}")
        return None, None

def predict_sequential_data(data):
    try:
        return hybrid_model.predict(data)
    except Exception as e:
        print(f"Error predicting data: {e}")
        return None

def predict_class(predictions):
    try:
        attack_threshold = 0.5
        if np.max(predictions) < attack_threshold:
            return "Attack"
        else:
            return "Normal"
    except Exception as e:
        print(f"Error predicting class: {e}")
        return "Unknown"

def predict_user_file(file_path):
    user_data_preprocessed, additional_features = preprocess_user_file(file_path)
    if user_data_preprocessed is None:
        return "Error in preprocessing"
    
    predictions = predict_sequential_data(user_data_preprocessed)
    if predictions is None:
        return "Error in prediction"
    
    predicted_class = predict_class(predictions)
    return predicted_class


@app.route('/')
def Home():
    return render_template('Home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction')
def prediction():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('result.html', predicted_class="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', predicted_class="No selected file")

    if file:
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)
        
        predicted_class = predict_user_file(file_path)
        return render_template('result.html', predicted_class=predicted_class)

    return redirect(url_for('prediction'))


if __name__ == '__main__':
    app.run(debug=True)
