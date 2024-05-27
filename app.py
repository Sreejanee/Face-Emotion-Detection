from flask import Flask, render_template, request, redirect, url_for, flash
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model('emotiondetector.h5') 

# The emotion labels and corresponding colors
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_colors = {
    'angry': 'red',
    'disgust': 'green',
    'fear': 'purple',  
    'happy': 'black',
    'neutral': 'blue',
    'sad': 'grey',
    'surprise': 'orange'
}

# Preprocess uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Load as grayscale
    img = img.resize((48, 48))  # Resize to 48x48
    img = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Checking if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            # Saving the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Preprocess the image
            img = preprocess_image(filename)

            # Make prediction
            prediction = model.predict(img)
            predicted_label = labels[np.argmax(prediction)]
            color = label_colors[predicted_label]

            return render_template('index.html', message=f'Prediction: {predicted_label}', color=color, filename=filename)

    return render_template('index.html')

@app.route('/start-realtime-detection', methods=['POST'])
def start_realtime_detection():
    # Run the realtimedetection.py script
    subprocess.Popen(['python', 'realtimedetection.py'])
    flash('Real-time detection started. Check your webcam.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.secret_key = 'sree@123'
    app.run(debug=True)
