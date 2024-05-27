# MoodScan: Real-Time Facial Emotion Recognition Web Application

MoodScan is a web application that utilizes Convolutional Neural Networks (CNN) to recognize facial emotions in real-time. It provides users with the ability to upload images for emotion detection analysis or use their webcam for live emotion recognition.

## Model Training

- Trained a CNN model to classify 7 different emotions: angry, sad, happy, fear, surprise, neutral and disgust.
- Achieved an accuracy of 74% on the validation dataset.

## Testing and Evaluation

- Tested the trained model with unseen images to evaluate its performance.
- Plotted a confusion matrix to visualize the model's performance.
- Calculated the classification report to assess precision, recall, F1-score, and ROC-AUC for comprehensive evaluation.

## Real-Time Detection

- Implemented real-time facial emotion detection using OpenCV to capture video feed from the device webcam.
- Utilized the trained CNN model to predict emotions in real-time.

## MoodScan Web Application

- Developed the MoodScan website with the following functionalities:
  - Image upload: Users can upload grayscale images, and the system will analyze facial expressions to determine the emotion.
  - Real-time detection: Users can activate their webcam to detect emotions in real-time and receive instant results.

## Technologies Used

- Python
- TensorFlow/Keras (for machine learning model)
- OpenCV (for real-time detection)
- Flask (web framework)
- HTML/CSS/Bootstrap (for front-end styling)



