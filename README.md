# Spam Detection using SVM

This project implements a **Support Vector Machine (SVM)** model to classify SMS messages as spam or not spam.  
It uses **TF-IDF Vectorization** for text feature extraction and Flask for a simple web interface.

## Features

- Machine learning classification using SVM
- Text preprocessing with TF-IDF
- Simple Flask-based web app for predictions
- Small sample dataset included

## Project Structure

Spam Detection/
│-- app.py # Flask web app
│-- train_model.py # Model training script
│-- spam_data.csv # Dataset
│-- svm_spam_model.pkl # Saved trained model
│-- tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
│-- templates/
│ └── index.html # HTML template
│-- static/
│ └── style.css # CSS styles
│-- requirements.txt # Dependencies
│-- README.md # Project documentation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Spam_detection.git
   cd Spam_detection```
2. Install the dependencies:
```pip install -r requirements.txt
```
##Usage

1. Train the model:
```python train_model.py```
2. Start the flask application:
```python app.py```
3. Open your browser and go to:
```http://127.0.0.1:5000```

