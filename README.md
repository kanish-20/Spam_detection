# Spam Detection using SVM

This project implements a **Support Vector Machine (SVM)** model to classify SMS messages as spam or not spam.  
It uses **TF-IDF Vectorization** for text feature extraction and Flask for a simple web interface.

## Features
- Machine learning classification using SVM
- Text preprocessing with TF-IDF
- Simple Flask-based web app for predictions
- Small sample dataset included

## Project Structure
```
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

```

## Installation

1. Clone the repository:
    ```
   git clone https://github.com/yourusername/Spam_detection.git
   cd Spam_detection
    ```
2.Install the dependencies:
    ```
   pip install -r requirements.txt
    ```

## Usage
1. Train the model:
   ```
   python train_model.py
   ```
2. Start the Flask application:
```
python app.py
```
3. Open your browser and go to:
```
http://127.0.0.1:5000
```
### 4.Enter a message and click "Predict" to see if it is spam or not.

## Screeshots

![WhatsApp Image 2025-08-13 at 12 22 05](https://github.com/user-attachments/assets/231c0e8b-1655-4695-b658-effc92d78020)
![WhatsApp Image 2025-08-13 at 12 22 05 (1)](https://github.com/user-attachments/assets/cc452cf3-4a56-447c-a24f-ca52145c6817)
![WhatsApp Image 2025-08-13 at 12 22 05 (2)](https://github.com/user-attachments/assets/1b50cd7c-660b-4184-ba07-09fc10118691)



