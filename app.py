from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        email_text = request.form["email_text"]

        # Transform input text using the trained TF-IDF vectorizer
        text_vector = vectorizer.transform([email_text])

        # Predict using SVM model
        result = model.predict(text_vector)[0]
        
        if result == "spam":
            prediction = "🚨 Spam Email"
        else:
            prediction = "✅ Not Spam (Ham Email)"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
