import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("spam_data.csv")

print("📦 Dataset shape before cleanup:", df.shape)

# 2. Drop rows with missing values
df.dropna(subset=['message', 'label'], inplace=True)

print("✅ Dataset shape after cleanup:", df.shape)
print("🔍 First few rows:\n", df.head())

# 3. Features (X) and Labels (y)
X = df['message']
y = df['label']

# 4. Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# 6. Train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 9. Save model & vectorizer
joblib.dump(model, "svm_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n Model and vectorizer saved successfully!")
