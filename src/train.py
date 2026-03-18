import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import clean_text


def train_model():
    # 🔹 Get ROOT directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 🔹 Paths
    data_path = os.path.join(BASE_DIR, "data", "fake_reviews.csv")
    model_dir = os.path.join(BASE_DIR, "model")

    # 🔹 Create model folder if not exists
    os.makedirs(model_dir, exist_ok=True)

    # 🔹 Load dataset
    df = pd.read_csv(data_path)

    # 🔹 Label encoding
    df['label'] = df['label'].map({'OR': 1, 'CG': 0})

    print("Label Distribution:\n", df['label'].value_counts())

    # 🔹 Preprocessing
    df['clean_review'] = df['text_'].apply(clean_text)

    # 🔹 Feature extraction
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_review'])
    y = df['label']

    # 🔹 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 Models
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced')
    }

    best_model = None
    best_accuracy = 0

    # 🔁 Train all models
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy:", acc)

        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    # 🔹 Save paths
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    # 💾 Save
    pickle.dump(best_model, open(model_path, "wb"))
    pickle.dump(vectorizer, open(vectorizer_path, "wb"))

    print(f"\n✅ Best Model Accuracy: {best_accuracy}")
    print("✅ Model saved at:", model_path)


if __name__ == "__main__":
    train_model()