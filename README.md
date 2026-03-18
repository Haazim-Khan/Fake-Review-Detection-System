# 🛡️ Fake Review Detection System

## 📌 Overview

**Fake Review Detection System** is a machine learning-based web application that detects whether a product review is **genuine** or **fake**.
It uses Natural Language Processing (NLP) techniques and classification models to analyze review text and provide real-time predictions.

---

## 🚀 Features

* ✅ Detects fake vs genuine reviews
* ✅ Real-time prediction using Streamlit UI
* ✅ Uses TF-IDF for feature extraction
* ✅ Multiple ML models comparison
* ✅ Automatically selects the best model
* ✅ Clean and modular code structure

---

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**

* Lowercasing
* Removing special characters
* Stopword removal (NLTK)

2. **Feature Extraction**

* TF-IDF Vectorization (max_features=5000)

3. **Model Training**

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Random Forest

4. **Model Selection**

* Best model selected based on highest accuracy

5. **Prediction**

* User input → cleaned → vectorized → classified

---

## 📊 Dataset

* Total: **40,000 reviews**

* 20,000 Genuine (OR)
* 20,000 Fake (CG)
* Balanced dataset ensures fair model training

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```
---

## 👨‍💻 Author

**Haazim**

This project is for academic and educational
