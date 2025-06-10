# OIB-SIP--task-3
data science project
#  Spam Detector Model Using Machine Learning

This project is a practical implementation of a *Spam Detection Model* using machine learning classification techniques. The goal is to identify whether a given message (usually SMS or email) is *spam* or *not spam (ham)* based on its content.

---

##  Objective

To build a text classification model that can:
Process and clean message text
Convert text into numerical features
Train a model to detect spam messages with high accuracy

---

##  Dataset Overview

*Source*: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
*Records*: ~5,500 SMS messages
*Columns*:
  - label: spam or ham
  - message: text content of the SMS

 Example:

| label | message                             |
|-------|--------------------------------------|
| ham   | Hey! Are we still meeting today?     |
| spam  | WINNER! You have won a $1000 prize!  |

---

##  Tools & Libraries Used

Python
Pandas, NumPy
Scikit-learn
NLTK (Natural Language Toolkit)
Matplotlib / Seaborn (for visualizations)

---

##  Preprocessing Steps

Lowercasing
Removing punctuation and stopwords
Tokenization
Lemmatization
Text vectorization using *TF-IDF* (Term Frequency–Inverse Document Frequency)

---

##  Models Trained

*Multinomial Naive Bayes*  (best performance)
Logistic Regression
Support Vector Machine (SVM)
Decision Tree

---

##  Evaluation Metrics

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

 *Naive Bayes* performed the best, achieving high accuracy and precision for spam detection.

---

##  Project Structure
spam-detector/
│
├── spam.csv                      # Dataset
├── spam_detector.ipynb           # Jupyter notebook with full workflow
├── vectorizer.pkl                # Saved TF-IDF vectorizer
├── model.pkl                     # Trained ML model
└── README.md                     # Project overview
