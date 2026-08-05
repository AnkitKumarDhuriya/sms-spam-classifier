import numpy as np
import pandas as pd
import pickle
import import_ipynb
from utils import transform_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed_spam.csv")
#print(df.isnull().sum())
df = df.dropna(subset=["transformed_text"])

# Features & Labels
X = df["transformed_text"]
y = df["label"]

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Vectorization
tfidf = TfidfVectorizer(max_features=3000)

X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LinearSVC": LinearSVC(),
    
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

results_df = pd.DataFrame(results).T
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print(results_df)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/model.pkl", "wb") as f:
    pickle.dump(models["LinearSVC"], f)


#Testing Phase
# Load Vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Load Model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

messages = [
    "Congratulations! You have won a free iPhone.",
    "Call me when you reach home.",
    "Win ₹50000 cash now. Click the link.",
    "Where are you?"
]


for msg in messages:
    clean_msg = transform_text(msg)
    print("Original :", msg)
    print("Processed:", clean_msg)

    vector = tfidf.transform([clean_msg])
    pred = model.predict(vector)[0]

    print("Prediction :", "Spam" if pred == 1 else "Ham")
    print("-"*50)