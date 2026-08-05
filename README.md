# 📩 SMS Spam Classifier using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based web application that classifies SMS messages as **Spam** or **Ham (Not Spam)**. It uses Natural Language Processing (NLP) techniques to preprocess text and a **Linear Support Vector Classifier (LinearSVC)** to classify SMS messages. A simple Streamlit interface allows users to enter an SMS message and instantly check whether it is spam or not.

---

## 🎯 Objectives

- Detect spam SMS messages automatically.
- Apply Natural Language Processing (NLP) techniques for text preprocessing.
- Convert text into numerical features using TF-IDF Vectorization.
- Train an accurate Machine Learning classification model.
- Build an interactive web application using Streamlit.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Pickle

---

## 📂 Project Structure

```text
sms-spam-classifier/
│
├── app.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
├── Project Synopsis.pdf
├── .gitignore
│
├── data/
│   ├── spam.csv
│   └── processed_spam.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
└── outputs/
```

---

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset**, which contains labeled SMS messages classified as **Spam** and **Ham (Not Spam)**.

- **Total Messages:** 5,500+
- **Classes:**
  - Spam
  - Ham

---

## ⚙️ Workflow

1. Load the SMS Spam Collection Dataset.
2. Perform data preprocessing:
   - Convert text to lowercase.
   - Remove punctuation and special characters.
   - Remove stopwords.
   - Apply stemming.
3. Convert the processed text into numerical features using **TF-IDF Vectorization**.
4. Train and evaluate multiple Machine Learning models.
5. Select **Linear Support Vector Classifier (LinearSVC)** as the final model based on performance.
6. Save the trained model and TF-IDF vectorizer using Pickle.
7. Build a Streamlit web application for real-time SMS spam prediction.

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/AnkitKumarDhuriya/sms-spam-classifier.git
```

### Move to the project directory

```bash
cd sms-spam-classifier
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## 💻 Demo

1. Enter an SMS message in the input box.
2. Click the **Predict** button.
3. The application will classify the message as:

- ✅ Ham (Not Spam)
- 🚫 Spam

---

## 📈 Future Improvements

- Email Spam Detection
- Multilingual Spam Classification
- Deep Learning Models (LSTM, GRU, BERT)
- Real-time API Integration
- Cloud Deployment

---

## 👨‍💻 Author

**Ankit Dhuriya**

B.Tech CSE (Artificial Intelligence & Machine Learning)

DAV Institute of Engineering & Technology (DAVIET), Jalandhar

GitHub: https://github.com/AnkitKumarDhuriya

---

## 📄 License

This project is created for educational and learning purposes.