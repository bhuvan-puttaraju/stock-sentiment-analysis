# 📈 Stock Sentiment Analysis using NLP & Machine Learning

Predicting stock market movement (📈 UP / 📉 DOWN) using financial news headlines with Natural Language Processing and ensemble machine learning models.

## 🚀 Project Overview

This project builds an end-to-end NLP pipeline to analyze stock-related news headlines and predict market sentiment.

The system:

* Cleans and preprocesses raw text
* Converts text into numerical features using TF-IDF
* Trains multiple ML models
* Combines predictions using ensemble learning
* Deploys the model using Streamlit

## 🎯 Problem Statement

Stock markets are heavily influenced by news sentiment.
The goal of this project is to:

* Extract sentiment from financial headlines
* Classify news as Positive (Stock Up) or Negative (Stock Down)
* Improve robustness using ensemble learning

## 🧠 Models Used

* Logistic Regression
* Multinomial Naive Bayes
* Random Forest
* Majority Voting Ensemble

## 🔍 NLP Pipeline

1. Text Cleaning (Regex)
2. Lowercasing
3. Tokenization
4. Stopword Removal
5. Stemming (PorterStemmer)
6. TF-IDF Vectorization (Unigrams + Bigrams)

```python
TfidfVectorizer(max_features=10000, ngram_range=(1,2))
```

## ⚙️ Hyperparameter Tuning

Used **RandomizedSearchCV** with cross-validation to optimize model performance efficiently.

## 📈 Ensemble Strategy

Predictions from multiple models are combined using majority voting to improve stability and reduce bias.

Confidence score is calculated from model probabilities.

## 🖥️ Deployment (Streamlit App)

### Features:

* Manual headline input
* Random news testing
* Model selection option
* Ensemble prediction summary
* Confidence score display

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Tech Stack

* Python
* Scikit-learn
* NLTK
* Pandas
* Streamlit
* Git & GitHub

## 📂 Project Structure

stock-sentiment-analysis/
│
├── app.py
├── Stock Sentiment Analysis.ipynb
├── requirements.txt
├── .gitignore
└── README.md

## 🔮 Future Improvements

* Implement Stacking Ensemble
* Use XGBoost / LightGBM
* Upgrade to BERT embeddings
* Integrate live financial news API

## 👤 Author

**Bhuvan Puttaraju**
Data Science Enthusiast
