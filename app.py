import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from random import randint
import pandas as pd

# -----------------------------
# 📦 Load Models & Vectorizer
# -----------------------------
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# 📊 Load Dataset
# -----------------------------
df = pd.read_csv("Stock Headlines.csv", encoding='ISO-8859-1')
df_copy = df.copy()

sample_test = df_copy[df_copy['Date'] > '20141231']
sample_test.reset_index(inplace=True)
sample_test = sample_test['Top1']

# -----------------------------
# 🔥 Prediction Function
# -----------------------------
def predict_text(text, model_choice):

    # Preprocessing
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    text = ' '.join(words)

    # Vectorization
    vector = cv.transform([text]).toarray()

    # Predictions
    pred_lr = lr_model.predict(vector)[0]
    pred_rf = rf_model.predict(vector)[0]
    pred_nb = nb_model.predict(vector)[0]

    # Probabilities
    proba_lr = lr_model.predict_proba(vector)[0]
    proba_rf = rf_model.predict_proba(vector)[0]
    proba_nb = nb_model.predict_proba(vector)[0]

    # Store all predictions
    all_preds = {
        "Logistic Regression": pred_lr,
        "Random Forest": pred_rf,
        "Naive Bayes": pred_nb
    }

    # -----------------------------
    # 🎯 Model Selection
    # -----------------------------
    if model_choice == "Logistic Regression":
        return pred_lr, max(proba_lr), all_preds

    elif model_choice == "Random Forest":
        return pred_rf, max(proba_rf), all_preds

    elif model_choice == "Naive Bayes":
        return pred_nb, max(proba_nb), all_preds

    # -----------------------------
    # 🔥 Weighted Ensemble
    # -----------------------------
    else:
        votes = [pred_lr, pred_rf, pred_nb]
        final_pred = round(sum(votes) / len(votes))

        confidence = (
            max(proba_lr) +
            max(proba_rf) +
            max(proba_nb)
        ) / 3

        return final_pred, confidence, all_preds


# -----------------------------
# 🎛️ Streamlit UI
# -----------------------------
st.title("📈 Stock Sentiment Analysis")

# Model selection
model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest", "Naive Bayes", "Ensemble (Weighted)"]
)

# Input type selection
option = st.radio(
    "Choose Input Type",
    ["Manual Input", "Random News"]
)

# -----------------------------
# ✍️ Manual Input
# -----------------------------
if option == "Manual Input":

    user_input = st.text_area("Enter News Headline")

    if st.button("Predict"):

        if user_input.strip() != "":

            pred, confidence, all_preds = predict_text(user_input, model_choice)

            # Final result
            if pred == 0:
                st.success(f"📊 Positive Sentiment (Stock may go UP)\nConfidence: {confidence:.2f}")
            else:
                st.error(f"📉 Negative Sentiment (Stock may go DOWN)\nConfidence: {confidence:.2f}")

            # -----------------------------
            # 📦 Conclusion Box
            # -----------------------------
            st.subheader("📦 Model Predictions Summary")

            for model, value in all_preds.items():
                label = "UP 📈" if value == 0 else "DOWN 📉"
                st.write(f"{model}: {label}")

            up_votes = list(all_preds.values()).count(0)
            down_votes = list(all_preds.values()).count(1)

            st.info(f"""
👉 Majority Voting Result:
- UP votes: {up_votes}
- DOWN votes: {down_votes}

Final prediction is based on majority voting.
""")

        else:
            st.warning("Please enter some text")


# -----------------------------
# 🎲 Random News
# -----------------------------
else:

    if st.button("Generate & Predict"):

        row = randint(0, sample_test.shape[0] - 1)
        sample_news = sample_test[row]

        st.write("📰 News:", sample_news)

        pred, confidence, all_preds = predict_text(sample_news, model_choice)

        # Final result
        if pred == 0:
            st.success(f"📊 Prediction: Stock may go UP\nConfidence: {confidence:.2f}")
        else:
            st.error(f"📉 Prediction: Stock may go DOWN\nConfidence: {confidence:.2f}")

        # -----------------------------
        # 📦 Conclusion Box
        # -----------------------------
        st.subheader("📦 Model Predictions Summary")

        for model, value in all_preds.items():
            label = "UP 📈" if value == 0 else "DOWN 📉"
            st.write(f"{model}: {label}")

        up_votes = list(all_preds.values()).count(0)
        down_votes = list(all_preds.values()).count(1)

        st.info(f"""
👉 Majority Voting Result:
- UP votes: {up_votes}
- DOWN votes: {down_votes}

Final prediction is based on majority voting.
""")
