{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "45397c72-0764-41f4-9ff5-1025d568c1ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-02-18 13:37:13.072 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.530 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run D:\\Anaconda Navigator\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-02-18 13:37:13.531 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.531 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.532 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.533 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.533 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.534 Session state does not function when running a script without `streamlit run`\n",
      "2026-02-18 13:37:13.535 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.535 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.536 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.537 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.538 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.538 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-02-18 13:37:13.539 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import re\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.stem import PorterStemmer\n",
    "from random import randint\n",
    "import pandas as pd\n",
    "\n",
    "# Load models\n",
    "lr_model = pickle.load(open('lr_model.pkl', 'rb'))\n",
    "rf_model = pickle.load(open('rf_model.pkl', 'rb'))\n",
    "nb_model = pickle.load(open('nb_model.pkl', 'rb'))\n",
    "cv = pickle.load(open('vectorizer.pkl', 'rb'))\n",
    "\n",
    "ps = PorterStemmer()\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(\"Stock Headlines.csv\", encoding='ISO-8859-1')\n",
    "df_copy = df.copy()\n",
    "\n",
    "# Filter test data\n",
    "sample_test = df_copy[df_copy['Date'] > '20141231']\n",
    "sample_test.reset_index(inplace=True)\n",
    "sample_test = sample_test['Top1']\n",
    "\n",
    "# 🔥 Preprocessing + prediction\n",
    "def predict_text(text):\n",
    "    text = re.sub('[^a-zA-Z]', ' ', text)\n",
    "    text = text.lower()\n",
    "    words = text.split()\n",
    "    words = [word for word in words if word not in stop_words]\n",
    "    words = [ps.stem(word) for word in words]\n",
    "    text = ' '.join(words)\n",
    "\n",
    "    vector = cv.transform([text]).toarray()\n",
    "\n",
    "    # Predictions from all models\n",
    "    pred_lr = lr_model.predict(vector)[0]\n",
    "    pred_rf = rf_model.predict(vector)[0]\n",
    "    pred_nb = nb_model.predict(vector)[0]\n",
    "\n",
    "    # 🔥 Majority Voting\n",
    "    final_pred = round((pred_lr + pred_rf + pred_nb) / 3)\n",
    "\n",
    "    return final_pred\n",
    "\n",
    "# UI\n",
    "st.title(\"📈 Stock Sentiment Analysis\")\n",
    "\n",
    "option = st.radio(\n",
    "    \"Choose Input Type\",\n",
    "    [\"Manual Input\", \"Random News\"]\n",
    ")\n",
    "\n",
    "# 🔹 Manual Input\n",
    "if option == \"Manual Input\":\n",
    "\n",
    "    user_input = st.text_area(\"Enter News Headline\")\n",
    "\n",
    "    if st.button(\"Predict\"):\n",
    "\n",
    "        if user_input.strip() != \"\":\n",
    "\n",
    "            pred = predict_text(user_input)\n",
    "\n",
    "            # Label mapping\n",
    "            if pred == 0:\n",
    "                st.success(\"📊 Positive Sentiment (Stock may go UP)\")\n",
    "            else:\n",
    "                st.error(\"📉 Negative Sentiment (Stock may go DOWN)\")\n",
    "\n",
    "        else:\n",
    "            st.warning(\"Please enter some text\")\n",
    "\n",
    "# 🔹 Random News\n",
    "else:\n",
    "\n",
    "    if st.button(\"Generate & Predict\"):\n",
    "\n",
    "        row = randint(0, sample_test.shape[0] - 1)\n",
    "        sample_news = sample_test[row]\n",
    "\n",
    "        st.write(\"📰 News:\", sample_news)\n",
    "\n",
    "        pred = predict_text(sample_news)\n",
    "\n",
    "        if pred == 0:\n",
    "            st.success(\"📊 Prediction: Stock may go UP\")\n",
    "        else:\n",
    "            st.error(\"📉 Prediction: Stock may go DOWN\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0a607e36-8fa7-41ae-b525-49a52d8c1c58",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to C:\\Users\\Bhuvan\n",
      "[nltk_data]     P\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('stopwords')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
