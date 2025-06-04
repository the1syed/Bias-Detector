# 📰 Media Bias Detection App

This project is a **Streamlit-based web application** that detects **bias in media headlines** using multiple Natural Language Processing (NLP) techniques. Built to analyze how language might indicate political or emotional bias.

---

## 🚀 Live Demo

👉 [Click here to use the app](https://your-username-your-repo-name.streamlit.app/)  
*(replace with your actual Streamlit Cloud link)*

---

## 💡 Features

- 🔍 Input a news headline and detect possible bias
- ✅ Uses 4 different NLP models:
  - **VADER Sentiment Analysis**
  - **TextBlob Polarity**
  - **SpaCy Named Entity Recognition (NER)**
  - **BERT (Transformer-based Sentiment Classifier)**
- 📊 Returns model-wise classification
- 💬 Easy-to-use Streamlit interface

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [Transformers (HuggingFace)](https://huggingface.co/transformers/)
- [TextBlob](https://textblob.readthedocs.io/)
- [SpaCy](https://spacy.io/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)

---

## ⚙️ Installation

To run locally:

```bash
git clone https://github.com/your-username/media-bias-detector.git
cd media-bias-detector
pip install -r requirements.txt
streamlit run app.py
