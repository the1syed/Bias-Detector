import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Load BERT classifier once
bert_classifier = pipeline('sentiment-analysis')

# Functions
def vader_bias_score(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound > 0.05:
        return 'Possibly Biased (Positive)'
    elif compound < -0.05:
        return 'Possibly Biased (Negative)'
    else:
        return 'Neutral/Unbiased'

def textblob_bias_score(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Possibly Biased (Positive)'
    elif polarity < -0.1:
        return 'Possibly Biased (Negative)'
    else:
        return 'Neutral/Unbiased'

def spacy_bias_score(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    if entities:
        return f"Entities detected: {entities}"
    else:
        return "No strong entities detected"

def bert_bias_score(text):
    result = bert_classifier(text)[0]
    return f"{result['label']} ({round(result['score'], 2)} confidence)"

# Streamlit UI
st.title("📰 Media Bias Detection App")
headline = st.text_input("Enter a news headline:")

if headline:
    st.write("## 🔍 Analysis Results:")
    st.write("**VADER Result:**", vader_bias_score(headline))
    st.write("**TextBlob Result:**", textblob_bias_score(headline))
    st.write("**SpaCy Result:**", spacy_bias_score(headline))
    st.write("**BERT Result:**", bert_bias_score(headline))
