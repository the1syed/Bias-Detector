import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import spacy
import plotly.graph_objects as go

# Load models
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()
bert_pipeline = pipeline("sentiment-analysis")

# Streamlit config
st.set_page_config(page_title="Media Bias Detector", layout="wide")

# Header UI
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(to right, #4facfe, #00f2fe); border-radius: 12px;'>
        <h1 style='color: white;'>📰 Media Bias Detection</h1>
        <p style='color: white; font-size: 18px;'>Analyze political or emotional bias in media headlines using powerful NLP models.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("## ✏️ Input Headline")
headline = st.text_input("Enter a news headline...", placeholder="Example: 'Government gives tax relief to rich companies'")

# Bias decision logic
def is_biased(v_label, v_score, t_label, t_score, b_label, b_score, threshold=0.3):
    bias_votes = 0
    reasons = []

    if v_label != "Neutral" and abs(v_score) >= threshold:
        bias_votes += 1
        reasons.append(f"VADER marked it **{v_label}** with score {v_score:.2f}")
    if t_label != "Neutral" and abs(t_score) >= threshold:
        bias_votes += 1
        reasons.append(f"TextBlob marked it **{t_label}** with score {t_score:.2f}")
    if b_label != "NEUTRAL" and b_score >= threshold:
        bias_votes += 1
        reasons.append(f"BERT marked it **{b_label}** with confidence {b_score:.2f}")

    verdict = "Biased" if bias_votes >= 2 else "Neutral / Unbiased"
    return verdict, reasons

# When headline is entered
if headline:
    # Model predictions
    blob = TextBlob(headline)
    blob_polarity = blob.sentiment.polarity
    blob_label = "Positive" if blob_polarity > 0.1 else "Negative" if blob_polarity < -0.1 else "Neutral"

    vader = analyzer.polarity_scores(headline)
    vader_compound = vader['compound']
    vader_label = "Positive" if vader_compound > 0.05 else "Negative" if vader_compound < -0.05 else "Neutral"

    bert_output = bert_pipeline(headline)[0]
    bert_label = bert_output['label']
    bert_score = round(bert_output['score'], 2)

    doc = nlp(headline)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Bias check
    bias_result, reasons = is_biased(
        vader_label, vader_compound,
        blob_label, blob_polarity,
        bert_label.upper(), bert_score
    )

    # Display metrics
    st.markdown("### 🧠 Model Predictions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("VADER", vader_label, round(vader_compound, 3))
        st.caption("Rule-based sentiment")

    with col2:
        st.metric("TextBlob", blob_label, round(blob_polarity, 3))
        st.caption("Lexicon-based polarity")

    with col3:
        st.metric("BERT", bert_label, f"{bert_score*100:.1f}%")
        st.caption("Transformer-based sentiment")

    # Show bar graph
    st.markdown("### 📊 Confidence Scores")
    fig = go.Figure(data=[
        go.Bar(name="VADER", x=["VADER"], y=[abs(vader_compound)], marker_color='#1f77b4'),
        go.Bar(name="TextBlob", x=["TextBlob"], y=[abs(blob_polarity)], marker_color='#ff7f0e'),
        go.Bar(name="BERT", x=["BERT"], y=[bert_score], marker_color='#2ca02c')
    ])
    fig.update_layout(barmode='group', title='Model Confidence/Polarity', yaxis=dict(title='Score'), height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Bias judgment
    st.markdown("### 🧾 Final Bias Judgment")
    if bias_result == "Biased":
        st.error("🚨 This headline appears to be **Biased**.")
    else:
        st.success("✅ This headline appears to be **Neutral / Unbiased**.")

    # Explanation
    with st.expander("🧐 Why is this biased / unbiased?"):
        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown("No strong evidence of polarity was found.")

    # NER
    with st.expander("📌 Named Entities (NER from SpaCy)"):
        if entities:
            for ent, label in entities:
                st.write(f"🔹 `{ent}` → {label}")
        else:
            st.write("No named entities detected.")

    st.success("✅ Analysis complete.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Built with ❤️ by <b>Syed Raza Ali</b> | MCA | Amity Noida</p>", unsafe_allow_html=True)
