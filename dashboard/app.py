import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline
import re

st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None
    )

model = load_model()

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def analyze_text(text):
    sentences = split_sentences(text)
    if not sentences:
        sentences = [text]
    
    all_scores = {'positive': [], 'negative': [], 'neutral': []}
    sentence_results = []
    
    for sentence in sentences:
        result = model(sentence[:512])[0]
        s = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        for r in result:
            label = r['label'].lower()
            if label in s:
                s[label] = round(float(r['score']) * 100, 2)
        
        all_scores['positive'].append(s['positive'])
        all_scores['negative'].append(s['negative'])
        all_scores['neutral'].append(s['neutral'])
        sentence_results.append({'sentence': sentence, **s})
    
    # Average scores
    avg = {
        'positive': round(sum(all_scores['positive']) / len(all_scores['positive']), 2),
        'negative': round(sum(all_scores['negative']) / len(all_scores['negative']), 2),
        'neutral': round(sum(all_scores['neutral']) / len(all_scores['neutral']), 2)
    }
    
    # Normalize to 100%
    total = sum(avg.values())
    avg = {k: round(v / total * 100, 2) for k, v in avg.items()}
    
    best_label = max(avg, key=avg.get)
    return avg, best_label, sentence_results

# UI
st.title("🧠 NLP Sentiment Analyzer")
st.markdown("**Sentence-level multi-class sentiment analysis — Positive / Negative / Neutral**")
st.markdown("---")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📝 Enter Text")
    user_input = st.text_area(
        "Paste your paragraph here (up to 1000 words)",
        height=200,
        placeholder="Type or paste any review, feedback, or paragraph here..."
    )
    word_count = len(user_input.split()) if user_input else 0
    st.caption(f"Word count: {word_count}/1000")
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    st.subheader("ℹ️ About")
    st.info("""
    **Model:** RoBERTa (Cardiff NLP)
    
    **How it works:**
    - Splits text into sentences
    - Analyzes each sentence
    - Averages scores for final result
    
    **Classes:**
    - 🟢 Positive
    - 🔴 Negative
    - 🟡 Neutral
    """)

if analyze_btn and user_input:
    with st.spinner("Analyzing sentence by sentence..."):
        scores, best_label, sentence_results = analyze_text(user_input)

    st.markdown("---")
    st.subheader("📊 Overall Sentiment")

    if best_label == 'positive':
        st.success(f"### ✅ POSITIVE — {scores['positive']}% Confidence")
    elif best_label == 'negative':
        st.error(f"### ❌ NEGATIVE — {scores['negative']}% Confidence")
    else:
        st.warning(f"### 😐 NEUTRAL — {scores['neutral']}% Confidence")

    # Gauge meters
    def gauge(value, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': title, 'font': {'size': 16}},
            number={'suffix': '%', 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0.0, 100.0]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': '#f5f5f5'},
                    {'range': [30, 70], 'color': '#eeeeee'},
                    {'range': [70, 100], 'color': '#e0e0e0'}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=0, l=20, r=20))
        return fig

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(gauge(scores['positive'], '🟢 Positive', '#43A047'),
                        use_container_width=True)
    with col2:
        st.plotly_chart(gauge(scores['negative'], '🔴 Negative', '#E53935'),
                        use_container_width=True)
    with col3:
        st.plotly_chart(gauge(scores['neutral'], '🟡 Neutral', '#FF9800'),
                        use_container_width=True)

    # Detailed scores
    st.markdown("---")
    st.subheader("📋 Detailed Scores")
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Positive", f"{scores['positive']}%")
    col2.metric("🔴 Negative", f"{scores['negative']}%")
    col3.metric("🟡 Neutral", f"{scores['neutral']}%")

    # Sentence level breakdown
    st.markdown("---")
    st.subheader("🔍 Sentence-by-Sentence Breakdown")
    for i, s in enumerate(sentence_results):
        with st.expander(f"Sentence {i+1}: {s['sentence'][:60]}..."):
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Positive", f"{s['positive']}%")
            c2.metric("🔴 Negative", f"{s['negative']}%")
            c3.metric("🟡 Neutral", f"{s['neutral']}%")

elif analyze_btn and not user_input:
    st.warning("⚠️ Please enter some text!")

st.markdown("---")
st.caption("Built by Krishan Kumar Chauhan | M.Tech Data Science, GBU | Powered by HuggingFace RoBERTa")