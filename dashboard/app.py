import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go

st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True
    )

model = load_model()

st.title("🧠 NLP Sentiment Analysis")
st.markdown("**Multi-class sentiment analyzer powered by RoBERTa transformer model**")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Text")
    user_input = st.text_area(
        "Paste your text here (up to 1000 words)",
        height=200,
        placeholder="Type or paste any review, comment, feedback, or paragraph here..."
    )
    word_count = len(user_input.split()) if user_input else 0
    st.caption(f"Word count: {word_count}/1000")
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

with col2:
    st.subheader("ℹ️ About")
    st.info("""
    **Model:** RoBERTa (Cardiff NLP)
    
    **Classes:**
    - 🟢 Positive
    - 🔴 Negative  
    - 🟡 Neutral
    
    **Use cases:**
    - Product reviews
    - Customer feedback
    - Social media posts
    - Survey responses
    """)

if analyze_btn and user_input:
    with st.spinner("Analyzing sentiment..."):
        truncated = ' '.join(user_input.split()[:512])
        raw_results = model(truncated)
        
        # raw_results is [[{label, score}, ...]]
        results = raw_results[0] if isinstance(raw_results[0], list) else raw_results

        scores = {}
        for r in results:
            scores[r['label']] = round(float(r['score']) * 100, 2)

        best_label = max(scores, key=scores.get)
        confidence = scores[best_label]

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    if best_label == 'positive':
        st.success(f"### ✅ POSITIVE — {confidence}% Confidence")
    elif best_label == 'negative':
        st.error(f"### ❌ NEGATIVE — {confidence}% Confidence")
    else:
        st.warning(f"### 😐 NEUTRAL — {confidence}% Confidence")

    st.markdown("---")

    def gauge(value, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': title, 'font': {'size': 16}},
            number={'suffix': '%', 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0.0, 100.0], 'tickwidth': 1},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': '#f5f5f5'},
                    {'range': [30, 70], 'color': '#eeeeee'},
                    {'range': [70, 100], 'color': '#e0e0e0'}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': float(value)
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=0, l=20, r=20))
        return fig

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(gauge(scores.get('positive', 0.0), '🟢 Positive', '#43A047'),
                        use_container_width=True)
    with col2:
        st.plotly_chart(gauge(scores.get('negative', 0.0), '🔴 Negative', '#E53935'),
                        use_container_width=True)
    with col3:
        st.plotly_chart(gauge(scores.get('neutral', 0.0), '🟡 Neutral', '#FF9800'),
                        use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Detailed Scores")
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Positive", f"{scores.get('positive', 0.0)}%")
    col2.metric("🔴 Negative", f"{scores.get('negative', 0.0)}%")
    col3.metric("🟡 Neutral", f"{scores.get('neutral', 0.0)}%")

elif analyze_btn and not user_input:
    st.warning("⚠️ Please enter some text to analyze!")

st.markdown("---")
st.caption("Built by Krishan Kumar Chauhan | M.Tech Data Science, GBU | Powered by HuggingFace RoBERTa")