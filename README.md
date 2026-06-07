# 🧠 NLP Sentiment Analyzer

A sentence-level multi-class sentiment analysis web app powered by **RoBERTa transformer model** (Cardiff NLP). Analyzes text paragraph sentence-by-sentence and returns Positive, Negative, and Neutral scores with interactive gauge meters.

## 🔴 Live Demo
👉 [Click here to try the live app](https://nlp-sentiment-analysis292003.streamlit.app/)

## 🤔 How It Works
1. Input text is split into individual sentences
2. Each sentence is analyzed by RoBERTa model separately
3. Scores are averaged across all sentences
4. Final result shows mixed sentiment with all 3 class percentages

## 🧪 Example
Input: *"This product is amazing! However, the delivery was very slow. Overall it was okay."*
- 🟢 Positive: 62.3%
- 🔴 Negative: 28.1%
- 🟡 Neutral: 9.6%

## 🛠️ Tech Stack
- **Model:** RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) via HuggingFace
- **Framework:** Streamlit — interactive web dashboard
- **Visualization:** Plotly — gauge meters
- **NLP:** HuggingFace Transformers pipeline

## 🤖 About the Model
**HuggingFace** is an open-source platform hosting thousands of pre-trained AI models — like GitHub for AI. **RoBERTa** is Facebook's improved version of BERT, fine-tuned by Cardiff NLP specifically for sentiment classification on real-world social media and review data.

## 🚀 Run Locally
```bash
git clone https://github.com/krishanchauhan29/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 👤 Author
**Krishan Kumar Chauhan**
M.Tech Data Science | Gautam Buddha University
[LinkedIn](https://www.linkedin.com/in/krishan-chauhan-714011232/) | [GitHub](https://github.com/krishanchauhan29)
