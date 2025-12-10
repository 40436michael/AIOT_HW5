# app.py
"""
Streamlit app: Simple AI vs Human text detector (heuristic + GPT-2 perplexity)
Requirements:
  pip install streamlit transformers torch nltk numpy pandas matplotlib

How it works (short):
  - Compute GPT-2 perplexity of input text (lower perplexity -> more 'model-like')
  - Compute simple text features (sentence length variance, stopword ratio,
    punctuation ratio, average token length, repetition score)
  - Map features to heuristic AI-score and combine with perplexity to produce final AI% / Human%
"""
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# Setup / Caching model loads
# ---------------------------
st.set_page_config(page_title="AI vs Human Detector", layout="centered")

@st.cache_resource(show_spinner=False)
def load_gpt2_model(model_name="gpt2"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

# ensure NLTK resources
@st.cache_resource(show_spinner=False)
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    return True

ensure_nltk()
tokenizer, gpt2_model = load_gpt2_model("gpt2")

# ---------------------------
# Utility functions
# ---------------------------
def chunked_perplexity(text, tokenizer, model, max_length=1024, stride=512, device=None):
    """
    Compute an approximate perplexity for a long text by windowing.
    Returns geometric mean perplexity across windows.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = tokenizer.encode(text)
    if len(enc) == 0:
        return float("inf")

    lls = []
    input_ids = torch.tensor(enc, dtype=torch.long).unsqueeze(0).to(device)
    n = input_ids.size(1)
    if n <= max_length:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss.item() * input_ids.size(1)  # loss is avg per token (CrossEntropy)
            ppl = math.exp(outputs.loss.item())
            return ppl
    # windowing
    stride = min(stride, max_length // 2)
    start_positions = list(range(0, n, stride))
    losses = []
    for start in start_positions:
        end = min(start + max_length, n)
        input_ids_window = input_ids[:, start:end].to(device)
        with torch.no_grad():
            outputs = model(input_ids_window, labels=input_ids_window)
            loss = outputs.loss.item()
            losses.append(loss)
        if end == n:
            break
    # geometric mean of per-token perplexity ~ exp(mean(losses))
    mean_loss = float(np.mean(losses))
    ppl = math.exp(mean_loss)
    return ppl

def repetition_score(text):
    """
    Measure token-level repetition: fraction of repeated 3-grams vs unique 3-grams.
    Higher repetition -> possibly AI (some models repeat).
    """
    toks = word_tokenize(text.lower())
    n = len(toks)
    if n < 3:
        return 0.0
    trigrams = [tuple(toks[i:i+3]) for i in range(n-2)]
    total = len(trigrams)
    unique = len(set(trigrams))
    rep = 1 - unique / total
    return rep

def punctuation_ratio(text):
    p = sum(1 for c in text if c in ".,;:!?()[]\"'—-")
    return p / max(1, len(text))

def stopword_ratio(text):
    sw = set(stopwords.words("english"))
    toks = word_tokenize(text.lower())
    if len(toks) == 0:
        return 0.0
    return sum(1 for t in toks if t in sw) / len(toks)

def sentence_length_variance(text):
    sents = sent_tokenize(text)
    if len(sents) <= 1:
        return 0.0
    lens = [len(word_tokenize(s)) for s in sents]
    return float(np.var(lens))

def avg_token_length(text):
    toks = word_tokenize(text)
    if not toks:
        return 0.0
    return np.mean([len(t) for t in toks])

# Map raw metric to normalized 0..1 (higher => more 'AI-like' depending on metric)
def normalize(value, vmin, vmax, invert=False):
    if np.isinf(value):
        return 0.0 if invert else 1.0
    x = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    x = np.clip(x, 0.0, 1.0)
    return 1 - x if invert else x

# ---------------------------
# Heuristic scoring pipeline
# ---------------------------
def compute_features(text):
    feats = {}
    feats["perplexity"] = chunked_perplexity(text, tokenizer, gpt2_model)
    feats["repetition"] = repetition_score(text)
    feats["punctuation_ratio"] = punctuation_ratio(text)
    feats["stopword_ratio"] = stopword_ratio(text)
    feats["sent_len_var"] = sentence_length_variance(text)
    feats["avg_token_len"] = avg_token_length(text)
    feats["char_count"] = len(text)
    feats["word_count"] = len(word_tokenize(text))
    return feats

def heuristic_ai_score(feats):
    """
    Combine features into a heuristic score in [0,1].
    Weights chosen heuristically:
      - lower perplexity => higher AI-likeness
      - lower sentence variance (very uniform short sentences) => higher AI-likeness
      - moderate repetition may indicate AI
      - punctuation patterns and stopword ratio provide signal
    """
    # Normalize metrics to [0,1] with reasonable ranges (tunable)
    # Perplexity: typical GPT-2 on coherent English might be ~10-50; map 5..80
    p_norm = normalize(feats["perplexity"], vmin=5.0, vmax=80.0, invert=True)

    rep_norm = normalize(feats["repetition"], vmin=0.0, vmax=0.5)  # 0-0.5
    punct_norm = normalize(feats["punctuation_ratio"], vmin=0.005, vmax=0.08)  # depends on text
    stop_norm = normalize(feats["stopword_ratio"], vmin=0.2, vmax=0.6, invert=True)  # high stopwords -> human-like
    sentvar_norm = normalize(feats["sent_len_var"], vmin=0.0, vmax=100.0, invert=True)  # low variance -> more AI
    avg_tok_len_norm = normalize(feats["avg_token_len"], vmin=3.0, vmax=6.0, invert=False)  # short tokens maybe simpler -> AI

    # weights (tunable)
    w = {
        "p": 0.45,
        "rep": 0.12,
        "punct": 0.08,
        "stop": 0.10,
        "sentvar": 0.15,
        "avgtok": 0.10
    }

    score = (w["p"]*p_norm +
             w["rep"]*rep_norm +
             w["punct"]*punct_norm +
             w["stop"]*stop_norm +
             w["sentvar"]*sentvar_norm +
             w["avgtok"]*avg_tok_len_norm)
    # clip
    score = float(np.clip(score, 0.0, 1.0))
    return score, {
        "p_norm": p_norm,
        "rep_norm": rep_norm,
        "punct_norm": punct_norm,
        "stop_norm": stop_norm,
        "sentvar_norm": sentvar_norm,
        "avgtok_norm": avg_tok_len_norm
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("AI vs Human — Simple Text Detector")
st.write("輸入一段英文或中文（英文效果較佳）。本工具使用 GPT-2 perplexity 與簡單文字特徵的啟發式組合來估計「AI 產生」的機率。")

col1, col2 = st.columns([3,1])
with col1:
    text = st.text_area("輸入文字 (Input text)", height=280, value=(
        "Paste or type text here. Example:\n\n"
        "Artificial intelligence has transformed many industries. Large language models can write articles, "
        "generate code, and help brainstorm ideas. However, distinguishing human writing from machine-generated "
        "text can be challenging.\n\n"
        "This is a short sample."))
with col2:
    st.markdown("**Options**")
    method = st.radio("Method", ["Ensemble (perplexity + features)", "Perplexity only", "Heuristic features only"])
    show_details = st.checkbox("Show feature details", value=True)
    st.markdown("---")
    st.markdown("**Notes**")
    st.caption("This is a heuristic demo. For production accuracy, train on labelled data or use specialized detectors.")

if not text.strip():
    st.info("請輸入一些文字以便分析。")
    st.stop()

# Compute
with st.spinner("Analyzing... (this may take a few seconds on first run)"):
    feats = compute_features(text)
    score_feat, normed = heuristic_ai_score(feats)
    # Perplexity mapping alone to ai score
    p_norm_alone = normalize(feats["perplexity"], vmin=5.0, vmax=80.0, invert=True)
    # combine according to method
    if method == "Perplexity only":
        ai_score = p_norm_alone
    elif method == "Heuristic features only":
        ai_score = score_feat
    else:  # ensemble
        ai_score = 0.6 * p_norm_alone + 0.4 * score_feat
    # map to percentages
    ai_pct = int(round(ai_score * 100))
    human_pct = 100 - ai_pct

# Display main result
st.metric(label="Predicted: AI vs Human", value=f"{ai_pct}% AI — {human_pct}% Human")

# Bar chart
df = pd.DataFrame({"label": ["AI", "Human"], "prob": [ai_pct, human_pct]})
fig, ax = plt.subplots(figsize=(5,2))
ax.barh(df["label"], df["prob"])
ax.set_xlim(0,100)
ax.set_xlabel("Percent")
for i, v in enumerate(df["prob"]):
    ax.text(v + 1, i, f"{v}%", va="center")
st.pyplot(fig)

# Show computed features and interpretations
if show_details:
    st.subheader("Computed features & diagnostics")
    st.write(f"**Perplexity (GPT-2 estimate):** {feats['perplexity']:.2f}")
    st.write(f"**Repetition score (3-gram):** {feats['repetition']:.3f} (higher → more repetitive)")
    st.write(f"**Punctuation ratio:** {feats['punctuation_ratio']:.4f}")
    st.write(f"**Stopword ratio:** {feats['stopword_ratio']:.3f} (higher → more human-like)")
    st.write(f"**Sentence length variance:** {feats['sent_len_var']:.3f} (lower → more uniform → may be AI-like)")
    st.write(f"**Average token length:** {feats['avg_token_len']:.3f}")
    st.write("---")
    st.subheader("Normalized feature contributions (0..1)")
    st.write(pd.DataFrame.from_dict(normed, orient="index", columns=["normalized"]).round(3))
    st.write("---")
    st.subheader("How the final score was computed")
    st.write("Perplexity is mapped into a normalized value (lower ppl → higher AI-likeness). "
             "Then combined with text-feature heuristics. Weights are tuned for demo purposes.")

# Provide suggestions & caveats
st.subheader("Interpretation & caveats")
st.write("""
- 這是一個示範性質的偵測器（heuristic）。若要達到高準確度，請收集標註資料並用 supervised learning（例如用 Transformers 或 sklearn 訓練分類器），或使用現成的 detector 模型。
- GPT-2 perplexity 只是一個 proxy：不同生成模型、溫度、修改後的文字都會改變 perplexity 分佈。
- 中文文本可能表現較差（此範例的 tokenization / stopwords 使用英文資源）。
""")

st.markdown("----")
st.markdown("### Quick tips to improve detection accuracy")
st.write("- Use labeled dataset (human vs ai) and train a classifier (TF-IDF + LogisticRegression 或 Transformer-based classifier)。")
st.write("- Add language-specific preprocessing (中文斷詞、停用詞)。")
st.write("- Consider ensemble of detectors and adversarially augmented data.")

# Footer: download features as CSV
if st.button("Download feature diagnostics CSV"):
    csv = pd.DataFrame([feats]).T.rename(columns={0:"value"})
    st.download_button("Download CSV", csv.to_csv(), file_name="diagnostics.csv", mime="text/csv")
