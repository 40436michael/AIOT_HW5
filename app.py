# app.py
"""
Streamlit app: AI 文章偵測器
Method: GPT-2 Perplexity + Heuristic Text Features
"""

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import re

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI 文章偵測器",
    layout="centered"
)

# =========================
# Load & cache resources
# =========================
@st.cache_resource(show_spinner=False)
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

@st.cache_resource(show_spinner=False)
def load_gpt2():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return tokenizer, model

ensure_nltk()
tokenizer_gpt2, gpt2_model = load_gpt2()

# ✅ 穩定 tokenizer（不依賴 punkt / punkt_tab）
tb_tokenizer = TreebankWordTokenizer()

# =========================
# Utility functions
# =========================
def simple_sentence_split(text):
    """
    Regex-based sentence splitter (Streamlit-safe).
    """
    sents = re.split(r'[.!?]+\s*', text)
    return [s.strip() for s in sents if s.strip()]

def compute_perplexity(text, tokenizer, model, max_len=1024, stride=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tokenizer.encode(text)
    if len(enc) < 2:
        return float("inf")

    if len(enc) <= max_len:
        ids = torch.tensor(enc).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        return math.exp(out.loss.item())

    losses = []
    for i in range(0, len(enc), stride):
        window = enc[i:i + max_len]
        ids = torch.tensor(window).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        losses.append(out.loss.item())
        if i + max_len >= len(enc):
            break

    return math.exp(float(np.mean(losses)))

def repetition_score(text):
    toks = tb_tokenizer.tokenize(text.lower())
    if len(toks) < 3:
        return 0.0
    trigrams = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
    return 1 - len(set(trigrams)) / len(trigrams)

def punctuation_ratio(text):
    punct = sum(1 for c in text if c in ".,;:!?()[]\"'—-")
    return punct / max(1, len(text))

def stopword_ratio(text):
    sw = set(stopwords.words("english"))
    toks = tb_tokenizer.tokenize(text.lower())
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in sw) / len(toks)

def sentence_length_variance(text):
    sents = simple_sentence_split(text)
    if len(sents) <= 1:
        return 0.0
    lens = [len(tb_tokenizer.tokenize(s)) for s in sents]
    return float(np.var(lens))

def avg_token_length(text):
    toks = tb_tokenizer.tokenize(text)
    if not toks:
        return 0.0
    return float(np.mean([len(t) for t in toks]))

def normalize(val, vmin, vmax, invert=False):
    if np.isinf(val):
        return 0.0 if invert else 1.0
    x = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    x = np.clip(x, 0.0, 1.0)
    return 1 - x if invert else x

# =========================
# Feature pipeline
# =========================
def extract_features(text):
    return {
        "perplexity": compute_perplexity(text, tokenizer_gpt2, gpt2_model),
        "repetition": repetition_score(text),
        "punctuation": punctuation_ratio(text),
        "stopword": stopword_ratio(text),
        "sent_var": sentence_length_variance(text),
        "avg_tok": avg_token_length(text),
    }

def heuristic_score(f):
    p = normalize(f["perplexity"], 5, 80, invert=True)
    rep = normalize(f["repetition"], 0, 0.5)
    punct = normalize(f["punctuation"], 0.005, 0.08)
    stop = normalize(f["stopword"], 0.2, 0.6, invert=True)
    sentv = normalize(f["sent_var"], 0, 100, invert=True)
    tok = normalize(f["avg_tok"], 3, 6)

    return float(np.clip(
        0.45*p + 0.12*rep + 0.08*punct +
        0.10*stop + 0.15*sentv + 0.10*tok,
        0.0, 1.0
    ))

# =========================
# Streamlit UI
# =========================
st.title("AI 文章偵測器")
st.caption("參考 JustDone AI Detector｜GPT-2 Perplexity + 啟發式分析")

text = st.text_area(
    "請輸入文章內容（英文效果最佳）",
    height=260,
    placeholder="Paste or type text here..."
)

method = st.radio(
    "分析模式",
    ["智慧整合模式（建議）", "僅 Perplexity", "僅文字特徵"]
)

if not text.strip():
    st.info("請輸入文字後再進行分析。")
    st.stop()

with st.spinner("分析中，請稍候..."):
    feat = extract_features(text)
    h_score = heuristic_score(feat)
    p_score = normalize(feat["perplexity"], 5, 80, invert=True)

    if method == "僅 Perplexity":
        ai_score = p_score
    elif method == "僅文字特徵":
        ai_score = h_score
    else:
        ai_score = 0.6 * p_score + 0.4 * h_score

    ai_pct = int(round(ai_score * 100))
    human_pct = 100 - ai_pct

# =========================
# Result
# =========================
st.metric(
    "判斷結果",
    f"{ai_pct}% 為 AI 生成",
    f"{human_pct}% 為人類撰寫"
)

df = pd.DataFrame({
    "類型": ["AI-generated", "Human-written"],
    "比例": [ai_pct, human_pct]
})

fig, ax = plt.subplots(figsize=(5, 2))
ax.barh(df["類型"], df["比例"])
ax.set_xlim(0, 100)
for i, v in enumerate(df["比例"]):
    ax.text(v + 1, i, f"{v}%")
st.pyplot(fig)

# =========================
# Diagnostics
# =========================
with st.expander("顯示分析細節"):
    for k, v in feat.items():
        st.write(f"**{k}**：{v:.4f}")

st.markdown("---")
st.write(
    "⚠ 本系統為示範性 AI 文章偵測工具，使用啟發式規則與語言模型困惑度，"
    "僅供教學與展示用途，不適合用於學術不端或法律判定。"
)


