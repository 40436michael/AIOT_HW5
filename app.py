# app_upgrade.py
"""
Streamlit app: AI 文章偵測器 (升級版)
Features:
- GPT-2 Perplexity + 自建文字特徵
- 智慧整合模式
- 批次上傳分析
- 即時判定
- 特徵雷達圖與結果下載
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
import io

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI 文章偵測器 (升級版)",
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
tb_tokenizer = TreebankWordTokenizer()
sw_set = set(stopwords.words("english"))

# =========================
# Utility functions
# =========================
def simple_sentence_split(text):
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
    toks = tb_tokenizer.tokenize(text.lower())
    if not toks:
        return 0.0
