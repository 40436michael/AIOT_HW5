# AI / Human 文章偵測器（Streamlit 應用）

本專題實作一個簡易的 **AI 與人類文章判別工具（AI Detector）**，使用者輸入一段文章後，系統即時判斷該文字為 AI 生成或人類撰寫，並以 **AI % / Human %** 機率方式呈現結果。

👉 **Streamlit 線上 Demo**  
【https://YOUR-STREAMLIT-APP.streamlit.app】

👉 **GitHub Repository**  
【https://github.com/YOUR-USERNAME/YOUR-REPO-NAME】

---

## 一、專題介紹（Project Description）

本專題目的在於實作一個可即時互動的 AI 文章偵測系統，展示 **自然語言處理（NLP）** 與 **預訓練語言模型（Pretrained Language Model）** 在文字分析上的應用。

系統並非依賴單一模型，而是結合：
- **GPT-2 Perplexity（困惑度）分析**
- **人工設計文字統計特徵（Heuristic Features）**
- **特徵融合（Ensemble）策略**

最終輸出為文字屬於 AI 或 Human 的機率比例。

---

## 二、系統功能（Features）

✅ 使用者輸入文字後即時分析  
✅ 顯示 AI / Human 百分比結果  
✅ 提供文字統計與特徵分析資訊  
✅ 視覺化信心分數（長條圖）  
✅ Streamlit Web UI 介面操作  

---

## 三、系統架構（System Architecture）

```text
使用者輸入文字
        ↓
文字前處理與斷詞
        ↓
特徵擷取
  - GPT-2 Perplexity
  - 文字統計特徵
        ↓
特徵評分與加權整合
        ↓
AI / Human 機率輸出
        ↓
Streamlit 視覺化結果
