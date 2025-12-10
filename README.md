# AI / Human 文章偵測器（Streamlit 應用）
## 專題主旨
隨著人工智慧生成文字技術（如 GPT 系列）的快速發展，辨識 AI 與人類撰寫文章的能力變得愈加重要。本專題旨在：

建立示範性 AI 文章偵測工具
結合 GPT-2 Perplexity 與文字啟發式特徵，提供簡單易用的判定方法。

探索 AI 與人類文字的特性差異
從語言困惑度、句子結構、詞彙重複度等角度分析文字，理解 AI 生成文本的模式。

提供教學與實驗平台
透過可視化圖表與特徵分析，幫助使用者直觀了解 AI 文章偵測的原理與方法。

專題目標並非提供最終的法律或學術判定工具，而是 展示 AI 文本偵測的技術流程與教學示範。

**Streamlit 線上 Demo**  
【[https://YOUR-STREAMLIT-APP.streamlit.ap](https://aiothw5-hlgfnupygxcwfoypzrq7df.streamlit.app/)p】



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
## 結論

AI 文章偵測器透過 GPT-2 Perplexity 與 文字啟發式特徵 的結合，提供一個直觀的方式來判斷文章是 AI 生成或人類撰寫。

Perplexity 可有效捕捉語言模式與生成特徵。

啟發式文字特徵 則提供額外線索，例如重複度、句子結構與詞彙使用。

智慧整合模式 將兩者結合，可提供更平衡、合理的判定結果。

本工具適合用於 教學、研究或實驗示範，幫助使用者了解 AI 生成文本的特性與檢測方法。但請注意，結果僅供參考，並不能替代正式的專業判斷。

