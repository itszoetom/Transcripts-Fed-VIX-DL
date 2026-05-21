# DL410 Project Milestone Report

**Project:** Frozen FinBERT + learned sentence-attention for 3-day VIX prediction from Federal Reserve text across political regimes
**Name:** Zoe Tomlinson
**Repo:** https://github.com/itszoetom/Transcripts-Fed-VIX-DL
**Notebook:** `notebooks/data_demo.ipynb`

## 1. What problem will you be investigating? Why is it interesting to you?
Whether Federal Reserve communications contain predictive signal for 3-day forward VIX change, and whether that signal structurally degrades across U.S. political regimes (Obama → Trump 1 → Biden → Trump 2). Monetary-policy text is information-dense but qualitative, and the language-volatility relationship is plausibly state-dependent under political pressure on the Fed. Combines my interest in political effects on markets with quantitative finance and text modeling.

## 2. What dataset will you use, and how will you get it?
326 documents scraped from federalreserve.gov: FOMC minutes (1993–present, 268 docs across four URL conventions the Fed used over the period) and Humphrey-Hawkins / Semiannual Monetary Policy Report testimony (1997–present, 58 docs). Target: 3-day close-to-close VIX change (FRED `VIXCLS`), aligned to the next trading day for weekend/holiday releases (no look-ahead leakage). Fully reproducible via `scripts/build_data.py`; the demo notebook runs against 10 bundled example documents and needs no FRED key or Talapas access.

## 3. Have people worked on this problem before? What's different?
Alexopoulos et al. (2023) on testimony→Treasury rates with classical NLP; *Digital Finance* (2023) uses pretrained FinBERT sentiment as a fixed feature; Gössi et al. (2023) fine-tune FinBERT on FOMC for sentiment classification. **This project's contribution:** frozen FinBERT + learned sentence-attention aggregator on a VIX *regression* target across a merged FOMC + testimony corpus, with explicit structural-break analysis at U.S. inauguration days.

## 4. How will you evaluate?
- **Regression:** MSE and Pearson r vs. TF-IDF + Ridge baseline.
- **Classification:** target binarized at training median; AUC-ROC and F1 vs. BoW + Logistic Regression baseline.
- **Temporal generalization:** train pre-2017-01-20, evaluate on 2017–2021, 2021–2025, 2025–present; report R² degradation; Chow F-test on residuals at 2017-01-20 (in-sample-bias caveat documented) and 2025-01-20.

## Changes from original proposal
- **Encoder:** fine-tuned FinBERT → fully *frozen* FinBERT + learned sentence-attention (~100k trained params) + linear head. ~233 train docs is under-data for fine-tuning 110M params.
- **Encoding:** 512-token chunking → sentence segmentation (NLTK Punkt, first 80 sentences; opening sections carry the policy rationale).
- **Training:** HuggingFace Trainer → manual PyTorch loop (cleaner control over the custom head + frozen-encoder embedding cache).
- **Splits:** added 2025-01-20 (Trump 2) as a third regime boundary.
- **Deferred past milestone:** frozen-vs-fine-tuned ablation, directional trading-strategy visualization.
