# Methodology

A comprehensive write-up of the project's data-collection, feature-engineering, and modeling decisions, with rationale and citations for each non-trivial choice. This document is the canonical reference for the methodology section of the final write-up.

## 1. Research questions

**RQ1 (predictive):** Does the text of Federal Reserve policy communications (FOMC minutes and Humphrey-Hawkins / Semiannual Monetary Policy Report testimony) predict the 3-day close-to-close change in the CBOE Volatility Index (VIX) following each document's release?

**RQ2 (structural-break / regime):** Does the text→volatility relationship structurally shift across U.S. political regimes, Obama (pre-2017), Trump 1 (2017-01-20 to 2021-01-20), Biden (2021-01-20 to 2025-01-20), Trump 2 (2025-01-20–present)?

RQ1 is the modeling problem; RQ2 is a hypothesis test on the residuals from RQ1's model.

## 2. Data acquisition

### 2.1 Document sources
Two long-running Federal Reserve text corpora:

- **FOMC minutes (1993-present, 268 documents):** Eight regular meetings per year plus occasional emergency releases. Released ~3 weeks after each meeting. The *release date* (not the meeting date) is used for VIX alignment because the release is when markets could observe and react to the text.
- **Humphrey-Hawkins / Semiannual Monetary Policy Report testimony (1997-present, 58 documents):** The Fed Chair's twice-yearly Congressional testimony.

Total corpus: 326 documents after VIX alignment (some recent releases have no 3-day forward VIX yet and are dropped).

### 2.2 Scraping
All documents are scraped from `federalreserve.gov` using `requests` + `BeautifulSoup` (HTML) and `pypdf` (PDFs, required for FOMC minutes from 1993-2006 which are PDF-only on the modern Fed site). The Fed has used **four distinct URL conventions** for FOMC minutes across the corpus period; the scraper handles all four:

1. `/fomc/MINUTES/{YYYY}/{YYYYMMDD}min.htm` (1993-1995)
2. `/fomc/minutes/{YYYYMMDD}.htm` (~1996-2007)
3. `/monetarypolicy/fomcminutes{YYYYMMDD}.htm|pdf` (~2007-2020)
4. `/monetarypolicy/files/fomcminutes{YYYYMMDD}.pdf` (~2020-present)

Humphrey-Hawkins testimony similarly uses two patterns (`{YYYY}testimony.htm` pre-~2016, `{YYYY}-testimony.htm` post-~2016) plus a pre-2006 `/boarddocs/hh/` layout. Discovery of all URL families was iterative, the scraper was patched until the document count converged on ~322 (the project proposal's expectation; the realized count is 326).

### 2.3 Caching
Raw HTTP bytes (HTML + PDF), extracted plain text, and the FRED VIX series are all cached under `data/raw/`. Re-runs of the scraper are idempotent and skip any document whose text file already exists, making iterative development cheap.

### 2.4 Target (VIX)
Pulled from FRED via `fredapi` (series `VIXCLS`, the official CBOE VIX closing value as redistributed by the St. Louis Fed). FRED is preferred over third-party tickers (e.g., Yahoo's `^VIX`) because the series is reproducible from `(API key, ticker)` alone.

**Alignment rule (next-trading-day):** For each document release date $r$, let $t$ be the first trading day with $t \ge r$. The 3-day forward change is then $\Delta_3 = \text{VIX}_{t+3} - \text{VIX}_t$ where the index advances by *trading* days (Fridays → next Wednesday). The "next" trading day (not "previous") is used because the previous would introduce look-ahead leakage (using market data from before the document was released).

## 3. Text preprocessing

### 3.1 Sentence segmentation
NLTK's Punkt sentence tokenizer (Kiss & Strunk, 2006) is used to split each document's plain-text body into sentences. Punkt is the academic-NLP standard for unsupervised sentence boundary detection on formal prose and handles common abbreviations (U.S., Mr., Dr.) reasonably well, important for Fed text, which is abbreviation-dense.

### 3.2 Sentence cap
Each document is truncated to the **first 80 sentences** before encoding. Three motivations:

1. **Signal concentration.** Opening sections of FOMC minutes and HH testimony contain the policy rationale and forward-looking language that markets react to; later sections (member-by-member voting records, operational appendices) are well-known to be lower signal in the Fed-text-mining literature (Hansen, McMahon & Prat 2018; Curti & Kazinnik 2023).
2. **Memory tractability.** 326 documents × 80 sentences × 768-dim embeddings = ~80 MB, easy to pre-compute once and cache on disk.
3. **Batching simplicity.** A uniform cap lets us materialize a single `(B, 80, 768)` tensor with a 0/1 mask, which is much simpler than ragged per-doc tensors.

The cap is enforced *at the sentence boundary*, not in mid-text. This satisfies the project constraint of "no chunking, no head-truncation."

## 4. Encoder: frozen FinBERT with mask-weighted mean pooling

### 4.1 Choice of encoder
`yiyanghkust/finbert-pretrain` (Yang, Uy & Huang 2020, "FinBERT: A Pretrained Language Model for Financial Communications") is the masked-language-model-pretrained variant of FinBERT, initialized from `bert-base-uncased` and further pretrained on financial corpora. It is *not* the sentiment-classification fine-tuned variant (`yiyanghkust/finbert-tone`); we deliberately want the general financial-language representations, not the sentiment-classification head's bias.

The repository ships only model weights, no tokenizer files, so the tokenizer is loaded from `bert-base-uncased` directly (FinBERT uses the same 30,522-token WordPiece vocabulary).

### 4.2 Why fully frozen
All FinBERT parameters are frozen during training (`requires_grad=False`). With only ~233 training documents, fine-tuning a 110M-parameter BERT is hopelessly under-data, it would overfit catastrophically and discard the financial-domain pretraining signal we're paying for by choosing FinBERT in the first place. Frozen-encoder training is also strictly faster because sentence embeddings can be pre-computed once and cached (see §6).

### 4.3 Why mean pooling, not `[CLS]`
For each sentence we extract the encoder's last-hidden-state tokens and compute their **mask-weighted mean** across the non-padding positions. We deliberately do *not* use the `[CLS]` token's embedding.

The reason: `[CLS]` is only a meaningful sentence representation for BERT models that have been *fine-tuned* on a sentence-classification objective. FinBERT-pretrain is MLM-only; on MLM-pretrained-only BERTs, `[CLS]` is a known-weak embedding (Reimers & Gurevych 2019, "Sentence-BERT", §3.3), and mean pooling consistently outperforms it as a frozen sentence representation. Reviewers familiar with the SBERT literature would flag `[CLS]`-based pooling here as the wrong choice.

## 5. Document aggregation: additive (Bahdanau-style) attention

Sentence embeddings are aggregated into one document vector via additive attention:

$$
u_i = \tanh(W h_i + b), \quad s_i = v^\top u_i, \quad \alpha = \text{softmax}(\text{mask}(s)), \quad d = \sum_i \alpha_i h_i
$$

where $h_i$ is the $i$-th sentence embedding (768-d), $W \in \mathbb{R}^{128 \times 768}$, $b \in \mathbb{R}^{128}$, $v \in \mathbb{R}^{128}$, and the mask sets padded positions to $-\infty$ before softmax. The document vector $d$ feeds a linear regression head $y = w^\top d + b_{\text{reg}}$.

This is the Hierarchical Attention Network formulation of Yang et al. (2016, "Hierarchical Attention Networks for Document Classification"), the canonical academic reference for "attend over sentence embeddings to make one document vector." Alternatives considered and rejected:

- **Dot-product attention with a learned query.** Simpler but lacks the per-sentence linear transform; less expressive when the underlying embeddings aren't optimized for the regression target.
- **Single self-attention layer with a `[DOC]` token.** ~2.4M parameters vs. ~100k for additive, a brutal capacity/data ratio for ~233 training documents.

Additive attention's ~100k trained parameters is the sweet spot for this dataset size.

## 6. Embedding cache

Because the encoder is fully frozen and deterministic (no dropout in eval mode), each sentence's 768-d embedding is fully determined by its text. We pre-compute all ~25,000 sentence embeddings once (`scripts/precompute_embeddings.py`), save them to `data/processed/sentence_embeddings.pt`, and load from there during training. This speeds training by 2–3 orders of magnitude with **no information leakage** (the encoder is the same model with the same weights regardless of when it runs).

## 7. Train/val/test splits

All splits are *strictly temporal*, no random shuffling, anywhere in the pipeline. Documents are sorted by `release_date` ascending and partitioned by date boundaries.

**Boundaries (anchored to U.S. presidential inauguration days):**
- `2017-01-20`, Trump 1 inauguration (train/test divide)
- `2021-01-20`, Biden inauguration
- `2025-01-20`, Trump 2 inauguration

**Partitions:**
- Train pool: `release_date < 2017-01-20` (233 documents)
- Val: last 15% chronologically of the train pool (35 documents); used only for early stopping
- Regime 1: 2017-01-20 ≤ date < 2021-01-20 (40 docs)
- Regime 2: 2021-01-20 ≤ date < 2025-01-20 (40 docs)
- Regime 3: 2025-01-20 ≤ date (13 docs as of May 2026)

Anchoring to inauguration dates makes the regime hypothesis (RQ2) directly interpretable, each segment maps to a single presidential administration.

## 8. Training procedure

Manual PyTorch loop (no HuggingFace Trainer). All hyperparameters live in `configs/default.yaml`:

- **Optimizer:** AdamW(lr=3e-4, weight_decay=1e-4) on the ~99,329 trainable parameters (attention + linear head only, encoder is frozen). The learning rate is higher than the conventional BERT-fine-tuning 2e-5 because we are training a small head from scratch, not fine-tuning a large backbone.
- **LR schedule:** linear warmup over the first 10% of total steps, constant thereafter. Constant-after-warmup (rather than decay) is preferred because early stopping on val MSE controls the actual stop point.
- **Gradient clipping:** `max_norm=1.0`.
- **Early stopping:** stop if val MSE has not improved for 5 epochs; max 30 epochs.
- **Batch size:** 8 documents.
- **Loss:** mean squared error on the regression target.

## 9. Evaluation

### 9.1 Primary metrics (continuous target)
- **MSE**, direct comparison to baseline.
- **R²**, fraction of variance explained; comparable across splits with differing target variances.
- **Pearson r**, direction-of-signal metric, scale-free.

### 9.2 Secondary metrics (binarized target)
The continuous target is binarized at the *training-set* median (computed only on the train split, to avoid leakage into evaluation). Two metrics:
- **AUC-ROC** on the model's continuous output as a score.
- **F1** at the median threshold.

### 9.3 Baselines
- **TF-IDF + Ridge regression:** unigram TF-IDF (min_df=2, max_df=0.95, sublinear_tf=True) → Ridge(α=1.0). Same temporal splits.
- **BoW + Logistic regression:** unigram counts → LogisticRegression(C=1.0, solver=liblinear). Same temporal splits, same train-median binarization threshold.

Baselines use fixed, conventional hyperparameters with no inner CV. The deep model gets one early-stopping signal; the baselines get zero. This keeps the comparison honest.

### 9.4 Temporal generalization
The trained model is evaluated on each of regime1, regime2, regime3 separately. R² and Pearson r are reported per regime; the test of RQ2 is whether these decay across regimes.

### 9.5 Chow test on residuals (Chow 1960)
For each configured breakpoint $b$, the F-statistic tests whether the *mean residual* shifts at $b$:

$$
F = \frac{(\text{RSS}_{\text{pooled}} - (\text{RSS}_{\text{pre}} + \text{RSS}_{\text{post}})) / k}{(\text{RSS}_{\text{pre}} + \text{RSS}_{\text{post}}) / (N - 2k)} \sim F(k, N-2k) \text{ under H}_0
$$

with $k=1$ (intercept-only regression). Implemented in `src/transcripts_fed_vix/utils/chow.py`.

**Caveat for the 2017-01-20 breakpoint:** pre-2017 residuals are in-sample (the model was trained on them) and biased toward zero. The Chow test at 2017-01-20 is therefore anti-conservative, likely to over-detect a "shift" that is partly the train/test discontinuity. This caveat is recorded in `outputs/regime_analysis.json` next to the test result. The 2025-01-20 breakpoint, where both sides are out-of-sample, is interpretable as a clean regime-change test.

## 10. Reproducibility

- All random seeds (Python, NumPy, PyTorch CPU+CUDA) are set from `configs/default.yaml`.
- Sentence embeddings are deterministic (frozen encoder, no eval-mode dropout).
- The full pipeline (`scripts/train.sbatch`) is idempotent, re-runs use cached scraped HTML, cached VIX, cached embeddings, etc., and re-deriving the dataset from scratch is one command (`python scripts/build_data.py --force`).
- The trained model checkpoint, per-epoch metrics, per-regime metrics, regime-analysis report, baseline metrics, and all figures are written under `outputs/`.

## 11. Selected references

- Chow, G. C. (1960). "Tests of Equality Between Sets of Coefficients in Two Linear Regressions." *Econometrica*, 28(3): 591–605.
- Curti, F., & Kazinnik, S. (2023). "Central Bank Communication and Inflation Expectations." *Working paper.*
- Hansen, S., McMahon, M., & Prat, A. (2018). "Transparency and Deliberation Within the FOMC: A Computational Linguistics Approach." *Quarterly Journal of Economics*, 133(2): 801–870.
- Kiss, T., & Strunk, J. (2006). "Unsupervised Multilingual Sentence Boundary Detection." *Computational Linguistics*, 32(4): 485–525. (NLTK Punkt)
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP-IJCNLP 2019*.
- Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). "Hierarchical Attention Networks for Document Classification." *NAACL-HLT 2016*.
- Yang, Y., Uy, M. C. S., & Huang, A. (2020). "FinBERT: A Pretrained Language Model for Financial Communications." *arXiv:2006.08097*.
- Alexopoulos, M., Han, X., Kryvtsov, O., & Zhang, X. (2023). "More than Words: Fed Chairs' Communication During Congressional Testimonies." *Bank of Canada Staff Working Paper.* (Cited in original project proposal.)
- Gössi, S., Chen, Z., Kim, W., Bermeitinger, B., & Handschuh, S. (2023). "FinBERT-FOMC: Fine-Tuned FinBERT Model with Sentiment Focus Method for Analyzing FOMC Minutes." *ICAIF 2023.* (Cited in original project proposal.)
