# Methodology

A focused write-up of the project's data-collection, feature-engineering, and modeling decisions, with rationale and citations for each non-trivial choice.

## 0. Context for non-finance readers

This section gives the background needed to understand why the project's research questions are well-posed.

### 0.1 The Federal Reserve and the FOMC

The Federal Reserve (the Fed) is the central bank of the United States. Its primary tool is monetary policy: it sets the federal funds rate (the overnight interbank lending rate), which propagates into all other interest rates in the economy, and adjusts the size of its balance sheet (quantitative easing or tightening). These choices govern the cost of credit, the supply of money, and ultimately inflation and unemployment.

The Federal Open Market Committee (FOMC) is the 12-member subcommittee that actually decides monetary policy. It meets eight times per year. After each meeting it releases (a) a brief statement on the same day and (b) detailed minutes about three weeks later. The minutes contain the discussion behind the rate decision: how committee members read the economy, what risks they emphasized, and what they expect to do next.

The Fed Chair also testifies before Congress twice a year under the Humphrey-Hawkins (Full Employment and Balanced Growth) framework. This testimony, formally the Semiannual Monetary Policy Report, gives the Chair an extended forum to explain the Fed's outlook in their own words.

Together, FOMC minutes and Humphrey-Hawkins testimony form the most policy-rich body of Fed text. Markets parse them line by line for signals about future rate moves.

### 0.2 The VIX

The CBOE Volatility Index (VIX) is a real-time index published by the Chicago Board Options Exchange. It is computed from the prices of short-dated S&P 500 options and is designed to express the market's expectation of 30-day forward volatility in the S&P 500. Higher VIX means the market is paying more for protection against large moves; lower VIX means the market expects calm. Whaley (2000) coined the now-standard nickname "investor fear gauge."

We use the *change* in VIX (specifically, the 3-day close-to-close change) as the target rather than the VIX level. A change target asks the right question for our purposes: did the document move expected volatility, conditional on whatever it was already? Levels are dominated by the broad volatility regime (e.g., post-2008, COVID), which has nothing to do with the text content of a particular release.

### 0.3 Why Fed text could move the VIX

Fed communications are information-dense for two reasons: the Fed is deliberately the most consequential single source of monetary-policy signal in U.S. financial markets, and its language is calibrated. Even small word choices (for example, "patient" vs. "data-dependent") are read as meaningful. If a release sounds hawkish (more concerned about inflation, willing to tighten), or surprising in any direction, traders update their expectations about future policy, which raises uncertainty about the path of asset prices, which shows up as higher implied volatility (higher VIX). If the release is calm and predictable, the opposite. Lucca and Trebbi (2009), Hansen, McMahon and Prat (2018), and Boukus and Rosenberg (2006) document that FOMC text moves Treasury yields and equity-market measures in statistically reliable ways. The 3-day forward VIX change is a natural and previously studied (if noisy) target for this kind of analysis.

### 0.4 Why the political regime matters

Different U.S. presidents have had very different relationships with the Fed. The Fed is formally independent, but Fed chairs are presidential appointees, and the political pressure they face has fluctuated. President Trump (2017 to 2021) broke a long-standing convention by publicly pressuring the Fed to cut rates. President Biden (2021 to 2025) returned to the hands-off norm. President Trump's second term (2025 onward) has resumed public pressure on Fed leadership. If the text-to-volatility relationship is conditional on the political environment around the Fed (for example, if traders read the same language differently when they believe the Chair is being pressured), then a model trained on pre-2017 text may generalize poorly to later regimes. This is the project's secondary research question.

## 1. Research questions

**RQ1 (predictive):** Does the text of Federal Reserve policy communications (FOMC minutes and Humphrey-Hawkins testimony) predict the 3-day close-to-close change in the CBOE VIX following each document's release?

**RQ2 (temporal generalization across political regimes):** Does the text-to-volatility relationship shift across U.S. political regimes: Obama (pre-2017), Trump 1 (2017-01-20 to 2021-01-20), Biden (2021-01-20 to 2025-01-20), Trump 2 (2025-01-20 to present)?

RQ1 is the modeling problem. RQ2 is answered descriptively by computing R^2 and Pearson r per regime on the model's predictions and inspecting the residuals-over-time figure.

## 2. Data acquisition

### 2.1 Document sources

- **FOMC minutes (1993 to present, 268 documents):** Eight regular meetings per year plus occasional emergency releases. Released about three weeks after each meeting. The *release date* (not the meeting date) is used for VIX alignment because the release is when markets could observe and react to the text.
- **Humphrey-Hawkins / Semiannual Monetary Policy Report testimony (1997 to present, 58 documents):** The Fed Chair's twice-yearly Congressional testimony.

Total corpus: 326 documents after VIX alignment (recent releases without 3-day forward VIX data are dropped).

### 2.2 Scraping

All documents are scraped from `federalreserve.gov` using `requests` + `BeautifulSoup` (HTML) and `pypdf` (PDFs, required for FOMC minutes from 1993 to 2006). The Fed has used four distinct URL conventions for FOMC minutes over the corpus period; the scraper handles all four:

1. `/fomc/MINUTES/{YYYY}/{YYYYMMDD}min.htm` (1993 to 1995)
2. `/fomc/minutes/{YYYYMMDD}.htm` (~1996 to 2007)
3. `/monetarypolicy/fomcminutes{YYYYMMDD}.htm|pdf` (~2007 to 2020)
4. `/monetarypolicy/files/fomcminutes{YYYYMMDD}.pdf` (~2020 to present)

Humphrey-Hawkins testimony similarly uses two patterns plus a pre-2006 layout, all handled.

### 2.3 Caching

Raw HTTP bytes (HTML + PDF), extracted plain text, and the FRED VIX series are all cached under `data/raw/`. Re-runs of the scraper are idempotent and skip any document whose text file already exists.

### 2.4 Target (VIX)

Pulled from FRED via `fredapi` (series `VIXCLS`, the official CBOE VIX closing value as redistributed by the St. Louis Fed). FRED is preferred over third-party tickers because the series is reproducible from `(API key, ticker)` alone.

**Alignment rule (next-trading-day):** For each document release date `r`, let `t` be the first trading day with `t >= r`. The 3-day forward change is `VIX[t+3] - VIX[t]` where the index advances by *trading* days (so a Friday release rolls to the following Wednesday). The "next" trading day (not "previous") is used because the previous would introduce look-ahead leakage (using market data from before the document was released).

## 3. Text preprocessing

### 3.1 Sentence segmentation

NLTK's Punkt sentence tokenizer (Kiss & Strunk, 2006) splits each document's plain-text body into sentences. Punkt is the academic-NLP standard for unsupervised sentence boundary detection on formal prose and handles common abbreviations (U.S., Mr., Dr.) reasonably well, which matters for Fed text.

### 3.2 Sentence cap

Each document is truncated to the **first 80 sentences** before encoding. Three motivations:

1. **Signal concentration.** Opening sections of FOMC minutes and HH testimony contain the policy rationale and forward-looking language that markets react to; later sections (member-by-member voting records, operational appendices) are well-known to be lower signal in the Fed-text-mining literature (Hansen, McMahon & Prat 2018; Curti & Kazinnik 2023).
2. **Memory tractability.** 326 documents x 80 sentences x 768-dim embeddings is about 80 MB, easy to pre-compute once and cache on disk.
3. **Batching simplicity.** A uniform cap lets us materialize a single `(B, 80, 768)` tensor with a 0/1 mask, which is simpler than ragged per-doc tensors.

The cap is enforced at the sentence boundary, never mid-text.

## 4. Encoder: frozen FinBERT with mask-weighted mean pooling

### 4.1 Why FinBERT (not generic BERT)

`yiyanghkust/finbert-pretrain` (Yang, Uy & Huang, 2020) is BERT-base initialized from `bert-base-uncased` and further pretrained on 4.9B tokens of financial corpora (analyst reports, earnings calls, financial news). Domain-pretrained transformer embeddings consistently beat generic embeddings on downstream financial NLP tasks, because the in-domain pretraining exposes the model to specialized terminology and discourse patterns (interest-rate jargon, balance-sheet language, hawkish/dovish stance markers) that are rare in Wikipedia plus BookCorpus. For Fed minutes specifically, FinBERT has been used as a frozen feature extractor by Goessi et al. (2023) and as a sentiment backbone by Huang et al. (2023).

We deliberately use `finbert-pretrain` (the MLM-pretrained variant) rather than `finbert-tone` (the sentiment-fine-tuned variant), because we want general financial-language representations, not the sentiment classification head's bias.

The repository ships only model weights, no tokenizer files, so the tokenizer is loaded from `bert-base-uncased` directly (FinBERT uses the same 30,522-token WordPiece vocabulary).

### 4.2 Why fully frozen

All FinBERT parameters are frozen during training (`requires_grad=False`). With about 233 training documents, fine-tuning a 110M-parameter BERT is hopelessly under-data; it would overfit catastrophically and discard the financial-domain pretraining signal. Peters, Ruder and Smith (2019) show that for small downstream datasets, feature-extraction (frozen backbone) often matches or beats fine-tuning, especially when the domain match is already strong. Frozen-encoder training is also strictly faster because sentence embeddings can be pre-computed once and cached (Section 6).

### 4.3 Why mean pooling, not `[CLS]`

For each sentence we extract the encoder's last-hidden-state tokens and compute their mask-weighted mean across the non-padding positions. We deliberately do *not* use the `[CLS]` token's embedding.

`[CLS]` is only a meaningful sentence representation for BERT models that have been fine-tuned on a sentence-classification objective. FinBERT-pretrain is MLM-only; on MLM-pretrained-only BERTs, `[CLS]` is a known-weak embedding, and mean pooling consistently outperforms it as a frozen sentence representation (Reimers & Gurevych, 2019, "Sentence-BERT," Section 3.3).

## 5. Document aggregation: additive (Bahdanau-style) attention

Sentence embeddings are aggregated into a single document vector via additive attention:

```
u_i    = tanh(W h_i + b)             # (attn_dim,)   per-sentence projection
s_i    = v^T u_i                     # scalar score per sentence
s_i   := s_i if mask_i = 1 else -inf  # padded sentences ignored by softmax
alpha  = softmax(s)                  # (N,) attention weights
d      = sum_i alpha_i * h_i         # (768,) document vector
y      = w^T d + b_reg               # scalar prediction (3-day VIX change)
```

Here `h_i` is the i-th sentence embedding (768-d), `W` is a `(128, 768)` matrix, `b` is a `(128,)` bias, and `v` is a `(128,)` query vector. About 99,000 trainable parameters total.

This is the Hierarchical Attention Network formulation of Yang et al. (2016), the canonical reference for "attend over sentence embeddings to make one document vector." The score function is the additive form from Bahdanau, Cho and Bengio (2014).

### 5.1 Why attention, not mean pooling

Fed documents are not uniform in signal. The opening paragraphs of FOMC minutes carry the policy rationale; later sections (committee voting records, operational appendices) carry essentially no signal for our target. Mean pooling weights all sentences equally and is therefore guaranteed to dilute. Attention learns the soft weighting we need. As a side benefit, the per-sentence attention weights are the project's main interpretability output: the attention-heatmap figure shows which sentences the model relied on for each prediction.

### 5.2 Why additive attention, not dot-product or multi-head

- **Dot-product attention with a learned query.** Simpler but lacks the per-sentence learned linear transform; less expressive when the underlying embeddings are not optimized for the regression target.
- **Single-layer self-attention with a `[DOC]` token.** About 2.4M parameters vs. 100k for additive, a brutal capacity-to-data ratio for 233 training documents.
- **Multi-head attention.** Multiplies parameter count without obvious benefit at this dataset size and adds explanatory burden.

Additive attention's parameter count (about 100k) is the sweet spot for this dataset size.

## 6. Embedding cache

Because the encoder is fully frozen and deterministic (no eval-mode dropout), each sentence's 768-d embedding is fully determined by its text. We pre-compute all ~25,000 sentence embeddings once (`scripts/precompute_embeddings.py`), save them to `data/processed/sentence_embeddings.pt`, and load from there during training. This speeds training by 2 to 3 orders of magnitude with no information leakage.

## 7. Train, val, test splits

All splits are strictly temporal; no random shuffling anywhere in the pipeline. Documents are sorted by `release_date` ascending and partitioned by date boundaries.

**Boundaries (anchored to U.S. presidential inauguration days):**
- `2017-01-20`, Trump 1 inauguration (train / test divide)
- `2021-01-20`, Biden inauguration
- `2025-01-20`, Trump 2 inauguration

**Partitions:**
- Train pool: `release_date < 2017-01-20` (about 233 documents)
- Val: last 15 percent chronologically of the train pool (about 35 documents); used only for early stopping
- Regime 1: `2017-01-20 <= date < 2021-01-20` (about 40 docs)
- Regime 2: `2021-01-20 <= date < 2025-01-20` (about 40 docs)
- Regime 3: `2025-01-20 <= date` (about 13 docs as of mid-2026)

Anchoring to inauguration dates makes RQ2 directly interpretable: each segment maps to one presidential administration.

## 8. Training procedure

Manual PyTorch loop. All hyperparameters live in `configs/default.yaml`:

- **Optimizer:** AdamW(lr=3e-4, weight_decay=1e-4) on the about 99,329 trainable parameters (attention plus linear head only). The learning rate is higher than the conventional BERT-fine-tuning 2e-5 because we are training a small head from scratch, not fine-tuning a large backbone.
- **LR schedule:** linear warmup over the first 10 percent of total steps, constant thereafter. Constant-after-warmup is preferred because early stopping on val MSE controls the actual stop point.
- **Gradient clipping:** `max_norm=1.0`.
- **Early stopping:** stop if val MSE has not improved for 5 epochs; max 30 epochs in the default config (extended to 100 epochs with patience 10 inside the hyperparameter sweep; see Section 10).
- **Batch size:** 8 documents.
- **Loss:** mean squared error on the regression target.

## 9. Evaluation

### 9.1 Metrics

- **MSE:** direct comparison vs. the baseline.
- **R^2:** fraction of variance explained; comparable across splits with differing target variances.
- **Pearson r:** direction-of-signal metric, scale-free.

### 9.2 Baseline

**TF-IDF plus Ridge regression:** unigram TF-IDF (min_df=2, max_df=0.95, sublinear_tf=True), Ridge(alpha=1.0). Same temporal splits. Fixed, conventional hyperparameters with no inner CV. The deep model gets one early-stopping signal; the baseline gets zero. This keeps the comparison honest.

### 9.3 Per-regime evaluation

The trained model is evaluated on each of regime1, regime2, regime3 separately. R^2 and Pearson r are reported per regime; RQ2 is answered descriptively by whether these decay across regimes and by the residuals-over-time figure.

## 10. Hyperparameter sweep

A small Cartesian sweep over `learning_rate` in {1e-4, 3e-4, 1e-3} crossed with `attn_dim` in {128, 256} (6 runs) is provided via `scripts/run_sweep.py` and `configs/sweep.yaml`. The sweep uses an extended 100-epoch budget with patience 10 so each candidate has room to converge. The winner is selected by lowest val MSE and promoted to `outputs/model.pt` for downstream evaluation, regime analysis, and figure generation.

## 11. Reproducibility

- All random seeds (Python, NumPy, PyTorch CPU and CUDA) are set from `configs/default.yaml`.
- Sentence embeddings are deterministic (frozen encoder, no eval-mode dropout).
- The full pipeline (`scripts/train.sbatch`) is idempotent: re-runs use cached scraped HTML, cached VIX, cached embeddings.
- The trained model checkpoint, per-epoch metrics, per-regime metrics, baseline metrics, and all figures are written under `outputs/`.

## 12. Selected references

- Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473.*
- Boukus, E., & Rosenberg, J. V. (2006). "The information content of FOMC minutes." *Federal Reserve Bank of New York working paper.*
- Curti, F., & Kazinnik, S. (2023). "Central Bank Communication and Inflation Expectations." *Working paper.*
- Goessi, S., Chen, Z., Kim, W., Bermeitinger, B., & Handschuh, S. (2023). "FinBERT-FOMC: Fine-Tuned FinBERT Model with Sentiment Focus Method for Analyzing FOMC Minutes." *ICAIF 2023.*
- Hansen, S., McMahon, M., & Prat, A. (2018). "Transparency and Deliberation Within the FOMC: A Computational Linguistics Approach." *Quarterly Journal of Economics*, 133(2): 801 to 870.
- Huang, A. H., Wang, H., & Yang, Y. (2023). "FinBERT: A Large Language Model for Extracting Information from Financial Text." *Contemporary Accounting Research.*
- Kiss, T., & Strunk, J. (2006). "Unsupervised Multilingual Sentence Boundary Detection." *Computational Linguistics*, 32(4): 485 to 525.
- Lucca, D. O., & Trebbi, F. (2009). "Measuring central bank communication: an automated approach with application to FOMC statements." *NBER working paper 15367.*
- Peters, M. E., Ruder, S., & Smith, N. A. (2019). "To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks." *RepL4NLP 2019.*
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP-IJCNLP 2019.*
- Whaley, R. E. (2000). "The Investor Fear Gauge." *Journal of Portfolio Management*, 26(3): 12 to 17.
- Yang, Y., Uy, M. C. S., & Huang, A. (2020). "FinBERT: A Pretrained Language Model for Financial Communications." *arXiv:2006.08097.*
- Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). "Hierarchical Attention Networks for Document Classification." *NAACL-HLT 2016.*
- Alexopoulos, M., Han, X., Kryvtsov, O., & Zhang, X. (2023). "More than Words: Fed Chairs' Communication During Congressional Testimonies." *Bank of Canada Staff Working Paper.*
