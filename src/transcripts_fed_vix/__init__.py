"""transcripts_fed_vix: Predict 10-day VIX change from Federal Reserve text.

Architecture overview (see README + configs/default.yaml for details):

    raw FOMC minutes / Humphrey-Hawkins testimony
        -> sentence segmentation (NLTK punkt, first 80 sentences)
            -> frozen FinBERT encoder (yiyanghkust/finbert-pretrain)
                -> mask-weighted mean pooling per sentence
                    -> additive (Bahdanau-style) attention over sentences  [TRAINED]
                        -> linear regression head                          [TRAINED]
                            -> predicted 10-day forward VIX change

Only the attention aggregator and the regression head are trained; the FinBERT
encoder is fully frozen so that sentence embeddings are deterministic and can
be cached to disk before training (large speedup, no information leakage).

Target: 10-day close-to-close VIX change after each document's release date,
aligned to the next available trading day when the release falls on a weekend
or U.S. market holiday. Temporal splits respect calendar order, no random
shuffling anywhere in the pipeline.
"""

__version__ = "0.1.0"
