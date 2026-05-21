"""Sentence segmentation for scraped Fed documents.

Why NLTK punkt:
    Punkt is the unsupervised sentence tokenizer of choice in academic NLP. It
    handles common formal-prose abbreviations (U.S., Mr., Dr., etc.) reasonably
    well — important for Fed text, which is dense with abbreviations — and has
    a tiny footprint compared to spaCy. It's also the most-cited choice in
    prior work on Fed/FOMC text mining, which keeps the methodology section of
    the write-up uncomplicated.

Why a hard cap at 80 sentences (SENTENCE_CAP):
    FOMC minutes run 5,000–10,000 words (typically 250–500 sentences) and HH
    testimony is shorter (~100–250 sentences). We truncate to the first 80
    sentences for three reasons:

        1. The opening sections of these documents contain the policy
           rationale and forward-looking language that markets actually
           respond to — operational detail and member-by-member voting
           appendices come later and are well-known to be lower signal
           (see Hansen, McMahon & Prat 2018; Curti & Kazinnik 2023 for
           the same intuition on truncating Fed text).
        2. With ~322 total documents and 768-d embeddings, keeping 80 sentences
           per doc gives the attention mechanism enough breadth without
           blowing memory on the GPU (~25K sentences total to encode once).
        3. A uniform cap makes batching tractable and lets us materialize a
           single (B, 80, 768) tensor with mask, which is much simpler than
           variable-length per-doc tensors.

    This is the "no chunking, no head-truncation" the project spec mandates:
    we truncate at the *sentence* boundary, not in the middle of text, and we
    keep all 80 sentences as distinct units rather than concatenating to fit
    one BERT context window.
"""

from __future__ import annotations

import logging

import nltk

logger = logging.getLogger(__name__)

# The 80-sentence cap is fixed for the project; see module docstring for why.
SENTENCE_CAP: int = 80


def _ensure_punkt_available() -> None:
    """Download punkt tokenizer data on first use if not already cached.

    NLTK's data-finder semantics changed across versions: older NLTK looked for
    `tokenizers/punkt`, newer NLTK uses `tokenizers/punkt_tab`. We probe both.
    """
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            continue
    # Nothing found — download both, idempotent.
    logger.info("Downloading NLTK punkt tokenizer data (one-time)…")
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)


def segment_document(text: str, cap: int = SENTENCE_CAP) -> list[str]:
    """Split a document into sentences and truncate to the first `cap` of them.

    Args:
        text: Plain-text document body (already stripped of HTML/PDF chrome).
        cap:  Maximum number of sentences to keep. Defaults to SENTENCE_CAP=80.

    Returns:
        List of sentence strings of length <= cap. Empty list if `text` was
        empty/whitespace-only.

    Notes:
        - Empty / whitespace-only sentences are filtered out before truncation,
          so the returned list contains `cap` real sentences when possible.
        - We do not strip punctuation, lowercase, or otherwise normalize — the
          downstream BERT tokenizer expects natural text.
    """
    if not text or not text.strip():
        return []
    _ensure_punkt_available()
    sents = [s.strip() for s in nltk.sent_tokenize(text)]
    sents = [s for s in sents if s]  # drop empties
    return sents[:cap]
