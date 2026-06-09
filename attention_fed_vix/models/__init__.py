"""Models subpackage: frozen FinBERT encoder and learned sentence-attention head.

Public surface:
    encoder.FrozenFinBERTEncoder
    attention.SentenceAttentionModel
"""

from .encoder import FrozenFinBERTEncoder
from .attention import SentenceAttentionModel

__all__ = ["FrozenFinBERTEncoder", "SentenceAttentionModel"]
