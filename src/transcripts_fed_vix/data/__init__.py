"""Data subpackage: scraping, sentence segmentation, VIX alignment, dataset.

Public surface:
    scrape.scrape_fomc_minutes, scrape.scrape_humphrey_hawkins
    segment.segment_document, segment.SENTENCE_CAP
    vix.fetch_vix, vix.align_to_next_trading_day, vix.compute_forward_change
    dataset.EmbeddingDocDataset, dataset.collate_padded
    build.build_processed_dataset
"""

from .segment import segment_document, SENTENCE_CAP
from .dataset import get_example_dataloader, load_example_documents

__all__ = [
    "segment_document",
    "SENTENCE_CAP",
    "get_example_dataloader",
    "load_example_documents",
]
