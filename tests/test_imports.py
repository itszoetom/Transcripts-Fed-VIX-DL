"""Smoke tests: package imports, default-config schema, model wiring.

Kept intentionally minimal, these are CI-style sanity checks, not unit tests
of model behavior. They confirm:

    1. All sub-packages import.
    2. configs/default.yaml has the schema the scripts expect.
    3. SentenceAttentionModel can be instantiated and runs a forward pass on
       a tiny fake batch (proves the model graph is wired up correctly).
"""

from pathlib import Path

import torch
import yaml


def test_package_imports():
    import transcripts_fed_vix  # noqa: F401
    from transcripts_fed_vix import data, models, training, utils  # noqa: F401


def test_default_config_schema():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    # Top-level sections.
    for key in ("seed", "data", "splits", "model", "training",
                "regime_analysis", "outputs"):
        assert key in cfg, f"missing top-level key: {key}"

    # Critical leaf values, anything the scripts dereference.
    assert cfg["data"]["sentence_cap"] == 80
    assert cfg["data"]["target_horizon_trading_days"] == 3
    assert cfg["splits"]["train_end_date"] == "2017-01-20"
    assert cfg["splits"]["regime2_start_date"] == "2021-01-20"
    assert cfg["model"]["freeze_encoder"] is True
    assert cfg["model"]["encoder_checkpoint"] == "yiyanghkust/finbert-pretrain"
    assert cfg["model"]["attn_dim"] > 0


def test_sentence_attention_model_forward():
    """Verify the trained model can run a forward pass on a tiny fake batch.

    Uses random embeddings so this test does NOT depend on transformers/FinBERT
    being installed, it only exercises the trained-component graph.
    """
    from transcripts_fed_vix.models import SentenceAttentionModel
    from transcripts_fed_vix.models.attention import AttentionConfig

    cfg = AttentionConfig(embed_dim=16, attn_dim=8, dropout=0.0)  # tiny for the test
    model = SentenceAttentionModel(cfg)

    B, N, E = 2, 5, 16
    embeddings = torch.randn(B, N, E)
    # mask: doc 0 has 5 real sentences, doc 1 has 3 real + 2 pad
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    out = model(embeddings, mask)
    assert out.prediction.shape == (B,)
    assert out.attention_weights.shape == (B, N)
    assert out.doc_vector.shape == (B, E)
    # Attention on padded positions should be exactly zero.
    assert torch.allclose(out.attention_weights[1, 3:], torch.zeros(2))
    # Attention weights should sum to ~1 across real sentences (modulo the
    # zero-masked positions which were already softmaxed away).
    assert torch.allclose(out.attention_weights.sum(dim=-1), torch.ones(B), atol=1e-5)


if __name__ == "__main__":
    test_package_imports()
    test_default_config_schema()
    test_sentence_attention_model_forward()
    print("SUCCESS: smoke tests pass")
