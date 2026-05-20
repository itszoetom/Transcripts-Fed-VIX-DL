"""Smoke test: package imports and basic instantiation work."""
import yaml
from pathlib import Path


def test_package_imports():
    import transcripts_fed_vix
    from transcripts_fed_vix import data, models, training, utils


def test_default_config_parses():
    cfg_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    assert "data_dir" in cfg
    assert "pretrained_model" in cfg


if __name__ == "__main__":
    test_package_imports()
    test_default_config_parses()
    print("SUCCESS: smoke tests pass")
