"""Unit tests for .env loading and config handling."""

from web_harness.config.loader import config_hash, load_env_file
from web_harness.core.errors import ConfigError


def test_load_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nMODEL_BASE_URL=https://example.com/v1\n\nMODEL_API_KEY='sk-test'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    assert load_env_file(env_file) is True
    import os

    assert os.environ["MODEL_BASE_URL"] == "https://example.com/v1"
    assert os.environ["MODEL_API_KEY"] == "sk-test"


def test_load_env_file_existing_env_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_BASE_URL=https://file.example\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_BASE_URL", "https://shell.example")
    assert load_env_file(env_file) is True
    import os

    assert os.environ["MODEL_BASE_URL"] == "https://shell.example"


def test_load_env_file_missing(tmp_path):
    assert load_env_file(tmp_path / "nope.env") is False


def test_config_hash_stable():
    a = {"x": 1, "y": [1, 2]}
    b = {"y": [1, 2], "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 2, "y": [1, 2]})


def test_harness_config_validation(tmp_path):
    from web_harness.config.loader import HarnessConfig

    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  provider: nope\n", encoding="utf-8")
    try:
        HarnessConfig.from_yaml(bad)
    except ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")
