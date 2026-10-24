import pytest

from src.config.deployment import require_explicit_env


def test_empty_keiba_env_is_rejected(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "")
    monkeypatch.delenv("KEIBA_ALLOW_IMPLICIT_PROD", raising=False)
    with pytest.raises(SystemExit):
        require_explicit_env()


def test_unset_keiba_env_is_rejected(monkeypatch):
    monkeypatch.delenv("KEIBA_ENV", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("KEIBA_ALLOW_IMPLICIT_PROD", raising=False)
    with pytest.raises(SystemExit):
        require_explicit_env()


@pytest.mark.parametrize("value", ["dev", "stg", "prod"])
def test_explicit_env_is_accepted(monkeypatch, value):
    monkeypatch.setenv("KEIBA_ENV", value)
    require_explicit_env()


def test_implicit_prod_escape_hatch(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "")
    monkeypatch.setenv("KEIBA_ALLOW_IMPLICIT_PROD", "1")
    require_explicit_env()
