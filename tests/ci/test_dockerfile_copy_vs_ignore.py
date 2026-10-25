"""Dockerfile の COPY 元が、同じイメージの dockerignore で除外されていないこと（除外されていると docker build が '"/x": not found' で失敗する）。

2026-10-26: Dockerfile.jobs の `COPY main.py` が Dockerfile.jobs.dockerignore の `main.py` に当たり、jobs イメージが作られていなかった。
"""
import fnmatch
import re
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[2] / "deploy/docker"
FILES = ["Dockerfile.cloudrun", "Dockerfile.jobs"]


def _copy_sources(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(.+)", line)
        if not m:
            continue
        parts = [p for p in m.group(1).split() if not p.startswith("--")]
        out += [p.rstrip("/") for p in parts[:-1]]          # 最後は COPY 先
    return out


def _ignored(src: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        if pat.startswith("!"):
            continue
        p = pat.rstrip("/")
        if fnmatch.fnmatch(src, p) or src == p or src.startswith(p + "/"):
            return pat
    return None


@pytest.mark.parametrize("name", FILES)
def test_copy_sources_are_not_dockerignored(name):
    df = DOCKER / name
    ign = DOCKER / f"{name}.dockerignore"
    assert ign.is_file(), f"{ign.name} がありません"
    pats = [l.strip() for l in ign.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]
    for src in _copy_sources(df.read_text(encoding="utf-8")):
        hit = _ignored(src, pats)
        assert hit is None, f"{name}: COPY {src} は {ign.name} の `{hit}` で除外されていて、docker build が失敗する"
