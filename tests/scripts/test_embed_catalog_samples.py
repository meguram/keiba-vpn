"""DATASET_CATALOG.html へのサンプル埋め込み（embed_catalog_samples）のテスト。"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

import pytest

from src.scripts.docs import embed_catalog_samples as M


@pytest.fixture()
def catalog(tmp_path):
    p = tmp_path / "DATASET_CATALOG.html"
    shutil.copy(M.CAT, p)
    return p


def test_every_dataset_row_gets_a_collapsible_sample(catalog):
    M.embed(catalog)
    s = catalog.read_text(encoding="utf-8")
    sec = s[s.index('<h2 id="catalog">'):s.index('<h2 id="matrix">')]
    ids = re.findall(r'<tr><td class="id">([A-G]\d\d)</td>|<tr class="ds" data-id="([A-G]\d\d)">', sec)
    ids = {a or b for a, b in ids}
    assert len(ids) == 54 and ids == set(re.findall(r'<tr class="ds" data-id="([A-G]\d\d)">', sec))
    assert sec.count('<tr class="sample">') == 54 and sec.count("<details>") == 54        # 行ごとに折り畳み
    assert "すべて開く" in sec and "tr.ds" in s and "addEventListener('click'" in s


def test_embedding_is_idempotent(catalog):
    M.embed(catalog)
    first = catalog.read_text(encoding="utf-8")
    M.embed(catalog)
    assert catalog.read_text(encoding="utf-8") == first


def test_json_samples_are_dicts_and_valid_json(catalog):
    M.embed(catalog)
    s = catalog.read_text(encoding="utf-8")
    blocks = [html.unescape(b) for b in re.findall(r'<pre class="smp"><code>(.*?)</code></pre>', s, re.S)]
    assert len(blocks) > 40
    kinds = [type(json.loads(b)) for b in blocks]
    assert kinds.count(dict) >= len(blocks) - 1                                          # ほぼすべて辞書型（最上位がリストの実ファイルは 1 つだけ）
    assert s.count("辞書型（dict）") == kinds.count(dict) and "リスト（list）" in s
    assert '<table class="smp">' in s                                                    # 特徴量などは表


def test_samples_are_schema_conformant_and_secrets_are_masked():
    from src.scraper import schema_infer, schemas

    S = M.build_samples()
    ids = {f"{g}{i:02d}" for g, n in zip("ABCDEFG", (18, 4, 7, 9, 6, 5, 5)) for i in range(1, n + 1)}
    assert set(S) == ids
    env = next(p for p in S["E06"] if p["kind"] == "json")["data"]
    assert env["APP_SECRET_KEY"] == "<秘密>" and env["GCS_PRIVATE_KEY"] == "<秘密>"      # 秘密の値は載せない
    for name, cat in schema_infer.SAMPLE_CATEGORY.items():                               # 使っている実サンプルは現行スキーマに適合
        data = json.loads((M.SAMP / f"{name}.json").read_text(encoding="utf-8"))
        assert schemas.validate(cat, data)["passed"], name
    for panels in S.values():                                                            # モックの時刻は固定（再生成しても差分が出ない）
        for q in panels:
            if q.get("source") == M.MOCK:
                assert "2026-10-06 00:00:00" in json.dumps(q["data"]) or "scraped_at_jst" not in json.dumps(q["data"])
