"""docs/ の HTML と、コード・設定の事実がずれていないかのテスト。

文書が古くなる主な原因（パスの移動、件数・ポートの手書き、既定値の変更）を、機械的に検出する。
意図的に実在しないパスを書く場合（予定のファイル、削除済みの旨の記載、生成されるファイル、別プロジェクトのファイル）は ALLOW_MISSING に理由つきで足す。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

# 実在しなくてよい参照（理由）
ALLOW_MISSING = {
    "config/exotic_alpha.json": "validate_exotic_probabilities --write-config で生成される",
    "config/pace_type_models.json": "学習済みの重みは公開リポジトリに置かない（各環境に配置。docs/operations/model_distribution.md）",
    "config/expected_megu/model.json": "学習済みの重みは公開リポジトリに置かない（各環境に配置。docs/operations/model_distribution.md）",
    "scripts/cron/setup_baba_cron.sh": "削除済みである旨の記載",
    "scripts/cron/update_entity_stats_snapshot.sh": "決定記録 AREA-13 の当時の記述",
    "src/app.py": "別プロジェクト(daily_news)のファイルの記述",
    "src/pipeline/features/persona_umauma_features.py": "設計案（予定のファイル）",
    "src/pipeline/inference/persona_umauma_predict.py": "設計案（予定のファイル）",
    "src/pipeline/models/train_persona_umauma.py": "設計案（予定のファイル）",
    "src/pipeline/inference/market_baseline.py": "決定 DEC-026 の予定のファイル",
    "src/pipeline/mlflow/calibration_eval.py": "決定 DEC-026 の予定のファイル",
}
_PATH = re.compile(r"((?:docs/html/|docs/decisions/|docs/|src/|scripts/|tests/|config/)[A-Za-z0-9_./\-]+\.(?:py|sh|html|md|json|yaml|env))")


def _is_fitted(ref: str) -> bool:
    """公開リポジトリに置かない当て込み済みファイル（config/fitted_artifacts.txt）への言及は、実在しなくてよい。"""
    f = ROOT / "config" / "fitted_artifacts.txt"
    if not f.is_file():
        return False
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ref == line or (line.endswith("/") and ref.startswith(line)):
            return True
    return False


def _pages() -> list[Path]:
    return [p for p in DOCS.rglob("*.html") if p.name != "book.html" and "node_modules" not in p.parts]


def _text(p: Path) -> str:
    return re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)\b.*?</\1>", "", p.read_text(encoding="utf-8", errors="ignore"), flags=re.S))


def test_paths_mentioned_in_docs_exist():
    missing: dict[str, set[str]] = {}
    for p in _pages():
        for ref in set(_PATH.findall(_text(p))):
            if "*" in ref or "YYYY" in ref or ref in ALLOW_MISSING or _is_fitted(ref):
                continue
            if not (ROOT / ref).exists():
                missing.setdefault(ref, set()).add(str(p.relative_to(DOCS)))
    assert not missing, "文書内の参照パスが実在しない（移動・改名したら文書も直す）:\n" + "\n".join(f"  {k} ← {sorted(v)[:3]}" for k, v in sorted(missing.items()))


def test_category_count_claims_match_schema_defs():
    d = json.loads((ROOT / "src/scraper/schema_defs.json").read_text(encoding="utf-8"))
    n = len(d.get("categories", d))
    for rel in ("html/data/DATASET_CATALOG.html", "html/data/database_schemas.html"):
        txt = _text(DOCS / rel)
        claims = {int(x) for x in re.findall(r"(?:全|スキーマ定義済みの全)\s*(\d+)\s*カテゴリ", txt)}
        assert claims <= {n}, f"{rel}: 『全N カテゴリ』が実数 {n} と違う: {claims}"


def test_docs_do_not_claim_old_queue_parallel_default():
    from src.scraper import queue_load_settings as q

    assert q._default_parallel_env() == 4, "既定の並列数を変えたら文書（data_scraping_storage_reference・LAYER_DESIGN）も直す"
    for p in _pages():
        assert "SCRAPE_QUEUE_PARALLEL=6" not in _text(p).replace(" ", ""), f"{p.relative_to(DOCS)}: 並列数の既定が古い"


def test_server_docs_show_current_ports():
    env = dict(re.findall(r"^(KEIBA_PORT_[A-Z_]+)=(\d+)", (ROOT / "config/ports.env").read_text(), flags=re.M))
    txt = _text(DOCS / "operations/server_architecture.html")
    for key in ("KEIBA_PORT_DEV_FASTAPI", "KEIBA_PORT_STG_FASTAPI", "KEIBA_PORT_PROD_FASTAPI"):
        if key in env:
            assert env[key] in txt, f"server_architecture.html に {key}={env[key]} が無い"


@pytest.mark.parametrize("rel", ["html/design/ARCHITECTURE.html", "html/design/FULLSTACK_ARCHITECTURE.html", "html/design/LAYER_DESIGN.html"])
def test_architecture_docs_point_to_the_port_source_of_truth(rel):
    txt = _text(DOCS / rel)
    assert "ports.env" in txt, f"{rel}: ポートの正（config/ports.env）への言及が無い"


def test_every_page_has_a_last_updated_line():
    no = [str(p.relative_to(DOCS)) for p in (DOCS / "html").rglob("*.html") if p.name != "book.html" and "chapters" not in p.parts
          and not re.search(r"最終更新\s*[:：]?\s*20\d\d-\d\d-\d\d|更新日\s*[:：]?\s*20\d\d|作成\s*[:：]?\s*20\d\d|生成\s*[:：]?\s*20\d\d|20\d\d-\d\d-\d\d", _text(p))]
    assert not no, f"最終更新日の記載が無い: {no}"
