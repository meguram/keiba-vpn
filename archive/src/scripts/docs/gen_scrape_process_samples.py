"""
`docs/requirements/data/scrape_process.md` の「保存 JSON のサンプル」マーカー区間を、
`docs/requirements/data/scrape_process_samples/*.json` の内容から再生成する。

各 `<details>` ブロックは **要件表の1行（row_id）** に対応する。
行固有派生カテゴリ（race_shutuba_meta 等）は `src/scraper/row_data_extractor.py` の
抽出関数をキャッシュ JSON に適用して生成する。

実データの入れ方（いずれか）:

1. **ローカル L2 / page_reference から取り込み**（キャッシュ済み JSON のみコピー）::

     python3 -m src.scripts.docs.gen_scrape_process_samples --from-cache

2. **要件サンプル手動テストで取得しつつ書き出し**（netkeiba ログイン要・未取得は API 取得）::

     python3 tests/scraper/manual/requirements_sample_scrape_test.py --export-samples

   上記は各スクレイプの戻り値を JSON に保存したうえで、本モジュールの Markdown 更新を呼ぶ。

サンプル ID は `docs/requirements/data/scrape_process.md` の config 節に合わせる:
  race_date=20230625, race_id=202309030811, horse_id=2019105219
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

RACE_ID = "202309030811"
LIST_DATE = "20230625"
HORSE_ID = "2019105219"

_REPO = Path(__file__).resolve().parents[3]
_SAMPLES = _REPO / "docs" / "requirements" / "data" / "scrape_process_samples"
_DOC = _REPO / "docs" / "requirements" / "data" / "scrape_process.md"
_BEGIN = "<!-- SCRAPE_PROCESS_SAMPLES_AUTO_BEGIN -->"
_END = "<!-- SCRAPE_PROCESS_SAMPLES_AUTO_END -->"


# ---------------------------------------------------------------------------- #
# 行単位スペック
# (row_id, display_title, target_category, key_selector)
# key_selector: "race" → RACE_ID / "horse" → HORSE_ID / "date" → LIST_DATE
# target_category: 最終的に表示する行固有カテゴリ（派生カテゴリを含む）
# ---------------------------------------------------------------------------- #

_ROW_SAMPLE_SPECS: list[tuple[str, str, str, str]] = [
    ("nk_shutuba_entries",    "出馬表HTML（`race_shutuba`）",                "race_shutuba",                "race"),
    ("nk_shutuba_race_meta",  "レース情報HTML（`race_shutuba_meta`）",        "race_shutuba_meta",           "race"),
    ("nk_speed_index",        "タイム指数HTML（`race_index`）",               "race_index",                  "race"),
    ("nk_barometer",          "調子偏差値HTML（`race_barometer`）",           "race_barometer",              "race"),
    ("nk_paddock",            "パドックHTML（`race_paddock`）",               "race_paddock",                "race"),
    ("nk_odds",               "オッズHTML（`race_odds`）",                    "race_odds",                   "race"),
    ("nk_result_on_time",     "結果HTML（`race_result_on_time`）",            "race_result_on_time",         "race"),
    ("nk_payoff_html",        "払戻HTML 速報（`race_result_on_time_payoff`）","race_result_on_time_payoff",  "race"),
    ("nk_lap_html",           "ラップHTML 速報（`race_result_on_time_lap`）", "race_result_on_time_lap",     "race"),
    ("nk_corner_html",        "通過順位HTML 速報（`race_result_on_time_corner`）","race_result_on_time_corner","race"),
    ("nk_per_horse_lap_html", "個別ラップHTML（`race_result_lap`）",          "race_result_lap",             "race"),
    ("nk_horse_profile",      "馬プロフィール（`horse_profile`）",            "horse_profile",               "horse"),
    ("nk_horse_history",      "馬過去成績（`horse_race_history`）",           "horse_race_history",          "horse"),
    ("nk_horse_pedigree",     "馬血統データ（`horse_pedigree_5gen`）",        "horse_pedigree_5gen",         "horse"),
    ("nk_horse_training",     "馬調教（`horse_training`）",                   "horse_training",              "horse"),
    ("nk_race_list",          f"レースID一覧（`race_lists`・`{LIST_DATE}`）", "race_lists",                  "date"),
    ("nk_race_day_schedule",  f"発走時刻表（`race_day_schedule`・`{LIST_DATE}`）","race_day_schedule",        "date"),
    ("nk_db_race_result",     "結果DB（`race_result`）",                      "race_result",                 "race"),
    ("nk_db_race_info",       "レース情報DB（`race_result_meta`）",           "race_result_meta",            "race"),
    ("nk_db_payoff",          "払戻DB（`race_result_payoff`）",               "race_result_payoff",          "race"),
    ("nk_db_track",           "馬場情報DB（`race_result_track`）",            "race_result_track",           "race"),
    ("nk_db_corner",          "通過順位DB（`race_result_corner`）",           "race_result_corner",          "race"),
    ("nk_db_lap",             "ラップDB（`race_result_lap_times`）",          "race_result_lap_times",       "race"),
    ("nk_db_per_horse_lap",   "個別ラップDB（`race_result_lap`）",            "race_result_lap",             "race"),
]

# backward-compat: _DETAILS_ORDER を参照する外部モジュールがあればそのまま維持
_DETAILS_ORDER: list[tuple[str, str]] = [
    (row_id, title) for row_id, title, _cat, _sel in _ROW_SAMPLE_SPECS
]


def _key_for_selector(sel: str) -> str:
    if sel == "race":
        return RACE_ID
    if sel == "horse":
        return HORSE_ID
    return LIST_DATE


def _dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _details_block(title: str, body_json: str) -> str:
    return (
        f"<details>\n<summary>{title}</summary>\n\n"
        f"```json\n{body_json.rstrip()}\n```\n\n</details>\n\n"
    )


def _local_cache_json_path(category: str, key: str) -> Path:
    from src.scraper.storage import HybridStorage

    st = HybridStorage(base_dir=str(_REPO))
    return st._local_cache_path(category, key)


def _race_lists_json_path(date_key: str) -> Path:
    return _REPO / "data" / "page_reference" / "race_lists" / f"{date_key}.json"


def _race_day_schedule_json_path(date_key: str) -> Path:
    return _REPO / "data" / "page_reference" / "race_day_schedule" / f"{date_key}.json"


def _load_source_json(category: str, key: str) -> dict[str, Any] | None:
    """L2 キャッシュ / page_reference から JSON を読む。"""
    if category == "race_lists":
        p = _race_lists_json_path(key)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        return None
    if category == "race_day_schedule":
        p = _race_day_schedule_json_path(key)
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        # スナップショット不在時は合成
        try:
            from src.scraper.storage import HybridStorage
            from src.scraper.race_day_schedule import synthesize_race_day_schedule_payload
            st = HybridStorage(base_dir=str(_REPO))
            return synthesize_race_day_schedule_payload(st, key)
        except Exception:
            return None
    p = _local_cache_json_path(category, key)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _generate_row_json(
    row_id: str,
    target_category: str,
    key: str,
) -> dict[str, Any] | None:
    """
    target_category が派生カテゴリの場合は、ソース JSON から抽出して返す。
    それ以外はキャッシュから直接ロードする。
    """
    from src.scraper.row_data_extractor import DERIVED_CATEGORY_MAP

    # 1. 派生カテゴリなら: ソースからの抽出を試みる
    if target_category in DERIVED_CATEGORY_MAP:
        source_category, extract_fn = DERIVED_CATEGORY_MAP[target_category]
        source_data = _load_source_json(source_category, key)
        if source_data is not None:
            try:
                return extract_fn(source_data)
            except Exception:
                pass
        return None

    # 2. 通常カテゴリ: キャッシュ直読み
    return _load_source_json(target_category, key)


def import_samples_from_workspace_cache() -> dict[str, str]:
    """
    L2 キャッシュ / page_reference / 派生抽出から行単位 JSON を生成し
    `scrape_process_samples/{row_id}.json` へ書き出す。

    Returns:
        row_id -> ステータス説明
    """
    _SAMPLES.mkdir(parents=True, exist_ok=True)
    for p in _SAMPLES.glob("*.json"):
        p.unlink()
    results: dict[str, str] = {}

    for row_id, _title, target_category, sel in _ROW_SAMPLE_SPECS:
        key = _key_for_selector(sel)
        data = _generate_row_json(row_id, target_category, key)
        if data is None:
            results[row_id] = "NOT FOUND"
            continue
        dst = _SAMPLES / f"{row_id}.json"
        dst.write_text(_dump(data), encoding="utf-8")
        results[row_id] = f"OK → {target_category}/{key}"

    return results


def build_markdown(samples_dir: Path) -> str:
    parts = [
        _BEGIN,
        "",
        "以下は **実際のスクレイプ保存 JSON**（ローカル L2 `data/cache/` または `requirements_sample_scrape_test.py --export-samples` の戻り値を `scrape_process_samples/` に書き出したもの）を整形したものです。",
        f"- レース ID: `{RACE_ID}`",
        f"- 一覧日付: `{LIST_DATE}`",
        f"- 馬 ID: `{HORSE_ID}`",
        "",
    ]
    missing: list[str] = []
    for row_id, title, _cat, _sel in _ROW_SAMPLE_SPECS:
        p = samples_dir / f"{row_id}.json"
        if not p.is_file():
            missing.append(row_id)
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            parts.append(f"<!-- {row_id}.json 読込失敗: {e} -->\n")
            continue
        parts.append(_details_block(f"<code>{row_id}</code> {title}", _dump(data)))

    if missing:
        parts.append(
            "<!-- 次の行はサンプルディレクトリに .json が無いため省略: "
            + ", ".join(missing)
            + " — `python3 tests/scraper/manual/requirements_sample_scrape_test.py --export-samples` "
            "で未取得分を取得してから再実行してください。 -->\n\n"
        )

    parts.append(_END)
    return "\n".join(parts)


def _ensure_row_id_samples(samples_dir: Path) -> None:
    """
    `--export-samples` が書き出したカテゴリ名 JSON（`race_shutuba.json` 等）が存在するが
    row_id 名 JSON（`nk_shutuba_entries.json` 等）がない場合に、変換・生成する。

    既に row_id 名ファイルが揃っている場合は何もしない。
    """
    from src.scraper.row_data_extractor import DERIVED_CATEGORY_MAP

    for row_id, _title, target_category, sel in _ROW_SAMPLE_SPECS:
        dst = samples_dir / f"{row_id}.json"
        if dst.exists():
            continue  # すでに生成済み

        key = _key_for_selector(sel)

        # 1. target_category のカテゴリファイルが直接あれば使う
        src_file = samples_dir / f"{target_category}.json"
        if src_file.exists():
            dst.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
            continue

        # 2. 派生カテゴリの場合: ソースカテゴリファイルから抽出
        if target_category in DERIVED_CATEGORY_MAP:
            source_category, extract_fn = DERIVED_CATEGORY_MAP[target_category]
            source_file = samples_dir / f"{source_category}.json"
            if source_file.exists():
                try:
                    source_data = json.loads(source_file.read_text(encoding="utf-8"))
                    derived = extract_fn(source_data)
                    dst.write_text(_dump(derived), encoding="utf-8")
                    continue
                except Exception:
                    pass

        # 3. special cases: race_lists / race_day_schedule はローカル保存
        if target_category in ("race_lists", "race_day_schedule"):
            data = _load_source_json(target_category, key)
            if data is not None:
                dst.write_text(_dump(data), encoding="utf-8")


def patch_scrape_process_doc(samples_dir: Path | None = None) -> None:
    samples_dir = samples_dir or _SAMPLES
    # カテゴリ名ファイルから row_id 名ファイルへ変換（--export-samples 連携）
    _ensure_row_id_samples(samples_dir)
    text = _DOC.read_text(encoding="utf-8")
    if _BEGIN not in text or _END not in text:
        raise SystemExit(
            f"マーカー {_BEGIN} … {_END} が {_DOC} にありません。"
        )
    pre, rest = text.split(_BEGIN, 1)
    _, post = rest.split(_END, 1)
    new_body = build_markdown(samples_dir)
    _DOC.write_text(pre + new_body + post, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-cache",
        action="store_true",
        help="data/cache と data/page_reference から行単位 JSON を生成して MD 更新",
    )
    args = ap.parse_args()

    if args.from_cache:
        results = import_samples_from_workspace_cache()
        ok = sum(1 for v in results.values() if v.startswith("OK"))
        nf = sum(1 for v in results.values() if v == "NOT FOUND")
        print(f"行単位 JSON 生成: OK={ok} / NOT_FOUND={nf}")
        for k, v in sorted(results.items()):
            if v != "NOT FOUND":
                print(f"  {k}: {v}")
        if nf:
            print("  (NOT FOUND の行はキャッシュなし)")

    patch_scrape_process_doc()
    print(f"更新: {_DOC}")


if __name__ == "__main__":
    main()
