"""
血統×レース結果 フラットインデックス構築スクリプト（10世代対応版）

概要
----
data/local/tables/{year}/race_result_flat.parquet と
data/local/horse_pedigree_5gen/{prefix}/{horse_id}.json を結合し、
血統カテゴリ別の種牡馬フラットテーブルを生成する。

5世代表の gen5 牡アンカーにさらにその5世代表を接木することで最大10世代まで展開する。
（pipeline/horse_pedigree_expand.py の merge_primary_and_branches を使用）

血統カテゴリ定義（path_fm 経路文字列による判定）
----------------------------------------------
path_fm は主馬→祖先の F（父側）/ M（母側）経路文字列。
F=父系、M=母系。文字列長 = 世代数。

① cat=1 : 父（path_fm='F'）、父父（path_fm='FF'）
② cat=2 : 母父母（'MFM'）以降の父側 → path_fm が 'MFM' で始まり末尾が 'F' かつ長さ≥4
③ cat=3 : 母母（'MM'）以降の父側   → path_fm が 'MM' で始まり末尾が 'F' かつ長さ≥3
② と ③ は UI 上で統合表示（共通カテゴリとして扱う）

出力
----
data/research/pedigree_race_index/race_result_slim.parquet
  └ race_id, date, venue, surface, distance, grade, track_condition,
    finish_position, popularity, horse_id, horse_name,
    win_odds_real, win_payout, place_payout
    (※ payoff JSON から正規パースした単勝オッズ・払戻。元 odds 列は信頼できないため不使用)

data/research/pedigree_race_index/horse_pedigree_cats.parquet
  └ horse_id, cat (int 1/2/3), sub_cat, gen (世代数), stallion_id, stallion_name

Usage
-----
python -m src.research.pedigree.build_pedigree_race_index
python -m src.research.pedigree.build_pedigree_race_index --years 2020 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# リポジトリルートを sys.path に追加（pipeline モジュールを import するため）
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.features.horse_pedigree_expand import (  # noqa: E402
    iter_gen5_male_anchor_horse_ids,
    merge_primary_and_branches,
)

logger = logging.getLogger(__name__)

BASE_DIR = _ROOT

TABLES_DIR = BASE_DIR / "data" / "local" / "tables"
PED_DIR = BASE_DIR / "data" / "local" / "horse_pedigree_5gen"
OUT_DIR = BASE_DIR / "data" / "research" / "pedigree_race_index"

# ────────────────────────────────────────────────────────────────
# path_fm ベースの血統カテゴリ判定
#
# path_fm: 主馬→祖先の F（父側）/ M（母側）経路。F で終わる = 牡スロット。
# ────────────────────────────────────────────────────────────────


def _cat_from_path(path: str) -> int | None:
    """
    path_fm から血統カテゴリを返す。該当しない場合は None。

    cat=1: 'F'（父）, 'FF'（父父）
    cat=2: 'MFM' 始まり、末尾 'F'、長さ≥4（母父母以降の父系）
    cat=3: 'MM' 始まり、末尾 'F'、長さ≥3（母母以降の父系）
    """
    if not path or path[-1] != "F":
        return None
    if path in ("F", "FF"):
        return 1
    if path.startswith("MFM") and len(path) >= 4:
        return 2
    if path.startswith("MM") and len(path) >= 3:
        return 3
    return None


def _sub_cat_from_path(cat: int, path: str) -> str:
    """cat=1 は sire/grandsire、cat=2/3 は世代数で区別する。"""
    if cat == 1:
        return "sire" if path == "F" else "grandsire"
    return f"gen{len(path)}"


# ──────────────────────────────────
# 全血統 JSON をメモリキャッシュ
# ──────────────────────────────────

_PED_INDEX: dict[str, dict] | None = None


def _build_ped_index() -> dict[str, dict]:
    """PED_DIR 配下の全 *.json を horse_id → record のインデックスに展開する。"""
    global _PED_INDEX
    if _PED_INDEX is not None:
        return _PED_INDEX
    t0 = time.time()
    idx: dict[str, dict] = {}
    for f in sorted(PED_DIR.rglob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hid = str(rec.get("horse_id") or "").strip()
        if hid:
            idx[hid] = rec
    logger.info("血統インデックス構築: %d 頭 (%.1fs)", len(idx), time.time() - t0)
    _PED_INDEX = idx
    return idx


def _extract_stallion_entries_10gen(
    horse_id: str,
    ped_index: dict[str, dict],
) -> list[dict]:
    """
    5世代表に gen5 牡アンカーの5世代を接木した最大10世代から
    血統カテゴリ別の種牡馬エントリを返す。

    Returns list of dicts with keys:
        horse_id, cat, sub_cat, gen, stallion_id, stallion_name
    """
    rec = ped_index.get(horse_id)
    if not rec:
        return []

    primary_rows: list[dict] = rec.get("ancestors") or []

    # gen5 牡アンカーの枝表を収集
    anchors = iter_gen5_male_anchor_horse_ids(primary_rows)
    branch_by_anchor: dict[str, list[dict]] = {}
    for anchor_id, _ in anchors:
        anchor_rec = ped_index.get(anchor_id)
        if anchor_rec:
            branch_by_anchor[anchor_id] = anchor_rec.get("ancestors") or []

    # 主表 + 枝を統合（最大 gen10）
    merged = merge_primary_and_branches(
        horse_id, primary_rows, branch_by_anchor
    )

    rows: list[dict] = []
    seen_paths: set[str] = set()
    for row in merged:
        path = str(row.get("path_fm") or "")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        cat = _cat_from_path(path)
        if cat is None:
            continue
        stallion_id = str(row.get("horse_id") or "").strip()
        stallion_name = str(row.get("name") or "").strip()
        if not stallion_id and not stallion_name:
            continue
        rows.append({
            "horse_id": horse_id,
            "cat": cat,
            "path_fm": path,
            "gen": len(path),
            "stallion_id": stallion_id,
            "stallion_name": stallion_name,
        })
    return rows


RESULT_COLS = [
    "race_id", "horse_id", "horse_name", "horse_number",
    "finish_position", "popularity", "grade", "payoff", "odds",
]


# ────────────────────────────────────────────────────────────────
# payoff JSON から「単勝払戻」「複勝払戻 (1〜3 着順)」を抽出するパーサ
#
# 仕様メモ:
#   payoff 例: {"単勝": {"numbers":"3","payout":"480"},
#               "複勝": {"numbers":"342","payout":"130110140"}, ...}
#   payout 文字列は「100〜999 円は 3 桁」「1,000 円以上はカンマ付き 4 桁 (1,xxx)」
#   が連結された形式。numbers と payout の順番は **着順順**(1着→2着→3着)。
#   レース結果 parquet の `odds` / `popularity` 列は破損例があるため、ここで
#   payoff から逆算した単勝オッズ (= 単勝払戻 / 100) を真の値として採用する。
# ────────────────────────────────────────────────────────────────

_PAY_RE = __import__("re").compile(r"\d{1,2},\d{3}|\d{3}")


def _parse_payouts(s: str) -> list[int]:
    """payout 文字列から整数リストを抽出 (3 桁 or 1,xxx 形式)。"""
    if not s:
        return []
    return [int(x.replace(",", "")) for x in _PAY_RE.findall(str(s).strip())]


def _flatten_payout_field(v) -> str:
    """単勝/複勝フィールドが dict / list[dict] / None どれでも payout 文字列を連結して返す。

    同着のときに list 形式 [{numbers, payout}, ...] となるケースがあるため、
    要素の payout を空白区切りで連結する。
    """
    if not v:
        return ""
    if isinstance(v, dict):
        return str(v.get("payout") or "")
    if isinstance(v, list):
        return " ".join(str(d.get("payout") or "") for d in v if isinstance(d, dict))
    return str(v)


def _parse_race_payoff(p_str: str) -> dict | None:
    """payoff JSON 文字列から単勝/複勝の払戻金 (整数) を返す。

    Returns:
        {"win_payout": int|None,         # 1 着馬の単勝払戻 (100 円賭け)
         "place_payouts": list[int]}     # 1〜3 着馬の複勝払戻 (順番=着順、最大 3)
        パース失敗時は None。
    """
    if not p_str:
        return None
    try:
        po = json.loads(p_str)
    except (json.JSONDecodeError, TypeError):
        return None
    win_pays = _parse_payouts(_flatten_payout_field(po.get("単勝")))
    place_pays = _parse_payouts(_flatten_payout_field(po.get("複勝")))
    return {
        "win_payout": int(win_pays[0]) if win_pays else None,
        "place_payouts": place_pays[:3],
    }

# race_shutuba_flat から取得する race レベル情報（grade は result_flat の方が正確なため除外）
SHUTUBA_RACE_COLS = [
    "race_id", "date", "venue", "surface", "distance", "track_condition",
]


def build_index(years: list[str] | None = None) -> None:
    """インデックスを構築して parquet として保存する（10世代対応）。"""
    t0 = time.time()

    # ── レース結果読み込み ──
    all_years = sorted(
        d.name for d in TABLES_DIR.iterdir() if d.is_dir() and d.name.isdigit()
    ) if years is None else years

    result_dfs: list[pd.DataFrame] = []
    shutuba_dfs: list[pd.DataFrame] = []

    for yr in all_years:
        p_result = TABLES_DIR / yr / "race_result_flat.parquet"
        p_shutuba = TABLES_DIR / yr / "race_shutuba_flat.parquet"
        if not p_result.exists():
            logger.warning("race_result_flat.parquet が存在しません: %s", p_result)
            continue
        df_r = pd.read_parquet(p_result, columns=RESULT_COLS)
        result_dfs.append(df_r)

        if p_shutuba.exists():
            df_sh = pd.read_parquet(p_shutuba, columns=SHUTUBA_RACE_COLS)
            shutuba_dfs.append(df_sh.drop_duplicates("race_id"))
        else:
            logger.warning("race_shutuba_flat.parquet が存在しません: %s", p_shutuba)

        logger.info("  %s: result=%d行", yr, len(df_r))

    if not result_dfs:
        logger.error("レース結果ファイルが見つかりません")
        return

    result_df = pd.concat(result_dfs, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=["race_id", "horse_id"])

    # race_shutuba_flat から race レベルの正確な距離・会場情報を取得して JOIN
    if shutuba_dfs:
        shutuba_df = pd.concat(shutuba_dfs, ignore_index=True).drop_duplicates("race_id")
        race_df = result_df.merge(shutuba_df, on="race_id", how="left")
    else:
        logger.warning("race_shutuba_flat が存在しません。race_result_flat のみ使用します")
        add_cols = pd.read_parquet(
            TABLES_DIR / all_years[0] / "race_result_flat.parquet",
            columns=["race_id"] + SHUTUBA_RACE_COLS[1:],
        ).drop_duplicates("race_id")
        race_df = result_df.merge(add_cols, on="race_id", how="left")

    race_df = race_df.drop_duplicates(subset=["race_id", "horse_id"])
    logger.info("レース結果: %d 行 / ユニーク馬: %d", len(race_df), race_df["horse_id"].nunique())

    # ── 血統インデックス構築（全JSONをメモリに展開）──
    ped_index = _build_ped_index()

    unique_horse_ids = race_df["horse_id"].dropna().unique().tolist()
    logger.info("血統カテゴリ抽出開始: %d 頭（最大10世代）", len(unique_horse_ids))

    ped_rows: list[dict] = []
    missing = 0
    for i, hid in enumerate(unique_horse_ids):
        if i % 5000 == 0 and i > 0:
            logger.info("  進捗: %d / %d (%.0fs)", i, len(unique_horse_ids), time.time() - t0)
        entries = _extract_stallion_entries_10gen(str(hid), ped_index)
        if not entries:
            missing += 1
        ped_rows.extend(entries)

    logger.info(
        "血統エントリ: %d 件 (血統なし: %d 頭)",
        len(ped_rows), missing,
    )

    ped_df = pd.DataFrame(ped_rows)
    if ped_df.empty:
        logger.error("血統データが空です")
        return

    # ── カテゴリ最適化 ──
    ped_df["cat"] = ped_df["cat"].astype("int8")
    ped_df["gen"] = ped_df["gen"].astype("int8")
    ped_df["path_fm"] = ped_df["path_fm"].astype("category")
    ped_df["stallion_id"] = ped_df["stallion_id"].astype("category")
    # path_fm は1頭の血統ツリー内で一意な座標 → (horse_id, cat, path_fm) で重複除去
    ped_df = ped_df.drop_duplicates(subset=["horse_id", "cat", "path_fm"])

    # ── payoff JSON から「単勝払戻 / 複勝払戻」を導出 (race_id 単位で 1 回だけパース) ──
    if "payoff" in race_df.columns:
        unique_pf = race_df.drop_duplicates("race_id")[["race_id", "payoff"]]
        race_pay_map: dict[str, dict] = {}
        for _row in unique_pf.itertuples(index=False):
            parsed = _parse_race_payoff(getattr(_row, "payoff", None))
            if parsed:
                race_pay_map[str(_row.race_id)] = parsed

        win_payouts: list[int] = []
        place_payouts: list[int] = []
        for rid, fp in zip(race_df["race_id"].astype(str), race_df["finish_position"].astype(int)):
            d = race_pay_map.get(rid)
            if not d:
                win_payouts.append(0)
                place_payouts.append(0)
                continue
            wp = d.get("win_payout") or 0
            pps = d.get("place_payouts") or []
            win_payouts.append(int(wp) if fp == 1 else 0)
            if 1 <= fp <= 3 and len(pps) >= fp:
                place_payouts.append(int(pps[fp - 1]))
            else:
                place_payouts.append(0)
        race_df["win_payout"] = pd.Series(win_payouts, index=race_df.index, dtype="int32")
        race_df["place_payout"] = pd.Series(place_payouts, index=race_df.index, dtype="int32")
        # 「単勝オッズ」= 単勝払戻 / 100 (浮動小数)。1 着馬以外もそのレース全体の単勝オッズを保持
        win_odds_real = []
        for rid in race_df["race_id"].astype(str):
            d = race_pay_map.get(rid)
            if not d or not d.get("win_payout"):
                win_odds_real.append(0.0)
            else:
                win_odds_real.append(round(int(d["win_payout"]) / 100.0, 2))
        race_df["win_odds_real"] = pd.Series(win_odds_real, index=race_df.index, dtype="float32")
        race_df = race_df.drop(columns=["payoff"])  # raw JSON は不要 (重い)
        logger.info(
            "payoff 解析: %d レースから単勝/複勝払戻を抽出",
            len(race_pay_map),
        )
    else:
        logger.warning("payoff 列が無いため、回収率計算は不可")
        race_df["win_payout"] = 0
        race_df["place_payout"] = 0
        race_df["win_odds_real"] = 0.0

    # race_result_slim の型最適化
    for col in ["venue", "surface", "grade", "track_condition"]:
        if col in race_df.columns:
            race_df[col] = race_df[col].astype("category")
    race_df["distance"] = pd.to_numeric(race_df["distance"], errors="coerce").fillna(0).astype("int32")
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce").fillna(0).astype("int16")
    race_df["popularity"] = pd.to_numeric(race_df["popularity"], errors="coerce").fillna(0).astype("int16")
    if "odds" in race_df.columns:
        race_df["odds"] = pd.to_numeric(race_df["odds"], errors="coerce").astype("float32")
        race_df.loc[~((race_df["odds"] > 0) & (race_df["odds"] <= 999.9)), "odds"] = np.nan
    else:
        race_df["odds"] = np.nan
    if "horse_number" in race_df.columns:
        race_df["horse_number"] = pd.to_numeric(race_df["horse_number"], errors="coerce").fillna(0).astype("int16")

    # ── 保存 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slim_path = OUT_DIR / "race_result_slim.parquet"
    race_df.to_parquet(slim_path, index=False, compression="zstd")
    logger.info("保存: %s (%.1f MB)", slim_path, slim_path.stat().st_size / 2**20)

    ped_path = OUT_DIR / "horse_pedigree_cats.parquet"
    ped_df.to_parquet(ped_path, index=False, compression="zstd")
    logger.info("保存: %s (%.1f MB)", ped_path, ped_path.stat().st_size / 2**20)

    # メタ情報保存
    import json as _json
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "years": all_years,
        "race_rows": len(race_df),
        "unique_horses": int(race_df["horse_id"].nunique()),
        "pedigree_entries": len(ped_df),
        "missing_pedigree": missing,
        "max_gen": int(ped_df["gen"].max()) if not ped_df.empty else 5,
        "venues": sorted(race_df["venue"].dropna().unique().tolist()),
        "surfaces": sorted(race_df["surface"].dropna().unique().tolist()),
        "grades": sorted(race_df["grade"].dropna().unique().tolist()),
        "track_conditions": sorted(race_df["track_condition"].dropna().unique().tolist()),
        "dist_min": int(race_df["distance"].min()),
        "dist_max": int(race_df["distance"].max()),
    }
    (OUT_DIR / "meta.json").write_text(_json.dumps(meta, ensure_ascii=False, indent=2))

    logger.info("完了 (%.1f 秒)", time.time() - t0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="血統×レース結果インデックス構築")
    parser.add_argument("--years", nargs="*", help="対象年度（例: 2020 2021 2022）")
    args = parser.parse_args()
    build_index(years=args.years)


if __name__ == "__main__":
    main()
