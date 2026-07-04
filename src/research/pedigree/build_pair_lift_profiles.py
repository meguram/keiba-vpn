"""父 × 母父系 のペア lift プロファイルを生成するバッチ。

★ 設計方針 (2026-05-15 改訂):
  ユーザー要件「父×母父個体だと件数が減るので、母父"系"で集計してほしい」
  に従い、母父個体ペアは廃止。母父の "父系ルート種牡馬" (= 母父父系root,
  例「ディープインパクト系」「Sunday Silence系」「Sadler's Wells系」等の
  227 系統) を中間粒度として採用する。

  PAIR_BMS_ROOT : 父個体 × 母父父系root  (n_records >= 50)  ← ★ メイン
                  ドゥラメンテ × Sunday Silence系 母父 のようなペア
                  227 系統あり、5大系統より細かく、母父個体より粗い実用粒度
  PAIR_GROUP    : 父個体 × 母父5大系統   (n_records >= 50)
                  Turn-To/Native/Northern/Nasrullah/非主流 (5個)
                  PAIR_BMS_ROOT で n が薄い時のフォールバック
  PAIR_GXG      : 父5系統 × 母父5系統   (n_records >= 50)
                  最終フォールバック (5×5 = 25 ペア)

出力:
  data/research/bloodline_meta_cluster/pair_lift_profiles_bms_root.parquet  ← ★ 新規
  data/research/bloodline_meta_cluster/pair_lift_profiles_group.parquet
  data/research/bloodline_meta_cluster/pair_lift_profiles_gxg.parquet
  data/research/bloodline_meta_cluster/pair_lift_meta.json

  互換: pair_lift_profiles_indiv.parquet は **削除** される (新フォールバック階層に
       無いため)。bloodline_meta_cluster.py のフォールバック階層も同時に更新が必要。

使用:
  python -m src.research.pedigree.build_pair_lift_profiles
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger("research.pair_lift")

ROOT = Path(__file__).resolve().parents[3]
IDX_DIR = ROOT / "data/page_reference/pedigree_race_index"
ART_DIR = ROOT / "data/page_reference/note_aptitude_race"

EB_PRIOR_N = 30
GLOBAL_WIN = 0.139

# 距離区分は父視点 (build_meta_cluster_artifacts.py) と統一:
#   短距離 [0,1400) / マイル [1400,1800) / 中距離 [1800,2400) / 長距離 [2400,9999)
DIST_BINS = [0, 1400, 1800, 2400, 9999]
DIST_LABELS = ["短距離", "マイル", "中距離", "長距離"]
STEEP_VENUES = ("中山", "阪神", "中京")
COND_COLS: list[str] = (
    [f"win_v_{v}" for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]]
    + [f"win_s_{s}" for s in ["芝","ダート"]]
    + [f"win_d_{d}" for d in DIST_LABELS]
    + ["win_steep","win_flat","win_heavy","win_outer","win_inner"]
)

MIN_N_BMS_ROOT = 50   # 父個体×母父父系root ペア最低出走数 (新メイン軸)
MIN_N_GROUP = 50      # 父個体×母父5大系統 ペア最低出走数 (フォールバック)
MAIN_GROUPS = ("Turn-To系", "Native Dancer系", "Northern Dancer系", "Nasrullah系", "非主流")


def _eb(w, n, prior_p=GLOBAL_WIN):
    if n < 5: return None
    return (w + EB_PRIOR_N * prior_p) / (n + EB_PRIOR_N)


def _load_priors() -> dict[str, float]:
    p = ART_DIR / "overall_raw.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {c: GLOBAL_WIN for c in COND_COLS}


def build():
    script_basic_config(level=logging.INFO)
    logger.info("[load] race / bms")
    bms = pd.read_parquet(
        IDX_DIR / "horse_bms.parquet",
        columns=["horse_id","sire_id","bms_id","bms_root_id","bms_root_name"],
    )
    bms["horse_id"] = bms["horse_id"].astype(str)
    bms["sire_id"] = bms["sire_id"].astype(str)
    bms["bms_id"] = bms["bms_id"].astype(str)
    bms["bms_root_id"] = bms["bms_root_id"].astype(str).fillna("").replace({"None": ""})
    bms["bms_root_name"] = bms["bms_root_name"].astype(str).fillna("").replace({"None": ""})
    # 母父父系root が空のレコードはペア集計から除外 (海外馬等 不明系統)
    n_total_bms = len(bms)
    bms_root_ok = bms[bms["bms_root_id"].str.len() > 0].copy()
    logger.info(
        "[ok] bms_root_id 有効レコード = %d / %d (%.1f%%)",
        len(bms_root_ok), n_total_bms, 100.0 * len(bms_root_ok) / max(n_total_bms, 1),
    )
    # bms_root_id → bms_root_name のマップを構築
    bms_root_name_map: dict[str, str] = (
        bms_root_ok[["bms_root_id","bms_root_name"]]
        .drop_duplicates()
        .set_index("bms_root_id")["bms_root_name"].to_dict()
    )

    race = pd.read_parquet(IDX_DIR / "race_result_slim.parquet")
    race["horse_id"] = race["horse_id"].astype(str)
    race = race[race["finish_position"].notna() & (race["finish_position"] > 0)].copy()
    race["win"] = (race["finish_position"] == 1).astype(int)
    race["dist_cat"] = pd.cut(
        race["distance"], bins=DIST_BINS, labels=DIST_LABELS, right=False
    ).astype(str)
    race["is_steep"] = race["venue"].astype(str).isin(STEEP_VENUES)
    race["is_heavy"] = race["track_condition"].astype(str).isin(["重","不良"])
    # 内/外回り (course_profiles.json + 公知 OVERRIDE) — 父視点と同じヘルパーを使う
    from src.research.pedigree.build_meta_cluster_artifacts import _attach_course_type
    race = _attach_course_type(race)
    race["is_outer"] = race["course_type"] == "外"
    race["is_inner"] = race["course_type"] == "内"

    sid_to_main: dict[str, str] = json.loads((ART_DIR / "sid_to_main.json").read_text(encoding="utf-8"))
    sid_to_name: dict[str, str] = {}
    for fn in ("sid_to_name_full.json", "sid_to_name.json"):
        p = ART_DIR / fn
        if p.exists():
            try:
                sid_to_name.update(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass

    bms["sire_main"] = bms["sire_id"].map(sid_to_main)
    bms["bms_main"] = bms["bms_id"].map(sid_to_main)
    logger.info("[ok] race=%d, bms=%d", len(race), len(bms))

    priors = _load_priors()

    # ── per-horse per-cond の集計 (高速化) ──
    logger.info("[pre-agg] horse × cond の (n,w) 事前集計")
    g_total = race.groupby("horse_id")["win"].agg(["count","sum"])
    horse_total_n = g_total["count"].to_dict()
    horse_total_w = g_total["sum"].to_dict()

    def _mask(c):
        if c.startswith("win_v_"): return race["venue"].astype(str) == c.split("_")[-1]
        if c.startswith("win_s_"): return race["surface"].astype(str) == c.split("_")[-1]
        if c.startswith("win_d_"): return race["dist_cat"] == c.split("_")[-1]
        if c == "win_steep": return race["is_steep"]
        if c == "win_flat":  return ~race["is_steep"]
        if c == "win_heavy": return race["is_heavy"]
        if c == "win_outer": return race["is_outer"]
        if c == "win_inner": return race["is_inner"]
        raise ValueError(c)

    cond_horse_n: dict[str, dict[str, int]] = {}
    cond_horse_w: dict[str, dict[str, int]] = {}
    for c in COND_COLS:
        sub = race[_mask(c)]
        g = sub.groupby("horse_id")["win"].agg(["count","sum"])
        cond_horse_n[c] = g["count"].to_dict()
        cond_horse_w[c] = g["sum"].to_dict()

    def _profile(horse_ids: set[str], min_n: int) -> dict[str, float] | None:
        if not horse_ids: return None
        n_tot = sum(horse_total_n.get(h, 0) for h in horse_ids)
        if n_tot < min_n: return None
        w_tot = sum(horse_total_w.get(h, 0) for h in horse_ids)
        eb_tot = _eb(w_tot, n_tot, priors.get("win_eb_total", GLOBAL_WIN))
        if eb_tot is None or eb_tot <= 0: return None
        out = {"_n_total": int(n_tot), "_n_horses": int(len(horse_ids)), "_eb_total": float(eb_tot)}
        for c in COND_COLS:
            cn = cond_horse_n.get(c, {})
            cw = cond_horse_w.get(c, {})
            n_c = sum(cn.get(h, 0) for h in horse_ids)
            if n_c < 5: continue
            w_c = sum(cw.get(h, 0) for h in horse_ids)
            e = _eb(w_c, n_c, priors.get(c, GLOBAL_WIN))
            if e is None: continue
            out[f"lift_{c}"] = float(e / eb_tot)
            out[f"n_{c}"] = int(n_c)
        return out

    # ── 父個体 × 母父父系root (=母父系) ペア [メイン] ──
    logger.info(
        "[run] PAIR_BMS_ROOT (父個体×母父父系root '母父系', n>=%d)", MIN_N_BMS_ROOT,
    )
    pair_root = bms_root_ok.groupby(["sire_id","bms_root_id"])["horse_id"].apply(set)
    rows_root = []
    n_done = 0
    for (sid, brid), horses in pair_root.items():
        n_done += 1
        if n_done % 2000 == 0:
            logger.info(
                "  ... PAIR_BMS_ROOT %d/%d, accepted=%d",
                n_done, len(pair_root), len(rows_root),
            )
        prof = _profile(horses, MIN_N_BMS_ROOT)
        if prof is None: continue
        bms_root_name = bms_root_name_map.get(brid, brid)
        row = {
            "sire_id": sid, "sire_name": sid_to_name.get(sid, sid),
            "sire_main": sid_to_main.get(sid),
            "bms_root_id": brid,
            "bms_root_name": bms_root_name,
            # 互換のために bms_main (5大系統) も保持
            "bms_main": sid_to_main.get(brid),
            "n_horses": prof["_n_horses"],
            "n_records": prof["_n_total"],
            "eb_total": prof["_eb_total"],
        }
        for c in COND_COLS:
            if f"lift_{c}" in prof:
                row[f"lift_{c}"] = prof[f"lift_{c}"]
                row[f"n_{c}"] = prof[f"n_{c}"]
        rows_root.append(row)
    df_root = pd.DataFrame(rows_root)
    out1 = ART_DIR / "pair_lift_profiles_bms_root.parquet"
    df_root.to_parquet(out1, index=False)
    logger.info("[save] %s : %d rows", out1, len(df_root))

    # 旧ファイル (pair_lift_profiles_indiv.parquet) を削除
    legacy = ART_DIR / "pair_lift_profiles_indiv.parquet"
    if legacy.exists():
        legacy.unlink()
        logger.info("[cleanup] removed legacy %s (PAIR_INDIV 廃止)", legacy.name)

    # ── 父個体 × 母父系統 ペア ──
    logger.info("[run] PAIR_GROUP (父個体×母父系統, n>=%d)", MIN_N_GROUP)
    rows_group = []
    sire_groups = bms.groupby(["sire_id","bms_main"])["horse_id"].apply(set)
    n_done = 0
    for (sid, mg), horses in sire_groups.items():
        n_done += 1
        if mg not in MAIN_GROUPS: continue
        if n_done % 2000 == 0:
            logger.info("  ... PAIR_GROUP %d/%d, accepted=%d", n_done, len(sire_groups), len(rows_group))
        prof = _profile(horses, MIN_N_GROUP)
        if prof is None: continue
        row = {
            "sire_id": sid, "sire_name": sid_to_name.get(sid, sid),
            "sire_main": sid_to_main.get(sid),
            "bms_main": mg,
            "n_horses": prof["_n_horses"],
            "n_records": prof["_n_total"],
            "eb_total": prof["_eb_total"],
        }
        for c in COND_COLS:
            if f"lift_{c}" in prof:
                row[f"lift_{c}"] = prof[f"lift_{c}"]
                row[f"n_{c}"] = prof[f"n_{c}"]
        rows_group.append(row)
    df_group = pd.DataFrame(rows_group)
    out2 = ART_DIR / "pair_lift_profiles_group.parquet"
    df_group.to_parquet(out2, index=False)
    logger.info("[save] %s : %d rows", out2, len(df_group))

    # ── 父系統 × 母父系統 マトリクス (最終フォールバック用) ──
    logger.info("[run] PAIR_GROUP_X_GROUP (父系統×母父系統)")
    rows_gxg = []
    grp_grp = bms.groupby(["sire_main","bms_main"])["horse_id"].apply(set)
    for (sg, mg), horses in grp_grp.items():
        if sg not in MAIN_GROUPS or mg not in MAIN_GROUPS: continue
        prof = _profile(horses, MIN_N_GROUP)
        if prof is None: continue
        row = {
            "sire_main": sg, "bms_main": mg,
            "n_horses": prof["_n_horses"],
            "n_records": prof["_n_total"],
            "eb_total": prof["_eb_total"],
        }
        for c in COND_COLS:
            if f"lift_{c}" in prof:
                row[f"lift_{c}"] = prof[f"lift_{c}"]
                row[f"n_{c}"] = prof[f"n_{c}"]
        rows_gxg.append(row)
    df_gxg = pd.DataFrame(rows_gxg)
    out3 = ART_DIR / "pair_lift_profiles_gxg.parquet"
    df_gxg.to_parquet(out3, index=False)
    logger.info("[save] %s : %d rows", out3, len(df_gxg))

    # メタ
    meta = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "min_n_bms_root": MIN_N_BMS_ROOT,
        "min_n_group": MIN_N_GROUP,
        "eb_prior_n": EB_PRIOR_N,
        "main_groups": list(MAIN_GROUPS),
        "cond_cols": COND_COLS,
        "n_bms_root": int(len(df_root)),
        "n_group": int(len(df_group)),
        "n_gxg": int(len(df_gxg)),
        "n_unique_sires_bms_root": int(df_root["sire_id"].nunique()) if len(df_root) else 0,
        "n_unique_bms_root_systems": int(df_root["bms_root_id"].nunique()) if len(df_root) else 0,
        "fallback_chain": [
            "PAIR_BMS_ROOT (sire_id, bms_root_id)  ← メイン軸: 父個体 × 母父系",
            "PAIR_GROUP    (sire_id, bms_main)     ← 母父5大系統",
            "ROLE_BLEND    (F+MF+FF+MMF weighted)",
            "PAIR_GXG      (sire_main, bms_main)   ← 最終フォールバック",
        ],
        "notes": (
            "lift_<cond> = EB(win_in_cond) / EB(win_total). 階層フォールバック順で参照する。"
            " bms_root_id は 母父の '父系ルート種牡馬' (= 母父系)。227 系統。"
            " 旧 PAIR_INDIV (父個体×母父個体) はサンプル件数不足のため廃止。"
        ),
        "schema_version": "2026-05-15-bms-root",
    }
    (ART_DIR / "pair_lift_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[done]")


if __name__ == "__main__":
    build()
