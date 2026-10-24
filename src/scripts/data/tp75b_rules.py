"""tp-75 第 2 ラウンド: 原子条件（事前に分かる情報だけ。市場情報なし）の連言（2 条件・3 条件）で、買い・消しの絞り込みを探す。

python -m src.scripts.data.tp75b_rules prereg     # 原子条件・列挙規則・候補総数を tp75b_prereg.json に書く（評価の前に実行しコミットする。結果の列は使わない）
python -m src.scripts.data.tp75b_rules eval       # 検定・採用（親・第 1 ラウンドに対する増分）・クラスタ・ペルソナの再評価
python -m src.scripts.data.tp75b_rules reeval FROZEN.json [--years 2026]

第 1 ラウンド（tp75_rules.py）と同じ基準: 現行モデル（v6e）の予測に対する残差、A=95%・B=90%、年をまたぐ再現、BH（FDR 10%）。
BH の母数は第 1 ラウンドの検定可能なルールと第 2 ラウンドの候補の合算。発見 2022〜2024、2025 は評価 1 回、2026 は記述のみ。
血統の系統は、現状の不完全な 10 世代表（5 世代 + 牡系アンカー）に基づく。完全な 10 世代表が揃ったら再実行する（prereg の rerun_when にも印を付ける）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp75_rules as T1

HERE = Path(__file__).parent
OUT = Path(os.environ.get("TP75_OUT", "/home/jovyan/work/tp75_out"))
PREREG2 = HERE / "tp75b_prereg.json"
LINEAGE_DIR = Path(os.environ.get("TP75B_LINEAGE", "/home/jovyan/work/keiba-vpn/data/page_reference/pedigree_race_index"))
MIN_RUNS2 = int(os.environ.get("TP75B_MIN_RUNS", "300"))       # 発見期間の該当走数の下限（頻出集合の最小サポート）
MIN_RACES2 = int(os.environ.get("TP75B_MIN_RACES", "100"))
MAX_ORDER = 3
N_COARSE = 12

GRADE_RANK = {"新馬": 0, "未勝利": 0, "1勝": 1, "2勝": 2, "3勝": 3, "OP": 4, "L": 4, "G3": 5, "G2": 6, "G1": 7}


# ====================================================================================== 追加列（全期間の履歴が要るもの）
def _lineage_maps() -> dict:
    sl = pd.read_parquet(LINEAGE_DIR / "stallion_lineage.parquet")
    meta = json.loads((LINEAGE_DIR / "stallion_lineage_meta.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for m in meta:
        counts[m["anchor_name"]] = counts.get(m["anchor_name"], 0) + int(m.get("count") or 0)
    top = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:N_COARSE]]
    sl["sub"] = sl["sub_group_label"].astype(str)
    sl["coarse"] = np.where(sl["sub"].isin(top), sl["sub"], "その他")
    by_id = sl.drop_duplicates("stallion_id").set_index(sl["stallion_id"].astype(str))
    by_name = sl.drop_duplicates("stallion_name").set_index("stallion_name")
    return {"by_id": by_id, "by_name": by_name, "top": top}


def make_extra():
    """load_table(extra=...) に渡す関数を返す。"""
    lm = _lineage_maps()
    hs = pd.read_parquet(LINEAGE_DIR / "horse_sire_damsire.parquet", columns=["horse_id", "dam_sire"]).drop_duplicates("horse_id").set_index("horse_id")["dam_sire"]

    def extra(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values(["horse_id", "date", "race_id"]).copy()
        g = d.groupby("horse_id", sort=False)
        d["grade_rank"] = d["grade"].map(GRADE_RANK)
        wt = d[(d["finish_position"] == 1) & (d["time_sec"] > 0)].groupby("race_id")["time_sec"].min()
        d["margin"] = np.where(d["time_sec"] > 0, d["time_sec"] - d["race_id"].map(wt), np.nan)
        g = d.groupby("horse_id", sort=False)
        d["prev_margin"] = g["margin"].shift(1)
        d["prev_grade_rank"] = g["grade_rank"].shift(1)
        d["prev_venue"] = g["venue_code"].shift(1)
        d["prev_track"] = g["track_condition"].shift(1)
        d["prev_pos1_frac"] = g["pos1_frac"].shift(1)
        d["surf_chg"] = np.where(d["prev_surface"].isna(), None, np.where(d["prev_surface"] == "芝", np.where(d["is_turf"] == 0, "turf_to_dirt", "same"), np.where(d["is_turf"] == 1, "dirt_to_turf", "same")))
        d["cls_chg"] = np.where(d["prev_grade_rank"].isna() | d["grade_rank"].isna(), None, np.where(d["grade_rank"] > d["prev_grade_rank"], "up", np.where(d["grade_rank"] < d["prev_grade_rank"], "down", "same")))
        d["venue_chg"] = np.where(d["prev_venue"].isna(), None, np.where(d["prev_venue"] == d["venue_code"], "same", "diff"))
        d["going_prev"] = np.where(d["prev_track"].isna(), None, np.where(d["prev_track"] == "良", "good", "heavy"))
        for col, key in (("sire", "sire"), ("dam_sire", "ds")):
            ids = d[col].astype(str)
            d[f"{key}_main"] = ids.map(lm["by_id"]["main_group_name"])
            d[f"{key}_fine"] = ids.map(lm["by_id"]["sub"])
            d[f"{key}_coarse"] = ids.map(lm["by_id"]["coarse"])
        dds_name = d["dam_id"].map(hs)
        d["dds_main"] = dds_name.map(lm["by_name"]["main_group_name"])
        d["dds_coarse"] = dds_name.map(lm["by_name"]["coarse"])
        return d

    return extra


# ====================================================================================== 原子条件
def atom_specs(d: pd.DataFrame) -> list[dict]:
    """原子条件の一覧 = {name, dim, expr}。dim が同じ条件は同じ連言に入れない（排他・包含）。市場情報の列は使わない。"""
    A: list[dict] = []

    def add(dim: str, name: str, expr: str) -> None:
        A.append({"name": name, "dim": dim, "expr": expr})

    # 血統の系統（父・母父・母母父。main = 4 大系統 + 非主流、coarse = 上位 12 + その他、fine = 36 のサブ系統）
    for key, dim in (("sire", "sire_line"), ("ds", "dsire_line"), ("dds", "ddsire_line")):
        for lvl in ("main", "coarse", "fine"):
            col = f"{key}_{lvl}"
            if col not in d.columns:
                continue
            for v in sorted(d[col].dropna().astype(str).unique()):
                add(dim, f"{col}={v}", f'{col}=="{v}"')
    # 前走からの条件
    for k, e in {"ext200_399": "(dist_chg>=200)&(dist_chg<400)", "ext400p": "dist_chg>=400", "short200_399": "(dist_chg<=-200)&(dist_chg>-400)", "short400p": "dist_chg<=-400", "same_dist": "(dist_chg>-200)&(dist_chg<200)"}.items():
        add("p_dist", f"p_dist_{k}", e)
    for k in ("turf_to_dirt", "dirt_to_turf", "same"):
        add("p_surf", f"p_surf_{k}", f'surf_chg=="{k}"')
    for k in ("up", "down", "same"):
        add("p_cls", f"p_cls_{k}", f'cls_chg=="{k}"')
    for k, e in {"1": "prev_fin==1", "2_3": "(prev_fin>=2)&(prev_fin<=3)", "4_5": "(prev_fin>=4)&(prev_fin<=5)", "6_9": "(prev_fin>=6)&(prev_fin<=9)", "10p": "prev_fin>=10"}.items():
        add("p_fin", f"p_fin_{k}", e)
    for k, e in {"0": "prev_margin==0", "lt0.3": "(prev_margin>0)&(prev_margin<0.3)", "0.3_0.8": "(prev_margin>=0.3)&(prev_margin<0.8)", "0.8_1.5": "(prev_margin>=0.8)&(prev_margin<1.5)", "1.5p": "prev_margin>=1.5"}.items():
        add("p_margin", f"p_margin_{k}", e)
    for k, e in {"1": "prev_agari_rank==1", "2_3": "(prev_agari_rank>=2)&(prev_agari_rank<=3)", "4p": "prev_agari_rank>=4"}.items():
        add("p_agari", f"p_agari_{k}", e)
    for col, dim in (("prev_pos1_frac", "p_pos1"), ("prev_posL_frac", "p_posL")):
        for k, e in {"front": f"{col}<0.25", "mid": f"({col}>=0.25)&({col}<0.6)", "back": f"{col}>=0.6"}.items():
            add(dim, f"{dim}_{k}", e)
    for k in ("good", "heavy"):
        add("p_going", f"p_going_{k}", f'going_prev=="{k}"')
    for k in ("same", "diff"):
        add("p_venue", f"p_venue_{k}", f'venue_chg=="{k}"')
    for k, e in {"lt14": "lay_days<14", "14_27": "(lay_days>=14)&(lay_days<28)", "28_55": "(lay_days>=28)&(lay_days<56)", "56_89": "(lay_days>=56)&(lay_days<90)", "90_179": "(lay_days>=90)&(lay_days<180)", "180p": "lay_days>=180"}.items():
        add("lay", f"lay_{k}", e)
    for k in ("none", "up", "down", "top_to_top", "non_to_non"):
        add("jockey", f"jockey_{k}", f'jchg_cls=="{k}"')
    # 今走の条件
    for v in range(1, 11):
        add("venue", f"venue_{v:02d}", f"venue_code=='{v:02d}'")
    add("surface", "turf", "is_turf==1"); add("surface", "dirt", "is_turf==0")
    for k in ("d_sprint", "d_mile_s", "d_mile", "d_mid", "d_long"):
        add("dist", f"dist_{k}", f"dist_band=='{k}'")
    add("going", "going_good", "track_condition=='良'"); add("going", "going_soft", "track_condition!='良'")
    for k in ("spring", "summer", "autumn", "winter"):
        add("season", f"season_{k}", f"season=='{k}'")
    for k, e in {"inner": "bracket_number<=2", "mid": "(bracket_number>=3)&(bracket_number<=6)", "outer": "bracket_number>=7"}.items():
        add("bracket", f"bracket_{k}", e)
    for k, e in {"le11": "fs<=11", "12_15": "(fs>=12)&(fs<=15)", "ge16": "fs>=16"}.items():
        add("field", f"field_{k}", e)
    for k in ("c0_newcomer", "c1_maiden", "c2_1win", "c3_2win", "c4_3win", "op", "g3", "g2"):
        add("cls", f"cls_{k}", f"cls=='{k}'")
    for k, e in (("male", "牡"), ("female", "牝"), ("gelding", "セ")):
        add("sex", f"sex_{k}", f"sex=='{e}'")
    for k, e in {"2": "age==2", "3": "age==3", "4p": "age>=4"}.items():
        add("age", f"age_{k}", e)
    for k, e in {"le-2": "jw_chg<=-2", "mid": "(jw_chg>-2)&(jw_chg<2)", "ge2": "jw_chg>=2"}.items():
        add("jw", f"jw_{k}", e)
    for k, e in {"le-10": "weight_change<=-10", "-9_-4": "(weight_change>-10)&(weight_change<=-4)", "-3_3": "(weight_change>-4)&(weight_change<4)", "4_9": "(weight_change>=4)&(weight_change<10)", "ge10": "weight_change>=10"}.items():
        add("wtchg", f"wtchg_{k}", e)
    # 馬ごとの状態
    for k, e in {"n1": "geld_n==1", "n2": "geld_n==2", "n3": "geld_n==3", "n4p": "geld_n>=4"}.items():
        add("geld", f"geld_{k}", e)
    for k in ("first", "again", "off", "cont"):
        add("blk", f"blk_{k}", f"blk_state=='{k}'")
    for k, e in {"le6": "dam_age_at_foaling<=6", "7_14": "(dam_age_at_foaling>=7)&(dam_age_at_foaling<=14)", "ge15": "dam_age_at_foaling>=15"}.items():
        add("dam_age", f"dam_age_{k}", e)
    add("first_foal", "first_foal", "dam_first_foal==1")
    for k in ("1", "2", "既存"):
        add("sire_year", f"sire_year_{k}", f"sire_year_group=='{k}'")
    for k in ("nige", "senko", "sashi", "oikomi"):
        add("style", f"style_{k}", f"style4=='{k}'")
    add("first_surf", "first_surface", "first_surface==1")
    return A


def atom_masks(d: pd.DataFrame, specs: list[dict]) -> np.ndarray:
    return np.stack([T1.rule_mask(d, a["expr"]) for a in specs])


# ====================================================================================== 頻出集合の列挙（Apriori 型。結果の列は使わない）
def _races(rc: np.ndarray, m: np.ndarray) -> int:
    r = rc[m]
    return 0 if r.size == 0 else int(np.count_nonzero(np.diff(r)) + 1)     # rc は行順に非減少（レース順）


def enumerate_itemsets(M: np.ndarray, dims: list[str], rc: np.ndarray, disc: np.ndarray, min_runs: int, min_races: int, max_order: int = MAX_ORDER,
                       known_hashes: set[bytes] | None = None) -> dict:
    """M: (原子数, 全行) の bool。発見期間の行だけでサポートを数える。同じ該当集合（既知の第 1 ラウンドのルールを含む）は重複として除く。"""
    Md = M[:, disc]
    rcd = rc[disc]
    n = M.shape[0]
    known = set() if known_hashes is None else set(known_hashes)
    stats = {"atoms": n, "frequent_atoms": 0, "pairs_checked": 0, "pairs": 0, "triples_checked": 0, "triples": 0, "duplicates_dropped": 0}
    cands: list[tuple[int, ...]] = []
    sup1 = Md.sum(1)
    f1 = [i for i in range(n) if sup1[i] >= min_runs and _races(rcd, Md[i]) >= min_races]
    stats["frequent_atoms"] = len(f1)

    def keep(mask: np.ndarray) -> bool:
        h = hashlib.md5(np.packbits(mask).tobytes()).digest()
        if h in known:
            stats["duplicates_dropped"] += 1
            return False
        known.add(h)
        return True

    for i in f1:
        if keep(Md[i]):
            cands.append((i,))
    pair_ok: dict[tuple[int, int], np.ndarray] = {}
    for a, i in enumerate(f1):
        for j in f1[a + 1:]:
            if dims[i] == dims[j]:
                continue
            stats["pairs_checked"] += 1
            m = Md[i] & Md[j]
            if np.count_nonzero(m) >= min_runs and _races(rcd, m) >= min_races:
                pair_ok[(i, j)] = m
    stats["pairs"] = len(pair_ok)
    for (i, j), m in pair_ok.items():
        if keep(m):
            cands.append((i, j))
    if max_order >= 3:
        f1s = f1
        for (i, j), mij in pair_ok.items():
            for k in f1s:
                if k <= j or dims[k] in (dims[i], dims[j]):
                    continue
                if (i, k) not in pair_ok or (j, k) not in pair_ok:
                    continue
                stats["triples_checked"] += 1
                m = mij & Md[k]
                if np.count_nonzero(m) >= min_runs and _races(rcd, m) >= min_races:
                    stats["triples"] += 1
                    if keep(m):
                        cands.append((i, j, k))
    stats["candidates"] = len(cands)
    return {"candidates": cands, "stats": stats}


def cand_name(specs: list[dict], c: tuple[int, ...]) -> str:
    return "&".join(specs[i]["name"] for i in c)


def cand_expr(specs: list[dict], c: tuple[int, ...]) -> str:
    return " & ".join(f"({specs[i]['expr']})" for i in c)


def known_round1_hashes(d: pd.DataFrame, disc: np.ndarray) -> set[bytes]:
    hs = set()
    for r in T1.library():
        try:
            m = T1.rule_mask(d, r["expr"])[disc]
        except Exception:  # noqa: BLE001
            continue
        hs.add(hashlib.md5(np.packbits(m).tobytes()).digest())
    return hs


def prereg2() -> dict:
    d = T1.load_table(extra=make_extra())
    specs = atom_specs(d)
    M = atom_masks(d, specs)
    disc = np.isin(d["year"].to_numpy(), T1.DISC)
    rc = d["rc"].to_numpy()
    en = enumerate_itemsets(M, [a["dim"] for a in specs], rc, disc, MIN_RUNS2, MIN_RACES2, known_hashes=known_round1_hashes(d, disc))
    names = sorted(cand_name(specs, c) for c in en["candidates"])
    by_order = {k: sum(1 for c in en["candidates"] if len(c) == k) for k in (1, 2, 3)}
    sup = M[:, disc].sum(1)
    spec = {
        "version": 1, "round": 2,
        "design": {
            "purpose": "原子条件の連言（2・3 条件）で、買い・消しの絞り込みを探す。第 1 ラウンドの台帳（1,070 型）と合わせて検定する。市場情報は使わない。",
            "market_free": "原子条件は事前に分かる情報だけ。人気・オッズ・市場確率・想定オッズ・期待値は使わない（T1.market_free で検査）。",
            "max_order": MAX_ORDER, "order_4_plus": "標本が不足するので、事前に打ち切り",
            "enumeration": {"method": "Apriori 型の頻出集合。原子 → 2 条件 → 3 条件。サブセットがすべて頻出のものだけ延ばす。同じ次元の原子（排他・包含）は同じ連言に入れない。",
                            "min_runs_discovery": MIN_RUNS2, "min_races_discovery": MIN_RACES2, "support_uses": "発見期間（2022–24）の該当走数・レース数だけ。結果（残差・着順・払戻）は使わない",
                            "dedup": "発見期間の該当行が第 1 ラウンドのルールまたは先に列挙した候補と完全に同じものは重複として除く"},
            "adoption": {"criteria": "第 1 ラウンドと同じ（現行モデルの残差、A=95%・B=90%、年をまたぐ再現、BH FDR 10%）",
                         "bh_denominator": "第 1 ラウンドの検定可能なルールと第 2 ラウンドの候補の合算（族ごと）",
                         "increment_parent": "連言は、すべての直近の親（1 条件減らしたもの）に対し、『連言 − 親の残り』の差が B（片側 95%）の水準で採用の向きにあるときだけ採用。親と同じ効果なら親が代表",
                         "increment_round1": "第 1 ラウンドの採用ルールと該当行が重なるもの（包含≥0.9 または Jaccard≥0.5）は、重なりを除いた部分が採用の向きに有意なときだけ採用（それ以外は第 1 ラウンドが代表）",
                         "clusters": "第 1 ラウンドと合わせて、Jaccard≥0.5 または包含≥0.9 を束ねる"},
            "dose_response": "合致した買い（消し）クラスタの数が増えるほど残差が大きくなるか、条件の数（1・2・3）ごとの効果と再現率",
            "lineage_note": "血統の系統は、現状の不完全な 10 世代表（5 世代 + 牡系アンカー）に基づく。完全な 10 世代表が揃ったら再実行する（rerun_when）。",
            "rerun_when": ["完全な 10 世代表が揃ったとき（血統の系統・母母父の系統）", "v6e の 2026 年全体の予測が揃ったとき"],
            "unavailable_atoms": ["前走の不利（不利を表す列が無い）"],
        },
        "counts": {"atoms": len(specs), "min_support_runs": MIN_RUNS2, "min_support_races": MIN_RACES2, **en["stats"], "by_order": by_order,
                   "round1_testable": 624, "combined_tests_per_family": 624 + en["stats"]["candidates"], "families": 4},
        "candidates_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "atoms": [{"name": a["name"], "dim": a["dim"], "expr": a["expr"], "support_disc_runs": int(s)} for a, s in zip(specs, sup)],
    }
    PREREG2.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates_r2.txt").write_text("\n".join(names), encoding="utf-8")
    return spec


# ====================================================================================== 検定
def fast_disc_z(e1: np.ndarray, e3: np.ndarray, rcd: np.ndarray, nrace: int, m: np.ndarray) -> tuple[float, float, int, float]:
    """発見期間の行（rcd は 0..nrace-1 に詰めたレース番号）で、勝ち・3着内の残差の z と走数、レース単位の割合。"""
    rc = rcd[m]
    n = int(m.sum())
    C = np.bincount(rc, minlength=nrace)
    out = []
    for e in (e1, e3):
        R = np.bincount(rc, weights=e[m], minlength=nrace)
        mean = R.sum() / n
        se = math.sqrt(float(((R - mean * C) ** 2).sum())) / n
        out.append(mean / se if se > 0 else 0.0)
    return out[0], out[1], n, 0.0


def inc_z(ctx: T1.Ctx, child: np.ndarray, parent: np.ndarray, col: str, sign: int) -> float:
    """『子 − 親の残り』の差（採用の向きを正）の z。子と残りを別のクラスタ集合として分散を足す（同じレース内の相関は無視 = 保守的）。"""
    rest = parent & ~child
    a, b = T1.cstat(ctx, child, col), T1.cstat(ctx, rest, col)
    if a["n"] == 0 or b["n"] == 0 or a["se"] is None or b["se"] is None:
        return -9.0
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return sign * (a["mean"] - b["mean"]) / se if se > 0 else 0.0


def run_eval2(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = T1.load_table(extra=make_extra())
    specs = atom_specs(d)
    M = atom_masks(d, specs)
    disc = np.isin(d["year"].to_numpy(), T1.DISC)
    rc = d["rc"].to_numpy()
    dims = [a["dim"] for a in specs]
    en = enumerate_itemsets(M, dims, rc, disc, MIN_RUNS2, MIN_RACES2, known_hashes=known_round1_hashes(d, disc))
    cands = en["candidates"]
    pre = json.loads(PREREG2.read_text(encoding="utf-8"))
    names = sorted(cand_name(specs, c) for c in cands)
    assert hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest() == pre["candidates_sha256"], "候補が事前登録と一致しない（データか列挙規則が変わった）"
    # 第 1 ラウンド
    ledger1 = json.loads((out_dir / "ledger.json").read_text(encoding="utf-8"))
    res1 = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    ctx = T1.Ctx(d)
    # 発見期間だけの高速な z（全候補）
    rcd_raw = rc[disc]
    uniq, rcd = np.unique(rcd_raw, return_inverse=True)
    nrace = len(uniq)
    field_d = np.bincount(rcd, minlength=nrace)
    e1d, e3d = d["e_m1"].to_numpy()[disc], d["e_m3"].to_numpy()[disc]
    Md = M[:, disc]
    rows2 = []
    for c in cands:
        m = Md[c[0]].copy()
        for k in c[1:]:
            m &= Md[k]
        z1, z3, n, _ = fast_disc_z(e1d, e3d, rcd, nrace, m)
        nsel = np.bincount(rcd[m], minlength=nrace)
        race_level = bool((nsel[rcd[m]] == field_d[rcd[m]]).mean() >= 0.9)
        rows2.append({"c": c, "n": n, "z1": z1, "z3": z3, "race_level": race_level})
    # 合算した BH（族ごと）
    fams = T1.FAMILIES
    comb = {}
    for fam, (col, sign) in fams.items():
        p1 = [r["adopt"][fam]["p"] for r in ledger1 if fam in r["adopt"]]
        p2 = [1 - T1.phi(sign * (r["z1"] if col == "e_m1" else r["z3"])) if not r["race_level"] else 1.0 for r in rows2]
        q = T1.bh(np.array(p1 + p2))
        comb[fam] = {"q1": q[:len(p1)], "q2": q[len(p1):], "n_total": len(p1) + len(p2)}
    for k, r in enumerate(rows2):
        r["q"] = {fam: float(comb[fam]["q2"][k]) for fam in fams}
    # 有望（片側 B 水準）なものだけ全期間の統計を計算
    prom = [k for k, r in enumerate(rows2) if (not r["race_level"]) and (max(abs(r["z1"]), abs(r["z3"])) > T1.ZB)]
    rules = [{"id": "R2/" + cand_name(specs, rows2[k]["c"]), "family": "R2_order%d" % len(rows2[k]["c"]), "name": cand_name(specs, rows2[k]["c"]),
              "expr": cand_expr(specs, rows2[k]["c"]), "source": "第 2 ラウンド（原子条件の連言）", "market_free": True, "from_tp72": False} for k in prom]
    full = T1.evaluate_rules(d, rules)
    for r, k in zip(full, prom):
        r["c"] = rows2[k]["c"]; r["q2"] = rows2[k]["q"]
    # 採用（第 1 ラウンドと同じ基準。p は合算 BH の q を使う）
    for r in full:
        r["adopt"] = {}
        for fam, (col, sign) in fams.items():
            z = sign * (r["stats"]["disc"][col]["z"] or 0.0)
            tier = "A" if z > T1.ZA else "B" if z > T1.ZB else None
            rep = T1._years_ok(r, col, sign)
            r["adopt"][fam] = {"z": float(z), "p": float(1 - T1.phi(z)), "q": float(r["q2"][fam]), "tier": tier, "replicated": rep, "robust_bh": bool(r["q2"][fam] < T1.FDR),
                               "adopted": bool(tier is not None and rep), "effect": r["stats"]["disc"][col]["mean"]}
    # 増分: 親・第 1 ラウンド
    ad1 = {fam: [x for x in ledger1 if x["adopt"].get(fam, {}).get("adopted")] for fam in fams}
    r1_masks = {fam: [(x["id"], T1.rule_mask(d, x["expr"]) & ctx.periods["disc"]) for x in ad1[fam]] for fam in fams}
    n_checked = {fam: 0 for fam in fams}
    for r in full:
        mask = T1.rule_mask(d, r["expr"])
        r["mask_n_disc"] = int((mask & ctx.periods["disc"]).sum())
        for fam, (col, sign) in fams.items():
            a = r["adopt"][fam]
            if not a["adopted"]:
                continue
            n_checked[fam] += 1
            inc = {}
            if len(r["c"]) >= 2:
                for drop in range(len(r["c"])):
                    parent_atoms = [x for j, x in enumerate(r["c"]) if j != drop]
                    pm = M[parent_atoms[0]].copy()
                    for x in parent_atoms[1:]:
                        pm &= M[x]
                    inc[cand_name(specs, tuple(parent_atoms))] = inc_z(ctx, mask & ctx.periods["disc"], pm & ctx.periods["disc"], col, sign)
            a["inc_parents"] = inc
            a["inc_parent_ok"] = all(v > T1.ZB for v in inc.values()) if inc else True
            md = mask & ctx.periods["disc"]
            ov = []
            for rid, rm in r1_masks[fam]:
                inter = int((md & rm).sum())
                if inter == 0:
                    continue
                n_c, n_r = int(md.sum()), int(rm.sum())
                if inter / (n_c + n_r - inter) >= T1.OVERLAP_JAC or inter / min(n_c, n_r) >= T1.OVERLAP_CONT:
                    only_new = md & ~rm
                    zn = sign * (T1.cstat(ctx, only_new, col)["z"] or 0.0) if only_new.sum() >= 50 else -9.0
                    ov.append((rid, float(zn)))
            a["inc_round1"] = ov
            a["inc_round1_ok"] = all(zv > T1.ZB for _, zv in ov) if ov else True
            a["final"] = bool(a["adopted"] and a["inc_parent_ok"] and a["inc_round1_ok"])
    summary = {}
    for fam in fams:
        ad = [r for r in full if r["adopt"][fam]["adopted"]]
        fin = [r for r in ad if r["adopt"][fam].get("final")]
        summary[fam] = {"candidates": len(rows2), "promising": len(prom), "adopted_before_increment": len(ad), "pass_parent_increment": sum(1 for r in ad if r["adopt"][fam]["inc_parent_ok"]),
                        "pass_round1_increment": sum(1 for r in ad if r["adopt"][fam]["inc_round1_ok"]), "final": len(fin), "final_A": sum(r["adopt"][fam]["tier"] == "A" for r in fin),
                        "final_robust_bh": sum(r["adopt"][fam]["robust_bh"] for r in fin), "by_order_final": {k: sum(1 for r in fin if len(r["c"]) == k) for k in (1, 2, 3)},
                        "combined_tests": comb[fam]["n_total"], "expected_false_pos_at_1.645": 0.05 * comb[fam]["n_total"]}
    # クラスタ（第 1 ラウンド + 第 2 ラウンド）。第 2 ラウンドの最終採用だけを足す
    ledger_rows = []
    by_id1 = {x["id"]: x for x in ledger1}
    class _Row(dict):
        pass
    clusters = {}
    for fam in fams:
        rows_f = []
        for x in ledger1:
            if x["adopt"].get(fam, {}).get("adopted"):
                st = _rebuild_stats(d, ctx, x)
                rows_f.append({"id": x["id"], "expr": x["expr"], "source": x["source"], "stats": st, "adopt": {fam: x["adopt"][fam]}, "round": 1})
        for r in full:
            if r["adopt"][fam].get("final"):
                rows_f.append({"id": r["id"], "expr": r["expr"], "source": r["source"], "stats": r["stats"], "adopt": {fam: r["adopt"][fam]}, "round": 2, "c": r["c"]})
        clusters[fam] = T1.cluster_set(d, rows_f, fam)
        for c in clusters[fam]:
            c["cover"] = float(c["union"][ctx.periods["disc"]].mean())
    res = {"summary": summary, "prereg_counts": pre["counts"], "enumeration": en["stats"]}
    res["clusters"] = {fam: [{"rep": c["rep"], "round": c["rep_row"]["round"], "n_members": len(c["members"]), "members": c["members"], "weight": T1.cluster_weight(c, fam), "cover": c["cover"],
                              "expr": c["rep_row"]["expr"], "tier": c["rep_row"]["adopt"][fam]["tier"], "robust_bh": c["rep_row"]["adopt"][fam]["robust_bh"],
                              "effect_pt": 100 * c["rep_row"]["adopt"][fam]["effect"]} for c in clusters[fam]] for fam in fams}
    res["v2"] = T1.evaluate_personas(d, ctx, clusters)
    narrow = {fam: [c for c in cl if c["cover"] <= T1.NARROW_MAX_COVER] for fam, cl in clusters.items()}
    res["v2_narrow_posthoc"] = T1.evaluate_personas(d, ctx, narrow)
    res["dose_response"] = dose_response(d, ctx, clusters)
    res["order_effect"] = order_effect(full)
    # 台帳（第 2 ラウンドの有望な候補 + 親・増分・クラスタ）
    rep_of = {fam: {m: c["rep"] for c in clusters[fam] for m in c["members"]} for fam in fams}
    for r in full:
        s = r["stats"]
        rec = {"id": r["id"], "order": len(r["c"]), "expr": r["expr"], "n_disc": s["disc"]["e_m1"]["n"], "races_disc": s["disc"]["e_m1"]["races"],
               "win_resid_pt": None if s["disc"]["e_m1"]["mean"] is None else 100 * s["disc"]["e_m1"]["mean"], "win_resid_z": s["disc"]["e_m1"]["z"],
               "top3_resid_pt": None if s["disc"]["e_m3"]["mean"] is None else 100 * s["disc"]["e_m3"]["mean"], "top3_resid_z": s["disc"]["e_m3"]["z"],
               "z_2025_win": s["y2025"]["e_m1"]["z"], "z_2025_top3": s["y2025"]["e_m3"]["z"], "z_2026_win": s["y2026"]["e_m1"]["z"], "z_2026_top3": s["y2026"]["e_m3"]["z"]}
        for fam in fams:
            a = r["adopt"][fam]
            rec[fam] = {"tier": a["tier"], "adopted": a["adopted"], "final": a.get("final", False), "robust_bh": a["robust_bh"], "q_combined": a["q"], "z": a["z"], "inc_parents": a.get("inc_parents"),
                        "inc_round1": a.get("inc_round1"), "cluster": rep_of[fam].get(r["id"])}
        ledger_rows.append(rec)
    (out_dir / "ledger_r2.json").write_text(json.dumps(ledger_rows, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out_dir / "results_r2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # v2 の再評価で使う、凍結用のルール定義（第 2 ラウンドの最終採用）
    return res


def _rebuild_stats(d: pd.DataFrame, ctx: T1.Ctx, x: dict) -> dict:
    """第 1 ラウンドの台帳の行から、重みの計算に必要な発見期間の統計だけを作り直す。"""
    m = T1.rule_mask(d, x["expr"]) & ctx.periods["disc"]
    return {"disc": {"e_m1": T1.cstat(ctx, m, "e_m1"), "e_m3": T1.cstat(ctx, m, "e_m3")}}


def order_effect(full: list[dict]) -> dict:
    """条件の数（1・2・3）ごとの、採用ルールの効果の大きさと、後の年での再現率。"""
    out = {}
    for fam, (col, sign) in T1.FAMILIES.items():
        o = {}
        for k in (1, 2, 3):
            ad = [r for r in full if len(r["c"]) == k and r["adopt"][fam].get("final")]
            zc = "e_m1" if col == "e_m1" else "e_m3"
            eff = [sign * r["stats"]["disc"][zc]["mean"] * 100 for r in ad]
            z25 = [sign * r["stats"]["y2025"][zc]["z"] for r in ad if r["stats"]["y2025"][zc]["z"] is not None]
            o[k] = {"n": len(ad), "mean_effect_pt": float(np.mean(eff)) if eff else None, "median_n_disc": float(np.median([r["stats"]["disc"][zc]["n"] for r in ad])) if ad else None,
                    "same_sign_2025": int(sum(z > 0 for z in z25)), "n_2025": len(z25), "z_gt_1645_2025": int(sum(z > 1.645 for z in z25))}
        out[fam] = o
    return out


def dose_response(d: pd.DataFrame, ctx: T1.Ctx, clusters: dict) -> dict:
    """合致した買い・消しクラスタの数 k ごとの、残差（採用の向きを正）と回収率。"""
    out = {}
    for fam, (col, sign) in T1.FAMILIES.items():
        k = np.zeros(len(d), int)
        for c in clusters[fam]:
            k += c["union"].astype(int)
        o = {}
        for pn in ("disc", "y2025", "y2026"):
            sel = ctx.periods[pn]
            o[pn] = {}
            for lab, m in (("0", k == 0), ("1", k == 1), ("2", k == 2), ("3+", k >= 3)):
                mm = m & sel
                o[pn][lab] = {"n": int(mm.sum()), "resid": T1.cstat(ctx, mm, col), "roi_win": T1.boot_mean(ctx, mm, "r1", 300), "roi_place": T1.boot_mean(ctx, mm, "r3", 300), "top3": T1.boot_mean(ctx, mm, "y3", 300)}
        out[fam] = o
    return out


def eval_personas_r2(out_dir: Path, robust_only: bool = True) -> dict:
    """第 2 ラウンドの最終採用のうち、BH 頑健（合算の母数）なものだけを第 1 ラウンドに足して、クラスタとペルソナを再評価する。
    事後の判断: 事前登録の基準（A/B・再現・増分）だけで採用すると 2.6〜3.5 千個になり、2025 年の標本外では偶然の期待とほぼ同じだった。BH 頑健なものだけが標本外でも再現したので、順位付けと凍結にはそれだけを使う。"""
    d = T1.load_table(extra=make_extra())
    ctx = T1.Ctx(d)
    ledger1 = json.loads((out_dir / "ledger.json").read_text(encoding="utf-8"))
    ledger2 = json.loads((out_dir / "ledger_r2.json").read_text(encoding="utf-8"))
    fams = T1.FAMILIES
    clusters = {}
    used = {}
    for fam, (col, sign) in fams.items():
        rows_f = []
        for x in ledger1:
            if x["adopt"].get(fam, {}).get("adopted"):
                rows_f.append({"id": x["id"], "expr": x["expr"], "source": x["source"], "stats": _rebuild_stats(d, ctx, x), "adopt": {fam: x["adopt"][fam]}, "round": 1})
        n2 = 0
        for x in ledger2:
            a = x[fam]
            if a["final"] and (a["robust_bh"] or not robust_only):
                st = _rebuild_stats(d, ctx, x)
                z = x["win_resid_z"] if col == "e_m1" else x["top3_resid_z"]
                rows_f.append({"id": x["id"], "expr": x["expr"], "source": "第 2 ラウンド", "stats": st, "adopt": {fam: {**a, "z": sign * z, "effect": st["disc"][col]["mean"]}}, "round": 2, "c": None})
                n2 += 1
        used[fam] = n2
        clusters[fam] = T1.cluster_set(d, rows_f, fam)
        for c in clusters[fam]:
            c["cover"] = float(c["union"][ctx.periods["disc"]].mean())
    res = {"robust_only": robust_only, "round2_rules_used": used}
    res["clusters"] = {fam: [{"rep": c["rep"], "round": c["rep_row"]["round"], "n_members": len(c["members"]), "members": c["members"], "weight": T1.cluster_weight(c, fam), "cover": c["cover"],
                              "expr": c["rep_row"]["expr"], "tier": c["rep_row"]["adopt"][fam]["tier"], "robust_bh": c["rep_row"]["adopt"][fam].get("robust_bh"),
                              "effect_pt": 100 * c["rep_row"]["adopt"][fam]["effect"]} for c in clusters[fam]] for fam in fams}
    res["v2"] = T1.evaluate_personas(d, ctx, clusters)
    res["dose_response"] = dose_response(d, ctx, clusters)
    (out_dir / ("results_r2b.json" if robust_only else "results_r2_all.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prereg", "eval", "personas", "reeval"])
    ap.add_argument("frozen", nargs="?")
    ap.add_argument("--years", nargs="*", type=int, default=[2026])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.cmd == "prereg":
        s = prereg2()
        print(json.dumps(s["counts"], ensure_ascii=False, indent=1))
    elif a.cmd == "eval":
        r = run_eval2(Path(a.out))
        print(json.dumps(r["summary"], ensure_ascii=False, indent=1))
    elif a.cmd == "personas":
        r = eval_personas_r2(Path(a.out))
        print(json.dumps(r["round2_rules_used"], ensure_ascii=False))
    else:
        print(json.dumps(T1.reeval(Path(a.frozen), a.years, loader=lambda: T1.load_table(extra=make_extra())), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
