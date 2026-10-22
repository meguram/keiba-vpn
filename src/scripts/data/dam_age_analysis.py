"""母の初仔・母の出産時年齢・産次・種牡馬の供用年（初年度産駒）と成績の関係。

python -m src.scripts.data.dam_age_analysis lineage [--out DIR]   # 5gen JSON から (子, 父, 母) の三つ組を集める
python -m src.scripts.data.dam_age_analysis table [--out DIR]     # 共有表 foal_lineage を出す（キー horse_id）
python -m src.scripts.data.dam_age_analysis run [--out DIR]       # 集計して results.json
python -m src.scripts.data.dam_age_analysis plot [--out DIR]      # 図（PNG）

定義
  馬の生年 = 馬 ID の先頭 4 桁（netkeiba の 10 桁 ID）。母の生年 = 母 ID の先頭 4 桁（外国の母 000a… は birth_manual.csv にあれば補う）。
  母の出産時年齢 = 馬の生年 − 母の生年。
  産次の下限 = 『その母の既知の産駒のうち、この馬より前に生まれた頭数』+ 1。既知の産駒 = 5gen の血統表にある馬と、その木の中の祖先。
    2017〜2023 年生まれはほぼ全頭が既知（JRA の出走馬）。2016 年以前は一部だけ。
  初仔の判定 = 母が 2014 年以降の生まれ（産駒がすべて 2017 年以降＝既知の範囲に入る）で、既知の前の産駒が無い → 推定の初仔。
    既知の前の産駒がある → 確実に 2 仔目以降。母の出産時年齢が 3 歳以下 → 確実に初仔。それ以外（母が古い世代）は不明。
  種牡馬の供用年 k = 馬の生年 − 『その種牡馬の既知の産駒の最小の生年』+ 1。最小の生年が 2018 年以降の種牡馬だけ k を決める
    （2017 年は 2016 年生まれの一部しか見えていないので不明側）。それ以前から産駒がいる種牡馬は『既存』（k は 7 以上と確実に言える）。
  市場オッズは回収率と市場との差（評価）にだけ使う。特徴量にはしない。2026 年は記述統計のみ。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("DAM_ROOT") or os.environ.get("GELD_ROOT") or Path(__file__).resolve().parents[3])
os.environ.setdefault("GELD_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))
try:
    from src.scripts.data import geld_analysis as G
except ImportError:  # ホスト上では単独のモジュールとして置く
    import geld_analysis as G  # type: ignore

BIRTH_CSVS = [Path(p) for p in (os.environ.get("DAM_BIRTH_CSV") or str(ROOT / "data/meta/dormello/birth_manual.csv")).split(":")]
FRAME = ROOT / "data/local/modeling/betting/tp36/frame_all.parquet"
TABLES = ROOT / "data/page_reference/tables"
B = 1000
SEED = 20261023
COMPLETE_FROM_FOAL = 2017          # この年以降に生まれた馬は、JRA の出走馬がほぼ全頭 5gen にある
COHORT_DAM_FROM = 2014             # 母がこの年以降の生まれなら、産駒は全員 COMPLETE_FROM_FOAL 以降に生まれる
SIRE_FIRST_CROP_FROM = 2018        # 既知の最小の生年がこの年以降の種牡馬だけ、供用年 k を決める
SIRE_ESTABLISHED_TO = 2015         # 既知の最小の生年がこの年以前なら『既存』（2016・2017 年は 2016 年以前の初年度産駒が見えていない恐れがあるので、どちらにも入れない）
SIRE_GAP = (4, 10)                 # 初産駒の生年 − 父の生年 がこの範囲なら、観測の最小の生年を本当の初年度とみなす（外れる種牡馬は不明）
AGE_BANDS = [("≤6", -99, 6), ("7-11", 7, 11), ("12-14", 12, 14), ("15-17", 15, 17), ("18+", 18, 99)]
METRICS = ["d_megu", "d_fp", "d_win", "d_top3", "m_win", "m_mkt", "roi_win", "roi_place"]
METRIC_LABEL = {
    "d_megu": "めぐ指数 − 同条件平均", "d_fp": "着順の位置 − 同条件平均（小さいほど良い）", "d_win": "勝率 − 同条件平均", "d_top3": "複勝率 − 同条件平均",
    "m_win": "勝率 − 現行モデルの予測", "m_mkt": "勝率 − 市場の勝率", "roi_win": "単勝回収率", "roi_place": "複勝回収率",
}


# ───────────────────────── 血統の三つ組と共有表
def build_lineage(out: Path) -> pd.DataFrame:
    rows: dict[str, tuple] = {}
    files = glob.glob(str(ROOT / "data/local/horse_pedigree_5gen/*/*.json"))
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hid = d.get("horse_id")
        anc = {(a["generation"], a["position"]): a.get("horse_id") for a in d.get("ancestors", []) if a.get("horse_id")}
        if hid:
            rows[hid] = (anc.get((1, 0)), anc.get((1, 1)), "subject")
        for (g, p), cid in anc.items():           # 木の中の祖先 (g,p) の親は (g+1,2p) と (g+1,2p+1)
            if g >= 5 or not cid or cid in rows:
                continue
            s, m = anc.get((g + 1, 2 * p)), anc.get((g + 1, 2 * p + 1))
            if s or m:
                rows[cid] = (s, m, "tree")
    df = pd.DataFrame([(k, v[0], v[1], v[2]) for k, v in rows.items()], columns=["horse_id", "sire_id", "dam_id", "src"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "lineage.parquet", index=False)
    return df


def _year(ids: pd.Series) -> pd.Series:
    s = ids.astype("string")
    y = pd.to_numeric(s.str[:4], errors="coerce").astype("float64")
    return y.where(s.str.fullmatch(r"\d{10}").fillna(False).astype(bool))


def _manual_births() -> dict[str, float]:
    out: dict[str, float] = {}
    for p in BIRTH_CSVS:
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype=str)
        if "horse_id" in d.columns and "birth_year" in d.columns:
            for h, y in zip(d["horse_id"], d["birth_year"]):
                try:
                    out[h] = float(y)
                except (TypeError, ValueError):
                    pass
    return out


def age_band(a: float) -> str | None:
    if not np.isfinite(a):
        return None
    for name, lo, hi in AGE_BANDS:
        if lo <= a <= hi:
            return name
    return None


def foal_table(L: pd.DataFrame) -> pd.DataFrame:
    """horse_id をキーにした共有表。出走前に分かる情報だけ（as-of）。"""
    L = L.drop_duplicates("horse_id").copy()
    man = _manual_births()
    L["foal_year"] = _year(L["horse_id"])
    L["dam_birth_year"] = _year(L["dam_id"])
    L["dam_birth_src"] = np.where(L["dam_birth_year"].notna(), "id", None)
    mm = L["dam_id"].map(man)
    fill = L["dam_birth_year"].isna() & mm.notna()
    L.loc[fill, "dam_birth_year"], L.loc[fill, "dam_birth_src"] = mm[fill], "manual"
    L["dam_age_at_foaling"] = L["foal_year"] - L["dam_birth_year"]
    L["dam_age_band"] = L["dam_age_at_foaling"].map(age_band)

    g = L[L["foal_year"].notna()].groupby("dam_id")["foal_year"]
    L["dam_known_foals"] = L["dam_id"].map(g.size())
    ranks = g.rank(method="min") - 1                     # 自分より前に生まれた既知の産駒の頭数
    L.loc[ranks.index, "dam_earlier_known"] = ranks
    L["parity_min"] = L["dam_earlier_known"] + 1
    L["dam_cohort_complete"] = ((L["dam_birth_year"] >= COHORT_DAM_FROM) & (L["foal_year"] >= COMPLETE_FROM_FOAL)).astype(float)
    L.loc[L["dam_birth_year"].isna() | L["foal_year"].isna(), "dam_cohort_complete"] = np.nan

    first, conf = [], []
    for e, age, comp in zip(L["dam_earlier_known"], L["dam_age_at_foaling"], L["dam_cohort_complete"]):
        if np.isfinite(e) and e >= 1:
            first.append(0.0); conf.append("確実")
        elif np.isfinite(age) and age <= 3:
            first.append(1.0); conf.append("確実")
        elif comp == 1.0 and np.isfinite(e):
            first.append(1.0); conf.append("推定")
        else:
            first.append(np.nan); conf.append("不明")
    L["dam_first_foal"], L["dam_first_foal_conf"] = first, conf
    pc = np.where(L["dam_cohort_complete"] == 1.0, "推定", np.where(L["dam_earlier_known"] >= 1, "下限", "不明"))
    L["parity_conf"] = pc
    L["parity_group"] = np.where(L["dam_cohort_complete"] == 1.0, np.minimum(L["parity_min"], 4).map(lambda x: "4+" if x == 4 else str(int(x)) if np.isfinite(x) else None), None)

    # 種牡馬の供用年
    L["sire_birth_year"] = _year(L["sire_id"])
    sm = L["sire_id"].map(man)
    L.loc[L["sire_birth_year"].isna() & sm.notna(), "sire_birth_year"] = sm
    y0 = L[L["foal_year"].notna()].groupby("sire_id")["foal_year"].min()
    L["sire_first_crop_year"] = L["sire_id"].map(y0)
    L["sire_n_known_foals"] = L["sire_id"].map(L[L["foal_year"].notna()].groupby("sire_id").size())
    k = L["foal_year"] - L["sire_first_crop_year"] + 1
    gap = L["sire_first_crop_year"] - L["sire_birth_year"]       # 初産駒の生年 − 父の生年（通常 5〜8）
    new = (L["sire_first_crop_year"] >= SIRE_FIRST_CROP_FROM) & gap.between(SIRE_GAP[0], SIRE_GAP[1])
    old = L["sire_first_crop_year"] <= SIRE_ESTABLISHED_TO
    L["sire_stud_year"] = np.where(new, k, np.nan)
    L["sire_year_group"] = np.where(L["foal_year"].isna(), None, np.where(new, np.select([k == 1, k == 2, k == 3], ["1", "2", "3"], "4+"), np.where(old, "既存", None)))
    L["sire_crop_conf"] = np.where(L["sire_year_group"].isna(), "不明", "確実")
    cols = ["horse_id", "foal_year", "sire_id", "dam_id", "dam_birth_year", "dam_birth_src", "dam_age_at_foaling", "dam_age_band",
            "dam_known_foals", "dam_earlier_known", "parity_min", "parity_group", "parity_conf", "dam_cohort_complete",
            "dam_first_foal", "dam_first_foal_conf", "sire_birth_year", "sire_first_crop_year", "sire_n_known_foals",
            "sire_stud_year", "sire_year_group", "sire_crop_conf"]
    return L[cols]


README_COLUMNS = {
    "horse_id": "キー（netkeiba の馬 ID）", "foal_year": "馬の生年（ID の先頭 4 桁）", "sire_id": "父の ID", "dam_id": "母の ID",
    "dam_birth_year": "母の生年（ID の先頭 4 桁。外国の母は birth_manual.csv にあれば補う）", "dam_birth_src": "母の生年の出どころ（id / manual）",
    "dam_age_at_foaling": "母の出産時年齢 = 馬の生年 − 母の生年", "dam_age_band": "出産時年齢の帯（≤6 / 7-11 / 12-14 / 15-17 / 18+）",
    "dam_known_foals": "既知の母の産駒の総数", "dam_earlier_known": "この馬より前の生年の既知の産駒の頭数",
    "parity_min": "産次の下限 = dam_earlier_known + 1（2016 年以前の産駒は一部しか既知でないので下限）",
    "parity_group": "産次の群（1 / 2 / 3 / 4+）。dam_cohort_complete=1 のときだけ", "parity_conf": "産次の信頼度（推定=母が 2014 年以降生まれで産駒が全員既知 / 下限 / 不明）",
    "dam_cohort_complete": "1 = 母が 2014 年以降の生まれで、産駒が全員 2017 年以降に生まれる（既知の範囲）",
    "dam_first_foal": "母の初仔なら 1、2 仔目以降なら 0、不明は欠損", "dam_first_foal_conf": "初仔判定の信頼度（確実 / 推定 / 不明）。確実=前の産駒が既知、または母の出産時年齢が 3 歳以下",
    "sire_birth_year": "父の生年", "sire_first_crop_year": "父の既知の産駒の最小の生年", "sire_n_known_foals": "父の既知の産駒の総数",
    "sire_stud_year": "父の供用年 k（1=初年度産駒）。sire_first_crop_year ≥ 2018 で、父の生年が分かり初産駒の生年 − 父の生年 が 4〜10 の父だけ", "sire_year_group": "1 / 2 / 3 / 4+ / 既存（既知の最小の産駒の生年が 2015 年以前）。それ以外（2016・2017 年、輸入で生年不明、初産駒が遅い父）は欠損",
    "sire_crop_conf": "供用年の信頼度（確実=sire_year_group が決まった / 不明=決まらない）",
}


def write_table(out: Path, shared: Path | None = None) -> pd.DataFrame:
    L = pd.read_parquet(out / "lineage.parquet")
    T = foal_table(L)
    for d in {out, *([shared] if shared else [])}:
        d.mkdir(parents=True, exist_ok=True)
        T.to_parquet(d / "foal_lineage.parquet", index=False)
        readme = {
            "rows": int(len(T)), "key": "horse_id", "source": "data/local/horse_pedigree_5gen の (子, 父, 母) の三つ組（木の中の祖先を含む）と ID の先頭 4 桁（生年）",
            "note": "出走前に分かる情報だけ（as-of）。馬の生年・母の生年・既知の産駒は、その馬の出走時点で変わらない。先頭の打ち切り（2016 年以前は既知の産駒が一部だけ）に注意し、信頼度の列で絞る。",
            "complete_from_foal_year": COMPLETE_FROM_FOAL, "cohort_dam_from": COHORT_DAM_FROM, "sire_first_crop_from": SIRE_FIRST_CROP_FROM,
            "columns": README_COLUMNS,
            "counts": {
                "dam_age_known": int(T["dam_age_at_foaling"].notna().sum()),
                "foals_2017_2023": int(T["foal_year"].between(2017, 2023).sum()),
                "dam_first_foal_conf": T[T["foal_year"].between(2017, 2023)]["dam_first_foal_conf"].value_counts().to_dict(),
                "sire_year_group": T[T["foal_year"].between(2017, 2023)]["sire_year_group"].value_counts(dropna=False).to_dict(),
            },
        }
        (d / "README.json").write_text(json.dumps(readme, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return T


# ───────────────────────── データ（成績・予測・市場）
def build_data(T: pd.DataFrame) -> pd.DataFrame:
    r = G.build_runs()
    fr = pd.read_parquet(FRAME, columns=["race_id", "horse_id", "q_win"])
    fr["race_id"], fr["horse_id"] = fr["race_id"].astype(str), fr["horse_id"].astype(str)
    r = r.merge(fr.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    mg = []
    for y in range(2020, 2027):
        p = TABLES / str(y) / "megu_index_flat.parquet"
        if p.exists():
            m = pd.read_parquet(p, columns=["race_id", "horse_id", "megu_index_v3a_robust", "megu_v3a_low_confidence"])
            mg.append(m)
    m = pd.concat(mg, ignore_index=True)
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    m["megu"] = np.where(m["megu_v3a_low_confidence"].astype(bool), np.nan, pd.to_numeric(m["megu_index_v3a_robust"], errors="coerce"))
    r = r.merge(m[["race_id", "horse_id", "megu"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    r = r.merge(T, on="horse_id", how="left")
    pool = r["year"] <= 2024
    E = G.cell_expect(r, pool, cols=("win", "top3", "fp", "megu"))
    r = pd.concat([r, E], axis=1)
    r["d_megu"], r["d_fp"] = r["megu"] - r["exp_megu"], r["fp"] - r["exp_fp"]
    r["d_win"], r["d_top3"] = r["win"] - r["exp_win"], r["top3"] - r["exp_top3"]
    r["m_win"], r["m_mkt"] = r["win"] - r["p_win"], r["win"] - r["q_win"]
    r["roi_win"], r["roi_place"] = r["ret_win"] / 100.0, r["ret_place"] / 100.0
    r["stage"] = np.where(r["age"] <= 2, "2歳", np.where(r["age"] == 3, "3歳", "4歳以上"))
    r["age_lvl"] = r["dam_age_at_foaling"].clip(lower=5, upper=21)
    return r


# ───────────────────────── 群と比較（事前に固定）
def group_masks(r: pd.DataFrame) -> dict[str, pd.Series]:
    a = r["dam_age_at_foaling"]
    comp = r["dam_cohort_complete"] == 1.0
    ff = r["dam_first_foal"]
    sg = r["sire_year_group"]
    m: dict[str, pd.Series] = {}
    for name, lo, hi in AGE_BANDS:
        m[f"age_{name}"] = a.between(lo, hi)
    m["age_15+"], m["age_7-14"] = a >= 15, a.between(7, 14)
    m["first"], m["later"] = comp & (ff == 1.0), comp & (ff == 0.0)
    pm = r["parity_min"]
    m["par1"], m["par2"], m["par3"], m["par4+"] = comp & (pm == 1), comp & (pm == 2), comp & (pm == 3), comp & (pm >= 4)
    young, old = a <= 6, a >= 7
    m["F_old"], m["L_young"], m["F_young"], m["L_old"] = m["first"] & old, m["later"] & young, m["first"] & young, m["later"] & old
    for k in ("1", "2", "3", "4+"):
        m[f"sire_k{k}"] = sg == k
    m["sire_est"] = sg == "既存"
    m["F_x_k1"] = m["first"] & (sg == "1")
    m["L_x_est"] = m["later"] & (sg == "既存")
    return m


CONTRASTS = [
    # (id, ラベル, 群, 基準, 系統)
    (1, "母の出産時年齢 ≤6 vs 7–11", "age_≤6", "age_7-11", "母の年齢"),
    (2, "母の出産時年齢 12–14 vs 7–11", "age_12-14", "age_7-11", "母の年齢"),
    (3, "母の出産時年齢 15–17 vs 7–11", "age_15-17", "age_7-11", "母の年齢"),
    (4, "母の出産時年齢 18+ vs 7–11", "age_18+", "age_7-11", "母の年齢"),
    (5, "高齢出産（母 15 歳以上）vs 7–14 歳", "age_15+", "age_7-14", "母の年齢"),
    (6, "母の初仔（推定）vs 2 仔目以降", "first", "later", "初仔"),
    (7, "産次 2 vs 1", "par2", "par1", "産次"),
    (8, "産次 3 vs 1", "par3", "par1", "産次"),
    (9, "産次 4+ vs 1", "par4+", "par1", "産次"),
    (10, "初仔 × 母 7 歳以上 vs 2 仔目以降 × 7 歳以上", "F_old", "L_old", "初仔×年齢"),
    (11, "2 仔目以降 × 母 ≤6 歳 vs 2 仔目以降 × 7 歳以上", "L_young", "L_old", "初仔×年齢"),
    (12, "初仔 × 母 ≤6 歳 vs 2 仔目以降 × 7 歳以上", "F_young", "L_old", "初仔×年齢"),
    (13, "種牡馬の初年度産駒 vs 既存の種牡馬", "sire_k1", "sire_est", "種牡馬の供用年"),
    (14, "種牡馬の 2 年目 vs 既存", "sire_k2", "sire_est", "種牡馬の供用年"),
    (15, "種牡馬の 3 年目 vs 既存", "sire_k3", "sire_est", "種牡馬の供用年"),
    (16, "種牡馬の 4 年目以降 vs 既存", "sire_k4+", "sire_est", "種牡馬の供用年"),
    (17, "種牡馬の初年度 vs 2 年目", "sire_k1", "sire_k2", "種牡馬の供用年"),
    (18, "母の初仔 × 種牡馬の初年度 vs 2 仔目以降 × 既存", "F_x_k1", "L_x_est", "組み合わせ"),
]
HEADLINE = (5, 6, 13)


def run_contrasts(df: pd.DataFrame, masks: dict[str, pd.Series], metrics=METRICS, seed: int = SEED, ids=None) -> dict:
    """群 − 基準の差（指標ごとの平均の差）と、レース単位ブートストラップの 95% 区間・p 値。"""
    cons = [c for c in CONTRASTS if ids is None or c[0] in ids]
    if len(df) == 0:
        return {}
    races = df["race_id"].astype(str).to_numpy()
    uniq, inv = np.unique(races, return_inverse=True)
    nR = len(uniq)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(nR, np.full(nR, 1.0 / nR), size=B).astype(np.float32)
    names = sorted({c[2] for c in cons} | {c[3] for c in cons})
    cols, idx = [], {}
    for nm in names:
        sel = masks[nm].reindex(df.index).fillna(False).to_numpy()
        for mt in metrics:
            v = pd.to_numeric(df[mt], errors="coerce").to_numpy(float)
            ok = np.isfinite(v) & sel
            S = np.bincount(inv, weights=np.where(ok, v, 0.0), minlength=nR)
            C = np.bincount(inv, weights=ok.astype(float), minlength=nR)
            idx[(nm, mt)] = len(cols)
            cols.append(S); cols.append(C)
    M = np.stack(cols, axis=1)
    NUM = W @ M
    out: dict = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for cid, label, g, r_, fam in cons:
            o = {"label": label, "family": fam}
            for mt in metrics:
                ig, ir = idx[(g, mt)], idx[(r_, mt)]
                Sg, Cg, Sr, Cr = M[:, ig].sum(), M[:, ig + 1].sum(), M[:, ir].sum(), M[:, ir + 1].sum()
                if Cg < 30 or Cr < 30:
                    o[mt] = {"ng": int(Cg), "nr": int(Cr)}
                    continue
                pg, pr = Sg / Cg, Sr / Cr
                bs = NUM[:, ig] / NUM[:, ig + 1] - NUM[:, ir] / NUM[:, ir + 1]
                bs = bs[np.isfinite(bs)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                p = float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))
                o[mt] = {"g": float(pg), "r": float(pr), "diff": float(pg - pr), "lo": float(lo), "hi": float(hi), "p": max(p, 1.0 / B), "ng": int(Cg), "nr": int(Cr)}
            out[cid] = o
    return out


def bh(pvals: list[tuple[tuple, float]], q: float = 0.05) -> dict:
    ps = sorted(pvals, key=lambda x: x[1])
    m = len(ps)
    thr, k = 0.0, 0
    for i, (_, p) in enumerate(ps, 1):
        if p <= q * i / m:
            k, thr = i, p
    return {"m": m, "k": k, "threshold": thr, "passed": [key for key, p in ps[:k]]}


# ───────────────────────── 水準別（折れ線用）
def level_stats(df: pd.DataFrame, col: str, levels: list) -> dict:
    cols = {m: m for m in METRICS}
    out: dict = {}
    for stage in ("全体", "2歳", "3歳", "4歳以上"):
        sub = df if stage == "全体" else df[df["stage"] == stage]
        out[stage] = {}
        for lv in levels:
            s = sub[sub[col] == lv]
            if len(s) < 40:
                continue
            res = G.stats(s, cols)
            res["n_rows"] = int(len(s))
            out[stage][str(lv)] = res
    return out


# ───────────────────────── 特徴量化の増分（簡易。学習はしない）
def increment(r: pd.DataFrame) -> dict:
    from scipy.optimize import minimize

    d = r[r["p_win"].notna() & r["dam_age_at_foaling"].notna() & r["year"].between(2022, 2025)].copy()
    a = (d["dam_age_at_foaling"] - 10.0) / 5.0
    d["x_age"], d["x_age2"] = a, a * a
    d["x_first"] = (d["dam_first_foal"] == 1.0).astype(float)
    d["x_k1"], d["x_k2"] = (d["sire_year_group"] == "1").astype(float), (d["sire_year_group"] == "2").astype(float)
    variants = {"出産時年齢（1次+2次）": ["x_age", "x_age2"], "出産時年齢+初仔": ["x_age", "x_age2", "x_first"], "種牡馬の初年度・2年目": ["x_k1", "x_k2"],
                "年齢+初仔+種牡馬の供用年": ["x_age", "x_age2", "x_first", "x_k1", "x_k2"]}
    res: dict = {}
    for tgt, ycol, pcol in (("win", "win", "p_win"), ("top3", "top3", "p_top3")):
        res[tgt] = {}
        tr = d[d["year"] <= 2024]
        te = d[d["year"] == 2025].copy()
        eps = 1e-6
        for vn, xs in variants.items():
            def ll(beta, df):
                p = df[pcol].clip(eps, 1 - eps).to_numpy()
                z = np.log(p / (1 - p)) + beta[0] + df[xs].to_numpy() @ beta[1:]
                q = 1 / (1 + np.exp(-z))
                y = df[ycol].to_numpy()
                return -(y * np.log(q.clip(eps)) + (1 - y) * np.log((1 - q).clip(eps)))
            f = lambda b: ll(b, tr).mean() + 1e-3 * np.sum(b[1:] ** 2)
            b = minimize(f, np.zeros(len(xs) + 1), method="L-BFGS-B").x
            te[f"dll_{vn}"] = ll(b, te) - ll(np.zeros(len(xs) + 1), te)
            res[tgt][vn] = {"coef": [float(x) for x in b], **{"test2025": G.stats(te, {"dll": f"dll_{vn}"})["dll"]}}
    return res


# ───────────────────────── 本体
def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    T = pd.read_parquet(out / "foal_lineage.parquet") if (out / "foal_lineage.parquet").exists() else write_table(out)
    r = build_data(T)
    r = r[r["year"] <= 2026].copy()
    masks = group_masks(r)
    R: dict = {"prereg": {"contrasts": len(CONTRASTS), "metrics": METRICS, "tests": len(CONTRASTS) * len(METRICS), "headline": list(HEADLINE)}}
    R["counts"] = {
        "runs": int(len(r)), "horses": int(r["horse_id"].nunique()),
        "horses_dam_age_known": int(r.loc[r["dam_age_at_foaling"].notna(), "horse_id"].nunique()),
        "horses_foal_2017_2023": int(r.loc[r["foal_year"].between(2017, 2023), "horse_id"].nunique()),
        "first_foal_conf": r.drop_duplicates("horse_id").query("foal_year>=2017 and foal_year<=2023")["dam_first_foal_conf"].value_counts().to_dict(),
        "first_foal_flag": r.drop_duplicates("horse_id").query("foal_year>=2017 and foal_year<=2023")["dam_first_foal"].value_counts(dropna=False).to_dict(),
        "sire_year_group": r.drop_duplicates("horse_id").query("foal_year>=2017 and foal_year<=2023")["sire_year_group"].value_counts(dropna=False).to_dict(),
        "sire_crop_conf": r.drop_duplicates("horse_id").query("foal_year>=2017 and foal_year<=2023")["sire_crop_conf"].value_counts().to_dict(),
        "dam_age_hist": r.drop_duplicates("horse_id")["dam_age_at_foaling"].round().value_counts().sort_index().to_dict(),
        "dam_birth_src": r.drop_duplicates("horse_id")["dam_birth_src"].value_counts(dropna=False).to_dict(),
        "n_new_sires": int(r.loc[r["sire_year_group"].isin(["1", "2", "3", "4+"]), "sire_id"].nunique()),
        "n_first_crop_sires": int(r.loc[r["sire_year_group"] == "1", "sire_id"].nunique()),
    }
    disc = r["year"].between(2020, 2024)
    R["discovery"] = run_contrasts(r[disc], masks)
    allp = [((cid, mt), v[mt]["p"]) for cid, v in R["discovery"].items() for mt in METRICS if "p" in v.get(mt, {})]
    R["bh"] = bh(allp, 0.05)
    R["bh"]["passed"] = [list(k) for k in R["bh"]["passed"]]
    R["test2025"] = run_contrasts(r[r["year"] == 2025], masks)
    R["y2026"] = run_contrasts(r[r["year"] == 2026], masks)
    R["by_race_year"] = {str(y): run_contrasts(r[r["year"] == y], masks, ids=HEADLINE) for y in range(2020, 2027)}
    R["by_foal_year"] = {str(y): run_contrasts(r[(r["foal_year"] == y) & (r["year"] <= 2025)], masks, ids=HEADLINE) for y in range(2017, 2024)}
    R["by_stage"] = {st: run_contrasts(r[r["year"].between(2020, 2025) & (r["stage"] == st)], masks, ids=HEADLINE) for st in ("2歳", "3歳", "4歳以上")}
    # 種牡馬を揃えた頑健性: 指標から『その種牡馬の平均』を引く
    ms = []
    for mt in METRICS:
        r[f"{mt}__s"] = r[mt] - r.groupby("sire_id")[mt].transform("mean")
        ms.append(f"{mt}__s")
    sub = run_contrasts(r[r["year"].between(2020, 2025)], masks, metrics=ms, ids=(1, 5, 6))
    R["sire_adjusted"] = {cid: {mt[:-3]: v for mt, v in o.items() if mt.endswith("__s")} | {"label": o["label"]} for cid, o in sub.items()}
    R["increment"] = increment(r)

    base = r[r["year"].between(2020, 2025)]
    R["lv_dam_age"] = level_stats(base, "age_lvl", [float(x) for x in range(5, 22)])
    comp = base[base["dam_cohort_complete"] == 1.0].copy()
    comp["par_lvl"] = np.minimum(comp["parity_min"], 6)
    R["lv_parity"] = level_stats(comp, "par_lvl", [float(x) for x in range(1, 7)])
    newsire = base[base["sire_stud_year"].notna()].copy()
    newsire["k_lvl"] = np.minimum(newsire["sire_stud_year"], 6)
    R["lv_sire_year"] = level_stats(newsire, "k_lvl", [float(x) for x in range(1, 7)])
    R["sire_established"] = G.stats(base[base["sire_year_group"] == "既存"], {m: m for m in METRICS})
    R["all_rows"] = G.stats(base, {m: m for m in METRICS})
    hm = comp[comp["dam_age_at_foaling"].between(4, 11)].copy()
    hm["a"], hm["p"] = hm["dam_age_at_foaling"], np.minimum(hm["parity_min"], 6)
    R["heat"] = {f"{int(a)}|{int(p)}": {"n": int(len(s)), **{m: (float(s[m].mean()) if s[m].notna().any() else None) for m in ("m_win", "d_win", "d_megu", "roi_win")}}
                 for (a, p), s in hm.groupby(["a", "p"])}
    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: None if o is None else str(o)), encoding="utf-8")
    print("written", out / "results.json")


# ───────────────────────── 図
def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for cand in [*Path.home().glob(".fonts/*JP*.*"), *Path("/usr/share/fonts").glob("**/*CJK*JP*.*")]:
        try:
            fm.fontManager.addfont(str(cand))
        except (OSError, RuntimeError):
            pass
    have = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = [n for n in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "Yu Gothic", "Meiryo", "DejaVu Sans") if n in have][:1] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


PANELS = [("d_megu", "めぐ指数 − 同条件平均（点）", 1.0, 0.0), ("d_win", "勝率 − 同条件平均（pt）", 100.0, 0.0), ("d_top3", "複勝率 − 同条件平均（pt）", 100.0, 0.0),
          ("m_win", "勝率 − 現行モデルの予測（pt）", 100.0, 0.0), ("roi_win", "単勝回収率（%）", 100.0, 100.0), ("roi_place", "複勝回収率（%）", 100.0, 100.0)]
STAGE_COL = {"全体": "#111111", "2歳": "#e07b39", "3歳": "#2a9d8f", "4歳以上": "#6a4c93"}


def _line_fig(plt, S: dict, levels: list, xlabels: list[str], title: str, fname: Path, ref: dict | None = None, xlabel: str = "") -> None:
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, (mt, lab, mult, base) in zip(axs.ravel(), PANELS):
        for stage, col in STAGE_COL.items():
            xs, ys, lo, hi, ns = [], [], [], [], []
            for i, lv in enumerate(levels):
                e = S.get(stage, {}).get(str(lv))
                if not e or not e.get(mt):
                    continue
                xs.append(i); ys.append(e[mt]["v"] * mult); lo.append(e[mt]["lo"] * mult); hi.append(e[mt]["hi"] * mult); ns.append(e["n_rows"])
            if not xs:
                continue
            xs, ys, lo, hi, ns = map(np.array, (xs, ys, lo, hi, ns))
            thick = 2.2 if stage == "全体" else 1.1
            ax.fill_between(xs, lo, hi, color=col, alpha=0.16 if stage == "全体" else 0.07, linewidth=0)
            ax.plot(xs, ys, color=col, lw=thick, alpha=0.9, label=stage)
            weak = ns < 300
            ax.scatter(xs[~weak], ys[~weak], color=col, s=18 if stage == "全体" else 9, zorder=3)
            ax.scatter(xs[weak], ys[weak], facecolors="none", edgecolors=col, s=18 if stage == "全体" else 9, alpha=0.5, zorder=3)
        if ref and ref.get(mt):
            ax.axhline(ref[mt]["v"] * mult, color="#999999", lw=1, ls=":")
        ax.axhline(base, color="#bbbbbb", lw=0.8)
        ax.set_title(lab, fontsize=10)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(title + "\n（帯 = 95% 区間、白抜き = 標本 300 走未満、2020–2025 年）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fname, dpi=120)
    plt.close(fig)


def plot(out: Path) -> None:
    plt = _setup_plt()
    R = json.loads((out / "results.json").read_text(encoding="utf-8"))
    lv = [float(x) for x in range(5, 22)]
    labels = ["≤5"] + [str(int(x)) for x in lv[1:-1]] + ["≥21"]
    _line_fig(plt, R["lv_dam_age"], lv, labels, "母の出産時年齢（1 歳刻み）と成績", out / "fig_dam_age.png", xlabel="母の出産時年齢（歳）")
    _line_fig(plt, R["lv_parity"], [float(x) for x in range(1, 7)], ["1", "2", "3", "4", "5", "6+"], "産次（母の既知の産駒の順。母 2014 年以降生まれのみ）と成績", out / "fig_parity.png", xlabel="産次（下限）")
    _line_fig(plt, R["lv_sire_year"], [float(x) for x in range(1, 7)], ["1（初年度）", "2", "3", "4", "5", "6+"], "種牡馬の供用年（初年度産駒の最初の年から数える。2018 年以降に産駒が出た種牡馬）と成績", out / "fig_sire_year.png",
              ref=R["sire_established"], xlabel="供用年（点線 = 既存の種牡馬の平均）")
    # ヒートマップ
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ages, pars = list(range(4, 12)), list(range(1, 7))
    for ax, key, ttl, mult in ((axs[0], "n", "標本数（走）", 1), (axs[1], "m_win", "勝率 − 現行モデルの予測（pt）", 100)):
        Z = np.full((len(pars), len(ages)), np.nan)
        N = np.zeros_like(Z)
        for i, p in enumerate(pars):
            for j, a in enumerate(ages):
                c = R["heat"].get(f"{a}|{p}")
                if c:
                    N[i, j] = c["n"]
                    v = c[key]
                    Z[i, j] = (v * mult) if (v is not None and (key == "n" or c["n"] >= 150)) else np.nan
        im = ax.imshow(Z, cmap="viridis" if key == "n" else "RdBu", aspect="auto", origin="lower",
                       vmin=None if key == "n" else -3, vmax=None if key == "n" else 3)
        ax.set_xticks(range(len(ages))); ax.set_xticklabels(ages); ax.set_yticks(range(len(pars))); ax.set_yticklabels(["1", "2", "3", "4", "5", "6+"])
        ax.set_xlabel("母の出産時年齢（歳）"); ax.set_ylabel("産次（下限）"); ax.set_title(ttl, fontsize=10)
        for i in range(len(pars)):
            for j in range(len(ages)):
                if np.isfinite(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.0f}" if key == "n" else f"{Z[i, j]:+.1f}", ha="center", va="center", fontsize=7, color="white" if key == "n" else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("母の出産時年齢 × 産次（母 2014 年以降生まれ。右は 150 走以上のセルのみ）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out / "fig_heat.png", dpi=120); plt.close(fig)
    # 生年・開催年ごとの推移（群 − 基準）
    for key, ttl, fn, xs in (("by_foal_year", "馬の生年（世代）ごとの差（群 − 基準）", "fig_foalyear.png", [str(y) for y in range(2017, 2024)]),
                             ("by_race_year", "出走した年ごとの差（群 − 基準。2026 年は記述のみ）", "fig_raceyear.png", [str(y) for y in range(2020, 2027)])):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
        col = {5: "#d1495b", 6: "#2a9d8f", 13: "#6a4c93"}
        name = {5: "高齢出産（15+）vs 7–14", 6: "母の初仔 vs 2 仔目以降", 13: "種牡馬の初年度 vs 既存"}
        for ax, (mt, lab, mult, base) in zip(axs.ravel(), PANELS):
            for cid in HEADLINE:
                xx, yy, lo, hi, ns = [], [], [], [], []
                for i, y in enumerate(xs):
                    e = (R[key].get(y) or {}).get(str(cid), {}).get(mt) or {}
                    if "diff" not in e:
                        continue
                    xx.append(i); yy.append(e["diff"] * mult); lo.append(e["lo"] * mult); hi.append(e["hi"] * mult); ns.append(e["ng"])
                if not xx:
                    continue
                xx, yy, lo, hi, ns = map(np.array, (xx, yy, lo, hi, ns))
                ax.fill_between(xx, lo, hi, color=col[cid], alpha=0.12, linewidth=0)
                ax.plot(xx, yy, color=col[cid], lw=1.6, label=name[cid])
                weak = ns < 300
                ax.scatter(xx[~weak], yy[~weak], color=col[cid], s=14)
                ax.scatter(xx[weak], yy[weak], facecolors="none", edgecolors=col[cid], s=14, alpha=0.6)
            ax.axhline(0, color="#999999", lw=0.8)
            ax.set_title(lab.replace("（", "の差（", 1) if mt.startswith("roi") else lab, fontsize=10)
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8); ax.grid(alpha=0.2)
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(ttl + "（帯 = 95% 区間、白抜き = 群の標本 300 走未満）", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out / fn, dpi=120); plt.close(fig)
    print("figures written", out)


# ───────────────────────── 章
def _d(e, mult=100.0, d=1, signed=True):
    if not e or "diff" not in e:
        return "–"
    return G._ci({"v": e["diff"], "lo": e["lo"], "hi": e["hi"]}, d, signed, mult)


def _png_b64(path: Path) -> str:
    import base64
    import io

    from PIL import Image

    im = Image.open(path).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=96)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def chapter(out: Path, dest: Path) -> None:
    import re

    R = json.loads((out / "results.json").read_text(encoding="utf-8"))
    css = re.search(r"<style>.*?</style>", (ROOT / "docs/html/modeling/chapters/equipment_and_change.html").read_text(encoding="utf-8"), re.S).group(0)
    C, D, T25, Y26 = R["counts"], R["discovery"], R["test2025"], R["y2026"]
    imgs = {k: _png_b64(out / f"fig_{k}.png") for k in ("dam_age", "parity", "sire_year", "heat", "foalyear", "raceyear")}
    meg = lambda e: _d(e, 1.0, 2)
    rows = ""
    for cid, o in D.items():
        a = o["d_top3"]
        rows += (f"<tr><td>{cid}</td><td>{o['label']}</td><td class='num'>{a.get('ng', 0):,} / {a.get('nr', 0):,}</td>"
                 f"<td class='num'>{_d(o['d_top3'])}</td><td class='num'>{_d(T25[str(cid)]['d_top3'])}</td><td class='num'>{_d(Y26[str(cid)]['d_top3'])}</td>"
                 f"<td class='num'>{_d(o['m_win'])}</td><td class='num'>{_d(T25[str(cid)]['m_win'])}</td>"
                 f"<td class='num'>{_d(o['roi_win'], 100.0, 0)}</td><td class='num'>{_d(o['roi_place'], 100.0, 0)}</td></tr>")
    head_rows = ""
    for cid in R["prereg"]["headline"]:
        o, t, y = D[str(cid)], T25[str(cid)], Y26[str(cid)]
        for mt, lab in (("d_megu", "めぐ指数 − 同条件平均（点）"), ("d_top3", "複勝率 − 同条件平均（pt）"), ("m_win", "勝率 − 現行モデルの予測（pt）"), ("m_mkt", "勝率 − 市場の勝率（pt）"), ("roi_win", "単勝回収率の差（pt）"), ("roi_place", "複勝回収率の差（pt）")):
            f = (lambda e: _d(e, 1.0, 2)) if mt == "d_megu" else (lambda e: _d(e, 100.0, 0 if mt.startswith("roi") else 1))
            head_rows += f"<tr><td>{o['label'] if mt == 'd_megu' else ''}</td><td>{lab}</td><td class='num'>{f(o[mt])}</td><td class='num'>{f(t[mt])}</td><td class='num'>{f(y[mt])}</td></tr>"
    st_rows = ""
    for st, d in R["by_stage"].items():
        for cid in ("5", "6", "13"):
            o = d[cid]
            st_rows += (f"<tr><td>{st}</td><td>{o['label']}</td><td class='num'>{_d(o['d_megu'], 1.0, 2)}</td><td class='num'>{_d(o['d_top3'])}</td>"
                        f"<td class='num'>{_d(o['m_win'])}</td><td class='num'>{_d(o['roi_win'], 100.0, 0)}</td></tr>")
    sa_rows = "".join(f"<tr><td>{o['label']}</td><td class='num'>{_d(o['d_megu'], 1.0, 2)}</td><td class='num'>{_d(o['d_top3'])}</td><td class='num'>{_d(o['m_win'])}</td></tr>" for o in R["sire_adjusted"].values())
    inc_rows = ""
    for tgt, nm in (("win", "勝率"), ("top3", "複勝率")):
        for vn, x in R["increment"][tgt].items():
            inc_rows += f"<tr><td>{nm}</td><td>{vn}</td><td class='num'>{G._ci(x['test2025'], 2, True, 1e4)}</td></tr>"
    bh = R["bh"]
    cnt = C["first_foal_conf"]
    html = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>母の初仔・出産時年齢・産次と種牡馬の初年度産駒の成績 — keiba-vpn</title>
{css}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>母の初仔・出産時年齢・産次と、種牡馬の初年度産駒の成績</h1>
<p class="muted">ユーザーの問い（2026-10-22）: 『母の初年度産駒と高齢出産産駒の成績は悪いように思う』『種牡馬の初年度産駒の傾向と、年が進むときの変化も見たい』。2020–2025 年の JRA（芝・ダート、障害を除く）{C['runs']:,} 走・{C['horses']:,} 頭。2026 年は記述統計のみ。市場オッズは回収率と市場との差（評価）にだけ使い、特徴量にはしない。</p>
<div class="note"><b>要点.</b>
<ul>
<li><b>『悪い』は本当だが、現行モデルと市場はすでに織り込んでいる。</b> 高齢出産（母 15 歳以上）の産駒は同条件の平均より複勝率が {_d(D['5']['d_top3'])}pt（2025 年 {_d(T25['5']['d_top3'])}）低く、母の初仔（推定）は {_d(D['6']['d_top3'])}pt（2025 年 {_d(T25['6']['d_top3'])}・2026 年 {_d(Y26['6']['d_top3'])}）低い。年ごとに同じ向きで、種牡馬を揃えても変わらない。一方、現行モデルの予測との差は高齢出産で {_d(D['5']['m_win'])}pt、初仔で {_d(D['6']['m_win'])}pt、市場との差も同水準で、<b>0 と区別できない</b>。</li>
<li><b>回収率に優位はない。</b> 144 本の検定（18 比較 × 8 指標）のうち、BH（FDR 5%）を通ったのは {bh['k']} 本で、すべて『同条件平均との差』の指標（めぐ指数・勝率・複勝率・着順の位置）。モデル・市場との差と回収率は 1 本も通らない。回収率の差が選択期間で 0 を外れたのは 4 本（初仔の複勝 {_d(D['6']['roi_place'], 100.0, 0)}pt、初仔 × 母 ≤6 歳の複勝 {_d(D['12']['roi_place'], 100.0, 0)}pt、供用 2 年目の単勝 {_d(D['14']['roi_win'], 100.0, 0)}pt、供用 3 年目の複勝 {_d(D['15']['roi_place'], 100.0, 0)}pt）で、2025 年に同じ向きで再現したのは『初仔 × 母 ≤6 歳の複勝』（{_d(T25['12']['roi_place'], 100.0, 0)}pt）だけ。これは<b>買わない側（消し）の目安</b>にはなるが、多重比較の範囲では偶然を排除できない（p=0.048）。買う側の優位は無い。</li>
<li><b>効く場所は 2〜3 歳。</b> 高齢出産の複勝率の差は 2 歳 {_d(R['by_stage']['2歳']['5']['d_top3'])}pt・3 歳 {_d(R['by_stage']['3歳']['5']['d_top3'])}pt に対し、4 歳以上 {_d(R['by_stage']['4歳以上']['5']['d_top3'])}pt（めぐ指数は 0）。影響は成長とともに薄れる。</li>
<li><b>母が若すぎる（5〜6 歳以下）ときも悪い。</b> 出産時年齢は両端で悪い U 字（≤6 歳の複勝率 {_d(D['1']['d_top3'])}pt）。『初仔 × 母 ≤6 歳』は {_d(D['12']['d_top3'])}pt と最も悪いが、これも現行モデルは織り込み済み（{_d(D['12']['m_win'])}pt）。2 仔目・3 仔目は 1 仔目より複勝率が高い（{_d(D['7']['d_top3'])}pt・{_d(D['8']['d_top3'])}pt）。</li>
<li><b>種牡馬の初年度産駒は、初年度だから悪い／良いという差は無い。</b> 初年度 vs 既存の種牡馬: 複勝率 {_d(D['13']['d_top3'])}pt（2025 年は {_d(T25['13']['d_top3'])}pt だが、モデルとの差は {_d(T25['13']['m_win'])}pt）。供用 2 年目の単勝回収率が選択期間で {_d(D['14']['roi_win'], 100.0, 0)}pt に見えるが、2025 年 {_d(T25['14']['roi_win'], 100.0, 0)}pt・2026 年 {_d(Y26['14']['roi_win'], 100.0, 0)}pt で再現しない。供用 4 年目以降は既存より複勝率が高い（成功した種牡馬だけが供用を続ける選抜）。</li>
<li><b>特徴量化は見送り。</b> 『母の出産時年齢（1 次+2 次）＋初仔＋種牡馬の供用年』を現行モデルの確率に足しても、test 2025 の対数損失の改善は勝率 {G._ci(R['increment']['win']['年齢+初仔+種牡馬の供用年']['test2025'], 2, True, 1e4)}（×10⁻⁴）、複勝率 {G._ci(R['increment']['top3']['年齢+初仔+種牡馬の供用年']['test2025'], 2, True, 1e4)}（×10⁻⁴）。採用基準（0.0002）の半分にも届かない。</li>
</ul></div>

<h2>1. データと『初仔』『初年度』の判定の信頼度</h2>
<ul>
<li><b>馬の生年・母の生年は ID の先頭 4 桁</b>（netkeiba の 10 桁 ID）から取る。母の生年が分かった馬は {C['horses_dam_age_known']:,} 頭（全体の {C['horses_dam_age_known'] / C['horses'] * 100:.0f}%。残りは外国産の母で、birth_manual.csv にも無いもの）。母の出産時年齢 = 馬の生年 − 母の生年。</li>
<li><b>産駒の一覧は 5gen の血統表（出走馬と、その木の中の祖先）から作る。</b> 2017〜2023 年生まれはほぼ全頭が既知だが、2016 年以前は一部だけ。したがって『前の産駒が既知か』で 2 仔目以降を<b>確実</b>に判定でき（{cnt.get('確実', 0):,} 頭。うち母の出産時年齢 3 歳以下の初仔が 2 頭）、母が 2014 年以降の生まれ（産駒がすべて 2017 年以降＝既知の範囲）で前の産駒が無いものを<b>推定</b>の初仔とする（{cnt.get('推定', 0):,} 頭）。母が古い世代で前の産駒が無いものは<b>不明</b>（{cnt.get('不明', 0):,} 頭。初仔かどうか決められない）。</li>
<li><b>推定の初仔の限界.</b> 地方競馬・未出走の前の産駒は見えないので、推定の初仔には 2 仔目以降が混ざり得る（過大に拾う側）。初仔の判定は母が 4〜9 歳の範囲に偏る（若い母の世代に限られるため）。産次は『既知の前の産駒の数 + 1』の<b>下限</b>で、5 以上はほぼ標本が無い。</li>
<li><b>種牡馬の供用年 k.</b> 『既知の産駒の最小の生年』が 2018 年以降で、父の生年が分かり、その差が 4〜10 年の種牡馬（{C['n_new_sires']} 頭）だけ k=1（初年度）, 2, 3, 4+ を決める。2015 年以前から産駒がいる種牡馬を『既存』とし、2016・2017 年は初年度が見えていない恐れがあるので、どちらにも入れない。輸入種牡馬・生年不明・初産駒が遅い種牡馬は不明（2017〜2023 年生まれの {C['horses_foal_2017_2023']:,} 頭のうち約 {C['sire_year_group'].get('null', 0) / max(C['horses_foal_2017_2023'], 1) * 100:.0f}%）。</li>
<li>共有表: <code>data/local/modeling/shared_cache/foal_lineage/foal_lineage.parquet</code>（キー horse_id。列の定義は同じディレクトリの README.json）。出走前に分かる情報だけ（as-of）。</li>
</ul>

<h2>2. 検定の枠組み（事前に固定）</h2>
<ul>
<li><b>18 の比較 × 8 の指標 = 144 本.</b> 比較: 母の出産時年齢の帯（≤6・12–14・15–17・18+ を 7–11 と比較、高齢出産 15+ vs 7–14）、母の初仔 vs 2 仔目以降、産次 2・3・4+ vs 1、初仔 × 年齢の 2 × 2、種牡馬の供用年（初年度・2 年目・3 年目・4 年目以降 vs 既存、初年度 vs 2 年目）、初仔 × 種牡馬の初年度。指標: めぐ指数・着順の位置・勝率・複勝率（いずれも『同条件平均との差』）、勝率 − 現行モデルの予測、勝率 − 市場の勝率、単勝・複勝の回収率。</li>
<li>『同条件平均』= 芝ダ × 距離帯 × クラス × 年齢（6+ 区分）× 休み明けのセルの平均（2020–2024 年のプール）。現行モデルの予測と市場（確定単勝）は 2022 年〜。差と区間はレース単位のブートストラップ（{B} 回）。</li>
<li>発見は 2020–2024 年（モデル・市場・回収率は 2022–2024 年）。p 値の BH（FDR 5%）で {bh['k']} / {bh['m']} 本が通過。確認は 2025 年（1 回）、2026 年は記述のみ。</li>
<li><b>1 年ごとの図は、この事前に固定した指標の記述的な図示で、検定の対象を増やすものではない。</b> 標本が 300 走未満の点は白抜きにしてある。</li>
</ul>

<h2>3. 主要な 3 つの比較</h2>
<table><tr><th>比較</th><th>指標</th><th>発見 2020–24</th><th>test 2025</th><th>2026（記述）</th></tr>{head_rows}</table>
<p class="muted">[ ] はレース単位ブートストラップの 95% 区間。群 − 基準の差。</p>

<h2>4. 全ての比較</h2>
<table><tr><th>#</th><th>比較</th><th>走数（群 / 基準）</th><th>複勝率 − 同条件平均（pt）発見</th><th>2025</th><th>2026</th><th>勝率 − モデル（pt）発見</th><th>2025</th><th>単勝回収率の差（pt）発見</th><th>複勝回収率の差（pt）発見</th></tr>{rows}</table>

<h2>5. 1 年ごとの図</h2>
<h3>5.1 母の出産時年齢（1 歳刻み）</h3>
<img alt="母の出産時年齢と成績" style="max-width:100%" src="data:image/png;base64,{imgs['dam_age']}" />
<p>5 歳以下・21 歳以上は標本が少ないので束ねた。めぐ指数・勝率・複勝率の『同条件平均との差』は両端で下がる U 字（若すぎる母と、15 歳を超える母）だが、『現行モデルの予測との差』は全年齢でほぼ 0 の帯の中、回収率は 60〜90% の水準で年齢に依らない。系列は馬の年齢段階（2 歳・3 歳・4 歳以上）で分けてあり、高齢側の低下は 2 歳・3 歳で目立つ。</p>
<h3>5.2 産次</h3>
<img alt="産次と成績" style="max-width:100%" src="data:image/png;base64,{imgs['parity']}" />
<p>母が 2014 年以降の生まれの馬だけ（産次が決められる範囲）。1 仔目が低く、2・3 仔目で上がるが、モデルとの差はほぼ 0。4 仔目以上は標本が小さい。</p>
<h3>5.3 種牡馬の供用年</h3>
<img alt="種牡馬の供用年と成績" style="max-width:100%" src="data:image/png;base64,{imgs['sire_year']}" />
<p>点線は既存の種牡馬の平均。初年度産駒は既存と同水準で、5 年目以降は選抜（成功した種牡馬だけが供用を続ける）で複勝率が高く見える。</p>
<h3>5.4 母の出産時年齢 × 産次</h3>
<img alt="母の出産時年齢と産次のヒートマップ" style="max-width:100%" src="data:image/png;base64,{imgs['heat']}" />
<h3>5.5 馬の生年（世代）ごと／出走した年ごとの推移</h3>
<img alt="世代ごとの差" style="max-width:100%" src="data:image/png;base64,{imgs['foalyear']}" />
<img alt="出走年ごとの差" style="max-width:100%" src="data:image/png;base64,{imgs['raceyear']}" />
<p>高齢出産の複勝率の差は、どの世代・どの年も負で再現する。母の初仔も 2020 年生まれ以降は負で揃う。モデルとの差は、どの世代・年も 0 の帯の中（初仔の 2026 年だけ {_d(Y26['6']['m_win'])}pt と区間が 0 を外れるが、発見期間・2025 年では出ておらず、選択には使わない）。</p>

<h2>6. 層別・頑健性</h2>
<h3>6.1 馬の年齢段階別（2020–2025 年）</h3>
<table><tr><th>段階</th><th>比較</th><th>めぐ指数 − 同条件</th><th>複勝率 − 同条件（pt）</th><th>勝率 − モデル（pt）</th><th>単勝回収率の差（pt）</th></tr>{st_rows}</table>
<h3>6.2 種牡馬を揃えた比較（指標から『その種牡馬の平均』を引く）</h3>
<table><tr><th>比較</th><th>めぐ指数 − 同条件</th><th>複勝率 − 同条件（pt）</th><th>勝率 − モデル（pt）</th></tr>{sa_rows}</table>
<p class="muted">種牡馬の効果を引いても差はほとんど変わらない（差は種牡馬の違いではない）。</p>

<h2>7. 特徴量化の見込み（学習はしていない）</h2>
<p>現行モデルの確率（logit）に、母の出産時年齢・初仔・種牡馬の供用年を足した補正を 2022–2024 年で当てはめ、test 2025 で 1 回だけ対数損失の変化を測った（負 = 改善、×10⁻⁴）。</p>
<table><tr><th>目的</th><th>追加する説明変数</th><th>対数損失の変化（×10⁻⁴）</th></tr>{inc_rows}</table>
<p>複勝率では出産時年齢で −0.8 程度の改善が見えるが、採用基準（0.0002 = 2 × 10⁻⁴）の半分にも届かない。『同条件平均との差』は大きくても、それは現行モデルがすでに持つ情報（種牡馬・母・馬の履歴）で説明されている。</p>

<h2>8. 結論</h2>
<ul>
<li>『高齢出産・母の初仔は成績が悪い』は事実（同条件平均との差は年ごとに再現）。ただし<b>現行モデルと市場はそれを織り込み済みで、回収率の妙味は無い</b>。『買う要素』にも『消す要素』にも使えない。</li>
<li>種牡馬の初年度産駒に、初年度であることによる傾向は見えない。供用 2 年目の単勝回収率の高さは再現しない。『初仔 × 母 ≤6 歳』の複勝回収率が低い（選択期間・2025 年とも −12〜−14pt）のは、買わない側の目安として監視に値するが、独立の確認が 2025 年の 1 回だけで、凍結して前向き追跡する対象にはしていない。</li>
<li>共有表 <code>foal_lineage</code> は、他の分析（血統クラスタ、ペア走査の記述子）の入力に使える。初仔の判定は『推定』を含むので、信頼度の列で絞って使うこと。</li>
</ul>
<h2>9. 限界</h2>
<ul>
<li>初仔の判定は母が 2014 年以降の生まれに限られる（若い母が中心）。母が古い世代の初仔は、データからは判定できない。</li>
<li>産駒の一覧は 5gen の既知の範囲。地方競馬・未出走の前の産駒は見えない。</li>
<li>モデルとの差は <code>betting/tp36/frame_all.parquet</code> の予測（2022 年〜）に対するもの。市場との差は確定単勝オッズ。</li>
<li>2026 年は記述のみで、選択・閾値に使っていない。</li>
</ul>
<p class="muted">再現: <code>python -m src.scripts.data.dam_age_analysis lineage → table → run → plot → chapter</code>（ホストでは heavy_run.sh 経由）。結果は <code>data/local/modeling/dam_age/</code>、共有表は <code>data/local/modeling/shared_cache/foal_lineage/</code>。</p>
</body></html>
"""
    dest.write_text(html, encoding="utf-8")
    print("chapter written", dest, len(html))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lineage", "table", "run", "plot", "chapter", "all"])
    ap.add_argument("--out", default=str(ROOT / "data/local/modeling/dam_age"))
    ap.add_argument("--shared", default="")
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd in ("lineage", "all"):
        build_lineage(out)
    if a.cmd in ("table", "all"):
        write_table(out, Path(a.shared) if a.shared else None)
    if a.cmd in ("run", "all"):
        run(out)
    if a.cmd in ("plot", "all"):
        plot(out)
    if a.cmd in ("chapter",):
        chapter(out, ROOT / "docs/html/modeling/chapters/dam_age_firstfoal.html")


if __name__ == "__main__":
    main()
