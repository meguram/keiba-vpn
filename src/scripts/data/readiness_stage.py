"""導入可否の確認用: 前段（s_time_pred / s_time_gap / s_rank_pred）と補助ヘッド（mtl_*）を、2026 年の抜き取りで 2025 年の OOF と比べる（readiness-audit）。

  python -m src.scripts.data.readiness_stage --stage-dir <readiness_v5_sample の出力>/stage --out <json>

基準は readiness_criteria.json の stage_aux_models（実測との順位相関が 2025 年 OOF の 90% 以上、欠損率が学習時 +5 ポイント以内）。
ラベルは 2025・2026 とも race_result_flat から同じ式で作る（レース内の走破タイム偏差・着順の正規化・上がり偏差）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {  # 出力列 → ラベル列
    "s_time_pred": "t_time_dev", "s_time_gap": "t_time_dev", "s_rank_pred": "t_rank_norm",
    "mtl_t_time_dev": "t_time_dev", "mtl_t_rank_norm": "t_rank_norm", "mtl_t_l3f_dev": "t_l3f_dev",
}
MIN_RACE_N = 5


def build_labels(tables: Path, year: int) -> pd.DataFrame:
    r = pd.read_parquet(tables / str(year) / "race_result_flat.parquet",
                        columns=["race_id", "horse_id", "surface", "finish_position", "time_sec", "last_3f"]).drop_duplicates(["race_id", "horse_id"])
    r["race_id"] = r["race_id"].astype(str)
    r["horse_id"] = r["horse_id"].astype(str)
    fp = pd.to_numeric(r["finish_position"], errors="coerce").where(lambda s: s >= 1)
    n_fin = fp.groupby(r["race_id"]).transform("count")
    g = r["race_id"]
    ts = pd.to_numeric(r["time_sec"], errors="coerce").where(lambda s: s > 0)
    l3 = pd.to_numeric(r["last_3f"], errors="coerce").where(lambda s: s > 0).where(~r["surface"].astype(str).str.contains("障"))
    out = r[["race_id", "horse_id"]].copy()
    out["t_time_dev"] = (ts - ts.groupby(g).transform("mean")).where(ts.groupby(g).transform("count") >= MIN_RACE_N)
    out["t_l3f_dev"] = (l3 - l3.groupby(g).transform("mean")).where(l3.groupby(g).transform("count") >= MIN_RACE_N)
    out["t_rank_norm"] = ((fp - 1) / (n_fin - 1)).where(n_fin >= 2).clip(0, 1)
    return out


def within_race_spearman(df: pd.DataFrame, col: str, lab: str) -> dict:
    """レース内の Spearman（出力と実測のレース内順位）。平均とレース数。出力の符号はそのまま（時間・着順は小さいほど良い＝正の相関が期待値）。"""
    d = df[["race_id", col, lab]].dropna()
    cs = []
    for _, g in d.groupby("race_id"):
        if len(g) >= MIN_RACE_N and g[col].nunique() > 1 and g[lab].nunique() > 1:
            cs.append(g[col].rank().corr(g[lab].rank()))
    return {"races": len(cs), "spearman_within_race": float(np.mean(cs)) if cs else None, "pearson_pooled": float(d[col].corr(d[lab])) if len(d) > 2 else None}


def evaluate(pred: pd.DataFrame, lab: pd.DataFrame, cols: list[str]) -> dict:
    m = pred.merge(lab, on=["race_id", "horse_id"], how="inner")
    out = {"rows": int(len(m)), "races": int(m["race_id"].nunique()), "missing_rate": {c: float(pred[c].isna().mean()) for c in cols if c in pred.columns}}
    for c in cols:
        if c in m.columns and PAIRS[c] in m.columns:
            out[c] = within_race_spearman(m, c, PAIRS[c])
    return out


def judge(ref: dict, cur: dict, cols: list[str], ratio_min: float = 0.90, miss_pp: float = 0.05) -> dict:
    res = {}
    for c in cols:
        a, b = (ref.get(c) or {}).get("spearman_within_race"), (cur.get(c) or {}).get("spearman_within_race")
        mr_ref, mr_cur = ref["missing_rate"].get(c), cur["missing_rate"].get(c)
        if a is None or b is None:
            res[c] = {"ok": None}
            continue
        ratio = b / a if a else None
        res[c] = {"ref": a, "cur": b, "ratio": ratio, "corr_ok": bool(ratio is not None and ratio >= ratio_min),
                  "missing_ref": mr_ref, "missing_cur": mr_cur, "missing_ok": bool(mr_cur is not None and mr_ref is not None and mr_cur <= mr_ref + miss_pp)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="data/page_reference/tables")
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--oof-stage", default="data/local/modeling/training_pipeline_v4/oof/stage_features_expanding.parquet")
    ap.add_argument("--oof-aux", default="data/local/modeling/training_pipeline/final_aux/aux_oof_2025.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tables = Path(a.tables)
    cols = list(PAIRS)
    lab25, lab26 = build_labels(tables, 2025), build_labels(tables, 2026)

    parts = [pd.read_parquet(f) for f in sorted(Path(a.stage_dir).glob("*/*.parquet"))]
    cur = pd.concat(parts, ignore_index=True)
    cur["race_id"] = cur["race_id"].astype(str)
    cur["horse_id"] = cur["horse_id"].astype(str)

    st = pd.read_parquet(a.oof_stage)
    st["race_id"] = st["race_id"].astype(str)
    st["horse_id"] = st["horse_id"].astype(str)
    ax = pd.read_parquet(a.oof_aux)
    ax["race_id"] = ax["race_id"].astype(str)
    ax["horse_id"] = ax["horse_id"].astype(str)
    ref_df = st.merge(ax, on=["race_id", "horse_id"], how="inner")
    ref_df = ref_df[ref_df["race_id"].isin(set(lab25["race_id"]))]

    # 同じ頭数規模で比べるため、2025 年を抜き取りと同じ数のレースで繰り返し切り出した平均も出す
    ref_full = evaluate(ref_df, lab25, cols)
    ids = ref_df["race_id"].unique()
    k = min(cur["race_id"].nunique(), len(ids))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(20):
        pick = set(rng.choice(ids, k, replace=False))
        boots.append(evaluate(ref_df[ref_df["race_id"].isin(pick)], lab25, cols))
    ref_sd = {c: float(np.std([b[c]["spearman_within_race"] for b in boots if c in b and b[c]["spearman_within_race"] is not None])) for c in cols}
    cur_ev = evaluate(cur, lab26, cols)
    out = {"sample_races_2026": cur_ev["races"], "ref_2025_full": ref_full, "ref_2025_sd_at_sample_size": ref_sd, "cur_2026": cur_ev, "judge": judge(ref_full, cur_ev, cols)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"judge": out["judge"], "sd": ref_sd, "races": cur_ev["races"]}, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
