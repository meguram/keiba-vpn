"""学習と推論の X の欠けの率のふるい（読み取りのみ・手元のファイルだけ。GCS は読まない）。

  python -m src.scripts.data.skew_sieve --served-glob '/home/jovyan/work/retrain_out/candidate_live/inf/stage2_x/*/*.parquet' \
      --train-x /home/jovyan/work/retrain_out/cycles16_fix3/X_xgb.parquet --train-meta /home/jovyan/work/retrain_out/cycles16_fix3/meta.parquet \
      --years 2024,2025 --months 10-12 --out retrain_out/train_serve_skew/s0/sieve.csv

推論で実際に見えた X（served）と、学習の X の同じクラス・同じ季節の行で、列ごとの欠けの割合を比べる。
クラス（新馬・未勝利・障害・その他）で層に分け、推論の側の層の割合で重みを付ける（推論の対象が未勝利・新馬だけでも比べられる）。
学習の meta からは race_id・date・race_class だけを読む（着順は読まない）。詳しくは docs/operations/train_serve_skew_20261126.md。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 列の群（接頭辞）。先に当たったものを採る。棚卸しと見張りの集計に使う
GROUPS: list[tuple[str, str]] = [
    ("tl_w1", r"^tl_w1_"), ("tl_w2", r"^tl_w2_"), ("tl", r"^tl_"), ("wc", r"^wc_"), ("oikiri", r"^oikiri_"),
    ("weight", r"^(weight|weight_change)$|^d_wt_"), ("prev_day", r"prev_day"), ("rb", r"^rb_"),
    ("stage", r"^(s_|EASE2_|lp_|ld_|pf|pp_|z_hat)"), ("attr", r"^attr_"), ("px", r"^px[a-z]?_"), ("mtl", r"^mtl_"),
    ("fm", r"^fm_"), ("fd", r"^fd_"), ("fe", r"^fe_"), ("f1", r"^f1_"), ("sr", r"^sr_"), ("seq", r"^seq_"), ("lag", r"^lag\d"),
    ("comment", r"^comment_"), ("pc", r"^pc_"),
]


def group_of(col: str) -> str:
    for g, pat in GROUPS:
        if re.search(pat, col):
            return g
    m = re.match(r"^([a-z0-9]+)_", col)
    return m.group(1) if m else "other"


def stratum(race_class: pd.Series, surface: pd.Series | None = None) -> np.ndarray:
    """層: shinba・misho・jump・other。障害は race_class が空で面が「障」（学習の行も推論の行も同じ）。"""
    c = race_class.astype("string").fillna("")
    jump = c.str.strip().eq("")
    if surface is not None:
        jump = jump & surface.astype("string").fillna("").eq("障")
    conds = [c.str.contains("新馬").to_numpy(dtype=bool), c.str.contains("未勝利").to_numpy(dtype=bool), jump.to_numpy(dtype=bool)]
    return np.select(conds, np.array(["shinba", "misho", "jump"], dtype=object), default="other")


def is_missing(s: pd.Series) -> np.ndarray:
    """欠け: NaN・None に加え、文字列の列の空（""・"None"・"nan"。推論の出馬表の馬場の空と同じ判定）。"""
    m = s.isna().to_numpy()
    if s.dtype == object or str(s.dtype).startswith(("string", "category")):
        m = m | s.astype("string").fillna("").str.strip().isin(["", "None", "nan", "<NA>"]).to_numpy(dtype=bool)
    return m


def compare(served: pd.DataFrame, train_x_path: str | Path, meta: pd.DataFrame, years: list[int], months: list[int],
            train_strata: np.ndarray | None = None, chunk: int = 100) -> pd.DataFrame:
    """列ごとの欠けの割合（学習・推論、層ごとと重み付きの全体）。meta は学習の X と同じ行の並び・行数。
    片方にしか無い列は、無い側の欠けを 100% とみなして行に出す（side 列で区別）。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(train_x_path))
    if pf.metadata.num_rows != len(meta):
        raise ValueError(f"学習の X と meta の行数が違う: {pf.metadata.num_rows} ≠ {len(meta)}")
    d = pd.to_datetime(meta["date"])
    win = (d.dt.year.isin(years) & d.dt.month.isin(months)).to_numpy()
    names = pf.schema_arrow.names
    if train_strata is not None:
        k_tr = train_strata
    else:
        surf = pf.read(columns=["surface"]).to_pandas()["surface"] if "surface" in names else None
        k_tr = stratum(meta["race_class"], surf)
    k_sv = stratum(served["race_class"], served["surface"] if "surface" in served else None)
    w = pd.Series(k_sv).value_counts(normalize=True).to_dict()
    keys = ("race_id", "horse_id")
    cols = [c for c in names if c in served.columns and c not in keys]
    rows = []
    for i in range(0, len(cols), chunk):
        T = pf.read(columns=cols[i:i + chunk]).to_pandas()
        for c in T.columns:
            v = is_missing(T[c])
            s = is_missing(served[c])
            rec: dict = {"col": c, "group": group_of(c), "side": "both", "na_train": 0.0, "na_served": 0.0}
            for k, wk in w.items():
                mt = win & (k_tr == k)
                tr = float(v[mt].mean()) if mt.any() else np.nan
                sv = float(s[k_sv == k].mean())
                rec[f"na_train_{k}"], rec[f"na_served_{k}"] = tr, sv
                rec["na_train"] += wk * (tr if np.isfinite(tr) else sv)      # 学習に無い層は差に数えない
                rec["na_served"] += wk * sv
            rows.append(rec)
        del T
    for c in [c for c in names if c not in served.columns and c not in keys]:
        rows.append({"col": c, "group": group_of(c), "side": "train_only", "na_train": np.nan, "na_served": 1.0})
    for c in [c for c in served.columns if c not in names and c not in keys]:
        rows.append({"col": c, "group": group_of(c), "side": "served_only", "na_train": 1.0, "na_served": float(is_missing(served[c]).mean())})
    o = pd.DataFrame(rows)
    o["d_pt"] = (o["na_served"] - o["na_train"]) * 100
    return o


def summary(o: pd.DataFrame, thr_pt: float = 5.0) -> dict:
    both = o[o["side"] == "both"] if "side" in o else o
    big = both[both["d_pt"].abs() > thr_pt]
    one = o[o["side"] != "both"] if "side" in o else o.iloc[:0]
    return {"cols": int(len(both)), "over": int(len(big)), "over_share": round(len(big) / max(len(both), 1), 4), "thr_pt": thr_pt,
            "by_group": {g: {"n": int(len(x)), "max_d_pt": round(float(x["d_pt"].abs().max()), 1)} for g, x in big.groupby("group")},
            "train_only": sorted(one.loc[one["side"] == "train_only", "col"]), "served_only": sorted(one.loc[one["side"] == "served_only", "col"])}


def _months(s: str) -> list[int]:
    """「10-12」「10,12」「1-3,10-12」を月の一覧にする。"""
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--served-glob", required=True)
    ap.add_argument("--train-x", required=True)
    ap.add_argument("--train-meta", required=True)
    ap.add_argument("--years", default="2024,2025")
    ap.add_argument("--months", default="1-12")
    ap.add_argument("--thr-pt", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fs = sorted(glob.glob(a.served_glob))
    if not fs:
        raise SystemExit(f"推論の X が無い: {a.served_glob}")
    parts = [pd.read_parquet(f) for f in fs]
    sets = {tuple(p.columns) for p in parts}
    if len(sets) > 1:      # 版の違うファイルが混ざると、無い列が NaN になって欠けに数えられる
        raise SystemExit(f"推論の X の列の組が {len(sets)} 通りある。版ごとに分けて流す")
    served = pd.concat(parts, ignore_index=True)
    meta = pd.read_parquet(a.train_meta, columns=["race_id", "date", "race_class"])
    o = compare(served, a.train_x, meta, [int(y) for y in a.years.split(",")], _months(a.months))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_csv(out, index=False)
    s = {"served_files": len(fs), "served_rows": int(len(served)), **summary(o, a.thr_pt)}
    out.with_suffix(".json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
