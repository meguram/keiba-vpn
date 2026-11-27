"""市場と相関の高い列（担当 model-relabel、11/27 夜のユーザーの指定）: 各列と市場の logit（確定の単勝の確率 q_final）の順位相関。

- 学習の年（2016〜21）の行だけで測る。評価の年（2022〜24）の値で列を選ばない。
- 列は mr-6 の v5bag_mf（v5bag − 追い切り評価の 3 列、1,080 列。course_type を含む）。mr-8 で v5bag_mf から course_type を外したが、
  表は作り直さない（凍結。作り直しを止める確かめを main に入れた）。mfc の列の選びは、この表の |ρ| で決める。相関は Spearman の |ρ|（全行をまとめて。q の欠けた行は落とす）。
- 閾値は数を見る前に固定（spec の market_corr: |ρ| ≥ 0.5 と ≥ 0.3 の 2 段）。外す列の一覧を docs に残し、longshot-ai と同じ表を使う。
- 着順は使わない（load_matrix の meta に着順の列はあるが、相関には市場の q と特徴量だけを使う）。値が一定の列（例 fr_name_jump）は ρ=NaN で、どちらの版でも外れない。費用 0 円（手元の表だけ）。
  KEIBA_SESSION=model-relabel flock /tmp/claude-1000/heavy.lock nice -n 19 python -m src.research.model_relabel.mr_market corr
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.model_relabel import mr_train as TR

ROOT = Path(__file__).resolve().parents[3]
TABLE = ROOT / "docs/operations/model_relabel/market_corr_2016_2021.csv"
RUNNERS = Path("/home/jovyan/work/megu_cycles_out/runners.parquet")


def corr_table() -> pd.DataFrame:
    X, meta, names = TR.load_matrix("v5bag_mf")
    yr = pd.to_numeric(meta["year"]).to_numpy()
    keep = yr <= 2021
    q = pd.read_parquet(RUNNERS, columns=["race_id", "horse_id", "year", "q_final"], filters=[("year", "<=", 2021)])
    q["race_id"], q["horse_id"] = q["race_id"].astype(str), q["horse_id"].astype(str)
    k = meta[["race_id", "horse_id"]].astype(str).merge(q.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    assert len(k) == len(X)
    qq = k["q_final"].to_numpy(float)
    ok = keep & np.isfinite(qq) & (qq > 0) & (qq < 1)
    lq = np.log(qq[ok] / (1 - qq[ok]))
    rq = pd.Series(lq).rank().to_numpy()
    rows = []
    Xk = X[ok]
    for c in names:
        v = Xk[c].to_numpy(float)
        m = np.isfinite(v)
        if m.sum() < 1000 or np.nanstd(v) == 0:
            rows.append((c, np.nan, int(m.sum())))
            continue
        rv = pd.Series(v[m]).rank().to_numpy()
        rho = np.corrcoef(rv, pd.Series(rq[m]).rank().to_numpy())[0, 1]
        rows.append((c, float(rho), int(m.sum())))
    t = pd.DataFrame(rows, columns=["column", "spearman_rho", "n"])
    t["abs_rho"] = t["spearman_rho"].abs()
    return t.sort_values("abs_rho", ascending=False).reset_index(drop=True)


def main() -> None:
    from src.ops import data_ledger as DL
    sp = TR.spec()["market_corr"]
    tok = DL.guard(tool="src/research/model_relabel/mr_market.py:main", family="model_relabel", version="market-corr-2016-2021",
                   period=("2016-01-01", "2021-12-31"), purpose="select", reads=("オッズ",), frozen_paths=[TR.SPEC],
                   reason="列と確定の市場の logit の順位相関（着順は読まない）。閾値は spec で先に固定")
    if TABLE.exists():
        raise SystemExit(f"{TABLE} は凍結（mr-6）。作り直さない")
    t = corr_table()
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLE, index=False, float_format="%.5f")
    counts = {str(th): int((t["abs_rho"] >= th).sum()) for th in sp["thresholds"]}
    (TABLE.with_suffix(".json")).write_text(json.dumps({"n_columns": len(t), "counts": counts, "top20": t.head(20)[["column", "spearman_rho"]].values.tolist()},
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    tok.finish(result_path=TABLE)
    print("列", len(t), "閾値ごとの列の数", counts, flush=True)
    print(t.head(25).to_string(), flush=True)


if __name__ == "__main__":
    main()
