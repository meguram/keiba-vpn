"""v6 の 2026 年の抜き取り（readiness_v6_sample）を、同じレースの v5 の確率と、確定結果で比べる（勝率・連対・複勝の LogLoss とレース単位ブートストラップ）。

python -m src.scripts.data.v6_sample_compare --v5 <readiness_out/v5sample/v5> --v6 <readiness_out/v6sample/v6>
2026 年はテストセット②。この比較は最後に 1 回だけ（選択には使わない）。レース数が少ない（抜き取り）ので区間は広い。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("data/page_reference/tables")


def load(d: Path) -> pd.DataFrame:
    fs = sorted(d.glob("*/*.parquet"))
    x = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    x["race_id"], x["horse_id"] = x["race_id"].astype(str), x["horse_id"].astype(str)
    return x


def ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    v5, v6 = load(Path(a.v5)), load(Path(a.v6))
    res = pd.read_parquet(TABLES / "2026/race_result_flat.parquet", columns=["race_id", "horse_id", "finish_position"])
    res["race_id"], res["horse_id"] = res["race_id"].astype(str), res["horse_id"].astype(str)
    cols = ["p_win", "p_top2", "p_top3"]
    m = v5[["race_id", "horse_id"] + cols].merge(v6[["race_id", "horse_id"] + cols], on=["race_id", "horse_id"], suffixes=("_5", "_6")).merge(res, on=["race_id", "horse_id"])
    m["fp"] = pd.to_numeric(m["finish_position"], errors="coerce")
    m = m[m["fp"] > 0]
    rng = np.random.default_rng(0)
    out = {"rows": int(len(m)), "races": int(m["race_id"].nunique())}
    ids = m["race_id"].to_numpy()
    uniq, inv = np.unique(ids, return_inverse=True)
    for name, k in (("win", 1), ("top2", 2), ("top3", 3)):
        y = (m["fp"].to_numpy() <= k).astype(float)
        d = ll(y, m[f"p_{name}_6"].to_numpy()) - ll(y, m[f"p_{name}_5"].to_numpy())
        per = np.bincount(inv, weights=d, minlength=len(uniq)), np.bincount(inv, minlength=len(uniq))
        boot = []
        for _ in range(2000):
            s = rng.integers(0, len(uniq), len(uniq))
            boot.append(per[0][s].sum() / per[1][s].sum())
        out[name] = {"v5": float(ll(y, m[f"p_{name}_5"].to_numpy()).mean()), "v6": float(ll(y, m[f"p_{name}_6"].to_numpy()).mean()),
                     "delta": float(d.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
