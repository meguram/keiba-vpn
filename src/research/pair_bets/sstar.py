"""組の s*（T-60 → 確定のオッズの縮み）。発走前の組のオッズと確定の組のオッズだけを読む（着順・払戻は結ばない）。

    KEIBA_SESSION=pair-bets nice -n 19 python -m src.research.pair_bets.sstar

- 元: data/page_reference/tables/2026/race_pair_odds_race.parquet の phase=final の行（トップが確定、snapshots に t60・t15）。手元の写しだけ（GCS は読まない）。
- 1 号の条件 8（decisions.md「R0-12 の追記: pair-bets の段 2」）: s* は門 0 の模擬の払戻の縮みにだけ使い、行動を選ぶことには使わない。
- 組ごとに s = 確定のオッズ ÷ T-60 のオッズ（ワイドは下限と中点の 2 通り）。T-60 の組の人気の帯ごとに、中央値・平均・幾何平均を出す。
  単勝の s*（0.86）は「選んだ馬の払戻の縮み」で、ここは選ぶ前の全部の組の縮み。選んだ組ほど縮む分（R0-14）は入らないので、楽観の下限として読む。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("PAIR_RACE_2026") or "/home/jovyan/work/keiba-vpn/data/page_reference/tables/2026/race_pair_odds_race.parquet")
OUT = ROOT / "docs/operations/pair_bets/sstar.json"
BANDS = [(1, 3), (4, 10), (11, 30), (31, 80), (81, 9999)]
FROM = "2026-10-24"


def _load(v):
    return json.loads(v) if isinstance(v, str) else v


def frame(items, ty: str) -> pd.DataFrame:
    rows = []
    for x in items or []:
        a, b = x["pair"]
        if ty == "wide":
            lo, hi = x.get("odds_min"), x.get("odds_max")
            if lo and hi:
                rows.append((int(a), int(b), float(lo), (float(lo) + float(hi)) / 2, x.get("popularity")))
        elif x.get("odds"):
            rows.append((int(a), int(b), float(x["odds"]), float(x["odds"]), x.get("popularity")))
    return pd.DataFrame(rows, columns=["a", "b", "o", "o_mid", "pop"])


def main() -> None:
    from src.ops import data_ledger as DL
    tok = DL.guard(tool="src/research/pair_bets/sstar.py", family="rules_council", version="pair_bets_sstar", period=(FROM, "2026-11-27"),
                   purpose="eval_only", reads=("組のオッズ（T-60・確定）",),
                   reason="1 号の条件 8: 組の s* を門 0 の模擬の払戻の縮みにだけ使う。着順・払戻は結ばない（統括役の整理: 結果を見たことにならない）")
    d = pd.read_parquet(SRC, columns=["race_id", "phase", "snapshot_at", "snapshots", "umaren", "wide"])
    d = d[(d["phase"] == "final") & (d["snapshot_at"].astype(str) >= FROM)]
    res = {"from": FROM, "races_final": int(len(d)), "bands": [f"{lo}〜{hi}" if hi < 9999 else f"{lo}〜" for lo, hi in BANDS]}
    for ty in ("umaren", "wide"):
        parts = []
        for _, r in d.iterrows():
            snaps = {s.get("phase"): s for s in (_load(r["snapshots"]) or [])}
            if "t60" not in snaps:
                continue
            F = frame(_load(r[ty]), ty)
            T = frame(snaps["t60"].get(ty), ty)
            if F.empty or T.empty:
                continue
            m = T.merge(F, on=["a", "b"], suffixes=("_t60", "_fin"))
            m["race_id"] = r["race_id"]
            parts.append(m)
        M = pd.concat(parts, ignore_index=True)
        M = M[(M["o_t60"] > 0) & (M["o_fin"] > 0)]
        out = {"races": int(M["race_id"].nunique()), "pairs": int(len(M))}
        for col in ("o", "o_mid") if ty == "wide" else ("o",):
            s = M[f"{col}_fin"] / M[f"{col}_t60"]
            pop = pd.to_numeric(M["pop_t60"], errors="coerce")
            tab = {}
            for lo, hi in BANDS:
                k = (pop >= lo) & (pop <= hi)
                v = s[k]
                tab[f"{lo}〜{hi}" if hi < 9999 else f"{lo}〜"] = {"n": int(k.sum()), "median": round(float(v.median()), 4),
                                                                  "mean": round(float(v.mean()), 4), "geo": round(float(np.exp(np.log(v).mean())), 4),
                                                                  "p10": round(float(v.quantile(0.1)), 4), "p90": round(float(v.quantile(0.9)), 4)}
            tab["全体"] = {"n": int(len(s)), "median": round(float(s.median()), 4), "mean": round(float(s.mean()), 4),
                           "geo": round(float(np.exp(np.log(s).mean())), 4)}
            out[col] = tab
        res[ty] = out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    tok.finish(result_path=OUT)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
