"""まとめの材料: 父ごとの適性の一覧（2024 年末までの as-of の縮小した率の最後の値。記述だけ・買い方に使わない）。

父 × 芝ダ（3 着内率の芝 − ダート）、父 × 芝ダ × 距離帯の得意な帯、父 × レースの年齢条件（2 歳戦 − 3 歳以上戦の 3 着内率）。
産駒が 2,000 走以上の父だけ。名前は 5 代血統の表（ancestors の name）から引く。
    nice -n 19 python -m src.research.pedigree_lab.sire_profile
出力: OUT/sire_profile.json。費用 0 円。
"""
from __future__ import annotations

import json

import pandas as pd

from src.research.pedigree_lab import build_base, data


def sire_names(ids: set[str]) -> dict[str, str]:
    out = {}
    import json as _j
    for base in build_base.PED_DIRS:
        for f in base.glob("*/*.json"):
            if len(out) >= len(ids):
                break
            try:
                d = _j.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            for a in d.get("ancestors") or []:
                i = str(a.get("horse_id") or "")
                if i in ids and i not in out and a.get("generation") == 1:
                    out[i] = str(a.get("name") or "")
        if len(out) >= len(ids):
            break
    return out


def main() -> None:
    X = pd.read_parquet(data.OUT / "feat_v4.parquet",
                        columns=["date", "sire", "surf", "dband", "rage", "sire_surf_top3", "sire_sd_top3", "sire_rasurf_top3", "sire_all_n"])
    big = X.groupby("sire").size()
    big = big[big >= 2000].index
    X = X[X["sire"].isin(big)].sort_values("date")
    last = lambda cols: X.groupby(["sire", *cols]).tail(1).set_index(["sire", *cols])
    s = last(["surf"])["sire_surf_top3"].unstack("surf")
    sd = last(["surf", "dband"])["sire_sd_top3"].unstack(["surf", "dband"])
    ra = last(["rage", "surf"])["sire_rasurf_top3"].unstack(["rage", "surf"])
    names = sire_names(set(big))
    rows = []
    for sid in big:
        r = {"sire": names.get(sid, sid), "id": sid, "runs": int((X["sire"] == sid).sum())}
        if sid in s.index:
            r["turf"], r["dirt"] = round(float(s.loc[sid].get("T", float("nan"))), 3), round(float(s.loc[sid].get("D", float("nan"))), 3)
            r["turf_minus_dirt"] = round(r["turf"] - r["dirt"], 3)
        if sid in sd.index:
            row = sd.loc[sid].dropna()
            if len(row):
                (sf, band) = row.idxmax()
                r["best_band"] = f"{'芝' if sf == 'T' else 'ダ'}{ {'S': '〜1300', 'M': '1400-1600', 'I': '1700-2000', 'L': '2100-2400', 'E': '2500〜'}[band]}"
        if sid in ra.index:
            a2 = ra.loc[sid].get(("A2", "T")), ra.loc[sid].get(("A3U", "T"))
            if all(pd.notna(v) for v in a2):
                r["early_turf"] = round(float(a2[0] - a2[1]), 3)
        rows.append(r)
    D = pd.DataFrame(rows)
    out = {"n_sires": len(D),
           "turf_lean_top": D.sort_values("turf_minus_dirt", ascending=False).head(10).to_dict("records"),
           "dirt_lean_top": D.sort_values("turf_minus_dirt").head(10).to_dict("records"),
           "early_top": D.dropna(subset=["early_turf"]).sort_values("early_turf", ascending=False).head(10).to_dict("records"),
           "late_top": D.dropna(subset=["early_turf"]).sort_values("early_turf").head(10).to_dict("records")}
    (data.OUT / "sire_profile.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
