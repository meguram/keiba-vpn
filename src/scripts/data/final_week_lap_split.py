"""A3/B3（片方が11秒台以下）を、もう片方のラップや差で細分化すべきかを、サンプル数・率・回収率・再現性で見る（坂路、2020〜2025、単勝50倍未満）。

python -m src.scripts.data.final_week_lap_split [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import final_week_workout as W
from src.pipeline.features import tp35_lap_groups as L
from src.scripts.data import final_week_pattern_eval as P
from src.scripts.data import final_week_roi_eval as RO
from src.scripts.data import trainer_pattern_table as T

OUT_DEFAULT = "/home/jovyan/work/retrain_out/base_2016/article_check/final_week/lap_split"
rng = np.random.default_rng(0)


def load_ext():
    t = pd.read_parquet(P.TABLE, columns=["race_id", "date", "surface", "distance", "field_size", "horse_number", "horse_id", "trainer_id", "trainer_name", "finish_position"])
    t = t[t["surface"].astype(str).isin(["芝", "ダート"]) & (t["finish_position"] > 0) & (t["date"].astype(str) < "2026-01-01")].copy()
    t["race_id"], t["horse_id"], t["trainer_id"] = t["race_id"].astype(str), t["horse_id"].astype(str), t["trainer_id"].astype(str)
    t["trainer_name"] = t["trainer_name"].astype(str)
    t["race_date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    t["year"] = t["race_date"].str[:4].astype(int)
    raw = pd.read_parquet(P.LAPS)
    raw = raw[(raw["race_date"] >= "2016-01-01") & (raw["race_date"] < "2026-01-01")].copy()
    raw["horse_id"] = raw["horse_id"].astype(str)
    bw = W.pick_best_weekday(L.prepare_laps(raw))
    bw["horse_id"] = bw["horse_id"].astype(str)
    d = t.merge(bw[["horse_id", "race_date", "BW_place", "BW_pat", "BW_l1", "BW_l2", "BW_kind"]], on=["horse_id", "race_date"], how="left")
    assert len(d) == len(t)
    d["pat"] = d["BW_place"] + "|" + d["BW_pat"]
    return d.reset_index(drop=True)


def band(v):
    return np.where(v < 12.0, "11台以下", np.where(v < 13.0, "12台", "13以上"))


def gapb(g):  # 手前と終いの差（秒）
    return pd.cut(g, [-0.01, 0.0, 0.5, 1.0, 99], labels=["差0", "差0.1-0.5", "差0.6-1.0", "差1.1以上"])


def boot(g, col, B=1500):
    r = g.groupby("race_id").agg(s=(col, "sum"), n=("c1", "size"))
    s, n = r["s"].to_numpy(), r["n"].to_numpy()
    idx = rng.integers(0, len(r), (B, len(r)))
    v = s[idx].sum(1) / n[idx].sum(1)
    return np.percentile(v, [2.5, 97.5])


def row(g, parent):
    n = len(g)
    p1, p3 = parent["c1"].mean(), parent["c3"].mean()
    z3 = (g["c3"].mean() - p3) / np.sqrt(p3 * (1 - p3) / n) if n else np.nan
    z1 = (g["c1"].mean() - p1) / np.sqrt(p1 * (1 - p1) / n) if n else np.nan
    lo, hi = boot(g, "place_ret") if n >= 30 else (np.nan, np.nan)
    wlo, whi = boot(g, "win_ret") if n >= 30 else (np.nan, np.nan)
    h = {k: grp for k, grp in g.groupby("half")}
    lift = {k: (grp["c3"].mean() / parent[parent["half"] == k]["c3"].mean() if len(grp) >= 30 else np.nan) for k, grp in h.items()}
    return {"走": n, "頭": g["horse_id"].nunique(), "調教師": g["trainer_id"].nunique(), "勝率": g["c1"].mean(), "z勝": z1, "複勝率": g["c3"].mean(), "z複": z3,
            "単勝ROI": g["win_ret"].mean(), "単lo": wlo, "単hi": whi, "複勝ROI": g["place_ret"].mean(), "複lo": lo, "複hi": hi,
            "中央オッズ": g["odds"].median(), "前半複勝率比": lift.get("前半2020-22", np.nan), "後半複勝率比": lift.get("後半2023-25", np.nan)}


def table(parent, keys, title):
    rows = []
    for k, g in parent.groupby(keys, observed=True):
        rows.append({"区分": " / ".join(map(str, k)) if isinstance(k, tuple) else str(k), **row(g, parent)})
    r = pd.DataFrame(rows)
    r.loc[len(r)] = {"区分": "（親・全体）", **row(parent, parent)}
    print(f"\n## {title}\n")
    print(r.round(3).to_string(index=False))
    return r



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    OUT = Path(ap.parse_args().out)
    OUT.mkdir(parents=True, exist_ok=True)
    d = T.add_odds(RO.add_returns(load_ext()))
    x = d[d["year"].between(2020, 2025) & d["pat"].notna() & d["BW_place"].astype(str).str.contains("坂路") & d["win_ret"].notna() & d["place_ret"].notna()].copy()
    x["cls"] = x["BW_pat"]
    x["l1"], x["l2"] = x["BW_l1"].astype(float), x["BW_l2"].astype(float)
    print(f"坂路・パターン付き・払戻あり: {len(x):,} 走（2020〜2025）。うち単勝50倍以上 {(x['odds'] >= 50).mean():.1%}")
    x = x[~(x["odds"] >= 50)].copy()
    x["c1"], x["c3"] = (x["finish_position"] == 1).astype(int), (x["finish_position"] <= 3).astype(int)
    x["half"] = np.where(x["year"] <= 2022, "前半2020-22", "後半2023-25")


    allr = []
    print("\n# 0. 七種別の基準（坂路・50倍未満）")
    base = pd.DataFrame([{"区分": c, **row(g, x)} for c, g in x.groupby("cls")]).round(3)
    print(base.to_string(index=False))
    base.to_csv(OUT / "base.csv", index=False, encoding="utf-8-sig")

    for cls, other, mine in (("A3", "l2", "l1"), ("B3", "l1", "l2")):
        p = x[x["cls"] == cls].copy()
        p["相手帯"] = band(p[other])
        p["差"] = gapb((p["l2"] - p["l1"]).abs())
        p["速い側"] = pd.cut(p[mine], [0, 11.0, 11.5, 12.0], right=False, labels=["〜10.9", "11.0-11.4", "11.5-11.9"])
        print(f"\n# {cls}（n={len(p):,}）")
        for keys, title in ((["相手帯"], f"{cls}: {'手前' if other=='l2' else '終い'}（相手側）が何秒台か"), (["差"], f"{cls}: 2つの差"), (["速い側"], f"{cls}: 速い側の時計"), (["速い側", "相手帯"], f"{cls}: 速い側 × 相手側")):
            r = table(p, keys, title)
            r.insert(0, "種別", cls)
            r.insert(1, "切り方", "×".join(keys))
            allr.append(r)
        for place, g in p.groupby(p["BW_place"].astype(str)):
            table(g, ["相手帯"], f"{cls}: {place} 内で相手側の帯")
    pd.concat(allr).to_csv(OUT / "a3_b3_split.csv", index=False, encoding="utf-8-sig")

    print("\n# 調教師×場所×種別セルのサンプル数（30走以上のセルの数と、それが覆う走数）")
    x["place"] = x["BW_place"].astype(str)
    for cls in ("A3", "B3"):
        p = x[x["cls"] == cls].copy()
        p["相手帯"] = band(p["l2" if cls == "A3" else "l1"])
        for lab, keys in (("分けない", ["trainer_id", "place"]), ("相手側の帯で3分割", ["trainer_id", "place", "相手帯"])):
            c = p.groupby(keys).size()
            print(f"{cls} {lab}: セル {len(c):,}、30走以上 {(c >= 30).sum():,}（走数の {c[c >= 30].sum() / c.sum():.0%}）、10走未満 {(c < 10).sum():,}（{c[c < 10].sum() / c.sum():.0%}）")
    print("\n完了")


if __name__ == "__main__":
    main()
