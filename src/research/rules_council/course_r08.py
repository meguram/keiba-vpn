"""R0-08 コース別の段 1 の準備（着順を使わない。1 号の決定「コース別の段 1 の R3」）。

- units   : 単位の表（場 × 芝ダ × 距離 × 内外。京都は改修の前〔〜2020〕と後〔2023〜〕を別の単位）。
            30% の側のレースが 150 に満たない単位は、同じ場・同じ芝ダ・同じ内外で、合わせた群の端から端までが 200m 以内になる単位と、近い順に合わせる（鎖は許さない）。
            それでも 150 に届かない単位は「件数不足」。出力 config/rules_council/course_units_r08.json（中身の sha を書く）。
- balance : 釣り合いの表（着順なし）。単位ごとに、70%・30% のレース数、年の割合、クラスの割合、頭数の平均、再現 T-60 < 10 倍の割合。
            出力 docs/operations/retrain_trials/rules_council_r08_balance.json。
読むのは full.parquet の race_id・開催・年・クラス・頭数と、v3 frame の course（コースの名前）だけ。着順・払戻・オッズの結果は読まない
（再現 T-60 の値は判定の入力として読む）。費用 0 円。土日の 07:30〜17:00 は止める。
使い方: python -m src.research.rules_council.course_r08 units|balance（nice -n 19）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E
from src.research.rules_council import rc_frame as F

REPO = F.CR.REPO
UNITS = REPO / "config/rules_council/course_units_r08.json"
BAL = REPO / "docs/operations/retrain_trials/rules_council_r08_balance.json"
MIN_RACES_30 = 150
MERGE_DIST = 200


def races() -> pd.DataFrame:
    d = pd.read_parquet(F.OUTW / "full.parquet", columns=["race_id", "kaisai", "year", "venue", "surf", "dist", "cls", "field"])
    d["race_id"] = d["race_id"].astype(str)
    d["pre0"] = np.load(F.OUTW / "full_pre.npy")[0]
    d["fav"] = d["pre0"] < 10
    r = d.groupby("race_id").agg(kaisai=("kaisai", "first"), year=("year", "first"), venue=("venue", "first"), surf=("surf", "first"),
                                 dist=("dist", "first"), cls=("cls", "first"), field=("field", "first"), fav=("fav", "mean")).reset_index()
    fr = pd.read_parquet(F.CR.OUT / "v3" / "frame.parquet", columns=["race_id", "course"], filters=[("year", "<=", F.YEAR_END)]).drop_duplicates("race_id")
    fr["race_id"] = fr["race_id"].astype(str)
    r = r.merge(fr, on="race_id", how="left")
    r["io"] = r["course"].astype(str).map(lambda s: "内外不明" if s.endswith("内外不明") else (m.group(1) if (m := re.search(r"\d+(外|内|直)$", s)) else ""))
    dt = pd.to_datetime(r["race_id"].str[:8], format="%Y%m%d", errors="coerce")
    kyoto = r["venue"].astype(str) == "京都"
    r["era"] = np.where(kyoto, np.where(dt < pd.Timestamp("2020-11-01"), "改修前", np.where(dt >= pd.Timestamp("2023-04-01"), "改修後", "改修中")), "")
    r["s70"] = E.search70(r["kaisai"].to_numpy())
    r["base"] = r["venue"].astype(str) + r["surf"].astype(str) + r["io"] + r["era"]
    r["unit0"] = r["base"] + r["dist"].astype(int).astype(str)
    return r


def cmd_units() -> dict:
    E.check_stop()
    r = races()
    g = r.groupby(["base", "venue", "surf", "io", "era", "dist"]).agg(races=("race_id", "size"), s30=("s70", lambda s: int((~s).sum()))).reset_index()
    units = []
    for b, gg in g.groupby("base"):
        rows = gg.sort_values("dist").to_dict("records")
        groups = [[x] for x in rows]
        changed = True
        while changed:                                   # 150 に満たない群を、距離の差 200m 以内の近い隣と合わせる（小さい群から）
            changed = False
            small = sorted([i for i, x in enumerate(groups) if sum(y["s30"] for y in x) < MIN_RACES_30], key=lambda i: sum(y["s30"] for y in groups[i]))
            for i in small:
                cand = []
                for j in (i - 1, i + 1):
                    if 0 <= j < len(groups):
                        both = groups[i] + groups[j]                 # 鎖は許さない: 合わせた群の端から端までが 200m 以内（1 号の判断）
                        span = max(y["dist"] for y in both) - min(y["dist"] for y in both)
                        if span <= MERGE_DIST:
                            cand.append((span, j))
                if cand:
                    _, j = min(cand)
                    a, c = sorted((i, j))
                    groups[a] = groups[a] + groups[c]
                    del groups[c]
                    changed = True
                    break
        for x in groups:
            s30 = int(sum(y["s30"] for y in x))
            unclear = "内外不明" in b or "改修中" in b                       # 内外の分からない・改修中の単位は 30% の検定に入れない
            units.append({"unit": b + "/".join(str(int(y["dist"])) for y in x), "base": b, "dists": [int(y["dist"]) for y in x],
                          "races": int(sum(y["races"] for y in x)), "races_30": s30, "merged": len(x) > 1,
                          "short": s30 < MIN_RACES_30 or unclear, "unclear": unclear})
    info = json.loads((F.OUTW / "full_info.json").read_text(encoding="utf-8"))
    body = {"inputs": {"full_info": {k: info.get(k) for k in ("rows", "races", "kaisai", "pre_sha256", "fold_sha256") if k in info},
                       "split70_seed": E.SPLIT70_SEED, "split70_share": E.SPLIT70_SHARE},
            "rule": f"場 × 芝ダ × 距離 × 内外（京都は改修の前後を別）。30% の側が {MIN_RACES_30} レース未満なら、同じ場・芝ダ・内外で、合わせた群の端から端までが {MERGE_DIST}m 以内になる相手と近い順に合わせる（鎖は許さない）。届かなければ件数不足。内外不明・京都の改修中は 30% の検定に入れない",
            "units": units}
    sha = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    out = body | {"sha256": sha, "n_units": len(units), "n_ok": int(sum(not u["short"] for u in units)), "n_single": int(sum(not u["merged"] for u in units))}
    UNITS.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_units", "n_ok", "n_single", "sha256")}, ensure_ascii=False))
    return out


def cmd_balance() -> dict:
    E.check_stop()
    U = json.loads(UNITS.read_text(encoding="utf-8"))
    r = races()
    key = {}
    for u in U["units"]:
        for dd in u["dists"]:
            key[(u["base"], dd)] = u["unit"]
    r["unit"] = [key.get((b, int(x))) for b, x in zip(r["base"], r["dist"])]
    miss = int(r["unit"].isna().sum())
    assert miss == 0, f"単位の表に無いレースが {miss} ある（データが変わった。units を作り直す）"
    out = {}
    for u, g in r.groupby("unit"):
        o = {}
        for side, m in (("70", g["s70"]), ("30", ~g["s70"])):
            x = g[m.to_numpy()]
            o[side] = {"races": int(len(x)), "years": {int(k): int(v) for k, v in x["year"].value_counts().sort_index().items()},
                       "cls_share": {int(k): round(float(v), 3) for k, v in x["cls"].value_counts(normalize=True).sort_index().items()},
                       "field_mean": round(float(x["field"].mean()), 2) if len(x) else None, "fav_share": round(float(x["fav"].mean()), 3) if len(x) else None}
        out[u] = o
    BAL.write_text(json.dumps({"note": "着順なしの釣り合いの表（R0-08）。単位は course_units_r08.json", "units_sha256": U["sha256"], "units": out},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print("balance", len(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["units", "balance"])
    a = ap.parse_args()
    {"units": cmd_units, "balance": cmd_balance}[a.cmd]()


if __name__ == "__main__":
    main()
