"""tp-61: 実測の展開（ペース × 隊列の凝縮度）と脚質別の成績を 1 表にまとめる。

python -m src.scripts.data.tp61_build --years 2020 2021 2022 2023 2024 2025 2026   # ホストで実行（heavy_run 経由）
出力: data/local/modeling/tp61/race.parquet（1 行 = 1 レース）, horse.parquet（1 行 = 1 頭）

- 隊列の凝縮度: 通過順位の記法（レース結果の corner_passing の order_text）から、最終コーナー（4C）と 3C の
  『馬の間の差』を数える。記法は、かっこ内=横並び、並べて書く=概ね 1 馬身以内、'-'=1 馬身以上、'='=大きな差。
  gap 単位は '-' を 1、'=' を 3 とする（隣り合う馬の境ごと）。
- ペース: レース前半 3F / 後半 3F（pace JSON）。同条件（場×面×距離×馬場）の中央値からの偏差の差を balance とする（tp-55 の
  ラップ傾向と同じ向き: 小さい=スロー寄り、大きい=ハイ寄り）。
- 着差: margin（前の馬との着差）を馬身に直し、先頭から 5 着までの幅など。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
OUT = ROOT / "data/local/modeling/tp61"

_INT = re.compile(r"\d+")
GAP_DASH = 1
GAP_EQ = 3

MARGIN = {"クビ": 0.2, "ハナ": 0.1, "アタマ": 0.3, "同着": 0.0, "大": 10.0}


def parse_order(text: str) -> tuple[list[int], list[int]]:
    """order_text → (馬番の並び, 隣り合う馬の境ごとの gap 単位)。記法に合わない文字は無視する。"""
    horses: list[int] = []
    gaps: list[int] = []
    pending = 0
    i, n = 0, len(text or "")
    t = str(text or "")
    while i < n:
        ch = t[i]
        if ch.isdigit():
            j = i
            while j < n and t[j].isdigit():
                j += 1
            if horses:
                gaps.append(pending)
            pending = 0
            horses.append(int(t[i:j]))
            i = j
            continue
        if ch == "=":
            pending = max(pending, GAP_EQ)
        elif ch == "-":
            pending = max(pending, GAP_DASH)
        i += 1
    return horses, gaps


def margin_len(s) -> float:
    s = str(s or "").strip()
    if not s:
        return np.nan
    if s in MARGIN:
        return MARGIN[s]
    m = re.fullmatch(r"(?:(\d+)\.)?(\d+)/(\d+)", s)
    if m:
        return float(m.group(1) or 0) + int(m.group(2)) / int(m.group(3))
    m = re.fullmatch(r"\d+(?:\.\d+)?", s)
    return float(s) if m else np.nan


def shape_metrics(text: str, field: int) -> dict:
    horses, gaps = parse_order(text)
    n = len(horses)
    if n < max(6, int(field * 0.6)):
        return {}
    g = np.array(gaps, dtype=float)
    nz = np.flatnonzero(g > 0)
    first_break = int(nz[0]) + 1 if len(nz) else n
    return {
        "n": n,
        "gap_total": float(g.sum()),
        "gap_per_horse": float(g.sum() / max(n - 1, 1)),
        "gap_top5": float(g[:4].sum()),
        "gap_lead": float(g[0]) if len(g) else 0.0,                 # 先頭と 2 番手の間
        "n_groups": int(len(nz) + 1),
        "lead_group_frac": float(first_break / n),
        "tail_gap": float(g[-3:].sum()) if len(g) >= 3 else float(g.sum()),
        "horses": horses,
        "gaps": gaps,
    }


def _going_group(tc: str) -> str:
    return "良" if str(tc) == "良" else "道悪"


def load_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["race_id", "date", "venue", "surface", "distance", "track_condition", "field_size", "grade", "race_class",
            "horse_number", "horse_id", "finish_position", "passing_order", "last_3f", "pace", "margin", "odds"]
    r = pd.read_parquet(TABLES / str(year) / "race_result_flat.parquet", columns=cols)
    r = r[r["surface"].isin(["芝", "ダート"]) & (r["field_size"].fillna(0) >= 8)].copy()
    r["finish_position"] = pd.to_numeric(r["finish_position"], errors="coerce")
    ps = r["passing_order"].map(lambda s: [int(x) for x in _INT.findall(str(s or ""))])
    fs = r["field_size"].astype(float)
    r["pass_first"] = ps.map(lambda x: x[0] if x else np.nan)
    r["pass_last"] = ps.map(lambda x: x[-1] if x else np.nan)
    r["early_norm"] = (r["pass_first"] - 1) / (fs - 1)
    r["late_norm"] = (r["pass_last"] - 1) / (fs - 1)
    r["l3f"] = pd.to_numeric(r["last_3f"], errors="coerce")
    r.loc[(r["l3f"] < 30) | (r["l3f"] > 50), "l3f"] = np.nan
    r["margin_len"] = r["margin"].map(margin_len)
    lap = pd.read_parquet(TABLES / str(year) / "race_result_lap_flat.parquet", columns=["race_id", "corner_passing"]).drop_duplicates("race_id")
    rows = []
    for rid, cp in zip(lap["race_id"], lap["corner_passing"]):
        try:
            cl = json.loads(cp) if cp else []
        except Exception:
            cl = []
        cs = sorted([c for c in cl if int(c.get("corner") or 0) > 0], key=lambda c: int(c["corner"]))
        rec = {"race_id": rid}
        rec["_cs"] = [(int(c["corner"]), str(c.get("order_text") or "")) for c in cs]
        rows.append(rec)
    return r, pd.DataFrame(rows)


def build_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    r, cs = load_year(year)
    fsz = r.drop_duplicates("race_id").set_index("race_id")["field_size"].astype(int)
    cs = cs[cs["race_id"].isin(fsz.index)].copy()
    shp = {}
    for rid, lst in zip(cs["race_id"], cs["_cs"]):
        f = int(fsz[rid])
        rec = {}
        if lst:
            last = shape_metrics(lst[-1][1], f)
            rec.update({f"c4_{k}": v for k, v in last.items() if k not in ("horses", "gaps")})
            rec["_c4_horses"] = last.get("horses")
            rec["_c4_gaps"] = last.get("gaps")
            if len(lst) >= 2:
                prev = shape_metrics(lst[-2][1], f)
                rec.update({f"c3_{k}": v for k, v in prev.items() if k not in ("horses", "gaps")})
        shp[rid] = rec
    sh = pd.DataFrame.from_dict(shp, orient="index").rename_axis("race_id").reset_index()

    base = r.drop_duplicates("race_id")[["race_id", "date", "venue", "surface", "distance", "track_condition", "field_size", "grade", "race_class", "pace"]].copy()

    def _pace(s):
        try:
            d = json.loads(s) if isinstance(s, str) else (s or {})
        except Exception:
            d = {}
        return d.get("t3f") or d.get("first_half_3f"), d.get("l3f") or d.get("second_half_3f")

    base[["t3f", "race_l3f"]] = base["pace"].map(_pace).apply(pd.Series)
    base = base.drop(columns=["pace"])
    base["going_group"] = base["track_condition"].map(_going_group)
    base["year"] = year
    race = base.merge(sh, on="race_id", how="left")

    # 着差の幅（先頭から 5 着まで・10 着まで）
    fin = r.sort_values(["race_id", "finish_position"])
    def _spread(g):
        m = g["margin_len"].to_numpy(dtype=float)
        mm = m[1:]
        return pd.Series({"fin_spread5": float(np.nansum(mm[:4])) if len(mm) >= 4 else np.nan,
                          "fin_spread10": float(np.nansum(mm[:9])) if len(mm) >= 9 else np.nan})
    sp = fin.groupby("race_id").apply(_spread).reset_index()
    race = race.merge(sp, on="race_id", how="left")

    med = r.groupby("race_id")["l3f"].median().rename("l3f_med")
    race = race.merge(med, on="race_id", how="left")
    r = r.merge(med, on="race_id", how="left")
    r["l3f_dev"] = r["l3f"] - r["l3f_med"]
    r["l3f_rank"] = r.groupby("race_id")["l3f"].rank(method="min")
    r["l3f_rank_pct"] = (r["l3f_rank"] - 1) / (r["field_size"].astype(float) - 1)
    r["year"] = year
    r["is_win"] = (r["finish_position"] == 1).astype(int)
    r["is_top3"] = (r["finish_position"] <= 3).astype(int)
    r["is_top2"] = (r["finish_position"] <= 2).astype(int)
    r["fin_pct"] = (r["finish_position"] - 1) / (r["field_size"].astype(float) - 1)
    r["pos_gain"] = r["early_norm"] - r["late_norm"]            # 正=前半より後半に順位を上げた
    horse = r.drop(columns=["passing_order", "last_3f", "pace", "margin"])
    race = race.drop(columns=[c for c in race.columns if c.startswith("_")])
    return horse, race


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hs, rs = [], []
    for y in a.years:
        h, rc = build_year(y)
        print(y, len(h), len(rc), "c4 shape coverage", float(rc["c4_gap_total"].notna().mean()), flush=True)
        hs.append(h)
        rs.append(rc)
    pd.concat(hs, ignore_index=True).to_parquet(out / "horse.parquet", index=False)
    pd.concat(rs, ignore_index=True).to_parquet(out / "race.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
