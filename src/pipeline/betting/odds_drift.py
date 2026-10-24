"""購入時点のオッズと確定オッズのずれ（締切直前の資金流入 = late money）を測る監視と、判断オッズの補正。

背景: 2020〜2025 年のオッズは確定値だけで、T-60 / T-15 のスナップショットは 2026-10-17 以降に貯まる（``odds_snapshot``）。
確定オッズで買い目を選んで確定オッズで精算する検証は、締切直前の動き（AI・大口の集中購入で人気馬が売れ、穴馬のオッズが下がる等）を無視する。
ここでは、race_odds の文書（トップ = 最新、``snapshots`` = それ以前の phase）から「T-15 → 確定」「T-60 → 確定」の変化率を、オッズ帯・月ごとに集計し、
- 帯ごとの平均 log 変化率（= 判断オッズに掛ける補正）、
- 直近と過去の変化率の差の検定（ドリフト警報）
を出す。

  python -m src.pipeline.betting.odds_drift --from 20261017 --to 20261231 --out data/local/meta/odds_drift/latest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BANDS = [1.0, 3.0, 6.0, 10.0, 20.0, 50.0, 1e9]
BAND_LABELS = ["<3", "3-6", "6-10", "10-20", "20-50", "50+"]


def flatten_docs(docs: Iterable[dict]) -> pd.DataFrame:
    """race_odds 文書の列 → 行（race_id, phase, snapshot_at, horse_number, win_odds, place_odds_min, popularity）。トップ（最新）と snapshots の両方を展開する。"""
    rows = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        rid = str(d.get("race_id") or "")
        parts = [(d.get("phase") or "latest", d.get("snapshot_at"), d.get("entries") or [])]
        for s in d.get("snapshots") or []:
            if isinstance(s, dict) and s.get("phase"):
                parts.append((s["phase"], s.get("snapshot_at"), s.get("entries") or []))
        for phase, at, entries in parts:
            for e in entries:
                try:
                    rows.append({"race_id": rid, "phase": str(phase), "snapshot_at": at, "horse_number": int(e["horse_number"]), "win_odds": float(e.get("win_odds")),
                                 "place_odds_min": float(e["place_odds_min"]) if e.get("place_odds_min") is not None else np.nan, "popularity": e.get("popularity")})
                except (KeyError, TypeError, ValueError):
                    continue
    df = pd.DataFrame(rows, columns=["race_id", "phase", "snapshot_at", "horse_number", "win_odds", "place_odds_min", "popularity"])
    return df[df["win_odds"] > 0].reset_index(drop=True)       # 取消・除外は -3.0


def pair_phases(flat: pd.DataFrame, early: str = "t15", final: str = "latest") -> pd.DataFrame:
    """同じレース・馬で early と final の単勝オッズが両方ある行。final は最新（トップ）で、phase 名が t15/t60 のときは最新ではなく別の phase。"""
    f = flat.copy()
    # トップの phase が t15 のように名前を持つ場合もあるので、レースごとに snapshot_at が最大のものを final とする
    f["rank"] = f.groupby("race_id")["snapshot_at"].transform(lambda s: s.rank(method="dense", ascending=False))
    fin = f[f["rank"] == 1].rename(columns={"win_odds": "odds_final", "place_odds_min": "place_final", "phase": "phase_final"})[["race_id", "horse_number", "odds_final", "place_final", "phase_final", "snapshot_at"]]
    ear = f[f["phase"] == early].rename(columns={"win_odds": "odds_early", "place_odds_min": "place_early"})[["race_id", "horse_number", "odds_early", "place_early"]]
    m = ear.merge(fin, on=["race_id", "horse_number"], how="inner")
    m = m[m["phase_final"] != early]
    m["log_ratio"] = np.log(m["odds_final"] / m["odds_early"])
    m["band"] = pd.cut(m["odds_early"], BANDS, labels=BAND_LABELS, right=False)
    m["month"] = m["snapshot_at"].astype(str).str[:7]
    return m


def band_stats(m: pd.DataFrame) -> pd.DataFrame:
    """帯ごとの変化率。log_ratio < 0 はオッズ低下（締切直前に買われた）、> 0 は上昇。"""
    g = m.groupby("band", observed=True)["log_ratio"]
    out = g.agg(n="size", mean="mean", median="median", sd="std").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out["倍率(平均)"] = np.exp(out["mean"])
    out["20%超低下の割合"] = m.groupby("band", observed=True)["log_ratio"].apply(lambda s: float((s < np.log(0.8)).mean())).to_numpy()
    return out


def haircut_table(m: pd.DataFrame, min_n: int = 200, shrink: float = 0.5) -> dict[str, float]:
    """判断オッズ（T-15 の値）に掛ける補正係数 exp(shrink × 平均 log 変化率)。サンプルが少ない帯は 1.0（補正なし）。shrink で過補正を避ける。"""
    st = band_stats(m)
    return {str(r["band"]): float(np.exp(shrink * r["mean"])) if r["n"] >= min_n else 1.0 for _, r in st.iterrows()}


def apply_haircut(odds: np.ndarray, table: dict[str, float]) -> np.ndarray:
    """T-15 のオッズから、確定時に期待されるオッズへ。"""
    band = pd.cut(pd.Series(odds), BANDS, labels=BAND_LABELS, right=False).astype(str)
    return odds * band.map(table).fillna(1.0).to_numpy(dtype=float)


def drift_alert(recent: pd.DataFrame, base: pd.DataFrame, z_thresh: float = 3.0, min_n: int = 200) -> pd.DataFrame:
    """帯ごとの平均 log 変化率が、基準期間から直近でどれだけずれたか（z 値）。|z| ≥ z_thresh で警報。"""
    a, b = band_stats(recent).set_index("band"), band_stats(base).set_index("band")
    rows = []
    for band in a.index.intersection(b.index):
        if a.loc[band, "n"] < min_n or b.loc[band, "n"] < min_n:
            continue
        diff = a.loc[band, "mean"] - b.loc[band, "mean"]
        se = float(np.sqrt(a.loc[band, "se"] ** 2 + b.loc[band, "se"] ** 2))
        z = diff / se if se > 0 else 0.0
        rows.append({"band": str(band), "直近の平均": a.loc[band, "mean"], "基準の平均": b.loc[band, "mean"], "差": diff, "z": z, "警報": abs(z) >= z_thresh})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d_from", required=True)
    ap.add_argument("--to", dest="d_to", required=True)
    ap.add_argument("--early", default="t15")
    ap.add_argument("--out", type=Path, default=Path("data/local/meta/odds_drift/latest.json"))
    a = ap.parse_args()
    from src.utils.project_env import load_project_dotenv
    from src.scraper.storage import HybridStorage

    load_project_dotenv()
    st = HybridStorage()
    keys = [k for y in sorted({a.d_from[:4], a.d_to[:4]}) for k in st.batch_list_blobs("race_odds", y)]
    docs = [st.load("race_odds", k, bypass_cache=True) for k in keys]
    docs = [d for d in docs if isinstance(d, dict) and d.get("snapshots") and str(d.get("snapshot_at") or "")[:10].replace("-", "") >= a.d_from and str(d.get("snapshot_at") or "")[:10].replace("-", "") <= a.d_to]
    flat = flatten_docs(docs)
    m = pair_phases(flat, early=a.early)
    res = {"docs": len(docs), "pairs": int(len(m)), "early": a.early, "bands": band_stats(m).to_dict("records") if len(m) else [], "haircut": haircut_table(m) if len(m) else {}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "bands"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
