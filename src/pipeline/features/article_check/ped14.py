"""V3: 血統の項目（父・母父・母の産駒成績）と、出走数による減衰（記事の主張 B）。

産駒の成績は、産駒の過去走の着順の相対位置（1 着 = 1）を、そのレースの日より前（同日を含まない）で集計する。
  - pow  : 伝える強さ。産駒全体の平均 − 0.5。
  - apt_surf : 今日の面への適性。今日の面の産駒平均 − 産駒全体の平均。
  - apt_dist : 今日の距離帯への適性。同上。
  - pref : 芝ダ得意度。芝の産駒平均 − ダートの産駒平均。
どれも K=20 で縮約（セルは産駒全体の平均へ、全体は 0.5 へ）。キー 3 種（父 sire・母父 dsire・母 dam）× 4 項目 = 12 本。
減衰版: 12 本 × 1/(1 + n/TAU)。n はその馬の過去の平地出走数（記事の「戦績が増えるほど血統の比重を下げる」）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.race_env.asof import prior_cell_mean

from .common import finish_pct

ROOT = Path(__file__).resolve().parents[4]
PED_DIRS = [ROOT / "data/local/horse_pedigree_5gen", ROOT / "data/local/horse_pedigree_10gen"]
KEYS = ("sire", "dsire", "dam")
ITEMS = ("pow", "apt_surf", "apt_dist", "pref")
SHRINK_K = 20.0
TAU = 4.0
DIST_CUTS = (1400, 1800, 2200)


def raw_cols() -> list[str]:
    return [f"ped14_{k}_{i}" for k in KEYS for i in ITEMS]


def gated_cols() -> list[str]:
    return [f"{c}_g" for c in raw_cols()]


def parent_ids(horse_ids, dirs: list[Path] | None = None) -> pd.DataFrame:
    """horse_id → 父・母父・母の ID（generation=1/position=0、generation=2/position=2、generation=1/position=1）。不明は空文字。"""
    dirs = [Path(x) for x in (dirs or PED_DIRS)]
    rows = []
    for h in sorted({str(x) for x in horse_ids}):
        sire = dsire = dam = ""
        for d in dirs:
            f = d / h[:4] / f"{h}.json"
            if not f.exists():
                continue
            try:
                anc = json.loads(f.read_text(encoding="utf-8")).get("ancestors") or []
            except Exception:
                continue
            m = {(a.get("generation"), a.get("position")): str(a.get("horse_id") or "") for a in anc}
            sire, dsire, dam = m.get((1, 0), ""), m.get((2, 2), ""), m.get((1, 1), "")
            if sire or dsire or dam:
                break
        rows.append((h, sire, dsire, dam))
    return pd.DataFrame(rows, columns=["horse_id", "sire", "dsire", "dam"])


def _dband(dist: pd.Series) -> pd.Series:
    return pd.Series(np.digitize(dist.to_numpy(), DIST_CUTS), index=dist.index)


def _shr(mean: pd.Series, n: pd.Series, prior) -> pd.Series:
    return (mean.fillna(0.0) * n + SHRINK_K * prior) / (n + SHRINK_K)


def build(runs: pd.DataFrame, parents: pd.DataFrame) -> pd.DataFrame:
    """runs: load_runs() の出力（全期間）。parents: parent_ids() の出力。返り値は race_id・horse_id と raw_cols()・gated_cols()、出走数 ped14_nruns。"""
    r = runs.merge(parents, on="horse_id", how="left").reset_index(drop=True)
    r["pct"] = finish_pct(r)
    r["db"] = _dband(r["distance"])
    r["turf"] = r["pct"].where(r["surface"] == "芝")
    r["dirt"] = r["pct"].where(r["surface"] == "ダート")
    out = r[["race_id", "horse_id"]].copy()
    for k in KEYS:
        valid = r[k].fillna("") != ""
        sub = r[valid].copy()
        m_all, n_all = prior_cell_mean(sub, [k], "pct")
        ov = _shr(m_all, n_all, 0.5)
        m_s, n_s = prior_cell_mean(sub, [k, "surface"], "pct")
        m_d, n_d = prior_cell_mean(sub, [k, "db"], "pct")
        m_t, n_t = prior_cell_mean(sub, [k], "turf")
        m_r, n_r = prior_cell_mean(sub, [k], "dirt")
        res = {
            "pow": ov - 0.5,
            "apt_surf": _shr(m_s, n_s, ov) - ov,
            "apt_dist": _shr(m_d, n_d, ov) - ov,
            "pref": _shr(m_t, n_t, ov) - _shr(m_r, n_r, ov),
        }
        for i in ITEMS:
            col = pd.Series(np.nan, index=r.index)
            col[valid.to_numpy()] = res[i].to_numpy()
            out[f"ped14_{k}_{i}"] = col.to_numpy()
    n = r.groupby("horse_id", sort=False).cumcount()
    out["ped14_nruns"] = n.to_numpy()
    w = 1.0 / (1.0 + n.to_numpy() / TAU)
    for c in raw_cols():
        out[f"{c}_g"] = out[c] * w
    return out
