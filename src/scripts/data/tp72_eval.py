"""tp-72 第 1 段の評価: 事前登録した候補（tp72_candidates.json）を、モデル（V4 の OOF / 保留年の予測）と市場に対する残差、回収率で検定する。

python -m src.scripts.data.tp72_eval stage1 [--cands PATH] [--out DIR]

発見 2022〜2024、確認 2025（通過した候補だけを 1 回）、2026 は記述統計のみ（選択に使わない）。
各候補・各アウトカム（勝ち / 複勝）について、レース単位のクラスタ頑健な z（候補に当たる馬の残差のレース内合計 ÷ その二乗和の平方根）を出す:
  z_m: 勝ち(複勝)の実績 − モデル確率、z_k: 実績 − 市場確率（単勝オッズの正規化 / Harville）、z_r: 回収の超過（同じ単勝オッズ帯の同年平均との差）。
スクリーニングは min(z_m, z_k)（両方が正でなければ通らない）の片側 p に BH（候補数 × 2 アウトカム）。回収率は通過後の確認で見る。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("TP72_OUT", "/home/jovyan/work/tp72_out"))
DISC = (2022, 2023, 2024)
CONF = 2025
DESC = 2026
MIN_RUNS, MIN_RACES = 300, 200
EDGES = [0, 1.5, 2, 3, 4, 6, 8, 12, 16, 25, 40, 70, 150, 1e9]


def phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def prepare(path: Path | None = None) -> pd.DataFrame:
    from src.utils.race_probabilities import harville_top3_prob

    d = pd.read_parquet(path or (OUT / "mining.parquet"))
    d = d[d["p_win"].notna() & d["p_top3"].notna() & (d["win_odds"] > 0) & (d["fs"] >= 8) & d["finish_position"].notna()].copy()
    d = d.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    inv = 1.0 / d["win_odds"].astype(float)
    d["m_win"] = inv / inv.groupby(d["race_id"]).transform("sum")
    m3 = np.zeros(len(d))
    for _, idx in d.groupby("race_id", sort=False).indices.items():
        m3[idx] = harville_top3_prob(d["m_win"].to_numpy()[idx])
    d["m_top3"] = m3
    d["r1"] = d["ret_win"].astype(float) / 100.0
    d["r3"] = d["ret_place"].astype(float) / 100.0
    d["odds_band"] = pd.cut(d["win_odds"], EDGES, labels=False)
    d["fs_c"] = d["fs"].clip(8, 18)
    d["b1"] = d.groupby(["year", "odds_band", "fs_c"])["r1"].transform("mean")      # 同年・同単勝オッズ帯・同頭数の平均（頭数が違うと複勝圏の確率が違うので揃える）
    d["b3"] = d.groupby(["year", "odds_band", "fs_c"])["r3"].transform("mean")
    d["y1"] = d["win"].astype(float)
    d["y3"] = d["top3"].astype(float)
    d["rc"] = pd.factorize(d["race_id"])[0]
    for k, (y, p) in {"m1": ("y1", "p_win"), "k1": ("y1", "m_win"), "m3": ("y3", "p_top3"), "k3": ("y3", "m_top3")}.items():
        d["e_" + k] = d[y] - d[p]
    d["e_r1"] = d["r1"] - d["b1"]
    d["e_r3"] = d["r3"] - d["b3"]
    return d


def cluster_z(d: pd.DataFrame, mask: np.ndarray, col: str, nrace: int) -> tuple[float, float, int]:
    """(z, 合計, 馬数)。レース内合計 R をクラスタとして z = ΣR / sqrt(ΣR²)。"""
    if mask.sum() == 0:
        return float("nan"), 0.0, 0
    rc = d["rc"].to_numpy()[mask]
    R = np.bincount(rc, weights=d[col].to_numpy()[mask], minlength=nrace)
    den = math.sqrt(float((R ** 2).sum()))
    return (float(R.sum() / den) if den > 0 else float("nan")), float(R.sum()), int(mask.sum())


def eval_mask(d: pd.DataFrame, mask: np.ndarray, nrace: int) -> dict:
    out: dict = {}
    yr = d["year"].to_numpy()
    for tag, sel in (("disc", np.isin(yr, DISC)), ("y2022", yr == 2022), ("y2023", yr == 2023), ("y2024", yr == 2024), ("y2025", yr == CONF), ("y2026", yr == DESC)):
        mm = mask & sel
        n = int(mm.sum())
        o = {"n": n, "races": int(len(np.unique(d["rc"].to_numpy()[mm]))) if n else 0}
        if n:
            for k in ("m1", "k1", "r1", "m3", "k3", "r3"):
                z, tot, _ = cluster_z(d, mm, "e_" + k, nrace)
                o["z_" + k] = z
                o["mean_" + k] = tot / n
            o["roi_win"] = float(d["r1"].to_numpy()[mm].mean())
            o["roi_place"] = float(d["r3"].to_numpy()[mm].mean())
            o["hit_win"] = float(d["y1"].to_numpy()[mm].mean())
            o["p_win_mean"] = float(d["p_win"].to_numpy()[mm].mean())
            o["hit_top3"] = float(d["y3"].to_numpy()[mm].mean())
            o["p_top3_mean"] = float(d["p_top3"].to_numpy()[mm].mean())
        out[tag] = o
    return out


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def boot_roi(d: pd.DataFrame, mask: np.ndarray, col: str, reps: int = 1000, seed: int = 0) -> list[float]:
    rc = d["rc"].to_numpy()[mask]
    v = d[col].to_numpy()[mask]
    if len(v) == 0:
        return [float("nan")] * 3
    nr = int(rc.max()) + 1
    s = np.bincount(rc, weights=v, minlength=nr)
    c = np.bincount(rc, minlength=nr).astype(float)
    keep = c > 0
    s, c = s[keep], c[keep]
    rng = np.random.default_rng(seed)
    n = len(s)
    vals = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        vals.append(s[i].sum() / c[i].sum())
    return [float(s.sum() / c.sum()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def stage1(cands_path: Path, out_dir: Path) -> dict:
    d = prepare()
    nrace = int(d["rc"].max()) + 1
    nfield = np.bincount(d["rc"].to_numpy(), minlength=nrace)
    spec = json.loads(Path(cands_path).read_text(encoding="utf-8"))
    cands = spec["candidates"]
    rows = []
    for c in cands:
        try:
            mask = d.eval(c["expr"], engine="python").fillna(False).to_numpy().astype(bool)
            err = None
        except Exception as exc:  # 列が無い等。検定不能として数に含める
            mask = np.zeros(len(d), bool)
            err = str(exc)[:120]
        r = eval_mask(d, mask, nrace)
        dm = mask & np.isin(d["year"].to_numpy(), DISC)
        full = 0.0
        if dm.sum():
            nsel = np.bincount(d["rc"].to_numpy()[dm], minlength=nrace)
            full = float(nsel[d["rc"].to_numpy()[dm]].__eq__(nfield[d["rc"].to_numpy()[dm]]).mean())
        rows.append({**c, "stats": r, "error": err, "full_race_frac": full, "race_level": bool(full >= 0.9)})
    # 発見期間の検定
    for outcome, (zm, zk, zr) in (("win", ("z_m1", "z_k1", "z_r1")), ("place", ("z_m3", "z_k3", "z_r3"))):
        ps = []
        for r in rows:
            s = r["stats"]["disc"]
            ok = s["n"] >= MIN_RUNS and s["races"] >= MIN_RACES and not (np.isnan(s.get(zm, np.nan)) or np.isnan(s.get(zk, np.nan)))
            # 出走馬全員が当たるレース単位の条件は、勝ち（複勝）の確率の合計が 1（3）に決まっているので残差の検定が成り立たない → 回収の超過 z_r だけを検定する
            z = (s[zr] if r["race_level"] else min(s[zm], s[zk])) if ok else float("nan")
            r[f"z_min_{outcome}"] = None if not ok else float(z)
            r[f"p_{outcome}"] = 1.0 if not ok else float(1 - phi(z))
            ps.append(r[f"p_{outcome}"])
        ps = np.array(ps)
        # BH は 2 アウトカムを合わせた全検定数で補正する（下で再計算）
        for r, p in zip(rows, ps):
            r[f"p_{outcome}_raw"] = float(p)
    n = len(rows)
    blk = np.array([2 if r["family"].startswith(("F21", "F22", "F23")) else 1 for r in rows])
    for r in rows:
        r["q_win"] = r["q_place"] = 1.0
    for b in (1, 2):           # BH はブロックごと（1: F1〜F20 / 2: 血統×騎手 F21〜F23）。全体で補正した q も併記
        ids = np.where(blk == b)[0]
        pb = np.array([rows[i]["p_win"] for i in ids] + [rows[i]["p_place"] for i in ids])
        qb = bh(pb)
        for k, i in enumerate(ids):
            rows[i]["q_win"] = float(qb[k])
            rows[i]["q_place"] = float(qb[len(ids) + k])
    allp = np.array([r["p_win"] for r in rows] + [r["p_place"] for r in rows])
    qg = bh(allp)
    for i, r in enumerate(rows):
        r["q_win_global"] = float(qg[i])
        r["q_place_global"] = float(qg[n + i])
    q = np.array([r["q_win"] for r in rows] + [r["q_place"] for r in rows])
    zs_w = np.array([r["z_min_win"] for r in rows if r["z_min_win"] is not None])
    zs_p = np.array([r["z_min_place"] for r in rows if r["z_min_place"] is not None])
    summary = {
        "n_candidates": n, "n_tests": 2 * n, "n_testable": int(len(zs_w)),
        "z_min_win_median": float(np.median(zs_w)), "z_min_win_mad_sd": float(1.4826 * np.median(np.abs(zs_w - np.median(zs_w)))),
        "z_min_place_median": float(np.median(zs_p)), "z_min_place_mad_sd": float(1.4826 * np.median(np.abs(zs_p - np.median(zs_p)))),
        "expected_false_pos_at_z1.645": float(0.05 * 2 * len(zs_w)),
        "n_z_gt_1.645_win": int((zs_w > 1.645).sum()), "n_z_gt_1.645_place": int((zs_p > 1.645).sum()),
        "n_q_lt_0.10": int(((q < 0.10)).sum()), "n_q_global_lt_0.10": int((qg < 0.10).sum()),
    }
    # 確認（2025 を 1 回）: 発見で q<0.10 のものと、記述用の上位 25（z_min が高い順）
    passed = [r for r in rows if r["q_win"] < 0.10 or r["q_place"] < 0.10]
    top = sorted([r for r in rows if r["z_min_win"] is not None], key=lambda r: -max(r["z_min_win"], r["z_min_place"] if r["z_min_place"] is not None else -9))[:25]
    for r in {id(x): x for x in passed + top}.values():
        nm = d.eval(r["expr"], engine="python").fillna(False).to_numpy().astype(bool)
        sel_d = nm & np.isin(d["year"].to_numpy(), DISC)
        sel_5 = nm & (d["year"].to_numpy() == CONF)
        sel_6 = nm & (d["year"].to_numpy() == DESC)
        r["roi_ci"] = {tag: {c: boot_roi(d, s, col) for c, col in (("win", "r1"), ("place", "r3"))} for tag, s in (("disc", sel_d), ("y2025", sel_5), ("y2026", sel_6))}
    out = {"summary": summary, "rows": rows, "passed_names": [r["name"] for r in passed], "top_names": [r["name"] for r in top]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1"])
    ap.add_argument("--cands", default=str(Path(__file__).with_name("tp72_candidates.json")))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    r = stage1(Path(a.cands), Path(a.out))
    print(json.dumps(r["summary"], ensure_ascii=False, indent=1))
    print("passed:", r["passed_names"])
