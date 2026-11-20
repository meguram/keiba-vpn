"""TS（ラスト 3F の最速ラップ）が、コース・距離帯ごとにどんな分布で、どれだけ再現するかを数える（読むだけ・JSON を書く）。

入力（ホストのローカルだけ。GCS は読まない）: 馬別ラップ（horse_lap の JSONL）。run_axes.with_ts と同じ作りで、
  補正後ピーク x = 最速ラップ − レース水準のずれ、コース平均 mu_s・標準偏差 sd_s、偏差値 ts を出す。
出力:
  course … コース（場・芝ダ・距離）ごとの 走数・生ピークの平均・x の平均・標準偏差・歪み・コース内の 5/50/95%点
  band   … 芝ダ × 距離帯ごとの同じ値 + レース内の標準偏差 + 0.1 秒刻みの粗さ + 再現性
  どちらにも 現行 TS の分布: ts_p05/25/50/75/95・ts_sd・ts_skew・ts_min/max・ts_hist（10〜90 を 4 点刻み 20 階級 + 両端の外れ値 = 22 階級）・peak20（最速ラップ 20 秒以上の走数）
  再現性 … 馬ごとに同じ芝ダ・距離帯の古い順 4 走を (1,3) と (2,4) に分けた平均の相関 → Spearman-Brown で 4 走の信頼性 R →
           1 走の信頼性 rho = R/(4-3R)（docs/ideas/running_axes_scale_1to10.md の問題 6 と同じ定義）。
           あわせて、2 走以上ある馬の ICC(1)（1 走の信頼性。馬の間の分散 ÷ 全体）。区間はレースでなく馬単位のブートストラップ。
使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I src/scripts/data/ts_course_resolution.py OUT.json /home/jovyan/acq/horse_lap/horse_lap_*.jsonl
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from src.research.race import run_axes as ra  # noqa: E402

DIST_ORDER = ["〜1200m", "1300〜1400m", "1500〜1700m", "1800〜2000m", "2100〜2400m", "2500m〜"]
COURSE_MIN_RUNS = 300
HIST_LO, HIST_HI, HIST_STEP = 10.0, 90.0, 4.0   # ts のヒストグラム: 10〜90 を 4 点刻み（20 階級）+ 両端の外れ値の階級（計 22）
PEAK_OUTLIER = 20.0                              # 最速ラップがこれ以上の走 = 異常走（中止・歩様など）
BOOT = 300
MIN_HORSES = 150


def dist_band(d: float) -> str:
    d = int(d)
    return DIST_ORDER[0] if d <= 1200 else DIST_ORDER[1] if d <= 1400 else DIST_ORDER[2] if d <= 1700 else DIST_ORDER[3] if d <= 2000 else DIST_ORDER[4] if d <= 2400 else DIST_ORDER[5]


def dist_stats(x: pd.Series) -> dict:
    q = x.quantile([0.05, 0.5, 0.95])
    return {"n": int(len(x)), "mean": round(float(x.mean()), 4), "sd": round(float(x.std()), 4), "skew": round(float(x.skew()), 3),
            "p05": round(float(q[0.05]), 3), "p50": round(float(q[0.5]), 3), "p95": round(float(q[0.95]), 3)}


def ts_block(g: pd.DataFrame) -> dict:
    """現行 TS の分布（5/25/50/75/95% 点・標準偏差・歪み・最小/最大・ヒストグラム）と、異常走（peak が 20 秒以上）の数。"""
    t = g["ts"]
    q = t.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    inner = np.arange(HIST_LO, HIST_HI + 1e-9, HIST_STEP)
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    hist = np.histogram(t.to_numpy(), bins=edges)[0]
    return {"ts_sd": round(float(t.std()), 3), "ts_skew": round(float(t.skew()), 3),
            "ts_p05": round(float(q[0.05]), 2), "ts_p25": round(float(q[0.25]), 2), "ts_p50": round(float(q[0.5]), 2),
            "ts_p75": round(float(q[0.75]), 2), "ts_p95": round(float(q[0.95]), 2),
            "ts_min": round(float(t.min()), 2), "ts_max": round(float(t.max()), 2),
            "ts_hist": [int(x) for x in hist], "peak20": int((g["peak"] >= PEAK_OUTLIER).sum())}


def split_half(first4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return first4[:, [0, 2]].mean(1), first4[:, [1, 3]].mean(1)


def rho_from_r(r: float) -> float:
    big_r = 2 * r / (1 + r)
    return float(big_r / (4 - 3 * big_r))


def icc1(groups: list[np.ndarray]) -> float:
    """一元配置の ICC(1)（馬ごとの走数が違ってよい）。馬の間の分散 ÷ (馬の間 + 馬の内)。"""
    k = np.array([len(g) for g in groups], float)
    n, m = k.sum(), len(groups)
    allv = np.concatenate(groups)
    gm = allv.mean()
    ssb = float(sum(len(g) * (g.mean() - gm) ** 2 for g in groups))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb, msw = ssb / (m - 1), ssw / (n - m)
    k0 = (n - (k ** 2).sum() / n) / (m - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def reliability(g: pd.DataFrame, rng: np.random.Generator) -> dict | None:
    """g: horse_id, date, ts（同じ芝ダ・距離帯の走）。"""
    g = g.sort_values(["horse_id", "date"])
    series = [v.to_numpy() for _, v in g.groupby("horse_id")["ts"] if len(v) >= 2]
    four = np.array([v[:4] for v in series if len(v) >= 4])
    if len(four) < MIN_HORSES:
        return None
    a, b = split_half(four)
    r = float(np.corrcoef(a, b)[0, 1])
    boot_r, boot_i = [], []
    for _ in range(BOOT):
        i = rng.integers(0, len(four), len(four))
        boot_r.append(rho_from_r(float(np.corrcoef(a[i], b[i])[0, 1])))
    j = rng.integers(0, len(series), (BOOT, len(series)))
    for row in j[:100]:
        boot_i.append(icc1([series[t] for t in row]))
    return {"horses4": int(len(four)), "horses2": int(len(series)), "r_half": round(r, 3), "rho_split": round(rho_from_r(r), 3),
            "rho_split_ci": [round(float(np.percentile(boot_r, 2.5)), 3), round(float(np.percentile(boot_r, 97.5)), 3)],
            "icc1": round(icc1(series), 3), "icc1_ci": [round(float(np.percentile(boot_i, 2.5)), 3), round(float(np.percentile(boot_i, 97.5)), 3)]}


def main() -> None:
    out_path = Path(sys.argv[1])
    paths = sys.argv[2:]
    runs = ra.read_runs(paths)
    h = ra.with_ts(runs, ra.default_years())
    h = h[h["surface"].isin(["芝", "ダート"]) & h["offset"].notna()].copy()
    h["date"] = pd.to_datetime(h["date"])
    h["dband"] = [dist_band(d) for d in h["distance"]]
    rng = np.random.default_rng(20261118)

    # コース内の順位の細かさ: 最速ラップは 0.1 秒刻みの記録なので、x の標準偏差に対する刻みの比も出す
    h["peak_front"] = h["peak"] <= 11.5
    h["race_n"] = h.groupby("race_id")["x"].transform("size")

    courses = []
    for (v, s, d), g in h.groupby(["venue", "surface", "distance"]):
        if len(g) < COURSE_MIN_RUNS:
            continue
        r = dist_stats(g["x"])
        r.update({"venue": v, "surface": s, "distance": int(d), "mu_s": round(float(g["mu_s"].iloc[0]), 4), "sd_s": round(float(g["sd_s"].iloc[0]), 4),
                  "peak_mean": round(float(g["peak"].mean()), 4), "races": int(g["race_id"].nunique()), "front115": round(float(g["peak_front"].mean()), 4)})
        r.update(ts_block(g))
        courses.append(r)

    bands = []
    for s in ("芝", "ダート"):
        for b in DIST_ORDER:
            g = h[(h["surface"] == s) & (h["dband"] == b)]
            if len(g) < 500:
                continue
            r = dist_stats(g["x"])
            within = g["x"] - g.groupby("race_id")["x"].transform("mean")
            r.update({"surface": s, "band": b, "races": int(g["race_id"].nunique()), "peak_mean": round(float(g["peak"].mean()), 4),
                      "peak_sd": round(float(g["peak"].std()), 4), "front115": round(float(g["peak_front"].mean()), 4),
                      "within_race_sd": round(float(within.std()), 4), "grain_over_sd": round(0.1 / float(g["x"].std()), 3),
                      "unique_peak": int(g["peak"].round(1).nunique()), "sd_s_mean": round(float(g["sd_s"].mean()), 4)})
            r.update(ts_block(g))
            r["rel"] = reliability(g[["horse_id", "date", "ts"]], rng)
            bands.append(r)
            print(s, b, r["n"], "sd", r["sd"], "skew", r["skew"], "rel", r["rel"], flush=True)

    rel_all = reliability(h[["horse_id", "date", "ts"]], rng)
    meta = {"built": date.today().isoformat(), "runs": int(len(h)), "races": int(h["race_id"].nunique()),
            "date_from": str(h["date"].min().date()), "date_to": str(h["date"].max().date()), "course_min_runs": COURSE_MIN_RUNS, "boot": BOOT,
            "hist": {"lo": HIST_LO, "hi": HIST_HI, "step": HIST_STEP, "peak_outlier": PEAK_OUTLIER}}
    out_path.write_text(json.dumps({"meta": meta, "all_rel": rel_all, "bands": bands, "courses": courses}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_path, meta, "all_rel", rel_all)


if __name__ == "__main__":
    main()
