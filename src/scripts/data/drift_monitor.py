"""ドリフトの週次監視: 毎週、週末の結果（着順・払戻・確定オッズ）が揃った後に 1 回、直近 8 週のローリング窓と全期間の累積で測る。

  python -m src.scripts.data.drift_monitor [--root .] [--out-dir data/local/meta/drift_monitor] [--no-features] [--no-shadow] [--no-coverage]
  python -m src.scripts.data.drift_monitor --ack-reset 2026-11-02 --note "調教ページの取り直しを完了"
  python -m src.scripts.data.drift_backtest   # 過去の各週にさかのぼって、どの週に警報が出たか・誤警報の率

主指標は『モデルと市場（確定単勝オッズ）の勝率 LogLoss の差』（レースの難しさの変動を打ち消す）。勝率/連対/複勝の LogLoss・Brier・ECE・較正曲線・的中の順位相関も併記する。
窓の揺れは、週次の EWMA/CUSUM 管理図と、レースでクラスタした標準誤差・ブートストラップ区間で、『ばらつきの範囲』と『有意な悪化』を分ける。窓内のレース数が min_races 未満は判定保留。
ラベルなしの先行指標（予測分布・新しい ID・特徴量の PSI・調教の被覆）は結果が出る前にも出せるので別枠にする（drift_labelfree）。
警報の段階: 注意（1 項目が 2σ 超え）／警告（独立な 2 群以上が連続 2 週超え）／要対応（差が 3σ 超えを 4 週連続、または較正のずれが回収率に効く水準を 2 週連続）。
手順は docs/operations/drift_monitor.md。**指標は報告・警報のみ。モデルや閾値の選択・学習には使わない**（2026 年はテストセット②）。
自動実行は KEIBA_DRIFT_MONITOR=1 のとき（drift_monitor_hook）。手動・cron は環境変数に依らず実行する。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M
from src.scripts.data.drift_data import PCOLS, add_strata, load_cfg, load_eval_frame

ENV_FLAG = "KEIBA_DRIFT_MONITOR"
ENV_DIR = "KEIBA_DRIFT_MONITOR_DIR"
STAGES = ["ok", "注意", "警告", "要対応"]
STAGE_COLOR = {"ok": "green", "注意": "yellow", "警告": "orange", "要対応": "red", "保留": "gray"}
LIMIT_KEYS = ("gap", "gap_top2", "gap_top3", "ece", "rank_gap", "sd_ratio", "psi_p")
ECE_ROI = 0.012                         # 較正のずれが回収率に効く水準（readiness 基準 ECE 勝率 0.006 の 2 倍）
SLOPE_RANGE = (0.85, 1.15)              # readiness 基準の logit 傾き


def enabled() -> bool:
    return (os.environ.get(ENV_FLAG) or "").strip().lower() in ("1", "true", "on", "yes")


def default_out_dir(root: Path) -> Path:
    return Path(os.environ.get(ENV_DIR) or root / "data/local/meta/drift_monitor")


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), M.EPS, 1 - M.EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- 行ごとの前計算

def prepare(df: pd.DataFrame, with_market_top: bool = True) -> pd.DataFrame:
    """行ごとの LL・Brier・差、市場の連対/複勝（Harville）を前計算し、週（月曜）を付ける。"""
    d = df.copy()
    d["week"] = M.iso_week_of(d["date"])
    d["ll_m"] = M.bin_ll(d["y_win"], d["p_win"])
    d["ll_k"] = M.bin_ll(d["y_win"], d["p_mkt"])
    d["d"] = d["ll_m"] - d["ll_k"]
    d["br_m"] = (d["p_win"] - d["y_win"]) ** 2
    d["ll2_m"], d["ll3_m"] = M.bin_ll(d["y_top2"], d["p_top2"]), M.bin_ll(d["y_top3"], d["p_top3"])
    if with_market_top and "p2_k" not in d:
        d["p2_k"], d["p3_k"] = M.market_top_probs(d)
    if "p2_k" in d:
        d["d2"] = d["ll2_m"] - M.bin_ll(d["y_top2"], d["p2_k"])
        d["d3"] = d["ll3_m"] - M.bin_ll(d["y_top3"], d["p3_k"])
    else:
        d["d2"] = d["d3"] = np.nan
    d["lm"], d["lk"] = _logit(d["p_win"]), _logit(d["p_mkt"])
    return d


def race_table(d: pd.DataFrame) -> pd.DataFrame:
    """レースごとの順位相関（モデル/市場）と週。窓の的中の順位相関に使う。"""
    rc_m = M.race_rank_corr_by_race(d, "p_win")
    rc_k = M.race_rank_corr_by_race(d, "p_mkt")
    wk = d.groupby("race_id")["week"].first()
    return pd.DataFrame({"week": wk, "rc_m": rc_m, "rc_k": rc_k}).dropna()


def window_metrics(g: pd.DataFrame, rt: pd.DataFrame | None = None, base_edges_p: np.ndarray | None = None) -> dict:
    """窓（行の集合）の指標。g は prepare 済み。rt は race_table のうち窓に入るレース。"""
    if len(g) == 0:
        return {}
    gap, se = M.cluster_gap_se(g, "d")
    a, b = M.calib_slope(g["p_win"], g["y_win"])
    top = g.loc[g.groupby("race_id")["p_win"].idxmax()]
    out = {"races": int(g["race_id"].nunique()), "rows": int(len(g)), "gap": gap, "gap_se": se,
           "gap_top2": float(g["d2"].mean()), "gap_top3": float(g["d3"].mean()),
           "ll_win": float(g["ll_m"].mean()), "ll_win_mkt": float(g["ll_k"].mean()), "ll_top2": float(g["ll2_m"].mean()), "ll_top3": float(g["ll3_m"].mean()),
           "brier_win": float(g["br_m"].mean()), "brier_win_mkt": M.brier(g["y_win"], g["p_mkt"]), "brier_top3": M.brier(g["y_top3"], g["p_top3"]),
           "ece": M.ece(g["p_win"], g["y_win"]), "ece_top2": M.ece(g["p_top2"], g["y_top2"]), "ece_top3": M.ece(g["p_top3"], g["y_top3"]),
           "slope": b, "intercept": a, "sd_ratio": float(g["lm"].std() / g["lk"].std()), "corr": float(np.corrcoef(g["lm"], g["lk"])[0, 1]),
           "top_pick_win": float(top["y_win"].mean())}
    if rt is not None and len(rt):
        out["rank_corr"], out["rank_corr_mkt"] = float(rt["rc_m"].mean()), float(rt["rc_k"].mean())
        out["rank_gap"] = out["rank_corr_mkt"] - out["rank_corr"]
    if base_edges_p is not None and len(base_edges_p):
        out["psi_p"] = M.psi_with_edges(base_edges_p, g["p_win"].to_numpy())
    return out


def rolling_series(d: pd.DataFrame, window: int = 8, rt: pd.DataFrame | None = None, base_edges_p: np.ndarray | None = None) -> pd.DataFrame:
    """週(月曜)ごとに、その週を末尾とする window 週の窓の指標。窓が満たない最初の週は除く。週単独の値（CUSUM/EWMA 用）も付ける。"""
    weeks = sorted(d["week"].unique())
    by_week = {w: g for w, g in d.groupby("week")}
    rows = []
    for i in range(window - 1, len(weeks)):
        ws = weeks[i - window + 1: i + 1]
        sub = pd.concat([by_week[w] for w in ws])
        rts = rt[rt["week"].isin(ws)] if rt is not None else None
        m = window_metrics(sub, rts, base_edges_p)
        g1 = by_week[weeks[i]]
        m.update({"week": weeks[i], "week_races": int(g1["race_id"].nunique()), "week_gap": float(g1["d"].mean()),
                  "week_sd_ratio": float(g1["lm"].std() / g1["lk"].std())})
        rows.append(m)
    return pd.DataFrame(rows)


def baseline_limits(series: pd.DataFrame, start: str, end: str, k: float) -> dict:
    """基準期間（窓が全部基準に入る週）の窓の指標の平均と ±kσ（と 2σ）。"""
    s = series[(series["week"] >= start) & (series["week"] <= end)]
    if len(s) < 20:
        return {}
    out = {}
    for key in LIMIT_KEYS + ("corr", "rank_corr"):
        if key in s.columns and s[key].notna().sum() > 5:
            out[key] = M.control_limits(s[key].to_numpy(), k)
            out[key]["k2_upper"] = out[key]["mean"] + 2 * out[key]["sd"]
            out[key]["k2_lower"] = out[key]["mean"] - 2 * out[key]["sd"]
    return out


def weekly_baseline(d: pd.DataFrame, start: str, end: str, min_races: int | None = None) -> dict:
    """基準期間の『週ごとの値』の平均と標準偏差（CUSUM・EWMA の基準）。レース数が少ない週（祝日の短い開催など）は除く（既定: 週のレース数の中央値の 4 割）。"""
    x = d[(d["week"] >= start) & (d["week"] <= end)]
    if min_races is None:
        min_races = max(5, int(0.4 * x.groupby("week")["race_id"].nunique().median()))
    rows = []
    for _, g in x.groupby("week"):
        if g["race_id"].nunique() >= min_races:
            rows.append({"gap": g["d"].mean(), "sd_ratio": g["lm"].std() / g["lk"].std()})
    t = pd.DataFrame(rows)
    return {k: {"mean": float(t[k].mean()), "sd": float(t[k].std(ddof=1)), "n": int(len(t))} for k in ("gap", "sd_ratio")}


# ---------------------------------------------------------------- 警報の段階

def _flag(v, lim: dict | None, which: str, k: int) -> bool:
    if v is None or lim is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if which == "upper":
        return bool(v > (lim["ucl"] if k == 3 else lim["k2_upper"]))
    return bool(v < (lim["lcl"] if k == 3 else lim["k2_lower"]))


def evaluate_sequence(series: pd.DataFrame, limits: dict, wk_base: dict, start: str, alert: dict, coverage_by_week: dict | None = None,
                      cusum_from: str | None = None, end: str | None = None) -> pd.DataFrame:
    """start 以降の各週について、指標ごとの超過・EWMA/CUSUM・有意性の区分・警報の段階（その週までの情報だけで決める）を出す。"""
    cur = series[(series["week"] >= start) & ((series["week"] <= end) if end else True)].reset_index(drop=True).copy()
    if len(cur) == 0 or not limits:
        return cur
    min_races = int(alert.get("min_races_window", 150))
    h = float(alert.get("cusum_h", 5.0))
    ew = M.ewma(cur["week_gap"].tolist(), float(alert.get("ewma_lambda", 0.2)), wk_base["gap"]["mean"], wk_base["gap"]["sd"], float(alert.get("gap_sigma_k", 3.0)))
    cs_idx = cur.index[cur["week"] >= (cusum_from or start)]
    cus = M.cusum(cur.loc[cs_idx, "week_gap"].tolist(), wk_base["gap"]["mean"], wk_base["gap"]["sd"], float(alert.get("cusum_k", 0.5)), h)
    cus_s = pd.Series(0.0, index=cur.index)
    cus_s.loc[cs_idx] = cus["path"]
    cur["ewma_gap"], cur["ewma_ucl"], cur["ewma_flag"] = ew["path"], ew["ucl"], ew["flags"]
    cur["cusum_gap"], cur["cusum_flag"] = cus_s, cus_s > h
    g = limits.get("gap", {})
    sig = []
    for _, r in cur.iterrows():
        if r["races"] < min_races:
            sig.append("保留")
        elif r["gap"] - 2 * r["gap_se"] > g.get("mean", 0) and r["gap"] > g.get("ucl", 1):
            sig.append("有意な悪化")
        elif r["gap"] <= g.get("k2_upper", 1):
            sig.append("ばらつきの範囲")
        else:
            sig.append("境界")
    cur["gap_class"] = sig
    ece_roi = float(alert.get("ece_roi", ECE_ROI))
    s_lo, s_hi = float(alert.get("slope_lo", SLOPE_RANGE[0])), float(alert.get("slope_hi", SLOPE_RANGE[1]))
    grp: dict[str, list[bool]] = {k: [] for k in ("perf", "cal", "dist", "inputs")}
    roi, perf3, hold = [], [], []
    for _, r in cur.iterrows():
        cov = (coverage_by_week or {}).get(r["week"])
        perf2 = (_flag(r["gap"], limits.get("gap"), "upper", 2) or _flag(r["gap_top2"], limits.get("gap_top2"), "upper", 2)
                 or _flag(r["gap_top3"], limits.get("gap_top3"), "upper", 2) or _flag(r.get("rank_gap"), limits.get("rank_gap"), "upper", 2) or bool(r["ewma_flag"]))
        cal2 = _flag(r["ece"], limits.get("ece"), "upper", 2) or not (s_lo <= r["slope"] <= s_hi)
        psi_p = r.get("psi_p")
        dist2 = _flag(r["sd_ratio"], limits.get("sd_ratio"), "lower", 2) or _flag(psi_p, limits.get("psi_p"), "upper", 2)
        inp2 = bool(cov is not None and cov < float(alert.get("coverage_floor", 0.90)))
        grp["perf"].append(perf2), grp["cal"].append(cal2), grp["dist"].append(dist2), grp["inputs"].append(inp2)
        perf3.append(_flag(r["gap"], limits.get("gap"), "upper", 3))
        roi.append(bool(r["ece"] > ece_roi or not (s_lo <= r["slope"] <= s_hi)))
        hold.append(bool(r["races"] < min_races))
    for k, v in grp.items():
        cur[f"flag_{k}"] = v
    cur["flag_perf3"], cur["flag_roi_cal"], cur["hold"] = perf3, roi, hold
    n_groups = [sum(grp[k][i] for k in grp) for i in range(len(cur))]
    stages = []
    for i in range(len(cur)):
        if hold[i]:
            stages.append("保留")
            continue
        s = "ok"
        if n_groups[i] >= 1:
            s = "注意"
        if i >= 1 and n_groups[i] >= 2 and n_groups[i - 1] >= 2:
            s = "警告"
        if (i >= 3 and all(perf3[i - 3: i + 1])) or (i >= 1 and all(roi[i - 1: i + 1])):
            s = "要対応"
        stages.append(s)
    cur["stage"] = stages
    return cur


def segments_for_backtest(cfg: dict) -> list[tuple[str, str, str]]:
    """(監視の開始週, 監視の終了, 基準期間の終わり)。2025 年の週は 2022-2024 の OOF だけを基準にする（先読みを避ける）。"""
    return [("2025-01-01", "2025-12-31", "2024-12-31"), (cfg["test2"]["from"], "2099-12-31", cfg["baseline"]["to"])]


def backtest(d: pd.DataFrame, cfg: dict, window: int | None = None, k: float | None = None, coverage_by_week: dict | None = None,
             segments: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    """2025 年の週は 2022-2024（OOF）だけを基準に、2026 年以降の週は 2022-2025 を基準に、各週の警報の段階を再現する（先読みなし）。
    d は prepare 済みの全期間。window/k で感度を見る（本番の設定は変えない）。"""
    al = dict(cfg["alert"])
    if k is not None:
        al["gap_sigma_k"] = k
    w = int(window or al.get("window_weeks", 8))
    rt = race_table(d)
    parts = []
    for s_from, s_to, b_end in (segments or segments_for_backtest(cfg)):
        edges = M.quantile_edges(d[(d["week"] >= cfg["baseline"]["from"]) & (d["week"] <= b_end)]["p_win"].to_numpy(), 10)
        series = rolling_series(d, w, rt, edges)
        lim = baseline_limits(series, cfg["baseline"]["from"], b_end, float(al["gap_sigma_k"]))
        wkb = weekly_baseline(d, cfg["baseline"]["from"], b_end)
        seq = evaluate_sequence(series, lim, wkb, s_from, al, coverage_by_week, end=s_to)
        seq["baseline_end"] = b_end
        parts.append(seq)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- 層別（FDR）

def strata_fdr(d: pd.DataFrame, cur_weeks: list[str], base_end: str, base_start: str, cols: list[str], q_alert: float = 0.10, min_rows: int = 150) -> list[dict]:
    """直近窓と基準期間の、層ごとの差の比較（片側 z 検定、BH の FDR 補正）。参考扱い。"""
    d = add_strata(d)
    d["is_new"] = np.where(d["grade"].astype(str) == "新馬", "新馬", np.where(d["grade"].astype(str) == "未勝利", "未勝利", "その他"))
    d["month_s"] = d["date"].dt.strftime("%m月")
    base = d[(d["week"] >= base_start) & (d["week"] <= base_end)]
    cur = d[d["week"].isin(cur_weeks)]
    out = []
    for col in cols:
        if col not in d.columns:
            continue
        for key, gc in cur.groupby(col):
            gb = base[base[col] == key]
            if len(gc) < min_rows or len(gb) < min_rows:
                continue
            mc, sc = M.cluster_gap_se(gc, "d")
            mb, sb = M.cluster_gap_se(gb, "d")
            z = (mc - mb) / float(np.hypot(sc, sb))
            out.append({"strata": col, "value": str(key), "rows": int(len(gc)), "gap_cur": mc, "gap_base": mb, "z": float(z), "p": M.norm_sf(z)})
    q = M.bh_fdr([o["p"] for o in out])
    for o, qq in zip(out, q):
        o["q"] = qq
        o["flag"] = bool(qq < q_alert and o["gap_cur"] > o["gap_base"])
    return sorted(out, key=lambda o: o["q"])


# ---------------------------------------------------------------- ラベルなし・入力の質・特徴量・shadow

def labelfree_section(root: Path, cfg: dict, window: int, base_end: str) -> dict:
    from src.scripts.data import drift_labelfree as LF

    years = list(range(int(cfg["baseline"]["from"][:4]), datetime.date.today().year + 1))
    p = LF.load_pred_with_card(root, cfg, years)
    if len(p) == 0:
        return {"status": "no_predictions"}
    p["week"] = M.iso_week_of(p["date"])
    base = p[(p["week"] >= cfg["baseline"]["from"]) & (p["week"] <= base_end)]
    weeks = sorted(p["week"].unique())
    last = weeks[-window:]
    cur = p[p["week"].isin(last)]
    shift = LF.dist_shift(base, cur)
    psi = shift["p_win"]["psi"]
    series = []
    for i in range(window - 1, len(weeks)):
        ws = weeks[i - window + 1: i + 1]
        m = LF.dist_metrics(p[p["week"].isin(ws)])
        m["week"] = weeks[i]
        series.append(m)
    ser = pd.DataFrame(series)
    k = float(cfg["alert"].get("gap_sigma_k", 3.0))
    bs = ser[(ser["week"] >= cfg["baseline"]["from"]) & (ser["week"] <= base_end)]
    lim = {c: M.control_limits(bs[c].to_numpy(), k) for c in ("mean_top_p", "logit_sd", "entropy_rel") if len(bs) > 20}
    lat = ser.iloc[-1] if len(ser) else None
    flags = {}
    if lat is not None and lim:
        flags = {"mean_top_p_low": bool(lat["mean_top_p"] < lim["mean_top_p"]["lcl"]), "logit_sd_low": bool(lat["logit_sd"] < lim["logit_sd"]["lcl"]),
                 "entropy_high": bool(lat["entropy_rel"] > lim["entropy_rel"]["ucl"])}
        flags["n_flagged"] = int(sum(flags.values()))
    status = "alert" if (psi >= 0.25 or flags.get("n_flagged", 0) >= 2) else ("warn" if (psi >= float(cfg["alert"].get("psi_warn", 0.10)) or flags.get("n_flagged", 0) == 1) else "ok")
    j, t, first = LF.known_ids(root, int(cfg["train"]["to"][:4]))
    latest_w = weeks[-1]
    new_ids = LF.new_id_rates(p[p["week"] == latest_w], first, latest_w, j, t) if "jockey_id" in p else {}
    return {"status": status, "latest_prediction_week": latest_w, "window_weeks": last, "shift": shift, "base_metrics": LF.dist_metrics(base),
            "cur_metrics": LF.dist_metrics(cur), "series": series, "limits": lim, "flags": flags, "new_ids": new_ids}


def coverage_section(df: pd.DataFrame, cfg: dict, weeks: int = 4, load_many=None) -> dict:
    """入力データの供給（調教ページの被覆）。直近 weeks 週の出走馬について、保存記録に『そのレース日の調教の行』があるかを数える。"""
    from src.scripts.data import drift_coverage as C

    try:
        wk = M.iso_week_of(df["date"])
        last = sorted(wk.unique())[-weeks:]
        sub = df[wk.isin(last)]
        if len(sub) == 0:
            return {"status": "no_data"}
        return C.coverage_status(C.census(sub, load_many=load_many), float(cfg["alert"].get("coverage_floor", 0.90)))
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "error": str(exc)[:200]}


def coverage_by_week(df: pd.DataFrame, start: str, load_many=None) -> dict:
    """週ごとの調教被覆（バックテスト用）。"""
    from src.scripts.data import drift_coverage as C

    sub = df[M.iso_week_of(df["date"]) >= start]
    recs = (load_many or C.default_loader())(sorted(sub["horse_id"].astype(str).unique().tolist()))
    blocks = C.block_dates(recs)
    s = sub[["race_id", "horse_id", "date"]].copy()
    s["has"] = [x in blocks.get(str(h), set()) for h, x in zip(s["horse_id"], pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d"))]
    s["week"] = M.iso_week_of(s["date"])
    return s.groupby("week")["has"].mean().to_dict()


def load_dumps(dump_dir: Path, years: list[int] | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted(dump_dir.glob("*/*.parquet")):
        if years and int(f.parent.name) not in years:
            continue
        parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def top_features(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("features", []))[:n]


def feature_section(root: Path, cfg: dict, ref_year: int = 2025, since: str | None = None) -> dict:
    dump_dir = root / "data/local/inference/drift_features"
    ref = load_dumps(dump_dir, [ref_year])
    cur = load_dumps(dump_dir, [y for y in range(ref_year + 1, 2100)])
    if len(ref) == 0 or len(cur) == 0:
        return {"status": "no_dump", "note": "drift_dump_features で 2025 年(基準)と監視期間の抜き取りを作ると PSI を計算する"}
    if since:
        cur = cur[cur["day"].astype(str) >= since.replace("-", "")]
        if len(cur) == 0:
            return {"status": "no_dump_in_range"}
    al = cfg["alert"]
    feats = top_features(root / "config/drift_top_features.json", al["top_features"])
    cols = feats or [c for c in ref.columns if c not in ("race_id", "horse_id", "day") + tuple(PCOLS)]
    sh = M.feature_shift(ref, cur, cols)
    nz = sh.sort_values("psi", ascending=False)
    big = nz[nz["psi"] >= al["psi_alert"]]
    jump = nz[(nz["miss_cur"] - nz["miss_ref"]) >= al["missing_rate_jump"]]
    status = "alert" if (len(big) >= al["psi_alert_min_features"] or len(jump) > 0) else ("warn" if (nz["psi"] >= al["psi_warn"]).any() else "ok")
    return {"status": status, "ref_rows": int(len(ref)), "cur_rows": int(len(cur)), "n_features": int(len(sh)), "n_psi_ge_alert": int(len(big)),
            "n_psi_ge_warn": int((nz["psi"] >= al["psi_warn"]).sum()), "n_missing_jump": int(len(jump)),
            "top_psi": nz.head(15).round(4).to_dict("records"), "missing_jump": jump.head(15).round(4).to_dict("records")}


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, key: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """b − a の LogLoss 差（同じ馬だけ・レース単位ブートストラップ）。a, b は race_id, horse_id, y_<key>, p_<key>。"""
    col = {"win": "p_win", "top2": "p_top2", "top3": "p_top3"}[key]
    m = a.merge(b[["race_id", "horse_id", col]], on=["race_id", "horse_id"], suffixes=("_a", "_b"))
    if len(m) == 0:
        return {"races": 0}
    y = m[f"y_{key}"].to_numpy()
    dd = M.bin_ll(y, m[f"{col}_b"].to_numpy()) - M.bin_ll(y, m[f"{col}_a"].to_numpy())
    g = pd.DataFrame({"race_id": m["race_id"], "d": dd}).groupby("race_id")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return {"races": int(len(g)), "delta": float(s.sum() / n.sum()), "lo": float(np.quantile(est, 0.025)), "hi": float(np.quantile(est, 0.975))}


def shadow_section(root: Path, cfg: dict, df: pd.DataFrame, window: int = 8) -> dict:
    """v4 shadow（predictions_v3）と v5 shadow（predictions_v5）を同じ馬で突き合わせる（全期間の累積と直近窓）。df は v4 の評価表。"""
    from src.scripts.data import readiness_eval as R

    v5dir = root / cfg["prediction_sources"]["v5_shadow_dir"]
    if not v5dir.exists():
        return {"status": "no_v5_shadow"}
    p5 = R.load_preds([v5dir])
    if len(p5) == 0:
        return {"status": "no_v5_shadow"}
    p5 = p5[["race_id", "horse_id"] + PCOLS]
    v4 = df.merge(p5[["race_id", "horse_id"]], on=["race_id", "horse_id"])
    if len(v4) == 0:
        return {"status": "no_overlap", "v5_races": int(p5["race_id"].nunique())}
    v5 = v4[["race_id", "horse_id", "y_win", "y_top2", "y_top3", "p_mkt"]].merge(p5, on=["race_id", "horse_id"])
    v5["lm"], v5["lk"] = _logit(v5["p_win"]), _logit(v5["p_mkt"])
    v4 = v4.assign(lm=_logit(v4["p_win"]), lk=_logit(v4["p_mkt"]))

    def gap(x: pd.DataFrame) -> float:
        return float((M.bin_ll(x["y_win"], x["p_win"]) - M.bin_ll(x["y_win"], x["p_mkt"])).mean())

    wk = M.iso_week_of(v4["date"])
    last = sorted(wk.unique())[-window:]
    v4w = v4[wk.isin(last)]
    return {"status": "ok", "races": int(v4["race_id"].nunique()), "rows": int(len(v4)),
            "delta_v5_minus_v4": {k: paired_delta(v4, v5, k) for k in ("win", "top2", "top3")},
            "gap_v4": gap(v4), "gap_v5": gap(v5), "sd_ratio_v4": float(v4["lm"].std() / v4["lk"].std()), "sd_ratio_v5": float(v5["lm"].std() / v5["lk"].std()),
            "latest_window": {"weeks": last, "races": int(v4w["race_id"].nunique()), "delta_win": paired_delta(v4w, v5[v5["race_id"].isin(v4w["race_id"])], "win")},
            "note": "優劣の検出には約 2,000 レースが要る。区間が 0 を跨ぐ間は優劣を言わない（切替判断に使うのは readiness の基準）"}


# ---------------------------------------------------------------- 本体

def read_ack(out_dir: Path) -> dict:
    p = out_dir / "ack.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def scorecard(rep: dict) -> list[dict]:
    """緑/黄/赤のスコアカード（群ごと）。"""
    seq = rep["performance"].get("sequence") or []
    last = seq[-1] if seq else {}
    cards = [{"name": "総合（警報の段階）", "color": STAGE_COLOR.get(rep["stage"], "gray"), "detail": rep["stage"]}]

    def lamp(flag) -> str:
        return "gray" if flag is None else ("yellow" if flag else "green")

    if last:
        cards += [
            {"name": "性能: モデル−市場の勝率 LL 差", "color": lamp(last["flag_perf"]), "detail": f"{last['gap']:+.4f}（{last['gap_class']}）"},
            {"name": "較正: ECE・傾き", "color": lamp(last["flag_cal"]), "detail": f"ECE {last['ece']:.4f} / 傾き {last['slope']:.3f}"},
            {"name": "予測の尖り（先行）", "color": lamp(last["flag_dist"]), "detail": f"sd_ratio {last['sd_ratio']:.3f}"},
        ]
    cv = rep.get("coverage", {})
    cards.append({"name": "入力: 調教の被覆", "color": {"ok": "green", "alert": "red"}.get(cv.get("status"), "gray"), "detail": f"{cv.get('block', float('nan')):.2f}"})
    lf = rep.get("labelfree", {})
    cards.append({"name": "先行: 予測分布の PSI", "color": {"ok": "green", "warn": "yellow", "alert": "red"}.get(lf.get("status"), "gray"),
                  "detail": f"{lf.get('shift', {}).get('p_win', {}).get('psi', float('nan')):.3f}"})
    ft = rep.get("features", {})
    cards.append({"name": "先行: 特徴量の PSI", "color": {"ok": "green", "warn": "yellow", "alert": "red"}.get(ft.get("status"), "gray"), "detail": ft.get("status", "")})
    cards.append({"name": "shadow v5−v4", "color": "gray", "detail": rep.get("shadow_v5", {}).get("status", "")})
    return cards


def build_report(root: Path, cfg: dict, with_features: bool = True, with_shadow: bool = True, with_coverage: bool = True, asof: str | None = None,
                 cusum_from: str | None = None, df: pd.DataFrame | None = None) -> dict:
    al = cfg["alert"]
    window = int(al.get("window_weeks", 8))
    start_year = int(cfg["baseline"]["from"][:4])
    raw = df if df is not None else load_eval_frame(root, list(range(start_year, datetime.date.today().year + 1)), cfg, with_prior=False)
    d = prepare(raw)
    wk_n = d.groupby("week")["race_id"].nunique()
    partial = None
    if len(wk_n) > 4 and wk_n.iloc[-1] < 0.6 * wk_n.iloc[-9:-1].median():      # 結果がまだ揃っていない週（週の途中で実行）は判定に使わない
        partial = {"week": wk_n.index[-1], "races": int(wk_n.iloc[-1]), "typical": float(wk_n.iloc[-9:-1].median())}
        d = d[d["week"] != wk_n.index[-1]]
    rt = race_table(d)
    base_end, mon_start = cfg["baseline"]["to"], cfg["test2"]["from"]
    edges = M.quantile_edges(d[(d["week"] >= cfg["baseline"]["from"]) & (d["week"] <= base_end)]["p_win"].to_numpy(), 10)
    series = rolling_series(d, window, rt, edges)
    limits = baseline_limits(series, cfg["baseline"]["from"], base_end, float(al["gap_sigma_k"]))
    wk_base = weekly_baseline(d, cfg["baseline"]["from"], base_end)
    cov = coverage_section(d, cfg) if with_coverage else {"status": "skipped"}
    seq = evaluate_sequence(series, limits, wk_base, mon_start, al, None, cusum_from)
    last_weeks = sorted(d["week"].unique())[-window:]
    cur = d[d["week"].isin(last_weeks)]
    cum = d[d["week"] >= mon_start]
    base_rows = d[(d["week"] >= cfg["baseline"]["from"]) & (d["week"] <= base_end)]
    rep = {"asof": asof or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "usage": cfg.get("usage"), "window_weeks": window,
           "periods": {k: cfg[k] for k in ("train", "test1", "test2", "monitor", "baseline")}, "rows": int(len(d)), "latest_result_week": last_weeks[-1],
           "performance": {"limits": limits, "weekly_baseline": wk_base, "min_races_window": int(al.get("min_races_window", 150)),
                           "latest_window": {**window_metrics(cur, rt[rt["week"].isin(last_weeks)], edges), "gap_ci": M.gap_with_ci(cur, n_boot=500)},
                           "cumulative_since_monitor_start": ({**window_metrics(cum, rt[rt["week"] >= mon_start], edges), "gap_ci": M.gap_with_ci(cum, n_boot=300)} if len(cum) else {}),
                           "sequence": seq.round(6).to_dict("records"),
                           "calibration": {"window": M.calib_bins(cur["p_win"], cur["y_win"]).round(5).to_dict("records"),
                                           "window_top3": M.calib_bins(cur["p_top3"], cur["y_top3"]).round(5).to_dict("records"),
                                           "baseline": M.calib_bins(base_rows["p_win"], base_rows["y_win"]).round(5).to_dict("records")},
                           "series_baseline": series[series["week"] >= cfg["baseline"]["from"]].round(6).to_dict("records")},
           "coverage": cov, "cusum_from": cusum_from, "partial_week_excluded": partial}
    rep["strata"] = strata_fdr(d, last_weeks, base_end, cfg["baseline"]["from"], ["surface", "dist_band", "f_size", "venue", "is_new", "age_c", "month_s"])
    try:
        rep["labelfree"] = labelfree_section(root, cfg, window, base_end)
    except Exception as exc:  # noqa: BLE001
        rep["labelfree"] = {"status": "unavailable", "error": str(exc)[:200]}
    rep["features"] = feature_section(root, cfg) if with_features else {"status": "skipped"}
    rep["shadow_v5"] = shadow_section(root, cfg, raw[raw["src"] == "forward"], window) if with_shadow else {"status": "skipped"}
    stage = seq.iloc[-1]["stage"] if len(seq) else "保留"
    idx = STAGES.index(stage) if stage in STAGES else 0
    for extra in (cov.get("status") == "alert", rep["features"].get("status") == "alert", rep["labelfree"].get("status") == "alert"):
        idx = max(idx, 1) if extra else idx
    rep["stage"] = stage if (stage == "保留" and idx == 0) else STAGES[idx]
    rep["scorecard"] = scorecard(rep)
    return rep


def write_outputs(rep: dict, out_dir: Path) -> Path:
    """追記のみ: レポートは時刻つきフォルダに新規保存し（上書きしない）、history.jsonl に 1 行足す。latest.* は最新を指す写し。"""
    from src.scripts.data.drift_report import render_html

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = rep["asof"].replace("-", "").replace(":", "").replace(" ", "_")
    rd = out_dir / "reports" / stamp
    rd.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(rep, ensure_ascii=False, indent=1, default=float)
    page = render_html(rep)
    (rd / "report.json").write_text(txt, encoding="utf-8")
    (rd / "report.html").write_text(page, encoding="utf-8")
    (out_dir / "latest.json").write_text(txt, encoding="utf-8")
    (out_dir / "latest.html").write_text(page, encoding="utf-8")
    seq = rep["performance"].get("sequence") or []
    last = seq[-1] if seq else {}
    row = {"asof": rep["asof"], "stage": rep["stage"], "latest_result_week": rep.get("latest_result_week"), "gap": last.get("gap"), "gap_class": last.get("gap_class"),
           "sd_ratio": last.get("sd_ratio"), "ece": last.get("ece"), "slope": last.get("slope"), "cusum": last.get("cusum_gap"), "ewma": last.get("ewma_gap"),
           "coverage_block": rep["coverage"].get("block"), "labelfree_psi": rep.get("labelfree", {}).get("shift", {}).get("p_win", {}).get("psi"),
           "features": rep["features"].get("status"), "shadow_v5": rep["shadow_v5"].get("status"), "report": str(rd)}
    with open(out_dir / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    return rd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--no-coverage", action="store_true", help="入力データの被覆(調教ページ)の点検を省く")
    ap.add_argument("--dry-run", action="store_true", help="保存せず要約だけ出す")
    ap.add_argument("--ack-reset", default="", help="YYYY-MM-DD: 原因の対処を確認したので、この週以降で CUSUM をやり直す（ack.json に記録して終了）")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    cfg = load_cfg()
    out_dir = Path(a.out_dir) if a.out_dir else default_out_dir(root)
    if a.ack_reset:
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {"cusum_reset_from": a.ack_reset, "note": a.note, "at": datetime.datetime.now().isoformat(timespec="seconds")}
        with open(out_dir / "ack_history.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (out_dir / "ack.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        print(f"ack: CUSUM を {a.ack_reset} 以降でやり直す")
        return 0
    rep = build_report(root, cfg, with_features=not a.no_features, with_shadow=not a.no_shadow, with_coverage=not a.no_coverage,
                       cusum_from=read_ack(out_dir).get("cusum_reset_from"))
    rd = None if a.dry_run else write_outputs(rep, out_dir)
    last = (rep["performance"]["sequence"] or [{}])[-1]
    print(json.dumps({"stage": rep["stage"], "week": rep["latest_result_week"], "gap": round(last.get("gap", float("nan")), 4), "gap_class": last.get("gap_class"),
                      "sd_ratio": round(last.get("sd_ratio", float("nan")), 3), "ece": round(last.get("ece", float("nan")), 4), "coverage": rep["coverage"].get("status"),
                      "labelfree": rep.get("labelfree", {}).get("status"), "features": rep["features"].get("status"), "shadow_v5": rep["shadow_v5"].get("status"),
                      "report": str(rd) if rd else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
