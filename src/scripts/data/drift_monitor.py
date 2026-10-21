"""ドリフトの週次監視: モデルの勝率予測が市場に比べて劣化していないかを、基準期間の変動幅から決めた管理図で見る。

  python -m src.scripts.data.drift_monitor [--root .] [--out-dir data/local/meta/drift_monitor] [--no-features] [--no-shadow]

見るもの（すべて『報告・警報のみ』。モデルや閾値の選択・学習には使わない。期間の定義は config/drift_periods.json）:
  gap        モデルと市場(確定単勝)の勝率 LogLoss の差の 8 週移動窓。基準(2022-2025)の窓の平均 + k×シグマを 3 週続けて超えたら警報
  sharpness  モデルの勝率の対数オッズの標準偏差 ÷ 市場のそれ（予測の尖り方。結果が無くても計算できる。低下は入力の情報の欠落を疑う）
  ece        モデル勝率の ECE（10 分位）
  features   特徴量の PSI（上位 50 列）と欠損率の跳ね。drift_dump_features で抜き取った日の特徴量を、2025 年の同経路の抜き取りと比べる
  shadow_v5  v4 shadow と v5 shadow(predictions_v5/) を同じ馬で突き合わせ、LogLoss の差(v5−v4)とレース単位の区間
自動実行: KEIBA_DRIFT_MONITOR=1 のとき金曜の weekly-update の後に別プロセスで起動(drift_monitor_hook)。手動・cron は環境変数に依らず実行する。
出力: <out-dir>/latest.json / latest.html / history.jsonl / <asof>/report.json。仕様: docs/operations/drift_monitor.md
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import drift_metrics as M
from src.scripts.data.drift_data import PCOLS, load_cfg, load_eval_frame

ENV_FLAG = "KEIBA_DRIFT_MONITOR"
ENV_DIR = "KEIBA_DRIFT_MONITOR_DIR"
WINDOW_WEEKS = 8
MIN_BASELINE_WINDOWS = 20


def enabled() -> bool:
    return (os.environ.get(ENV_FLAG) or "").strip().lower() in ("1", "true", "on", "yes")


def default_out_dir(root: Path) -> Path:
    return Path(os.environ.get(ENV_DIR) or root / "data/local/meta/drift_monitor")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), M.EPS, 1 - M.EPS)
    return np.log(p / (1 - p))


def window_metrics(g: pd.DataFrame) -> dict:
    """窓内のレースの指標。"""
    if len(g) == 0:
        return {}
    lm, lk = _logit(g["p_win"].to_numpy()), _logit(g["p_mkt"].to_numpy())
    top = g.loc[g.groupby("race_id")["p_win"].idxmax()]
    return {
        "races": int(g["race_id"].nunique()), "rows": int(len(g)),
        "gap": float((M.bin_ll(g["y_win"], g["p_win"]) - M.bin_ll(g["y_win"], g["p_mkt"])).mean()),
        "ll_model": float(M.bin_ll(g["y_win"], g["p_win"]).mean()), "ll_market": float(M.bin_ll(g["y_win"], g["p_mkt"]).mean()),
        "ece": M.ece(g["p_win"], g["y_win"]), "sd_ratio": float(lm.std() / lk.std()),
        "corr": float(np.corrcoef(lm, lk)[0, 1]), "top_pick_win": float(top["y_win"].mean()),
    }


def rolling_series(df: pd.DataFrame, window: int = WINDOW_WEEKS) -> pd.DataFrame:
    """週(月曜)ごとに、その週を末尾とする window 週の窓の指標。窓が満たない最初の週は除く。"""
    d = df.assign(week=M.iso_week_of(df["date"]))
    weeks = sorted(d["week"].unique())
    rows = []
    for i in range(window - 1, len(weeks)):
        sub = d[d["week"].isin(weeks[i - window + 1: i + 1])]
        m = window_metrics(sub)
        wk = window_metrics(d[d["week"] == weeks[i]])
        m.update({"week": weeks[i], "week_races": wk.get("races", 0), "week_gap": wk.get("gap", float("nan")), "week_sd_ratio": wk.get("sd_ratio", float("nan"))})
        rows.append(m)
    return pd.DataFrame(rows)


def baseline_limits(series: pd.DataFrame, start: str, end: str, k: float) -> dict:
    s = series[(series["week"] >= start) & (series["week"] <= end)]
    if len(s) < MIN_BASELINE_WINDOWS:
        return {}
    out = {}
    for key in ("gap", "sd_ratio", "ece", "corr"):
        out[key] = M.control_limits(s[key].to_numpy(), k)
    return out


def weekly_gap_sigma(df: pd.DataFrame, start: str, end: str) -> dict:
    """基準期間の『週ごとの差』の平均と標準偏差（CUSUM の基準）。週のレース数が少ない週は除く。"""
    d = df.assign(week=M.iso_week_of(df["date"]))
    d = d[(d["week"] >= start) & (d["week"] <= end)]
    rows = [{"week": w, **window_metrics(g)} for w, g in d.groupby("week")]
    t = pd.DataFrame(rows)
    t = t[t["races"] >= 20]
    out = {}
    for key in ("gap", "sd_ratio"):
        out[key] = {"mean": float(t[key].mean()), "sd": float(t[key].std(ddof=1)), "n": int(len(t))}
    return out


def evaluate_alerts(series: pd.DataFrame, limits: dict, start: str, consecutive: int, weekly: dict | None = None,
                    cusum_k: float = 0.5, cusum_h: float = 5.0, cusum_from: str | None = None) -> dict:
    """監視期間(start 以降)の窓について、管理限界を超えた週を数え、続けて超えた週数と警報を出す。
    警報は (a) 8 週窓が限界を consecutive 週続けて超える（遅いが誤報が少ない）か、(b) 週次の差が CUSUM の h を超える（数週で検出）。"""
    cur = series[series["week"] >= start].reset_index(drop=True)
    if len(cur) == 0 or not limits:
        return {"status": "no_data" if len(cur) == 0 else "no_baseline", "triggers": {}}
    rules = {"gap": ("gap", ">", "ucl"), "sharpness": ("sd_ratio", "<", "lcl"), "ece": ("ece", ">", "ucl")}
    trig = {}
    for name, (col, op, lim) in rules.items():
        lv = limits[col][lim]
        flags = [bool(v > lv) if op == ">" else bool(v < lv) for v in cur[col]]
        run = M.run_length(flags)
        max_run, r = 0, 0
        first_alert = None
        for w, f in zip(cur["week"], flags):
            r = r + 1 if f else 0
            max_run = max(max_run, r)
            if first_alert is None and r >= consecutive:
                first_alert = w
        t = {"limit": lv, "latest": float(cur[col].iloc[-1]), "run_weeks": run, "max_run_weeks": max_run, "weeks_over": int(sum(flags)),
             "weeks": int(len(flags)), "alert": run >= consecutive, "warn": run >= 1, "first_alert_week": first_alert,
             "first_over_week": next((w for w, f in zip(cur["week"], flags) if f), None)}
        if weekly and col in weekly:
            direction = "up" if op == ">" else "down"
            cc = cur[cur["week"] >= cusum_from].reset_index(drop=True) if cusum_from else cur      # 原因を直したあと ack で CUSUM をやり直す
            cs = M.cusum(cc["week_" + col].tolist() if ("week_" + col) in cc.columns else cc[col].tolist(), weekly[col]["mean"], weekly[col]["sd"],
                         cusum_k, cusum_h, direction)
            t["cusum_latest"], t["cusum_h"] = cs["latest"], cusum_h
            t["cusum_first_alarm_week"] = cc["week"].iloc[cs["first_alarm_index"]] if cs["first_alarm_index"] is not None else None
            t["cusum_alarm_now"] = cs["alarm_now"]
            t["alert"] = bool(t["alert"] or cs["alarm_now"])
        trig[name] = t
    status = "alert" if any(t["alert"] for t in trig.values()) else ("warn" if any(t["warn"] for t in trig.values()) else "ok")
    return {"status": status, "triggers": trig}


# ---------------------------------------------------------------- 特徴量

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
    d = json.loads(path.read_text(encoding="utf-8"))
    return list(d.get("features", []))[:n]


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
    sh["rank"] = range(1, len(sh) + 1) if feats else np.nan
    nz = sh.sort_values("psi", ascending=False)
    big = nz[nz["psi"] >= al["psi_alert"]]
    jump = nz[(nz["miss_cur"] - nz["miss_ref"]) >= al["missing_rate_jump"]]
    status = "alert" if (len(big) >= al["psi_alert_min_features"] or len(jump) > 0) else ("warn" if (nz["psi"] >= al["psi_warn"]).any() else "ok")
    return {"status": status, "ref_rows": int(len(ref)), "cur_rows": int(len(cur)), "n_features": int(len(sh)),
            "n_psi_ge_alert": int(len(big)), "n_psi_ge_warn": int((nz["psi"] >= al["psi_warn"]).sum()), "n_missing_jump": int(len(jump)),
            "top_psi": nz.head(15).round(4).to_dict("records"), "missing_jump": jump.head(15).round(4).to_dict("records")}


# ---------------------------------------------------------------- v4 と v5 の shadow の並べ比較

def paired_delta(a: pd.DataFrame, b: pd.DataFrame, key: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """b − a の LogLoss 差（同じ馬だけ・レース単位ブートストラップ）。a, b は race_id, horse_id, y_<key>, p_<key>。"""
    col = {"win": "p_win", "top2": "p_top2", "top3": "p_top3"}[key]
    m = a.merge(b[["race_id", "horse_id", col]], on=["race_id", "horse_id"], suffixes=("_a", "_b"))
    if len(m) == 0:
        return {"races": 0}
    y = m[f"y_{key}"].to_numpy()
    d = M.bin_ll(y, m[f"{col}_b"].to_numpy()) - M.bin_ll(y, m[f"{col}_a"].to_numpy())
    g = pd.DataFrame({"race_id": m["race_id"], "d": d}).groupby("race_id")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return {"races": int(len(g)), "delta": float(s.sum() / n.sum()), "lo": float(np.quantile(est, 0.025)), "hi": float(np.quantile(est, 0.975))}


def shadow_section(root: Path, cfg: dict, df: pd.DataFrame) -> dict:
    """df は v4 の評価表(forward)。predictions_v5/ の確率を同じ馬で突き合わせる。"""
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
    out = {"status": "ok", "races": int(v4["race_id"].nunique()), "rows": int(len(v4)),
           "delta_v5_minus_v4": {k: paired_delta(v4, v5, k) for k in ("win", "top2", "top3")},
           "gap_v4": window_metrics(v4).get("gap"), "gap_v5": window_metrics(v5)["gap"],
           "sd_ratio_v4": window_metrics(v4).get("sd_ratio"), "sd_ratio_v5": window_metrics(v5)["sd_ratio"]}
    out["note"] = "優劣の検出には約 2,000 レースが要る。区間が 0 を跨ぐ間は優劣を言わない（切替判断に使うのは readiness の基準）"
    return out


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


# ---------------------------------------------------------------- 出力

def svg_line(xs: list[str], series: dict[str, list[float]], limits: dict[str, float] | None = None, w: int = 640, h: int = 170) -> str:
    vals = [v for s in series.values() for v in s if v is not None and not np.isnan(v)] + list((limits or {}).values())
    if not vals or not xs:
        return ""
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1e-3
    lo, hi = lo - pad, hi + pad
    sx = lambda i: 40 + (w - 50) * i / max(len(xs) - 1, 1)  # noqa: E731
    sy = lambda v: h - 20 - (h - 35) * (v - lo) / (hi - lo)  # noqa: E731
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    parts = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border:1px solid #ddd">']
    for name, lv in (limits or {}).items():
        parts.append(f'<line x1="40" x2="{w - 10}" y1="{sy(lv):.1f}" y2="{sy(lv):.1f}" stroke="#d62728" stroke-dasharray="4 3"/>'
                     f'<text x="{w - 12}" y="{sy(lv) - 3:.1f}" font-size="10" text-anchor="end" fill="#d62728">{html.escape(name)} {lv:.4f}</text>')
    for j, (name, s) in enumerate(series.items()):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(s) if v is not None and not np.isnan(v))
        parts.append(f'<polyline fill="none" stroke="{colors[j % 3]}" stroke-width="1.6" points="{pts}"/>')
    parts.append(f'<text x="40" y="{h - 5}" font-size="10">{html.escape(xs[0])}</text><text x="{w - 10}" y="{h - 5}" font-size="10" text-anchor="end">{html.escape(xs[-1])}</text>')
    parts.append(f'<text x="4" y="{sy(hi - pad):.0f}" font-size="10">{hi - pad:.3f}</text><text x="4" y="{sy(lo + pad):.0f}" font-size="10">{lo + pad:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(rep: dict) -> str:
    st = rep["alerts"]["status"]
    color = {"ok": "#2e7d32", "warn": "#ef6c00", "alert": "#c62828"}.get(st, "#555")
    s = rep["series"]
    xs = [r["week"] for r in s]
    lim = rep["baseline"]
    parts = [f"<html><head><meta charset='utf-8'><title>ドリフト監視</title><style>body{{font-family:sans-serif;margin:20px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}</style></head><body>",
             f"<h1>ドリフト監視 <span style='color:{color}'>[{st}]</span></h1>",
             f"<p>作成: {html.escape(rep['asof'])} / 窓: {rep['window_weeks']} 週 / 基準期間: {html.escape(rep['baseline_period'])}。"
             "指標は報告・警報のみで、モデルや閾値の選択・学習には使わない（2026 年はテストセット②）。</p>"]
    if s and lim:
        parts += ["<h2>モデル − 市場 の勝率 LogLoss 差（8 週窓）</h2>", svg_line(xs, {"gap": [r["gap"] for r in s]}, {"UCL": lim["gap"]["ucl"], "mean": lim["gap"]["mean"]}),
                  "<h2>予測の尖り（モデル ÷ 市場 の対数オッズ SD）</h2>", svg_line(xs, {"sd_ratio": [r["sd_ratio"] for r in s]}, {"LCL": lim["sd_ratio"]["lcl"], "mean": lim["sd_ratio"]["mean"]}),
                  "<h2>ECE（モデル勝率）</h2>", svg_line(xs, {"ece": [r["ece"] for r in s]}, {"UCL": lim["ece"]["ucl"]})]
    parts.append("<h2>警報の条件と現在</h2><table><tr><th>指標</th><th>限界</th><th>最新</th><th>続けて超えた週</th><th>超えた週/全週</th><th>最初に超えた週</th><th>最初の警報週(窓)</th><th>CUSUM(最新/h)</th><th>最初のCUSUM警報週</th><th>警報</th></tr>")
    for name, t in rep["alerts"].get("triggers", {}).items():
        parts.append(f"<tr><td>{name}</td><td>{t['limit']:.4f}</td><td>{t['latest']:.4f}</td><td>{t['run_weeks']}</td><td>{t['weeks_over']}/{t['weeks']}</td><td>{t['first_over_week'] or '—'}</td><td>{t.get('first_alert_week') or '—'}</td><td>{t.get('cusum_latest', float('nan')):.2f}/{t.get('cusum_h', float('nan')):.1f}</td><td>{t.get('cusum_first_alarm_week') or '—'}</td><td>{'警報' if t['alert'] else ('注意' if t['warn'] else '—')}</td></tr>")
    parts.append("</table>")
    f = rep.get("features", {})
    parts.append(f"<h2>特徴量（PSI 上位）: {f.get('status')}</h2>")
    if f.get("top_psi"):
        parts.append("<table><tr><th>列</th><th>PSI</th><th>KS</th><th>欠損(基準)</th><th>欠損(今回)</th></tr>" + "".join(
            f"<tr><td>{html.escape(str(r['feature']))}</td><td>{r['psi']:.3f}</td><td>{r['ks']:.3f}</td><td>{r['miss_ref']:.3f}</td><td>{r['miss_cur']:.3f}</td></tr>" for r in f["top_psi"]) + "</table>")
    cv = rep.get("coverage", {})
    parts.append(f"<h2>入力データの供給（調教ページの被覆）: {cv.get('status')}</h2>")
    if cv.get("table"):
        parts.append("<p class='muted'>学習側(2020-2025)は約 100%。低いと tl_・wc_ が欠損のまま予測され、予測が平らになる。</p><table><tr><th>月</th><th>行数</th><th>調教の行がある率</th><th>記録がある率</th></tr>" + "".join(
            f"<tr><td>{r['ym']}</td><td>{r['rows']}</td><td>{r['block']:.3f}</td><td>{r['rec']:.3f}</td></tr>" for r in cv["table"]) + "</table>")
    sh = rep.get("shadow_v5", {})
    parts.append(f"<h2>v4 と v5 の shadow: {sh.get('status')}</h2>")
    if sh.get("status") == "ok":
        parts.append("<table><tr><th>指標</th><th>差(v5−v4)</th><th>95%区間</th><th>レース数</th></tr>" + "".join(
            f"<tr><td>{k}</td><td>{v.get('delta', float('nan')):+.5f}</td><td>[{v.get('lo', float('nan')):+.5f}, {v.get('hi', float('nan')):+.5f}]</td><td>{v.get('races')}</td></tr>"
            for k, v in sh["delta_v5_minus_v4"].items()) + "</table>")
    parts.append("</body></html>")
    return "".join(parts)


def read_ack(out_dir: Path) -> dict:
    p = out_dir / "ack.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build_report(root: Path, cfg: dict, with_features: bool = True, with_shadow: bool = True, asof: str | None = None,
                 cusum_from: str | None = None, with_coverage: bool = True) -> dict:
    al = cfg["alert"]
    start_year = int(cfg["baseline"]["from"][:4])
    end_year = datetime.date.today().year
    df = load_eval_frame(root, list(range(start_year, end_year + 1)), cfg, with_prior=False)
    window = int(al.get("window_weeks", WINDOW_WEEKS))
    series = rolling_series(df, window)
    base_end = cfg["baseline"]["to"]
    mon_start = cfg["test2"]["from"]
    limits = baseline_limits(series, cfg["baseline"]["from"], base_end, al["gap_sigma_k"])
    weekly = weekly_gap_sigma(df, cfg["baseline"]["from"], base_end)
    alerts = evaluate_alerts(series, limits, mon_start, al["gap_consecutive_weeks"], {"gap": weekly["gap"], "sd_ratio": weekly["sd_ratio"]},
                             float(al.get("cusum_k", 0.5)), float(al.get("cusum_h", 5.0)), cusum_from)
    wk_all = M.iso_week_of(df["date"])
    last_weeks = sorted(wk_all.unique())[-window:]
    latest_win = df[wk_all.isin(last_weeks)]
    cal = M.calib_bins(latest_win["p_win"], latest_win["y_win"]).round(5).to_dict("records") if len(latest_win) else []
    rep = {"asof": asof or datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "window_weeks": window,
           "baseline_period": f"{cfg['baseline']['from']}〜{base_end}", "monitor_from": mon_start, "usage": cfg.get("usage"),
           "cusum_from": cusum_from, "baseline": limits, "baseline_weekly": weekly, "alerts": alerts, "rows": int(len(df)),
           "series": series[series["week"] >= cfg["baseline"]["from"]].round(6).to_dict("records"), "calibration_latest_window": cal}
    fwd = df[df["src"] == "forward"]
    rep["features"] = feature_section(root, cfg) if with_features else {"status": "skipped"}
    rep["shadow_v5"] = shadow_section(root, cfg, fwd) if with_shadow else {"status": "skipped"}
    rep["coverage"] = coverage_section(df, cfg) if with_coverage else {"status": "skipped"}
    st = [alerts["status"], rep["features"].get("status"), rep["coverage"].get("status")]
    rep["status"] = "alert" if "alert" in st else ("warn" if "warn" in st else alerts["status"])
    rep["alerts"]["status_overall"] = rep["status"]
    return rep


def write_outputs(rep: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    day = rep["asof"][:10]
    (out_dir / day).mkdir(exist_ok=True)
    txt = json.dumps(rep, ensure_ascii=False, indent=1, default=float)
    (out_dir / day / "report.json").write_text(txt, encoding="utf-8")
    (out_dir / "latest.json").write_text(txt, encoding="utf-8")
    page = render_html({**rep, "alerts": {**rep["alerts"], "status": rep["status"]}})
    (out_dir / "latest.html").write_text(page, encoding="utf-8")
    (out_dir / day / "report.html").write_text(page, encoding="utf-8")
    last = rep["series"][-1] if rep["series"] else {}
    row = {"asof": rep["asof"], "status": rep["status"], "gap": last.get("gap"), "sd_ratio": last.get("sd_ratio"), "ece": last.get("ece"),
           "triggers": {k: {"run_weeks": v["run_weeks"], "alert": v["alert"]} for k, v in rep["alerts"].get("triggers", {}).items()},
           "features": rep["features"].get("status"), "shadow_v5": rep["shadow_v5"].get("status"), "coverage": rep["coverage"].get("status"),
           "coverage_block": rep["coverage"].get("block")}
    with open(out_dir / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


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
        (out_dir / "ack.json").write_text(json.dumps({"cusum_reset_from": a.ack_reset, "note": a.note, "at": datetime.datetime.now().isoformat(timespec="seconds")},
                                                     ensure_ascii=False), encoding="utf-8")
        print(f"ack: CUSUM を {a.ack_reset} 以降でやり直す")
        return 0
    rep = build_report(root, cfg, with_features=not a.no_features, with_shadow=not a.no_shadow, with_coverage=not a.no_coverage, cusum_from=read_ack(out_dir).get("cusum_reset_from"))
    if not a.dry_run:
        write_outputs(rep, out_dir)
    print(json.dumps({"status": rep["status"], "alerts": {k: {"latest": round(v["latest"], 4), "limit": round(v["limit"], 4), "run_weeks": v["run_weeks"]}
                                                     for k, v in rep["alerts"].get("triggers", {}).items()},
                      "features": rep["features"].get("status"), "shadow_v5": rep["shadow_v5"].get("status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
