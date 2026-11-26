"""トラックバイアスの CLI（費用 0 円・手元の表だけ。nice -n 19 で流す）。

  python -m src.research.track_bias calibrate                  # 2016〜24 だけで k と校正を決め、config/track_bias/calibration.json に書く
  python -m src.research.track_bias build-all [--out DIR]      # 全期間（2016〜最新）の日の名札を作る（既定 data/local/research/track_bias/）
  python -m src.research.track_bias day --date 2026-11-22 --venue 東京   # その日・その場の芝・ダートの判定とレースごとの根拠（JSON）
  python -m src.research.track_bias html                       # docs/html/operations/final/track_bias.html と track_bias_labels.html

開催日の夜に新しい日を足すときは build-all を流し直す（as-of なので過去の日の値は変わらない。1 回 30 秒ほど）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from src.research.track_bias import baseline as B
from src.research.track_bias import data as D
from src.research.track_bias import day as Y

CAL_PATH = D.REPO / "config/track_bias/calibration.json"
OUT_DIR = D.REPO / "data/local/research/track_bias"
LEDGER_FROM = "2022-01-01"          # これより前は台帳の「開発」の期間（記録しない）


def build_races(ks: dict | None = None):
    h = D.horse_values(D.load_horses())
    R = D.race_values(h)
    if ks is None:
        ks = {}
        for c in D.METRICS:
            ks[c], _ = B.choose_k(R, c)
        for c in D.AUX:
            ks[c] = ks["front_early"]
    R = B.add_baseline(R, D.METRICS + D.AUX, ks)
    return h, R, ks


def load_cal(path: Path = CAL_PATH) -> dict:
    if not path.exists():
        sys.exit(f"校正が無い: {path}（先に calibrate を流す）")
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_calibrate(a) -> int:
    """2016〜24 だけを使う（k は 2018〜24 の as-of の誤差・校正は 2017〜24 の日）。2025・26 年は入れない。"""
    from src.research.track_bias import tb_ledger
    with tb_ledger.open_token((LEDGER_FROM, "2024-12-31"), _code_version(), "k と日の z の校正（μ・s）を決める。2016〜24 だけ", purpose="select") as tok:
        out = _calibrate_inner()
        tok.finish(CAL_PATH)
    print(json.dumps({"k": out["k"], "persist": out["persist"]}, ensure_ascii=False))
    return 0


def _calibrate_inner() -> dict:
    h = D.horse_values(D.load_horses(range(D.FIRST_YEAR, 2025)))
    R = D.race_values(h)
    ks, mses = {}, {}
    for c in D.METRICS:
        ks[c], mses[c] = B.choose_k(R, c)
    for c in D.AUX:
        ks[c] = ks["front_early"]
    R = B.add_baseline(R, D.METRICS + D.AUX, ks)
    days = Y.day_raw(R)
    cal = Y.calibrate(days)
    # 日の持続: 同じ場・芝ダで、次の開催日（1〜2 日後）の z_raw との相関（土日の続き）
    persist = {}
    for c in (Y.FRONT_MAIN, Y.INNER_MAIN):
        d = days[(days["n_races"] >= Y.MIN_RACES) & (days["date"] >= Y.CAL_FROM)].copy()
        d["dt"] = pd.to_datetime(d["date"])
        d = d.sort_values(["venue", "surf", "dt"])
        nxt = d.groupby(["venue", "surf"])[[f"zr_{c}", "dt"]].shift(-1)
        ok = (nxt["dt"] - d["dt"]).dt.days.between(1, 2)
        persist[c] = {"corr_next_day": float(d.loc[ok, f"zr_{c}"].corr(nxt.loc[ok, f"zr_{c}"])), "pairs": int(ok.sum())}
    out = {"made": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"), "data_to": "2024-12-31",
           "k": ks, "k_mse": {c: {str(k): v for k, v in m.items()} for c, m in mses.items()},
           "persist": persist, **cal}
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAL_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def _code_version() -> str:
    """道具のコード（掃除・S の定義・基準線・区分）の hash。"""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        if p.name not in ("report.py", "tb_ledger.py"):
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _version(cal_path: Path) -> str:
    return f"code-{_code_version()}-cal-" + hashlib.sha256(cal_path.read_bytes()).hexdigest()[:12]


def build_all(out: Path, cal_path: Path = CAL_PATH) -> dict:
    cal = load_cal(cal_path)
    h, R, _ = build_races(cal["k"])
    last = str(R["date"].max())
    if last < LEDGER_FROM:
        return _write_all(h, R, cal, out, None)
    from src.research.track_bias import tb_ledger
    with tb_ledger.open_token((LEDGER_FROM, last), _version(cal_path),
                              "記述（トラックバイアスの日の名札）。買い方・採否・選び直しに使わない。設計と校正は 2016〜24 で固定") as tok:
        return _write_all(h, R, cal, out, tok)


def _write_all(h, R, cal: dict, out: Path, tok) -> dict:
    last = str(R["date"].max())
    days = Y.apply(Y.day_raw(R), cal)
    contrib = Y.race_contrib(R, days, cal)
    segs = Y.segments(R, cal)
    out.mkdir(parents=True, exist_ok=True)
    days.to_parquet(out / "daily_labels.parquet", index=False)
    contrib.to_parquet(out / "race_metrics.parquet", index=False)
    h.drop(columns=["corner_passing"]).to_parquet(out / "horses.parquet", index=False)
    segs.to_parquet(out / "segments.parquet", index=False)
    cols = ["date", "venue", "surf", "label", "front_label", "inner_label", "inner_provisional", "n_races",
            f"z_{Y.FRONT_MAIN}", f"lo_{Y.FRONT_MAIN}", f"hi_{Y.FRONT_MAIN}", "z_front_4c",
            f"z_{Y.INNER_MAIN}", f"lo_{Y.INNER_MAIN}", f"hi_{Y.INNER_MAIN}", "z_inner_a", "z_inner_b"]
    for y, g in days.groupby(days["date"].str[:4]):
        g[cols].to_csv(out / f"daily_labels_{y}.csv", index=False, encoding="utf-8")
    res = {"days": int(len(days)), "races": int(len(R)), "from": str(R["date"].min()), "to": last, "out": str(out)}
    if tok is not None:
        res["ledger"] = tok.summary()
        tok.finish(out / "daily_labels.parquet")
    return res


def cmd_build_all(a) -> int:
    print(json.dumps(build_all(Path(a.out)), ensure_ascii=False, default=str))
    return 0


def day_report(date: str, venue: str, out: Path = OUT_DIR) -> dict:
    """その日・その場の芝・ダートの判定と、レースごとの根拠（build-all の出力から）。"""
    days = pd.read_parquet(out / "daily_labels.parquet")
    rm = pd.read_parquet(out / "race_metrics.parquet")
    hs = pd.read_parquet(out / "horses.parquet")
    sg = pd.read_parquet(out / "segments.parquet")
    res = {"date": date, "venue": venue, "surfaces": {}}
    for _, d in days[(days["date"] == date) & (days["venue"] == venue)].iterrows():
        s = d["surf"]
        races = rm[(rm["date"] == date) & (rm["venue"] == venue) & (rm["surf"] == s)].sort_values("round")
        res["surfaces"][s] = {
            "label": d["label"], "front": d["front_label"], "inner": d["inner_label"] + ("（仮・案 C）" if d["inner_provisional"] else ""),
            "n_races": int(d["n_races"]),
            "z": {m: round(float(d[f"z_{m}"]), 2) for m in D.METRICS},
            "D_front_80": [round(float(d[f"lo_{Y.FRONT_MAIN}"]), 2), round(float(d[f"hi_{Y.FRONT_MAIN}"]), 2)],
            "segments": sg[(sg["date"] == date) & (sg["venue"] == venue) & (sg["surf"] == s)][["segment", "n_races", "z"]].round(2).to_dict("records"),
            "races": [{"round": int(r["round"]), "race_id": r["race_id"], "distance": int(r["distance"]), "class": r["grade"],
                       "cond": r["track_condition"], "N": int(r["N"]),
                       "S-E_front": round(float(r[f"SE_{Y.FRONT_MAIN}"]), 3), "z_contrib_front": round(float(r[f"zc_{Y.FRONT_MAIN}"]), 3),
                       "S-E_front_4c": round(float(r["SE_front_4c"]), 3), "S-E_inner_c": round(float(r[f"SE_{Y.INNER_MAIN}"]), 3)}
                      for _, r in races.iterrows()],
        }
        res["surfaces"][s]["n_horses"] = int(hs["race_id"].isin(races["race_id"]).sum())
    return res


def cmd_day(a) -> int:
    print(json.dumps(day_report(a.date, a.venue, Path(a.out)), ensure_ascii=False, indent=1, default=str))
    return 0


def cmd_html(a) -> int:
    from src.research.track_bias import report
    print(report.write(Path(a.out)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.research.track_bias")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    for name in ("build-all", "html"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT_DIR))
    p = sub.add_parser("day")
    p.add_argument("--date", required=True)
    p.add_argument("--venue", required=True)
    p.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    return {"calibrate": cmd_calibrate, "build-all": cmd_build_all, "day": cmd_day, "html": cmd_html}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
