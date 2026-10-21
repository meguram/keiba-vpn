"""ドリフト監視のバックテスト: 2025〜2026 年の各週にさかのぼって監視を回し、どの週に警報が出たか、誤警報の率、検知の遅れを見る。

  python -m src.scripts.data.drift_backtest [--out-dir data/local/meta/drift_monitor] [--no-coverage]

2025 年の週は 2022-2024（OOF）だけを基準に、2026 年の週は 2022-2025 を基準に再現する（その週までの情報だけ。先読みなし）。
2025 年は『ドリフトが無かった年』とみなして誤警報の率を数える。2026 年は 6 月ごろに差が開いた（診断の章）ので、検知の遅れを見る。
窓の長さ（4/8/12 週）と限界の幅（2/2.5/3σ）の感度表は、検知力と誤警報のトレードオフを見るためのもので、本番の設定を 2026 年で選び直すものではない。
結果は <out-dir>/backtest/<時刻>/ に追記（上書きしない）。報告・警報のみ。モデルや閾値の選択には使わない。
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import pandas as pd

from src.scripts.data import drift_monitor as D
from src.scripts.data.drift_data import load_cfg, load_eval_frame


def summarize(seq: pd.DataFrame) -> dict:
    out = {}
    for label, sel in (("2025", seq["week"] < "2026-01-01"), ("2026", seq["week"] >= "2026-01-01")):
        s = seq[sel & (seq["stage"] != "保留")]
        n = len(s)
        rec = {"weeks": n}
        for st, rank in (("注意以上", 1), ("警告以上", 2), ("要対応", 3)):
            flagged = s[s["stage"].map(lambda x: D.STAGES.index(x) if x in D.STAGES else 0) >= rank]
            rec[st] = {"weeks": int(len(flagged)), "rate": float(len(flagged) / n) if n else float("nan"),
                       "first_week": flagged["week"].iloc[0] if len(flagged) else None}
        rec["有意な悪化(gap)"] = int((s["gap_class"] == "有意な悪化").sum())
        out[label] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--no-coverage", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    cfg = load_cfg()
    raw = load_eval_frame(root, list(range(int(cfg["baseline"]["from"][:4]), datetime.date.today().year + 1)), cfg, with_prior=False)
    d = D.prepare(raw)
    cov = None if a.no_coverage else D.coverage_by_week(raw, "2026-01-01")
    main_seq = D.backtest(d, cfg, coverage_by_week=cov)
    res = {"asof": datetime.datetime.now().isoformat(timespec="seconds"), "config": {k: cfg["alert"][k] for k in ("window_weeks", "gap_sigma_k", "min_races_window")},
           "summary": summarize(main_seq),
           "coverage_first_week_below_floor": next((w for w, v in sorted((cov or {}).items()) if v < cfg["alert"].get("coverage_floor", 0.9)), None),
           "weeks": main_seq[["week", "races", "gap", "gap_se", "gap_class", "sd_ratio", "ece", "slope", "ewma_gap", "ewma_flag", "cusum_gap", "flag_perf", "flag_cal", "flag_dist",
                              "flag_inputs", "stage", "baseline_end"]].round(5).to_dict("records")}
    sens = []
    for w in (4, 8, 12):
        for k in (2.0, 2.5, 3.0):
            s = summarize(D.backtest(d, cfg, window=w, k=k, coverage_by_week=cov))
            sens.append({"window_weeks": w, "sigma_k": k, "false_alarm_2025": {x: s["2025"][x]["rate"] for x in ("注意以上", "警告以上", "要対応")},
                         "first_week_2026": {x: s["2026"][x]["first_week"] for x in ("注意以上", "警告以上", "要対応")},
                         "flagged_rate_2026": {x: s["2026"][x]["rate"] for x in ("注意以上", "警告以上", "要対応")}})
    res["sensitivity"] = sens
    out_dir = (Path(a.out_dir) if a.out_dir else D.default_out_dir(root)) / "backtest" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"summary": res["summary"], "coverage_first_week_below_floor": res["coverage_first_week_below_floor"], "out": str(out_dir)}, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
