"""候補の影のランナーのリハーサル: 過去の開催日を、当日と同じ順（本体の推論の完了 → T-60 オッズ → 期限）で再現する。

時計は進めるだけで待たない。本体の推論の完了は、本体の events/<日>.jsonl の時刻で絞る。T-60 のオッズは手元の
pre_odds.parquet（週次の pre_odds_table が作る）から、その時刻までに取れたものだけ見せる。GCS のオッズは読まない。
出力（3 つの木・X・見張りの記録）は --out の下だけに書く（本番の木・GCS には書かない）。

  python -m src.scripts.data.candidate_replay --day 20261115 --out /home/jovyan/work/retrain_out/candidate_rehearsal [--max-races 4]

1 レースごとの計算時間・メモリ・GCS の読み出しの量は、--out/launch_watch/<日>.jsonl と標準出力の要約に出る。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_JST = timezone(timedelta(hours=9))
PRE_ODDS = Path(__file__).resolve().parents[3] / "data/local/research/persona1/pre_odds.parquet"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--max-races", type=int, default=0)
    ap.add_argument("--pre-odds", default=str(PRE_ODDS))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from src.pipeline.inference import candidate_runner as cr

    cfg = cr.load_config(a.config or None)
    out = Path(a.out)
    cfg = {**cfg, "enabled": True, "publish_dir": str(out / "publish"), "watch_dir": str(out / "launch_watch"),
           "x_dir": str(out / "stage2_x"), "rss_cap_gb": cfg.get("rss_cap_gb", 0)}
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    cr.apply_env(cfg)
    if "src.pipeline.models.training_years" in sys.modules:
        raise SystemExit("training_years が先に import された")
    import pandas as pd

    po = pd.read_parquet(a.pre_odds)
    po = po[po["phase"] == "t60"].copy()
    po["race_id"] = po["race_id"].astype(str)
    po["snap"] = pd.to_datetime(po["snapshot_at"].astype(str), utc=True).dt.tz_convert(_JST)
    odds_by = {rid: g for rid, g in po.groupby("race_id")}

    ev_path = Path(cfg["main_inference_dir"]) / "events" / f"{a.day}.jsonl"
    evs = []
    for ln in ev_path.read_text(encoding="utf-8").splitlines() if ev_path.exists() else []:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("status") == "ok":
            evs.append((str(e["race_id"]), datetime.fromisoformat(e["at"]).astimezone(_JST)))

    miss = cr.preflight(cfg, a.day)
    if miss:
        print("足りないもの:", miss, file=sys.stderr)
    r = cr.Runner(cfg, a.day, replay=True)
    t_warm = time.time()
    r.warm()
    warm_sec = time.time() - t_warm
    races = [x for x in r.schedule() if x.get("post_time")]
    if a.max_races:
        keep = {x["race_id"] for x in sorted(races, key=lambda x: x["post_time"])[: a.max_races]}
        r.schedule = lambda: [x for x in races if x["race_id"] in keep]  # type: ignore[method-assign]
        races = [x for x in races if x["race_id"] in keep]
    clock = [min(x["post_time"] for x in races) - timedelta(minutes=70)]

    def now_fn():
        return clock[0]

    def sleep_fn(sec):
        clock[0] = clock[0] + timedelta(seconds=sec)

    r.main_done_fn = lambda now: {rid for rid, at in evs if at <= now}

    def odds_fn(rid, now):
        g = odds_by.get(rid)
        if g is None:
            return None
        g = g[g["snap"] <= now]
        if g.empty:
            return None
        g = g.sort_values("snap").drop_duplicates("horse_number", keep="last")
        return {"race_id": rid, "snapshot_at": g["snap"].max().isoformat(),
                "entries": [{"horse_number": int(h), "win_odds": float(o), "popularity": p} for h, o, p in
                            zip(g["horse_number"], g["win_odds"], g["popularity"])]}

    r.odds_fn = odds_fn
    r.sleep_fn = sleep_fn
    t0 = time.time()
    r.loop(now_fn=now_fn)
    w = [json.loads(ln) for ln in (out / "launch_watch" / f"{a.day}.jsonl").read_text(encoding="utf-8").splitlines()]
    dec = [x for x in w if x.get("event") == "decision"]
    v6c = [x for x in w if x.get("event") == "v6c"]
    summ = {"day": a.day, "races": len(races), "warm_sec": round(warm_sec), "loop_sec": round(time.time() - t0),
            "v6c_ok": len(v6c), "v6c_err": sum(x.get("event") == "v6c_error" for x in w),
            "v6c_sec_max": max((x["sec"] for x in v6c), default=None), "v6c_sec_median": (sorted(x["sec"] for x in v6c)[len(v6c) // 2] if v6c else None),
            "rss_gb_max": max((x.get("rss_gb", 0) for x in w), default=None), "gcs_mb": r.meter.bytes / 1e6, "gcs_reads": r.meter.reads,
            "decisions": len(dec), "buy": sum(x.get("status") == "buy" for x in dec),
            "skip_reasons": {k: sum(x.get("skip_reason") == k for x in dec) for k in {x.get("skip_reason") for x in dec}}}
    (out / f"summary_{a.day}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
