"""tp-59: 凍結した候補規則の前向き追跡。

規則（2026-10-21 に凍結）: 単勝を 100 円買う。条件 = V4（predictions_v3）の勝率 × 確定単勝オッズ ≥ 1.5、かつ騎手・調教師がともに『2025 年までの勝ち数が市場（確定単勝オッズ由来の勝率）の見込みを上回る』。
python -m src.scripts.data.tp59_track freeze    # 騎手・調教師の条件表を作って config/tp59_frozen_rule.json に書く（凍結。既にあれば上書きしない）
python -m src.scripts.data.tp59_track track --from 20261022    # 凍結日以降のレースで、規則の買い目・的中・回収率（レース単位ブートストラップ区間）を集計
2027 年のレースには 2026 年までの条件表が要る（freeze --year 2027 は 2027-01 に別ファイルとして作る）。確定オッズで評価するので、購入時点（T-15 など）のオッズとは違う点に注意。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp59_lib as L

ROOT = Path(__file__).resolve().parents[3]
CFG = Path(os.environ.get("TP59_CFG", str(ROOT / "config/tp59_frozen_rule.json")))
OUT = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
PRED = ROOT / "data/local/inference/predictions_v3"
TABLES = ROOT / "data/page_reference/tables"
EV_MIN = 1.5


def residual_flags(master: pd.DataFrame, upto_year: int) -> dict:
    """upto_year 以前の全出走で、勝ち数 − 市場の見込み（Σq）> 0 の騎手・調教師の ID。"""
    h = master[(master["year"] <= upto_year) & master["jockey_id"].notna() & master["trainer_id"].notna()]
    out = {}
    for nm, key in (("jockey_pos", "jockey_id"), ("trainer_pos", "trainer_id")):
        g = (h["win"] - h["q_win"]).groupby(h[key].astype(str)).sum()
        out[nm] = sorted(g.index[g > 0].tolist())
    return out


def cmd_freeze(a) -> None:
    if CFG.exists() and not a.force:
        raise SystemExit(f"{CFG} は既にある（凍結済み）。上書きは --force")
    m = pd.read_parquet(OUT / "master.parquet")
    flags = residual_flags(m, a.year)
    cfg = {"name": "tp59_win_ev15_rider_residual", "frozen_at": "2026-10-21", "flags_upto_year": a.year, "applies_to_races_in_year": a.year + 1,
           "rule": {"bet": "tansho", "stake_yen": 100, "ev_min": EV_MIN, "ev": "p_win(V4: predictions_v3) x final win odds", "jockey_residual_positive": True, "trainer_residual_positive": True},
           "backtest": json.loads((OUT / "ev_robust.json").read_text())["pooled"] if (OUT / "ev_robust.json").exists() else None, **flags}
    CFG.parent.mkdir(parents=True, exist_ok=True)
    CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    print("frozen", {k: len(v) for k, v in flags.items()})


def payoff_win(payoff_json: str) -> dict[int, float]:
    try:
        p = json.loads(payoff_json)
    except Exception:
        return {}
    w = p.get("単勝")
    if w is None:
        return {}
    rows = w if isinstance(w, list) else [w]
    out = {}
    for r in rows:
        try:
            out[int(str(r["numbers"]).strip())] = float(r["payout"])
        except Exception:
            continue
    return out


def cmd_track(a) -> None:
    cfg = json.loads(CFG.read_text())
    jpos, tpos = set(cfg["jockey_pos"]), set(cfg["trainer_pos"])
    rows = []
    for day_dir in sorted(PRED.iterdir()):
        if not day_dir.is_dir() or day_dir.name < a.__dict__["from"]:
            continue
        year = int(day_dir.name[:4])
        if year != cfg["applies_to_races_in_year"]:
            continue
        shut = pd.read_parquet(TABLES / str(year) / "race_shutuba_flat.parquet", columns=["race_id", "horse_number", "horse_id", "jockey_id", "trainer_id"])
        odds = pd.read_parquet(TABLES / str(year) / "race_odds_flat.parquet", columns=["race_id", "horse_number", "win_odds"])
        res = pd.read_parquet(TABLES / str(year) / "race_result_flat.parquet", columns=["race_id", "horse_number", "finish_position", "payoff", "date"])
        for f in sorted(day_dir.glob("*.parquet")):
            p = pd.read_parquet(f)
            rid = str(p["race_id"].iloc[0])
            r = res[res["race_id"].astype(str) == rid]
            if r.empty:
                continue
            s = shut[shut["race_id"].astype(str) == rid]
            o = odds[odds["race_id"].astype(str) == rid]
            x = p.merge(s[["horse_id", "horse_number", "jockey_id", "trainer_id"]].astype({"horse_id": str}), on="horse_id", how="left").merge(o[["horse_number", "win_odds"]], on="horse_number", how="left")
            pay = payoff_win(r["payoff"].iloc[0])
            x["ret"] = x["horse_number"].map(lambda n: pay.get(int(n), 0.0) if pd.notna(n) else 0.0)
            x["date"] = str(r["date"].iloc[0])
            rows.append(x)
    if not rows:
        print("対象のレースなし")
        return
    d = pd.concat(rows, ignore_index=True)
    d["ev"] = d["p_win"] * d["win_odds"]
    d["j_pos"] = d["jockey_id"].astype(str).isin(jpos)
    d["t_pos"] = d["trainer_id"].astype(str).isin(tpos)
    d["bet"] = (d["ev"] >= EV_MIN) & d["j_pos"] & d["t_pos"]
    b = d[d["bet"]]
    base = d[d["ev"] >= EV_MIN]
    rep = {"from": a.__dict__["from"], "races": int(d["race_id"].nunique()), "runners": int(len(d)), "bets": int(len(b)), "hits": int((b["ret"] > 0).sum()),
           "rule": L.boot_roi(b["race_id"].to_numpy(), b["ret"].to_numpy(), 1000) if len(b) else None,
           "ev15_without_rider_condition": L.boot_roi(base["race_id"].to_numpy(), base["ret"].to_numpy(), 500) if len(base) else None,
           "profit_yen": float(b["ret"].sum() - 100.0 * len(b)), "plan_note": "区間の下限が 100% を超えるまでの目安は約 3,300 件（真の回収率が 130% の場合。1 年に約 1,700 件なので約 2 年）"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tp59_track.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(rep, ensure_ascii=False, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--year", type=int, default=2025)
    f.add_argument("--force", action="store_true")
    t = sub.add_parser("track")
    t.add_argument("--from", dest="from", default="20261022")
    a = ap.parse_args()
    {"freeze": cmd_freeze, "track": cmd_track}[a.cmd](a)


if __name__ == "__main__":
    main()
