"""T-60 の段（血統予想家）。5 分おきの軽い cron から呼ぶ。手元のファイルだけを読み、対象が無ければすぐ終わる。

    nice -n 19 /opt/conda/bin/python -m src.research.pedigree_persona.t60 [--date 2026-12-06]

- 対象: 前の夜の段が置いた欄が pending のレースで、T-60 の単勝オッズの手元の写し（odds_t60/<YYYYMMDD>/<race_id>.json、
  raceday-runner が書く）が届いたもの。写しの無いまま発走の時刻を過ぎたレースは unavailable（一覧を出さない）にする。
- 処理: T-60 の時点で単勝オッズのある馬（取消を外す）でレースの中を揃え直し、凍結の規則で 0〜2 頭を選び、理由の文を付ける。
- 同時に 2 本走らない（flock）。GCS を読まない。成績は計算しない。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime
from pathlib import Path

import logging

from src.research.pedigree_persona import store
from src.research.pedigree_persona.store import JST

logger = logging.getLogger(__name__)

LOCK = Path(os.environ.get("KEIBA_PEDIGREE_PERSONA_LOCK") or "/tmp/keiba_pedigree_persona_t60.lock")


def odds_copy(day8: str, race_id: str) -> dict | None:
    base = os.environ.get("KEIBA_ODDS_T60_DIR")
    if base:
        p = Path(base) / day8 / f"{race_id}.json"
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    from src.pipeline.inference import odds_t60_copy as oc

    return oc.read_copy(day8, race_id)


def past_start(day: str, start_time: str, now: datetime) -> bool:
    try:
        hh, mm = (int(x) for x in start_time.split(":")[:2])
    except ValueError:
        hh, mm = 23, 59                           # 発走の時刻が分からなければ、その日の終わりで締める
    y, m, d = (int(x) for x in day.split("-"))
    return now >= datetime(y, m, d, hh, mm, tzinfo=JST)


def finalize(g, copy: dict, R, fz: dict, old: dict, now: datetime) -> dict:
    """1 レースの欄を T-60 の値で確定する。g はそのレースの生の値（前の夜の段）。

    人気の順は、写しの全頭（オッズ > 0）で付けてから、生の値のある馬に絞る（閾値の母集団と同じく、全出走馬の中の順。critic 6）。
    有効なオッズが 6 頭未満の写し（発売前など）は、一覧を出さない（unavailable。critic 7）。"""
    import numpy as np

    from src.research.pedigree_persona import columns as pc

    ents = [(int(_num(e.get("horse_number"))), _num(e.get("win_odds"))) for e in copy.get("entries") or []]
    ents = [(h, o) for h, o in ents if h > 0 and o > 0]
    fig = dict(old["figure"])
    surface = str(fig["surface"])
    base = old | {"made_at": now.isoformat(timespec="seconds"), "t60_snapshot_at": copy.get("snapshot_at"), "visible": store.visible()}
    if len(ents) < R.POP_MIN:
        return base | {"figure": fig | {"picks_state": "unavailable", "picks": []}}
    hn = np.array([h for h, _ in ents])
    rank_of = dict(zip(hn.tolist(), R.pop_rank(np.zeros(len(hn)), np.array([o for _, o in ents]), hn).tolist()))
    g = g[pd_num(g["horse_number"]).isin(rank_of)].copy()                  # T-60 の時点の出走馬（取消を外す）
    g["horse_number"] = pd_num(g["horse_number"]).astype(int)
    fig["facts"] = R.facts(surface, g["FD"], g["layoff_y"], g["career"], False)
    if R.banned_in(fig["facts"]):
        raise ValueError(f"禁止の語が入った: {fig['facts']}")
    picks = []
    if len(g):
        z = R.score(g[R.SCORE_COL[surface]].to_numpy(), g["race_id"].to_numpy(), g["no_pedigree"].to_numpy(), fz["sd"][surface])
        rank = g["horse_number"].map(rank_of).to_numpy()
        for i in R.pick(z, rank, fz["threshold"][surface]):
            row = g.iloc[int(i)]
            top = json.loads(row["anc_top"] or "[]")[:2]
            names = pc.ancestor_names(str(row["horse_id"]), {t[0] for t in top})
            picks.append({"horse_number": int(row["horse_number"]), "horse_name": str(row["horse_name"] or ""),
                          "reason": R.reason(surface, [names.get(t[0], "") for t in top], bool(row["FD"]), int(row["sire_runs"]))})
    return base | {"figure": fig | {"picks_state": "ready" if picks else "none", "picks": picks}}


def pd_num(s):
    import pandas as pd

    return pd.to_numeric(s, errors="coerce")


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="対象の開催日（既定は JST の今日）")
    a = ap.parse_args()
    now = datetime.now(JST)
    day = a.date or now.date().isoformat()
    raw = store.raw_dir(day) / "raw.parquet"
    if not raw.exists() or not (store.raw_dir(day) / "races.json").exists():
        return                                    # 開催日でない・前の夜の段が無い: すぐ終わる
    try:
        lk = open(LOCK, "a")
    except OSError as exc:                        # 別の使用者のロックのファイルなど
        print(f"ロックを開けない: {LOCK} {exc}", flush=True)
        return
    with lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return                                # 前の回がまだ動いている
        todo = []
        try:
            ids = json.loads((store.raw_dir(day) / "races.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for rid in ids:                            # その日のレースだけを見る（置き場の全件を一覧しない）
            doc = store.read(str(rid))
            if doc and doc.get("asof") == day and (doc.get("figure") or {}).get("picks_state") == "pending":
                todo.append(doc)
        if not todo:
            return
        day8 = day.replace("-", "")
        ready = [(d, c) for d in todo if (c := odds_copy(day8, d["race_id"]))]
        late = [d for d in todo if d["race_id"] not in {x[0]["race_id"] for x in ready} and past_start(day, d.get("start_time") or "", now)]
        for d in late:
            fig = dict(d["figure"]) | {"picks_state": "unavailable", "picks": []}
            store.write(d["race_id"], d | {"figure": fig, "made_at": now.isoformat(timespec="seconds"), "visible": store.visible()})
        if not ready:
            return
        import pandas as pd

        from src.research.pedigree_persona import rules as R

        fz = R.load_frozen()
        R_ = pd.read_parquet(raw)
        R_["race_id"] = R_["race_id"].astype(str)
        failed = 0
        for d, c in ready:
            g = R_[R_["race_id"] == d["race_id"]]
            try:
                doc = finalize(g, c, R, fz, d, now)
            except Exception:                     # 1 レースの失敗で後ろのレースを止めない（critic 4）
                logger.exception("血統予想家の T-60 の段に失敗: %s", d["race_id"])
                failed += 1
                doc = d | {"figure": dict(d["figure"]) | {"picks_state": "unavailable", "picks": []},
                           "made_at": now.isoformat(timespec="seconds"), "visible": store.visible()}
            store.write(d["race_id"], doc)
        print(json.dumps({"date": day, "ready": len(ready) - failed, "failed": failed, "unavailable": len(late)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
