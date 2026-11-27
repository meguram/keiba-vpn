"""前の夜の段（血統予想家）。翌日の出馬表の馬に、as-of の生の値と理由の材料を作り、欄を「一覧は発走の約 60 分前に出す」の状態で置く。

    nice -n 19 /opt/conda/bin/python -m src.research.pedigree_persona.eve [--date 2026-12-06]

- 手元の表（race_result_flat・race_shutuba_flat）と 5 代血統の JSON だけを読む。GCS を読まない・オッズを使わない。
- 開催日の 07:30〜17:00 には流さない（重い計算の決まり。--force で外す）。
- 前向きの記録（forward/<日付>.csv）とは別のファイルに書く（会議の 1 号の条件 4）。記録の CSV は書かない・読まない。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

from src.research.pedigree_persona import store
from src.research.pedigree_persona.store import JST


def _hhmm(v) -> str:
    """発走の時刻を「H:MM」の文字にする。欠け（NaN・空）は空の文字（critic 9）。"""
    s = "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()
    return s if len(s.split(":")) >= 2 and s.split(":")[0].isdigit() else ""


def pending_doc(race_id: str, g, R, fz: dict, meta: dict) -> dict:
    surface = str(g["surface"].iat[0])
    new_race = bool(g["new_race"].any())
    state = "not_applicable" if new_race or surface not in R.SCORE_COL else "pending"
    return R.module(race_id=race_id, surface=surface,
                    facts_text=R.facts(surface, g["FD"], g["layoff_y"], g["career"], new_race),
                    state=state, picks=[], asof=str(g["date"].iat[0]), made_at=meta["made_at"], rule_sha=fz["rules_sha256"][:16],
                    inputs=meta["inputs"]) | {"visible": store.visible(), "start_time": _hhmm(g["start_time"].iat[0])}


def main() -> None:
    import pandas as pd

    from src.research.pedigree_persona import columns as pc
    from src.research.pedigree_persona import rules as R

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="対象の開催日（既定は JST の翌日）")
    ap.add_argument("--force", action="store_true", help="開催日の 07:30〜17:00 でも流す（手で流すときだけ）")
    a = ap.parse_args()
    now = datetime.now(JST)
    day = pd.Timestamp(a.date) if a.date else pd.Timestamp((now + timedelta(days=1)).date())
    if not a.force and 7 * 60 + 30 <= now.hour * 60 + now.minute < 17 * 60:
        raise SystemExit("07:30〜17:00 は重い計算を流さない（--force で外す）")
    fz = R.load_frozen()
    if not pc.has_entries(day):
        print(json.dumps({"date": str(day.date()), "skip": "出馬表が無い（開催日でない）"}, ensure_ascii=False), flush=True)
        return
    out, meta = pc.build_day(day)
    d = store.raw_dir(str(day.date()))
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "raw.parquet", index=False)
    meta["rule_sha"] = fz["rules_sha256"][:16]
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "races.json").write_text(json.dumps(sorted(out["race_id"].astype(str).unique().tolist())), encoding="utf-8")
    n = {"pending": 0, "not_applicable": 0, "kept": 0}
    for rid, g in out.groupby("race_id"):
        old = store.read(str(rid))
        if old and (old.get("figure") or {}).get("picks_state") in ("ready", "none", "unavailable"):
            n["kept"] += 1            # T-60 の段が確定させた欄は上書きしない
            continue
        doc = pending_doc(str(rid), g, R, fz, meta)
        store.write(str(rid), doc)
        n[doc["figure"]["picks_state"]] += 1
    print(json.dumps({"date": str(day.date()), **n, "races": meta["races"], "horses": meta["horses"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
