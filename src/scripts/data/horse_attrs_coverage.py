"""推論が使う馬の産地（と誕生日）の埋まり具合を見張る（読み取りと data/local への書き込みだけ。GCS・GCP は読まない。費用 0 円）。

python -m src.scripts.data.horse_attrs_coverage [--days 28] [--warn 0.05]

推論の V4Features は、産地・誕生日の表（factory/self/horse_attrs.parquet）を、馬主・生産者の表（factory/conn/horse_profile.parquet）と
日次の補い（state/profile_dyn.parquet）で埋めて使う（serving_v4.fill_attrs_from_profile）。その合わせた結果で、直近の開催で走った馬
（race_result_flat）と出走予定の馬（race_shutuba_flat）のうち、産地が無い馬の割合を出す。
- 走った馬・出走予定の既走馬の欠けが --warn（既定 5%）を超えたら WARNING を出し、記録の is_warn を立てる（終了コードは 0。週次の処理を止めない）。
  出走予定の初出走の馬は、走る前には日次の補いが届かない（推論のときに読む）ので、数を出すだけで判定に使わない。
  走った馬が 0 頭（表が読めない）のときも WARNING。
- 記録: data/local/meta/horse_attrs_coverage/latest.json と history.jsonl（追記）。
欠けが増えたときに見直すもの: 馬主・生産者の表の作り直し（factory_conn_fetch）と、日次の補い（update_serving_state profile、cron の serving-state-profile）。
背景: docs/operations/retrain_trials/v6e_2026_gap_20261114.md（2026 年に新しい馬の産地が最大 48% 欠け、予測の上積みが消えていた）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("HORSE_ATTRS_COVERAGE_DIR", str(ROOT / "data/local/meta/horse_attrs_coverage")))


def serving_attrs() -> pd.DataFrame:
    """推論と同じ合わせ方の産地・誕生日の表（重い履歴は読まない）。"""
    from src.pipeline.features import factory_conn as fc
    from src.pipeline.features import serving_v4 as sv4

    at = pd.read_parquet(sv4.ATTRS_PATH).set_index("horse_id") if sv4.ATTRS_PATH.exists() else pd.DataFrame(columns=["birthplace", "birthday"])
    at.index = at.index.astype(str)
    if Path(fc.PROFILE).exists():                       # 推論と同じく、馬主・生産者の表からは学習の期間に走った馬を埋めない
        at = sv4.fill_attrs_from_profile(at, pd.read_parquet(fc.PROFILE), exclude=trained_horses())
    dyn = sv4.STATE_DIR / "profile_dyn.parquet"
    if dyn.exists():
        at = sv4.fill_attrs_from_profile(at, pd.read_parquet(dyn))
    return at


def trained_horses() -> set[str]:
    """学習の期間（FIRST_YEAR〜2025 年）に走った馬（推論の V4History.trained_horses と同じ範囲）。"""
    from src.pipeline.features import serving_v4 as sv4
    from src.pipeline.models.training_years import FIRST_YEAR
    from src.scripts.data import frozen_tracking as ft

    out: set[str] = set()
    for y in range(FIRST_YEAR, sv4.TRAIN_LAST_YEAR + 1):
        f = ft.TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            out |= set(pd.read_parquet(f, columns=["horse_id"])["horse_id"].astype(str))
    return out


def runners(days: int, today: date, tables: Path) -> pd.DataFrame:
    """直近 days 日に走った馬（確定オッズのある馬）と、今日以降の出走予定の馬（馬番 0 の登録だけ・除外の行は除く）。
    出走予定は、初めて走る馬（race_result_flat に走った行が無い馬）と、既に走ったことのある馬に分ける。"""
    since = (today - timedelta(days=days)).isoformat()
    years = range((today - timedelta(days=days)).year - 0, today.year + 2)          # 年末に翌年 1 月の出馬表も読む
    res, sh = [], []
    for y in years:
        f = tables / str(y) / "race_result_flat.parquet"
        if f.exists():
            res.append(pd.read_parquet(f, columns=["race_id", "date", "horse_id", "odds"]))
        f = tables / str(y) / "race_shutuba_flat.parquet"
        if f.exists():
            sh.append(pd.read_parquet(f, columns=["race_id", "date", "horse_id", "horse_number"]))
    out = []
    ran_ever: set[str] = set()
    for y in range(2016, today.year + 1):                                          # 既に走ったことがあるか（初出走の判定）
        f = tables / str(y) / "race_result_flat.parquet"
        if f.exists():
            r = pd.read_parquet(f, columns=["horse_id", "odds", "date"])
            r = r[(pd.to_numeric(r["odds"], errors="coerce") > 0) & (pd.to_datetime(r["date"], errors="coerce") < pd.Timestamp(today))]
            ran_ever |= set(r["horse_id"].astype(str))
    if res:
        d = pd.concat(res, ignore_index=True)
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        d = d[(pd.to_numeric(d["odds"], errors="coerce") > 0) & (d["date"] >= since) & (d["date"] < today.isoformat())]
        out.append(d.assign(kind="走った馬")[["race_id", "date", "horse_id", "kind"]])
    if sh:
        d = pd.concat(sh, ignore_index=True)
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        d = d[(pd.to_numeric(d["horse_number"], errors="coerce").fillna(0) > 0) & (d["date"] >= today.isoformat())]
        d["horse_id"] = d["horse_id"].astype(str)
        d["kind"] = d["horse_id"].map(lambda h: "出走予定（既走）" if h in ran_ever else "出走予定（初出走）")
        out.append(d[["race_id", "date", "horse_id", "kind"]])
    r = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["race_id", "date", "horse_id", "kind"])
    r["horse_id"] = r["horse_id"].astype(str)
    return r[r["horse_id"].str.len() > 0]


WARN_KINDS = ("走った馬", "出走予定（既走）")      # 初出走の馬は、走る前には日次の補いが届かない（推論のときに読む）ので、判定に使わない


def coverage(r: pd.DataFrame, at: pd.DataFrame) -> dict:
    bp = at["birthplace"] if "birthplace" in at.columns else pd.Series(dtype=object)
    have = set(at.index[bp.notna() & (bp.astype(str).str.strip() != "")])
    res = {}
    for kind, g in r.groupby("kind"):
        u = g.drop_duplicates("horse_id")
        miss = ~u["horse_id"].isin(have)
        res[kind] = {"horses": int(len(u)), "missing": int(miss.sum()), "missing_rate": float(miss.mean()) if len(u) else 0.0,
                     "missing_sample": sorted(u.loc[miss, "horse_id"])[:20]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--warn", type=float, default=0.05)
    ap.add_argument("--today", default="")
    a = ap.parse_args()
    from src.scripts.data import frozen_tracking as ft

    today = date.fromisoformat(a.today) if a.today else date.today()
    cov = coverage(runners(a.days, today, ft.TABLES), serving_attrs())
    worst = max((v["missing_rate"] for k, v in cov.items() if k in WARN_KINDS), default=0.0)
    no_data = "走った馬" not in cov or cov["走った馬"]["horses"] == 0          # 表が読めない・向き先違いを正常扱いにしない
    rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "today": today.isoformat(), "days": a.days, "warn_threshold": a.warn,
           "is_warn": bool(worst > a.warn or no_data), "no_data": bool(no_data), "worst_missing_rate": worst, "warn_kinds": list(WARN_KINDS), "by_kind": cov}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / "history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({k: v for k, v in rec.items() if k != "by_kind"} | {"by_kind": {k: {kk: vv for kk, vv in v.items() if kk != "missing_sample"} for k, v in cov.items()}},
                            ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in ("today", "is_warn", "worst_missing_rate")} | {"by_kind": {k: f"{v['missing']}/{v['horses']}" for k, v in cov.items()}}, ensure_ascii=False))
    if no_data:
        print(f"WARNING: 直近 {a.days} 日に走った馬が 0 頭（結果の表が読めない・向き先が違う可能性）: {ft.TABLES}", file=sys.stderr)
    elif rec["is_warn"]:
        print(f"WARNING: 推論で産地の無い馬が {worst:.1%}（しきい値 {a.warn:.0%}、走った馬と出走予定の既走馬）。日次の補い（serving-state-profile）を確かめる", file=sys.stderr)


if __name__ == "__main__":
    main()
