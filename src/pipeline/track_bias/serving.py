"""トラックバイアスの画面の値をホストで作る（表示のときに計算しない。Cloud Run は view.py で読むだけ）。

  nice -n 19 python -m src.pipeline.track_bias.serving build [--out DIR]

- 研究の道具（src/research/track_bias の data・baseline・day。純粋な関数）で全期間を作り、画面に要る列だけを年ごとに書く。
  校正は commit 済みの config/track_bias/calibration.json を読むだけ（変えない）。1 回 30 秒ほど（2016〜最新の表を読む）。
- 出力（data/page_reference/track_bias/。ミラーの木。書くのは一時ファイル → 置き換え）:
  - days_<年>.parquet（鍵 date・venue・surf）: 名札・前後と内外の名札・z・80% 区間・レース数。
  - races_<年>.parquet（鍵 race_id）: 前後と内外の S − E・日の z への寄与・4 角の版の S − E・1〜3 着の位置（JSON の配列）。
  - meta.json: 版（コードと校正の hash）・作った時刻・最後の日付・主の値。
- 画面に結果を出す本番の表示で、研究の評価ではない（race_level と同じ扱い）。そのため見た回数の台帳は通さない。
  研究の CLI（python -m src.research.track_bias build-all）は今のまま台帳を通す。
- ホストの開催日の後処理から流す（12/1 以降。race_level と同じ所）。費用 0 円（手元の表だけ・GCS を読まない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
REL = "data/page_reference/track_bias"
CAL_PATH = REPO / "config/track_bias/calibration.json"
DAY_COLS = ["date", "venue", "surf", "label", "front_label", "inner_label", "n_races",
            "z_front", "lo_front", "hi_front", "z_inner", "lo_inner", "hi_inner"]
RACE_COLS = ["race_id", "date", "venue", "surf", "round", "distance",
             "front_se", "front_zc", "inner_se", "inner_zc", "front4c_se", "top3"]


def code_version() -> str:
    """作りの版: 研究の data・baseline・day とこのファイル、校正の JSON の hash。"""
    h = hashlib.sha256()
    base = REPO / "src/research/track_bias"
    for p in [base / "data.py", base / "baseline.py", base / "day.py", Path(__file__), CAL_PATH]:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _num(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def _passing(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan") else str(v).strip()


def top3_rows(h: pd.DataFrame) -> dict[str, str]:
    """レースごとの 1〜3 着（同着はそのまま並べる）の JSON。通過順が無い馬は番手を None（画面は「通過なし」）。"""
    t = h[h["fin"] <= 3].sort_values(["race_id", "fin", "horse_number"])
    out: dict[str, list] = {}
    for r in t.itertuples():
        po = _passing(getattr(r, "passing_order", ""))
        out.setdefault(r.race_id, []).append({
            "pos": int(r.fin), "horse_number": int(r.horse_number),
            "popularity": None if pd.isna(r.popularity) else int(r.popularity),
            "passing_order": po or None,
            "early": int(r.pos_first) if po and pd.notna(r.pos_first) else None,
            "corner4": int(r.pos_last) if po and pd.notna(r.pos_last) else None,
            "field": int(r.N)})
    return {k: json.dumps(v, ensure_ascii=False) for k, v in out.items()}


def export(h: pd.DataFrame, days: pd.DataFrame, contrib: pd.DataFrame, front: str, inner: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """画面に要る列だけの表（日・レース）。front・inner は主の値の名前（校正の JSON の main）。"""
    d = days.rename(columns={f"z_{front}": "z_front", f"lo_{front}": "lo_front", f"hi_{front}": "hi_front",
                             f"z_{inner}": "z_inner", f"lo_{inner}": "lo_inner", f"hi_{inner}": "hi_inner"})[DAY_COLS].copy()
    r = contrib[["race_id", "date", "venue", "surf", "round", "distance"]].copy()
    r["front_se"], r["front_zc"] = contrib[f"SE_{front}"], contrib[f"zc_{front}"]     # 主が 4 角の版でも名前がぶつからないよう、1 列ずつ写す
    r["inner_se"], r["inner_zc"] = contrib[f"SE_{inner}"], contrib[f"zc_{inner}"]
    r["front4c_se"] = contrib["SE_front_4c"]
    r["top3"] = r["race_id"].map(top3_rows(h)).fillna("[]")
    for t in (d, r):
        for c in t.columns:
            if t[c].dtype == float:
                t[c] = t[c].round(3)
    return d, r


def _tmp_of(p: Path) -> Path:
    """一時ファイルはミラーの木の外（1 つ上）に、プロセスごとの名前で置く（push が拾わない・同時の実行でぶつからない）。"""
    return p.parent.parent / f".{p.parent.name}.{p.name}.{os.getpid()}.tmp"


def _atomic_parquet(df: pd.DataFrame, p: Path) -> None:
    tmp = _tmp_of(p)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def write(d: pd.DataFrame, r: pd.DataFrame, out: Path, meta: dict) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for y, g in d.groupby(d["date"].str[:4]):
        _atomic_parquet(g.reset_index(drop=True), out / f"days_{y}.parquet")
        files.append(f"days_{y}.parquet")
    for y, g in r.groupby(r["date"].str[:4]):
        _atomic_parquet(g.reset_index(drop=True), out / f"races_{y}.parquet")
        files.append(f"races_{y}.parquet")
    tmp = _tmp_of(out / "meta.json")          # meta は最後に置き換える（読む側は meta が新しければ表も新しい）
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out / "meta.json")
    return files + ["meta.json"]


def build(out: Path | None = None) -> dict:
    from src.research.track_bias import baseline as B
    from src.research.track_bias import data as D
    from src.research.track_bias import day as Y

    out = out or REPO / REL
    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    h = D.horse_values(D.load_horses())
    R = B.add_baseline(D.race_values(h), D.METRICS + D.AUX, cal["k"], cal.get("extras"))
    days = Y.apply(Y.day_raw(R), cal)             # 主の値（Y.FRONT_MAIN・INNER_MAIN）は校正の JSON の main に合わせる
    contrib = Y.race_contrib(R, days, cal)
    d, r = export(h, days, contrib, Y.FRONT_MAIN, Y.INNER_MAIN)
    meta = {"version": code_version(), "built": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"),
            "last_date": str(d["date"].max()), "main": {"front": Y.FRONT_MAIN, "inner": Y.INNER_MAIN},
            "days": int(len(d)), "races": int(len(r))}
    files = write(d, r, out, meta)
    return {**meta, "out": str(out), "files": len(files)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.pipeline.track_bias.serving")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", default=str(REPO / REL))
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.out)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
