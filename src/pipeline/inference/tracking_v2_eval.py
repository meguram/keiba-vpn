"""追走難度 v2 の「保存した予測」と「実測」の週次の突き合わせ。

実測のラベル Y = （実測の上がり3F − そのレースの中央値）− その馬の過去 5 走の『位置取りの効果を除いた上がりの力』w_adj。
Y が大きい＝終いが落ちた（余裕が無かった）。保存した予測の drain_sec（予測 Y）と ease_pct を、この Y と突き合わせる（tp-54 の学習のラベルと同じ定義）。
w_adj は保存時の入力（inputs.w_adj）を使うので、結果を見た後に過去の定義が変わっても同じ尺度で比べられる。

保存先: data/local/inference/tracking_v2_eval/{YYYYMMDD}.json（日次。pairs を含む）、weekly/{週の月曜 YYYYMMDD}.json（週次）、_trend.json（週ごとの要約の推移）。
自動実行は KEIBA_TRACKING_V2_EVAL=1 のときだけ（金曜の weekly-update の後、`refresh_for_dates`）。失敗しても呼び出し元を止めない。
手動: python -m src.scripts.data.tracking_v2_eval --date YYYYMMDD / --from A --to B / --week YYYYMMDD [--dry-run]
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.inference import tracking_v2_service as svc

logger = logging.getLogger("TrackingV2Eval")

ROOT = Path(__file__).resolve().parents[3]
ENV_FLAG = "KEIBA_TRACKING_V2_EVAL"
ENV_DIR = "KEIBA_TRACKING_V2_EVAL_DIR"
ENV_TABLES = "KEIBA_TABLES_DIR"
L3F_RANGE = (30.0, 50.0)
DIST_BANDS = [("短距離(~1300)", 0, 1300), ("マイル(1400~1800)", 1400, 1800), ("中距離(1900~2200)", 1900, 2200), ("中長距離(2300~)", 2300, 99999)]


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes")


def eval_dir() -> Path:
    return Path(os.environ.get(ENV_DIR) or ROOT / "data" / "local" / "inference" / "tracking_v2_eval")


def tables_dir() -> Path:
    return Path(os.environ.get(ENV_TABLES) or ROOT / "data" / "page_reference" / "tables")


def _ymd(d: str) -> str:
    return str(d).replace("-", "").replace("/", "")[:8]


def _iso(d: str) -> str:
    d = _ymd(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def week_monday(d: str) -> str:
    t = datetime.strptime(_ymd(d), "%Y%m%d")
    return (t - timedelta(days=t.weekday())).strftime("%Y%m%d")


# ───────────────────────── 実測

def load_actuals(date: str) -> pd.DataFrame:
    """その開催日の全馬の実測（race_id, horse_number, act_l3f, act_l3f_dev, surface, distance）。結果が無ければ空。"""
    iso = _iso(date)
    f = tables_dir() / iso[:4] / "race_result_flat.parquet"
    if not f.exists():
        return pd.DataFrame()
    cols = ["race_id", "horse_number", "last_3f", "date", "surface", "distance", "finish_position"]
    import pyarrow.parquet as pq

    have = set(pq.read_schema(f).names)
    d = pd.read_parquet(f, columns=[c for c in cols if c in have])
    d["date"] = d["date"].astype(str).str.replace("/", "-").str[:10]
    d = d[d["date"] == iso].copy()
    if d.empty:
        return d
    d["act_l3f"] = pd.to_numeric(d["last_3f"], errors="coerce")
    d.loc[(d["act_l3f"] < L3F_RANGE[0]) | (d["act_l3f"] > L3F_RANGE[1]), "act_l3f"] = np.nan
    med = d.groupby("race_id")["act_l3f"].transform("median")
    d["act_l3f_dev"] = d["act_l3f"] - med
    d["horse_number"] = pd.to_numeric(d["horse_number"], errors="coerce")
    return d.drop_duplicates(["race_id", "horse_number"])


def served_docs(date: str) -> dict[str, dict]:
    """その日の保存から、読み出しで返される版（取消で作り直したものがあればそれ、無ければ前夜）を race_id ごとに返す。"""
    day_dir = svc.base_dir() / _iso(date)
    out: dict[str, dict] = {}
    if not day_dir.is_dir():
        return out
    for p in sorted(day_dir.glob("*.json")):
        if p.name.endswith(".recomputed.json"):
            continue
        d = svc._read(p)
        if d:
            out[str(d.get("race_id") or p.stem)] = d
    for p in sorted(day_dir.glob("*.recomputed.json")):
        d = svc._read(p)
        if d:
            rid = str(d.get("race_id") or p.name.split(".")[0])
            d["_recomputed"] = True
            out[rid] = d
    return out


def band_of(distance: Any) -> str:
    try:
        x = int(distance)
    except Exception:  # noqa: BLE001
        return "不明"
    for name, lo, hi in DIST_BANDS:
        if lo <= x <= hi:
            return name
    return "不明"


# ───────────────────────── 集計（純粋関数）

def _corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float | None:
    m = a.notna() & b.notna()
    if m.sum() < 10 or a[m].nunique() < 2 or b[m].nunique() < 2:
        return None
    v = a[m].corr(b[m], method=method)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)


def metrics_from_pairs(pairs: pd.DataFrame) -> dict:
    """pairs: drain, ease, y（+任意の列）。予測 vs 実測の要約。"""
    d = pairs.dropna(subset=["drain", "y"]).copy()
    n = int(len(d))
    out: dict[str, Any] = {"n_pairs": n}
    if n < 10:
        return out
    out["corr_drain_y"] = _corr(d["drain"], d["y"])
    out["spearman_drain_y"] = _corr(d["drain"], d["y"], "spearman")
    out["corr_ease_y"] = _corr(d["ease"], d["y"])           # 楽なほど Y が小さいので、負が理想
    out["mean_y"] = round(float(d["y"].mean()), 4)
    try:
        d["dec"] = pd.qcut(d["drain"].rank(method="first"), 10, labels=False)
        g = d.groupby("dec").agg(n=("y", "size"), drain=("drain", "mean"), y=("y", "mean")).reset_index()
        out["deciles"] = [{"dec": int(r.dec) + 1, "n": int(r.n), "mean_drain": round(float(r.drain), 3), "mean_y": round(float(r.y), 3)} for r in g.itertuples()]
        out["decile_monotonic_spearman"] = _corr(g["dec"].astype(float), g["y"], "spearman")
        out["decile_top_minus_bottom_y"] = round(float(g["y"].iloc[-1] - g["y"].iloc[0]), 3)
    except Exception:  # noqa: BLE001
        pass
    e = d["ease"]
    out["ease_mean"] = round(float(e.mean()), 1)
    out["share_ease_ge60"] = round(float((e >= 60).mean()), 4)
    out["share_ease_lt30"] = round(float((e < 30).mean()), 4)
    return out


def breakdown(pairs: pd.DataFrame) -> dict:
    d = pairs.dropna(subset=["drain", "y"]).copy()
    out: dict[str, Any] = {}
    if d.empty:
        return out
    d["band"] = d["distance"].map(band_of)
    for key, col in (("surface", "surface"), ("distance_band", "band")):
        res = {}
        for name, g in d.groupby(col):
            m = metrics_from_pairs(g)
            res[str(name)] = {k: m.get(k) for k in ("n_pairs", "corr_drain_y", "spearman_drain_y", "share_ease_ge60", "share_ease_lt30", "ease_mean")}
        out[key] = res
    return out


# ───────────────────────── 日次・週次

def aggregate_day(date: str, *, save: bool = True) -> dict:
    """保存した v2 を、その日の実測と突き合わせて日次の JSON にする。結果が無い日は status=no_results（保存しない）。"""
    iso = _iso(date)
    docs = served_docs(date)
    act = load_actuals(date)
    races_held = int(act["race_id"].nunique()) if not act.empty else 0
    out: dict[str, Any] = {"date": iso, "computed_at": svc._now_jst(), "races_saved": len(docs), "races_recomputed": sum(1 for d in docs.values() if d.get("_recomputed")),
                           "races_held": races_held, "coverage": round(len(docs) / races_held, 4) if races_held else None}
    if act.empty:
        out["status"] = "no_results"
        return out
    rows = []
    n_saved_horses = n_v2 = 0
    for rid, doc in docs.items():
        n_saved_horses += int(doc.get("n_horses") or 0)
        n_v2 += int(doc.get("n_with_v2") or 0)
        a = act[act["race_id"] == rid].set_index("horse_number")
        for hn, h in (doc.get("horses") or {}).items():
            try:
                hn_i = int(hn)
            except Exception:  # noqa: BLE001
                continue
            t = h.get("tracking_v2") or {}
            inp = h.get("inputs") or {}
            dev = a["act_l3f_dev"].get(hn_i) if hn_i in a.index else None
            w = inp.get("w_adj")
            y = None if dev is None or w is None or (isinstance(dev, float) and math.isnan(dev)) else float(dev) - float(w)
            rows.append({"race_id": rid, "horse_number": hn_i, "drain": t.get("drain_sec"), "ease": t.get("ease_pct"), "y": y,
                         "surface": doc.get("surface"), "distance": doc.get("distance"), "recomputed": bool(doc.get("_recomputed"))})
    pairs = pd.DataFrame(rows)
    out["n_horses_saved"], out["n_horses_with_v2"] = n_saved_horses, n_v2
    out["v2_share"] = round(n_v2 / n_saved_horses, 4) if n_saved_horses else None
    if pairs.empty:
        out["status"] = "no_pairs"
    else:
        out["status"] = "ok"
        out["metrics"] = metrics_from_pairs(pairs)
        out["breakdown"] = breakdown(pairs)
        out["pairs"] = [[r.race_id, int(r.horse_number), None if pd.isna(r.drain) else round(float(r.drain), 3), None if pd.isna(r.ease) else round(float(r.ease), 1),
                         None if pd.isna(r.y) else round(float(r.y), 3), r.surface, r.distance] for r in pairs.itertuples()]
    if save and out["status"] in ("ok", "no_pairs"):
        d = eval_dir()
        d.mkdir(parents=True, exist_ok=True)
        svc._write(d / f"{_ymd(date)}.json", out)
    return out


def aggregate_week(date: str, *, save: bool = True) -> dict:
    """その日を含む週（月〜日）の日次の JSON を束ねて、週次の指標を作る。"""
    mon = week_monday(date)
    t0 = datetime.strptime(mon, "%Y%m%d")
    days = [(t0 + timedelta(days=i)).strftime("%Y%m%d") for i in range(7)]
    frames, used = [], []
    held = saved = 0
    for d in days:
        p = eval_dir() / f"{d}.json"
        doc = svc._read(p)
        if not doc or doc.get("status") != "ok":
            continue
        used.append(d)
        held += int(doc.get("races_held") or 0)
        saved += int(doc.get("races_saved") or 0)
        df = pd.DataFrame(doc.get("pairs") or [], columns=["race_id", "horse_number", "drain", "ease", "y", "surface", "distance"])
        frames.append(df)
    out: dict[str, Any] = {"week_start": _iso(mon), "days": [_iso(d) for d in used], "computed_at": svc._now_jst(), "races_held": held, "races_saved": saved,
                           "coverage": round(saved / held, 4) if held else None}
    if not frames:
        out["status"] = "no_days"
        return out
    pairs = pd.concat(frames, ignore_index=True)
    out["status"] = "ok"
    out["metrics"] = metrics_from_pairs(pairs)
    out["breakdown"] = breakdown(pairs)
    if save:
        d = eval_dir() / "weekly"
        d.mkdir(parents=True, exist_ok=True)
        svc._write(d / f"{mon}.json", out)
        _update_trend(mon, out)
    return out


def _update_trend(mon: str, wk: dict) -> None:
    p = eval_dir() / "_trend.json"
    cur = svc._read(p) or {"weeks": []}
    m = wk.get("metrics") or {}
    row = {"week_start": wk["week_start"], "n_pairs": m.get("n_pairs"), "corr_drain_y": m.get("corr_drain_y"), "spearman_drain_y": m.get("spearman_drain_y"),
           "decile_monotonic_spearman": m.get("decile_monotonic_spearman"), "decile_top_minus_bottom_y": m.get("decile_top_minus_bottom_y"),
           "share_ease_ge60": m.get("share_ease_ge60"), "share_ease_lt30": m.get("share_ease_lt30"), "coverage": wk.get("coverage")}
    weeks = [w for w in cur.get("weeks", []) if w.get("week_start") != row["week_start"]] + [row]
    cur["weeks"] = sorted(weeks, key=lambda w: w["week_start"])
    cur["updated_at"] = svc._now_jst()
    svc._write(p, cur)


def refresh_for_dates(dates: list[str], *, reason: str = "") -> dict:
    """指定した開催日を突き合わせて日次・週次を保存する（自動実行用。KEIBA_TRACKING_V2_EVAL=1 のときだけ。失敗しても呼び出し元を止めない）。"""
    if not enabled():
        return {"status": "disabled"}
    out: dict[str, str] = {}
    weeks: set[str] = set()
    try:
        for d in sorted({_ymd(x) for x in dates if x}):
            try:
                doc = aggregate_day(d)
                out[d] = f"{doc['status']}:{(doc.get('metrics') or {}).get('n_pairs', 0)}"
                if doc["status"] == "ok":
                    weeks.add(week_monday(d))
            except Exception as exc:  # noqa: BLE001
                logger.warning("追走難度 v2 の突き合わせに失敗 %s (%s): %s", d, reason, exc)
                out[d] = "error"
        for w in sorted(weeks):
            try:
                aggregate_week(w)
            except Exception as exc:  # noqa: BLE001
                logger.warning("追走難度 v2 の週次の突き合わせに失敗 %s (%s): %s", w, reason, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("追走難度 v2 の突き合わせを始められない (%s): %s", reason, exc)
        return {"status": "error", "error": str(exc)[:200]}
    logger.info("追走難度 v2 の突き合わせ (%s): %s", reason, out)
    return {"status": "ok", "days": out, "weeks": sorted(weeks)}
