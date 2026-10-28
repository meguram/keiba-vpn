"""想定めぐ指数の階層縮約モデル（HEM）の推論。学習・検証は src/scripts/data/hem_megu.py（決定ログ §75）。

出走馬それぞれの過去走（expected_megu.runs_for_horse の走の記録）から、学習時と同じ関数（Hist / horse_terms / cell_mean）で項を作り、
config/expected_megu/hem.json の係数で想定値を出す。騎手の項は、学習時に作った騎手の累積表（``jockey_table.parquet``）から日付の前日までの値を引く。
旧版（加重平均＋条件補正）へ戻すには環境変数 ``KEIBA_EXPECTED_MEGU=legacy``。HEM の係数ファイルが無いときも旧版で動く。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.pipeline.inference import horse_growth as hg

ROOT = Path(__file__).resolve().parents[3]
HEM_PATH = ROOT / "config" / "expected_megu" / "hem.json"
STATE_PATH = Path(os.environ.get("KEIBA_HEM_STATE") or ROOT / "data" / "local" / "modeling" / "hem" / "jockey_table.parquet")

_lock = threading.Lock()
_model_cache: dict[str, tuple[Any, dict | None]] = {}
_jockey_cache: dict[str, tuple[Any, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {}

N_BUCKETS = (("n1", 1, 1), ("n2", 2, 2), ("n3_4", 3, 4), ("n5p", 5, 10_000))
# adjust に出す項の名前（学習時の特徴名 → 画面の表示名）
TERM_LABELS = {
    "us": "同じ面の適性", "ud": "近い距離の適性", "jock": "騎手",
    "cls_up": "昇級", "cls_down": "降級", "to_dirt": "ダート替わり", "to_turf": "芝替わり", "dist_up": "距離延長", "dist_down": "距離短縮",
    "rest_long": "休み明け", "rest_short": "間隔が短い", "age_c": "馬齢", "wdiff": "斤量差",
}


def enabled() -> bool:
    return os.environ.get("KEIBA_EXPECTED_MEGU", "").strip().lower() != "legacy"


def load_hem(path: Path | None = None) -> dict | None:
    """HEM の係数ファイル。無い・壊れているときは None（旧版で動く）。更新時刻でキャッシュ。"""
    if not enabled():
        return None
    p = path or HEM_PATH
    m = hg._mtime(p)
    with _lock:
        hit = _model_cache.get(str(p))
        if hit and hit[0] == m:
            return hit[1]
    try:
        doc = json.loads(p.read_text(encoding="utf-8")) if m else None
        if doc is not None:
            doc["cells"] = {"glob": float(doc["cells"]["glob"]), "fine": {int(k): float(v) for k, v in doc["cells"]["fine"].items()},
                            "mid": {int(k): float(v) for k, v in doc["cells"]["mid"].items()}}
            doc["coef_arr"] = np.asarray(doc["coef"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError):
        doc = None
    with _lock:
        _model_cache[str(p)] = (m, doc)
    return doc


def load_jockey_table(path: Path | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{騎手ID: (日付の昇順 dnum, 累積 e の和, 累積の件数)}。無ければ空（騎手の項は 0）。更新時刻でキャッシュ。"""
    p = path or STATE_PATH
    m = hg._mtime(p)
    with _lock:
        hit = _jockey_cache.get(str(p))
        if hit and hit[0] == m:
            return hit[1]
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if m:
        try:
            d = pd.read_parquet(p, columns=["jockey_id", "dnum", "cse", "ccnt"]).sort_values(["jockey_id", "dnum"])
            for jid, g in d.groupby("jockey_id", sort=False):
                out[str(jid)] = (g["dnum"].to_numpy().astype(np.int64), g["cse"].to_numpy(dtype=float), g["ccnt"].to_numpy(dtype=float))
        except (OSError, ValueError, KeyError):
            out = {}
    with _lock:
        _jockey_cache[str(p)] = (m, out)
    return out


def jockey_term(table: dict, jockey_id: str | None, dnum: int, kj: float) -> float:
    """その日より前の騎乗だけの累積から、騎手の効果 Σe/(n+kj)。表に無い騎手は 0。"""
    rec = table.get(str(jockey_id or ""))
    if rec is None:
        return 0.0
    dn, cse, cnt = rec
    i = int(np.searchsorted(dn, dnum, side="left")) - 1      # dnum より厳密に前
    return float(cse[i] / (cnt[i] + kj)) if i >= 0 else 0.0


def _surf_code(s: Any) -> int:
    return 1 if s == "芝" else 0 if s == "ダ" else -1


def predict_horses(runs_by_key: dict[Any, list], race_date: str, todays: dict[Any, dict | None], jockeys: dict[Any, str | None],
                   hem: dict, jtable: dict | None = None) -> dict[Any, dict | None]:
    """馬ごと（key）の HEM 予測。走が 1 つも使えない馬・今回の面と距離が不明な馬は None。
    runs_by_key の走は expected_megu の記録 (YYYYMMDD, 指数, 低信頼, 面, 距離, 競馬場, クラス水準, 斤量, 馬齢, race_id)。"""
    from src.scripts.data import hem_megu as hm

    p, cols, coef = hem["params"], hem["stack_cols"], hem["coef_arr"]
    dn = int("".join(ch for ch in str(race_date) if ch.isdigit())[:8] or 0)
    if not dn:
        return {k: None for k in runs_by_key}
    rows, trows, keys = [], [], []
    for ci, (k, runs) in enumerate(runs_by_key.items()):
        today = todays.get(k)
        if not runs or not today or not today.get("surface") or not today.get("distance"):
            continue
        for r in runs:
            r = tuple(r) + (None,) * (10 - len(r))
            ymd = int("".join(ch for ch in str(r[0]) if ch.isdigit())[:8] or 0)
            if ymd <= 0:
                continue
            age = r[8] if r[8] else 0
            rows.append({"horse_id": str(ci), "jockey_id": "", "y": float(r[1]), "low": bool(r[2]), "dnum": ymd, "surf": _surf_code(r[3]),
                         "dist": float(r[4]) if r[4] is not None else np.nan, "lvl": int(r[6]) if r[6] is not None else 10,
                         "age": int(min(max(age, 0), 7)), "jw": float(r[7]) if r[7] is not None else np.nan})
        lvl = today.get("level")
        ta = today.get("age") or 0
        trows.append({"key_i": ci, "horse_id": str(ci), "dnum": dn, "surf": _surf_code(today["surface"]), "dist": float(today["distance"]),
                      "lvl": int(lvl) if lvl is not None else 10, "age": int(min(max(ta, 0), 7)),
                      "jw": float(today["weight"]) if today.get("weight") else np.nan})
        keys.append(k)
    out: dict[Any, dict | None] = {k: None for k in runs_by_key}
    if not rows or not trows:
        return out
    h = pd.DataFrame(rows)
    T = pd.DataFrame(trows).drop(columns=["key_i"])
    hist = hm.Hist(h, hem["cells"])
    ht = hm.horse_terms(hist, T, p)
    ht["m_T"] = hm.cell_mean(hem["cells"], T["lvl"], T["age"], T["surf"])
    table = jtable if jtable is not None else load_jockey_table()
    ht["jock"] = [jockey_term(table, jockeys.get(k), dn, float(p["kj"])) for k in keys]
    pred = hm.predict_stack(ht, coef, cols)
    for i, k in enumerate(keys):
        row = ht.iloc[i]
        n = int(row["n"])
        if n < 1:
            continue
        contrib = {c: float(w) * float(row[c]) for c, w in zip(cols, coef[1:])}
        out[k] = {"pred": float(pred[i]), "n": n, "intercept": float(coef[0]), "contrib": contrib, "wmean": float(row["wmean"])}
    return out


def range_for(hem: dict, n: int) -> float:
    """想定の幅（±）。検証年の残差の標準偏差を、使った走数の区分ごとに。"""
    for name, lo, hi in N_BUCKETS:
        if lo <= n <= hi:
            return float((hem.get("resid_sd") or {}).get(name, 12.0))
    return 12.0


def adjust_items(hem_out: dict, base_value: float, value: float, min_abs: float = 0.3) -> list[dict]:
    """value − base_value を、項ごとの寄与に分ける。足し上げると value − base_value に一致するよう、残りは『平均への縮約・全体の傾向』に入れる。"""
    items = []
    for key, label in TERM_LABELS.items():
        v = hem_out["contrib"].get(key)
        if v is not None and abs(v) >= min_abs:
            items.append({"key": key, "label": label, "delta": round(v, 1)})
    rest = (value - base_value) - sum(i["delta"] for i in items)       # 表示する項の丸めも含めて、合計がちょうど一致するように
    if abs(rest) >= min_abs:
        items.append({"key": "shrink", "label": "平均への縮約・全体の傾向", "delta": round(rest, 1)})
    return items
