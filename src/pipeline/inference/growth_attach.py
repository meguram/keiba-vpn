"""成長度（成長指数 growth_index）を、予測 API の出走馬ごと p["growth"] に付ける（馬体重の発表 T-60 時点で揃う情報だけを使う）。

有効化: KEIBA_GROWTH_INDEX=1（既定オフ）。LightGBM 未導入・設定不足・馬の入力不足のときは何も付けず、例外も出さない。
モデル・係数: config/growth_index/（params.json, model.txt, ped_profile.json）。仕様・検証: docs/html/modeling/chapters/weight_growth_performance.html。
呼び出し: expected_megu.attach_expected_megu の末尾（app.py は触らない）。

入力の組み立て（学習 src/research/pedigree/growth_index_dataset.py と同じ定義）:
  stage        馬齢（馬 ID 先頭 4 桁の生年との差）× 月 → 成長段階
  weight       今回の馬体重（発表前は欠損）、前走比、自分の過去平均からのずれ
  oikiri       TL_A（馬別調教の追い切りの自己比）+ 追い切り評価・厩舎コメントの語数
  ped          父・母父の成長プロファイル（ped_profile.json。父・母父の ID は horse_pedigree_5gen）
  統制         直近水準（過去 5 走の有効な指数を 0.7^k で加重。前走が 7 日以内なら前走を除く）・休養・距離・面・性
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.pipeline.inference import growth_index as gi

logger = logging.getLogger(__name__)

ENV_FLAG = "KEIBA_GROWTH_INDEX"
CONF_DIR = Path(__file__).resolve().parents[3] / "config" / "growth_index"
TL_REF_DEFAULT = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline/final_stage/tl_reference.parquet"
DECAY = 0.7
K_PRIOR = 5
STALE_DAYS = 7
MIN_VALID = 2           # 直近水準を作るのに要る有効な過去走の数（学習と同じ）

GROUP_JA = {"stage": "成長期", "weight": "馬体重", "oikiri": "追い切り", "ped": "血統"}


def enabled() -> bool:
    return (os.environ.get(ENV_FLAG) or "").strip().lower() in ("1", "true", "on", "yes")


# ── 純粋関数（テスト対象） ──────────────────────────────────────────────

def _digits(s: Any) -> str:
    return re.sub(r"\D", "", str(s or ""))[:8]


def _to_date(ymd: Any) -> date | None:
    d = _digits(ymd)
    if len(d) != 8:
        return None
    try:
        return date(int(d[:4]), int(d[4:6]), int(d[6:8]))
    except ValueError:
        return None


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def age_from_id(horse_id: str, race_year: int) -> int | None:
    s = str(horse_id or "")
    if len(s) < 4 or not s[:4].isdigit():
        return None
    a = int(race_year) - int(s[:4])
    return a if 1 <= a <= 12 else None


def level_inputs(index_runs: Iterable[Mapping[str, Any]], history_dates: Iterable[str], race_date: str) -> dict[str, Any] | None:
    """統制変数のうち、指数の履歴から決まるもの。
    index_runs: [{date(YYYYMMDD か YYYY-MM-DD), value, low}]（指数のある走）。history_dates: 出走履歴の全日付（指数の有無を問わない。休養日数用）。
    返り値: {level, n_valid_prior, stale_prev, rest_days}。有効な過去走が MIN_VALID 未満なら None（指数を出さない）。"""
    rd = _to_date(race_date)
    if rd is None:
        return None
    prior = [(_digits(r.get("date")), r) for r in index_runs if _digits(r.get("date")) and _digits(r.get("date")) < _digits(race_date)]
    prior.sort(key=lambda t: t[0], reverse=True)
    hist = sorted({d for d in (_digits(x) for x in history_dates) if d and d < _digits(race_date)}, reverse=True)
    last_any = _to_date(hist[0]) if hist else (_to_date(prior[0][0]) if prior else None)
    rest = (rd - last_any).days if last_any else None
    stale = rest is None or rest <= STALE_DAYS
    valid = [(d, r) for d, r in prior if not r.get("low") and _num(r.get("value")) is not None]
    n_valid = len(valid)
    prev_date = _digits(hist[0]) if hist else (prior[0][0] if prior else "")
    use = [(d, r) for d, r in valid if not (stale and d == prev_date)][:K_PRIOR]
    if len(use) < MIN_VALID:
        return None
    w = [DECAY ** k for k in range(len(use))]
    level = sum(wi * float(r["value"]) for wi, (_, r) in zip(w, use)) / sum(w)
    return {"level": level, "n_valid_prior": n_valid, "stale_prev": 1 if stale else 0, "rest_days": rest}


def weight_inputs(weight_today: Any, announced_change: Any, past_weights: Iterable[Any]) -> dict[str, float | None]:
    """今回の馬体重から、前走比と自分の過去平均からのずれ。past_weights は新しい順（None 可）。発表前（weight_today が無い）は両方 None。"""
    w = _num(weight_today)
    pw = [x for x in (_num(v) for v in past_weights) if x is not None and x > 0]
    if w is None or w <= 0:
        return {"weight_change": None, "weight_dev": None}
    dw = (w - pw[0]) if pw else _num(announced_change)          # 学習は自前の前走差。前走の体重が無ければ発表の増減
    dev = (w - sum(pw) / len(pw)) if pw else None
    return {"weight_change": dw, "weight_dev": dev}


def build_x(*, horse_id: str, sex_age: str, weight_today: Any, announced_change: Any, past_weights: Iterable[Any], index_runs: Iterable[Mapping[str, Any]],
            history_dates: Iterable[str], race_date: str, distance: Any, surface: Any, tl_row: Mapping[str, Any] | None,
            extras_row: Mapping[str, Any] | None, sire_profile: Mapping | None, dam_sire_profile: Mapping | None) -> dict[str, Any] | None:
    """growth_index_model に渡す x。直近水準が作れない馬は None。"""
    lv = level_inputs(index_runs, history_dates, race_date)
    if lv is None:
        return None
    rd = _to_date(race_date)
    age = age_from_id(horse_id, rd.year) if rd else None
    stage = gi.stage_of(age, rd.month) if rd else None
    x: dict[str, Any] = {**lv, "stage": stage, **weight_inputs(weight_today, announced_change, past_weights)}
    x["distance"] = _num(distance)
    x["surface"] = "ダ" if str(surface or "").startswith("ダ") else "芝"
    sx = str(sex_age or "")[:1]
    x["sex"] = sx if sx in ("牡", "牝", "セ") else "牡"
    for k in gi.OIKIRI_NUM:
        v = (tl_row or {}).get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = (extras_row or {}).get(k)
        x[k] = _num(v)
    x["has_training"] = _num((tl_row or {}).get("has_training"))
    x["ped"] = gi.ped_inputs(stage, sire_profile, dam_sire_profile)
    return x


def summarize(res: Mapping[str, Any], x: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """growth_index_model の返り値を、API 用の形に整える。馬体重が未発表なら体重の寄与を 0 にして段階を付け直し、provisional=true。"""
    contrib = dict(res["contributions"])
    idx = float(res["index"])
    provisional = x.get("weight_change") is None and x.get("weight_dev") is None
    if provisional:
        idx -= float(contrib.get("weight", 0.0))
        contrib["weight"] = 0.0
    level = gi.stage_of_index(idx, params["thresholds"])
    cov = dict(res.get("coverage") or {})
    cov["oikiri"] = x.get("f_t4z_vs_self") is not None            # 追い切りの時計（馬別調教）があるか。評価・コメントだけでは『あり』にしない
    reasons = []
    min_pts = float(params.get("reason_min_points", 0.3))
    for g in ("oikiri", "stage", "weight", "ped"):
        pts = float(contrib.get(g, 0.0))
        if abs(pts) < min_pts or (g == "weight" and provisional):
            continue
        t = gi._reason(g, pts, x)
        if t:
            reasons.append((abs(pts), t))
    reasons.sort(key=lambda t: -t[0])
    return {
        "index": round(idx, 2), "stage": level, "label": gi.LABELS[level],
        "contributions": {k: round(float(v), 2) for k, v in contrib.items()},
        "reasons": [t for _, t in reasons[:3]],
        "provisional": bool(provisional),
        "n_missing": sum(1 for k in ("weight", "oikiri", "ped") if not cov.get(k)) + (1 if provisional and cov.get("weight") else 0),
        "coverage": {k: bool(v) for k, v in cov.items()},
    }


def compute_growth(x: Mapping[str, Any], params: Mapping[str, Any], booster: Any) -> dict[str, Any]:
    res = gi.growth_index_model(x, params, booster)
    return summarize(res, x, params)


def meta_from(params: Mapping[str, Any]) -> dict[str, Any]:
    """UI に出す、段階ごとの実績（評価期間 2024〜2025 年）。"""
    rows = []
    for r in params.get("level_table") or []:
        rows.append({"level": r.get("level"), "label": r.get("label"), "share": r.get("share"), "n": r.get("n"),
                     "share_resid_up": r.get("share_resid_up"), "mean_resid": r.get("mean_resid"), "share_up": r.get("share_up"), "mean_y_now": r.get("mean_y_now")})
    return {"version": params.get("version"), "unit": params.get("unit"), "thresholds": params.get("thresholds"), "level_table": rows,
            "note": "過去の傾向から作った目安で、AI の予測ではありません。1走ごとのばらつきが大きいため、段階は目安です。"}


# ── 読み込み（キャッシュ） ───────────────────────────────────────────────

@functools.lru_cache(maxsize=2)
def load_assets(conf_dir: str = str(CONF_DIR)) -> tuple[dict, Any, dict] | None:
    """(params, booster, ped_profile)。LightGBM 未導入・ファイル不足は None。"""
    d = Path(conf_dir)
    try:
        import lightgbm as lgb

        params = json.loads((d / "params.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(d / params.get("model", "model.txt")))
        prof = json.loads((d / "ped_profile.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("成長度の設定を読めません（付けません）: %s", exc)
        return None
    return params, booster, prof


@functools.lru_cache(maxsize=1)
def _tl_reference(path: str) -> Any:
    import pandas as pd

    return pd.read_parquet(path)


def _storage() -> Any:
    app = sys.modules.get("src.api.app")
    if app is not None and hasattr(app, "_get_storage"):
        return app._get_storage()
    from src.scraper.storage import HybridStorage

    return HybridStorage(Path(__file__).resolve().parents[3])


def _ancestors(storage: Any, horse_id: str) -> tuple[str, str]:
    """父・母父の馬 ID（horse_pedigree_5gen の ancestors: 1 世代目の 0 番 = 父、2 世代目の 2 番 = 母父）。"""
    try:
        doc = storage.load("horse_pedigree_5gen", horse_id) or {}
    except Exception:  # noqa: BLE001
        return "", ""
    sire = dam_sire = ""
    for a in doc.get("ancestors") or []:
        g, p = a.get("generation"), a.get("position")
        if g == 1 and p == 0:
            sire = str(a.get("horse_id") or "")
        elif g == 2 and p == 2:
            dam_sire = str(a.get("horse_id") or "")
    return sire, dam_sire


def _history(storage: Any, horse_id: str, race_date: str) -> tuple[list[str], list[Any]]:
    """出走履歴の日付（全走）と体重（新しい順）。対象日より前だけ。"""
    try:
        doc = storage.load("horse_result", horse_id) or {}
    except Exception:  # noqa: BLE001
        return [], []
    rows = []
    for h in doc.get("race_history") or []:
        if not isinstance(h, dict):
            continue
        d = _digits(h.get("date"))
        if d and d < _digits(race_date):
            rows.append((d, _num(h.get("horse_weight") if h.get("horse_weight") is not None else h.get("weight"))))
    rows.sort(key=lambda t: t[0], reverse=True)
    return [d for d, _ in rows], [w for _, w in rows]


def attach_growth(resp: dict, race_date: str, items: Mapping[int, Mapping | None] | None = None, base: Path | None = None, storage: Any = None,
                  conf_dir: str | Path | None = None, tl_ref_path: str | Path | None = None) -> dict:
    """予測 API の応答の各馬に growth を付ける。無効・失敗のときは resp をそのまま返す。
    items: attach_expected_megu が作った {馬の位置: expected_megu}。その recent（実測の直近走）を指数の履歴に使う。"""
    if not enabled():
        return resp
    preds = resp.get("predictions") or []
    if not preds:
        return resp
    try:
        assets = load_assets(str(conf_dir or CONF_DIR))
        if assets is None:
            return resp
        params, booster, prof = assets
        st = storage or _storage()
        rid = str(resp.get("race_id") or "")
        rdate = _digits(race_date)
        if len(rdate) != 8:
            return resp
        race_date_s = f"{rdate[:4]}-{rdate[4:6]}-{rdate[6:]}"
        import pandas as pd

        from src.pipeline.features import serving_ped_extras as spe
        from src.pipeline.features import training_serving as ts

        hids = [str(p.get("horse_id") or "") for p in preds]

        def gather(h: str) -> tuple[Any, tuple[list[str], list[Any]], tuple[str, str]]:        # 馬ごとの読み込み 3 種を並列化（保存先が遅いため）
            if not h:
                return None, ([], []), ("", "")
            return st.load("horse_training", h), _history(st, h, race_date_s), _ancestors(st, h)

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(hids)))) as ex:
            got = dict(zip(hids, ex.map(gather, hids)))
        records = {h: got[h][0] for h in hids}
        tl = pd.DataFrame()
        ref_path = Path(tl_ref_path or os.environ.get("KEIBA_GROWTH_TL_REF") or TL_REF_DEFAULT)
        if any(records.values()) and ref_path.exists():
            tl = ts.tl_a_for_race({h: r for h, r in records.items() if h}, race_date_s, _tl_reference(str(ref_path)))
        extras = pd.DataFrame()
        if rid:
            keys = pd.DataFrame({"race_id": [rid] * len(hids), "horse_id": hids})
            extras = spe.extras_features(keys, spe.fetch_extras_rows([rid], st)).assign(horse_id=hids)
        course = {"distance": resp.get("distance"), "surface": resp.get("surface")}
        for i, p in enumerate(preds):
            hid = hids[i]
            if not hid:
                continue
            ex = (items or {}).get(i) or p.get("expected_megu") or {}
            (_, (hist_dates, hist_w), (sire_id, dam_id)) = got[hid]
            tl_row = tl.loc[hid].to_dict() if len(tl) and hid in tl.index else None
            ex_row = None
            if len(extras):
                m = extras[extras["horse_id"] == hid]
                ex_row = m.iloc[0].to_dict() if len(m) else None
            x = build_x(
                horse_id=hid, sex_age=p.get("sex_age") or "", weight_today=p.get("weight"), announced_change=p.get("weight_change"), past_weights=hist_w,
                index_runs=ex.get("recent") or [], history_dates=hist_dates, race_date=race_date_s, distance=course["distance"], surface=course["surface"],
                tl_row=tl_row, extras_row=ex_row, sire_profile=(prof.get("sire") or {}).get(sire_id), dam_sire_profile=(prof.get("dam_sire") or {}).get(dam_id),
            )
            if x is None:
                continue
            p["growth"] = compute_growth(x, params, booster)
        if any(p.get("growth") for p in preds):
            resp["growth_meta"] = meta_from(params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("成長度の付与に失敗（付けません）: %s", exc)
    return resp
