"""走法マップ: 性別・父・母父だけから出す「血統から見た傾向」（tp-85）。

事前計算（めぐ分析の running_map）の中で、出走馬ごとに 10 観点の期待値と 80% 区間を付ける。API・画面では計算しない。
モデルは data/local/modeling/tp85_ped_traits/model（ホスト）。無い環境では何も付けず、画面の欄も出さない。
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# (キー, 表示名, 表示形式, 説明)。pct=割合（0〜1 を % に）, m=メートル, q=0〜1, n=数値
TRAITS: list[tuple[str, str, str, str]] = [
    ("dirt_share", "ダート比率", "pct", "高いほどダートを多く走る"),
    ("dist_run", "走る距離", "m", "芝ダの平均との差。＋は長め"),
    ("dist_good", "好走する距離", "m", "芝ダの平均との差。＋は長め"),
    ("style_q", "脚質", "q", "0=先頭〜1=後方"),
    ("ax_ts", "TS", "n", "末脚の速さ"),
    ("ax_sustain", "TS持続力", "n", "ギアチェンジ・TS・Hペース耐性の血統予測から出した値"),
    ("ax_debut", "出脚", "n", "出足の速さ"),
    ("ax_gear", "ギアチェンジ", "n", "加速の切り替え"),
    ("ax_endure", "Hペース耐性", "n", "速い流れで脚を使っても終いに落ちない"),
    ("ax_front", "先行力", "n", "先行できる力"),
]


@lru_cache(maxsize=1)
def _load() -> tuple[dict, dict] | None:
    try:
        from src.pipeline.features.tp85_pedprior import MODEL_DIR, load_models

        man = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
        models = load_models(MODEL_DIR)
        return models, man
    except Exception:   # モデルの無い環境（VPS・開発 PC）では血統の欄を付けない
        logger.info("血統モデルを読めないため、血統の傾向は付けません", exc_info=True)
        return None


def clear_cache() -> None:
    _load.cache_clear()


def _read_ped(horse_ids: list[str]) -> tuple[dict[str, dict[int, str]], dict[str, dict[str, str | None]]]:
    from src.scripts.data import tp85_ped_traits as t85

    want = {v: k for k, v in t85.POS.items() if k in (1, 2)}
    ped: dict[str, dict[int, str]] = {}
    names: dict[str, dict[str, str | None]] = {}
    for h in horse_ids:
        for dr in t85.PED_DIRS:
            f = dr / h[:4] / f"{h}.json"
            if not f.exists():
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            r: dict[int, str] = {}
            for a in d.get("ancestors") or []:
                k = want.get((a.get("generation"), a.get("position")))
                if k and a.get("horse_id"):
                    r[k] = str(a["horse_id"])
            if 1 in r:
                ped[h] = r
                names[h] = {"sire": d.get("sire"), "damsire": d.get("dam_sire")}
                break
    return ped, names


def _sex_of(entry: dict) -> str:
    s = str(entry.get("sex_age") or entry.get("sex") or "")[:1]
    return s if s in ("牡", "牝", "セ") else "牡"


def _scale(key: str, meta: dict, models: dict) -> tuple[float | None, float | None, float | None]:
    """(基準=全体の平均, 上位下位1割の目安の幅, 当てはまり R²)。画面で『平均からどれだけ離れているか』の目盛りに使う。"""
    r2 = meta.get("r2_cv") if meta.get("kind") == "derived" else meta.get("r2_obs_cal")
    m = models.get(key)
    if m is None:
        return None, None, r2
    if meta.get("kind") == "derived":
        b = np.asarray(meta["beta"], dtype=float)
        base = float(b[0] + sum(b[i + 1] * models_in.st["m"] for i, models_in in enumerate(m.inputs)))
        half = (meta.get("tails") or {}).get("pred_diff")
        return base, None if half is None else abs(float(half)) / 2, r2
    top, bot = (meta.get("top10") or {}).get("pred_diff"), (meta.get("bottom10") or {}).get("pred_diff")
    half = None if top is None or bot is None else (abs(float(top)) + abs(float(bot))) / 2
    return float(m.st["m"]), half, r2


def pedigree_for(entries: list[dict]) -> dict[str, Any] | None:
    """entries（horse_id, sex_age）から {"traits": [...], "horses": {horse_id: {...}}}。付けられなければ None。"""
    loaded = _load()
    if loaded is None:
        return None
    models, man = loaded
    ids = [str(e["horse_id"]) for e in entries if e.get("horse_id")]
    ped, names = _read_ped(ids)
    sex = {str(e["horse_id"]): _sex_of(e) for e in entries if e.get("horse_id")}
    got = [h for h in ids if h in ped]
    if not got:
        return None
    hs = pd.DataFrame({"horse_id": got, "sex": [sex[h] for h in got]})
    preds = {k: m.predict(ped, hs).set_index("horse_id") for k, m in models.items()}
    traits, keep = [], []
    for key, label, fmt, hint in TRAITS:
        if key not in models:
            continue
        base, half, r2 = _scale(key, man.get(key, {}), models)
        traits.append({"key": key, "label": label, "fmt": fmt, "hint": hint,
                       "base": None if base is None else round(base, 4), "ref": None if half is None else round(half, 4),
                       "r2": None if r2 is None else round(float(r2), 3)})
        keep.append(key)
    horses: dict[str, Any] = {}
    for h in got:
        v = {}
        for k in keep:
            r = preds[k].loc[h]
            lo, hi, mean = float(r["lo80"]), float(r["hi80"]), float(r["mean"])
            if k == "dirt_share":
                lo, mean, hi = (min(1.0, max(0.0, x)) for x in (lo, mean, hi))
            v[k] = [round(mean, 4), round(lo, 4), round(hi, 4)]
        horses[h] = {"sex": sex[h], "sire": names[h]["sire"], "damsire": names[h]["damsire"], "has_damsire": 2 in ped[h], "v": v}
    return {"traits": traits, "horses": horses}
