"""想定単勝・想定複勝（決定ログ §77 のモデル）の推論。

出走馬の当日特徴量（V3Serving v6 の ``features()`` の戻り値）→ LightGBM（対数シェア・対数複勝）→ 較正の直線 → レース内で正規化 → 想定オッズ。
保存は最終オッズ予測のキャッシュ（FinalOddsCache。カテゴリ final_odds_prediction）と同じ形で行い、entries の source を ``sec77`` にする。
これにより predicted_single_odds（odds_source）を通るすべての表示（レース詳細・race-values・推奨馬券・予測成績）が同じ想定オッズを使う。
モデルが無い・特徴量が足りないときは何もしない（現行の想定オッズに自動で戻る）。KEIBA_ODDS_SEC77=0 で無効。

学習と保存: ``python -m src.scripts.data.odds_sec77_fit fit``（data/local/modeling/odds_pred/serving/）。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = Path(os.environ.get("ODDS_SEC77_DIR") or ROOT / "data/local/modeling/odds_pred/serving")
SOURCE = "sec77"
KEYS = ["race_id", "horse_id"]
DEFAULT_PLACE_HALF_RANGE = 0.12   # 想定複勝の中央から min/max を作る幅（実データの (max-min)/(2·mid) の中央値で上書きされる）
_cache: dict[str, Any] = {}


def enabled() -> bool:
    return (os.environ.get("KEIBA_ODDS_SEC77") or "1").strip().lower() not in ("0", "false", "no", "off")


def t6_enabled() -> bool:
    """再学習の想定オッズ（試行 6、odds_t6）で作る。KEIBA_ODDS_T6=1 かつモデルが揃っているときだけ（既定は従来の §77 のまま）。保存の形・source は sec77 のまま。"""
    from src.pipeline.inference import odds_t6

    return enabled() and (os.environ.get("KEIBA_ODDS_T6") or "").strip() == "1" and odds_t6.available()


def _sec77_files() -> bool:
    return (MODEL_DIR / "meta.json").exists() and (MODEL_DIR / "share_model.txt").exists() and (MODEL_DIR / "place_model.txt").exists()


def available() -> bool:
    return t6_enabled() or (enabled() and _sec77_files())


def reload() -> None:
    _cache.clear()


def _load() -> dict[str, Any]:
    key = str(MODEL_DIR)
    mt = (MODEL_DIR / "meta.json").stat().st_mtime
    ent = _cache.get(key)
    if ent and ent[0] == mt:
        return ent[1]
    import lightgbm as lgb

    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    pack = {"meta": meta, "share": lgb.Booster(model_file=str(MODEL_DIR / "share_model.txt")), "place": lgb.Booster(model_file=str(MODEL_DIR / "place_model.txt"))}
    _cache[key] = (mt, pack)
    return pack


def feature_columns() -> list[str]:
    return list(_load()["meta"]["cols"])


def predict_frame(X: pd.DataFrame) -> pd.DataFrame:
    """X: race_id・horse_id + 特徴量。戻り値: race_id・horse_id・win_odds・place_mid・place_min・place_max（odds_pred_eval の win_B / place_B と同じ変換）。"""
    if t6_enabled():
        from src.pipeline.inference import odds_t6

        try:
            return odds_t6.predict_frame(X)
        except KeyError:
            if not _sec77_files():
                raise
            logger.warning("試行 6 の特徴量が足りないため §77 で作る", exc_info=True)
    pack = _load()
    meta = pack["meta"]
    cols = meta["cols"]
    miss = [c for c in cols if c not in X.columns]
    if miss:
        raise KeyError(f"sec77 の特徴量が足りない列: {len(miss)} 列 {miss[:8]}")
    A = X[cols].astype("float32")
    share_raw = pack["share"].predict(A)
    place_raw = pack["place"].predict(A)
    a1, b1 = meta["win_line"]
    a2, b2 = meta["place_line"]
    rid = X["race_id"].astype(str).to_numpy()
    logv = a1 + b1 * share_raw
    v = np.exp(logv - logv.max())
    s = pd.Series(v).groupby(rid).transform("sum").to_numpy()
    share = v / s
    win = np.clip(float(meta["payback_win"]) / np.clip(share, 1e-6, None), 1.0, None)
    mid = np.clip(np.exp(a2 + b2 * place_raw), float(meta.get("place_floor", 1.05)), None)
    h = float(meta.get("place_half_range", DEFAULT_PLACE_HALF_RANGE))
    out = pd.DataFrame({"race_id": rid, "horse_id": X["horse_id"].astype(str).to_numpy(), "win_odds": win, "place_mid": mid})
    out["place_min"] = np.maximum(out["place_mid"] * (1 - h), 1.0)
    out["place_max"] = out["place_mid"] * (1 + h)
    return out


LOG_DIR: Path | None = None          # 試しで差し替える置き場（None なら _log_dir()）
_MODEL_INFO: dict = {}


def _log_dir() -> Path:
    """手元の追記の記録の置き場。推論の作業場所（KEIBA_INFERENCE_DIR）に従う（worktree から流しても本体の記録に混ざらない）。"""
    if LOG_DIR is not None:
        return LOG_DIR
    if os.environ.get("KEIBA_SEC77_LOG_DIR"):
        return Path(os.environ["KEIBA_SEC77_LOG_DIR"])
    from src.config.data_paths import LOCAL

    return Path(os.environ.get("KEIBA_INFERENCE_DIR") or LOCAL / "inference") / "sec77_log"


def model_info() -> dict:
    """想定オッズの模型の版（meta.json の fitted_at・trained_years と、meta と 2 つの模型のファイルの sha256 の先頭）。模型の更新時刻で読み直す。
    記録だけなので、読めないときは {} を返して保存・推奨を止めない。試行 6 の模型（KEIBA_ODDS_T6=1）のときはその旨を残す。"""
    import hashlib

    try:
        files = [MODEL_DIR / n for n in ("meta.json", "share_model.txt", "place_model.txt")]
        key = tuple(f.stat().st_mtime for f in files)
        if _MODEL_INFO.get("key") != key:
            h = hashlib.sha256()
            for f in files:
                h.update(f.read_bytes())
            meta = json.loads(files[0].read_text(encoding="utf-8"))
            _MODEL_INFO.update(key=key, info={"fitted_at": meta.get("fitted_at"), "trained_years": meta.get("trained_years"), "sha16": h.hexdigest()[:16]})
        info = dict(_MODEL_INFO["info"])
    except Exception:  # noqa: BLE001
        return {}
    if (os.environ.get("KEIBA_ODDS_T6") or "").strip() == "1":
        info["t6_requested"] = True          # 試行 6 の模型を使う設定（§77 に戻ったかは区別できないので、版は §77 の serving のもの）
    return info


def payload_for_race(rows: pd.DataFrame, horse_numbers: dict[str, int], race_name: str, elapsed: float = 0.0) -> dict | None:
    """1 レース分の予測行 → 最終オッズ予測のキャッシュと同じ形の応答。馬番が分からない馬がいれば None。"""
    entries = []
    for r in rows.itertuples(index=False):
        hn = horse_numbers.get(str(r.horse_id))
        if not hn:
            return None
        entries.append({"horse_number": int(hn), "predicted_win_odds": round(float(r.win_odds), 2),
                        "predicted_place_odds_min": round(float(r.place_min), 2), "predicted_place_odds_max": round(float(r.place_max), 2), "source": SOURCE})
    if not entries:
        return None
    entries.sort(key=lambda e: e["horse_number"])
    from datetime import datetime, timedelta, timezone

    created = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    return {"race_id": str(rows["race_id"].iloc[0]), "race_name": race_name or "-", "elapsed_sec": round(float(elapsed), 2), "entries": entries,
            "model": model_info(), "created_at": created}       # 記録だけ（推奨・画面の計算は entries だけを使う）


def save_for_race(storage, payload: dict, weights_known: bool | None = None, log: bool = True, extra: dict | None = None) -> bool:
    """保存する。log=True（開催日の予測の経路）のときだけ、手元の追記の記録にも 1 行を足す（過去のレースの取り込み直しでは記録しない）。"""
    from src.pipeline.inference.final_odds_service import FinalOddsCache

    ok = bool(FinalOddsCache.save_cached(storage, payload["race_id"], payload, source=SOURCE))
    if ok and log and (os.environ.get("KEIBA_RUN_LOCATION") or "").strip().lower() != "gcp":   # Cloud Run Jobs のコンテナでは消えるので書かない
        try:
            append_log(payload, weights_known, extra=extra)
        except Exception:  # noqa: BLE001  記録に失敗しても保存・推奨には影響させない
            logger.exception("[sec77] 手元の記録に書けません: %s", payload.get("race_id"))
    return ok


def training_null_share(X: pd.DataFrame, race_id: str) -> float | None:
    """そのレースの馬の、追い切りの列（tl_*・wc_*）の空の割合（記録だけ）。列やレースが無ければ None。"""
    cols = [c for c in X.columns if str(c).startswith(("tl_", "wc_"))]
    if not cols or "race_id" not in X.columns:
        return None
    g = X.loc[X["race_id"].astype(str) == str(race_id), cols]
    return round(float(g.isna().to_numpy().mean()), 4) if len(g) else None


def append_log(payload: dict, weights_known: bool | None = None, log_dir: Path | None = None, extra: dict | None = None) -> Path:
    """§77 を保存するたびに、手元の追記の記録 sec77_log/<作った日>.jsonl に 1 行を足す（GCS には上げない。記録だけ）。
    中身: race_id・作った時刻・模型の版・その時点で馬体重が入っていたか・馬ごとの q̂（想定単勝オッズの逆数を合計 1 に直した値）・
    同じレースの何回目の保存か（上書きの回数を数える）。"""
    import fcntl

    d = log_dir or _log_dir()
    created = str(payload.get("created_at") or "")
    day = str((extra or {}).get("race_day") or created[:10].replace("-", "") or "unknown")   # レースの日で分ける（前日の保存も同じファイル）
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{day}.jsonl"
    rid = str(payload["race_id"])
    inv = {int(e["horse_number"]): 1.0 / float(e["predicted_win_odds"]) for e in payload.get("entries") or [] if float(e.get("predicted_win_odds") or 0) > 0}
    z = sum(inv.values()) or 1.0
    row = {"race_id": rid, "created_at": created, "model": payload.get("model") or {}, "weights_known": weights_known,
           "env": os.environ.get("KEIBA_ENV") or "", "odds_model": SOURCE, **(extra or {}),
           "q_hat": {str(h): round(v / z, 6) for h, v in sorted(inv.items())},
           "win_odds": {str(e["horse_number"]): e["predicted_win_odds"] for e in payload.get("entries") or []}}
    with f.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)                  # 同時に書いても write_n が重ならない
        fh.seek(0)
        n_prev = 0
        for line in fh.read().splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            n_prev += int(isinstance(x, dict) and x.get("race_id") == rid)
        row["write_n"] = n_prev + 1
        fh.seek(0, 2)
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    return f


def is_sec77(single: dict | None) -> bool:
    """predicted_single_odds の結果が sec77 由来か。"""
    return bool(single) and (single.get("odds_model") == SOURCE)
