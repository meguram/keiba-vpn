"""追走難度 v2（tp-54）を前夜の定期実行で算出・保存し、予測 API は保存を優先して返す。

有効化は環境変数 KEIBA_TRACKING_V2=1（既定は無効＝何もしない。tracking_difficulty_v2.enabled と同じ）。
保存先は data/local/inference/tracking_v2/<日付>/<race_id>.json（KEIBA_TRACKING_V2_DIR で変更可）。

- 前夜（raceday-eve、毎日 18:00 JST、翌日が開催日のとき）に、出馬表・馬柱から全馬の v2 を作って保存する（`precompute_tracking_v2_for_race_ids`）。
  eve の失敗は本体のジョブを落とさない（例外は警告ログのみ）。
- 当日の読み出し（`attach_tracking_v2_cached`）: 保存があり、いまの出走馬の馬番の集合が保存時と同じなら保存を返す（固定値）。
  取消・除外・追加で馬番の集合が変わったレースだけ、その場で計算し直し（`<race_id>.recomputed.json` に別保存。前夜の保存は消さない）、それを返す。
  馬体重・増減の発表では値を動かさない（`KEIBA_TRACKING_V2_RECOMPUTE=none` で取消でも動かさない、`always` で毎回その場計算）。
- 保存が無ければ従来どおりその場で計算する（応答の形は同じ。tracking_v2_meta に source と computed_at が足されるだけ）。

保存ファイルの形:
  version, race_id, date, venue, surface, distance, track_condition, n_horses, n_with_v2, horse_numbers（署名）, computed_at（JST）,
  source（raceday_eve / recomputed / backfill）, card_fetched_at, past_fetched_at, model（version とファイルの短いハッシュ）,
  pace_forecast（位置取り側のペース）, pace_unified（整合層のペースと z。無ければ None）, tracking_v2_meta,
  horses: {馬番: {horse_id, tracking_v2{ease_pct,label,drain_sec,demand_sec?,capacity_sec?}, position_early, inputs{...}}},
  z_hat（レース単位の想定ペース。学習と同じ式）, race_only（馬ごとの v2 が無いレースの理由。新馬戦など。horses は位置取り予想の
  ある馬の inputs だけで tracking_v2 は無い。v6e の前段だけが使い、API の読み出し・突き合わせの集計は使わない）
inputs は週次の突き合わせ用（実測のラベル Y = 実測の上がり偏差 − w_adj）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("TrackingV2Service")

VERSION = "tv2-store-1"
ENV_DIR = "KEIBA_TRACKING_V2_DIR"
ENV_RECOMPUTE = "KEIBA_TRACKING_V2_RECOMPUTE"      # horses（既定: 馬番の集合が変わったら）/ none / always
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "local" / "inference" / "tracking_v2"
JST = timezone(timedelta(hours=9))


def enabled() -> bool:
    from src.pipeline.models.tracking_difficulty_v2 import enabled as _enabled

    return _enabled()


def base_dir() -> Path:
    return Path(os.environ.get(ENV_DIR) or DEFAULT_DIR)


def recompute_policy() -> str:
    v = os.environ.get(ENV_RECOMPUTE, "horses").strip().lower()
    return v if v in ("horses", "none", "always") else "horses"


def _day(date: str | None) -> str:
    return str(date or "")[:10].replace("/", "-")


def path_for(race_id: str, date: str, *, recomputed: bool = False) -> Path:
    name = f"{race_id}.recomputed.json" if recomputed else f"{race_id}.json"
    return base_dir() / _day(date) / name


def _write(p: Path, doc: dict) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def _read(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def load_saved(race_id: str, date: str, *, recomputed: bool = False) -> dict | None:
    from src.ops.gcs_mirror import lazy_fetch

    p = path_for(race_id, date, recomputed=recomputed)
    lazy_fetch(p, date)               # 窓より古い日は起動時に取り込まない → 読まれたときに GCS から 1 件
    return _read(p)


def _short_hash(p: Path) -> str | None:
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:10]
    except Exception:  # noqa: BLE001
        return None


def model_info() -> dict:
    from src.pipeline.inference import pace_coherence as pcoh
    from src.pipeline.models import tracking_difficulty_v2 as tv2

    m = tv2.load_model()
    d = tv2.CONFIG_DIR
    return {"model_version": (m or {}).get("version"), "model_hash": _short_hash(d / "model.txt"), "meta_hash": _short_hash(d / "meta.json"),
            "position_effect_hash": _short_hash(d / "position_effect.json"), "pace_v2_hash": _short_hash(pcoh.CONFIG_V2_PATH),
            "pace_config_hash": _short_hash(pcoh.CONFIG_PATH)}


def _now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _scraped(doc: dict | None) -> str | None:
    m = (doc or {}).get("_meta") or {}
    return m.get("scraped_at_jst")


def numbers_of(preds: list[dict]) -> list[int]:
    out = []
    for p in preds or []:
        try:
            if p.get("horse_number") is not None:
                out.append(int(p["horse_number"]))
        except Exception:  # noqa: BLE001
            continue
    return sorted(set(out))


# ───────────────────────── 作る（前夜・バックフィル・取消後の計算し直し）

def _minimal_resp(card: dict) -> dict:
    preds = []
    for e in card.get("entries") or []:
        if isinstance(e, dict) and e.get("horse_number") is not None:
            preds.append({"horse_number": int(e["horse_number"]), "horse_id": e.get("horse_id")})
    return {"venue": card.get("venue"), "surface": card.get("surface"), "distance": card.get("distance"),
            "track_condition": card.get("track_condition"), "predictions": preds}


def build_doc(card: dict, past: dict | None, race_id: str, *, source: str, resp: dict | None = None) -> dict | None:
    """出馬表・馬柱から v2 を作って保存用の文書にする。材料不足（モデル無し等）は None。

    位置取りのペースが unknown のレース（新馬戦など）は、条件を満たせばレース単位の z_hat だけの文書（race_only）を返す
    （build_doc_reason を参照）。resp を渡すと（予測 API の応答に位置取り予想が付いたもの）、それを土台にする。無ければ出馬表から作る。
    """
    return build_doc_reason(card, past, race_id, source=source, resp=resp)[0]


def build_doc_reason(card: dict, past: dict | None, race_id: str, *, source: str, resp: dict | None = None) -> tuple[dict | None, str]:
    """build_doc と同じ。作れなかったときは理由（no_entries / model_missing / pace_unknown / no_position / no_horses）も返す。

    ペースが unknown で馬ごとの v2 が作れないレース（新馬戦など）でも、馬柱があり 8 頭以上で z_hat が計算できれば、
    レース単位の z_hat だけの保存（理由 race_only）を返す。
    """
    from src.pipeline.inference import pace_coherence as pcoh
    from src.pipeline.inference.position_forecast import attach_position_forecast
    from src.pipeline.models.tracking_difficulty_v2 import attach_tracking_v2_to_response

    date = str(card.get("date") or "")
    course = {k: card.get(k) for k in ("venue", "surface", "distance")}
    course["track_condition"] = card.get("track_condition")
    if resp is None:
        resp = attach_position_forecast(_minimal_resp(card), past, date, {k: course[k] for k in ("venue", "surface", "distance")})
    capture: dict[int, dict] = {}
    resp = attach_tracking_v2_to_response(resp, past, date, course, card, capture=capture)
    preds = resp.get("predictions") or []
    meta = resp.get("tracking_v2_meta")
    pf = resp.get("pace_forecast") or {}
    z_race = _race_z(pf, preds, course)
    if not meta:
        if not preds:
            return None, "no_entries"
        if tv2_load_model() is None:      # v2 のモデルが無い環境は v6e の前段も動かさない前提（race_only も作らない）
            return None, "model_missing"
        if not pf or pf.get("level") == "unknown":
            # 新馬戦など、位置取りのペースが判定できないレース: 馬ごとの v2 は作れないが、学習（tp53/tp54）と同じく
            # レース単位の z_hat と、位置取り予想のある馬の pf_early・pf_rank は付ける（学習の新馬戦は z_hat が約 98% 埋まる。
            # 保存が無いと v6e の前段でこれらが全馬欠損になっていた）。馬柱が無い・8 頭未満（学習に z_hat が無い）は作らない。
            if z_race is not None and _past_count(past) >= RACE_ONLY_MIN_FIELD:
                return _doc(card, past, race_id, date, preds, source=source, pf=pf, unified=None, meta=None,
                            horses=_known_inputs(preds, z_race), z_race=z_race, race_only="pace_unknown"), "race_only"
            return None, "pace_unknown"
        return None, "no_position"
    unified = None
    try:
        cfg = pcoh.load_config()
        if cfg and pf and pf.get("level") != "unknown":
            horses = [{"horse_number": int(p["horse_number"]), "early_pos_norm": float((p.get("position_forecast") or {}).get("early_pos_norm")), "past_races": []}
                      for p in preds if p.get("horse_number") is not None and (p.get("position_forecast") or {}).get("early_pos_norm") is not None]
            if horses:
                unified = pcoh.coherent_forecast(cfg, horses, pf, course, date)["pace"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("整合層のペースの保存用計算に失敗 %s: %s", race_id, exc)
    horses_out: dict[str, Any] = {}
    for p in preds:
        if p.get("horse_number") is None or not p.get("tracking_v2"):
            continue
        hn = int(p["horse_number"])
        horses_out[str(hn)] = {"horse_id": p.get("horse_id"), "tracking_v2": p["tracking_v2"],
                               "position_early": (p.get("position_forecast") or {}).get("early_pos_norm"), "inputs": capture.get(hn)}
    if not horses_out:
        return None, "no_horses"
    return _doc(card, past, race_id, date, preds, source=source, pf=pf, unified=unified, meta=meta, horses=horses_out, z_race=z_race), "ok"


RACE_ONLY_MIN_FIELD = 8      # 学習（tp53_dataset.load_pf）は 8 頭未満のレースに位置取り・z_hat を付けない


def _past_count(past: dict | None) -> int:
    """馬柱の頭数（学習 tp53_dataset.load_pf の len(ents) と同じく、馬柱に載った馬を数える。馬柱が無ければ 0）。"""
    return sum(1 for e in (past or {}).get("entries") or [] if isinstance(e, dict))


def _known_inputs(preds: list[dict], z_race: float) -> dict[str, Any]:
    """race_only の保存の horses: 位置取り予想のある馬だけ、inputs（pf_early・pf_rank・z_hat）を持つ（tracking_v2 は無い）。"""
    out: dict[str, Any] = {}
    for p in preds:
        f = p.get("position_forecast") or {}
        if p.get("horse_number") is None or f.get("early_pos_norm") is None:
            continue
        est = f.get("early_pos_est")
        out[str(int(p["horse_number"]))] = {"horse_id": p.get("horse_id"), "position_early": f.get("early_pos_norm"),
                                             "inputs": {"pf_early": float(f["early_pos_norm"]), "pf_rank": None if est is None else float(est), "z_hat": z_race}}
    return out


def _race_z(pf: dict, preds: list[dict], course: dict) -> float | None:
    """レース単位の想定ペース z_hat（学習と同じ式。tracking_difficulty_v2.race_z_hat）。材料が無ければ None。"""
    if not pf or not preds:
        return None
    try:
        from src.pipeline.inference import pace_coherence as pcoh
        from src.pipeline.models.tracking_difficulty_v2 import race_z_hat

        field = int(pf.get("field_size") or len(preds))
        z = race_z_hat(pcoh.load_pace_v2(), pf, field, course.get("surface"), course.get("distance"), course.get("track_condition"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("レース単位の z_hat の計算に失敗: %s", exc)
        return None
    return None if z is None or z != z else float(z)


def _doc(card: dict, past: dict | None, race_id: str, date: str, preds: list[dict], *, source: str, pf: dict, unified: Any, meta: Any,
         horses: dict[str, Any], z_race: float | None, race_only: str | None = None) -> dict:
    doc = {
        "version": VERSION, "race_id": race_id, "date": _day(date), "venue": card.get("venue"), "surface": card.get("surface"),
        "distance": card.get("distance"), "track_condition": card.get("track_condition"),
        "n_horses": len(preds), "n_with_v2": sum(1 for h in horses.values() if h.get("tracking_v2")), "horse_numbers": numbers_of(preds),
        "computed_at": _now_jst(), "source": source, "card_fetched_at": _scraped(card), "past_fetched_at": _scraped(past),
        "model": model_info(),
        "pace_forecast": {k: (pf or {}).get(k) for k in ("level", "label", "front_runner_count", "stalker_count", "early_pressure", "known_horses", "field_size")},
        "pace_unified": unified, "tracking_v2_meta": meta, "horses": horses, "z_hat": z_race,
    }
    if race_only:
        doc["race_only"] = race_only    # 馬ごとの v2 が無い（理由）。API の読み出し・突き合わせの集計はこの保存を使わない
    return doc


def tv2_load_model():
    from src.pipeline.models.tracking_difficulty_v2 import load_model

    return load_model()


def build_for_race(storage: Any, race_id: str, *, source: str = "raceday_eve") -> dict | None:
    return build_for_race_reason(storage, race_id, source=source)[0]


def build_for_race_reason(storage: Any, race_id: str, *, source: str = "raceday_eve") -> tuple[dict | None, str]:
    card = storage.load("race_shutuba", race_id) or {}
    if not card.get("entries"):
        return None, "no_card"
    past = storage.load("race_shutuba_past", race_id)
    return build_doc_reason(card, past, race_id, source=source)


def precompute_tracking_v2_for_race_ids(race_ids: list[str], storage: Any = None, *, source: str = "raceday_eve", overwrite: bool = True) -> dict[str, Any]:
    """前夜 eve 等: 出馬表・馬柱の保存後に、追走難度 v2 を一括で作って保存する。1 レースの失敗は他に波及しない。

    KEIBA_TRACKING_V2 が無効なら何もしない。overwrite=False（バックフィル）は、保存済みのレースを作り直さない。
    """
    stats: dict[str, Any] = {"ok": 0, "skip": 0, "fail": 0, "total": len(race_ids), "skip_reasons": {}}
    if not enabled():
        stats["status"] = "disabled"
        return stats
    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()
    for rid in race_ids:
        try:
            card = storage.load("race_shutuba", rid) or {}
            date = str(card.get("date") or "")
            # race_only（レース単位の z_hat だけ）の保存は「保存済み」と見なさない（後から馬柱がそろえば v2 が作れる）
            if not overwrite and date and path_for(rid, date).exists() and not (_read(path_for(rid, date)) or {}).get("race_only"):
                stats["skip"] += 1
                stats["skip_reasons"]["exists"] = stats["skip_reasons"].get("exists", 0) + 1
                continue
            doc, reason = build_for_race_reason(storage, rid, source=source)
            if doc is None:
                stats["skip"] += 1
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                continue
            _write(path_for(rid, doc["date"]), doc)
            if reason == "race_only":    # v2 は作れていない（レース単位の z_hat だけ）。ok には数えない
                stats["race_only"] = stats.get("race_only", 0) + 1
            else:
                stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("追走難度 v2 の保存に失敗 %s: %s", rid, exc)
            stats["fail"] += 1
    return stats


# ───────────────────────── 読み出し（予測 API）

def _apply(resp: dict, saved: dict, *, source: str) -> dict:
    by_no = saved.get("horses") or {}
    for p in resp.get("predictions") or []:
        try:
            hn = None if p.get("horse_number") is None else str(int(p["horse_number"]))
        except Exception:  # noqa: BLE001
            hn = None
        h = by_no.get(hn) if hn is not None else None
        if h and h.get("tracking_v2"):
            p["tracking_v2"] = dict(h["tracking_v2"])
    meta = dict(saved.get("tracking_v2_meta") or {})
    meta.update({"source": source, "computed_at": saved.get("computed_at"), "store_version": saved.get("version")})
    resp["tracking_v2_meta"] = meta
    return resp


def attach_tracking_v2_cached(resp: dict, race_id: str, past_doc: dict | None, race_date: str | None, course: dict | None, card: dict | None) -> dict:
    """予測 API の応答に各馬の tracking_v2 を足す。保存（前夜の固定値）を優先し、無い・馬番の集合が変わったときはその場で計算する。

    無効（KEIBA_TRACKING_V2 が無い）なら何もしない。失敗しても応答は返す。
    """
    if not enabled():
        return resp
    from src.pipeline.models.tracking_difficulty_v2 import attach_tracking_v2_to_response

    date = _day(race_date or (card or {}).get("date"))
    policy = recompute_policy()
    nums = numbers_of(resp.get("predictions") or [])
    try:
        if policy != "always" and date:
            saved = load_saved(race_id, date)
            if saved is not None and saved.get("race_only"):
                saved = None    # レース単位の z_hat だけの保存（v6e の前段用）: API は従来どおりその場で計算する
            if saved is not None:
                same = sorted(saved.get("horse_numbers") or []) == nums
                if same or policy == "none":
                    return _apply(resp, saved, source="saved" if same else "saved_stale")
                rec = load_saved(race_id, date, recomputed=True)
                if rec is not None and not rec.get("race_only") and sorted(rec.get("horse_numbers") or []) == nums:
                    return _apply(resp, rec, source="recomputed_saved")
                # 馬番の集合が変わった（取消・除外・追加）: その場で計算し、別に保存する（前夜の保存は残す）
                doc = build_doc(card or {}, past_doc, race_id, source="recomputed", resp=resp)
                if doc is not None and not doc.get("race_only"):   # race_only は API が読まないので書かない（毎回の書き直しを防ぐ）
                    try:
                        _write(path_for(race_id, doc["date"] or date, recomputed=True), doc)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("追走難度 v2 の計算し直しの保存に失敗 %s: %s", race_id, exc)
                if resp.get("tracking_v2_meta"):
                    resp["tracking_v2_meta"] = {**resp["tracking_v2_meta"], "source": "recomputed", "computed_at": _now_jst()}
                return resp
    except Exception as exc:  # noqa: BLE001
        logger.warning("追走難度 v2 の保存の読み出しに失敗 %s: %s", race_id, exc)
    out = attach_tracking_v2_to_response(resp, past_doc, race_date, course, card)
    if out.get("tracking_v2_meta"):
        out["tracking_v2_meta"] = {**out["tracking_v2_meta"], "source": "live", "computed_at": _now_jst()}
    return out
