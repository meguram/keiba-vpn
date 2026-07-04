"""
レース質推定（ラベルなし・結果連動）

正解の「レース質」ラベルは持たず、各馬の事前プロファイル（血統因子・タイム指数・
過去成績・バロメーター等）と着順から、レースごとに「どのタイプが好走したか」を
非負最小二乗 (NNLS) で混合比として推定する。

- 8 軸: ユーザー定義のレース質タイプに対応する馬側スコア（説明変数）
- 第9カテゴリ: フィット悪・小頭数・分布の平坦さから「判定困難」確率を付与

距離・芝/ダ別セグメント:
  芝1200m と ダ1800m などでは「どの軸が意味を持つか」が違うため、
  segment_key（例: 芝_1200-1399）ごとに 8 軸への事前倍率を掛け、
  条件帯に合わせたスケールで NNLS する（別パラメータ学習ではなくドメイン事前）。

馬の過去成績 (horse_result.race_history) は距離帯（短距離〜長距離）別に集計し、
今走と同じ帯の上がり傾向・複勝率を 8 軸へ反映する（帯ごとに脚質が異なる前提）。

ラップ・ペース形状:
  race_result の前半3F/後半3F・セクションデータから
  消耗型・瞬発（溜め）型・均速リズム型の指標を作り、該当軸の倍率を上げる。
  race_lap / race_result_lap は extract_lap_times_from_blob で実JSONの揺れに対応。

データ駆動セグメント補正:
  data/research/race_quality_priors.json の segment_extra_log_mult を
  research/tune_race_quality_priors.py で推定可能（NNLS残差最小化）。

参照: research/sire_factor_stats.AXIS_IDS（種牡馬統計ベースの血統軸）
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from src.research.pedigree.sire_factor_aptitude import ANCESTOR_SLOTS, DEFAULT_WEIGHTS, _compute_aptitude_fast
from src.research.pedigree.sire_factor_stats import AXES, AXIS_IDS, load_sire_factor_stats

logger = logging.getLogger(__name__)

# データ駆動で上書きするセグメント事前倍率（8軸の log 補正）。チューニングスクリプトが生成。
PRIORS_JSON_PATH = Path(__file__).resolve().parents[3] / "data" / "local" / "research" / "race_quality_priors.json"
_prior_json_cache: dict[str, Any] | None = None
_prior_json_mtime: float = 0.0

# 8 軸 + 1（判定困難）— UI・API での固定順
QUALITY_AXES: list[dict[str, str]] = [
    {
        "id": "closing",
        "label_ja": "上がり・ラスト脚",
        "hint": "バロメ上がり指数と、今走と同じ距離帯（短距離〜長距離）に切った戦歴の上がり傾向・穴好走が上位馬と揃った度合い",
    },
    {
        "id": "baseline_speed",
        "label_ja": "基礎スピード",
        "hint": "血統TS・平均/最高タイム指数・距離指数・追走指数など、総合的なスピード・能力が好走と結びついた度合い",
    },
    {
        "id": "stamina",
        "label_ja": "スタミナ（消耗戦）",
        "hint": "長距離実績と持続系血統が活きた消耗寄りの展開の度合い",
    },
    {
        "id": "burst",
        "label_ja": "瞬発力・ギアチェンジ",
        "hint": "血統のギアチェンジ・差し脚が評価された度合い",
    },
    {
        "id": "sustain_run",
        "label_ja": "持続力（長く脚を使う）",
        "hint": "TS持続・安定性が活きた度合い（スタミナ軸と補完的）",
    },
    {
        "id": "position_draw",
        "label_ja": "位置取り・枠順",
        "hint": "スタート・内枠等、ポジションが結果と結びついた度合い",
    },
    {
        "id": "class_flat",
        "label_ja": "能力至上（適性バイアス小）",
        "hint": "通算実績に加え、今走距離帯での複勝率が高い馬が上位に来た度合い（重賞ほど解釈しやすい）",
    },
    {
        "id": "mud_going",
        "label_ja": "道悪・タフ馬場",
        "hint": "道悪適性血統・重め実績が活きた度合い",
    },
    {
        "id": "unknown",
        "label_ja": "判定困難",
        "hint": "データ不足・フィット不良・分布が平坦なときに質が特定しにくい",
    },
]

N_TYPE8 = 8

_IDX = {k: i for i, k in enumerate(AXIS_IDS)}

# 列順: closing, baseline, stamina, burst, sustain_run, position, class_flat, mud
_ONES8 = np.ones(N_TYPE8, dtype=np.float64)


def history_distance_band(distance_m: int) -> str:
    """
    馬の戦歴を距離帯で分類（芝ダは問わずメートルのみ）。
    短距離 / マイル / 中距離 / 中長距離 / 長距離
    """
    d = int(distance_m or 0)
    if d <= 0:
        return "不明"
    if d < 1400:
        return "短距離"
    if d < 1800:
        return "マイル"
    if d < 2200:
        return "中距離"
    if d < 2800:
        return "中長距離"
    return "長距離"


HISTORY_BAND_KEYS: tuple[str, ...] = (
    "短距離",
    "マイル",
    "中距離",
    "中長距離",
    "長距離",
)


def distance_surface_segment(surface: str, distance_m: int) -> str:
    """
    芝/ダ/障 + 距離帯。同一帯で事前倍率を共有（完全別モデルではなく条件付きスケーリング）。
    """
    s = (surface or "").strip()
    if "ダ" in s:
        sk = "ダート"
    elif "障" in s:
        sk = "障害"
    elif "芝" in s:
        sk = "芝"
    else:
        sk = "その他"

    d = int(distance_m or 0)
    if d <= 0:
        return f"{sk}_距離不明"
    if d < 1200:
        band = "~1199"
    elif d < 1400:
        band = "1200-1399"
    elif d < 1600:
        band = "1400-1599"
    elif d < 1800:
        band = "1600-1799"
    elif d < 2000:
        band = "1800-1999"
    elif d < 2200:
        band = "2000-2199"
    elif d < 2800:
        band = "2200-2799"
    else:
        band = "2800+"
    return f"{sk}_{band}"


def _load_priors_json() -> dict[str, Any]:
    """race_quality_priors.json をmtimeでキャッシュ読み。"""
    global _prior_json_cache, _prior_json_mtime
    try:
        if not PRIORS_JSON_PATH.is_file():
            return {}
        mt = PRIORS_JSON_PATH.stat().st_mtime
        if _prior_json_cache is not None and mt == _prior_json_mtime:
            return _prior_json_cache
        with open(PRIORS_JSON_PATH, "r", encoding="utf-8") as f:
            _prior_json_cache = json.load(f)
        _prior_json_mtime = mt
        return _prior_json_cache if isinstance(_prior_json_cache, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("race_quality priors JSON: %s", e)
        return {}


def list_tuned_prior_segments() -> list[str]:
    """チューニング済み segment_key 一覧。"""
    data = _load_priors_json()
    m = data.get("segment_extra_log_mult") or {}
    return sorted(m.keys()) if isinstance(m, dict) else []


def _hand_segment_archetype_prior(segment_key: str) -> np.ndarray:
    """
    手設計のセグメント事前倍率（ドメイン知識）。
    """
    w = _ONES8.copy()
    if segment_key.startswith("芝_") or segment_key.startswith("ダート_") or segment_key.startswith("障害_"):
        pass
    for tag, vec in (
        ("~1199", [1.12, 0.96, 0.78, 1.08, 0.9, 1.14, 1.02, 1.0]),
        ("1200-1399", [1.18, 0.98, 0.72, 1.12, 0.86, 1.2, 1.0, 1.0]),
        ("1400-1599", [1.06, 1.08, 0.85, 1.02, 0.98, 1.05, 1.04, 1.0]),
        ("1600-1799", [1.0, 1.12, 0.95, 0.98, 1.05, 1.0, 1.06, 1.0]),
        ("1800-1999", [0.92, 1.08, 1.12, 0.9, 1.1, 0.94, 1.02, 1.0]),
        ("2000-2199", [0.86, 1.04, 1.22, 0.84, 1.16, 0.88, 1.04, 1.0]),
        ("2200-2799", [0.8, 1.0, 1.3, 0.78, 1.24, 0.82, 1.08, 1.0]),
        ("2800+", [0.76, 0.98, 1.36, 0.72, 1.3, 0.78, 1.1, 1.02]),
    ):
        if tag in segment_key:
            w *= np.array(vec, dtype=np.float64)
            break

    if segment_key.startswith("ダート_"):
        w *= np.array([0.98, 1.12, 1.04, 1.02, 1.0, 1.1, 1.0, 1.06], dtype=np.float64)
    elif segment_key.startswith("障害_"):
        w *= np.array([0.85, 1.0, 1.35, 0.75, 1.25, 0.8, 1.05, 1.08], dtype=np.float64)

    return np.maximum(w, 0.25)


def segment_archetype_prior(segment_key: str) -> np.ndarray:
    """
    手設計 ×（任意）race_quality_priors.json の segment_extra_log_mult。
    JSON は過去レースの NNLS 残差最小化で推定した軸ごとの log 補正。
    """
    w = _hand_segment_archetype_prior(segment_key)
    data = _load_priors_json()
    raw = (data.get("segment_extra_log_mult") or {}).get(segment_key)
    if raw is None:
        return w
    try:
        ex = np.array([float(x) for x in raw], dtype=np.float64)
        if ex.shape != (N_TYPE8,):
            return w
        ex = np.clip(ex, -1.0, 1.0)
        w = w * np.exp(ex)
    except (TypeError, ValueError):
        return w
    return np.maximum(w, 0.25)


def _parse_lap_list(raw: Any) -> list[float]:
    if not raw:
        return []
    if isinstance(raw, list):
        out = []
        for x in raw:
            try:
                v = float(x)
                if math.isfinite(v) and v > 0:
                    out.append(v)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, str):
        return _parse_lap_string(raw)
    return []


def _parse_lap_string(raw: str) -> list[float]:
    """文字列化されたセクションタイム（例 12.3-11.4-10.5）を分解。"""
    if not raw or not isinstance(raw, str):
        return []
    s = raw.replace("－", "-").replace("—", "-").replace("・", ".")
    nums = re.findall(r"\d+\.?\d*", s)
    out: list[float] = []
    for n in nums:
        try:
            v = float(n)
            if math.isfinite(v) and 2.0 < v < 250.0:
                out.append(v)
        except ValueError:
            continue
    return out


def extract_lap_times_from_blob(obj: Any, depth: int = 0) -> list[float]:
    """
    race_lap / race_result_lap 等、実JSONの揺れに対応してセクション秒を抽出。
    ネストした dict・entries 配列・単一文字列を再帰探索。
    """
    if depth > 6 or obj is None:
        return []
    if isinstance(obj, str):
        return _parse_lap_string(obj)
    if isinstance(obj, (int, float)):
        v = float(obj)
        return [v] if math.isfinite(v) and v > 2.0 else []
    if isinstance(obj, list):
        if not obj:
            return []
        if isinstance(obj[0], dict):
            merged: list[float] = []
            for e in obj:
                merged.extend(extract_lap_times_from_blob(e, depth + 1))
            return merged
        return _parse_lap_list(obj)
    if isinstance(obj, dict):
        for key in (
            "lap_times", "section_times", "laps", "splits",
            "LapTime", "lap", "sections", "split_times", "corner_times",
            "race_lap", "data", "values", "rows", "entries",
        ):
            if key in obj:
                got = extract_lap_times_from_blob(obj[key], depth + 1)
                if len(got) >= 3:
                    return got
        for v in obj.values():
            if isinstance(v, (dict, list, str)):
                got = extract_lap_times_from_blob(v, depth + 1)
                if len(got) >= 4:
                    return got
    return []


def _merge_lap_from_dict(lap_times: list[float], pace: dict[str, Any], src: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    """補助 JSON からラップ・ペースをマージ。"""
    if not isinstance(src, dict):
        return lap_times, pace
    alt = _parse_lap_list(
        src.get("lap_times")
        or src.get("section_times")
        or src.get("laps")
        or src.get("splits")
    )
    if len(alt) < 3:
        ext = extract_lap_times_from_blob(src)
        if len(ext) > len(alt):
            alt = ext
    if len(alt) >= len(lap_times):
        lap_times = alt
    p2 = src.get("pace")
    if not pace.get("first_half_3f") and isinstance(p2, dict):
        if p2.get("first_half_3f") and p2.get("second_half_3f"):
            pace = {**pace, **p2}
    return lap_times, pace


def _resolve_lap_pace(
    storage: Any,
    race_id: str,
    rr: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    """race_result 優先。欠ける場合 race_lap / race_result_lap を試す。"""
    lap_times = _parse_lap_list(rr.get("lap_times"))
    pace = dict(rr.get("pace") or {})

    if storage:
        if not lap_times or len(lap_times) < 3 or not pace.get("first_half_3f"):
            rl = storage.load("race_lap", race_id)
            lap_times, pace = _merge_lap_from_dict(lap_times, pace, rl or {})
        if not lap_times or len(lap_times) < 3 or not pace.get("first_half_3f"):
            rrl = storage.load("race_result_lap", race_id)
            lap_times, pace = _merge_lap_from_dict(lap_times, pace, rrl or {})

    return lap_times, pace


def going_archetype_multiplier(track_condition: str) -> np.ndarray:
    """馬場が重いほど道悪軸の事前倍率を上げる。"""
    m = _ONES8.copy()
    tc = (track_condition or "").strip()
    if tc in ("稍重", "重", "不良"):
        m[7] *= 1.38
    elif tc == "良":
        m[7] *= 0.9
    return np.maximum(m, 0.2)


def get_race_quality_meta() -> dict[str, Any]:
    """UI・連携用の固定メタデータ。"""
    bands = [
        "~1199", "1200-1399", "1400-1599", "1600-1799",
        "1800-1999", "2000-2199", "2200-2799", "2800+",
    ]
    surfaces = ["芝", "ダート", "障害", "その他"]
    tuned = list_tuned_prior_segments()
    pj = PRIORS_JSON_PATH
    return {
        "version": 1,
        "priors_json_path": str(pj),
        "priors_json_exists": pj.is_file(),
        "tuned_segment_keys": tuned,
        "tuning_hint_ja": (
            "セグメント別のデータ駆動補正: "
            "python -m src.research.race.tune_race_quality_priors を実行すると "
            "race_quality_priors.json が生成・更新され、推定時に log 補正が乗ります。"
        ),
        "axes": QUALITY_AXES,
        "segment_surfaces": surfaces,
        "segment_distance_bands": bands,
        "history_distance_bands_ja": list(HISTORY_BAND_KEYS),
        "history_distance_bands_note_ja": (
            "馬戦歴は短距離/マイル/中距離/中長距離/長距離に分け、今走と同じ帯の上がり・複勝率を優先して 8 軸に反映する。"
        ),
        "segment_example_keys": [f"芝_{b}" for b in bands[:3]]
        + [f"ダート_{b}" for b in bands[4:6]],
        "pace_metrics": [
            {
                "id": "grind_index",
                "label_ja": "消耗寄り",
                "hint": "前半3Fが相対的に速く後半3Fが遅い（秒が大きい）ほど上がる",
            },
            {
                "id": "burst_index",
                "label_ja": "溜め・瞬発寄り",
                "hint": "前半が遅めで後半が伸びている形状ほど上がる",
            },
            {
                "id": "lap_evenness",
                "label_ja": "均速リズム",
                "hint": "セクションタイムのばらつきが小さいほど上がる（持続ペース）",
            },
            {
                "id": "has_half_pace",
                "label_ja": "前半後半3Fデータ",
                "hint": "1.0 でペース行から 3F ペアを取得済み",
            },
            {
                "id": "lap_sections",
                "label_ja": "ラップ分割数",
                "hint": "取得したセクションタイムの本数",
            },
        ],
        "api": {
            "meta": "/api/race-quality/meta",
            "day": "/api/race-quality/day?date=YYYYMMDD",
            "race": "/api/race-quality/race?race_id=",
            "entrants_aptitude": "/api/race-quality/entrants-aptitude?race_id=",
        },
        "aptitude_store_category": "horse_race_quality_aptitude",
        "aptitude_store_note_ja": (
            "馬ごとの血統8軸＋距離帯別戦歴を GCS（horse カテゴリと同階層）に JSON 保存。"
            "馬戦績・5代血統の scraped_at が新しくなると自動で再計算・上書き。"
        ),
        "page": "/race-quality",
    }


def compute_pace_shape(
    pace: dict[str, Any],
    lap_times: list[float],
    distance_m: int,
) -> dict[str, float]:
    """
    ラップ形状の特徴（0〜1 正規化を目安）。

    - grind_index: 前半が相対的に速く後半が落ちる（時間が伸びる）＝消耗寄り
    - burst_index: 前半遅め・後半まだ速い＝溜め→脚質勝負寄り
    - lap_evenness: セクションタイムのばらつきが小さい＝均速・持続リズム寄り
    """
    out: dict[str, float] = {
        "grind_index": 0.5,
        "burst_index": 0.5,
        "lap_evenness": 0.5,
        "has_half_pace": 0.0,
        "lap_sections": float(len(lap_times)),
    }

    fh = pace.get("first_half_3f")
    sh = pace.get("second_half_3f")
    if fh is not None and sh is not None:
        try:
            fhv = float(fh)
            shv = float(sh)
            if fhv > 0 and shv > 0:
                out["has_half_pace"] = 1.0
                # 同じ3F×2 の秒: 前半が小さいほどハイペース
                tot = fhv + shv
                # 後半の方が秒が大きいほど消耗（後半遅い）
                delta = (shv - fhv) / tot
                out["grind_index"] = _clamp01(0.5 + 2.8 * delta)
                out["burst_index"] = _clamp01(0.5 - 2.2 * delta)
        except (TypeError, ValueError):
            pass

    if len(lap_times) >= 4:
        arr = np.array(lap_times, dtype=np.float64)
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        cv = sd / (mu + 1e-6)
        out["lap_evenness"] = _clamp01(1.0 - min(1.2, cv * 2.5))

        if out["has_half_pace"] < 0.5:
            n = len(arr)
            k = max(2, n // 3)
            early = float(np.sum(arr[:k]))
            late = float(np.sum(arr[-k:]))
            t = early + late
            if t > 0:
                delta2 = (late - early) / t
                out["grind_index"] = _clamp01(0.5 + 1.8 * delta2)
                out["burst_index"] = _clamp01(0.5 - 1.5 * delta2)

    # 長距離では「均一ラップ」の解釈を少し強める（距離補正）
    if distance_m >= 2000 and len(lap_times) >= 4:
        out["lap_evenness"] = _clamp01(out["lap_evenness"] * 1.08)

    return out


def pace_archetype_multipliers(pf: dict[str, float]) -> np.ndarray:
    """ペース形状に応じて 8 軸の事前倍率を調整。"""
    m = _ONES8.copy()
    g = float(pf.get("grind_index") or 0.5)
    b = float(pf.get("burst_index") or 0.5)
    ev = float(pf.get("lap_evenness") or 0.5)
    # 中心 0.5 からの偏差
    dg = max(0.0, g - 0.5)
    db = max(0.0, b - 0.5)
    de = max(0.0, ev - 0.5)

    m[2] *= 1.0 + 0.55 * dg
    m[1] *= 1.0 + 0.28 * dg
    m[3] *= 1.0 + 0.5 * db
    m[0] *= 1.0 + 0.38 * db
    m[4] *= 1.0 + 0.52 * de
    m[1] *= 1.0 + 0.18 * de
    return np.maximum(m, 0.2)


def _sf(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _si(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_ped(val: float) -> float:
    """種牡馬軸は概ね偏差値想定。50 を中心に 0–1 に押し込む。"""
    return _clamp01((val - 45.0) / 25.0)


def _shrink_last3f_fast(last3f_fast: float, n: int) -> float:
    """出走が多いほどキャリア平均上がりを 0.5 へ収縮。"""
    if n >= 12:
        return 0.5 + (last3f_fast - 0.5) * 0.45
    if n >= 7:
        return 0.5 + (last3f_fast - 0.5) * 0.65
    if n >= 4:
        return 0.5 + (last3f_fast - 0.5) * 0.82
    return last3f_fast


def _last3f_fast_from_values(l3f_vals: list[float]) -> float:
    if not l3f_vals:
        return 0.5
    avg_l3f = sum(l3f_vals) / len(l3f_vals)
    return _clamp01((42.0 - avg_l3f) / 12.0)


def _aggregate_band(recs: list[dict]) -> dict[str, float]:
    """1 距離帯に属する戦歴のみで集計（今走と同条件の脚質・上がり傾向の目安）。"""
    n = len(recs)
    if n == 0:
        return {
            "n": 0.0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "last3f_fast": 0.5,
            "l3f_top_tertile_rate": 0.5,
            "surprise_top3_rate": 0.0,
        }

    wins = sum(1 for r in recs if _si(r.get("finish_position"), 0) == 1)
    top3 = sum(1 for r in recs if _si(r.get("finish_position"), 99) in (1, 2, 3))
    l3f_vals = [_sf(r.get("last_3f")) for r in recs if _sf(r.get("last_3f")) > 0]
    last3f_fast = _shrink_last3f_fast(_last3f_fast_from_values(l3f_vals), n)

    # 当該帯内で「自分の上がりが速いレース」の割合（レース間比較の近似）
    l3f_top_tertile_rate = 0.5
    if len(l3f_vals) >= 3:
        sorted_vals = sorted(l3f_vals)
        k = max(1, math.ceil(len(sorted_vals) / 3))
        cut = sorted_vals[k - 1]
        fast_count = sum(1 for v in l3f_vals if v <= cut)
        l3f_top_tertile_rate = _clamp01(fast_count / len(l3f_vals))
    elif len(l3f_vals) >= 1:
        l3f_top_tertile_rate = 0.55

    # 3着以内のうち、人気薄（7番人気以降）で入った割合＝「その条件でたまに爆発」
    top3_recs = [r for r in recs if _si(r.get("finish_position"), 99) in (1, 2, 3)]
    if top3_recs:
        surprise_top3_rate = sum(1 for r in top3_recs if _si(r.get("popularity"), 0) >= 7) / len(
            top3_recs
        )
    else:
        surprise_top3_rate = 0.0

    return {
        "n": float(n),
        "win_rate": wins / n,
        "top3_rate": top3 / n,
        "last3f_fast": last3f_fast,
        "l3f_top_tertile_rate": l3f_top_tertile_rate,
        "surprise_top3_rate": float(surprise_top3_rate),
    }


def _history_features(records: list[dict]) -> dict[str, Any]:
    recs = records or []
    n = len(recs)
    by_band_raw: dict[str, list[dict]] = {k: [] for k in HISTORY_BAND_KEYS}
    for r in recs:
        b = history_distance_band(_si(r.get("distance"), 0))
        if b in by_band_raw:
            by_band_raw[b].append(r)

    by_band: dict[str, dict[str, float]] = {k: _aggregate_band(v) for k, v in by_band_raw.items()}
    band_long_stats = _aggregate_band(
        by_band_raw["中長距離"] + by_band_raw["長距離"]
    )

    if n == 0:
        return {
            "starts": 0.0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "long_top3_rate": 0.0,
            "heavy_top3_rate": 0.0,
            "last3f_fast": 0.5,
            "by_band": by_band,
            "band_long_stats": band_long_stats,
        }

    wins = sum(1 for r in recs if _si(r.get("finish_position"), 0) == 1)
    top3 = sum(1 for r in recs if _si(r.get("finish_position"), 99) in (1, 2, 3))
    long_runs = [r for r in recs if _si(r.get("distance"), 0) >= 1800]
    lt3 = sum(
        1
        for r in long_runs
        if _si(r.get("finish_position"), 99) in (1, 2, 3)
    )
    heavy_runs = [
        r
        for r in recs
        if str(r.get("track_condition") or "") in ("稍重", "重", "不良")
    ]
    ht3 = sum(
        1
        for r in heavy_runs
        if _si(r.get("finish_position"), 99) in (1, 2, 3)
    )
    l3f_vals = [
        _sf(r.get("last_3f"))
        for r in recs
        if _sf(r.get("last_3f")) > 0
    ]
    last3f_fast = _shrink_last3f_fast(_last3f_fast_from_values(l3f_vals), n)

    return {
        "starts": float(n),
        "win_rate": wins / n,
        "top3_rate": top3 / n,
        "long_top3_rate": (lt3 / len(long_runs)) if long_runs else 0.0,
        "heavy_top3_rate": (ht3 / len(heavy_runs)) if heavy_runs else 0.0,
        "last3f_fast": last3f_fast,
        "by_band": by_band,
        "band_long_stats": band_long_stats,
    }


def _entries_by_num_id(entries: list[dict] | None) -> tuple[dict[int, dict], dict[str, dict]]:
    by_num: dict[int, dict] = {}
    by_id: dict[str, dict] = {}
    for e in entries or []:
        hn = _si(e.get("horse_number"), 0)
        hid = (e.get("horse_id") or "").strip()
        if hn > 0:
            by_num[hn] = e
        if hid:
            by_id[hid] = e
    return by_num, by_id


def _lookup_aux(
    by_num: dict[int, dict],
    by_id: dict[str, dict],
    result_entry: dict,
) -> dict | None:
    hid = (result_entry.get("horse_id") or "").strip()
    if hid and hid in by_id:
        return by_id[hid]
    hn = _si(result_entry.get("horse_number"), 0)
    if hn > 0 and hn in by_num:
        return by_num[hn]
    return None


def _column_minmax_nonneg(X: np.ndarray) -> np.ndarray:
    """列ごとに [0,1] に線形変換（NNLS 用）。"""
    out = np.zeros_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        lo = float(np.min(col))
        hi = float(np.max(col))
        if hi - lo < 1e-9:
            out[:, j] = 0.5
        else:
            out[:, j] = (col - lo) / (hi - lo)
    return np.clip(out, 1e-6, 1.0)


def _build_archetype_row(
    ped: np.ndarray,
    hist: dict[str, Any],
    speed_max: float,
    speed_avg: float,
    speed_distance: float,
    idx_start: float,
    idx_chase: float,
    idx_closing: float,
    bracket: int,
    field_size: int,
    race_distance: int,
) -> np.ndarray:
    """8 次元の非負に近い生スコア（列 min-max 前）。"""
    ts = float(ped[_IDX["ts"]])
    gear = float(ped[_IDX["gear_change"]])
    sustain = float(ped[_IDX["ts_sustain"]])
    wet = float(ped[_IDX["wet_turf"]])
    start_sp = float(ped[_IDX["start_speed"]])
    consistency = float(ped[_IDX["consistency"]])
    dirt = float(ped[_IDX["dirt"]])

    rk = history_distance_band(race_distance)
    bb = (hist.get("by_band") or {}).get(rk) or {}
    n_bb = int(bb.get("n", 0))
    l3f_g = float(hist.get("last3f_fast", 0.5))
    l3f_b = float(bb.get("last3f_fast", 0.5))
    l3f_tt = float(bb.get("l3f_top_tertile_rate", 0.5))
    surp = float(bb.get("surprise_top3_rate", 0.0))

    # 今走の距離帯に該当する戦歴が十分あるとき、帯別の上がり傾向を優先（短距離と長距離で別物にする）
    if n_bb >= 4:
        l3f_mix = 0.58 * l3f_b + 0.42 * l3f_g
        l3f_signal = 0.68 * l3f_mix + 0.32 * l3f_tt
    elif n_bb >= 2:
        l3f_mix = 0.45 * l3f_b + 0.55 * l3f_g
        l3f_signal = 0.72 * l3f_mix + 0.28 * l3f_tt
    else:
        l3f_signal = l3f_g

    # closing: バロメ上がり + 距離帯を切った実績上がり + 帯内で繰り返し上がりが尖るか + 穴での好走（その条件のベスト脚）
    closing = (
        _clamp01(idx_closing / 120.0) * 0.44
        + l3f_signal * 0.42
        + _clamp01(surp * 1.1) * 0.14
    )

    # baseline: 血統のTS（上がり3F偏差の種牡馬統計）とタイム指数の核を集約
    baseline = (
        _norm_ped(ts) * 0.30
        + _clamp01(speed_avg / 120.0) * 0.28
        + _clamp01(speed_distance / 120.0) * 0.20
        + _clamp01(speed_max / 120.0) * 0.14
        + _clamp01(idx_chase / 120.0) * 0.08
    )

    bls = hist.get("band_long_stats") or {}
    n_bl = int(bls.get("n", 0))
    dist_bonus = 1.0 if race_distance >= 2000 else 0.35
    if race_distance >= 2200 and n_bl >= 3:
        stamina = (
            float(hist.get("long_top3_rate", 0.0)) * 0.36
            + float(bls.get("top3_rate", 0.0)) * 0.24
            + _norm_ped(sustain) * 0.33
            + dist_bonus * 0.07
        )
    else:
        stamina = (
            float(hist.get("long_top3_rate", 0.0)) * 0.55
            + _norm_ped(sustain) * 0.35
            + dist_bonus * 0.1
        )

    if n_bb >= 2:
        burst = _clamp01(_norm_ped(gear) * 0.88 + surp * 0.32)
    else:
        burst = _norm_ped(gear)

    sustain_run = _norm_ped(sustain) * 0.55 + _norm_ped(consistency) * 0.45

    bracket = max(1, bracket) if bracket > 0 else max(1, field_size // 2)
    fs = max(field_size, 1)
    inner = (fs - bracket + 1) / fs
    position = _norm_ped(start_sp) * 0.55 + _clamp01(inner) * 0.45

    starts = max(float(hist.get("starts", 1.0)), 1.0)
    gwr = float(hist.get("win_rate", 0.0))
    gtr = float(hist.get("top3_rate", 0.0))
    if n_bb >= 4:
        eff_wr = 0.52 * gwr + 0.48 * float(bb.get("win_rate", 0.0))
        eff_tr = 0.48 * gtr + 0.52 * float(bb.get("top3_rate", 0.0))
    elif n_bb >= 2:
        eff_wr = 0.62 * gwr + 0.38 * float(bb.get("win_rate", 0.0))
        eff_tr = 0.62 * gtr + 0.38 * float(bb.get("top3_rate", 0.0))
    else:
        eff_wr = gwr
        eff_tr = gtr

    class_flat = (
        min(1.0, eff_wr * 4.0) * 0.45
        + min(1.0, eff_tr * 2.0) * 0.35
        + min(1.0, math.log1p(starts) / math.log1p(25.0)) * 0.2
    )

    mud = _norm_ped(wet) * 0.65 + float(hist.get("heavy_top3_rate", 0.0)) * 0.25 + _norm_ped(dirt) * 0.1

    row = np.array(
        [closing, baseline, stamina, burst, sustain_run, position, class_flat, mud],
        dtype=np.float64,
    )
    return np.maximum(row, 1e-9)


def _finish_scores(entries: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """有効な着順のみ。y は高いほど好走。"""
    rows: list[tuple[float, dict]] = []
    for e in entries:
        fp = _si(e.get("finish_position"), -1)
        if fp < 1:
            continue
        rows.append((float(fp), e))
    if len(rows) < 4:
        return np.array([]), []
    rows.sort(key=lambda x: x[0])
    n = len(rows)
    y = np.array([(n - fp + 1) / max(1, n - 1) for fp, _ in rows], dtype=np.float64)
    meta = [e for _, e in rows]
    return y, meta


def _fit_mixture(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    coef, res_norm = nnls(X, y)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2 = max(0.0, min(1.0, r2))
    return coef, r2, res_norm


def _nine_probs(coef: np.ndarray, r2: float, n: int) -> tuple[list[float], float]:
    s = float(np.sum(coef))
    if s < 1e-14:
        pi8 = np.ones(N_TYPE8, dtype=np.float64) / N_TYPE8
    else:
        pi8 = coef / s
    ent = float(-np.sum(pi8 * np.log(pi8 + 1e-12)))
    ent_norm = ent / math.log(N_TYPE8) if ent > 0 else 0.0

    p9 = (
        0.08
        + 0.42 * max(0.0, 1.0 - r2)
        + 0.22 * ent_norm
        + (0.12 if n < 7 else 0.0)
        + (0.10 if n < 5 else 0.0)
    )
    p9 = max(0.03, min(0.88, p9))
    rest = 1.0 - p9
    out = [float(pi8[i] * rest) for i in range(N_TYPE8)] + [p9]
    tot = sum(out)
    if tot > 0:
        out = [x / tot for x in out]
    return out, p9


def collect_race_xy_tensors(
    storage: Any,
    race_id: str,
    *,
    stats_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    NNLS 用の基礎行列・着順ベクトル・セグメント・ペース/馬場倍率を収集。
    チューニングスクリプト・検証用。analyze_race の入力層。
    """
    rr = storage.load("race_result", race_id)
    if not rr or not rr.get("entries"):
        return None

    y, valid_entries = _finish_scores(rr["entries"])
    n = len(valid_entries)
    if n < 4:
        return None

    stats = stats_data
    if stats is None:
        stats = load_sire_factor_stats(storage=storage)
    sires_db = stats.get("sires") or {}

    ri = storage.load("race_index", race_id) or {}
    rb = storage.load("race_barometer", race_id) or {}
    ri_n, ri_i = _entries_by_num_id(ri.get("entries"))
    rb_n, rb_i = _entries_by_num_id(rb.get("entries"))

    w_arr = np.array(
        [DEFAULT_WEIGHTS[s["key"]] for s in ANCESTOR_SLOTS],
        dtype=np.float64,
    )

    field_size = _si(rr.get("field_size"), n)
    race_dist = _si(rr.get("distance"), 0)
    surface = rr.get("surface") or ""
    segment_key = distance_surface_segment(surface, race_dist)

    lap_times, pace = _resolve_lap_pace(storage, race_id, rr)
    pace_shape = compute_pace_shape(pace, lap_times, race_dist)
    pace_w = pace_archetype_multipliers(pace_shape)
    track_condition = rr.get("track_condition") or ""
    going_w = going_archetype_multiplier(track_condition)

    X_list: list[np.ndarray] = []
    for e in valid_entries:
        hid = (e.get("horse_id") or "").strip()
        idx_e = _lookup_aux(ri_n, ri_i, e)
        bo_e = _lookup_aux(rb_n, rb_i, e)

        speed_max = _sf((idx_e or {}).get("speed_max"))
        speed_avg = _sf((idx_e or {}).get("speed_avg"))
        speed_distance = _sf((idx_e or {}).get("speed_distance"))
        idx_start = _sf((bo_e or {}).get("index_start"))
        idx_chase = _sf((bo_e or {}).get("index_chase"))
        idx_closing = _sf((bo_e or {}).get("index_closing"))

        ped = storage.load("horse_pedigree_5gen", hid) if hid else None
        ancestors = ped.get("ancestors", []) if ped else []
        ped_vec = _compute_aptitude_fast(ancestors, w_arr, sires_db)

        hr = storage.load("horse_result", hid) if hid else None
        hist = _history_features((hr or {}).get("race_history") or [])

        bracket = _si(e.get("bracket_number"), 0)
        row = _build_archetype_row(
            ped_vec,
            hist,
            speed_max,
            speed_avg,
            speed_distance,
            idx_start,
            idx_chase,
            idx_closing,
            bracket,
            field_size,
            race_dist,
        )
        X_list.append(row)

    base_rows = np.stack(X_list, axis=0)
    return {
        "race_id": race_id,
        "rr": rr,
        "y": y,
        "base_rows": base_rows,
        "segment_key": segment_key,
        "pace_w": pace_w,
        "going_w": going_w,
        "pace_shape": pace_shape,
        "lap_times": lap_times,
        "pace": pace,
        "track_condition": track_condition,
        "n_runners": n,
        "distance_m": race_dist,
    }


def analyze_race(
    storage: Any,
    race_id: str,
    *,
    stats_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    単一レースのレース質ベクトル（9 確率）とメタ情報を返す。
    結果未取得・頭数不足のときは None。
    """
    t = collect_race_xy_tensors(storage, race_id, stats_data=stats_data)
    if not t:
        return None

    rr = t["rr"]
    y = t["y"]
    n = t["n_runners"]
    segment_key = t["segment_key"]
    lap_times = t["lap_times"]
    pace = t["pace"]
    pace_shape = t["pace_shape"]
    track_condition = t["track_condition"]
    race_dist = t["distance_m"]

    seg_w = segment_archetype_prior(segment_key)
    pace_w = t["pace_w"]
    going_w = t["going_w"]
    row_scale = seg_w * pace_w * going_w

    X_raw = t["base_rows"] * row_scale
    X = _column_minmax_nonneg(X_raw)

    coef, r2, res_norm = _fit_mixture(X, y)
    probs, p9 = _nine_probs(coef, r2, n)

    axes_out = []
    for i, ax in enumerate(QUALITY_AXES):
        axes_out.append(
            {
                "id": ax["id"],
                "label_ja": ax["label_ja"],
                "hint": ax["hint"],
                "p": round(probs[i], 4),
            }
        )

    return {
        "race_id": race_id,
        "race_name": rr.get("race_name") or "",
        "date": rr.get("date") or "",
        "venue": rr.get("venue") or "",
        "distance": race_dist,
        "surface": rr.get("surface") or "",
        "grade": rr.get("grade") or "",
        "n_runners": n,
        "r2_fit": round(r2, 4),
        "residual_l2": round(res_norm, 6),
        "p_unknown_internal": round(p9, 4),
        "method": "nnls_mixture_segment_pace_going",
        "method_note_ja": (
            "各馬の8タイプ事前スコア（血統因子・指数・過去成績等）に、"
            "芝/ダ・距離帯・馬場状態・ラップ形状（消耗・溜め・均速）由来の倍率を掛けた上で、"
            "（任意で data/research/race_quality_priors.json のデータ駆動補正を乗算し）、"
            "着順スコアを非負最小二乗 (NNLS) で当てはめた混合比をレース質とみなす。"
            "教師ラベルは用いず、結果との整合のみで推定する。"
        ),
        "prior_tune_applied": segment_key in set(list_tuned_prior_segments()),
        "segment_key": segment_key,
        "track_condition": track_condition,
        "lap_times": [round(x, 2) for x in lap_times[:24]],
        "pace_harvest": {
            "first_half_3f": pace.get("first_half_3f"),
            "second_half_3f": pace.get("second_half_3f"),
            "t3f": pace.get("t3f"),
            "l3f": pace.get("l3f"),
        },
        "pace_shape": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in pace_shape.items()},
        "archetype_row_scale": [round(float(x), 4) for x in row_scale.tolist()],
        "axes": axes_out,
        "probs": probs,
    }


def aggregate_day(
    race_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """analyze_race の結果リストから加重平均（sqrt 頭数）。"""
    if not race_results:
        return {"axes": [], "probs": [], "n_races": 0}
    ws = np.array([math.sqrt(max(1, int(r.get("n_runners") or 1))) for r in race_results])
    P = np.array([r["probs"] for r in race_results], dtype=np.float64)
    mix = (ws[:, None] * P).sum(axis=0) / ws.sum()
    axes_out = []
    for i, ax in enumerate(QUALITY_AXES):
        axes_out.append(
            {
                "id": ax["id"],
                "label_ja": ax["label_ja"],
                "p": round(float(mix[i]), 4),
            }
        )
    return {
        "n_races": len(race_results),
        "axes": axes_out,
        "probs": [float(x) for x in mix],
    }


def analyze_date(
    storage: Any,
    date_compact: str,
    *,
    stats_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """race_lists の JRA レースを列挙し、結果のあるレースのみ推定。"""
    raw = storage.load("race_lists", date_compact)
    races_raw = (raw or {}).get("races") or []

    def _is_jra(rid: str) -> bool:
        return len(rid) >= 6 and rid[4:6] in {
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        }

    race_ids = [
        (r.get("race_id") or "").strip()
        for r in races_raw
        if _is_jra((r.get("race_id") or "").strip())
    ]

    per_race: list[dict[str, Any]] = []
    stats = stats_data
    if stats is None:
        stats = load_sire_factor_stats(storage=storage)
    valid_ids = [rid for rid in race_ids if rid]
    if not valid_ids:
        return {
            "date": date_compact,
            "day_summary": aggregate_day([]),
            "by_segment": {},
            "races": [],
            "skipped": 0,
        }

    max_workers = min(12, max(4, len(valid_ids)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(analyze_race, storage, rid, stats_data=stats): rid
            for rid in valid_ids
        }
        for fut in as_completed(futs):
            one = fut.result()
            if one:
                per_race.append(one)

    per_race.sort(key=lambda r: (r.get("venue", ""), r.get("race_id", "")))

    day = aggregate_day(per_race)

    by_seg: dict[str, Any] = defaultdict(list)
    for r in per_race:
        sk = r.get("segment_key") or "その他"
        by_seg[sk].append(r)
    by_segment = {
        sk: aggregate_day(lst)
        for sk, lst in sorted(by_seg.items(), key=lambda x: x[0])
    }

    return {
        "date": date_compact,
        "day_summary": day,
        "by_segment": by_segment,
        "races": per_race,
        "skipped": len(valid_ids) - len(per_race),
    }


# ── 馬単体適性キャッシュ（GCS: horse_race_quality_aptitude）────────────────

APTITUDE_CACHE_VERSION = 1
APTITUDE_STORE_CATEGORY = "horse_race_quality_aptitude"


def _aptitude_cache_stale(cached: dict[str, Any], hr: dict | None, ped: dict | None) -> bool:
    if int(cached.get("aptitude_cache_version", 0)) != APTITUDE_CACHE_VERSION:
        return True
    hr_ts = float(((hr or {}).get("_meta") or {}).get("scraped_at") or 0)
    ped_ts = float(((ped or {}).get("_meta") or {}).get("scraped_at") or 0)
    if hr_ts > float(cached.get("horse_result_scraped_at", 0)) + 1e-9:
        return True
    if ped_ts > float(cached.get("pedigree_scraped_at", 0)) + 1e-9:
        return True
    return False


def build_horse_aptitude_cache_payload(
    storage: Any,
    horse_id: str,
    *,
    stats_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """馬単体の適性スナップショット（血統ベクトル＋距離帯別戦歴）。保存用。"""
    hid = (horse_id or "").strip()
    if len(hid) < 6:
        return None
    stats = stats_data if stats_data is not None else load_sire_factor_stats(storage=storage)
    sires_db = stats.get("sires") or {}
    hr = storage.load("horse_result", hid) or {}
    ped = storage.load("horse_pedigree_5gen", hid) or {}
    ancestors = ped.get("ancestors", []) if isinstance(ped, dict) else []
    w_arr = np.array(
        [DEFAULT_WEIGHTS[s["key"]] for s in ANCESTOR_SLOTS],
        dtype=np.float64,
    )
    ped_vec = _compute_aptitude_fast(ancestors, w_arr, sires_db)
    hist = _history_features((hr.get("race_history") or []) if isinstance(hr, dict) else [])

    hr_ts = float((hr.get("_meta") or {}).get("scraped_at") or 0)
    ped_ts = float((ped.get("_meta") or {}).get("scraped_at") or 0)
    ped_list = [round(float(x), 4) for x in ped_vec.tolist()]
    ped_display = [
        {"id": a["id"], "label_ja": a["label_ja"], "value": round(ped_list[j], 2)}
        for j, a in enumerate(AXES)
    ]

    return {
        "aptitude_cache_version": APTITUDE_CACHE_VERSION,
        "horse_id": hid,
        "horse_name": (hr.get("horse_name") or "").strip(),
        "horse_result_scraped_at": hr_ts,
        "pedigree_scraped_at": ped_ts,
        "pedigree_vector": ped_list,
        "pedigree_axes": ped_display,
        "history": hist,
        "history_distance_bands_ja": list(HISTORY_BAND_KEYS),
    }


def get_horse_aptitude_cache(
    storage: Any,
    horse_id: str,
    *,
    force_refresh: bool = False,
    stats_data: dict[str, Any] | None = None,
    save: bool = True,
) -> tuple[dict[str, Any] | None, bool]:
    """
    適性キャッシュを返す。(payload, from_storage)
    from_storage が True のときストアの内容をそのまま利用（再計算なし）。
    """
    hid = (horse_id or "").strip()
    if len(hid) < 6:
        return None, False
    hr = storage.load("horse_result", hid)
    ped = storage.load("horse_pedigree_5gen", hid)
    cached: dict[str, Any] | None = None
    if not force_refresh:
        cached = storage.load(APTITUDE_STORE_CATEGORY, hid)
    if cached and not _aptitude_cache_stale(cached, hr, ped):
        return cached, True
    payload = build_horse_aptitude_cache_payload(storage, hid, stats_data=stats_data)
    if not payload:
        return cached, False
    if save:
        try:
            storage.save(APTITUDE_STORE_CATEGORY, hid, payload)
        except Exception as e:
            logger.warning("horse_race_quality_aptitude save failed %s: %s", hid, e)
    return payload, False


def _race_entries_for_aptitude(storage: Any, race_id: str) -> tuple[list[dict], dict[str, Any], str]:
    rr = storage.load("race_result", race_id)
    if rr and rr.get("entries"):
        return list(rr["entries"]), rr, "race_result"
    rs = storage.load("race_shutuba", race_id)
    if rs and rs.get("entries"):
        return list(rs["entries"]), rs, "race_shutuba"
    return [], {}, ""


def build_entrants_aptitude_response(
    storage: Any,
    race_id: str,
    *,
    force_refresh: bool = False,
    stats_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    指定レースの出走馬について、キャッシュ済み適性＋当該レース指数で 8 軸を可視化用に返す。
    """
    rid = (race_id or "").strip()
    entries, race_doc, src = _race_entries_for_aptitude(storage, rid)
    if not entries or not race_doc:
        return None

    seen: set[str] = set()
    uniq: list[dict] = []
    for e in entries:
        hid = (e.get("horse_id") or "").strip()
        if not hid or hid in seen:
            continue
        seen.add(hid)
        uniq.append(e)
    if not uniq:
        return None

    field_size = _si(race_doc.get("field_size"), len(uniq))
    race_dist = _si(race_doc.get("distance"), 0)
    surface = race_doc.get("surface") or ""
    track_condition = race_doc.get("track_condition") or ""
    segment_key = distance_surface_segment(surface, race_dist)
    hband = history_distance_band(race_dist)

    stats = stats_data if stats_data is not None else load_sire_factor_stats(storage=storage)

    ri = storage.load("race_index", rid) or {}
    rb = storage.load("race_barometer", rid) or {}
    ri_n, ri_i = _entries_by_num_id(ri.get("entries"))
    rb_n, rb_i = _entries_by_num_id(rb.get("entries"))

    rr_lap = storage.load("race_result", rid) or {}
    lap_times, pace = _resolve_lap_pace(storage, rid, rr_lap if rr_lap.get("entries") else race_doc)
    pace_shape = compute_pace_shape(pace, lap_times, race_dist)
    pace_w = pace_archetype_multipliers(pace_shape)
    going_w = going_archetype_multiplier(track_condition)
    seg_w = segment_archetype_prior(segment_key)
    row_scale = seg_w * pace_w * going_w

    archetype_axes = [
        {"id": a["id"], "label_ja": a["label_ja"], "hint": a.get("hint", "")}
        for a in QUALITY_AXES[:N_TYPE8]
    ]

    cache_hits = 0
    cache_misses = 0
    entrants_out: list[dict[str, Any]] = []

    for e in sorted(uniq, key=lambda x: _si(x.get("horse_number"), 999)):
        hid = (e.get("horse_id") or "").strip()
        cache, from_store = get_horse_aptitude_cache(
            storage,
            hid,
            force_refresh=force_refresh,
            stats_data=stats,
            save=True,
        )
        if from_store:
            cache_hits += 1
        else:
            cache_misses += 1
        if not cache:
            continue
        ped_vec = np.array(cache["pedigree_vector"], dtype=np.float64)
        hist = cache["history"]
        if not isinstance(hist, dict):
            hist = _history_features([])

        idx_e = _lookup_aux(ri_n, ri_i, e)
        bo_e = _lookup_aux(rb_n, rb_i, e)
        speed_max = _sf((idx_e or {}).get("speed_max"))
        speed_avg = _sf((idx_e or {}).get("speed_avg"))
        speed_distance = _sf((idx_e or {}).get("speed_distance"))
        idx_start = _sf((bo_e or {}).get("index_start"))
        idx_chase = _sf((bo_e or {}).get("index_chase"))
        idx_closing = _sf((bo_e or {}).get("index_closing"))
        bracket = _si(e.get("bracket_number"), 0)

        row = _build_archetype_row(
            ped_vec,
            hist,
            speed_max,
            speed_avg,
            speed_distance,
            idx_start,
            idx_chase,
            idx_closing,
            bracket,
            field_size,
            race_dist,
        )
        scaled = row * row_scale
        by_band = hist.get("by_band") or {}
        band_snap = by_band.get(hband) if isinstance(by_band, dict) else None

        entrants_out.append(
            {
                "horse_id": hid,
                "horse_name": cache.get("horse_name") or e.get("horse_name") or "",
                "horse_number": _si(e.get("horse_number"), 0),
                "bracket_number": _si(e.get("bracket_number"), 0),
                "finish_position": _si(e.get("finish_position"), 0) or None,
                "aptitude_from_cache": from_store,
                "pedigree_axes": cache.get("pedigree_axes") or [],
                "history_band_key": hband,
                "history_band_stats": band_snap,
                "base_row": [round(float(x), 4) for x in row.tolist()],
                "scaled_row": [round(float(x), 4) for x in scaled.tolist()],
            }
        )

    return {
        "race_id": rid,
        "race_name": race_doc.get("race_name") or "",
        "entries_source": src,
        "segment_key": segment_key,
        "history_distance_band": hband,
        "surface": surface,
        "distance_m": race_dist,
        "track_condition": track_condition,
        "pace_shape": {
            k: round(float(v), 4) if isinstance(v, float) else v
            for k, v in pace_shape.items()
        },
        "row_scale": [round(float(x), 4) for x in row_scale.tolist()],
        "archetype_axes": archetype_axes,
        "entrants": entrants_out,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "aptitude_cache_version": APTITUDE_CACHE_VERSION,
        "method_note_ja": (
            "各馬の血統・距離帯別戦歴は horse_race_quality_aptitude にキャッシュ。"
            "表示値は当該レースの race_index / race_barometer とセグメント・馬場・ラップ事前倍率を乗じた 8 軸。"
        ),
    }
