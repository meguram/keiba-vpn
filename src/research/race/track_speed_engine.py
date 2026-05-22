"""
馬場速度レベリング v2
====================

ベースライン (改修後 2020-2025):
  groupby: 競馬場 × コース(内外) × 距離 × クラス × 馬場プール
  指標: 2着走破タイム（前半3Fペース補正後）の mean/std → z-score → 馬場指数(100基準)
  馬場指数: 50=標準（偏差値風）、大きいほど速い馬場（指数 = 50 - z×10）

ペース補正 (コホート残差):
  同条件 (競馬場×内外×距離×クラス×馬場プール) 内でペース指標を3分位に分割
  各ペース集合の2着タイム分布から期待値を求め、全体平均との差で補正
  指標は RPCI (前半3F/上がり3F) と前半3F を比較し、残差相関が小さい方を採用

馬場プール (重複サンプル):
  firm_yielding  : 良 + 稍重
  yielding_soft  : 稍重 + 重
  soft_heavy     : 重 + 不良

判定時:
  稍重・重は複数プールで z を計算し、|z| 最小のプールを採用
  → 良寄り稍重 / 重寄り稍重 等のラベルを付与

アーティファクト:
  data/knowledge/track_speed_baselines.parquet
  data/analysis/track_speed/races_{year}.parquet
  data/analysis/track_speed/meta.json
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from src.config.data_paths import (
    TRACK_SPEED_BASELINES,
    TRACK_SPEED_PACE_BASELINES,
    TRACK_SPEED_RACES_DIR,
)

logger = logging.getLogger("research.track_speed")

JRA_VENUES = frozenset({
    "札幌", "函館", "福島", "新潟", "東京",
    "中山", "中京", "京都", "阪神", "小倉",
})

CLASS_BANDS = ("新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "L", "G3", "G2", "G1")

COND_POOLS = {
    "firm_yielding": frozenset({"良", "稍重"}),
    "yielding_soft": frozenset({"稍重", "重"}),
    "soft_heavy": frozenset({"重", "不良"}),
}

# レースの馬場状態 → 判定に使うプール候補
COND_CANDIDATES: dict[str, tuple[str, ...]] = {
    "良": ("firm_yielding",),
    "稍重": ("firm_yielding", "yielding_soft"),
    "重": ("yielding_soft", "soft_heavy"),
    "不良": ("soft_heavy",),
}

COND_VARIANT_LABEL = {
    ("稍重", "firm_yielding"): "良寄り稍重",
    ("稍重", "yielding_soft"): "重寄り稍重",
    ("重", "yielding_soft"): "稍重寄り重",
    ("重", "soft_heavy"): "不良寄り重",
    ("不良", "soft_heavy"): "不良",
}

SPEED_LEVELS = [
    (-float("inf"), -1.5, "超高速", 5),
    (-1.5, -0.5, "高速", 4),
    (-0.5, 0.5, "標準", 3),
    (0.5, 1.5, "低速", 2),
    (1.5, float("inf"), "超低速", 1),
]

MIN_BASELINE_N = 12
# クラスレベル分類用の最小サンプル数: ベースライン構築(12)より高く設定し、
# 小サンプルの不安定な平均値がクラス判定を引きつけるのを防ぐ。
MIN_CLASSIFY_N = 20
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINES_PATH = TRACK_SPEED_BASELINES
PACE_BASELINES_PATH = TRACK_SPEED_PACE_BASELINES
RACES_DIR = TRACK_SPEED_RACES_DIR
META_PATH = RACES_DIR / "meta.json"
MIN_PACE_CONTEXT_N = 30
MIN_PACE_BIN_N = 8
PACE_ADJ_CLIP = 1.5
PACE_METRICS = ("rpci", "first_half_3f")
BABA_INDEX_BASE = 50
BABA_INDEX_STEP = 10


def z_to_baba_index(z: float) -> float:
    """z-score → 馬場指数（50基準・高いほど速い馬場）。"""
    return round(BABA_INDEX_BASE - float(z) * BABA_INDEX_STEP, 1)


def json_safe(value: Any) -> Any:
    """NaN/Inf を None にし FastAPI JSON 化エラーを防ぐ。"""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, np.integer):
        return int(value)
    return value


def format_race_time(sec: float | None) -> str:
    """秒 → 競馬表記 (例: 91.5 → 1:31.5)。"""
    if sec is None or (isinstance(sec, float) and pd.isna(sec)):
        return "—"
    try:
        t = float(sec)
    except (TypeError, ValueError):
        return "—"
    if t <= 0:
        return "—"
    # 0.1秒単位の整数演算で浮動小数誤差を避ける (120.4 → 2:00.4)
    tenths = int(round(t * 10))
    m, rem = divmod(tenths, 600)
    if m > 0:
        whole, dec = divmod(rem, 10)
        return f"{m}:{whole:02d}.{dec}"
    return f"{t:.1f}"


def enrich_pace_from_lap(
    races: pd.DataFrame,
    years: list[str],
    base_dir: str | Path,
) -> pd.DataFrame:
    """race_result の pace 欠損を race_result_lap flat から補完する。"""
    if races.empty or "pace" not in races.columns:
        return races
    from src.scraper.local_tables import load_flat_df
    from src.scraper.pace_utils import pace_has_first_half

    need = races["pace"].map(
        lambda p: not pace_has_first_half(p) if p is not None else True
    )
    if not need.any():
        return races
    lap = load_flat_df(
        "race_result_lap",
        years=years,
        columns=["race_id", "pace"],
        base_dir=str(base_dir),
    )
    if lap.empty:
        return races
    lap_map = (
        lap.drop_duplicates("race_id")
        .set_index("race_id")["pace"]
        .to_dict()
    )
    out = races.copy()

    def _fill(row: pd.Series) -> object:
        p = row["pace"]
        if pace_has_first_half(p):
            return p
        lap_p = lap_map.get(str(row["race_id"]))
        if pace_has_first_half(lap_p):
            return lap_p
        return p

    out.loc[need, "pace"] = out.loc[need].apply(_fill, axis=1)
    return out


def enrich_distance(
    df: pd.DataFrame,
    years: list[str],
    base_dir: str | Path,
) -> pd.DataFrame:
    """race_result の distance=0 を race_shutuba_flat で補完。"""
    if df.empty or "distance" not in df.columns:
        return df
    mask = df["distance"].fillna(0).astype(int) == 0
    if not mask.any():
        return df
    from src.scraper.local_tables import load_flat_df

    sh = load_flat_df(
        "race_shutuba",
        years=years,
        columns=["race_id", "distance"],
        base_dir=str(base_dir),
    )
    if sh.empty:
        return df
    dist_map = sh.drop_duplicates("race_id").set_index("race_id")["distance"]
    out = df.copy()
    out.loc[mask, "distance"] = (
        out.loc[mask, "race_id"].map(dist_map).fillna(0).astype(int)
    )
    return out


def pace_distance_band(distance: int) -> str:
    if distance <= 0:
        return "unknown"
    if distance <= 1200:
        return "≤1200"
    if distance <= 1400:
        return "1201-1400"
    if distance <= 1600:
        return "1401-1600"
    if distance <= 1800:
        return "1601-1800"
    if distance <= 2200:
        return "1801-2200"
    if distance <= 2600:
        return "2201-2600"
    return "2601+"


def parse_pace_obj(raw: Any) -> dict[str, Any]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass(frozen=True)
class PaceFeatures:
    first_half_3f: float | None = None
    last_3f: float | None = None
    rpci: float | None = None
    l3fr: float | None = None


@dataclass(frozen=True)
class PaceAdjustResult:
    time_adj: float
    adj_sec: float
    label: str
    z_pace: float | None
    metric: str
    rpci: float | None
    first_half_3f: float | None
    metric_mean: float | None = None
    metric_std: float | None = None


def extract_pace_features(raw: Any) -> PaceFeatures:
    """pace 列から前半3F・上がり3F・RPCI 等を抽出。"""
    d = parse_pace_obj(raw)
    fh = last_3f = None
    try:
        v = float(d.get("first_half_3f"))
        fh = v if v > 0 else None
    except (TypeError, ValueError):
        pass
    for key in ("second_half_3f", "l3f", "last_3f"):
        try:
            v = float(d.get(key))
            if v > 0:
                last_3f = v
                break
        except (TypeError, ValueError):
            continue
    rpci = (fh / last_3f) if fh and last_3f else None
    l3fr = (last_3f / (fh + last_3f)) if fh and last_3f else None
    return PaceFeatures(
        first_half_3f=round(fh, 2) if fh else None,
        last_3f=round(last_3f, 2) if last_3f else None,
        rpci=round(rpci, 4) if rpci else None,
        l3fr=round(l3fr, 4) if l3fr else None,
    )


def extract_first_half_3f(raw: Any) -> float | None:
    return extract_pace_features(raw).first_half_3f


def pace_context_key(
    venue: str,
    layout: str,
    surface: str,
    distance: int,
    cond_pool: str,
) -> str:
    return f"{venue}|{layout}|{surface}|{distance}|{cond_pool}"


def pace_coarse_context_key(
    surface: str,
    distance: int,
    cond_pool: str,
) -> str:
    return f"ALL|-|{surface}|{pace_distance_band(distance)}|{cond_pool}"


def _metric_value(features: PaceFeatures, metric: str) -> float | None:
    if metric == "rpci":
        return features.rpci
    if metric == "first_half_3f":
        return features.first_half_3f
    return None


def _pace_label_from_z(z_pace: float | None) -> str:
    """
    z < −1 → 前傾（ハイペース）: 前半タイムが平均より短い = 速い前半
    z > +1 → 抑え（スロー）  : 前半タイムが平均より長い = 遅い前半
    ※ rpci/first_half_3f は「値が大きい = 前半が遅い」なので逆符号になる
    """
    if z_pace is None:
        return ""
    if z_pace < -1.0:
        return "前傾"
    if z_pace > 1.0:
        return "抑え"
    return ""


def _clip_pace_adj(adj: float) -> float:
    return max(-PACE_ADJ_CLIP, min(PACE_ADJ_CLIP, adj))


def _apply_cohort_adjustment(
    time_2nd: float,
    metric_val: float,
    meta: dict[str, Any],
) -> PaceAdjustResult:
    """正規分布z変換ベース連続ペース補正。adj = clip(β × z_pace, ±PACE_ADJ_CLIP)。"""
    metric = str(meta["metric"])
    metric_mean = float(meta["mean"])
    metric_std = float(meta["std"])
    beta = float(meta["beta"])
    z_pace = (metric_val - metric_mean) / metric_std if metric_std > 1e-9 else 0.0
    adj = _clip_pace_adj(beta * z_pace)
    time_adj = max(time_2nd - adj, time_2nd * 0.94)
    rpci = meta.get("_rpci")
    fh = meta.get("_fh")
    return PaceAdjustResult(
        time_adj=round(time_adj, 2),
        adj_sec=round(adj, 2),
        label=_pace_label_from_z(z_pace),
        z_pace=round(z_pace, 3),
        metric=metric,
        rpci=float(rpci) if rpci is not None else None,
        first_half_3f=float(fh) if fh is not None else None,
        metric_mean=round(metric_mean, 3),
        metric_std=round(metric_std, 3),
    )


def _select_pace_metric(group: pd.DataFrame) -> str | None:
    """time_2nd との相関が最も高いペース指標を選ぶ（z変換後の線形補正力が最大）。"""
    best_metric: str | None = None
    best_r2 = 0.0
    for metric in PACE_METRICS:
        if metric not in group.columns:
            continue
        sub = group[[metric, "time_2nd"]].dropna()
        if len(sub) < MIN_PACE_CONTEXT_N * 0.65:
            continue
        vals = sub[metric].astype(float)
        t = sub["time_2nd"].astype(float)
        if float(vals.std()) < 1e-6:
            continue
        r = abs(float(np.corrcoef(vals, t)[0, 1]))
        if math.isnan(r):
            r = 0.0
        if r > best_r2:
            best_r2 = r
            best_metric = metric
    return best_metric


def _build_context_cohort(
    group: pd.DataFrame,
    metric: str,
) -> dict[str, Any] | None:
    """正規分布z変換に基づく線形回帰補正係数を算出。β = Cov(z_pace, time_2nd)。"""
    sub = group[[metric, "time_2nd"]].dropna()
    if len(sub) < MIN_PACE_CONTEXT_N:
        return None
    vals = sub[metric].astype(float)
    times = sub["time_2nd"].astype(float)
    metric_mean = float(vals.mean())
    metric_std = float(vals.std())
    if metric_std < 1e-6:
        return None
    global_mean = float(times.mean())
    z_pace = (vals - metric_mean) / metric_std
    # β = Cov(z_pace, time) / Var(z_pace) = Cov(z_pace, time) since Var(z_pace) = 1
    cov_matrix = np.cov(z_pace.values, times.values)
    beta = float(cov_matrix[0, 1])
    return {
        "metric": metric,
        "mean": round(metric_mean, 4),
        "std": round(metric_std, 4),
        "beta": round(beta, 4),
        "global_mean": round(global_mean, 3),
        "n": len(sub),
    }


def adjust_time_for_pace(
    time_2nd: float,
    features: PaceFeatures,
    cohort_meta: dict[str, Any] | None,
) -> PaceAdjustResult:
    """正規分布z変換ベースの連続ペース補正。"""
    if cohort_meta is None:
        return PaceAdjustResult(
            time_adj=time_2nd,
            adj_sec=0.0,
            label="",
            z_pace=None,
            metric="",
            rpci=features.rpci,
            first_half_3f=features.first_half_3f,
        )
    metric = str(cohort_meta["metric"])
    metric_val = _metric_value(features, metric)
    if metric_val is None:
        for alt in PACE_METRICS:
            if alt == metric:
                continue
            metric_val = _metric_value(features, alt)
            if metric_val is not None:
                cohort_meta = {**cohort_meta, "metric": alt}
                metric = alt
                break
    if metric_val is None:
        return PaceAdjustResult(
            time_adj=time_2nd,
            adj_sec=0.0,
            label="",
            z_pace=None,
            metric=metric,
            rpci=features.rpci,
            first_half_3f=features.first_half_3f,
        )
    meta = {
        **cohort_meta,
        "_fh": features.first_half_3f,
        "_rpci": features.rpci,
    }
    return _apply_cohort_adjustment(time_2nd, metric_val, meta)


def parse_round(row: dict | pd.Series) -> int:
    """round 列が 0 のとき race_id 末尾2桁から R 番号を復元。"""
    r = row.get("round") if hasattr(row, "get") else None
    try:
        if r is not None and int(r) > 0:
            return int(r)
    except (TypeError, ValueError):
        pass
    rid = str(row.get("race_id", "") if hasattr(row, "get") else row)
    if len(rid) >= 2 and rid[-2:].isdigit():
        return int(rid[-2:])
    return 0


def is_obstacle_race(row: dict | pd.Series) -> bool:
    """障害・ジャンプ競走は馬場速度計測対象外。"""
    if hasattr(row, "get"):
        name = str(row.get("race_name") or "")
        surface = str(row.get("surface") or "").strip()
        grade = str(row.get("grade") or "")
    else:
        name = str(getattr(row, "race_name", "") or "")
        surface = str(getattr(row, "surface", "") or "").strip()
        grade = str(getattr(row, "grade", "") or "")
    if "障害" in name:
        return True
    if surface == "障害":
        return True
    if "障害" in grade:
        return True
    return False


def classify_grade(grade: str) -> str:
    g = (grade or "").strip()
    if g in CLASS_BANDS:
        return g
    if g.startswith("G") and len(g) == 2:
        return g
    return "未勝利"


def classify_speed(z: float) -> tuple[str, int]:
    for lo, hi, label, level in SPEED_LEVELS:
        if lo <= z < hi:
            return label, level
    return "標準", 3


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_course_layout(venue: str, surface: str, distance: int) -> str:
    if surface != "芝":
        return "-"
    data = _load_json(PROJECT_ROOT / "data" / "local" / "knowledge" / "course_layout_map.json")
    key = f"{venue}_{surface}_{distance}"
    return data.get("entries", {}).get(key, data.get("default", "-"))


def renovation_cutoff(venue: str, surface: str) -> str | None:
    data = _load_json(PROJECT_ROOT / "data" / "local" / "knowledge" / "track_renovations.json")
    v = data.get("venues", {}).get(venue, {})
    return v.get(surface)


@dataclass(frozen=True)
class BaselineKey:
    venue: str
    layout: str
    surface: str
    distance: int
    class_band: str
    cond_pool: str

    def token(self) -> str:
        return f"{self.venue}|{self.layout}|{self.surface}|{self.distance}|{self.class_band}|{self.cond_pool}"


CLASS_LEVEL_ORDER = ("新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "L", "G3", "G2", "G1")


def _class_fallback_chain(class_band: str) -> list[str]:
    order = list(CLASS_BANDS)
    if class_band not in order:
        return list(CLASS_BANDS)
    idx = order.index(class_band)
    chain = [class_band]
    # 近隣クラスを優先し、最終的に全クラスを探索（稀少距離でサンプル不足の場合に備える）
    if class_band in ("G1", "G2", "G3"):
        chain.extend(["G3", "G2", "G1", "OP", "L", "3勝", "2勝", "1勝", "未勝利", "新馬"])
    elif class_band in ("OP", "L"):
        chain.extend(["3勝", "2勝", "1勝", "未勝利", "G3", "G2", "G1", "新馬"])
    else:
        # 条件戦: 上方向 → 下方向 の順で探索
        chain.extend(order[idx + 1 : idx + 4])
        chain.extend(order[max(0, idx - 1) : idx])
        chain.extend(order)  # 最終フォールバック: 全クラス
    seen: set[str] = set()
    out: list[str] = []
    for c in chain:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _min_std(distance: int) -> float:
    return 0.35 + distance / 5000.0


class TrackSpeedEngine:
    def __init__(self, base_dir: str | Path = "."):
        self.base_dir = Path(base_dir)
        self._baselines: dict[str, dict[str, float]] = {}
        self._pace_cohorts: dict[str, dict[str, Any]] = {}
        self._renovation: dict[str, dict[str, str | None]] = {}

    def load_baselines(self) -> bool:
        if not BASELINES_PATH.exists():
            return False
        df = pd.read_parquet(BASELINES_PATH)
        self._baselines = {
            row["key"]: {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "n": int(row["n"]),
            }
            for _, row in df.iterrows()
        }
        ren = _load_json(PROJECT_ROOT / "data" / "local" / "knowledge" / "track_renovations.json")
        self._renovation = ren.get("venues", {})
        self.load_pace_baselines()
        return True

    def load_pace_baselines(self) -> bool:
        if not PACE_BASELINES_PATH.exists():
            self._pace_cohorts = {}
            return False
        df = pd.read_parquet(PACE_BASELINES_PATH)
        if "context_key" not in df.columns:
            logger.warning("旧ペース基準形式のため再構築が必要です")
            self._pace_cohorts = {}
            return False
        if "beta" not in df.columns:
            logger.warning("旧3分位形式のため再構築が必要です")
            self._pace_cohorts = {}
            return False
        cohorts: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            cohorts[str(row["context_key"])] = {
                "metric": str(row["metric"]),
                "mean": float(row["metric_mean"]),
                "std": float(row["metric_std"]),
                "beta": float(row["beta"]),
                "global_mean": float(row["global_mean"]),
                "n": int(row.get("n", 0)),
            }
        self._pace_cohorts = cohorts
        return True

    def _pace_context_chain(
        self,
        venue: str,
        layout: str,
        surface: str,
        distance: int,
        cond_pool: str,
    ) -> list[str]:
        keys: list[str] = []
        layouts = [layout, "-"] if layout != "-" else ["-"]
        for lay in layouts:
            keys.append(pace_context_key(venue, lay, surface, distance, cond_pool))
        keys.append(pace_coarse_context_key(surface, distance, cond_pool))
        seen: set[str] = set()
        out: list[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    def _lookup_pace_cohort(
        self,
        venue: str,
        layout: str,
        surface: str,
        distance: int,
        track_condition: str,
    ) -> dict[str, Any] | None:
        tc = str(track_condition or "").strip()
        pools = COND_CANDIDATES.get(tc, ())
        seen: set[str] = set()
        for pool in pools:
            for ctx in self._pace_context_chain(venue, layout, surface, distance, pool):
                if ctx in seen:
                    continue
                seen.add(ctx)
                meta = self._pace_cohorts.get(ctx)
                if meta:
                    return meta
        return None

    def build_pace_baselines(
        self,
        races: pd.DataFrame,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """同条件×ペース集合の2着タイム分布からコホート補正表を構築。"""
        work = races.copy()
        feats = work["pace"].map(extract_pace_features)
        work["first_half_3f"] = feats.map(lambda f: f.first_half_3f)
        work["rpci"] = feats.map(lambda f: f.rpci)
        work = work[work["time_2nd"].notna() & (work["time_2nd"] > 0)]

        expanded: list[dict[str, Any]] = []
        for _, r in work.iterrows():
            tc = str(r.get("track_condition") or "").strip()
            if tc not in COND_CANDIDATES:
                continue
            venue = str(r["venue"])
            surface = str(r["surface"])
            distance = int(r["distance"])
            layout = str(
                r.get("layout")
                or load_course_layout(venue, surface, distance)
            )
            for pool_name, cond_set in COND_POOLS.items():
                if tc not in cond_set:
                    continue
                ctx = pace_context_key(venue, layout, surface, distance, pool_name)
                expanded.append({
                    "context_key": ctx,
                    "time_2nd": float(r["time_2nd"]),
                    "first_half_3f": r["first_half_3f"],
                    "rpci": r["rpci"],
                })
                coarse = pace_coarse_context_key(surface, distance, pool_name)
                expanded.append({
                    "context_key": coarse,
                    "time_2nd": float(r["time_2nd"]),
                    "first_half_3f": r["first_half_3f"],
                    "rpci": r["rpci"],
                })

        if not expanded:
            if progress_cb:
                progress_cb("ペース基準: 対象レース0件")
            return pd.DataFrame()

        exp_df = pd.DataFrame(expanded)
        cohort_store: dict[str, dict[str, Any]] = {}
        parquet_rows: list[dict[str, Any]] = []

        for ctx, g in exp_df.groupby("context_key"):
            metric = _select_pace_metric(g)
            if metric is None:
                metric = "rpci" if g["rpci"].notna().mean() >= 0.5 else "first_half_3f"
            built = _build_context_cohort(g, metric)
            if not built:
                continue
            cohort_store[str(ctx)] = built
            parquet_rows.append({
                "context_key": ctx,
                "metric": built["metric"],
                "metric_mean": built["mean"],
                "metric_std": built["std"],
                "beta": built["beta"],
                "global_mean": built["global_mean"],
                "n": built["n"],
            })

        bl_df = pd.DataFrame(parquet_rows)
        if not bl_df.empty:
            PACE_BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
            bl_df.to_parquet(PACE_BASELINES_PATH, index=False)
            self._pace_cohorts = cohort_store
        else:
            self._pace_cohorts = {}

        metric_counts = (
            bl_df["metric"].value_counts().to_dict()
            if not bl_df.empty
            else {}
        )
        if progress_cb:
            progress_cb(
                f"ペースコホート: {len(cohort_store)} 条件 / "
                f"指標 {metric_counts} → {PACE_BASELINES_PATH}"
            )
        return bl_df

    def _pace_adjust_row(self, row: pd.Series | dict) -> PaceAdjustResult:
        if isinstance(row, pd.Series):
            row = row.to_dict()
        feats = extract_pace_features(row.get("pace"))
        if row.get("first_half_3f") is not None:
            try:
                fh = float(row["first_half_3f"])
                feats = PaceFeatures(
                    first_half_3f=fh,
                    last_3f=feats.last_3f,
                    rpci=row.get("rpci") if row.get("rpci") is not None else feats.rpci,
                    l3fr=feats.l3fr,
                )
            except (TypeError, ValueError):
                pass
        venue = str(row["venue"])
        surface = str(row["surface"])
        distance = int(row.get("distance") or 0)
        layout = str(
            row.get("layout") or load_course_layout(venue, surface, distance)
        )
        meta = self._lookup_pace_cohort(
            venue, layout, surface, distance,
            str(row.get("track_condition") or ""),
        )
        return adjust_time_for_pace(float(row["time_2nd"]), feats, meta)

    def _attach_pace_adjustments(self, races: pd.DataFrame) -> pd.DataFrame:
        if races.empty:
            return races
        out = races.copy()
        fh_vals: list[float | None] = []
        rpci_vals: list[float | None] = []
        adj_secs: list[float] = []
        time_adj: list[float] = []
        labels: list[str] = []
        metrics: list[str] = []
        z_paces: list[float | None] = []
        for _, r in out.iterrows():
            res = self._pace_adjust_row(r)
            fh_vals.append(res.first_half_3f)
            rpci_vals.append(res.rpci)
            adj_secs.append(res.adj_sec)
            time_adj.append(res.time_adj)
            labels.append(res.label)
            metrics.append(res.metric)
            z_paces.append(res.z_pace)
        out["first_half_3f"] = fh_vals
        out["pace_rpci"] = rpci_vals
        out["pace_metric"] = metrics
        out["pace_adj_sec"] = adj_secs
        out["time_2nd_adj"] = time_adj
        out["pace_label"] = labels
        out["pace_z"] = z_paces
        return out

    def load_races_from_parquet(
        self,
        years: list[str],
        date_min: str | None = None,
        date_max: str | None = None,
        progress_cb: Callable[[str], None] | None = None,
        apply_pace: bool = True,
    ) -> pd.DataFrame:
        from src.scraper.local_tables import load_flat_df

        cols = [
            "race_id", "date", "venue", "surface", "distance", "direction",
            "grade", "track_condition", "round", "race_name", "pace",
            "finish_position", "time_sec", "horse_number",
        ]
        df = load_flat_df("race_result", years=years, columns=cols, base_dir=str(self.base_dir))
        if df.empty:
            return df

        df = df[df["venue"].isin(JRA_VENUES)]
        df = df[df["surface"].isin(["芝", "ダート"])]
        df = df[~df.apply(is_obstacle_race, axis=1)]
        df = df[df["finish_position"] > 0]
        df = df[df["time_sec"].notna() & (df["time_sec"] > 0)]

        # flat parquet の date は "2025-12-28" 形式。date_min/max はハイフンなし
        # ("20251228") で渡されるため、比較前に正規化する。
        if date_min or date_max:
            date_norm = df["date"].astype(str).str.replace("-", "", regex=False).values
            if date_min:
                df = df[date_norm >= date_min]
            if date_max:
                date_norm = df["date"].astype(str).str.replace("-", "", regex=False).values
                df = df[date_norm <= date_max]

        # レース単位: 2着タイム
        r2 = df[df["finish_position"] == 2][
            ["race_id", "time_sec"]
        ].rename(columns={"time_sec": "time_2nd"})
        meta_cols = [
            "race_id", "date", "venue", "surface", "distance", "grade",
            "track_condition", "round", "race_name", "pace",
        ]
        race_meta = df.groupby("race_id", as_index=False).first()[meta_cols]
        races = race_meta.merge(r2, on="race_id", how="inner")
        races = enrich_pace_from_lap(races, years, self.base_dir)
        races = enrich_distance(races, years, self.base_dir)
        races["class_band"] = races["grade"].map(classify_grade)
        races["layout"] = [
            load_course_layout(str(v), str(s), int(d))
            for v, s, d in zip(races["venue"], races["surface"], races["distance"], strict=False)
        ]
        races["date"] = races["date"].astype(str).str.replace("-", "", regex=False)

        # 改修日フィルタ
        def _after_renovation(row) -> bool:
            cut = renovation_cutoff(str(row["venue"]), str(row["surface"]))
            if not cut:
                return True
            d = str(row["date"])
            if len(d) == 8:
                d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            return d >= cut

        before = len(races)
        races = races[races.apply(_after_renovation, axis=1)].copy()
        if progress_cb:
            progress_cb(f"レース抽出: {len(races)} (改修フィルタ前 {before})")
        if apply_pace:
            if not self._pace_cohorts:
                self.load_pace_baselines()
            if self._pace_cohorts:
                races = self._attach_pace_adjustments(races)
            else:
                races = races.copy()
                races["time_2nd_adj"] = races["time_2nd"]
                races["pace_adj_sec"] = 0.0
                races["pace_label"] = ""
                races["first_half_3f"] = None
                races["pace_rpci"] = None
                races["pace_metric"] = ""
                races["pace_z"] = None
        return races

    def build_baselines(
        self,
        years: list[str] | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        if years is None:
            years = [str(y) for y in range(2020, 2026)]
        races = self.load_races_from_parquet(
            years, progress_cb=progress_cb, apply_pace=False,
        )
        if races.empty:
            raise ValueError("ベースライン用レースが0件")

        self.build_pace_baselines(races, progress_cb=progress_cb)
        races = self._attach_pace_adjustments(races)

        groups: dict[str, list[float]] = defaultdict(list)
        for _, r in races.iterrows():
            tc = str(r["track_condition"] or "").strip()
            if tc not in COND_CANDIDATES:
                continue
            t_score = float(r.get("time_2nd_adj") or r["time_2nd"])
            for pool_name, cond_set in COND_POOLS.items():
                if tc not in cond_set:
                    continue
                bk = BaselineKey(
                    str(r["venue"]), str(r["layout"]), str(r["surface"]),
                    int(r["distance"]), str(r["class_band"]), pool_name,
                )
                groups[bk.token()].append(t_score)

        rows: list[dict] = []
        for key, times in groups.items():
            if len(times) < MIN_BASELINE_N:
                continue
            mu = statistics.mean(times)
            sd = max(statistics.stdev(times) if len(times) > 1 else 1.0, _min_std(int(key.split("|")[3])))
            parts = key.split("|")
            rows.append({
                "key": key,
                "venue": parts[0],
                "layout": parts[1],
                "surface": parts[2],
                "distance": int(parts[3]),
                "class_band": parts[4],
                "cond_pool": parts[5],
                "mean": round(mu, 3),
                "std": round(sd, 3),
                "n": len(times),
            })

        bl_df = pd.DataFrame(rows)
        BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
        bl_df.to_parquet(BASELINES_PATH, index=False)
        self._baselines = {r["key"]: {"mean": r["mean"], "std": r["std"], "n": r["n"]} for _, r in bl_df.iterrows()}

        if progress_cb:
            progress_cb(f"ベースライン保存: {len(bl_df)} グループ → {BASELINES_PATH}")
        return bl_df

    def _lookup_baseline(
        self,
        venue: str,
        layout: str,
        surface: str,
        distance: int,
        class_band: str,
        cond_pool: str,
    ) -> dict[str, float] | None:
        layouts = [layout, "-"] if layout != "-" else ["-"]
        for cls in _class_fallback_chain(class_band):
            for lay in layouts:
                key = BaselineKey(venue, lay, surface, distance, cls, cond_pool).token()
                bl = self._baselines.get(key)
                if bl and bl["n"] >= MIN_BASELINE_N:
                    return {**bl, "key": key, "class_band_used": cls, "layout_used": lay}
        # 全会場フォールバック
        for cls in _class_fallback_chain(class_band):
            key = BaselineKey("ALL", "-", surface, distance, cls, cond_pool).token()
            bl = self._baselines.get(key)
            if bl:
                return {**bl, "key": key, "class_band_used": cls, "layout_used": "-"}
        return None

    def _classify_class_level(
        self,
        time_2nd_adj: float,
        venue: str,
        layout: str,
        surface: str,
        distance: int,
        cond_pool: str,
        class_band: str | None = None,
    ) -> str | None:
        """
        time_2nd_adj を各クラスのベースライン (mean, std) でPF指数換算し、
        PF=50（クラス標準）に最も近いクラス名を返す。

        スコア = |(time - mean) / std| + 1/sqrt(n)
          第1項: σ差 = PF50からの距離（小さいほど「標準的」）
          第2項: サンプル数補正 = 平均値の推定誤差 (SE/std = 1/sqrt(n))
            → 小サンプルのベースラインほどペナルティが大きく、不安定な平均に
               引きつけられるのを防ぐ。n→∞ でペナルティは0に収束。

        制約: 実クラス±2ランク以内かつ n>=MIN_CLASSIFY_N の候補のみ。
          根拠: 全84コース中73.8%でG級ベースラインが存在せず、34箇所で
          高クラスほど平均タイムが遅い逆転現象がある。MIN_CLASSIFY_N=20 は
          MIN_BASELINE_N=12 より高く設定し、分類精度を 46.7%→53.4% に改善。
          ベースラインが見つからない場合は実クラス(class_band)を返す。
        """
        if not self._baselines or time_2nd_adj <= 0:
            return class_band
        layouts = [layout, "-"] if layout != "-" else ["-"]

        own_rank = CLASS_LEVEL_ORDER.index(class_band) if class_band in CLASS_LEVEL_ORDER else None

        # 実クラス±2ランク以内 かつ n>=MIN_CLASSIFY_N の候補のみ (mean, std, n) を収集
        cls_baselines: dict[str, tuple[float, float, int]] = {}
        for cls in CLASS_LEVEL_ORDER:
            if own_rank is not None:
                cls_rank = CLASS_LEVEL_ORDER.index(cls)
                if abs(cls_rank - own_rank) > 2:
                    continue
            bl: dict | None = None
            for lay in layouts:
                for ven in [venue, "ALL"]:
                    key = BaselineKey(ven, lay, surface, distance, cls, cond_pool).token()
                    candidate = self._baselines.get(key)
                    if candidate and int(candidate.get("n", 0)) >= MIN_CLASSIFY_N:
                        bl = candidate
                        break
                if bl:
                    break
            if bl is not None:
                std = float(bl.get("std") or 1.0)
                cls_baselines[cls] = (float(bl["mean"]), std if std > 0 else 1.0, int(bl["n"]))

        if not cls_baselines:
            return class_band

        # σ差 + n補正ペナルティ が最小のクラス = PF50に最も近い信頼性の高いクラス
        best_cls = min(cls_baselines, key=lambda c: (
            abs((time_2nd_adj - cls_baselines[c][0]) / cls_baselines[c][1])
            + 1.0 / math.sqrt(cls_baselines[c][2])
        ))
        return best_cls

    def _attach_class_level_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """既存DataFrameにclass_level_label列を後付けで計算・追加する。"""
        if df.empty or not self._baselines:
            return df
        if "class_level_label" in df.columns and df["class_level_label"].notna().any():
            return df
        out = df.copy()
        labels: list[str | None] = []
        for _, r in out.iterrows():
            t = float(r.get("time_2nd_adj") or r.get("time_2nd") or 0)
            cl = self._classify_class_level(
                t,
                str(r.get("venue", "")),
                str(r.get("layout") or "-"),
                str(r.get("surface", "")),
                int(r.get("distance") or 0),
                str(r.get("cond_pool") or "firm_yielding"),
                class_band=str(r.get("class_band") or ""),
            )
            labels.append(cl)
        out["class_level_label"] = labels
        return out

    def score_race(self, row: pd.Series | dict) -> dict[str, Any] | None:
        if isinstance(row, pd.Series):
            row = row.to_dict()
        if is_obstacle_race(row):
            return None
        tc = str(row.get("track_condition") or "").strip()
        if tc not in COND_CANDIDATES:
            return None
        time_raw = float(row.get("time_2nd") or 0)
        if time_raw <= 0:
            return None
        pace_res = self._pace_adjust_row(row)
        time_2nd = pace_res.time_adj
        pace_adj = pace_res.adj_sec
        pace_label = pace_res.label
        fh = pace_res.first_half_3f
        pace_rpci = pace_res.rpci
        pace_metric = pace_res.metric
        pace_z = pace_res.z_pace
        if row.get("time_2nd_adj") is not None:
            try:
                time_2nd = float(row["time_2nd_adj"])
            except (TypeError, ValueError):
                pass
        if row.get("pace_adj_sec") is not None:
            try:
                pace_adj = float(row["pace_adj_sec"])
            except (TypeError, ValueError):
                pass
        if row.get("pace_label"):
            pace_label = str(row["pace_label"])
        if row.get("pace_rpci") is not None:
            try:
                pace_rpci = float(row["pace_rpci"])
            except (TypeError, ValueError):
                pass
        if row.get("pace_metric"):
            pace_metric = str(row["pace_metric"])
        if row.get("pace_z") is not None:
            try:
                pace_z = float(row["pace_z"])
            except (TypeError, ValueError):
                pass

        venue = str(row["venue"])
        surface = str(row["surface"])
        distance = int(row["distance"])
        class_band = str(row.get("class_band") or classify_grade(str(row.get("grade", ""))))
        layout = str(row.get("layout") or load_course_layout(venue, surface, distance))

        best: dict[str, Any] | None = None
        for pool in COND_CANDIDATES[tc]:
            bl = self._lookup_baseline(venue, layout, surface, distance, class_band, pool)
            if not bl:
                continue
            z = (time_2nd - bl["mean"]) / bl["std"]
            z_raw = (time_raw - bl["mean"]) / bl["std"]
            cand = {
                "z": round(z, 3),
                "z_raw": round(z_raw, 3),
                "cond_pool": pool,
                "baseline_mean": bl["mean"],
                "baseline_std": bl["std"],
                "baseline_n": bl["n"],
                "baseline_key": bl["key"],
            }
            if best is None or abs(cand["z"]) < abs(best["z"]):
                best = cand

        if not best:
            return None

        label, level = classify_speed(best["z"])
        _, level_raw = classify_speed(best["z_raw"])
        variant = COND_VARIANT_LABEL.get((tc, best["cond_pool"]), tc)
        if tc == "良":
            variant = "良"
        elif tc == "不良" and best["cond_pool"] == "soft_heavy":
            variant = "不良"

        return {
            "race_id": str(row.get("race_id", "")),
            "date": str(row.get("date", "")),
            "venue": venue,
            "surface": surface,
            "layout": layout,
            "distance": distance,
            "grade": str(row.get("grade", "")),
            "class_band": class_band,
            "track_condition": tc,
            "cond_variant": variant,
            "cond_pool": best["cond_pool"],
            "round": parse_round(row),
            "race_name": str(row.get("race_name", "")),
            "time_2nd": round(time_raw, 1),
            "time_2nd_adj": round(time_2nd, 1),
            "time_2nd_fmt": format_race_time(time_raw),
            "first_half_3f": (
                round(fh, 1)
                if fh is not None and not (isinstance(fh, float) and math.isnan(fh))
                else None
            ),
            "pace_adj_sec": pace_adj,
            "pace_label": pace_label,
            "pace_rpci": (
                round(pace_rpci, 4)
                if pace_rpci is not None
                and not (isinstance(pace_rpci, float) and math.isnan(pace_rpci))
                else None
            ),
            "pace_z": (
                round(pace_z, 3)
                if pace_z is not None
                and not (isinstance(pace_z, float) and math.isnan(pace_z))
                else None
            ),
            "pace_metric": pace_metric or "",
            "pace_metric_mean": (
                round(float(pace_res.metric_mean), 3)
                if pace_res.metric_mean is not None else None
            ),
            "pace_metric_std": (
                round(float(pace_res.metric_std), 3)
                if pace_res.metric_std is not None else None
            ),
            "z": best["z"],
            "z_raw": best["z_raw"],
            "baba_index": z_to_baba_index(best["z"]),
            "label": label,
            "label_raw": classify_speed(best["z_raw"])[0],
            "level": level,
            "level_raw": level_raw,
            "baseline_mean": best["baseline_mean"],
            "baseline_std": best["baseline_std"],
            "baseline_key": best["baseline_key"],
            "class_level_label": self._classify_class_level(
                time_2nd, venue, layout, surface, distance, best["cond_pool"],
                class_band=class_band,
            ),
        }

    def assign_races(
        self,
        years: list[str],
        date_min: str | None = None,
        date_max: str | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> dict[str, Path]:
        if not self._baselines:
            if not self.load_baselines():
                raise FileNotFoundError(f"ベースライン未生成: {BASELINES_PATH}")

        races = self.load_races_from_parquet(
            years, date_min=date_min, date_max=date_max, progress_cb=progress_cb,
        )
        scored: list[dict] = []
        for _, r in races.iterrows():
            s = self.score_race(r)
            if s:
                scored.append(s)

        if not scored:
            raise ValueError("スコア付きレースが0件")

        from src.research.race.perf_index import (
            add_perf_index, add_perf_level, build_all_temporal_z_maps,
            invalidate_sigma_cache,
        )

        out_df = pd.DataFrame(scored)
        out_df["date"] = out_df["date"].astype(str).str.replace("-", "", regex=False)

        # テンポラルプーリング用: 対象期間の最初の1週間に前年データが必要な場合はロード
        extra_df: pd.DataFrame | None = None
        all_years_in_data = sorted(out_df["date"].str[:4].unique())
        if all_years_in_data:
            earliest_year = all_years_in_data[0]
            prev_year_p = RACES_DIR / f"races_{int(earliest_year) - 1}.parquet"
            if prev_year_p.exists():
                try:
                    extra_df = pd.read_parquet(
                        prev_year_p, columns=["date", "venue", "surface", "z", "track_condition"]
                    )
                except Exception:
                    extra_df = None

        # 全日付のテンポラルプーリングz-mapを一括構築
        invalidate_sigma_cache()
        temporal_z_map = build_all_temporal_z_maps(out_df, extra_df)
        out_df = add_perf_index(out_df, temporal_z_map)
        out_df = add_perf_level(out_df)

        RACES_DIR.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for year, grp in out_df.groupby(out_df["date"].astype(str).str[:4]):
            p = RACES_DIR / f"races_{year}.parquet"
            if p.exists():
                old = pd.read_parquet(p)
                grp = pd.concat([old, grp]).drop_duplicates("race_id", keep="last")
            grp.to_parquet(p, index=False)
            paths[str(year)] = p

        meta = {
            "built_at": pd.Timestamp.now().isoformat(),
            "baseline_path": str(BASELINES_PATH),
            "baseline_groups": len(self._baselines),
            "races_scored": len(scored),
            "years": list(paths.keys()),
        }
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress_cb:
            progress_cb(f"振り分け完了: {len(scored)} R → {RACES_DIR}")
        return paths

    def query_day(self, date: str, venue: str | None = None) -> dict[str, Any]:
        """date: YYYYMMDD or YYYY-MM-DD"""
        from datetime import datetime as _dt, timedelta as _td
        from src.research.race.perf_index import (
            add_perf_index, add_perf_level,
            build_temporal_pooled_z_stats,
        )

        d = date.replace("-", "")
        year = d[:4]
        p = RACES_DIR / f"races_{year}.parquet"
        if not p.exists():
            return {"date": d, "races": [], "summary": {}}

        df = pd.read_parquet(p)
        df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
        day_df = df[df["date"] == d]

        # parquetにperf_indexが既に保存されているか確認
        has_stored_perf = "perf_index" in day_df.columns and day_df["perf_index"].notna().any()

        if has_stored_perf:
            # 保存済みperf_indexを使用（確定済み履歴データ）
            pooled_z_stats: dict = {}
            daily_mean_z_map: dict = {}
        else:
            # フォールバック: テンポラルプーリングでリアルタイム計算
            dt_obj = _dt.strptime(d, "%Y%m%d")
            prior_day = (dt_obj - _td(days=1)).strftime("%Y%m%d")
            prior_week = (dt_obj - _td(days=7)).strftime("%Y%m%d")
            extra_years = {prior_day[:4], prior_week[:4]} - {year}
            for ey in extra_years:
                ep = RACES_DIR / f"races_{ey}.parquet"
                if ep.exists():
                    edf = pd.read_parquet(ep)
                    edf["date"] = edf["date"].astype(str).str.replace("-", "", regex=False)
                    df = pd.concat([df, edf], ignore_index=True)
            wide_df = df[df["date"].isin([d, prior_day, prior_week])]
            pooled_z_stats = build_temporal_pooled_z_stats(wide_df, d)
            daily_mean_z_map = {k: s["mean_z"] for k, s in pooled_z_stats.items() if "mean_z" in s}

        sub = day_df.copy()
        if venue:
            sub = sub[sub["venue"] == venue]
        if len(sub) and "race_name" in sub.columns:
            sub = sub[~sub.apply(is_obstacle_race, axis=1)]
        sub = enrich_distance(sub, [year], self.base_dir)
        if len(sub) and "race_id" in sub.columns:
            sub = sub.copy()
            bad = sub["round"].fillna(0).astype(int) == 0
            if bad.any():
                sub.loc[bad, "round"] = (
                    sub.loc[bad, "race_id"].astype(str).str[-2:].astype(int)
                )
        sort_cols = ["round"] if venue else ["venue", "round"]
        sub = sub.sort_values(sort_cols)

        if not has_stored_perf:
            sub = add_perf_index(sub, daily_mean_z_map)
            sub = add_perf_level(sub)

        sub = self._attach_class_level_labels(sub)
        races = sub.to_dict(orient="records")
        for r in races:
            r["time_2nd_fmt"] = format_race_time(r.get("time_2nd"))
            if r.get("z") is not None:
                r["baba_index"] = z_to_baba_index(float(r["z"]))

        # サマリー: 保存済みperf_indexがある場合もtemp pooling statsでsummaryを構築
        # (n_prior_sources等の情報は別途計算が必要なため)
        if not pooled_z_stats:
            from src.research.race.perf_index import build_temporal_pooled_z_stats as _btz
            from datetime import datetime as _dt2, timedelta as _td2
            dt_obj2 = _dt.strptime(d, "%Y%m%d")
            prior_day2 = (dt_obj2 - _td(days=1)).strftime("%Y%m%d")
            prior_week2 = (dt_obj2 - _td(days=7)).strftime("%Y%m%d")
            df2 = pd.read_parquet(p)
            df2["date"] = df2["date"].astype(str).str.replace("-", "", regex=False)
            extra_years2 = {prior_day2[:4], prior_week2[:4]} - {year}
            for ey in extra_years2:
                ep = RACES_DIR / f"races_{ey}.parquet"
                if ep.exists():
                    edf2 = pd.read_parquet(ep)
                    edf2["date"] = edf2["date"].astype(str).str.replace("-", "", regex=False)
                    df2 = pd.concat([df2, edf2], ignore_index=True)
            wide_df2 = df2[df2["date"].isin([d, prior_day2, prior_week2])]
            pooled_z_stats = _btz(wide_df2, d)

        summary: dict[str, dict] = {}
        for (v, surf), g in sub.groupby(["venue", "surface"]):
            if g["z"].dropna().empty:
                continue
            key = f"{d}|{v}|{surf}"
            stats = pooled_z_stats.get(key, {})
            mean_z = stats.get("mean_z") if stats else None
            if mean_z is None:
                mean_z = float(g["z"].dropna().mean())
            mean_index = z_to_baba_index(mean_z)
            lbl, lvl = classify_speed(mean_z)
            summary[f"{v}_{surf}"] = {
                "venue": v,
                "surface": surf,
                "mean_z": round(mean_z, 3),
                "mean_index": mean_index,
                "label": lbl,
                "level": lvl,
                "n_races": len(g),
                "n_inliers": stats.get("n_inliers", len(g)),
                "n_outliers": stats.get("n_outliers", 0),
                "z_std": stats.get("z_std"),
                "confidence": stats.get("confidence", "mid"),
                "n_prior_sources": stats.get("n_prior_sources", 0),
                "track_conditions": sorted(g["track_condition"].unique().tolist()),
            }
        return json_safe({"date": d, "races": races, "summary": summary})

    def list_dates(self, venue: str | None = None) -> list[str]:
        dates: set[str] = set()
        for p in sorted(RACES_DIR.glob("races_*.parquet")):
            df = pd.read_parquet(p, columns=["date", "venue"])
            df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
            if venue:
                df = df[df["venue"] == venue]
            dates.update(df["date"].unique())
        return sorted(dates, reverse=True)

    def list_venues(self) -> list[str]:
        venues: set[str] = set()
        for p in RACES_DIR.glob("races_*.parquet"):
            df = pd.read_parquet(p, columns=["venue"])
            venues.update(df["venue"].dropna().unique())
        return sorted(venues)

    def list_venues_for_date(self, date: str) -> list[str]:
        """指定日に開催がある競馬場のみ返す。"""
        d = date.replace("-", "")
        year = d[:4]
        p = RACES_DIR / f"races_{year}.parquet"
        if not p.exists():
            return []
        df = pd.read_parquet(p, columns=["date", "venue"])
        df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
        sub = df[df["date"] == d]
        return sorted(sub["venue"].dropna().unique().tolist())

    def query_by_category(
        self,
        venue: str,
        level: int,
        surface: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """競馬場×馬場速度カテゴリ(level 1-5)で直近レースをページング返却。"""
        from src.research.race.perf_index import add_perf_index, add_perf_level, build_daily_mean_z_map

        read_cols = [
            "race_id", "date", "venue", "surface", "layout", "distance",
            "grade", "class_band", "track_condition", "cond_variant", "cond_pool",
            "round", "race_name", "time_2nd", "time_2nd_adj",
            "first_half_3f", "pace_adj_sec", "pace_label", "pace_rpci",
            "pace_z", "pace_metric", "pace_metric_mean", "pace_metric_std",
            "z", "baba_index", "label", "level",
            "baseline_std", "perf_index", "perf_index_std2000",
            "perf_level", "perf_level_label",
        ]
        import pyarrow.parquet as _pq

        frames: list[pd.DataFrame] = []
        for p in sorted(RACES_DIR.glob("races_*.parquet"), reverse=True):
            available = set(_pq.read_schema(p).names)
            year_cols_needed = ["race_id", "date", "venue", "surface", "z", "baseline_std", "level", "round"]
            extra_cols = [c for c in read_cols if c in available]
            load_cols = list(set(year_cols_needed + extra_cols) & available)
            df_full = pd.read_parquet(p, columns=load_cols)
            df_full["date"] = df_full["date"].astype(str).str.replace("-", "", regex=False)

            # 保存済みperf_indexがあればそのまま使用; なければ従来の計算
            if "perf_index" not in df_full.columns or df_full["perf_index"].isna().all():
                daily_mean_z_map = build_daily_mean_z_map(df_full)
                df_full = add_perf_index(df_full, daily_mean_z_map)
                df_full = add_perf_level(df_full)

            # Filter by venue/level/surface
            mask = (df_full["venue"] == venue) & (df_full["level"].fillna(0).astype(int) == level)
            if surface:
                mask &= df_full["surface"] == surface
            sub = df_full[mask]
            if not sub.empty:
                frames.append(sub)

        if not frames:
            return {"races": [], "total": 0, "offset": offset, "limit": limit}

        all_df = pd.concat(frames, ignore_index=True)
        all_df = all_df.sort_values(["date", "round"], ascending=[False, False])
        all_df = all_df.drop_duplicates("race_id", keep="first")

        total = len(all_df)
        page_df = self._attach_class_level_labels(all_df.iloc[offset: offset + limit])
        races = page_df.to_dict(orient="records")
        for r in races:
            r["time_2nd_fmt"] = format_race_time(r.get("time_2nd"))
            if r.get("z") is not None:
                r["baba_index"] = z_to_baba_index(float(r["z"]))

        return json_safe({
            "races": races,
            "total": total,
            "offset": offset,
            "limit": limit,
        })
