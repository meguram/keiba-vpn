"""
レース適性マップ v2 — 血統メタクラスタベース

旧アプローチ（6祖先固定重みブレンド）に代わり、
bloodline_meta_cluster.compute_blended_prior_v2 を利用する。

2D 座標:
  l2_positions_2d.json の PCA 空間。父系(sire_id) 65% + 母父系(bms_root_id) 35%。
  どちらかしか pca.stallions にない場合は 100% 側、両方なければ entity_l2 → centroid。

ベストゾーン:
  race_result_slim.parquet の同会場×馬場タイプ全期間 top3 馬の重み付き重心
  （直近 look_back_weeks 週に限定）。

当日傾向:
  当日同会場×馬場の完了済みレース（HybridStorage 経由）top3 馬の重心。
"""
from __future__ import annotations

import json
import logging
import threading
import time as _time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[3]
_ART_DIR = _BASE / "data" / "local" / "research" / "bloodline_meta_cluster"
_IDX_DIR = _BASE / "data" / "local" / "research" / "pedigree_race_index"

# ---------------------------------------------------------------------------
# Position data cache
# ---------------------------------------------------------------------------
_pos_cache: dict | None = None
_pos_cache_lock = threading.Lock()
_POS_CACHE_TTL = 3600


def _compute_axis_labels(centroids: dict, l2_names: dict, evr: list | None = None) -> dict:
    """PCA centroid の両端クラスタ名から軸ラベルを生成する。"""
    if not centroids:
        return {"x_neg": "←", "x_pos": "→", "y_neg": "↓", "y_pos": "↑", "evr_x": 0.0, "evr_y": 0.0}

    def short(name: str) -> str:
        return name.replace("型", "").replace("系", "")[:10]

    items = [(k, v[0], v[1]) for k, v in centroids.items()]
    x_neg = min(items, key=lambda t: t[1])
    x_pos = max(items, key=lambda t: t[1])
    y_neg = min(items, key=lambda t: t[2])
    y_pos = max(items, key=lambda t: t[2])

    def name_of(k: str) -> str:
        return l2_names.get(k, {}).get("name", f"L2-{k}")

    evr = evr or [0.0, 0.0]
    return {
        "x_neg": short(name_of(x_neg[0])),
        "x_pos": short(name_of(x_pos[0])),
        "y_neg": short(name_of(y_neg[0])),
        "y_pos": short(name_of(y_pos[0])),
        "evr_x": round(evr[0] * 100, 1) if evr else 0.0,
        "evr_y": round(evr[1] * 100, 1) if len(evr) > 1 else 0.0,
    }


def _load_pos_data() -> dict:
    global _pos_cache
    now = _time.time()
    if _pos_cache and (now - _pos_cache.get("_ts", 0)) < _POS_CACHE_TTL:
        return _pos_cache
    with _pos_cache_lock:
        if _pos_cache and (now - _pos_cache.get("_ts", 0)) < _POS_CACHE_TTL:
            return _pos_cache

        pos: dict = {"_ts": now}

        # l2_positions_2d → pca.stallions / pca.centroids
        p = _ART_DIR / "l2_positions_2d.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            pca = d.get("pca", {})
            pos["pca_stallions"] = {k: v for k, v in pca.get("stallions", {}).items()}
            pos["pca_centroids"] = {k: v for k, v in pca.get("centroids", {}).items()}
            pos["pca_evr"] = pca.get("explained_variance_ratio", [0.0, 0.0])
        else:
            pos["pca_stallions"] = {}
            pos["pca_centroids"] = {}
            pos["pca_evr"] = [0.0, 0.0]

        # l2_names
        p2 = _ART_DIR / "l2_names.json"
        pos["l2_names"] = {}
        if p2.exists():
            d2 = json.loads(p2.read_text(encoding="utf-8"))
            pos["l2_names"] = d2.get("l2_names", {})

        # l2_profiles: {str(L2): {cond: delta_from_1.0, ...}}
        p2b = _ART_DIR / "l2_profiles.json"
        pos["l2_profiles"] = {}
        if p2b.exists():
            pos["l2_profiles"] = json.loads(p2b.read_text(encoding="utf-8"))

        # axis_labels: derive human-readable labels from centroid extremes
        pos["axis_labels"] = _compute_axis_labels(pos["pca_centroids"], pos["l2_names"], pos["pca_evr"])

        # horse_bms: horse_id → {sire_id, sire_name, bms_root_id, bms_root_name, bms_name}
        p3 = _IDX_DIR / "horse_bms.parquet"
        pos["horse_bms"] = {}
        if p3.exists():
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(p3, columns=[
                    "horse_id", "sire_id", "sire_name",
                    "bms_root_id", "bms_root_name", "bms_name",
                ])
                for row in t.to_pylist():
                    pos["horse_bms"][row["horse_id"]] = row
            except Exception as e:
                logger.warning("horse_bms load failed: %s", e)

        # entity_l2: entity_id → L2 cluster (int)
        p4 = _ART_DIR / "unified.parquet"
        pos["entity_l2"] = {}
        if p4.exists():
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(p4, columns=["entity_id", "L2"])
                for row in t.to_pylist():
                    pos["entity_l2"][row["entity_id"]] = row["L2"]
            except Exception as e:
                logger.warning("unified.parquet load failed: %s", e)

        # race_result_slim
        p5 = _IDX_DIR / "race_result_slim.parquet"
        pos["slim_df"] = None
        if p5.exists():
            try:
                import pyarrow.parquet as pq
                t = pq.read_table(p5, columns=[
                    "race_id", "horse_id", "finish_position",
                    "date", "venue", "surface", "distance",
                ])
                pos["slim_df"] = t.to_pandas()
            except Exception as e:
                logger.warning("race_result_slim load failed: %s", e)

        _pos_cache = pos
        return pos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stallion_pos(sid: str, pos: dict) -> list[float] | None:
    """種牡馬 ID → PCA 2D 座標。stallions → entity_l2+centroid の順でフォールバック。"""
    if sid in pos["pca_stallions"]:
        return pos["pca_stallions"][sid]
    l2 = pos["entity_l2"].get(sid)
    if l2 is not None and l2 >= 0:
        c = pos["pca_centroids"].get(str(l2))
        if c:
            return c
    return None


def _get_horse_pos(horse_id: str, pos: dict) -> tuple[float, float]:
    """馬の 2D 座標（父系 65% + 母父系 35%）。"""
    bms = pos["horse_bms"].get(horse_id, {})
    sire_id = bms.get("sire_id", "")
    bms_root_id = bms.get("bms_root_id", "")

    sp = _stallion_pos(sire_id, pos) if sire_id else None
    bp = _stallion_pos(bms_root_id, pos) if bms_root_id else None

    if sp and bp:
        return 0.65 * sp[0] + 0.35 * bp[0], 0.65 * sp[1] + 0.35 * bp[1]
    elif sp:
        return sp[0], sp[1]
    elif bp:
        return bp[0], bp[1]
    return 0.0, 0.0


def _get_dominant_l2(horse_id: str, pos: dict) -> int:
    """馬の支配的 L2 クラスタ（父の L2）。"""
    bms = pos["horse_bms"].get(horse_id, {})
    sire_id = bms.get("sire_id", "")
    return pos["entity_l2"].get(sire_id, -1)


def _distance_class(dist_m: int) -> str:
    """build_role_lift_profiles.py と同一の閾値を使用。"""
    if dist_m < 1400:
        return "短距離"
    elif dist_m < 1800:
        return "マイル"
    elif dist_m < 2400:
        return "中距離"
    return "長距離"


def _get_course_cond_keys(venue: str, surface: str, dist_m: int, track_cond: str) -> list[str]:
    keys: list[str] = []
    if venue:
        keys.append(f"win_v_{venue}")
    if surface:
        keys.append(f"win_s_{surface}")
    if dist_m:
        keys.append(f"win_d_{_distance_class(dist_m)}")
    if track_cond in ("重", "不良"):
        keys.append("win_heavy")
    return keys


def _course_match_score(lift: dict, cond_keys: list[str]) -> float:
    """条件別 lift を距離・馬場重み付き加重平均で集約。

    距離カテゴリが極端（長距離・短距離）な場合は距離 lift の比重を高め、
    マイル戦は会場特性を重視、ダートは馬場適性を重視する。
    重馬場フラグは単独で強い信号として扱う。
    """
    if not cond_keys:
        return 1.0

    # 距離カテゴリと馬場カテゴリを cond_keys から判定
    dist_key = next((k for k in cond_keys if k.startswith("win_d_")), None)
    surface_key = next((k for k in cond_keys if k.startswith("win_s_")), None)
    is_dart = surface_key == "win_s_ダート"
    is_jump = surface_key == "win_s_障害"

    # 距離重み: 極端な距離ほど距離適性が支配的
    if dist_key == "win_d_長距離":
        w_d = 3.0
    elif dist_key == "win_d_短距離":
        w_d = 3.0
    elif dist_key == "win_d_マイル":
        w_d = 1.5
    else:  # 中距離
        w_d = 2.0

    # 会場重み: マイル戦は会場特性（直線長など）が重要、長距離は普通
    w_v = 2.5 if dist_key == "win_d_マイル" else 1.5

    # 馬場重み: ダート・障害は芝と根本的に異なるので重視
    w_s = 2.5 if (is_dart or is_jump) else 0.8

    # 重馬場重み
    w_h = 2.0

    total_w = 0.0
    total = 0.0
    for k in cond_keys:
        if k not in lift:
            continue
        if k.startswith("win_d_"):
            w = w_d
        elif k.startswith("win_v_"):
            w = w_v
        elif k.startswith("win_s_"):
            w = w_s
        else:  # win_heavy
            w = w_h
        total += lift[k] * w
        total_w += w

    return round(total / total_w, 4) if total_w > 0 else 1.0


def _weighted_centroid(pts: list[tuple[float, float, float]]) -> dict | None:
    """pts = [(x, y, weight), ...] → center + radius。"""
    if not pts:
        return None
    total_w = sum(p[2] for p in pts)
    if total_w <= 0:
        return None
    cx = sum(p[0] * p[2] for p in pts) / total_w
    cy = sum(p[1] * p[2] for p in pts) / total_w
    var = sum(((p[0] - cx) ** 2 + (p[1] - cy) ** 2) * p[2] for p in pts) / total_w
    return {"x": round(cx, 3), "y": round(cy, 3), "r": round(var ** 0.5, 3), "n": len(pts)}


# ---------------------------------------------------------------------------
# Best zone: race_result_slim の同会場×馬場 top3 重心
# ---------------------------------------------------------------------------

def _compute_best_zone(
    pos: dict,
    venue: str,
    surface: str,
    look_back_weeks: int = 8,
    ref_date_str: str = "",
) -> dict | None:
    df = pos.get("slim_df")
    if df is None:
        return None

    try:
        mask = (df["venue"] == venue) & (df["surface"] == surface)
        if ref_date_str:
            from datetime import datetime, timedelta
            ref = datetime.strptime(ref_date_str[:10], "%Y-%m-%d")
            date_from = (ref - timedelta(weeks=look_back_weeks)).strftime("%Y-%m-%d")
            mask &= df["date"] >= date_from
            mask &= df["date"] < ref.strftime("%Y-%m-%d")
        sub = df[mask & (df["finish_position"] > 0) & (df["finish_position"] <= 3)]
        if sub.empty:
            # fallback: all history
            sub = df[mask & (df["finish_position"] > 0) & (df["finish_position"] <= 3)]
        if sub.empty:
            return None

        pts: list[tuple[float, float, float]] = []
        l2_cnt: dict[int, int] = {}
        for _, row in sub.iterrows():
            hid = str(row["horse_id"])
            px, py = _get_horse_pos(hid, pos)
            if px == 0.0 and py == 0.0:
                continue
            w = float(4 - int(row["finish_position"]))
            pts.append((px, py, w))
            l2 = _get_dominant_l2(hid, pos)
            l2_cnt[l2] = l2_cnt.get(l2, 0) + int(row["finish_position"] == 1)

        centroid = _weighted_centroid(pts)
        if not centroid:
            return None

        top_l2 = sorted(l2_cnt.items(), key=lambda x: -x[1])[:3]
        centroid["top_l2"] = [
            {
                "l2_id": l2,
                "win_count": cnt,
                "name": pos["l2_names"].get(str(l2), {}).get("name", ""),
                "color": pos["l2_names"].get(str(l2), {}).get("color", "#888"),
            }
            for l2, cnt in top_l2 if l2 >= 0
        ]
        centroid["label"] = f"過去{look_back_weeks}週 {venue}{surface} 好走域"
        return centroid
    except Exception as e:
        logger.warning("best_zone failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Same-day tendency: 当日完了済み同会場×馬場
# ---------------------------------------------------------------------------

def _compute_same_day_tendency(
    storage: Any,
    pos: dict,
    venue: str,
    surface: str,
    date_str: str,
    exclude_race_id: str = "",
) -> dict | None:
    try:
        from datetime import date as _date
        date_key = date_str.replace("-", "")
        # Future races can't have results yet — skip 36 serial GCS misses
        try:
            race_dt = _date(int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8]))
            if race_dt > _date.today():
                return None
        except Exception:
            pass
        rl = storage.load("race_lists", date_key)
        if not rl:
            return None
        pts: list[tuple[float, float, float]] = []
        seen_venues: set[str] = set()
        for race in rl.get("races", []):
            rid = str(race.get("race_id", ""))
            if not rid or rid == exclude_race_id:
                continue
            if str(race.get("venue", "")) != venue:
                continue
            result = storage.load("race_result", rid)
            if not result:
                continue
            if str(result.get("surface", "")) != surface:
                continue
            seen_venues.add(rid)
            for entry in result.get("entries", []):
                fp = int(entry.get("finish_position") or 0)
                if not 1 <= fp <= 3:
                    continue
                hid = str(entry.get("horse_id") or "")
                if not hid:
                    continue
                px, py = _get_horse_pos(hid, pos)
                if px == 0.0 and py == 0.0:
                    continue
                pts.append((px, py, float(4 - fp)))
        if not pts:
            return None
        centroid = _weighted_centroid(pts)
        if centroid:
            centroid["label"] = f"当日 {venue}{surface} 済レース傾向"
            centroid["races_done"] = len(seen_venues)
        return centroid
    except Exception as e:
        logger.warning("same_day_tendency failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Individual performance correction
# ---------------------------------------------------------------------------

def _compute_individual_performance_batch(
    horse_ids: list,
    pos: dict,
    cond_keys: list,
) -> dict:
    """
    race_result_slim から出走全馬の条件別個体成績を一括算出。

    Returns:
        {horse_id: {
            "ind_lift":       {key: float},   # 条件別 individual lift
            "n_total":        int,            # 全成績数
            "n_cond_races":   int,            # 最大条件別レース数
            "ind_weight":     float,          # ブレンド重み (0〜0.5)
        }}
    """
    import pandas as pd

    df = pos.get("slim_df")
    if df is None or not horse_ids or not cond_keys:
        return {}

    MIN_RACES_PER_COND = 3   # 補正を掛ける最低条件別レース数
    MAX_WEIGHT = 0.5         # ブレンド重みの上限
    WEIGHT_SCALE = 15        # このレース数で MAX_WEIGHT に到達

    # slim_df から対象馬を絞り込み（0 着を除外）
    valid = df[df["finish_position"].notna() & (df["finish_position"] > 0)]
    horse_df = valid[valid["horse_id"].isin(horse_ids)].copy()
    if horse_df.empty:
        return {}

    # distance → dist_class 列を追加（slim_df が整数 distance を持つ前提）
    def _dc(d):
        try:
            return _distance_class(int(d))
        except Exception:
            return ""

    horse_df["dist_class"] = horse_df["distance"].apply(_dc)

    # 母集団のベースライン top3 率（cond_keys に対応する分だけ計算）
    pop_valid = valid.copy()
    pop_valid["dist_class"] = pop_valid["distance"].apply(_dc)

    pop_baselines: dict = {}
    for key in cond_keys:
        if key.startswith("win_v_"):
            v = key[6:]
            sub = pop_valid[pop_valid["venue"] == v]
        elif key.startswith("win_s_"):
            s = key[6:]
            sub = pop_valid[pop_valid["surface"] == s]
        elif key.startswith("win_d_"):
            dc = key[6:]
            sub = pop_valid[pop_valid["dist_class"] == dc]
        elif key == "win_heavy":
            continue  # slim_df に track_condition 列なし
        else:
            continue
        if len(sub) >= 20:
            top3_rate = float((sub["finish_position"] <= 3).mean())
            if top3_rate > 0:
                pop_baselines[key] = top3_rate

    if not pop_baselines:
        return {}

    result: dict = {}
    for hid in horse_ids:
        h_df = horse_df[horse_df["horse_id"] == hid]
        n_total = len(h_df)

        ind_lift: dict = {}
        n_cond_max = 0

        for key, pop_rate in pop_baselines.items():
            if key.startswith("win_v_"):
                h_cond = h_df[h_df["venue"] == key[6:]]
            elif key.startswith("win_s_"):
                h_cond = h_df[h_df["surface"] == key[6:]]
            elif key.startswith("win_d_"):
                h_cond = h_df[h_df["dist_class"] == key[6:]]
            else:
                continue

            n_cond = len(h_cond)
            if n_cond < MIN_RACES_PER_COND:
                continue

            n_cond_max = max(n_cond_max, n_cond)
            horse_rate = float((h_cond["finish_position"] <= 3).mean())
            ind_lift[key] = round(horse_rate / pop_rate, 4)

        if not ind_lift or n_cond_max < MIN_RACES_PER_COND:
            continue

        weight = round(min(n_cond_max / WEIGHT_SCALE * MAX_WEIGHT, MAX_WEIGHT), 3)
        result[hid] = {
            "ind_lift": ind_lift,
            "n_total": n_total,
            "n_cond_races": n_cond_max,
            "ind_weight": weight,
        }

    return result


def _blend_lift(
    pedigree_lift: dict,
    ind_data: dict | None,
    cond_keys: list,
) -> tuple[dict, float]:
    """
    血統 prior と個体 posterior を混合した最終 lift を返す。
    Returns: (final_lift, blend_weight)
    """
    if not ind_data or not ind_data.get("ind_lift"):
        return pedigree_lift, 0.0

    w = ind_data["ind_weight"]
    ind_lift = ind_data["ind_lift"]

    final: dict = {}
    for k, ped_v in pedigree_lift.items():
        if k in ind_lift:
            final[k] = round((1 - w) * ped_v + w * ind_lift[k], 4)
        else:
            final[k] = ped_v
    # cond_keys にあって pedigree_lift にないキー
    for k in cond_keys:
        if k not in final and k in ind_lift:
            final[k] = ind_lift[k]

    return final, w


# ---------------------------------------------------------------------------
# Race list fallback
# ---------------------------------------------------------------------------

def _find_race_in_lists(storage: Any, race_id: str) -> dict:
    """race_id をスキャンして race_lists から基本情報を取得する (shutuba 未取得時フォールバック)。"""
    from datetime import date, timedelta
    today = date.today()
    for delta in range(-14, 22):
        d = today + timedelta(days=delta)
        date_key = d.strftime("%Y%m%d")
        rl = storage.load("race_lists", date_key)
        if not rl:
            continue
        for r in rl.get("races", []):
            if str(r.get("race_id", "")) == race_id:
                raw_date = str(r.get("date", ""))
                if len(raw_date) == 8:
                    fmt_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                else:
                    fmt_date = raw_date
                return {
                    "race_name": r.get("race_name", ""),
                    "venue": r.get("venue", ""),
                    "surface": "",
                    "distance": 0,
                    "track_condition": "",
                    "date": fmt_date,
                    "post_time": "",
                }
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_race_note_v2(storage: Any, race_id: str) -> dict:
    """レース適性マップ v2 データを構築して返す。"""
    from src.api.bloodline_meta_cluster import compute_blended_prior_v2, _load_artifacts

    rid = race_id.strip()
    pos = _load_pos_data()
    art = _load_artifacts()  # load once, reuse for all horses

    # ── レースデータ取得 ──────────────────────────────────
    race_meta: dict = {}
    entries: list[dict] = []
    source = "none"

    shutuba = storage.load("race_shutuba", rid)
    result = storage.load("race_result", rid)

    if shutuba:
        entries = shutuba.get("entries", [])
        source = "race_shutuba"
        race_meta = {
            "race_name": shutuba.get("race_name", ""),
            "venue": shutuba.get("venue", ""),
            "surface": shutuba.get("surface", ""),
            "distance": int(shutuba.get("distance") or 0),
            "track_condition": shutuba.get("track_condition", ""),
            "date": shutuba.get("date", ""),
            "post_time": shutuba.get("post_time", ""),
        }
    if not entries and result:
        entries = result.get("entries", [])
        source = "race_result"
        race_meta = {
            "race_name": result.get("race_name", ""),
            "venue": result.get("venue", ""),
            "surface": result.get("surface", ""),
            "distance": int(result.get("distance") or 0),
            "track_condition": result.get("track_condition", ""),
            "date": result.get("date", ""),
            "post_time": "",
        }
    # shutuba / result どちらもなければ race_lists からメタ情報だけ取得
    if not race_meta.get("venue"):
        race_meta = _find_race_in_lists(storage, rid)
        if race_meta:
            source = "race_lists_only"

    venue = race_meta.get("venue", "")
    surface = race_meta.get("surface", "")
    dist_m = race_meta.get("distance", 0)
    track_cond = race_meta.get("track_condition", "")
    race_date = race_meta.get("date", "")
    cond_keys = _get_course_cond_keys(venue, surface, dist_m, track_cond)

    # ── 出走馬ごとの適性プロファイル ─────────────────────
    horses_out: list[dict] = []

    # ── 個体成績の一括算出 ──────────────────────────────────
    all_horse_ids = [
        str(e.get("horse_id") or "").strip()
        for e in entries if e.get("horse_id")
    ]
    ind_perf = _compute_individual_performance_batch(
        all_horse_ids, pos, cond_keys
    )

    for entry in entries:
        hid = str(entry.get("horse_id") or "").strip()
        if not hid:
            continue

        prior = None
        try:
            prior = compute_blended_prior_v2(hid, art=art)
        except Exception:
            pass

        lift: dict = {}
        course_match = 1.0
        layer_meta: dict = {}
        ind_adj = False
        ind_weight = 0.0
        ind_n_races = 0
        ind_n_cond = 0
        if prior:
            ped_lift = prior.get("lift", {})
            ind_data = ind_perf.get(hid)
            lift, ind_weight = _blend_lift(ped_lift, ind_data, cond_keys)
            ind_adj = ind_weight > 0
            if ind_data:
                ind_n_races = ind_data.get("n_total", 0)
                ind_n_cond = ind_data.get("n_cond_races", 0)
            course_match = _course_match_score(lift, cond_keys)
            layer_meta = {
                "n_layers": prior.get("n_layers_used", 0),
                "weights": prior.get("weights_used", {}),
            }

        px, py = _get_horse_pos(hid, pos)
        l2 = _get_dominant_l2(hid, pos)
        l2_info = pos["l2_names"].get(str(l2), {}) if l2 >= 0 else {}
        bms_row = pos["horse_bms"].get(hid, {})

        # 条件別 lift の上位3件
        top_conds = sorted(
            [(k.replace("win_", ""), v) for k, v in lift.items() if isinstance(v, float)],
            key=lambda x: -x[1],
        )[:4]

        horses_out.append({
            "horse_id": hid,
            "horse_name": str(entry.get("horse_name") or ""),
            "horse_number": int(entry.get("horse_number") or 0),
            "bracket_number": int(entry.get("bracket_number") or 0),
            "pos_x": round(px, 3),
            "pos_y": round(py, 3),
            "dominant_l2": l2,
            "l2_name": l2_info.get("name", ""),
            "l2_color": l2_info.get("color", "#888"),
            "course_match": course_match,
            "lift": lift,
            "top_conds": [{"key": k, "lift": round(v, 3)} for k, v in top_conds],
            "sire_name": str(bms_row.get("sire_name") or ""),
            "bms_name": str(bms_row.get("bms_name") or ""),
            "bms_root_name": str(bms_row.get("bms_root_name") or ""),
            "layer_meta": layer_meta,
            "ind_adj": ind_adj,
            "ind_weight": round(ind_weight, 3),
            "ind_n_races": ind_n_races,
            "ind_n_cond": ind_n_cond,
        })

    # course_match_rank: 1 = best (lower number = better aptitude)
    for rank_i, h in enumerate(
        sorted(horses_out, key=lambda x: x["course_match"], reverse=True), start=1
    ):
        h["course_match_rank"] = rank_i

    # ── L2 クラスタ一覧 ───────────────────────────────────
    l2_clusters: list[dict] = []
    for l2_id_str, l2_info in pos["l2_names"].items():
        centroid = pos["pca_centroids"].get(l2_id_str)
        if not centroid:
            continue
        l2_id = int(l2_id_str)
        # クラスタ centroid の course_match: l2_profiles の delta → lift → score
        cluster_profile = pos["l2_profiles"].get(l2_id_str, {})
        cluster_lift = {k: v + 1.0 for k, v in cluster_profile.items()}
        cluster_course_match = _course_match_score(cluster_lift, cond_keys) if cond_keys else 1.0
        l2_clusters.append({
            "l2_id": l2_id,
            "name": l2_info.get("name", ""),
            "color": l2_info.get("color", "#888"),
            "icon": l2_info.get("icon", "●"),
            "pos_x": round(centroid[0], 3),
            "pos_y": round(centroid[1], 3),
            "horse_count": sum(1 for h in horses_out if h["dominant_l2"] == l2_id),
            "cluster_course_match": round(cluster_course_match, 4),
        })

    # ── ベストゾーン + 当日傾向 ───────────────────────────
    best_zone = _compute_best_zone(pos, venue, surface, look_back_weeks=8, ref_date_str=race_date)
    same_day = _compute_same_day_tendency(storage, pos, venue, surface, race_date, rid) if race_date else None

    return {
        "race_id": rid,
        "source": source,
        "course": {
            "venue": venue,
            "surface": surface,
            "distance": dist_m,
            "distance_class": _distance_class(dist_m) if dist_m else "",
            "track_condition": track_cond,
            "race_name": race_meta.get("race_name", ""),
            "date": race_date,
            "post_time": race_meta.get("post_time", ""),
            "cond_keys": cond_keys,
        },
        "horses": horses_out,
        "best_zone": best_zone,
        "same_day_tendency": same_day,
        "l2_clusters": l2_clusters,
        "axis_labels": pos.get("axis_labels", {}),
    }


# ---------------------------------------------------------------------------
# Week races helper
# ---------------------------------------------------------------------------

def get_week_races(storage: Any) -> list[dict]:
    """今週の開催（土・日）のレース一覧を返す。"""
    from datetime import date, timedelta

    today = date.today()
    wd = today.weekday()  # 0=Mon … 5=Sat, 6=Sun

    # This weekend: this Sat and Sun
    days_to_sat = (5 - wd) % 7
    sat = today + timedelta(days=days_to_sat)
    sun = sat + timedelta(days=1)
    # Also include last weekend if today is Mon-Fri
    prev_sat = sat - timedelta(days=7)
    prev_sun = sun - timedelta(days=7)

    candidates = [prev_sat, prev_sun, sat, sun]
    races: list[dict] = []
    seen: set[str] = set()

    for d in candidates:
        date_key = d.strftime("%Y%m%d")
        rl = storage.load("race_lists", date_key)
        if not rl:
            continue
        for r in rl.get("races", []):
            rid = str(r.get("race_id", ""))
            if not rid or rid in seen:
                continue
            seen.add(rid)
            races.append({
                "race_id": rid,
                "race_name": str(r.get("race_name") or ""),
                "venue": str(r.get("venue") or ""),
                "round": int(r.get("round") or 0),
                "date": d.strftime("%Y-%m-%d"),
                "date_label": d.strftime("%m/%d") + ("(土)" if d.weekday() == 5 else "(日)"),
            })

    return races
