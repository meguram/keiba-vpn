"""
出走馬の走法マップ（めぐ分析）。

走法プロファイルの 5 方面（tp-25 の TS/GC、tp-26 の前後半比・先行度・ポジション変化）を 2 次元へ写し、
「この馬とあの馬は走り方が近い」を見えるようにする。さらに、コース（競馬場・芝ダ・距離）ごとに
「地図のどのゾーンが好走しやすいか」を色で重ねる（running_map_aptitude.py）。

方式:
  - 各方面は、全馬の分布で順位 → 標準正規へ変換してそろえる（split_ratio の ±1600 のような外れ値に強い）。
  - TS と GC は馬ごとの値の順位相関が 1.00 で実質同じ軸なので、重みを 1/√2 ずつにして 2 軸で 1 方面ぶんにする。
  - 2 次元化は「2020 年以降の全馬」で 1 回だけ作った共通の主成分（reference に保存）。どのレースでも同じ座標系なので、
    同じ位置＝同じ走り方で、コース適性の色塗りをそのまま重ねられる。
    出走馬の数や顔ぶれで地図の形が変わらない代わりに、そのレースの出走馬だけで作る主成分（古典的 MDS）より
    元の距離の再現度は少し落ちる（fidelity.distance_spearman で毎回示す）。
  - 向きは固定（末脚の矢印が真上、先行力の矢印が左）。
  - グループは 2 次元ではなく元の 5 次元で Ward 法の階層クラスタリング。数はシルエット係数が最大の 2〜5。

走法データは src/research/race/running_profile.py で出走記録（data/page_reference/tables）から作る:
  - 各馬のプロファイルは、対象レースの日より前の走だけから作る（前走まで入る）。出走の少ない馬は父の産駒の平均へ寄せ、
    新馬は父の産駒の平均そのもの（estimated=True。地図には載せるが、推定であることを画面で示す）。
  - 全馬の分布（順位の基準）・共通の主成分・コース適性の表は、最初の 1 回だけ全期間のデータで作って保存し、以後は読むだけ。
    コース適性の倍率と有意性は 2020 年以降の全レースで測った値（標本内）なので、期間内のレースに対しては結果を含んだ値になる。
    データの最終日は data_until で返す。
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.research.race import running_profile as rp

# (キー, 列, 向き, 表示名, 高いときの説明, 低いときの説明)。向き -1 は値が小さいほど「高い」側
AXES: list[tuple[str, str, int, str, str, str]] = [
    ("ts", "ts_score", 1, "末脚の速さ", "末脚が速い", "末脚が遅い"),
    ("gc", "gc_score", 1, "加速の鋭さ", "加速が鋭い", "加速が鈍い"),
    ("split", "split_ratio", 1, "前半型のペース", "前半から速い", "後半に速い"),
    ("front", "front_ratio", -1, "先行力", "前に行く", "後ろから行く"),
    ("shift", "shift_ratio", -1, "道中の押し上げ", "道中で位置を上げる", "道中で位置を下げる"),
]
AXIS_KEYS = [a[0] for a in AXES]
WEIGHTS = np.array([1 / math.sqrt(2), 1 / math.sqrt(2), 1, 1, 1])
SIGNS = np.array([a[2] for a in AXES])
# グループの説明では TS と GC を 1 つの「末脚」として扱う
_TRAIT_AXES = [("末脚", [0, 1], "末脚が速い", "末脚が遅い")] + [
    (a[3], [i], a[4], a[5]) for i, a in enumerate(AXES) if a[0] not in ("ts", "gc")
]

VENUE_JA = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}

MIN_HORSES = 3
MAX_IMPUTED = 3  # これより多くの方面が欠けた馬は地図に載せない


# ── データ ────────────────────────────────────────────────────────────────
# 計算は最初の 1 回だけ。1 走ごとの値は running_profile.load_metrics が保存・追加計算し、
# 基準（全馬の分布の分位点・共通の主成分・コース適性の表）は REFERENCE_DIR に保存して、以後は読むだけ。
# 基準は作り直さない（データが増えても分布はほぼ変わらない）。作り方を変えたら REFERENCE_VERSION を上げる。

REFERENCE_DIR = rp.STORE_DIR / "reference"
REFERENCE_VERSION = "2"
_APTITUDE_FILE = "aptitude.json"


def _global_projection(table: pd.DataFrame, q: dict[str, list[float]], sd: list[float]) -> dict[str, Any]:
    """全馬共通の 2 次元座標（5 方面が全部そろった馬の主成分）。座標は table と同じ並びで返す（欠けた馬は NaN）。"""
    Z = np.column_stack([_to_normal(table[col].to_numpy(dtype=float), np.array(q[key])) for key, col, *_ in AXES])
    Zw = Z / np.array(sd) * SIGNS * WEIGHTS
    full = ~np.isnan(Zw).any(axis=1)
    mu = Zw[full].mean(0)
    Zc = Zw[full] - mu
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    Y, A = _orient(Zc @ Vt[:2].T, Vt[:2].T)
    recon = Y @ A.T
    var = Zc.var(0)
    coords = np.full((len(table), 2), np.nan)
    coords[full] = Y
    return {
        "mu": mu.tolist(), "A": A.tolist(),
        "r2": [None if v < 1e-12 else float(1 - (Zc[:, j] - recon[:, j]).var() / v) for j, v in enumerate(var)],
        "explained": float((S[:2] ** 2).sum() / (S ** 2).sum()),
        "_coords": coords,
    }


def _build_reference(metrics: pd.DataFrame, sires: pd.DataFrame) -> dict[str, Any]:
    from src.research.race import running_map_aptitude as apt

    table = rp.profile_table(metrics, sires)
    q = {key: np.quantile(table[col].dropna().to_numpy(dtype=float), np.linspace(0, 1, 1001)).tolist() for key, col, *_ in AXES}
    sample = table.sample(min(len(table), 50_000), random_state=0)
    sd = [float(np.nanstd(_to_normal(sample[col].to_numpy(dtype=float), np.array(q[key])))) for key, col, *_ in AXES]
    proj = _global_projection(table, q, sd)
    coords = proj.pop("_coords")
    extent = apt.extent_from(coords[~np.isnan(coords).any(axis=1)])

    keys = ["race_id", "horse_id", "date"]
    tab = table[keys].assign(x=coords[:, 0], y2=coords[:, 1]).merge(
        metrics[keys + ["venue", "surface", "distance", "finish_position"]].drop_duplicates(keys), on=keys, how="left")
    aptitude = apt.build(apt.prepare(tab.dropna(subset=["venue", "surface", "distance"])), extent)
    return {
        "version": REFERENCE_VERSION, "metrics_version": rp.METRICS_VERSION,
        "built_from": str(metrics["date"].max().date()), "quantiles": q, "sd": sd, "extent": extent, **proj,
        "_aptitude": aptitude,
    }


def _load_reference(metrics: pd.DataFrame, sires: pd.DataFrame) -> dict[str, Any]:
    import json

    meta_p, apt_p = REFERENCE_DIR / "reference.json", REFERENCE_DIR / _APTITUDE_FILE
    try:
        ref = json.loads(meta_p.read_text(encoding="utf-8"))
        if ref.get("version") == REFERENCE_VERSION and ref.get("metrics_version") == rp.METRICS_VERSION:
            ref["_aptitude"] = json.loads(apt_p.read_text(encoding="utf-8"))
            return ref
    except (OSError, ValueError):
        pass
    ref = _build_reference(metrics, sires)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    apt_p.write_text(json.dumps(ref["_aptitude"], ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    meta_p.write_text(json.dumps({k: v for k, v in ref.items() if not k.startswith("_")}, ensure_ascii=False), encoding="utf-8")
    return ref


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    """保存済みの 1 走ごとの値・父の対応・基準（無ければ最初の 1 回だけ作って保存）。"""
    metrics = rp.load_metrics()
    sires = rp._sire_map()
    return {"metrics": metrics, "sires": sires, "ref": _load_reference(metrics, sires)}


@lru_cache(maxsize=1)
def _quantiles() -> dict[str, np.ndarray]:
    """各方面の全馬分布の分位点（1001 点）と、変換後の標準偏差。"""
    ref = _data()["ref"]
    out: dict[str, Any] = {k: np.array(v) for k, v in ref["quantiles"].items()}
    out["_sd"] = np.array(ref["sd"])
    return out


def clear_cache() -> None:
    for f in (_data, _quantiles):
        f.cache_clear()


def aptitude_for(conditions: dict[str, Any]) -> Optional[dict[str, Any]]:
    """このレースのコース（競馬場・芝ダ・距離）の好走しやすさの地図。コースの出走が少なければ None。"""
    from src.research.race import running_map_aptitude as apt

    v, s, d = conditions.get("venue"), conditions.get("surface"), conditions.get("distance")
    if not (v and s and d):
        return None
    hit = _data()["ref"]["_aptitude"]["courses"].get(apt.course_key(v, s, d))
    return None if hit is None else {"course": f"{v} {s} {int(d)}m", **hit}


# ── 変換 ──────────────────────────────────────────────────────────────────

def _norm_ppf(p: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.ppf(p)


def _to_pct(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """全馬の中での順位（0〜1）。同じ値が続く区間は中央をとる（sklearn の QuantileTransformer と同じ）。"""
    grid = np.linspace(0, 1, len(q))
    up = np.interp(x, q, grid)
    down = 1 - np.interp(-x, -q[::-1], grid)
    p = 0.5 * (up + down)
    return np.where(np.isnan(x), np.nan, p)


def _to_normal(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    return _norm_ppf(np.clip(_to_pct(x, q), 1e-3, 1 - 1e-3))


# ── 入力の整形 ────────────────────────────────────────────────────────────

def _norm_date(s: Any) -> Optional[str]:
    d = re.sub(r"\D", "", str(s or ""))
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) >= 8 else None


def _norm_surface(s: Any) -> Optional[str]:
    s = str(s or "")
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ"):
        return "ダート"
    return None


def _norm_condition(s: Any) -> Optional[str]:
    """「稍」「不」のような略記も正式名にそろえる。"""
    s = str(s or "")
    for head, name in (("不", "不良"), ("稍", "稍重"), ("重", "重"), ("良", "良")):
        if s.startswith(head):
            return name
    return None


def race_conditions(card: dict, race_id: str) -> dict[str, Any]:
    try:
        distance = int(card.get("distance") or 0) or None
    except (TypeError, ValueError):
        distance = None
    return {
        "date": _norm_date(card.get("date")),
        "venue": card.get("venue") or VENUE_JA.get(str(race_id)[4:6]),
        "surface": _norm_surface(card.get("surface")),
        "distance": distance,
        "track_condition": _norm_condition(card.get("track_condition")),
    }


# ── 2 次元化とグループ分け ────────────────────────────────────────────────

def _orient(Y: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """末脚（TS+GC）の矢印が真上、先行力の矢印が左を向くように回転・反転する。距離は変わらない。"""
    up = V[0] + V[1]
    if np.linalg.norm(up) > 1e-9:
        ang = math.atan2(up[1], up[0])
        rot = math.pi / 2 - ang
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        Y, V = Y @ R.T, V @ R.T
    if V[3][0] > 0:
        Y, V = Y * [-1, 1], V * [-1, 1]
    return Y, V


def project(Z: np.ndarray) -> dict[str, Any]:
    """出走馬の 5 次元（重み付き）を全馬共通の 2 次元座標へ。座標・各方面の矢印と再現度・全体の再現度を返す。"""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    ref = _data()["ref"]
    A = np.array(ref["A"])
    Y = (Z - np.array(ref["mu"])) @ A
    rho = None
    if len(Z) >= 4 and not np.allclose(Z, Z[0]):
        r = spearmanr(pdist(Z), pdist(Y))[0]
        rho = None if np.isnan(r) else float(r)
    return {"coords": Y, "loadings": A, "r2": ref["r2"], "explained": ref["explained"], "distance_spearman": rho}


def group(Z: np.ndarray, max_k: int = 5) -> np.ndarray:
    """5 次元で Ward 法。グループ数はシルエット係数が最大のもの（2〜max_k）。"""
    n = len(Z)
    if n < 4 or np.allclose(Z, Z[0]):
        return np.zeros(n, dtype=int)
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score

    L = linkage(Z, method="ward")
    best, best_s = np.zeros(n, dtype=int), -1.0
    for k in range(2, min(max_k, n - 1) + 1):
        lab = fcluster(L, k, criterion="maxclust") - 1
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(Z, lab)
        if s > best_s:
            best, best_s = lab, s
    return best


def describe_group(Zg: np.ndarray, Zall: np.ndarray, threshold: float = 0.3) -> list[dict[str, Any]]:
    """グループの平均が出走馬全体の平均とどれだけ違うか（全馬の標準偏差単位）。差の大きい順。"""
    diff = Zg.mean(0) - Zall.mean(0)
    traits = []
    for name, idx, hi, lo in _TRAIT_AXES:
        d = float(diff[idx].mean())
        if abs(d) >= threshold:
            traits.append({"axis": name, "diff": round(d, 2), "text": hi if d > 0 else lo})
    return sorted(traits, key=lambda t: -abs(t["diff"]))


# ── 本体 ──────────────────────────────────────────────────────────────────

def build_running_map(entries: list[dict], race_id: str, conditions: dict[str, Any]) -> dict[str, Any]:
    """出走馬一覧とレース条件から走法マップを作る。"""
    qs = _quantiles()
    as_of = conditions.get("date") or pd.Timestamp.now(tz="Asia/Tokyo").date().isoformat()
    d = _data()

    ids = [str(e.get("horse_id") or "") for e in entries]
    prof = rp.profiles_as_of(d["metrics"], [i for i in ids if i], as_of, d["sires"]).set_index("horse_id") if any(ids) else pd.DataFrame()
    picked, unmapped = [], []
    for e, hid in zip(entries, ids):
        base = {"horse_number": e.get("horse_number"), "bracket_number": e.get("bracket_number"),
                "horse_name": e.get("horse_name"), "horse_id": hid or None}
        row = prof.loc[hid] if hid and hid in prof.index else None
        if row is None or (row["n_runs"] == 0 and row["sire_id"] is None):
            unmapped.append({**base, "reason": "過去の出走も父のデータもありません"})
            continue
        picked.append(({**base, "n_runs": int(row["n_runs"]), "estimated": bool(row["n_runs"] == 0),
                        "own_weight": round(float(np.mean([row[f"own_{col}"] for _, col, *_ in AXES])), 3)}, row))

    horses, Zs = [], []
    signs = SIGNS
    for base, row in picked:
        raw = np.array([row.get(col) for _, col, *_ in AXES], dtype=float)
        pct = np.array([_to_pct(np.array([raw[j]]), qs[k])[0] for j, k in enumerate(AXIS_KEYS)])
        z = np.array([_to_normal(np.array([raw[j]]), qs[k])[0] for j, k in enumerate(AXIS_KEYS)]) / qs["_sd"]
        z = z * signs
        missing = [AXIS_KEYS[j] for j in range(len(AXES)) if np.isnan(raw[j])]
        if len(missing) == len(AXES):
            unmapped.append({**base, "reason": "過去の走法データがありません"})
            continue
        if len(missing) > MAX_IMPUTED:
            unmapped.append({**base, "reason": "走法データの欠けが多い（" + "・".join(a[3] for a in AXES if a[0] in missing) + "）"})
            continue
        z = np.nan_to_num(z)  # 欠けた方面は全馬の中央値
        horses.append({
            **base,
            "values": {k: (None if np.isnan(raw[j]) else round(float(raw[j]), 4)) for j, k in enumerate(AXIS_KEYS)},
            # 全馬の中で「高い」側から見た位置（0〜100。100 が最も高い側）
            "percentiles": {k: (None if np.isnan(pct[j]) else round(float(pct[j] if signs[j] > 0 else 1 - pct[j]) * 100, 1))
                            for j, k in enumerate(AXIS_KEYS)},
            "imputed": missing,
            # 自分の走の重みが 0（父や全体の平均だけで決まった）方面
            "from_prior": [k for k, col, *_ in AXES if row[f"own_{col}"] == 0],
            "last_run_date": row["last_run_date"],
        })
        Zs.append(z)

    axes_out = [{"key": k, "label": lbl, "high": hi, "low": lo, "weight": round(float(WEIGHTS[j]), 3)}
                for j, (k, _, _, lbl, hi, lo) in enumerate(AXES)]
    out: dict[str, Any] = {
        "race_id": race_id,
        "conditions": conditions,
        "axes": axes_out,
        "horses": horses,
        "groups": [],
        "unmapped": unmapped,
        "fidelity": None,
        "data_until": str(d["metrics"]["date"].max().date()),
        "method": "各馬の直近 10 走（新しい走ほど重く、走が少ないほど父の産駒の平均へ寄せる）から作った 5 方面を、"
                  "全馬の分布で順位→正規化（TS・GC は重み 1/√2）し、2020 年以降の全馬で作った共通の主成分で 2 次元に置く"
                  "（どのレースでも同じ地図）。グループは 5 次元の Ward 法",
        "extent": _data()["ref"]["extent"],
        "aptitude": aptitude_for(conditions),
    }
    if len(horses) < MIN_HORSES:
        out["status"] = "insufficient"
        return out

    Z = np.array(Zs) * WEIGHTS
    pr = project(Z)
    for h, (x, y) in zip(horses, pr["coords"]):
        h["x"], h["y"] = round(float(x), 4), round(float(y), 4)
    for a, (lx, ly), r2 in zip(axes_out, pr["loadings"] * WEIGHTS[:, None], pr["r2"]):
        a["arrow"] = [round(float(lx), 4), round(float(ly), 4)]
        a["r2"] = None if r2 is None else round(r2, 3)
    out["fidelity"] = {
        "explained": None if pr["explained"] is None else round(pr["explained"], 3),
        "distance_spearman": None if pr["distance_spearman"] is None else round(pr["distance_spearman"], 3),
    }

    labels = group(Z)
    Zu = np.array(Zs)
    order = sorted(set(labels), key=lambda g: (-int((labels == g).sum()), g))
    for gi, g in enumerate(order):
        members = np.where(labels == g)[0]
        gid = chr(ord("A") + gi)
        for m in members:
            horses[m]["group"] = gid
        traits = describe_group(Zu[members], Zu)
        out["groups"].append({
            "id": gid,
            "label": "・".join(t["text"] for t in traits[:2]) or "平均的",
            "traits": traits,
            "horse_numbers": [horses[m]["horse_number"] for m in members],
        })
    out["status"] = "ok"
    return out


def load_entries(storage: Any, race_id: str) -> tuple[list[dict], dict]:
    """出馬表を優先し、無ければ結果から出走馬を取る。"""
    for cat in ("race_shutuba", "race_result"):
        card = storage.load(cat, race_id)
        if card and card.get("entries"):
            return [e for e in card["entries"] if e.get("horse_id")], card
    return [], {}


def running_map_for_race(storage: Any, race_id: str) -> tuple[dict[str, Any], int]:
    entries, card = load_entries(storage, race_id)
    if not entries:
        return {"error": "出走馬が見つかりません", "race_id": race_id}, 404
    try:
        return build_running_map(entries, race_id, race_conditions(card, race_id)), 200
    except FileNotFoundError as exc:
        return {"error": f"走法データがありません: {exc.filename or exc}"}, 503
