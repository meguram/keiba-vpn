"""
出走馬の走法マップ（めぐ分析）。

走法プロファイルの 8 方面（実測ラップ版: 末脚の速さ・末脚の持続力・出脚・ギアチェンジ・Hペース耐性・
先行力・Cr性能・まくり特性。定義は docs/ideas/running_map_axes_fixed.md）を 2 次元へ写し、
「この馬とあの馬は走り方が近い」を見えるようにする。さらに、コース（競馬場・芝ダ・距離）ごとに
「地図のどのゾーンが好走しやすいか」を色で重ねる（running_map_aptitude.py）。
地図の次元には入れない補助の軸として、急坂適性（中山・阪神のみ）と道悪の軸（芝=芝道悪適性、ダート=ダートタフ適正）を
馬ごとの詳細に付ける。判定できない馬は 0 や 50 ではなく None（画面では「不明」）。

方式:
  - 各方面は、全馬の分布で順位 → 標準正規へ変換してそろえる（外れ値に強い）。
  - 2 次元化は「基準を作った時点の全馬」で 1 回だけ作った共通の主成分（reference に保存）。どのレースでも同じ座標系なので、
    同じ位置＝同じ走り方で、コース適性の色塗りをそのまま重ねられる。
    出走馬の数や顔ぶれで地図の形が変わらない代わりに、そのレースの出走馬だけで作る主成分（古典的 MDS）より
    元の距離の再現度は少し落ちる（fidelity.distance_spearman で毎回示す）。
  - 向きは固定（末脚の矢印が真上、先行力の矢印が左）。
  - グループは 2 次元ではなく元の 8 次元で Ward 法の階層クラスタリング。数はシルエット係数が最大の 2〜5。

走法データは src/research/race/running_profile.py が作る（実測ラップは run_axes.py → run_axes.parquet）:
  - 各馬のプロファイルは、対象レースの日より前の走だけから作る（前走まで入る）。実測ラップのある走が少ない馬は
    父の産駒の平均へ寄せ、実測の走が 1 つも無い馬は父の産駒の平均そのもの（estimated=True。地図には載せるが、
    推定であることを画面で示す）。方面ごとの実測の走数は axis_runs で返す。
  - 全馬の分布（順位の基準）・共通の主成分・コース適性の表は、最初の 1 回だけ作って保存し、以後は読むだけ。
    実測ラップの取得年が増えたら REFERENCE_VERSION を上げて作り直す。
    コース適性の倍率と有意性は基準を作った期間の全レースで測った値（標本内）なので、期間内のレースに対しては結果を含んだ値になる。
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
# 前後半比（前半型のペース）は 2026-10-30 に外した: ばらつきの半分がレースの展開（そのレースのペース）で、レースの平均との差に
# 直すと Hペース耐性と相関 0.86（他の軸で 9 割説明できる）。1 走ごとの値（run_axes の split）は残している。
AXES: list[tuple[str, str, int, str, str, str]] = [
    ("ts", "ax_ts", 1, "末脚の速さ", "末脚が速い", "末脚が遅い"),
    ("sustain", "ax_sustain", 1, "末脚の持続力", "速い末脚を長く続ける", "末脚が続かない"),
    ("debut", "ax_debut", 1, "出脚", "出脚が速い", "出脚が遅い"),
    ("gear", "ax_gear", 1, "ギアチェンジ", "終盤に加速する", "終盤に加速しない"),
    ("endure", "ax_endure", 1, "Hペース耐性", "Hペースでも落ちない", "Hペースで止まる"),   # 追走効率（追走で脚を使わず終いに落ちない）の意味も持つ
    ("front", "ax_front", 1, "先行力", "前に行く", "後ろから行く"),
    ("cr", "ax_cr", 1, "Cr性能", "3〜4角で位置を上げる", "3〜4角で位置を下げる"),
    ("makuri", "ax_makuri", 1, "まくり特性", "道中〜3角で位置を上げる", "道中〜3角で位置を下げる"),
]
AXIS_KEYS = [a[0] for a in AXES]
WEIGHTS = np.ones(len(AXES))
SIGNS = np.array([a[2] for a in AXES])
# 地図の次元には入れない補助の軸（値は 50±10 の偏差値。判定できなければ None = 不明）
EXTRA_LABELS = {
    "slope": ("急坂適性", "坂で落ちない", "坂で落ちる"),
    "going_turf": ("芝道悪適性", "道悪の芝が得意", "道悪の芝が苦手"),
    "going_dirt": ("ダートタフ適正", "乾いたダートが得意", "乾いたダートが苦手"),
}
SLOPE_VENUES = ("中山", "阪神")
MAP_COLS = [a[1] for a in AXES] + ["ax_slope"]
# グループの説明に使う軸（1 軸 1 項目）
_TRAIT_AXES = [(a[3], [i], a[4], a[5]) for i, a in enumerate(AXES)]
_UP_IDX = [AXIS_KEYS.index("ts"), AXIS_KEYS.index("sustain")]   # 地図の「上」= 末脚
_FRONT_IDX = AXIS_KEYS.index("front")                          # 地図の「左」= 先行力

VENUE_JA = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}

MIN_HORSES = 3
MAX_IMPUTED = 3  # これより多くの方面が欠けた馬は地図に載せない（欠けた方面は全馬の中央値の位置に置く）
SIMILAR_K = 3    # 馬ごとに出す「8 方面で本当に近い馬」の数（地図の近さは 2 次元に落とした分だけずれるので、それを補う）
SIMILAR_MIN_COMMON = 6  # 距離を測るのに、2 頭とも実際に値がある方面がこれ以上必要（中央値で補った方面は距離に使わない）


# ── データ ────────────────────────────────────────────────────────────────
# 計算は最初の 1 回だけ。1 走ごとの値は running_profile.load_metrics が保存・追加計算し、
# 基準（全馬の分布の分位点・共通の主成分・コース適性の表）は REFERENCE_DIR に保存して、以後は読むだけ。
# 基準は作り直さない（データが増えても分布はほぼ変わらない）。作り方を変えたら REFERENCE_VERSION を上げる。

REFERENCE_DIR = rp.STORE_DIR / "reference"
REFERENCE_VERSION = "4"
_APTITUDE_FILE = "aptitude.json"


def _global_projection(table: pd.DataFrame, q: dict[str, list[float]], sd: list[float]) -> dict[str, Any]:
    """全馬共通の 2 次元座標（全方面がそろった馬の主成分）。座標は table と同じ並びで返す（欠けた馬は NaN）。"""
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

    table = rp.profile_table(metrics, sires, cols=[col for _, col, *_ in AXES])
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
    for f in (_data, _quantiles, _priors, _going_runs, _going_all):
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
    """末脚（速さ+持続力）の矢印が真上、先行力の矢印が左を向くように回転・反転する。距離は変わらない。"""
    up = V[_UP_IDX].sum(0)
    if np.linalg.norm(up) > 1e-9:
        ang = math.atan2(up[1], up[0])
        rot = math.pi / 2 - ang
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        Y, V = Y @ R.T, V @ R.T
    if V[_FRONT_IDX][0] > 0:
        Y, V = Y * [-1, 1], V * [-1, 1]
    return Y, V


def project(Z: np.ndarray) -> dict[str, Any]:
    """出走馬の全方面（重み付き）を全馬共通の 2 次元座標へ。座標・各方面の矢印と再現度・全体の再現度を返す。"""
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
    """全方面の次元で Ward 法。グループ数はシルエット係数が最大のもの（2〜max_k）。"""
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


def similar(Zraw: np.ndarray, k: int = SIMILAR_K, min_common: int = SIMILAR_MIN_COMMON) -> list[list[tuple[int, float]]]:
    """馬ごとに、8 方面の元の空間で近い馬（同じレースの中）を近い順に k 頭。距離は 2 頭とも値がある方面だけの二乗平均の平方根。

    地図（共通の主成分）は 8 方面の違いの半分ほどしか表さないので、地図で近くても実際は違う・地図で離れていても実際は近い
    ことがある（2026-10-30 の比較: 近い 3 頭のうち地図でも近いのは約 6 割）。画面では、選んだ馬のこの一覧を強調して補う。"""
    n = len(Zraw)
    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        cands = []
        for j in range(n):
            if j == i:
                continue
            both = ~np.isnan(Zraw[i]) & ~np.isnan(Zraw[j])
            if both.sum() < min_common:
                continue
            cands.append((j, float(np.sqrt(np.mean((Zraw[i, both] - Zraw[j, both]) ** 2)))))
        out.append(sorted(cands, key=lambda t: t[1])[:k])
    return out


def describe_group(Zg: np.ndarray, Zall: np.ndarray, threshold: float = 0.3) -> list[dict[str, Any]]:
    """グループの平均が出走馬全体の平均とどれだけ違うか（全馬の標準偏差単位）。差の大きい順。"""
    diff = Zg.mean(0) - Zall.mean(0)
    traits = []
    for name, idx, hi, lo in _TRAIT_AXES:
        d = float(diff[idx].mean())
        if abs(d) >= threshold:
            traits.append({"axis": name, "diff": round(d, 2), "text": hi if d > 0 else lo})
    return sorted(traits, key=lambda t: -abs(t["diff"]))


# ── 補助の軸（急坂・道悪） ─────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _priors(as_of: str) -> tuple[pd.DataFrame, pd.Series]:
    """対象日より前の父ごとの平均と全馬の平均。同じ日の複数レースで使い回す。"""
    d = _data()
    return rp.priors_as_of(d["metrics"], pd.Timestamp(as_of), d["sires"], MAP_COLS)


@lru_cache(maxsize=1)
def _going_runs() -> pd.DataFrame:
    from src.research.race import going_aptitude as ga

    runs = rp.load_runs(rp._years())
    return ga.attach_moisture(runs[["horse_id", "date", "venue", "surface", "finish_position", "field_size"]], ga.load_moisture())


@lru_cache(maxsize=4)
def _going_all(as_of: str) -> pd.DataFrame:
    """対象日時点の道悪の軸（走歴のある全馬）。馬ごとの引きは horse_id の索引。"""
    from src.research.race import going_aptitude as ga

    runs = _going_runs()
    past = runs[runs["date"] < pd.Timestamp(as_of)]
    return ga.going_scores(runs, list(past["horse_id"].unique()), as_of).set_index("horse_id")


def _extra_keys(conditions: dict[str, Any]) -> list[str]:
    keys = []
    if conditions.get("venue") in SLOPE_VENUES:
        keys.append("slope")
    surface = conditions.get("surface")
    if surface in ("芝", "ダート"):
        keys.append("going_turf" if surface == "芝" else "going_dirt")
    return keys


def _num(x: Any) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 1)


def _extras(hid: str, row: pd.Series, keys: list[str], going: Optional[pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k == "slope":
            out[k] = {"value": _num(row.get("ax_slope")), "runs": int(row.get("n_ax_slope", 0))}
        else:
            g = going.loc[hid] if going is not None and hid in going.index else None
            out[k] = ({"value": None, "n_power": 0, "n_normal": 0, "low": False} if g is None else
                      {"value": _num(g[k]), "n_power": int(g[f"n_power_{k}"]), "n_normal": int(g[f"n_normal_{k}"]), "low": bool(g[f"low_{k}"])})
    return out


# ── 本体 ──────────────────────────────────────────────────────────────────

def build_running_map(entries: list[dict], race_id: str, conditions: dict[str, Any]) -> dict[str, Any]:
    """出走馬一覧とレース条件から走法マップを作る。"""
    qs = _quantiles()
    as_of = str(conditions.get("date") or pd.Timestamp.now(tz="Asia/Tokyo").date().isoformat())
    d = _data()

    ids = [str(e.get("horse_id") or "") for e in entries]
    prof = (rp.profiles_as_of(d["metrics"], [i for i in ids if i], as_of, d["sires"], cols=MAP_COLS, priors=_priors(as_of)).set_index("horse_id")
            if any(ids) else pd.DataFrame())
    extra_keys = _extra_keys(conditions)
    going = _going_all(as_of) if any(k.startswith("going_") for k in extra_keys) else None
    picked, unmapped = [], []
    for e, hid in zip(entries, ids):
        base = {"horse_number": e.get("horse_number"), "bracket_number": e.get("bracket_number"),
                "horse_name": e.get("horse_name"), "horse_id": hid or None}
        row = prof.loc[hid] if hid and hid in prof.index else None
        axis_runs = None if row is None else {k: int(row[f"n_{col}"]) for k, col, *_ in AXES}
        n_lap = 0 if axis_runs is None else max(axis_runs.values())
        if row is None or (n_lap == 0 and row["sire_id"] is None):
            unmapped.append({**base, "reason": "実測ラップのある過去の出走も父のデータもありません"})
            continue
        picked.append(({**base, "n_runs": int(row["n_runs"]), "n_lap_runs": n_lap, "axis_runs": axis_runs, "estimated": n_lap == 0,
                        "own_weight": round(float(np.mean([row[f"own_{col}"] for _, col, *_ in AXES])), 3),
                        "extras": _extras(hid, row, extra_keys, going)}, row))

    horses, Zs, Zraw = [], [], []
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
        Zraw.append(z * WEIGHTS)
        z = np.nan_to_num(z)  # 欠けた方面は全馬の中央値
        horses.append({
            **base,
            "values": {k: (None if np.isnan(raw[j]) else round(float(raw[j]), 4)) for j, k in enumerate(AXIS_KEYS)},
            # 全馬の中で「高い」側から見た位置（0〜100。100 が最も高い側）
            "percentiles": {k: (None if np.isnan(pct[j]) else round(float(pct[j] if signs[j] > 0 else 1 - pct[j]) * 100, 1))
                            for j, k in enumerate(AXIS_KEYS)},
            "imputed": missing,
            # 自分の実測の走が 1 つも無い（父や全体の平均だけで決まった）方面
            "from_prior": [k for k, col, *_ in AXES if base["axis_runs"][k] == 0],
            "last_run_date": row["last_run_date"],
        })
        Zs.append(z)

    axes_out = [{"key": k, "label": lbl, "high": hi, "low": lo, "weight": round(float(WEIGHTS[j]), 3)}
                for j, (k, _, _, lbl, hi, lo) in enumerate(AXES)]
    out: dict[str, Any] = {
        "race_id": race_id,
        "conditions": conditions,
        "axes": axes_out,
        "extra_axes": [{"key": k, "label": EXTRA_LABELS[k][0], "high": EXTRA_LABELS[k][1], "low": EXTRA_LABELS[k][2]} for k in extra_keys],
        "horses": horses,
        "groups": [],
        "unmapped": unmapped,
        "fidelity": None,
        "data_until": str(d["metrics"]["date"].max().date()),
        "method": "各馬の直近 10 走（新しい走ほど重く、実測ラップのある走が少ないほど父の産駒の平均へ寄せる）から作った 8 方面を、"
                  "全馬の分布で順位→正規化し、基準を作った時点の全馬で作った共通の主成分で 2 次元に置く"
                  "（どのレースでも同じ地図）。グループは 8 次元の Ward 法",
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

    for h, near in zip(horses, similar(np.array(Zraw))):
        h["similar"] = [{"horse_id": horses[j]["horse_id"], "horse_number": horses[j]["horse_number"], "horse_name": horses[j]["horse_name"],
                         "distance": round(d, 3)} for j, d in near]

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
