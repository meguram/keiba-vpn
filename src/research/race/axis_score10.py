"""走法軸の「10 点満点」への換算（クラス基準）。

定義（ユーザー 2026-11-11）: 5=2勝クラスで安定して力を出せる水準、4=1勝、3=新馬・未勝利を勝てる、8〜9=G1/G2 で水準を保てる、
10=歴代最強級（G1 の最上位）。基準は「各クラスで好走（3着以内）した馬」の水準。検討の経緯は docs/ideas/running_axes_scale_1to10.md。

- **TS（水準軸）**: 馬ごとの値は「直近 10 走の TS の上位 2 走の平均」（ユーザー定義の「その馬の最高速度」。平均だと上位クラスが分離しない）。
  クラスごとの好走走の中央値をアンカーに、単調な折れ線で 1〜10 に写す。ただし実データでは 3勝より上（OP・G3・G2・G1）の TS 中央値が
  3勝とほぼ同じ（63〜65）で、TS の絶対値ではクラスが分かれない。アンカーは分離できる 未勝利〜3勝 だけにし、
  上側は 3勝の水準から「G1 好走走の上位 5% = 10」まで線形に補間する（8〜9 は TS 単独では決まらない）。
- **それ以外（タイプ軸）**: クラスの階段が無いので、点 = 5 + k·向き·z。z は 2勝の好走走の平均・SD で測る。
  k は TS のアンカーから決める（「3 → 5 の 2 点分」が TS の何 SD に当たるか）。実データでは 0.95 で、下限 1.0 に切り上げている
  （TS の 3→5 は約 2.1 SD 分。他の軸はそれより少し狭い幅で並ぶ）。
- 基準は「その走の直後」の値（次走の直前のプロファイル＝今の採点と同じ種類の量。その走の好走が入る）で作り、採点は「対象レースの直前」の値。
  この非対称は意図したもので、「このクラスで好走した馬はこの水準を持っている」を測る（これから好走する馬の点は、その好走を入れる前の値なので控えめに出る）。
- 値は馬ごとの as-of プロファイル（その走の直前時点）。未来の走を含まない。
- 表示は 0.5 点刻み・1〜10。値が無ければ None（不明）。0 や 5 は入れない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# クラスの目盛り。ユーザー定義は 3・4・5・8〜9・10 のみで、3勝・OP・G3 の間は補間（2026-11-11 の私の選択）
CLASS_LEVEL: dict[str, float] = {
    "新馬": 3.0, "未勝利": 3.0, "1勝": 4.0, "2勝": 5.0, "3勝": 5.5,
    "OP": 6.0, "L": 6.0, "G3": 7.0, "G2": 8.0, "G1": 9.0,
}
BASE_CLASS = "2勝"          # タイプ軸の基準（5 点）
TOP_CLASS = "G1"            # 10 点を決めるクラス
MAX_ANCHOR_LEVEL = 5.5      # TS のアンカーにするクラスの上限（これより上は TS の中央値で分離できない）
PEAK_LOOKBACK = 10          # TS のピーク: 直近この走数のうち
PEAK_TOP = 2                # 上位この走数の平均
PEAK_PRIOR = 50.0           # 走数が足りないときに足す仮の値（偏差値の中心）
TOP_QUANTILE = 0.95         # 10 点 = TOP_CLASS 好走走の TS 上位 5%
MIN_N = 30                  # アンカーにするクラスの最少走数
DEFAULT_K = 1.5             # TS から決められないときの 1 SD あたりの点（検討時の実測は約 1.5）
K_RANGE = (1.0, 2.5)
TS_KEY = "ts"
FLAT_DIR = Path(__file__).resolve().parents[3] / "data/page_reference/tables"


def load_grades(years: Iterable[int], base: Path = FLAT_DIR) -> pd.Series:
    """race_id → grade（新馬・未勝利・1勝・2勝・3勝・OP・L・G3・G2・G1）。取れない年は飛ばす。"""
    parts = []
    for y in years:
        p = base / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "grade"]).drop_duplicates("race_id"))
    if not parts:
        return pd.Series(dtype=object)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("race_id")
    return df.set_index(df["race_id"].astype(str))["grade"]


def _peak(values: np.ndarray) -> float:
    top = np.sort(values)[::-1][:PEAK_TOP]
    return float((top.sum() + (PEAK_TOP - len(top)) * PEAK_PRIOR) / PEAK_TOP)


def peak_ts_runs(metrics: pd.DataFrame, col: str = "ax_ts") -> pd.DataFrame:
    """各走の「直後」のピーク TS（その走を含む直近 PEAK_LOOKBACK 走の上位 PEAK_TOP 走の平均）。基準を作るためのもの。
    返す列: race_id, horse_id, date, peak_ts。TS の無い走は飛ばす。"""
    m = metrics[["race_id", "horse_id", "date", col]].dropna(subset=[col]).sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    v = m[col].to_numpy(float)
    out = np.empty(len(m))
    for _h, idx in m.groupby("horse_id").indices.items():
        for j in range(len(idx)):
            out[idx[j]] = _peak(v[idx[max(0, j - PEAK_LOOKBACK + 1): j + 1]])
    return m[["race_id", "horse_id", "date"]].assign(peak_ts=out)


def peak_ts_as_of(metrics: pd.DataFrame, horse_ids: Iterable[str], as_of: Any, col: str = "ax_ts") -> pd.Series:
    """as_of より前の走だけから出した、馬ごとのピーク TS（horse_id 索引）。走が 1 つも無い馬は NaN（不明）。"""
    ids = list(horse_ids)
    m = metrics[metrics["horse_id"].isin(ids) & (metrics["date"] < pd.Timestamp(as_of))].dropna(subset=[col]).sort_values(["horse_id", "date", "race_id"])
    got = {h: _peak(g[col].to_numpy(float)[-PEAK_LOOKBACK:]) for h, g in m.groupby("horse_id")}
    return pd.Series({h: got.get(h, np.nan) for h in ids}, dtype=float)


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """単調増加への等値回帰（重み付き。隣り合う逆転を平均で均す）。"""
    blocks = [[float(v), float(wt), 1] for v, wt in zip(y, w)]
    out: list[list[float]] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, n2 = out.pop()
            v1, w1, n1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    res: list[float] = []
    for v, _w, n in out:
        res += [v] * int(n)
    return np.array(res)


def _ts_knots(ts: pd.Series, grade: pd.Series) -> Optional[dict[str, Any]]:
    """TS のアンカー（クラスの好走走の中央値）を、目盛りの低い順に単調にそろえた折れ線の節点（[TS, 点]）にする。"""
    rows = []
    for g, lv in CLASS_LEVEL.items():
        x = ts[grade == g].dropna()
        if lv <= MAX_ANCHOR_LEVEL and len(x) >= MIN_N:
            rows.append((lv, g, float(x.median()), len(x)))
    if len(rows) < 3:
        return None
    rows.sort(key=lambda r: (r[0], r[2]))
    levels = np.array([r[0] for r in rows])
    med = _pav(np.array([r[2] for r in rows]), np.array([float(r[3]) for r in rows]))
    # 同じ目盛りのクラス（新馬と未勝利、OP と L）は 1 つにまとめる（件数で重み付け）
    pts: dict[float, list[float]] = {}
    for (lv, _g, _m, n), m in zip(rows, med):
        pts.setdefault(lv, []).append((m, n))
    knots = [[sum(m * n for m, n in v) / sum(n for _m, n in v), lv] for lv, v in sorted(pts.items())]
    # 等値回帰で同じ値になった隣り合う目盛りは、目盛りの平均で 1 節点にする（x が厳密に増えないと逆引きできない）
    merged: list[list[float]] = []
    for x, lv in knots:
        if merged and x <= merged[-1][0] + 1e-9:
            merged[-1][1] = (merged[-1][1] + lv) / 2
        else:
            merged.append([x, lv])
    top = ts[grade == TOP_CLASS].dropna()
    if len(top) >= MIN_N:
        q = float(top.quantile(TOP_QUANTILE))
        if q > merged[-1][0] and merged[-1][1] < 10:
            merged.append([q, 10.0])
    if len(merged) < 2:
        return None
    return {"knots": [[round(x, 4), round(y, 4)] for x, y in merged],
            "medians": {r[1]: round(r[2], 3) for r in rows}, "n": {r[1]: r[3] for r in rows}}


def _ts_points(x: float, knots: list[list[float]]) -> float:
    xs, ys = np.array([k[0] for k in knots]), np.array([k[1] for k in knots])
    if x <= xs[0]:   # 下側は最初の区間の傾きで外挿
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + s * (x - xs[0]))
    if x >= xs[-1]:  # 上側は 10 で止める
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


def _round_pt(p: float) -> float:
    return float(np.clip(round(p * 2) / 2, 1.0, 10.0))


def build_score10(table: pd.DataFrame, finish: pd.Series, grade: pd.Series, cols: dict[str, str], signs: dict[str, int]) -> Optional[dict[str, Any]]:
    """基準を作る。table = 走ごとの as-of プロファイル（race_id・horse_id と各軸の列）。finish・grade は table と同じ並びの着順・グレード。
    cols = {軸キー: 列名}、signs = {軸キー: 向き}。作れない（格付けが無い・標本が足りない）ときは None。"""
    good = (finish.to_numpy() <= 3) & grade.notna().to_numpy()
    t, g = table[good], grade[good].reset_index(drop=True)
    t = t.reset_index(drop=True)
    if TS_KEY not in cols or len(t) == 0:
        return None
    ts = t[cols[TS_KEY]]
    ts_ref = _ts_knots(ts, g)
    if ts_ref is None:
        return None
    knots = ts_ref["knots"]
    base_mask = g == BASE_CLASS
    k = DEFAULT_K
    kv = {y: x for x, y in knots}
    if 3.0 in kv and 5.0 in kv and base_mask.sum() >= MIN_N:
        sd_ts = float(ts[base_mask].std())
        if sd_ts > 0 and kv[5.0] > kv[3.0]:
            k = float(np.clip(2.0 / ((kv[5.0] - kv[3.0]) / sd_ts), *K_RANGE))
    axes: dict[str, Any] = {}
    for key, col in cols.items():
        if key == TS_KEY or col not in t.columns:
            continue
        x = t.loc[base_mask, col].dropna()
        if len(x) < MIN_N or float(x.std()) <= 0:
            continue
        axes[key] = {"mu": round(float(x.mean()), 5), "sd": round(float(x.std()), 5), "n": int(len(x)), "sign": int(signs.get(key, 1))}
    return {"version": 1, "k": round(k, 4), "ts": ts_ref, "axes": axes, "base_class": BASE_CLASS,
            "levels": CLASS_LEVEL, "top_quantile": TOP_QUANTILE}


def to_score10(key: str, value: Any, ref: Optional[dict[str, Any]]) -> Optional[float]:
    """軸 key の値を 1〜10（0.5 刻み）に。基準が無い・値が無い・その軸の基準が作れなかったときは None。"""
    if ref is None or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    if key == TS_KEY:
        return _round_pt(_ts_points(v, ref["ts"]["knots"]))
    a = ref["axes"].get(key)
    if a is None:
        return None
    return _round_pt(5.0 + ref["k"] * a["sign"] * (v - a["mu"]) / a["sd"])


def to_score10_deviation(value: Any, ref: Optional[dict[str, Any]], center: float = 50.0, scale: float = 10.0) -> Optional[float]:
    """50±10 の偏差値（道悪適性など、基準の分布を持たない補助の軸）を同じ幅で 1〜10 に。5 = 偏差値 50。"""
    if ref is None or value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return _round_pt(5.0 + ref["k"] * (v - center) / scale)


def score_series(key: str, values: pd.Series, ref: Optional[dict[str, Any]]) -> pd.Series:
    """to_score10 のベクトル版（pandas 用）。基準が無い・その軸の基準が作れなかったときは全て NaN。"""
    nan = pd.Series(np.nan, index=values.index)
    if ref is None:
        return nan
    v = pd.to_numeric(values, errors="coerce").astype(float)
    if key == TS_KEY:
        k = ref["ts"]["knots"]
        xs, ys = np.array([a[0] for a in k]), np.array([a[1] for a in k])
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        p = np.where(v <= xs[0], ys[0] + slope * (v - xs[0]), np.interp(v, xs, ys))   # 下側は外挿、上側は 10 で止まる
    else:
        a = ref["axes"].get(key)
        if a is None:
            return nan
        p = 5.0 + ref["k"] * a["sign"] * (v - a["mu"]) / a["sd"]
    out = np.clip(np.round(np.asarray(p, float) * 2) / 2, 1.0, 10.0)
    return pd.Series(np.where(v.notna(), out, np.nan), index=values.index)
