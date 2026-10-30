"""
走法データ（走法マップの 6 方面のもと）を、出走記録から作る。

tp-25 のノートブック（TS・TS持続力・GC）と scripts/run_tp26.py（前後半比・先行度・押し上げ）に分かれていた計算を、
年を固定せずに data/page_reference/tables/<年>/race_result_flat.parquet から作れるようにまとめたもの。

計算は 1 回だけ: 1 走ごとの値は data/local/running_profile/run_metrics.parquet に保存し、2 回目からは読むだけにする
（load_metrics）。新しく結果が入ったレースのぶんだけ追加で計算する（1 走ごとの値はそのレースの記録だけで決まるので、
追加分だけ計算しても全体を計算し直したのと同じ）。計算式を変えたら METRICS_VERSION を上げる（保存分を作り直す）。

2 段に分ける:
  1. run_metrics(): 1 走ごとの値。その馬・そのレースの記録（と、同じレースの全頭）だけで決まるので、未来の情報は入らない。
  2. profiles_as_of(): 馬ごとのプロファイル。対象日より前の走だけを使う（対象日当日・以降の走は使わない）。

データが欠けるときの扱い:
  - 障害戦は対象外（レース名に「障害」「ジャンプ」「JS」「JG」を含む。外れ値になるため）。取消・除外・中止（着順が無い）も対象外。
  - レースのラップが無い平地戦: 同じコース（競馬場・芝ダ・距離）のラップの形の中央値で推定する（laps_imputed=True）。
    コースのラップが 1 つも無ければ、同じ芝ダ・同じハロン数で推定する。2020 年以降の平地ではほぼ起きない（古い年の取り込み向け）。
  - 前後半比のハロン数は、ラップが無ければ距離から決める（距離 ÷ 200 の切り上げ。ラップがあるレースではラップの数と一致する）。
  - 頭数は「発表の頭数・出走した頭数・通過順の最大」の大きい方を使う（元データの頭数の誤りで先行度が 1 を超えるのを防ぐ）。
  - プロファイルは、方面ごとに値のある直近 LOOKBACK 走（値の無い走は飛ばす）の、新しい走ほど重い加重平均を、
    父の産駒の平均（対象日より前の走だけ、MIN_SIRE_RUNS 走以上あるとき。無ければ全馬の平均）へ引き寄せたもの:
        (Σ decay^i × 値_i + c × 父の平均) / (Σ decay^i + c)
    走が少ない馬ほど父の平均に近く、新馬は父の平均そのもの。自分の走の重みの割合を own_<方面>（0〜1）で返す。

何走前まで・どれだけ新しい走を重くするか・どれだけ引き寄せるか（PROFILE_PARAMS）は、
「プロファイルが次の 1 走の値をどれだけ当てるか（R²）」で 2024 年に決め、2025 年で確かめた（2026-10-29）:
    方面            旧（直近 3 走の単純平均）  新        さかのぼる走数の影響
    TS              -0.013                   0.147    7〜10 走でほぼ頭打ち（3 走 0.149 → 10 走 0.155、2024 年）
    GC              -0.016                   0.144    同上
    前後半比         0.093                   0.249
    先行度           0.109                   0.252
    押し上げ        -0.238                   0.041
    TS持続力        -0.369                   0.007    （1 走ごとのぶれが大きく、ほぼ当てられない）
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FLAT_DIR = REPO / "data/page_reference/tables"
SIRE_MAP_PATH = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
STORE_DIR = REPO / "data/local/running_profile"
METRICS_PATH = STORE_DIR / "run_metrics.parquet"
METRICS_VERSION = "1"       # run_metrics の計算式を変えたら上げる（保存分を作り直す）

# 1 走ごとの値（プロファイルにする方面）
METRICS = ["ts_score", "ts_stamina_score", "gc_score", "ts_fastest_f", "gc_raw", "split_ratio", "front_ratio", "shift_ratio"]
LOOKBACK = 10         # さかのぼる走数（値のある走だけを数える）。10 走より増やしても当たりは 0.001 も変わらない
MIN_SIRE_RUNS = 30    # 父の産駒の平均を使うのに要る走数
# 方面ごとの (新しい走ほど重くする係数 decay, 平均へ引き寄せる強さ c, 引き寄せ先 "sire"=父の産駒 / "overall"=全馬)
PROFILE_PARAMS: dict[str, tuple[float, float, str]] = {
    "ts_score": (0.7, 2.0, "sire"),
    "gc_score": (0.7, 2.0, "sire"),
    "ts_fastest_f": (0.7, 2.0, "sire"),
    "gc_raw": (0.7, 2.0, "sire"),
    "split_ratio": (0.7, 2.0, "sire"),
    "front_ratio": (0.7, 1.0, "sire"),
    "shift_ratio": (0.8, 5.0, "sire"),
    "ts_stamina_score": (0.7, 5.0, "overall"),
}
_JUMP = r"障害|ジャンプ|JS\(|JG|J・G"

_COLS = ["race_id", "date", "venue", "surface", "distance", "track_condition", "race_name", "field_size",
         "finish_position", "horse_id", "passing_order", "last_3f", "time_sec", "lap_times"]


# ── 読み込み ──────────────────────────────────────────────────────────────

def load_runs(years: Iterable[int], base: Path = FLAT_DIR) -> pd.DataFrame:
    """平地（障害を除く）で着順のある出走。"""
    parts = []
    for y in years:
        p = base / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=_COLS))
    if not parts:
        return pd.DataFrame(columns=_COLS)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["surface"].isin(["芝", "ダート"]) & ~df["race_name"].fillna("").str.contains(_JUMP)]
    for c in ("last_3f", "time_sec", "field_size", "finish_position", "distance"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["finish_position"] > 0].copy()
    # 上がり 3F・走破タイムが 0 以下の記録は欠損として扱う（tp-25 と同じ。0 だと最速の 1F が 0 秒になり TS が壊れる）
    df.loc[~(df["last_3f"] > 0), "last_3f"] = np.nan
    df.loc[~(df["time_sec"] > 0), "time_sec"] = np.nan
    df["horse_id"] = df["horse_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def _laps(s: object) -> list[float]:
    if isinstance(s, str):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(s, (list, tuple, np.ndarray)):
        v = list(s)
    else:
        return []
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return []


# ── 1 走ごとの値 ──────────────────────────────────────────────────────────

def _race_dev(df: pd.DataFrame, col: str) -> pd.Series:
    """レース内の偏差値（tp-25 と同じ: 標準偏差が 0・1 頭なら 1 で割る）。"""
    g = df.groupby("race_id")[col]
    mu = g.transform("mean")
    sd = g.transform("std").fillna(1.0).replace(0.0, 1.0)
    return (df[col] - mu) / sd * 10 + 50


def _lap_shape(df: pd.DataFrame) -> pd.DataFrame:
    """ラップのあるレースから、コースごとのラップの形（ラスト 1〜5F ÷ レースの上がり 3F）と、
    レースの上がり 3F ÷ 出走馬の上がり 3F の中央値 の中央値。ラップの無いレースの推定に使う。"""
    r = df[df["has_laps"]].groupby("race_id").agg(
        venue=("venue", "first"), surface=("surface", "first"), distance=("distance", "first"), n_laps=("n_laps", "first"),
        L1=("lap_L1", "first"), L2=("lap_L2", "first"), L3=("lap_L3", "first"), L4=("race_L4", "first"), L5=("race_L5", "first"),
        race_last3f=("race_last3f", "first"), field_l3f=("last_3f", "median"))
    for k in ("L1", "L2", "L3", "L4", "L5"):
        r[f"s_{k}"] = r[k] / r["race_last3f"]
    r["s_race"] = r["race_last3f"] / r["field_l3f"]
    return r


def _impute_laps(df: pd.DataFrame) -> pd.DataFrame:
    miss = ~df["has_laps"]
    if not miss.any():
        return df
    shape = _lap_shape(df)
    cols = ["s_L1", "s_L2", "s_L3", "s_L4", "s_L5", "s_race"]
    by_course = shape.groupby(["venue", "surface", "distance"])[cols].median()
    by_len = shape.groupby(["surface", "n_laps"])[cols].median()
    m = df.loc[miss, ["venue", "surface", "distance", "n_laps", "race_id", "last_3f"]].copy()
    est = m.join(by_course, on=["venue", "surface", "distance"])
    fb = m.join(by_len, on=["surface", "n_laps"])
    est[cols] = est[cols].fillna(fb[cols])
    field_l3f = df.loc[miss].groupby("race_id")["last_3f"].transform("median")
    race_last3f = est["s_race"] * field_l3f
    df.loc[miss, "race_last3f"] = race_last3f
    df.loc[miss, "lap_L1"] = est["s_L1"] * race_last3f
    df.loc[miss, "lap_L2"] = est["s_L2"] * race_last3f
    df.loc[miss, "lap_L3"] = est["s_L3"] * race_last3f
    df.loc[miss, "race_L4"] = est["s_L4"] * race_last3f
    df.loc[miss, "race_L5"] = est["s_L5"] * race_last3f
    df.loc[miss, "laps_imputed"] = est["s_L1"].notna()
    return df


def _passing(po: object, fs: float) -> tuple[float, float]:
    if not isinstance(po, str) or not po or not fs or fs <= 1:
        return (math.nan, math.nan)
    try:
        c = [int(x) for x in po.split("-")]
    except ValueError:
        return (math.nan, math.nan)
    if not c:
        return (math.nan, math.nan)
    return (c[0] / fs, (c[-1] - c[0]) / fs)


def run_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """1 走ごとの走法の値。列: race_id, horse_id, date, venue, surface, distance, track_condition, laps_imputed, METRICS。"""
    df = runs.copy()
    laps = [_laps(s) for s in df["lap_times"]]
    df["has_laps"] = [len(x) >= 5 for x in laps]
    df["n_laps"] = np.where(df["has_laps"], [len(x) for x in laps], np.ceil(df["distance"] / 200))

    def lap(i: int) -> list[float]:
        return [x[-i] if len(x) >= 5 else math.nan for x in laps]
    df["lap_L1"], df["lap_L2"], df["lap_L3"], df["race_L4"], df["race_L5"] = lap(1), lap(2), lap(3), lap(4), lap(5)
    df["race_last3f"] = df["lap_L1"] + df["lap_L2"] + df["lap_L3"]
    df["laps_imputed"] = False
    df = _impute_laps(df)

    # 馬ごとの推定ラップ（レースのラップを、その馬の上がり 3F に合わせて伸縮）
    scale = df["last_3f"] / df["race_last3f"].replace(0, np.nan)
    L1, L2, L3 = df["lap_L1"] * scale, df["lap_L2"] * scale, df["lap_L3"] * scale
    L4, L5 = df["race_L4"], df["race_L5"]
    est = np.column_stack([L3, L2, L1])

    # TS: ラスト 3F で最も速い 1F（tp-25）
    df["ts_fastest_f"] = np.nanmin(np.where(np.isnan(est), np.inf, est), axis=1)
    df.loc[~np.isfinite(df["ts_fastest_f"]), "ts_fastest_f"] = np.nan
    df["ts_idx"] = 12.0 / df["ts_fastest_f"] * 100
    df["ts_score"] = _race_dev(df, "ts_idx")

    # TS持続力: ピーク付近（+5%以内）を何 F 続けたか ＋ 連続加速・L4 の加点
    thr = df["ts_fastest_f"] * 1.05
    sustained = (L4 <= thr).astype(float) + (L3 <= thr).astype(float) + (L2 <= thr).astype(float) + (L1 <= thr).astype(float)
    bonus = 0.5 * ((L3 > L2) & (L2 > L1)).astype(float) + 0.5 * (L4 <= thr).astype(float)
    df["ts_stamina_raw"] = (sustained + bonus).clip(0, 5).where(L4.notna())
    df["ts_stamina_score"] = _race_dev(df, "ts_stamina_raw")

    # GC: L5 からピークまでの加速の大きさ × ピークの早さ
    peak_pos = np.where(np.isnan(est).all(axis=1), 0, np.nanargmin(np.where(np.isnan(est), np.inf, est), axis=1)) + 1
    accel = (L5 - df["ts_fastest_f"]) / L5.replace(0, np.nan)
    df["gc_raw"] = accel * (4 - peak_pos) / 3.0
    df["gc_score"] = _race_dev(df, "gc_raw")

    # 前後半比: 後半 3F の 1F あたり ÷ 前半の 1F あたり（大きい = 前半が速い）
    front_n = (df["n_laps"] - 3).clip(lower=1)
    pace_front = (df["time_sec"] - df["last_3f"]) / front_n
    df["split_ratio"] = (df["last_3f"] / 3) / pace_front.replace(0, np.nan)

    # 先行度・押し上げ: 最初と最後のコーナーの通過順 ÷ 頭数
    runners = df.groupby("race_id")["horse_id"].transform("size")
    maxpos = df["passing_order"].map(lambda s: max((int(x) for x in s.split("-") if x.isdigit()), default=0) if isinstance(s, str) else 0)
    race_maxpos = maxpos.groupby(df["race_id"]).transform("max")
    fs = np.fmax(np.fmax(df["field_size"].fillna(0), runners), race_maxpos)
    fr = [_passing(po, f) for po, f in zip(df["passing_order"], fs)]
    df["front_ratio"] = [a for a, _ in fr]
    df["shift_ratio"] = [b for _, b in fr]

    keep = ["race_id", "horse_id", "date", "venue", "surface", "distance", "track_condition", "finish_position", "n_laps", "laps_imputed"] + METRICS
    return df[keep].sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


# ── 保存（計算は 1 回だけ） ──────────────────────────────────────────────

def _years(base: Path = FLAT_DIR) -> list[int]:
    return sorted(int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) >= 2020)


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _source_sig(base: Path) -> dict[str, list[int]]:
    out = {}
    for y in _years(base):
        p = base / str(y) / "race_result_flat.parquet"
        if p.exists():
            st = p.stat()
            out[str(y)] = [st.st_mtime_ns, st.st_size]
    return out


def load_metrics(path: Path = METRICS_PATH, base: Path = FLAT_DIR) -> pd.DataFrame:
    """保存した 1 走ごとの値を読む。出走記録のファイルが前回から変わっていなければ、保存分を読むだけ。
    変わっていれば、保存に無いレースだけ計算して足す。保存が無い・版が違うときだけ全部計算する。"""
    import json

    stored = None
    mf = _manifest_path(path)
    sig = _source_sig(base)
    if path.exists() and mf.exists():
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
            if meta.get("version") == METRICS_VERSION:
                stored = pd.read_parquet(path)
                if meta.get("source") == sig:
                    return stored
        except (OSError, ValueError):
            stored = None
    runs = load_runs(_years(base), base)
    if stored is not None:
        new = runs[~runs["race_id"].isin(set(stored["race_id"]))]
        metrics = stored if new.empty else pd.concat([stored, run_metrics(new)], ignore_index=True)
    else:
        metrics = run_metrics(runs)
    metrics = metrics.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    metrics.to_parquet(tmp, index=False)
    tmp.replace(path)
    mf.write_text(json.dumps({"version": METRICS_VERSION, "source": sig, "races": int(metrics["race_id"].nunique()), "rows": len(metrics),
                              "data_until": str(metrics["date"].max().date()) if len(metrics) else None}, ensure_ascii=False), encoding="utf-8")
    return metrics


# ── 対象日時点のプロファイル ──────────────────────────────────────────────

def _sire_map() -> pd.DataFrame:
    m = pd.read_parquet(SIRE_MAP_PATH, columns=["horse_id", "sire_id"]).dropna().drop_duplicates("horse_id")
    m["horse_id"] = m["horse_id"].astype(str)
    m["sire_id"] = m["sire_id"].astype(str)
    return m


def priors_as_of(metrics: pd.DataFrame, as_of: pd.Timestamp, sires: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.Series]:
    """対象日より前の走だけで、父ごとの平均（MIN_SIRE_RUNS 走以上の父のみ）と全馬の平均。"""
    past = metrics[metrics["date"] < as_of]
    overall = past[METRICS].mean()
    sires = _sire_map() if sires is None else sires
    p = past[["horse_id"] + METRICS].merge(sires, on="horse_id", how="inner")
    g = p.groupby("sire_id")[METRICS]
    mean = g.mean().where(g.count() >= MIN_SIRE_RUNS)
    return mean, overall


def profiles_as_of(metrics: pd.DataFrame, horse_ids: Iterable[str], as_of: str | pd.Timestamp,
                   sires: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """対象日 as_of より前の走だけで、馬ごとの走法プロファイル。

    列: horse_id, sire_id, n_runs（対象日より前の平地の出走数）, last_run_date,
        方面ごとに <方面>（プロファイルの値）, n_<方面>（使った自分の走数）, own_<方面>（自分の走の重みの割合 0〜1）,
        prior_<方面>（引き寄せ先: "sire" / "overall" / None=引き寄せ先が無く自分の走だけ）。
    """
    as_of = pd.Timestamp(as_of)
    ids = [str(h) for h in horse_ids]
    sires = _sire_map() if sires is None else sires
    sire_mean, overall = priors_as_of(metrics, as_of, sires)
    sire_of = dict(zip(sires["horse_id"], sires["sire_id"]))
    hist = metrics[(metrics["date"] < as_of) & metrics["horse_id"].isin(ids)]
    by_horse = {k: g.sort_values(["date", "race_id"]) for k, g in hist.groupby("horse_id")}

    rows = []
    for hid in ids:
        h = by_horse.get(hid)
        sid = sire_of.get(hid)
        row: dict[str, object] = {"horse_id": hid, "sire_id": sid, "n_runs": 0 if h is None else len(h),
                                  "last_run_date": None if h is None else h["date"].iloc[-1].date().isoformat()}
        for m in METRICS:
            decay, c, prior_kind = PROFILE_PARAMS[m]
            own = h[m].dropna().to_numpy()[::-1][:LOOKBACK] if h is not None else np.array([])   # 新しい順
            w = decay ** np.arange(len(own))
            prior, src = math.nan, None
            if prior_kind == "sire" and sid is not None and sid in sire_mean.index and not math.isnan(sire_mean.at[sid, m]):
                prior, src = float(sire_mean.at[sid, m]), "sire"
            elif not math.isnan(overall[m]):
                prior, src = float(overall[m]), "overall"
            if math.isnan(prior):
                c, src = 0.0, None
            num, den = float((w * own).sum()), float(w.sum())
            row[m] = (num + c * (0.0 if math.isnan(prior) else prior)) / (den + c) if den + c > 0 else math.nan
            row[f"n_{m}"] = len(own)
            row[f"own_{m}"] = round(den / (den + c), 3) if den + c > 0 else 0.0
            row[f"prior_{m}"] = src
        rows.append(row)
    return pd.DataFrame(rows)


def profile_table(metrics: pd.DataFrame, sires: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """全出走について「その走の直前時点のプロファイル」（profiles_as_of と同じ式）をまとめて計算する。

    走法マップの順位の基準（全馬の分布）と、相性の標準化に使う。1 頭ずつの profiles_as_of と違い、
    引き寄せ先の父の平均は全期間の値（基準の分布を作るためだけの近似）。自分の走が 1 走も無い行は NaN。
    """
    sires = _sire_map() if sires is None else sires
    df = metrics.merge(sires, on="horse_id", how="left").sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    out = df[["race_id", "horse_id", "date"]].copy()
    for m in METRICS:
        decay, c, prior_kind = PROFILE_PARAMS[m]
        v = df[["horse_id", "sire_id", m]].dropna(subset=[m])
        overall = v[m].mean()
        g = v.groupby("sire_id")[m].agg(["mean", "size"])
        prior = v["sire_id"].map(g.loc[g["size"] >= MIN_SIRE_RUNS, "mean"]) if prior_kind == "sire" else pd.Series(np.nan, index=v.index)
        prior = prior.fillna(overall).to_numpy()
        gm = v.groupby("horse_id")[m]
        num = np.zeros(len(v))
        den = np.zeros(len(v))
        for i in range(1, LOOKBACK + 1):
            lag = gm.shift(i).to_numpy()
            ok = ~np.isnan(lag)
            num[ok] += decay ** (i - 1) * lag[ok]
            den[ok] += decay ** (i - 1)
        val = np.where(den > 0, (num + c * prior) / (den + c), np.nan)
        out[m] = pd.Series(val, index=v.index).reindex(df.index).to_numpy()
    return out


RQ_FEATS = ["ts_fastest_f", "gc_raw", "split_ratio", "front_ratio", "shift_ratio"]
RQ_COLS = [f"rq_{f}" for f in RQ_FEATS]
COND_KEY = ["venue", "distance", "surface", "track_condition"]
COURSE_KEY = ["venue", "distance", "surface"]


def race_quality(metrics: pd.DataFrame) -> pd.DataFrame:
    """レースごとのレース質（scripts/run_tp26.py の 6 章と同じ）: 各値のレース内順位と着順の順位相関。
    値がそろった 5 頭以上・5F 以上のレースだけ。"""
    df = metrics.dropna(subset=RQ_FEATS + ["finish_position"])
    df = df[df["n_laps"] >= 5]
    n = df.groupby("race_id")["race_id"].transform("size")
    df = df[n >= 5]
    out = df.groupby("race_id")[["venue", "distance", "surface", "track_condition", "date"]].first()
    rid = df["race_id"]
    size = df.groupby("race_id")["race_id"].transform("size")

    def centered(col: str) -> tuple[pd.Series, pd.Series]:
        r = df.groupby("race_id")[col].rank(method="average")
        return r - r.groupby(rid).transform("mean"), r.groupby(rid).transform("std")
    fin_c, fin_sd = centered("finish_position")
    for f in RQ_FEATS:              # 順位どうしのピアソン相関 = スピアマン相関
        fc, fsd = centered(f)
        r = (fc * fin_c).groupby(rid).transform("sum") / ((size - 1) * fsd * fin_sd).replace(0, np.nan)
        out[f"rq_{f}"] = r.groupby(rid).first()
    return out.dropna(subset=RQ_COLS).reset_index()


def course_tables(rq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """条件（競馬場・距離・芝ダ・馬場状態。3 レース以上）とコース（5 レース以上）ごとの、レース質の平均（run_tp26.py の 9 章）。"""
    def stats(key: list[str], min_n: int) -> pd.DataFrame:
        g = rq.groupby(key)
        t = g[RQ_COLS].mean()
        t["n_races"] = g.size()
        return t[t["n_races"] >= min_n].reset_index()
    return stats(COND_KEY, 3), stats(COURSE_KEY, 5)
