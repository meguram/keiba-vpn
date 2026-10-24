"""tp-30: 荒れやすいレースの分析（事後指標・事前予測の入力・温度調整・買い目の幅）。

区別:
  モデル用 = オッズ・人気を使わない入力（レース条件・出走メンバーの能力の散らばりなど。発走前に分かる）
  馬券用   = 上に加えて購入前のオッズ構造（履歴は確定オッズで代用）。馬券の幅の決定にだけ使う。
事後指標（荒れの正解）は結果から作る。特徴量にするのは『事前に予測した荒れ度』であって、事後指標そのものではない。
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("data/page_reference/tables")
JRA = {f"{i:02d}" for i in range(1, 11)}
RES_COLS = ["race_id", "date", "venue_code", "surface", "distance", "grade", "race_class", "weather", "track_condition", "field_size", "round",
            "horse_number", "horse_id", "finish_position", "popularity", "odds", "time_sec", "payoff"]


# ───────────────────────── 払戻
def _yen(v) -> int | None:
    s = re.sub(r"[^\d]", "", str(v or ""))
    return int(s) if s else None


def _nums(s) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", str(s or "")))


def parse_payoff(js) -> dict:
    """payoff JSON → {tansho:(馬番, 円), fukusho:[(馬番, 円)], umaren:(組, 円), wide:[(組, 円)], umatan, sanrenpuku:(組, 円), sanrentan}。"""
    try:
        d = json.loads(js) if isinstance(js, str) else (js or {})
    except Exception:
        return {}
    out: dict = {}

    def one(v):
        if isinstance(v, list):
            v = v[0] if v else None
        if not v:
            return None
        y = _yen(v.get("payout"))
        return (_nums(v.get("numbers")), y) if y else None

    def many(v, pair: bool = False):
        """複勝・ワイドは複数行。2024 年の一部は 1 つのセルに連結（numbers='3168'、payout='330120540'）されていて分解できないので、その場合は空にする。"""
        if isinstance(v, dict):
            nums = _nums(v.get("numbers"))
            if (pair and len(nums) != 2) or (not pair and len(str(v.get("numbers") or "").replace(" ", "")) > 2):
                out["_concat_" + ("wide" if pair else "fukusho")] = True
                return []
        v = v if isinstance(v, list) else ([v] if v else [])
        return [(_nums(x.get("numbers")), _yen(x.get("payout"))) for x in v if x and _yen(x.get("payout"))]

    for k, name in (("単勝", "tansho"), ("馬連", "umaren"), ("馬単", "umatan"), ("三連複", "sanrenpuku"), ("三連単", "sanrentan")):
        o = one(d.get(k))
        if o:
            out[name] = o
    out["fukusho"] = many(d.get("複勝"))
    out["wide"] = many(d.get("ワイド"), pair=True)
    return out


def load_results(years, root: Path = TABLES) -> pd.DataFrame:
    import polars as pl

    parts = []
    for y in years:
        f = root / str(y) / "race_result_flat.parquet"
        sch = pl.read_parquet_schema(f)
        d = pl.read_parquet(f, columns=[c for c in RES_COLS if c in sch]).to_pandas()
        parts.append(d)
    r = pd.concat(parts, ignore_index=True)
    r["race_id"] = r["race_id"].astype(str)
    r["horse_id"] = r["horse_id"].astype(str)
    r = r[r["venue_code"].astype(str).isin(JRA)]
    for c in ("finish_position", "popularity", "odds", "time_sec", "horse_number", "field_size", "distance"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    from src.scraper.odds_fix import apply_pandas

    r = apply_pandas(r)   # 2025 年は odds が指数列だった
    return r.reset_index(drop=True)


# ───────────────────────── 事後指標（荒れの正解、レース単位）
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4:
        return np.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def posthoc(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, g in res.groupby("race_id", sort=False):
        f = g[(g["finish_position"] > 0) & (g["popularity"] > 0)]
        if len(f) < 5:
            continue
        pay = parse_payoff(g["payoff"].iloc[0])
        top3 = f[f["finish_position"] <= 3]
        win = f[f["finish_position"] == 1]
        fav3 = set(f.nsmallest(3, "popularity")["horse_id"])
        r = {"race_id": rid, "n": len(f),
             "win_pop": float(win["popularity"].min()) if len(win) else np.nan,
             "mean_pop_top3": float(top3["popularity"].mean()) if len(top3) else np.nan,
             "rho": _spearman(f["popularity"].to_numpy(), f["finish_position"].to_numpy()),
             "fav3_all_top3": int(fav3 <= set(top3["horse_id"])),
             "n_long_top3": int((top3["popularity"] >= 6).sum()),
             "pay_tansho": pay.get("tansho", (None, np.nan))[1], "pay_umaren": pay.get("umaren", (None, np.nan))[1],
             "pay_sanrenpuku": pay.get("sanrenpuku", (None, np.nan))[1], "pay_sanrentan": pay.get("sanrentan", (None, np.nan))[1]}
        rows.append(r)
    ph = pd.DataFrame(rows)
    ph["log_win_pop"] = np.log(ph["win_pop"])
    for c in ("tansho", "umaren", "sanrenpuku", "sanrentan"):
        ph[f"log_pay_{c}"] = np.log(ph[f"pay_{c}"].astype(float))
    ph["manba"] = (ph["pay_sanrenpuku"] >= 10000).astype(float)
    return ph.replace([np.inf, -np.inf], np.nan)


def add_composite(ph: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """荒れ度の合成（標準化した log 3連複配当・log 勝ち馬人気・3着以内の平均人気の log・−rho の平均）。標準化の平均・標準偏差は ref（学習側の年）から。"""
    parts = {"log_pay_sanrenpuku": 1, "log_win_pop": 1, "log_mean_pop_top3": 1, "neg_rho": 1}
    for d in (ph, ref):
        d["log_mean_pop_top3"] = np.log(d["mean_pop_top3"])
        d["neg_rho"] = -d["rho"]
    z = []
    for c in parts:
        mu, sd = ref[c].mean(), ref[c].std()
        z.append(((ph[c] - mu) / sd).to_numpy())
    ph["vol"] = np.nanmean(np.vstack(z), axis=0)
    return ph


def odds_cols(res: pd.DataFrame) -> pd.DataFrame:
    """馬券用: 確定オッズ（購入前の代用）の構造。レース単位。"""
    rows = []
    for rid, g in res.groupby("race_id", sort=False):
        o = pd.to_numeric(g["odds"], errors="coerce")
        o = o[(o > 0)]
        if len(o) < 5:
            continue
        p = (1.0 / o).to_numpy()
        p = p / p.sum()
        ps = np.sort(p)[::-1]
        rows.append({"race_id": rid, "mk_fav_odds": float(o.min()), "mk_p1": ps[0], "mk_p12": ps[0] - ps[1], "mk_p3sum": ps[:3].sum(),
                     "mk_entropy": float(-(p * np.log(p)).sum()), "mk_n_lt10": int((o < 10).sum()), "mk_logodds_sd": float(np.log(o).std())})
    return pd.DataFrame(rows)


# ───────────────────────── 事前入力（モデル用: オッズ・人気なし）
AGG_COLS = ["sr_perf_ewm6", "sr_perf_ewm2", "horse_prior_win_rate", "jockey_prior_win_rate", "trainer_prior_win_rate", "f3_r", "c_res_ewm2",
            "a_rk_ewm2", "b_aff_all", "pc_adv", "f2_pace_match"]


def _entropy_softmax(x: np.ndarray) -> float:
    z = np.exp(x - np.nanmax(x))
    z = np.where(np.isfinite(z), z, 0)
    s = z.sum()
    if s <= 0:
        return np.nan
    p = z / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def race_context(tt: pd.DataFrame) -> pd.DataFrame:
    """学習表（馬単位）→ レース単位の事前入力。レース条件 + 出走メンバーの能力の散らばり。人気・オッズは一切入れない。"""
    tt = tt.copy()
    tt["race_id"] = tt["race_id"].astype(str)
    g = tt.groupby("race_id", sort=False)
    out = g.agg(field_size=("field_size", "max"), distance=("distance", "max"), round=("round", "max"), n_debut=("horse_prior_races", lambda s: int((s.fillna(0) == 0).sum())),
                date=("date", "min")).reset_index()
    for c in ("surface", "venue_code", "grade", "race_class", "weather", "track_condition"):
        out[c] = g[c].agg(lambda s: str(s.iloc[0])).to_numpy()
    out["month"] = pd.to_datetime(out["date"]).dt.month
    for c in AGG_COLS:
        if c not in tt.columns:
            continue
        x = tt[c].astype("float64")
        gg = x.groupby(tt["race_id"], sort=False)
        out[f"{c}_sd"] = gg.std().to_numpy()
        out[f"{c}_gap"] = (gg.max() - gg.mean()).to_numpy()
        out[f"{c}_top12"] = gg.apply(lambda s: (lambda v: v[0] - v[1] if len(v) > 1 else np.nan)(np.sort(s.dropna().to_numpy())[::-1])).to_numpy()
        out[f"{c}_ent"] = gg.apply(lambda s: _entropy_softmax(s.fillna(s.mean()).to_numpy())).to_numpy()
    out["style_mean"] = tt["a_style"].astype("float64").groupby(tt["race_id"], sort=False).mean().to_numpy()
    out["style_sd"] = tt["a_style"].astype("float64").groupby(tt["race_id"], sort=False).std().to_numpy()
    return out


# ───────────────────────── 温度調整
def temper(p: np.ndarray, tau: float, k: float = 1.0) -> np.ndarray:
    """logit(p)/tau + c で、合計が k になる c を二分法で探す（k=1: 勝率、2: 連対、3: 複勝）。"""
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    z = np.log(p / (1 - p)) / tau
    lo, hi = -30.0, 30.0
    for _ in range(50):
        c = (lo + hi) / 2
        s = (1 / (1 + np.exp(-(z + c)))).sum()
        lo, hi = (c, hi) if s < k else (lo, c)
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))


# ───────────────────────── 3連複の買い目
def formation_combos(cols: list[list[int]]) -> set[tuple[int, int, int]]:
    """3連複フォーメーション: 1列目×2列目×3列目から相異なる 3 頭を選ぶ組の集合（順不同）。"""
    out = set()
    for a in cols[0]:
        for b in cols[1]:
            if b == a:
                continue
            for c in cols[2]:
                if c in (a, b):
                    continue
                out.add(tuple(sorted((a, b, c))))
    return out


def box_combos(idx: list[int], r: int = 3) -> set[tuple[int, ...]]:
    return {tuple(sorted(t)) for t in combinations(idx, r)}


def nagashi_combos(axis: list[int], partners: list[int]) -> set[tuple[int, int, int]]:
    """3連複 軸 1 頭流し: 軸 + 相手 2 頭。"""
    out = set()
    for a in axis:
        for t in combinations([x for x in partners if x != a], 2):
            out.add(tuple(sorted((a, *t))))
    return out


def pad_races(race_id: np.ndarray, *cols: np.ndarray, width: int = 18):
    """レースごとに行をまとめて (R, width) の配列にする（欠けは mask=False）。戻り値: (race 順の race_id, mask, 各列の配列..., 元の行の位置 index)。"""
    order = pd.Series(np.arange(len(race_id))).groupby(race_id, sort=False).apply(list)
    R = len(order)
    mask = np.zeros((R, width), bool)
    outs = [np.zeros((R, width)) for _ in cols]
    pos = -np.ones((R, width), int)
    for r_, idx in enumerate(order.to_numpy()):
        n = min(len(idx), width)
        mask[r_, :n] = True
        pos[r_, :n] = idx[:n]
        for o, c in zip(outs, cols):
            o[r_, :n] = np.asarray(c)[idx[:n]]
    return order.index.to_numpy(), mask, outs, pos


def temper_batch(P: np.ndarray, mask: np.ndarray, tau: np.ndarray, k: float) -> np.ndarray:
    """temper をレースごとにまとめて計算（P: (R, W)、tau: (R,)）。合計が k になる c を全レース同時に二分法で探す。"""
    p = np.clip(P, 1e-9, 1 - 1e-9)
    z = np.where(mask, np.log(p / (1 - p)) / tau[:, None], 0.0)
    lo = np.full(len(P), -30.0)
    hi = np.full(len(P), 30.0)
    for _ in range(40):
        c = (lo + hi) / 2
        s = np.where(mask, 1 / (1 + np.exp(-(z + c[:, None]))), 0.0).sum(axis=1)
        lo = np.where(s < k, c, lo)
        hi = np.where(s < k, hi, c)
    q = 1 / (1 + np.exp(-(z + ((lo + hi) / 2)[:, None])))
    return np.where(mask, q, 0.0)
