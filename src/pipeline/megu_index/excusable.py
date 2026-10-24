"""『無視してよいめぐ指数の低下（excusable）』の判別（tp-69）。

過去走の指数が低かったとき、それが馬の力の低下を示すのか、不利・展開・馬場・状態などの一過性かを、
**その走の結果が確定した時点以降に分かる信号だけ**から判定する（先読みなし）。次走以降の指数は判定に使わない
（学習・評価のラベルにだけ使う: 低い走のあと次の 2 走で通常水準へ戻ったか＝反発）。
信号: 結果表の事実（通過順位・上がり順位・着差・頭数）、レース水準（展開・上がり）、備考のフラグ（不利・出遅れ）、
馬体重・間隔・初条件・斤量の変化、馬自身の履歴に対する外れ度。市場オッズ・人気は使わない。
判定器は standardize 済みのロジスティック回帰（係数は config/megu_excusable.json）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "megu_excusable.json"

SIGNALS = [
    "own_drop", "n_prior", "fp_pct", "l3f_vs_fp", "l3f_rank_pct", "late_gain", "c1_pct", "c4_pct", "margin_sec",
    "pace_dis", "race_l3f_dev", "trouble_rem", "slow_rem", "has_remarks", "wchg_abs", "wchg_big",
    "rest_log", "layoff90", "novel_dist", "novel_surf", "jw_change", "going_heavy", "big_field", "low_conf",
]

# 根拠の短文（係数が正で効いた信号のうち、値が大きいものを上位から使う）
REASON_JA = {
    "l3f_vs_fp": "上がり3Fは上位なのに着順が悪い（詰まり・仕掛け遅れの可能性）",
    "late_gain": "道中から着順にかけて順位を落としている（直線の不利の可能性）",
    "pace_dis": "このペース・隊列に対して脚質が不利だった",
    "trouble_rem": "備考に不利・躓き・挟まれ等の記載",
    "slow_rem": "備考に出遅れの記載",
    "wchg_big": "馬体重が大きく変動していた",
    "layoff90": "休み明け（90日以上）だった",
    "novel_dist": "初めての距離だった",
    "novel_surf": "初めての馬場（芝/ダ）だった",
    "jw_change": "斤量が大きく変わっていた",
    "going_heavy": "重・不良馬場だった",
    "big_field": "多頭数（16頭以上）だった",
    "race_l3f_dev": "レース全体の上がりが極端だった",
    "own_drop": "この馬の通常水準から大きく外れた1走だった",
    "low_conf": "基準タイムが薄く、指数の信頼が低い",
}

MARGIN_LEN = {"ハナ": 0.1, "アタマ": 0.2, "クビ": 0.3, "1/2": 0.5, "3/4": 0.75, "大": 12.0}


def parse_margin_len(s) -> float:
    """着差の表記（ハナ・クビ・1/2・1.1/4・3 など）→ 馬身。1着や空は 0。読めなければ NaN。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = str(s).strip()
    if t == "":
        return 0.0
    if t in MARGIN_LEN:
        return MARGIN_LEN[t]
    m = re.fullmatch(r"(\d+)\.(\d)/(\d)", t)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.fullmatch(r"(\d+)/(\d)", t)
    if m:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(t)
    except ValueError:
        return np.nan


def parse_passing(s) -> tuple[float, float]:
    """通過順位 '3-3-2-2' → (最初のコーナー, 最後のコーナー)。読めなければ NaN。"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan, np.nan
    nums = [int(x) for x in re.findall(r"\d+", str(s))]
    return (float(nums[0]), float(nums[-1])) if nums else (np.nan, np.nan)


def style_bucket(c1_pct: pd.Series) -> pd.Series:
    return pd.cut(c1_pct, [-0.01, 0.2, 0.45, 0.7, 1.01], labels=["front", "stalk", "mid", "back"]).astype(object)


def fit_lift_table(df: pd.DataFrame) -> dict[str, float]:
    """(脚質, ペース, 隊列) ごとの 3 着内率の比（全体に対する）。df は参照期間（評価年より前）だけ。キーは 'front|high|compact'。"""
    d = df.dropna(subset=["style", "pace_cls", "shape_cls", "top3"]).copy()
    base = float(d["top3"].mean())
    g = d.groupby(["style", "pace_cls", "shape_cls"])["top3"].agg(["sum", "size"])
    k = 30.0  # 縮約（小さいセルは全体へ）
    lift = ((g["sum"] + k * base) / (g["size"] + k)) / base
    return {f"{a}|{b}|{c}": float(v) for (a, b, c), v in lift.items()}


def run_signals(df: pd.DataFrame, lift: dict[str, float] | None = None, scale: float = 11.5) -> pd.DataFrame:
    """走ごとの信号。df: 馬×日付で並べ替え済みでなくてよい（内部で並べ替え、元の index で返す）。
    必須列: horse_id, race_id, date, fp, field_size, passing_order, last_3f, margin, distance, surface, x（指数。下限適用済み）
    任意列: weight_change, jockey_weight, track_condition, pace_cls, shape_cls, race_l3f_dev, trouble_rem, slow_rem, has_remarks, low_conf
    履歴の信号（own_drop・n_prior・休み・初条件・斤量変化）は、その走より前の走だけから作る。"""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    order = d.sort_values(["horse_id", "date", "race_id"]).index
    d = d.loc[order]
    n = d["field_size"].astype(float).clip(lower=1)
    fp = pd.to_numeric(d["fp"], errors="coerce")
    out = pd.DataFrame(index=d.index)
    out["fp_pct"] = (fp / n).where(fp > 0)
    pa = d["passing_order"].map(parse_passing)
    c1 = pd.Series([p[0] for p in pa], index=d.index)
    c4 = pd.Series([p[1] for p in pa], index=d.index)
    out["c1_pct"] = c1 / n
    out["c4_pct"] = c4 / n
    out["late_gain"] = out["c4_pct"] - out["fp_pct"]            # 負＝最後のコーナーから着順にかけて順位を落とした
    l3 = pd.to_numeric(d["last_3f"], errors="coerce")
    l3r = l3.where(l3 > 0).groupby(d["race_id"]).rank(method="min")
    out["l3f_rank_pct"] = l3r / n
    out["l3f_vs_fp"] = out["fp_pct"] - out["l3f_rank_pct"]       # 正＝上がり順位に比べて着順が悪い
    out["margin_sec"] = d["margin"].map(parse_margin_len) * 0.2
    st = style_bucket(out["c1_pct"])
    d = d.assign(style=st)
    if lift:
        key = d["style"].astype(str) + "|" + d["pace_cls"].astype(str) + "|" + d["shape_cls"].astype(str)
        lf = key.map(lift)
        out["pace_dis"] = -np.log(lf.clip(lower=0.2, upper=5.0))
    else:
        out["pace_dis"] = np.nan
    out["race_l3f_dev"] = pd.to_numeric(d.get("race_l3f_dev"), errors="coerce") if "race_l3f_dev" in d else np.nan
    for c in ("trouble_rem", "slow_rem", "has_remarks"):
        out[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    wc = pd.to_numeric(d.get("weight_change"), errors="coerce") if "weight_change" in d else pd.Series(np.nan, index=d.index)
    out["wchg_abs"] = wc.abs()
    out["wchg_big"] = (wc.abs() >= 10).astype(float).where(wc.notna())
    g = d["horse_id"]
    prev_date = d["date"].groupby(g).shift(1)
    rest = (d["date"] - prev_date).dt.days
    out["rest_log"] = np.log1p(rest.clip(lower=0))
    out["layoff90"] = (rest >= 90).astype(float).where(rest.notna())
    dist = pd.to_numeric(d["distance"], errors="coerce")
    seen_d = pd.Series(False, index=d.index)
    # 初めての距離帯（±200m 以内に過去走が無い）・初めての馬場（芝/ダ）
    pd_prev = pd.DataFrame({"g": g, "dist": dist, "surf": d["surface"]})
    nov_d = np.full(len(d), np.nan)
    nov_s = np.full(len(d), np.nan)
    prior_d: dict = {}
    prior_s: dict = {}
    for i, (h, dd, ss) in enumerate(zip(pd_prev["g"].to_numpy(), pd_prev["dist"].to_numpy(), pd_prev["surf"].to_numpy())):
        pdl = prior_d.setdefault(h, [])
        psl = prior_s.setdefault(h, set())
        if pdl:
            nov_d[i] = float(all(abs(dd - x) > 200 for x in pdl)) if not np.isnan(dd) else np.nan
            nov_s[i] = float(ss not in psl)
        pdl.append(dd)
        psl.add(ss)
    out["novel_dist"], out["novel_surf"] = nov_d, nov_s
    jw = pd.to_numeric(d.get("jockey_weight"), errors="coerce") if "jockey_weight" in d else pd.Series(np.nan, index=d.index)
    out["jw_change"] = (jw - jw.groupby(g).shift(1)).abs()
    tc = d["track_condition"].astype(str) if "track_condition" in d else pd.Series("", index=d.index)
    out["going_heavy"] = tc.isin(["重", "不良"]).astype(float)
    out["big_field"] = (n >= 16).astype(float)
    lc = d["low_conf"] if "low_conf" in d else pd.Series(np.nan, index=d.index)
    out["low_conf"] = pd.to_numeric(lc.astype(float), errors="coerce")
    # 馬自身の履歴に対する外れ度: その走の指数が、それ以前の直近 5 走の中央値よりどれだけ低いか（尺度で割る）
    x = pd.to_numeric(d["x"], errors="coerce")
    M = np.column_stack([x.groupby(g).shift(i).to_numpy() for i in range(1, 6)])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(M, axis=1)
    out["n_prior"] = np.sum(~np.isnan(M), axis=1).astype(float)
    out["own_drop"] = np.where(out["n_prior"] >= 3, np.maximum(0.0, (med - x.to_numpy()) / scale), np.nan)
    out["prior_med"] = med
    return out.reindex(df.index)


def rebound_label(df: pd.DataFrame, scale: float = 11.5, tol: float = 0.5) -> pd.Series:
    """評価用の事後ラベル: 低い走のあと、次の 2 走（指数がある走）の平均が、その前の通常水準（直前 5 走の中央値）− tol×scale 以上なら 1（反発）。
    次の走が 1 つも無い走は NaN。判定ロジックには使わない。df は run_signals の prior_med を持つこと。"""
    d = df.sort_values(["horse_id", "date", "race_id"])
    x = d["x"].astype(float)
    g = d["horse_id"]
    n1 = x.groupby(g).shift(-1)
    n2 = x.groupby(g).shift(-2)
    nxt = pd.concat([n1, n2], axis=1).mean(axis=1)
    lab = (nxt >= d["prior_med"] - tol * scale).astype(float).where(nxt.notna() & d["prior_med"].notna())
    return lab.reindex(df.index)


class ExcusableModel:
    """standardize 済みロジスティック回帰。欠損の信号は 0（平均）として扱う。"""

    def __init__(self, coef: dict[str, float], intercept: float, mean: dict[str, float], std: dict[str, float], meta: dict | None = None):
        self.coef, self.intercept, self.mean, self.std, self.meta = coef, intercept, mean, std, meta or {}

    def z(self, sig: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for c in self.coef:
            v = pd.to_numeric(sig[c], errors="coerce") if c in sig else pd.Series(np.nan, index=sig.index)
            cols[c] = ((v - self.mean[c]) / max(self.std[c], 1e-9)).fillna(0.0)
        return pd.DataFrame(cols, index=sig.index)

    def score(self, sig: pd.DataFrame) -> pd.Series:
        Z = self.z(sig)
        lin = sum(Z[c] * w for c, w in self.coef.items()) + self.intercept
        return 1.0 / (1.0 + np.exp(-lin))

    def reasons(self, sig_row: pd.Series, top: int = 2) -> list[str]:
        contrib = {}
        for c, w in self.coef.items():
            if w <= 0 or c not in REASON_JA:
                continue
            v = pd.to_numeric(pd.Series([sig_row.get(c)]), errors="coerce").iloc[0]
            if pd.isna(v):
                continue
            z = (v - self.mean[c]) / max(self.std[c], 1e-9)
            if z > 0.5:
                contrib[c] = w * z
        return [REASON_JA[c] for c, _ in sorted(contrib.items(), key=lambda kv: -kv[1])[:top]]

    def to_json(self) -> dict:
        return {"coef": self.coef, "intercept": self.intercept, "mean": self.mean, "std": self.std, "meta": self.meta}

    @classmethod
    def from_json(cls, j: dict) -> "ExcusableModel":
        return cls(j["coef"], j["intercept"], j["mean"], j["std"], j.get("meta"))

    @classmethod
    def load(cls, path: Path | None = None) -> "ExcusableModel | None":
        p = path or CONFIG
        if not Path(p).exists():
            return None
        return cls.from_json(json.loads(Path(p).read_text(encoding="utf-8")))


_DEFAULT: dict = {}


def load_default_model() -> "ExcusableModel | None":
    """config/megu_excusable.json の判定器（ファイルの更新時刻でキャッシュ）。無ければ None。"""
    if not CONFIG.exists():
        return None
    mt = CONFIG.stat().st_mtime
    if _DEFAULT.get("mt") != mt:
        _DEFAULT["mt"], _DEFAULT["m"] = mt, ExcusableModel.load(CONFIG)
    return _DEFAULT["m"]


def fit_model(sig: pd.DataFrame, y: pd.Series, mask: pd.Series, C: float = 0.5, cols: list[str] | None = None) -> ExcusableModel:
    """低い走（mask）だけで、反発（y=1）の確率ではなく **『反発』= excusable の方向** をロジスティック回帰で当てはめる。"""
    from sklearn.linear_model import LogisticRegression

    cols = cols or SIGNALS
    S = sig.loc[mask, cols].apply(pd.to_numeric, errors="coerce")
    mean = S.mean().fillna(0.0)
    std = S.std().replace(0, 1.0).fillna(1.0)
    Z = ((S - mean) / std).fillna(0.0)
    yy = y.loc[mask].astype(int)
    lr = LogisticRegression(C=C, max_iter=500)
    lr.fit(Z, yy)
    return ExcusableModel(dict(zip(cols, map(float, lr.coef_[0]))), float(lr.intercept_[0]), {c: float(mean[c]) for c in cols},
                          {c: float(std[c]) for c in cols}, {"n_train": int(mask.sum()), "base_rate": float(yy.mean()), "C": C})
