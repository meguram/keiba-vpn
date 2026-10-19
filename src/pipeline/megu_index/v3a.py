"""めぐ指数 v3a（改善版）。

工程1の実測（2026-10-19）で見つかった弱点を直す。既存の指数列は変えず、``megu_index_v3a*`` の列を足す。

  1. 斤量: 基準を全馬 55kg に統一し、係数 β を小さくする（2000m で 0.61 → 評価で選ぶ値。固定効果の推定は約 0.025）。
  2. 馬場: 『同日・同場・同面の他レースの補正前残差の中央値』（leave-one-race-out）を馬場差に使う。
     他レースが足りない日だけ、従来のクッション値/含水率の補正（TSI）にフォールバックする。
  3. ペース: 芝だけに掛ける（ダートの係数は根拠が無い）。上限は ``pace_cap``。
  4. 基準タイム: 世代（2歳/3歳/3歳以上/4歳以上）×クラスを区別し、薄いセルは親のセルに縮約する。
     世代×クラスの季節（月）の水準差は、平滑した月別オフセットで補う。
  5. 絶対化: クラスをまたいで比べられる指数（昇級馬を同じ軸で追える）を ``megu_index_v3a_abs`` として別に出す。
     クラス間差は『同一馬・同一年齢』で比べた固定効果から推定する。

基準タイム・月別オフセット・クラスオフセットは「基準期間」のデータから作ってファイルに保存し、日次でも同じ値を使う
（将来年にその年の結果から自己推定した値を使わない）。馬場差は同日のレースだけから作るので、日次でも再計算でも同じ値になる。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

V3A_MODEL_VERSION = "v3a"
V3A_VERSION_LABEL = "v3a-w55-loro-genpar"
BASE_INDEX = 50.0
POINTS_PER_SEC = 10.0
CLIP_MIN, CLIP_MAX = 0.0, 100.0

V3A_COLUMNS = [
    "megu_index_v3a", "megu_index_v3a_robust", "megu_index_v3a_abs", "megu_index_v3a_abs_robust",
    "megu_v3a_status", "megu_v3a_low_confidence", "megu_v3a_version",
    "v3a_track_sec", "v3a_track_src", "v3a_par_sec", "v3a_par_level", "v3a_par_n", "v3a_corrected_sec",
]

_CLASS_GROUP = {
    "新馬": "新馬", "未勝利": "未勝利", "1勝": "1勝", "2勝": "2勝", "3勝": "3勝",
    "L": "OP", "OP": "OP", "G3": "重賞", "G2": "重賞", "G1": "G1",
}
CLASS_ORDER = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "重賞", "G1"]
GENS = ("2歳", "3歳", "3歳以上", "4歳以上")
_USABLE = ("ok", "clipped", "restored_out_of_range", "restored_clipped")


@dataclass
class V3ASpec:
    beta_weight: float = 0.1       # 斤量補正（秒/kg、2000m 換算。距離に比例）
    base_kg: float = 55.0
    pace_cap: float = 5.0          # ペース補正の上限（秒）
    pace_turf_only: bool = True
    loro_min_other_races: int = 2  # 馬場差を作るのに必要な『同日・同場・同面の他レース』の数
    loro_min_horses: int = 6       # 馬場差の推定に使うレースの最小頭数
    min_par_n: int = 30            # これ未満のセルは low_confidence
    shrink_k: float = 30.0         # 基準タイムの親セルへの縮約の強さ
    month_smooth: int = 3          # 月別オフセットの移動平均の窓
    month_shrink_k: float = 30.0
    n_iter: int = 3                # 基準タイムと馬場差の往復回数
    abs_min_runs: int = 3          # 絶対化の固定効果推定に使う馬の最小走数


@dataclass
class V3AModel:
    spec: V3ASpec
    par: pd.DataFrame                      # level, surface, distance, direction, gen, cls, mean, n, par
    month_offsets: pd.DataFrame            # gen, cls, month, off_pts
    class_offsets: dict[str, float]        # cls -> gamma（相対指数が絶対指数より低く出る分。abs = rel − gamma）
    meta: dict = field(default_factory=dict)

    # ── 保存・読込 ───────────────────────────────────────────
    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        (d / "spec.json").write_text(json.dumps(asdict(self.spec), ensure_ascii=False, indent=1))
        self.par.to_csv(d / "par.csv", index=False)
        self.month_offsets.to_csv(d / "month_offsets.csv", index=False)
        (d / "class_offsets.json").write_text(json.dumps(self.class_offsets, ensure_ascii=False, indent=1))
        (d / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1, default=str))

    @classmethod
    def load(cls, d: Path) -> "V3AModel":
        d = Path(d)
        spec = V3ASpec(**json.loads((d / "spec.json").read_text()))
        par = pd.read_csv(d / "par.csv", keep_default_na=False, na_values=[""])
        for c in ("direction", "gen", "cls", "surface"):
            if c in par.columns:
                par[c] = par[c].fillna("").astype(str)
        return cls(
            spec=spec,
            par=par,
            month_offsets=pd.read_csv(d / "month_offsets.csv"),
            class_offsets=json.loads((d / "class_offsets.json").read_text()),
            meta=json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {},
        )


# ── 区分 ──────────────────────────────────────────────────────
def gen_of(race_class) -> str:
    s = str(race_class or "")
    if "２歳" in s or "2歳" in s:
        return "2歳"
    if "３歳以上" in s or "3歳以上" in s:
        return "3歳以上"
    if "３歳" in s or "3歳" in s:
        return "3歳"
    if "４歳以上" in s or "4歳以上" in s:
        return "4歳以上"
    return "他"


def class_of(grade, race_name=None, race_class=None) -> str:
    from src.pipeline.megu_index.career_prize import classify_grade

    return _CLASS_GROUP.get(classify_grade(grade, race_name, race_class), "OP")


# ── 作業フレーム ─────────────────────────────────────────────────
def _num(df: pd.DataFrame, c: str, default=np.nan) -> pd.Series:
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def prepare(df: pd.DataFrame, spec: V3ASpec, beta_pace: float) -> pd.DataFrame:
    """compute_for_dataframe の途中結果から、v3a の作業列を作って返す（行は増減しない）。"""
    w = df.copy()
    t = _num(w, "finish_time_sec")
    dev = _num(w, "front_split_dev").fillna(0.0)
    pace = (beta_pace * dev).clip(-spec.pace_cap, spec.pace_cap)
    if spec.pace_turf_only:
        pace = pace.where(w["surface"].eq("芝"), 0.0)
    wt = _num(w, "jockey_weight_num")
    ds = _num(w, "dist_scale").fillna(1.0)
    w["v3a_pace"] = pace
    w["v3a_weight"] = (spec.beta_weight * (wt - spec.base_kg) * ds).fillna(0.0)
    w["v3a_weight_missing"] = wt.isna()
    w["v3a_c0"] = t - w["v3a_pace"] - w["v3a_weight"]
    w["v3a_tsi_track"] = _num(w, "delta_track_sec").fillna(0.0)
    date = pd.to_datetime(w["date"], errors="coerce")
    w["v3a_month"] = date.dt.month
    w["v3a_gen"] = w["race_class"].map(gen_of) if "race_class" in w.columns else "他"
    trip = pd.DataFrame({
        "g": w["grade"] if "grade" in w.columns else "",
        "n": w["race_name"] if "race_name" in w.columns else "",
        "c": w["race_class"] if "race_class" in w.columns else "",
    }, index=w.index).fillna("")
    cache: dict[tuple, str] = {}
    cls_vals = []
    for g, n, c in zip(trip["g"], trip["n"], trip["c"]):
        k = (g, n, c)
        if k not in cache:
            cache[k] = class_of(g, n, c)
        cls_vals.append(cache[k])
    w["v3a_cls"] = cls_vals
    w["v3a_day"] = (
        date.dt.strftime("%Y-%m-%d").fillna("") + "|" + w.get("venue_code", pd.Series("", index=w.index)).astype(str)
        + "|" + w["surface"].astype(str)
    )
    oor = w["out_of_range"].astype(bool) if "out_of_range" in w.columns else pd.Series(False, index=w.index)
    w["v3a_oor"] = oor
    w["v3a_fp"] = _num(w, "finish_pos_num") if "finish_pos_num" in w.columns else _num(w, "finish_position")
    w["distance"] = _num(w, "distance")
    w["direction"] = w["direction"].fillna("").astype(str) if "direction" in w.columns else ""
    return w


# ── 基準タイム（世代×クラス×コース。親セルへの縮約つき）─────────────────────
_LEVELS = [
    ("L1_dir_gen_cls", ["surface", "distance", "direction", "gen", "cls"]),
    ("L2_gen_cls", ["surface", "distance", "gen", "cls"]),
    ("L3_cls", ["surface", "distance", "cls"]),
    ("L4_pooled", ["surface", "distance"]),
]
_KEYCOLS = ["surface", "distance", "direction", "gen", "cls"]


def build_par_table(rows: pd.DataFrame, spec: V3ASpec) -> pd.DataFrame:
    """rows: 2 着馬（無ければ 1 着）の行。列 surface, distance, direction, gen, cls, c（補正後タイム）。"""
    r = rows.rename(columns={"v3a_gen": "gen", "v3a_cls": "cls"}).copy()
    out = []
    parent = None
    for name, keys in reversed(_LEVELS):
        g = r.groupby(keys, as_index=False).agg(mean=("c", "mean"), n=("c", "size"))
        if parent is None:
            g["par"] = g["mean"]
        else:
            pkeys, ptab = parent
            g = g.merge(ptab[pkeys + ["par"]].rename(columns={"par": "par_parent"}), on=pkeys, how="left")
            pp = g["par_parent"].fillna(g["mean"])
            g["par"] = (g["n"] * g["mean"] + spec.shrink_k * pp) / (g["n"] + spec.shrink_k)
            g = g.drop(columns=["par_parent"])
        g["level"] = name
        parent = (keys, g[keys + ["par"]].copy())
        for c in _KEYCOLS:
            if c not in g.columns:
                g[c] = ""
        out.append(g)
    tab = pd.concat(out, ignore_index=True)
    return tab[["level", *_KEYCOLS, "mean", "n", "par"]]


def attach_par(w: pd.DataFrame, par: pd.DataFrame) -> pd.DataFrame:
    """作業フレームに v3a_par_sec / v3a_par_level / v3a_par_n を付ける（最も細かい、存在するセルを使う）。"""
    out = w.copy()
    out["v3a_par_sec"] = np.nan
    out["v3a_par_level"] = ""
    out["v3a_par_n"] = np.nan
    key_w = out.rename(columns={"v3a_gen": "gen", "v3a_cls": "cls"})
    for name, keys in _LEVELS:
        sub = par[par["level"] == name][keys + ["par", "n"]].drop_duplicates(keys)
        m = key_w[keys].merge(sub, on=keys, how="left")
        m.index = out.index
        need = out["v3a_par_sec"].isna() & m["par"].notna()
        out.loc[need, "v3a_par_sec"] = m.loc[need, "par"]
        out.loc[need, "v3a_par_level"] = name
        out.loc[need, "v3a_par_n"] = m.loc[need, "n"]
    return out


# ── 馬場差（同日・同場・同面の他レースの中央値。leave-one-race-out）──────────────────
def loro_track(w: pd.DataFrame, resid: pd.Series, spec: V3ASpec) -> pd.Series:
    """各レースについて、同じ日・同じ場・同じ面の『他のレース』の補正前残差の中央値の平均（秒）を返す。足りなければ NaN。"""
    x = w[["race_id", "v3a_day", "v3a_oor"]].copy()
    x["res"] = resid
    x = x[x["res"].notna() & ~x["v3a_oor"]]
    rm = x.groupby("race_id").agg(day=("v3a_day", "first"), b=("res", "median"), n=("res", "size"))
    elig = rm["n"] >= spec.loro_min_horses
    g = rm[elig].groupby("day")["b"].agg(["sum", "count"])
    races = w.drop_duplicates("race_id")[["race_id", "v3a_day"]].set_index("race_id")
    s = races["v3a_day"].map(g["sum"]).fillna(0.0)
    c = races["v3a_day"].map(g["count"]).fillna(0.0)
    own_b = rm["b"].where(elig).reindex(races.index)
    s = s - own_b.fillna(0.0)
    c = c - elig.reindex(races.index).fillna(False).astype(float)
    loro = (s / c.where(c >= spec.loro_min_other_races)).rename("loro")
    return w["race_id"].map(loro)


def track_term(w: pd.DataFrame, loro: pd.Series) -> tuple[pd.Series, pd.Series]:
    use = loro.notna()
    term = loro.where(use, w["v3a_tsi_track"])
    src = pd.Series(np.where(use, "loro", np.where(w["v3a_tsi_track"] != 0, "tsi", "none")), index=w.index)
    return term, src


# ── 基準期間からモデルを作る ──────────────────────────────────────────
def _second_rows(w: pd.DataFrame, c: pd.Series) -> pd.DataFrame:
    base = w[~w["v3a_oor"] & c.notna()].copy()
    base["c"] = c[base.index]
    s2 = base[base["v3a_fp"] == 2]
    have2 = set(s2["race_id"])
    s1 = base[(base["v3a_fp"] == 1) & ~base["race_id"].isin(have2)]
    return pd.concat([s2, s1])


def _month_offsets(w2: pd.DataFrame, spec: V3ASpec) -> pd.DataFrame:
    """2 着馬の相対指数の (世代, クラス, 月) 別平均 − 50 を、月方向に平滑して縮約したオフセット（指数の点）。"""
    g = w2[w2["v3a_gen"].isin(GENS)].groupby(["v3a_gen", "v3a_cls", "v3a_month"]).agg(
        m=("rel", "mean"), n=("rel", "size")).reset_index()
    rows = []
    for (gen, cls), s in g.groupby(["v3a_gen", "v3a_cls"]):
        s = s.sort_values("v3a_month")
        sm = s["m"].rolling(spec.month_smooth, center=True, min_periods=1).mean()
        sh = (sm - BASE_INDEX) * s["n"] / (s["n"] + spec.month_shrink_k)
        for mo, v in zip(s["v3a_month"], sh):
            rows.append((gen, cls, int(mo), float(v)))
    return pd.DataFrame(rows, columns=["gen", "cls", "month", "off_pts"])


def _month_off_series(w: pd.DataFrame, mo: pd.DataFrame) -> pd.Series:
    key = w.rename(columns={"v3a_gen": "gen", "v3a_cls": "cls", "v3a_month": "month"})[["gen", "cls", "month"]]
    key = key.assign(month=key["month"].astype("Int64"))
    m = key.merge(mo.assign(month=mo["month"].astype("Int64")), on=["gen", "cls", "month"], how="left")
    m.index = w.index
    return m["off_pts"].fillna(0.0)


def _class_offsets(w: pd.DataFrame, rel: pd.Series, spec: V3ASpec) -> dict[str, float]:
    """同一馬・同一年齢で比べたクラス間差（1勝=0 に正規化）。rel = クラス内相対の指数。"""
    d = pd.DataFrame({
        "h": w["horse_id"].values, "age": _num(w, "v3a_age").values, "cls": w["v3a_cls"].values,
        "y": rel.clip(CLIP_MIN, CLIP_MAX).values,
    }).dropna()
    cnt = d.groupby("h")["y"].transform("size")
    d = d[cnt >= spec.abs_min_runs].copy()
    gamma = {c: 0.0 for c in CLASS_ORDER}
    dh = pd.Series(0.0, index=d.index)
    da = pd.Series(0.0, index=d.index)
    gcol = d["cls"].map(gamma)
    for _ in range(25):
        dh = (d["y"] - da - gcol).groupby(d["h"]).transform("mean")
        da = (d["y"] - dh - gcol).groupby(d["age"]).transform("mean")
        gm = (d["y"] - dh - da).groupby(d["cls"]).mean()
        gamma = {c: float(gm.get(c, 0.0)) for c in CLASS_ORDER}
        gcol = d["cls"].map(gamma)
    base = gamma["1勝"]
    return {c: gamma[c] - base for c in CLASS_ORDER}


def build_model(df_ref: pd.DataFrame, spec: V3ASpec, beta_pace: float, meta: dict | None = None) -> V3AModel:
    """基準期間のフレーム（compute_for_dataframe の途中結果。複数年を連結）からモデルを作る。"""
    w = prepare(df_ref, spec, beta_pace)
    w["v3a_age"] = pd.to_numeric(w["sex_age"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    track = w["v3a_tsi_track"].copy()           # 初回は TSI の補正だけ
    par = None
    for it in range(spec.n_iter):
        c = w["v3a_c0"] - track
        par = build_par_table(_second_rows(w, c), spec)
        wp = attach_par(w, par)
        resid = w["v3a_c0"] - wp["v3a_par_sec"]
        loro = loro_track(wp, resid, spec)
        track, _ = track_term(w, loro)
        logger.info("v3a build iter %d: par cells=%d, loro coverage=%.3f", it + 1, len(par), float(loro.notna().mean()))
    c = w["v3a_c0"] - track
    par = build_par_table(_second_rows(w, c), spec)
    wp = attach_par(w, par)
    rel0 = BASE_INDEX + (wp["v3a_par_sec"] - c) * POINTS_PER_SEC
    sec = _second_rows(w, c)
    w2 = w.loc[sec.index].assign(rel=rel0.loc[sec.index])
    mo = _month_offsets(w2, spec)
    rel = rel0 - _month_off_series(w, mo)
    cls_off = _class_offsets(w, rel.where(~w["v3a_oor"]), spec)
    model = V3AModel(spec=spec, par=par, month_offsets=mo, class_offsets=cls_off, meta=meta or {})
    return model


# ── 適用 ──────────────────────────────────────────────────────
def apply_model(df: pd.DataFrame, model: V3AModel, beta_pace: float) -> pd.DataFrame:
    """compute_for_dataframe の途中結果に v3a の列（V3A_COLUMNS）を付けて返す（行は増減しない）。"""
    spec = model.spec
    w = prepare(df, spec, beta_pace)
    wp = attach_par(w, model.par)
    mo_pts = _month_off_series(wp, model.month_offsets)
    par_final = wp["v3a_par_sec"] - mo_pts / POINTS_PER_SEC
    resid = wp["v3a_c0"] - par_final
    loro = loro_track(wp, resid, spec)
    term, src = track_term(wp, loro)
    corrected = wp["v3a_c0"] - term
    rel = BASE_INDEX + (par_final - corrected) * POINTS_PER_SEC
    gamma = wp["v3a_cls"].map(model.class_offsets).fillna(0.0)
    absv = rel - gamma

    out = df.copy()
    usable = rel.notna() & wp["v3a_par_sec"].notna() & corrected.notna()
    fp = wp["v3a_fp"]
    invalid = fp.isna() | (fp <= 0)
    usable = usable & ~invalid
    rel = rel.where(usable)
    absv = absv.where(usable)
    clipped = (rel < CLIP_MIN) | (rel > CLIP_MAX) | (absv < CLIP_MIN) | (absv > CLIP_MAX)
    status = pd.Series("ok", index=out.index, dtype=object)
    status[usable & clipped & ~wp["v3a_oor"]] = "clipped"
    status[usable & wp["v3a_oor"] & ~clipped] = "restored_out_of_range"
    status[usable & wp["v3a_oor"] & clipped] = "restored_clipped"
    status[wp["v3a_par_sec"].isna()] = "no_par"
    status[invalid & wp["v3a_par_sec"].notna()] = "invalid"
    low = (wp["v3a_par_n"] < spec.min_par_n) | wp["v3a_par_level"].isin(["L4_pooled"]) | wp["v3a_weight_missing"]
    low = (low | (src == "tsi") | (src == "none")) & usable   # 馬場差が同日他レースで作れない行も低信頼

    out["megu_index_v3a"] = rel
    out["megu_index_v3a_robust"] = rel.clip(CLIP_MIN, CLIP_MAX)
    out["megu_index_v3a_abs"] = absv
    out["megu_index_v3a_abs_robust"] = absv.clip(CLIP_MIN, CLIP_MAX)
    out["megu_v3a_status"] = status
    out["megu_v3a_low_confidence"] = low
    out["megu_v3a_version"] = V3A_VERSION_LABEL
    out["v3a_track_sec"] = term
    out["v3a_track_src"] = src
    out["v3a_par_sec"] = par_final
    out["v3a_par_level"] = wp["v3a_par_level"]
    out["v3a_par_n"] = wp["v3a_par_n"]
    out["v3a_corrected_sec"] = corrected
    return out
