"""成長指数（馬体重発表 T-60 時点で作る『この馬は今回、成長で指数が上がる見込みか』）。

目的変数: 今走のめぐ指数（絶対値 megu_v3a_abs_robust。1 点 = 0.1 秒）− その馬の直近水準（過去 5 走の指数を新しい順に 0.7^k で重み付けした平均）。
成長指数 = 4 つの要素の寄与の和（指数ポイント）。平均への回帰（前走の水準・走数・休養・面・距離）は目的変数に対する統制として
学習時にだけ使い、指数には含めない。

  stage   : 成長期（月齢段階 2歳前半〜5歳以上）。母集団で段階ごとに指数が上がりやすいか
  weight  : 馬体重（前走比・自分の過去平均からのずれ。若馬の増加と古馬の増加で効き方を分ける）
  oikiri  : 追い切り（自己比の良化・加速・本数・評価・厩舎コメント）
  ped     : 血統（父・母父の、その成長段階での指数/体重の増え方と晩成度。経験ベイズで縮約した値）

入力は T-60 までに分かるものだけ。関数は外部読み込みなしの純粋関数（モデルは呼び出し側が読み込んで渡す）。
2 つの実装がある:
  growth_index_model : 採用版。統制変数 + 要素の入力を、小さな木のモデル（LightGBM、300 本 × 葉 7。config/growth_index/model.txt）に入れ、
                       『統制で説明できる分を引いた、次走の指数の上がり幅』を予測する。要素ごとの寄与は SHAP を要素ごとに足したもの。
  growth_index       : 要素の寄与の和（リッジ。lightgbm なしで動く簡易版。精度は採用版の約半分）
係数・列順は config/growth_index/params.json。
学習・評価: src/research/pedigree/growth_index_fit.py。章: docs/html/modeling/chapters/weight_growth_performance.html。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

STAGES = ["2歳前半", "2歳後半", "3歳前半", "3歳後半", "4歳", "5歳以上"]
YOUNG = ("2歳前半", "2歳後半", "3歳前半", "3歳後半")
GROUPS = ("stage", "weight", "oikiri", "ped")
CTRL_NAMES = ["level_c", "level_c2", "log_nvalid", "stale_prev", "log_rest", "dist_km", "surface_da", "sex_f", "sex_g"]
LABELS = ["下降", "やや停滞", "横ばい", "成長", "大きく成長"]

# 追い切りの入力列（TL_A の標準化済み列など）。欠けたら学習平均で埋めて寄与 0
OIKIRI_NUM = ["f_t4z_vs_self", "f_t1z_vs_self", "f_accelz_vs_self", "f_t1z", "f_accelz", "best_t1z_14", "mean_t4z_14", "n_work_14", "oikiri_eval", "comment_pos", "comment_neg"]
PED_KEYS = ["sire_didx", "sire_dwt", "sire_banse", "sire_wtgain", "dam_sire_didx", "dam_sire_dwt", "dam_sire_banse", "dam_sire_wtgain"]


def stage_of(age: int | None, month: int) -> str | None:
    """成長段階。2歳は 9 月までを前半、3歳は 6 月までを前半。"""
    if age is None or age < 2:
        return None
    if age == 2:
        return "2歳前半" if month <= 9 else "2歳後半"
    if age == 3:
        return "3歳前半" if month <= 6 else "3歳後半"
    return "4歳" if age == 4 else "5歳以上"


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def ped_inputs(stage: str | None, sire: Mapping | None, dam_sire: Mapping | None) -> dict[str, float | None]:
    """父・母父の血統プロファイル（ped_profile.json の 1 頭分）から、今回の成長段階に対応する値を取り出す。
    プロファイルのキー: banse（晩成度）, wtgain（体重の伸び）, didx[<前段階>→<段階>]（段階間の指数の増え方）, dwt[...]（体重）。"""
    prev = STAGES[STAGES.index(stage) - 1] if stage in STAGES and STAGES.index(stage) > 0 else None
    out: dict[str, float | None] = {}
    for tag, p in (("sire", sire), ("dam_sire", dam_sire)):
        p = p or {}
        key = f"{prev}→{stage}" if prev else None
        out[f"{tag}_didx"] = _num((p.get("didx") or {}).get(key)) if key else None
        out[f"{tag}_dwt"] = _num((p.get("dwt") or {}).get(key)) if key else None
        out[f"{tag}_banse"] = _num(p.get("banse"))
        out[f"{tag}_wtgain"] = _num(p.get("wtgain"))
    return out


def group_features(x: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    """入力 x から要素ごとの特徴量を作る（欠損は None のまま。中心化・欠損埋めは contributions 側）。
    x の主なキー: stage, weight_change（前走比 kg）, weight_dev（体重 − 自分の過去平均 kg）, rest_days,
    oikiri 系（OIKIRI_NUM）, has_training, ped（ped_inputs の返り値）。"""
    stage = x.get("stage")
    young = 1.0 if stage in YOUNG else 0.0
    out: dict[str, dict[str, float | None]] = {"stage": {f"stage={s}": (1.0 if stage == s else 0.0) for s in STAGES}}
    dw, dev = _num(x.get("weight_change")), _num(x.get("weight_dev"))
    rest = _num(x.get("rest_days"))
    long_rest = None if rest is None else (1.0 if rest > 35 else 0.0)
    w: dict[str, float | None] = {
        "dw10": None if dw is None else dw / 10.0,
        "dw10_sq": None if dw is None else (dw / 10.0) ** 2,
        "dw10_young": None if dw is None else dw / 10.0 * young,
        "dw10_long": None if dw is None or long_rest is None else dw / 10.0 * long_rest,
        "dev10": None if dev is None else dev / 10.0,
        "dev10_sq": None if dev is None else (dev / 10.0) ** 2,
        "dev10_young": None if dev is None else dev / 10.0 * young,
        "miss_dw": 1.0 if dw is None else 0.0,
        "miss_dev": 1.0 if dev is None else 0.0,
    }
    out["weight"] = w
    o: dict[str, float | None] = {k: _num(x.get(k)) for k in OIKIRI_NUM}
    o["has_training"] = _num(x.get("has_training"))
    o["miss_tl"] = 1.0 if _num(x.get("f_t4z_vs_self")) is None else 0.0
    o["miss_oik"] = 1.0 if _num(x.get("oikiri_eval")) is None else 0.0
    out["oikiri"] = o
    ped = x.get("ped") or {}
    p: dict[str, float | None] = {k: _num(ped.get(k)) for k in PED_KEYS}
    p["miss_ped"] = 1.0 if all(p[k] is None for k in PED_KEYS) else 0.0
    out["ped"] = p
    return out


def contributions(x: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, float]:
    """要素ごとの寄与（指数ポイント）。欠損は学習平均で埋める（= 寄与 0）。"""
    feats = group_features(x)
    out: dict[str, float] = {}
    for g in GROUPS:
        spec = params["groups"][g]
        s = 0.0
        for name, mean, coef in zip(spec["names"], spec["mean"], spec["coef"]):
            v = feats[g].get(name)
            s += coef * ((mean if v is None else v) - mean)
        out[g] = s
    return out


def stage_of_index(index: float, thresholds: list[float]) -> int:
    """thresholds は昇順 4 つ（下降/やや停滞/横ばい/成長/大きく成長 の境目）。戻り値は 0〜4。"""
    return sum(1 for t in thresholds if index >= t)


def _reason(g: str, c: float, x: Mapping[str, Any]) -> str | None:
    up = c > 0
    if g == "stage":
        st = x.get("stage") or ""
        return f"{st}は指数が上がりやすい時期" if up else f"{st}は指数が上がりにくい時期"
    if g == "weight":
        dw, dev = _num(x.get("weight_change")), _num(x.get("weight_dev"))
        bits = []
        if dev is not None and abs(dev) >= 2:
            bits.append(f"自己平均より{dev:+.0f}kg")
        if dw is not None and abs(dw) >= 2:
            bits.append(f"前走比{dw:+.0f}kg")
        base = "・".join(bits) or "馬体重"
        return f"{base}（成長方向）" if up else f"{base}（成長に結びつきにくい動き）"
    if g == "oikiri":
        return "追い切りが自己比で良化" if up else "追い切りに上積みが見えない"
    if g == "ped":
        return "父・母父は、この時期に伸びやすい血統" if up else "父・母父は、この時期に伸びにくい血統"
    return None


def growth_index(x: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """成長指数と段階・要素ごとの寄与・理由をまとめる。
    返り値: {index, level(0〜4), label, contributions{...}, reasons[{group, points, text}], coverage{weight,oikiri,ped}}。"""
    c = contributions(x, params)
    idx = sum(c.values()) + float(params.get("intercept", 0.0))
    lv = stage_of_index(idx, params["thresholds"])
    feats = group_features(x)
    reasons = []
    for g, pts in sorted(c.items(), key=lambda kv: -abs(kv[1])):
        if abs(pts) < float(params.get("reason_min_points", 0.3)):
            continue
        t = _reason(g, pts, x)
        if t:
            reasons.append({"group": g, "points": round(pts, 2), "text": t})
    cov = {"weight": feats["weight"]["miss_dev"] == 0.0 or feats["weight"]["miss_dw"] == 0.0,
           "oikiri": feats["oikiri"]["miss_tl"] == 0.0 or feats["oikiri"]["miss_oik"] == 0.0, "ped": feats["ped"]["miss_ped"] == 0.0}
    return {"index": round(idx, 2), "level": lv, "label": LABELS[lv], "contributions": {k: round(v, 2) for k, v in c.items()}, "reasons": reasons[:3], "coverage": cov}


# ── 採用版: 小さな木のモデル ────────────────────────────────────────────

def control_vector(x: Mapping[str, Any], params: Mapping[str, Any]) -> list[float]:
    """統制変数（目的変数を説明するだけで、指数には含めない）。
    x: level（過去 5 走の指数を新しい順に 0.7^k で重み付けした平均。前走が 7 日以内なら前走を除く）, n_valid_prior, stale_prev(0/1),
       rest_days, distance, surface('芝'|'ダ'), sex('牡'|'牝'|'セ')。level が無い馬には指数を出さない（呼び出し側で判定）。"""
    lv = float(x["level"]) - float(params["level_mean"])
    rest = _num(x.get("rest_days"))
    rest = float(params["rest_median"]) if rest is None else rest
    dist = _num(x.get("distance"))
    return [lv, lv**2 / 100.0, math.log1p(float(x.get("n_valid_prior") or 0)), float(x.get("stale_prev") or 0), math.log(max(rest, 1.0)),
            (1.6 if dist is None else dist / 1000.0), 1.0 if x.get("surface") == "ダ" else 0.0, 1.0 if x.get("sex") == "牝" else 0.0, 1.0 if x.get("sex") == "セ" else 0.0]


def feature_names() -> list[str]:
    feats = group_features({"stage": "3歳前半"})
    return CTRL_NAMES + [f"{g}:{n}" for g in GROUPS for n in feats[g]]


def feature_vector(x: Mapping[str, Any], params: Mapping[str, Any]) -> list[float]:
    feats = group_features(x)
    row = control_vector(x, params)
    for g in GROUPS:
        row += [math.nan if v is None else v for v in feats[g].values()]
    return row


def group_slices() -> dict[str, slice]:
    feats = group_features({"stage": "3歳前半"})
    out, pos = {"ctrl": slice(0, len(CTRL_NAMES))}, len(CTRL_NAMES)
    for g in GROUPS:
        out[g] = slice(pos, pos + len(feats[g]))
        pos += len(feats[g])
    return out


def growth_index_model(x: Mapping[str, Any], params: Mapping[str, Any], booster: Any) -> dict[str, Any]:
    """採用版。booster は lightgbm.Booster（config/growth_index/model.txt）。返り値の形は growth_index と同じ + ctrl（統制による調整分）。"""
    import numpy as np

    v = np.asarray([feature_vector(x, params)], dtype=float)
    contrib = booster.predict(v, pred_contrib=True)[0]
    sl = group_slices()
    c = {g: float(contrib[sl[g]].sum()) for g in GROUPS}
    ctrl = float(contrib[sl["ctrl"]].sum())
    idx = float(contrib[:-1].sum() + contrib[-1])             # = 予測値（バイアスを含む）
    lv = stage_of_index(idx, params["thresholds"])
    reasons = []
    for g, pts in sorted(c.items(), key=lambda kv: -abs(kv[1])):
        if abs(pts) < float(params.get("reason_min_points", 0.3)):
            continue
        t = _reason(g, pts, x)
        if t:
            reasons.append({"group": g, "points": round(pts, 2), "text": t})
    feats = group_features(x)
    cov = {"weight": feats["weight"]["miss_dev"] == 0.0 or feats["weight"]["miss_dw"] == 0.0,
           "oikiri": feats["oikiri"]["miss_tl"] == 0.0 or feats["oikiri"]["miss_oik"] == 0.0, "ped": feats["ped"]["miss_ped"] == 0.0}
    return {"index": round(idx, 2), "level": lv, "label": LABELS[lv], "contributions": {**{k: round(vv, 2) for k, vv in c.items()}, "ctrl": round(ctrl, 2)},
            "reasons": reasons[:3], "coverage": cov}
