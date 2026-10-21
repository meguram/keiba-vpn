"""tp-69: めぐ指数の『無視してよい低下（excusable）』の判別と、v3b 集約の評価。ホストで実行（nice 19 で足りる軽い集計）。

python -m src.scripts.data.tp69_build signals   # 走ごとの信号・as-of の判定確率・反発ラベル → <TP69_OUT>/signals.parquet、judge.json（判別の妥当性）
python -m src.scripts.data.tp69_build pred      # 次走予測力（v3b の集約 × excusable の除外/重み下げ）→ pred.json
python -m src.scripts.data.tp69_build feat      # V5 に足す as-of 特徴量 → <TP69_OUT>/rob.parquet（tp65_feat の run がそのまま読む）
python -m src.scripts.data.tp69_build final     # 全期間（〜2025）で当てはめた最終係数 → config/megu_excusable.json（コミット用に <TP69_OUT> にも出す）

入力: <TP65_OUT>/runs.parquet（tp65_prep）、race_quality 共有表、備考フラグ remarks_flags.parquet、各年の race_result_flat（weight_change）。
設計: 評価年 Y の判定器は、Y−1 年 6/30 までの走の低い走だけで当てはめる（ラベルは Y 年 1/1 より前の次走だけから作る）。2020〜2021 年の走は判定しない（除外なし）。
低い走 = 直前 5 走（3 走以上）の中央値より尺度（MAD×1.4826）1 つ分以上低い走。2026 年は観察のみ（選択に使わない）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.megu_index import excusable as ex
from src.pipeline.megu_index import v3b
from src.scripts.data import tp65_var as t65

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP69_OUT", "/home/jovyan/work/tp69_out"))
RQ = ROOT / "data/local/modeling/shared_cache/race_quality/race_quality.parquet"
FLAGS = ROOT / "data/local/modeling/training_pipeline_v4/slowreal/remarks_flags.parquet"
TABLES = ROOT / "data/page_reference/tables"
KEY = ["race_id", "horse_id"]
LOW_Z = 1.0          # 低い走: 直前 5 走の中央値より尺度 1 つ分以上低い
EVAL_YEARS = (2022, 2023, 2024, 2025, 2026)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_runs() -> tuple[pd.DataFrame, float]:
    d = t65.load()
    d = d[d[t65.MEGU].notna()].copy()
    d = d.drop_duplicates(KEY)
    d["x_raw"] = d[t65.MEGU].astype(float)
    ref = d.loc[d["year"].isin([2020, 2021]), "x_raw"]
    scale = float(1.4826 * np.median(np.abs(ref - ref.median())))
    d["x"] = v3b.floor_by_winner(d["x_raw"], d["race_id"], d["fp"], v3b.FLOOR)
    d["low_conf"] = d["megu_v3a_low_confidence"].astype(float)
    rq = pd.read_parquet(RQ, columns=["race_id", "pace_cls", "shape_cls", "race_l3f_dev"])
    d = d.merge(rq, on="race_id", how="left")
    fl = pd.read_parquet(FLAGS, columns=["race_id", "horse_id", "trouble_rem", "slow_rem", "has_remarks"]).drop_duplicates(KEY)
    d = d.merge(fl, on=KEY, how="left")
    wc = []
    for y in range(2020, 2027):
        p = TABLES / str(y) / "race_result_flat.parquet"
        if p.exists():
            wc.append(pd.read_parquet(p, columns=["race_id", "horse_id", "weight_change"]))
    d = d.merge(pd.concat(wc).drop_duplicates(KEY), on=KEY, how="left")
    d["top3"] = ((d["fp"] >= 1) & (d["fp"] <= 3)).astype(float)
    d = d.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    return d, scale


def build_signals(d: pd.DataFrame, scale: float) -> pd.DataFrame:
    sig = ex.run_signals(d, lift=None, scale=scale)
    sig["style"] = ex.style_bucket(sig["c1_pct"])
    sig["is_low"] = (sig["own_drop"] >= LOW_Z).astype(float).where(sig["own_drop"].notna())
    # 脚質×ペース×隊列の不利は、評価年より前の年だけで作った 3 着内率の比で、年ごとに付ける（as-of）
    pace = np.full(len(d), np.nan)
    key = sig["style"].astype(str) + "|" + d["pace_cls"].astype(str) + "|" + d["shape_cls"].astype(str)
    tmp = d[["year", "top3", "pace_cls", "shape_cls"]].copy()
    tmp["style"] = sig["style"]
    for y in range(2021, 2028):
        lift = ex.fit_lift_table(tmp[tmp["year"] < y])
        m = (d["year"] == y).to_numpy()
        if lift and m.any():
            pace[m] = -np.log(key[m].map(lift).astype(float).clip(0.2, 5.0))
    sig["pace_dis"] = pace
    return sig


def labels_asof(d: pd.DataFrame, sig: pd.DataFrame, scale: float, upto: pd.Timestamp | None) -> pd.Series:
    """反発ラベル。upto を指定すると、その日より前の走だけを『次の走』として使う（学習時の先読み防止）。"""
    base = d[["horse_id", "race_id", "date", "x"]].copy()
    base["prior_med"] = sig["prior_med"]
    if upto is not None:
        base.loc[base["date"] >= upto, "x"] = np.nan
    lab = ex.rebound_label(base, scale=scale)
    if upto is not None:
        lab[(d["date"] >= upto)] = np.nan
    return lab


def cmd_signals(a) -> None:
    d, scale = load_runs()
    sig = build_signals(d, scale)
    _log(f"runs {len(d)}  scale {scale:.2f}  低い走 {int(sig['is_low'].sum())}")
    full_label = labels_asof(d, sig, scale, None)
    p = pd.Series(np.nan, index=d.index)
    models: dict[int, dict] = {}
    judge: dict = {"scale": scale, "low_z": LOW_Z, "years": {}}
    for y in EVAL_YEARS:
        cut_train = pd.Timestamp(f"{y - 1}-06-30")
        upto = pd.Timestamp(f"{y}-01-01")
        lab_y = labels_asof(d, sig, scale, upto)
        tr = (d["date"] <= cut_train) & (sig["is_low"] == 1) & lab_y.notna()
        if tr.sum() < 500:
            continue
        mdl = ex.fit_model(sig, lab_y, tr)
        base = ex.fit_model(sig, lab_y, tr, cols=["own_drop", "n_prior"])
        models[y] = mdl.to_json()
        te = (d["year"] == y) & (sig["is_low"] == 1)
        p.loc[d["year"] == y] = mdl.score(sig.loc[d["year"] == y]).to_numpy()
        pb = base.score(sig)
        ev = te & full_label.notna()
        if ev.sum() < 100:
            continue
        from sklearn.metrics import roc_auc_score

        yv = full_label[ev].astype(int)
        auc = roc_auc_score(yv, p[ev])
        auc_b = roc_auc_score(yv, pb[ev])
        thr_tr = np.quantile(mdl.score(sig.loc[tr]), [0.5, 0.65, 0.8])  # 学習期間の p の分位（除外率 50/35/20%）
        res = {"n_train": int(tr.sum()), "n_low_eval": int(ev.sum()), "rebound_rate_low": float(yv.mean()), "auc": float(auc), "auc_baseline_own_only": float(auc_b),
               "thresholds_from_train": [float(t) for t in thr_tr]}
        for name, t in zip(("excl50", "excl35", "excl20"), thr_tr):
            fl = p[ev] >= t
            res[name] = {"flag_share_of_low": float(fl.mean()), "rebound_flagged": float(yv[fl].mean()), "rebound_other": float(yv[~fl].mean()),
                         "false_exclusion": float(1 - yv[fl].mean()),  # 反発しなかったのに除外した割合
                         "flag_share_of_all_runs": float(((p >= t) & (sig["is_low"] == 1) & (d["year"] == y)).sum() / max(int((d["year"] == y).sum()), 1))}
            res[name]["lift"] = res[name]["rebound_flagged"] / max(res[name]["rebound_other"], 1e-9)
        judge["years"][str(y)] = res
        _log(f"{y}: low {int(ev.sum())} 反発率 {yv.mean():.3f}  AUC {auc:.3f}（ベースライン {auc_b:.3f}）")
    sig["p_exc"] = p
    sig["label_full"] = full_label
    keep = d[["race_id", "horse_id", "date", "year", "x", "x_raw", "fp", "top3"]].join(sig)
    OUT.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(OUT / "signals.parquet", index=False)
    (OUT / "judge.json").write_text(json.dumps({**judge, "models": models}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log("signals.parquet / judge.json saved")


def thresholds(judge: dict, year: int, name: str) -> float:
    j = judge["years"].get(str(year))
    return j["thresholds_from_train"][{"excl50": 0, "excl35": 1, "excl20": 2}[name]] if j else np.inf


def agg_variants(S: pd.DataFrame, scale: float, judge: dict) -> pd.DataFrame:
    """as-of の集約（その走より前の直近 5 走）。S は horse_id, date で並べ替え済み。変種: ewm / 加重 Huber × 低信頼の重み × excusable の除外・重み下げ。"""
    g = S["horse_id"]
    X = v3b.asof_matrix(S["x"], g)
    LC = v3b.asof_matrix(S["low_conf"].fillna(0.0), g)
    P = v3b.asof_matrix(S["p_exc"], g)
    LOW = v3b.asof_matrix(S["is_low"], g)
    # 除外判定の閾値は、その走の属する年の判定器の学習期間から決める（走の年ごと）
    thr = {n: S["year"].map(lambda y, n=n: thresholds(judge, int(y), n)).to_numpy(float) for n in ("excl50", "excl35", "excl20")}
    # 過去走ごと（列ごと）の閾値: その走の年の閾値。asof_matrix と同様にずらす
    THR = {n: v3b.asof_matrix(pd.Series(thr[n], index=S.index), g) for n in thr}
    D = v3b.run_weights()[None, :] * np.where(np.isnan(X), 0.0, 1.0)
    out = {}

    def hub(w):
        r = v3b.weighted_huber(X, w, scale)
        base = v3b.weighted_huber(X, D, scale)
        return np.where(np.isnan(r), base, r)  # 全部除外された行は除外なしに戻す（比較対象の行を揃える）

    out["wh"] = v3b.weighted_huber(X, D, scale)
    for lcw in (0.0, 0.5):
        out[f"wh_lc{lcw}"] = hub(D * np.where(LC >= 1, lcw, 1.0))
    for n in ("excl50", "excl35", "excl20"):
        flagged = (LOW >= 1) & (P >= THR[n])
        out[f"wh_{n}_hard"] = hub(D * np.where(flagged, 0.0, 1.0))
        out[f"wh_{n}_half"] = hub(D * np.where(flagged, 0.5, 1.0))
    soft = D * np.where(LOW >= 1, 1.0 - np.nan_to_num(P, nan=0.0), 1.0)
    out["wh_soft"] = hub(soft)
    flagged = (LOW >= 1) & (P >= THR["excl35"])
    out["wh_lc0.5_excl35"] = hub(D * np.where(LC >= 1, 0.5, 1.0) * np.where(flagged, 0.0, 1.0))
    # 最高値（下限適用済みの値の最大。除外あり/なし）
    Xe = np.where(flagged, np.nan, X)
    with np.errstate(all="ignore"):
        out["max5"] = np.nanmax(X, axis=1)
        mx = np.nanmax(Xe, axis=1)
    out["max5_excl35"] = np.where(np.isnan(mx), out["max5"], mx)
    out["mean5"] = np.nanmean(X, axis=1)
    me = np.nanmean(Xe, axis=1)
    out["mean5_excl35"] = np.where(np.isnan(me), out["mean5"], me)
    out["n_excl35"] = flagged.sum(axis=1).astype(float)
    return pd.DataFrame(out, index=S.index)


def cmd_pred(a) -> None:
    S = pd.read_parquet(OUT / "signals.parquet")
    judge = json.loads((OUT / "judge.json").read_text(encoding="utf-8"))
    scale = judge["scale"]
    S = S.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    A = agg_variants(S, scale, judge)
    P = pd.concat([S[["race_id", "horse_id", "date", "year", "fp"]], A.add_prefix("f_")], axis=1)
    res: dict = {"features": {}}
    cols = [c for c in P.columns if c.startswith("f_") and not c.startswith("f_n_")]
    for c in cols:
        r = t65.eval_feature(P, c, years=(2023, 2024, 2025))
        res["features"][c[2:]] = r
        _log(f"{c}: {json.dumps({y: round(v['ll_gain'], 4) for y, v in r.items()})}")
    (OUT / "pred.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log("pred.json saved")


PAIRS = [("wh_lc0.5", "wh"), ("wh_lc0.0", "wh"), ("wh_soft", "wh"), ("wh_excl20_half", "wh"), ("wh_excl35_half", "wh"), ("wh_excl20_hard", "wh"),
         ("wh_excl35_hard", "wh"), ("wh_lc0.5_excl35", "wh_lc0.5")]


def _race_gain(P: pd.DataFrame, col: str, years: tuple) -> dict:
    """eval_feature と同じ手順で、年ごとの『レースごとの勝ち馬の対数尤度の改善』を返す（対の比較用）。"""
    base = P[P[col].notna() & P["year"].between(2021, 2025)].copy()
    cnt = base.groupby("race_id")[col].transform("size")
    base = base[cnt >= 6]
    base = base[base.groupby("race_id")["fp"].transform(lambda s: (s == 1).sum()) == 1].copy()
    base["ridx"] = pd.factorize(base["race_id"])[0]
    base["y"] = (base["fp"] == 1).astype(int)
    base["n_r"] = base.groupby("ridx")["y"].transform("size")
    sc = base[col].to_numpy()
    sc = (sc - base.groupby("ridx")[col].transform("mean").to_numpy()) / 10.0
    out = {}
    for y in years:
        tr = base["year"].isin([t for t in (2022, 2023, 2024) if t != y]).to_numpy()
        te = (base["year"] == y).to_numpy()
        beta = t65.fit_beta(sc[tr], base["ridx"].to_numpy()[tr], base["y"].to_numpy()[tr])
        ll = t65.race_softmax_ll(sc[te], base["ridx"].to_numpy()[te], base["y"].to_numpy()[te], beta)
        nr = base.loc[te].groupby("ridx")["n_r"].first().reindex(ll.index)
        gain = pd.Series(ll.to_numpy() + np.log(nr.to_numpy()), index=base.loc[te].groupby("ridx")["race_id"].first().reindex(ll.index).to_numpy())
        out[y] = gain
    return out


def cmd_paired(a) -> None:
    """同じレースでの対の差（ブートストラップ 95% 区間）: v3b の各設定 − 基準。正なら改善。"""
    S = pd.read_parquet(OUT / "signals.parquet")
    judge = json.loads((OUT / "judge.json").read_text(encoding="utf-8"))
    S = S.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    A = agg_variants(S, judge["scale"], judge)
    P = pd.concat([S[["race_id", "horse_id", "date", "year", "fp"]], A.add_prefix("f_")], axis=1)
    res = {}
    cache = {}
    rng = np.random.default_rng(0)
    for c1, c0 in PAIRS:
        for c in (c1, c0):
            if c not in cache:
                cache[c] = _race_gain(P, "f_" + c, (2023, 2024, 2025))
        row = {}
        for y in (2023, 2024, 2025):
            g1, g0 = cache[c1][y], cache[c0][y]
            common = g1.index.intersection(g0.index)
            diff = (g1.loc[common] - g0.loc[common]).to_numpy()
            bs = [rng.choice(diff, len(diff)).mean() for _ in range(500)]
            row[str(y)] = {"diff": float(diff.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "races": int(len(diff))}
        res[f"{c1} − {c0}"] = row
        _log(f"{c1} − {c0}: " + json.dumps({y: round(v['diff'], 4) for y, v in row.items()}))
    (OUT / "paired.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_feat(a) -> None:
    S = pd.read_parquet(OUT / "signals.parquet")
    judge = json.loads((OUT / "judge.json").read_text(encoding="utf-8"))
    S = S.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    A = agg_variants(S, judge["scale"], judge)
    # 組 A（v3b: 下限＋低信頼の重み＋加重 Huber。excusable の除外なし）と、組 X（A に excusable の除外を足したもの）を別の接頭辞で出す
    F = pd.DataFrame({"v3bA_wh": A["wh_lc0.5"], "v3bA_max5": A["max5"], "v3bA_mean5": A["mean5"],
                      "v3bX_wh": A["wh_lc0.5_excl35"], "v3bX_max5": A["max5_excl35"], "v3bX_mean5": A["mean5_excl35"], "v3bX_nexcl5": A["n_excl35"]})
    g = S["horse_id"]
    thr = S["year"].map(lambda y: thresholds(judge, int(y), "excl35")).to_numpy(float)
    flag = ((S["is_low"] == 1) & (S["p_exc"] >= thr)).astype(float)
    F["v3bX_last_excusable"] = flag.groupby(g).shift(1).to_numpy()
    F["v3bX_last_p"] = S["p_exc"].groupby(g).shift(1).to_numpy()
    F["v3bX_gap_wh"] = F["v3bX_wh"] - F["v3bA_wh"]
    F["race_id"], F["horse_id"] = S["race_id"].to_numpy(), S["horse_id"].to_numpy()
    for c in F.columns:
        if c not in KEY:
            F[c] = F[c].astype("float32")
    F = F.drop_duplicates(KEY)
    F.to_parquet(OUT / "rob.parquet", index=False)
    _log(f"rob.parquet {F.shape} cover {float(F['v3bA_wh'].notna().mean()):.3f}")


def cmd_final(a) -> None:
    d, scale = load_runs()
    sig = build_signals(d, scale)
    upto = pd.Timestamp("2026-01-01")
    lab = labels_asof(d, sig, scale, upto)
    tr = (d["date"] <= pd.Timestamp("2025-06-30")) & (sig["is_low"] == 1) & lab.notna()
    mdl = ex.fit_model(sig, lab, tr)
    j = mdl.to_json()
    j["meta"].update({"scale": scale, "low_z": LOW_Z, "trained_through": "2025-06-30 (labels through 2025-12-31)", "signals": ex.SIGNALS})
    # 運用の除外しきい値（学習期間の p の 65% 分位 = 低い走の約 35% を除外）
    ps = mdl.score(sig.loc[tr])
    j["meta"]["threshold_excl35"] = float(np.quantile(ps, 0.65))
    j["meta"]["threshold_excl20"] = float(np.quantile(ps, 0.80))   # UI の『参考: 一過性の可能性』の表示（低い走の約 20%）
    # 運用の脚質×ペース×隊列の不利の表（2025 年までの全体）
    tmp = d[["year", "top3", "pace_cls", "shape_cls"]].copy()
    tmp["style"] = sig["style"]
    j["meta"]["lift_table"] = ex.fit_lift_table(tmp[tmp["year"] <= 2025])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "megu_excusable.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"final model saved: n_train {int(tr.sum())} thr35 {j['meta']['threshold_excl35']:.3f}")
    print(json.dumps({k: round(v, 3) for k, v in j["coef"].items()}, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["signals", "pred", "paired", "feat", "final"])
    a = ap.parse_args()
    {"signals": cmd_signals, "pred": cmd_pred, "paired": cmd_paired, "feat": cmd_feat, "final": cmd_final}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
