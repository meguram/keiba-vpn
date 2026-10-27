"""tp-80: ユーザー手順の位置取り予測・ペース予測の再現と、既存との比較・部品別アブレーション・親への特徴量。

python -m src.scripts.data.tp80_eval position   # 位置取り: 既存（pf ルール・td・前段 ML）との精度比較 + アブレーション → tp80/position.json
python -m src.scripts.data.tp80_eval pace       # ペース: 既存（pf 由来の線形）との精度比較 → tp80/pace.json
python -m src.scripts.data.tp80_eval export     # 親モデル用の特徴量 → data/local/modeling/ptrain/usr_rows.parquet（v6_eval の外部群 USR）
環境変数 TP80_OUT（既定 data/local/modeling/tp80）。評価は 2022〜2025 年（係数・λ は評価年より前の年だけで決める）。2026 年は使わない。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp80_userpos as U

ROOT = Path(__file__).resolve().parents[3]
TP = ROOT / "data/local/modeling/training_pipeline"
SMART = TP / "smartrc_ten1f"
OUT = Path(os.environ.get("TP80_OUT") or ROOT / "data/local/modeling/tp80")
EVAL_YEARS = [2022, 2023, 2024, 2025]
KEY = ["race_id", "horse_id"]
LAMS = [0.4, 0.55, 0.7, 0.85, 1.0]
KS = [3, 5, 8]


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    h = pd.read_parquet(ROOT / "data/local/modeling/tp53/horse.parquet")
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    f = pd.read_parquet(TP / "tp25a/stage_frame.parquet", columns=KEY + ["ten_actual", "ten_rel_actual", "c1_rel_actual"])
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    o = pd.read_parquet(TP / "tp25a/start_stage_oof.parquet", columns=KEY + ["ts_ten_pred", "ts_c1_pred"])
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    D = h.merge(f, on=KEY, how="left").merge(o, on=KEY, how="left")
    D["date"] = pd.to_datetime(D["date"])
    D = D.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    D["act_first"] = D["act_pass_first"].where(D["act_pass_first"] >= 1)
    D["act_pos"] = D["act_early_norm"].where(D["act_first"].notna()).clip(0, 1)
    D["gate"] = U.gate_rel(D["horse_number"], D["field_size"])
    career = pd.read_parquet(SMART / "career.parquet")
    career = U.add_race_relative(career)
    return D, career


def race_z(race: np.ndarray, x: np.ndarray) -> np.ndarray:
    d = pd.DataFrame({"r": race, "x": x})
    g = d.groupby("r")["x"]
    sd = g.transform("std").replace(0, np.nan)
    return ((d["x"] - g.transform("mean")) / sd).to_numpy()


def histories(D: pd.DataFrame, career: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "ten": U.history_matrix(D, career, "ten"),
        "trel": U.history_matrix(D, career, "ten_rel"),
        "c1": U.history_matrix(D, career, "c1_rel"),
    }


# ───────────────────────── 指標
def _race_spearman(race: np.ndarray, pred_rank: np.ndarray, actual: np.ndarray) -> float:
    d = pd.DataFrame({"r": race, "p": pred_rank.astype("float64"), "a": actual}).dropna()
    d = d[d.groupby("r")["r"].transform("size") >= 5]
    d["ra"] = d.groupby("r")["a"].rank(method="average")
    d["rp"] = d.groupby("r")["p"].rank(method="average")
    for c in ("ra", "rp"):
        d[c] = d[c] - d.groupby("r")[c].transform("mean")
    num = (d["ra"] * d["rp"]).groupby(d["r"]).sum()
    den = np.sqrt((d["ra"] ** 2).groupby(d["r"]).sum() * (d["rp"] ** 2).groupby(d["r"]).sum())
    ok = den > 0
    return float((num[ok] / den[ok]).mean())


def metrics(D: pd.DataFrame, pred_rank: np.ndarray, mask: np.ndarray) -> dict:
    d = D.loc[mask, ["race_id", "field_size", "act_first", "act_pos", "ten_rel_actual"]].copy()
    d["rank"] = pred_rank[mask]
    out = {"n_rows": int(len(d)), "n_races": int(d["race_id"].nunique())}
    out["sp_c1"] = _race_spearman(d["race_id"].to_numpy(), d["rank"].to_numpy(), d["act_pos"].to_numpy())
    out["sp_ten"] = _race_spearman(d["race_id"].to_numpy(), d["rank"].to_numpy(), d["ten_rel_actual"].to_numpy())
    v = d.dropna(subset=["act_pos", "rank"])
    pos = np.where(v["field_size"] > 1, (v["rank"] - 1) / np.maximum(v["field_size"] - 1, 1), 0.5)
    pc, ac = U.style_class(pos), U.style_class(v["act_pos"].to_numpy())
    out["class_acc"] = float((pc == ac).mean())
    for k, nm in enumerate(["nige", "senko", "sashi", "oikomi"]):
        tp = float(((pc == k) & (ac == k)).sum())
        out[f"{nm}_prec"] = tp / max(float((pc == k).sum()), 1.0)
        out[f"{nm}_rec"] = tp / max(float((ac == k).sum()), 1.0)
    top = v[v["rank"] == 1]
    out["leader_hit"] = float((top["act_first"] == 1).mean())
    out["leader_top3"] = float((top["act_first"] <= 3).mean())
    r2 = v[v["rank"] <= 2]
    out["actual_leader_in_pred_top2"] = float(r2.groupby("race_id")["act_first"].min().eq(1).mean())
    return out


def by_year(D: pd.DataFrame, pred_rank: np.ndarray, years=EVAL_YEARS) -> dict:
    res = {}
    for y in years:
        res[str(y)] = metrics(D, pred_rank, (D["year"] == y).to_numpy())
    res["2022-24"] = metrics(D, pred_rank, D["year"].isin([2022, 2023, 2024]).to_numpy())
    return res


# ───────────────────────── 想定
def fitted_score(D: pd.DataFrame, H: dict, *, kind: str, lam: float, k: int, gate: bool, fit_years_fn=None, kappa: float = 300.0) -> np.ndarray:
    """評価年ごとに『評価年より前の年』で当てはめた枠補正を使った想定スコア（小さいほど前）。kind: ten=生のテン1F / trel=同レース内の相対テン。"""
    w = U.weighted_mean(H[kind], lam, k)
    if not gate:
        return w
    tgt = D["ten_actual"].to_numpy() if kind == "ten" else D["ten_rel_actual"].to_numpy()
    out = w.copy()
    g = D["gate"].to_numpy()
    ven, sf, dist = D["venue"].to_numpy(), D["surface"].to_numpy(), D["distance"].to_numpy()
    years = D["year"].to_numpy()
    for y in sorted(set(years)):
        train = (years < y) if y > 2021 else (years <= 2021)
        res = tgt - w
        ok = train & ~np.isnan(res)
        coef = U.fit_gate_coef(res[ok], g[ok], ven[ok], sf[ok], dist[ok], kappa)
        m = years == y
        out[m] = w[m] + U.gate_adjust(coef, g[m], ven[m], sf[m], dist[m])
    return out


def choose_lam_k(D: pd.DataFrame, H: dict, kind: str) -> tuple[float, int, list]:
    """λ と本数 k は 2020〜2021 年の位置取りの順位相関だけで選ぶ（評価年は使わない）。"""
    m = (D["year"] <= 2021).to_numpy()
    rows = []
    for k in KS:
        for lam in LAMS:
            s = U.weighted_mean(H[kind], lam, k)
            rk = U.order_in_race(D["race_id"].to_numpy(), s, D["horse_number"].to_numpy())
            sp = _race_spearman(D.loc[m, "race_id"].to_numpy(), rk[m], D.loc[m, "act_pos"].to_numpy())
            rows.append({"lam": lam, "k": k, "sp_c1_train": sp})
    best = max(rows, key=lambda r: r["sp_c1_train"])
    return best["lam"], best["k"], rows


def cmd_position() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D, career = load()
    H = histories(D, career)
    rid, hn = D["race_id"].to_numpy(), D["horse_number"].to_numpy()
    res: dict = {"data": {"rows": int(len(D)), "ten_hist_coverage": float((~np.isnan(H["ten"][:, 0])).mean()),
                           "c1_hist_coverage": float((~np.isnan(H["c1"][:, 0])).mean())}}
    lam_t, k_t, grid_t = choose_lam_k(D, H, "ten")
    lam_r, k_r, grid_r = choose_lam_k(D, H, "trel")
    res["chosen"] = {"ten": {"lam": lam_t, "k": k_t}, "trel": {"lam": lam_r, "k": k_r}, "grid_ten": grid_t, "grid_trel": grid_r}
    _log(f"選択（2020-21）: ten λ={lam_t} k={k_t} / trel λ={lam_r} k={k_r}")

    def rank_of(score, fill="median", tie=True):
        return U.order_in_race(rid, score, hn, fill=fill, tiebreak=tie)

    comp = {}
    comp["既存: pf ルール（通過順の加重＋枠補正）"] = by_year(D, rank_of(D["pf_early"].to_numpy()))
    comp["既存: td_early"] = by_year(D, rank_of(D["td_early"].to_numpy()))
    comp["既存: 前段 ML（c1 予測, 馬の特徴量多数）"] = by_year(D, rank_of(D["ts_c1_pred"].to_numpy()))
    comp["既存: 前段 ML（テン予測）"] = by_year(D, rank_of(D["ts_ten_pred"].to_numpy()))
    # ユーザー手順（本線 = 生テン1F, 加重平均, 枠補正, 同値は内枠が前, 履歴なしは中央値扱い）
    s_main = fitted_score(D, H, kind="ten", lam=lam_t, k=k_t, gate=True)
    comp["ユーザー手順（生テン）"] = by_year(D, rank_of(s_main))
    s_main_r = fitted_score(D, H, kind="trel", lam=lam_r, k=k_r, gate=True)
    comp["ユーザー手順（同レース内の相対テン）"] = by_year(D, rank_of(s_main_r))
    res["compare"] = comp
    # アブレーション（本線 = 生テン）
    ab = {}
    ab["本線"] = by_year(D, rank_of(s_main))
    ab["加重なし（λ=1）"] = by_year(D, rank_of(fitted_score(D, H, kind="ten", lam=1.0, k=k_t, gate=True)))
    ab["直近1走のみ"] = by_year(D, rank_of(fitted_score(D, H, kind="ten", lam=1.0, k=1, gate=True)))
    ab["枠補正なし"] = by_year(D, rank_of(fitted_score(D, H, kind="ten", lam=lam_t, k=k_t, gate=False)))
    ab["同値の並び: 乱数（内枠優先なし）"] = by_year(D, rank_of(s_main, tie=False))
    ab["履歴なしは最後尾"] = by_year(D, rank_of(s_main, fill="back"))
    ab["履歴なしは先頭側"] = by_year(D, rank_of(s_main, fill="front"))
    # 履歴なしを c1 の過去（通過位置）で補う
    wc1 = U.weighted_mean(H["c1"], lam_t, k_t)
    tr = (D["year"] <= 2021).to_numpy() & ~np.isnan(wc1) & ~np.isnan(s_main)
    A = np.vstack([np.ones(tr.sum()), wc1[tr]]).T
    beta = np.linalg.lstsq(A, s_main[tr], rcond=None)[0]
    s_fill = np.where(np.isnan(s_main) & ~np.isnan(wc1), beta[0] + beta[1] * wc1, s_main)
    ab["履歴なしは通過位置で補う"] = by_year(D, rank_of(s_fill))
    # テン履歴が薄い馬（直近の有効テンが 3 本未満）の扱い
    n_ten = (~np.isnan(H["ten"])).sum(axis=1)
    # 拡張（ユーザー手順 + 通過位置の履歴のブレンド。重みは 2020〜2021 年の位置取り順位相関だけで選ぶ）
    ext = {}
    z_t, z_c = race_z(rid, s_main_r), race_z(rid, wc1)
    tr_m = (D["year"] <= 2021).to_numpy()
    cands = {}
    for wgt in [0.0, 0.25, 0.5, 0.75, 1.0]:
        zs = np.stack([z_t, z_c]); ww = np.array([[wgt], [1 - wgt]]); ok = ~np.isnan(zs)
        den = (ok * ww).sum(axis=0)
        sc = np.where(den > 0, (np.where(ok, zs, 0.0) * ww).sum(axis=0) / np.where(den > 0, den, 1), np.nan)
        rk = rank_of(sc, fill="back")
        cands[wgt] = (sc, _race_spearman(rid[tr_m], rk[tr_m], D.loc[tr_m, "act_pos"].to_numpy()))
    wbest = max(cands, key=lambda x: cands[x][1])
    ext["ブレンド重みの選択(2020-21)"] = {str(k): v[1] for k, v in cands.items()}
    ext["相対テンのみ・履歴なしは最後尾"] = by_year(D, rank_of(s_main_r, fill="back"))
    ext[f"ブレンド(相対テン {wbest} + 通過位置 {1 - wbest}), 履歴なしは最後尾"] = by_year(D, rank_of(cands[wbest][0], fill="back"))
    res["extension"] = ext
    comp["拡張: ブレンド"] = ext[f"ブレンド(相対テン {wbest} + 通過位置 {1 - wbest}), 履歴なしは最後尾"]
    ab["_カバレッジ: テン履歴0本の割合"] = {"all": float((n_ten == 0).mean()), "by_year": {str(y): float((n_ten[D["year"].to_numpy() == y] == 0).mean()) for y in EVAL_YEARS}}
    res["ablation"] = ab
    # 距離・面別（本線 vs 既存 ML）
    seg = {}
    for nm, mk in {"芝": D["surface"].eq("芝"), "ダート": D["surface"].eq("ダート"), "短距離(<=1400)": D["distance"] <= 1400,
                    "中距離(1500-2000)": D["distance"].between(1500, 2000), "長距離(>=2100)": D["distance"] >= 2100}.items():
        mm = mk.to_numpy() & D["year"].isin([2022, 2023, 2024]).to_numpy()
        seg[nm] = {"ユーザー手順": metrics(D, rank_of(s_main), mm), "既存ML": metrics(D, rank_of(D["ts_c1_pred"].to_numpy()), mm), "pfルール": metrics(D, rank_of(D["pf_early"].to_numpy()), mm)}
    res["segments"] = seg
    (OUT / "position.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    _log("position.json を書きました")
    _print_compare(comp, ab)


def _print_compare(comp: dict, ab: dict) -> None:
    cols = ["sp_c1", "sp_ten", "class_acc", "nige_prec", "nige_rec", "leader_hit", "leader_top3", "actual_leader_in_pred_top2"]
    print("\n== 比較 2022-24 =="); print(pd.DataFrame({k: {c: v["2022-24"][c] for c in cols} for k, v in comp.items()}).T.round(4).to_string())
    print("\n== 比較 2025 =="); print(pd.DataFrame({k: {c: v["2025"][c] for c in cols} for k, v in comp.items()}).T.round(4).to_string())
    print("\n== アブレーション 2022-24 =="); print(pd.DataFrame({k: {c: v["2022-24"][c] for c in cols} for k, v in ab.items() if "2022-24" in v}).T.round(4).to_string())


# ───────────────────────── ペース
def _blend(D, H, rid, lam, k, wbest):
    wc1 = U.weighted_mean(H["c1"], lam, k)
    s_r = fitted_score(D, H, kind="trel", lam=lam, k=k, gate=True)
    z_t, z_c = race_z(rid, s_r), race_z(rid, wc1)
    zs = np.stack([z_t, z_c]); ww = np.array([[wbest], [1 - wbest]]); ok = ~np.isnan(zs)
    den = (ok * ww).sum(axis=0)
    return np.where(den > 0, (np.where(ok, zs, 0.0) * ww).sum(axis=0) / np.where(den > 0, den, 1), np.nan), s_r


def race_features(D: pd.DataFrame, H: dict, score: np.ndarray, ten_hat: np.ndarray, rank: np.ndarray) -> pd.DataFrame:
    """ユーザー手順のペース部品（レース単位）。先頭想定馬の過去の前半ラップと、先行争いの相手の強さ。"""
    d = pd.DataFrame({"race_id": D["race_id"].to_numpy(), "rank": rank, "ten": ten_hat, "zf": D["_zf"].to_numpy(), "za": D["_za"].to_numpy()})
    lead = d[d["rank"] == 1].drop_duplicates("race_id").set_index("race_id")
    g = d.groupby("race_id")
    f = pd.DataFrame(index=lead.index)
    f["u_leader_ten"] = lead["ten"]
    f["u_leader_zf"] = lead["zf"].fillna(lead["za"])
    d = d.join(lead["ten"].rename("lead_ten"), on="race_id")
    d["gap"] = d["ten"] - d["lead_ten"]
    f["u_n_close03"] = d[(d["gap"] <= 0.3) & (d["rank"] > 1)].groupby("race_id").size().reindex(f.index).fillna(0)
    f["u_n_close05"] = d[(d["gap"] <= 0.5) & (d["rank"] > 1)].groupby("race_id").size().reindex(f.index).fillna(0)
    d["pr"] = np.exp(-d["gap"].clip(lower=0) / 0.3) * (d["rank"] > 1)
    f["u_pressure"] = d.groupby("race_id")["pr"].sum().reindex(f.index).fillna(0)
    second = d[d["rank"] == 2].drop_duplicates("race_id").set_index("race_id")["gap"]
    f["u_gap2"] = second.reindex(f.index)
    f["u_known"] = g["ten"].apply(lambda x: x.notna().mean()).reindex(f.index)
    return f.reset_index()


def prepare_pace(D: pd.DataFrame, career: pd.DataFrame, H: dict):
    R = pd.read_parquet(ROOT / "data/local/modeling/tp53/race.parquet")
    R["race_id"] = R["race_id"].astype(str)
    R["date"] = pd.to_datetime(R["date"])
    key = ["venue", "surface", "distance", "going_group"]
    base = R[R["year"] <= 2021]
    st = base.groupby(key)["act_t3f"].agg(["median", "std", "count"]).reset_index()
    R = R.merge(st, on=key, how="left")
    R["band"] = U.dist_band(R["distance"].to_numpy())
    base = base.assign(band=U.dist_band(base["distance"].to_numpy()))
    st2 = base.groupby(["surface", "band", "going_group"])["act_t3f"].agg(["median", "std"]).reset_index().rename(columns={"median": "median2", "std": "std2"})
    R = R.merge(st2, on=["surface", "band", "going_group"], how="left")
    use1 = R["count"].fillna(0) >= 8
    R["med"] = np.where(use1, R["median"], R["median2"])
    R["sd"] = np.where(use1, R["std"], R["std2"])
    R["z_act"] = (R["act_t3f"] - R["med"]) / R["sd"].replace(0, np.nan)
    D["_za"] = D["race_id"].map(R.set_index("race_id")["z_act"]).to_numpy()
    D["_zf"] = np.where(D["act_pos"].to_numpy() <= 0.25, D["_za"].to_numpy(), np.nan)
    c2 = D[["horse_id", "date", "_za", "_zf"]].copy()
    c2["rdate"] = c2["date"].dt.strftime("%Y%m%d").astype(int)
    HZ = {"zf": U.history_matrix(D, c2, "_zf"), "za": U.history_matrix(D, c2, "_za")}
    return R, HZ


def pace_table(D, H, HZ, R, lam_t, k_t, lam_r, k_r, wbest):
    rid, hn = D["race_id"].to_numpy(), D["horse_number"].to_numpy()
    s_ten = fitted_score(D, H, kind="ten", lam=lam_t, k=k_t, gate=True)
    blend, s_r = _blend(D, H, rid, lam_r, k_r, wbest)
    rank = U.order_in_race(rid, blend, hn, fill="back")
    D2 = D.copy()
    D2["_zf"] = np.where(np.isnan(U.weighted_mean(HZ["zf"], 0.7, 8)), U.weighted_mean(HZ["za"], 0.7, 8), U.weighted_mean(HZ["zf"], 0.7, 8))
    D2["_za"] = U.weighted_mean(HZ["za"], 0.7, 8)
    rf = race_features(D2, H, blend, s_r, rank)        # 想定先頭馬の過去の前半ラップ z と、先行争いの相手（相対テンの差）
    T = R.merge(rf, on="race_id", how="left")
    T["turf"] = (T["surface"] == "芝").astype(float)
    T["dist_k"] = T["distance"] / 1000.0
    T["heavy"] = T["going_group"].astype(str).isin(["重", "不良", "heavy", "道悪"]).astype(float)
    T["fs"] = T["field_size"].astype(float)
    return T, rank, blend, s_r


FEATS = {
    "既存系(pf): 逃げ数・先行数・圧力": ["pf_front", "pf_stalker", "pf_pressure", "pf_known", "fs", "turf", "dist_k", "heavy"],
    "ユーザー手順: 先頭馬の前半ラップ+先行争い": ["u_leader_zf", "u_n_close03", "u_n_close05", "u_pressure", "u_gap2", "u_known", "fs", "turf", "dist_k", "heavy"],
    "ユーザー手順 + 既存系": ["u_leader_zf", "u_n_close03", "u_n_close05", "u_pressure", "u_gap2", "u_known", "pf_front", "pf_stalker", "pf_pressure", "pf_known", "fs", "turf", "dist_k", "heavy"],
    "コントロールのみ(頭数・面・距離・馬場)": ["fs", "turf", "dist_k", "heavy"],
}


def _fit_predict(T: pd.DataFrame, cols: list[str], year: int, ridge: float = 1.0) -> np.ndarray:
    tr = T[(T["year"] < year) if year > 2021 else (T["year"] <= 2021)]
    tr = tr.dropna(subset=["z_act"])
    X = tr[cols].astype(float)
    mu = X.mean()
    X = X.fillna(mu)
    sdv = X.std().replace(0, 1)
    A = np.hstack([np.ones((len(X), 1)), ((X - mu) / sdv).to_numpy()])
    reg = ridge * np.eye(A.shape[1]); reg[0, 0] = 0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ tr["z_act"].to_numpy())
    te = T[T["year"] == year]
    Xt = te[cols].astype(float).fillna(mu)
    return np.hstack([np.ones((len(Xt), 1)), ((Xt - mu) / sdv).to_numpy()]) @ beta


def cmd_pace() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D, career = load()
    H = histories(D, career)
    pos = json.loads((OUT / "position.json").read_text(encoding="utf-8"))
    lam_t, k_t = pos["chosen"]["ten"]["lam"], pos["chosen"]["ten"]["k"]
    lam_r, k_r = pos["chosen"]["trel"]["lam"], pos["chosen"]["trel"]["k"]
    wbest = float(max(pos["extension"]["ブレンド重みの選択(2020-21)"].items(), key=lambda kv: kv[1])[0])
    R, HZ = prepare_pace(D, career, H)
    T, rank, blend, s_r = pace_table(D, H, HZ, R, lam_t, k_t, lam_r, k_r, wbest)
    res = {"n_races": int(len(T)), "coverage_leader_zf": float(T["u_leader_zf"].notna().mean())}
    qs = T[T["year"] <= 2021]["z_act"].quantile([1 / 3, 2 / 3]).to_numpy()
    out = {}
    for name, cols in FEATS.items():
        rows = {}
        pred_all = []
        for y in EVAL_YEARS:
            te = T[T["year"] == y]
            p = _fit_predict(T, cols, y)
            ok = te["z_act"].notna().to_numpy()
            a = te["z_act"].to_numpy()[ok]; pp = p[ok]
            cls_a = np.digitize(a, qs); cls_p = np.digitize(pp, qs)
            rows[str(y)] = {"pearson": float(np.corrcoef(a, pp)[0, 1]), "spearman": float(pd.Series(a).corr(pd.Series(pp), method="spearman")),
                            "rmse": float(np.sqrt(np.mean((a - pp) ** 2))), "rmse_const": float(np.sqrt(np.mean((a - a.mean()) ** 2))), "tercile_acc": float((cls_a == cls_p).mean()), "n": int(ok.sum())}
        out[name] = rows
    for nm, col in {"既存: pf_level(数値化) ": "pf_pressure", "既存: td_pace_index": "td_pace_index", "ユーザー: 先行争いの圧力": "u_pressure", "ユーザー: 先頭馬の前半ラップ z": "u_leader_zf"}.items():
        rows = {}
        for y in EVAL_YEARS:
            te = T[(T["year"] == y)].dropna(subset=["z_act", col])
            rows[str(y)] = {"spearman": float(te["z_act"].corr(te[col], method="spearman")), "n": int(len(te))}
        out["単独: " + nm] = rows
    res["models"] = out
    (OUT / "pace.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for name, rows in out.items():
        print(name, {y: {k: round(v, 3) for k, v in r.items() if k != "n"} for y, r in rows.items() if y in ("2022", "2023", "2024", "2025")})
    T.to_parquet(OUT / "pace_table.parquet", index=False)


# ───────────────────────── 親モデル用の特徴量
def cmd_export() -> None:
    D, career = load()
    H = histories(D, career)
    pos = json.loads((OUT / "position.json").read_text(encoding="utf-8"))
    lam_t, k_t = pos["chosen"]["ten"]["lam"], pos["chosen"]["ten"]["k"]
    lam_r, k_r = pos["chosen"]["trel"]["lam"], pos["chosen"]["trel"]["k"]
    wbest = float(max(pos["extension"]["ブレンド重みの選択(2020-21)"].items(), key=lambda kv: kv[1])[0])
    R, HZ = prepare_pace(D, career, H)
    T, rank, blend, s_r = pace_table(D, H, HZ, R, lam_t, k_t, lam_r, k_r, wbest)
    rid, hn = D["race_id"].to_numpy(), D["horse_number"].to_numpy()
    s_ten = fitted_score(D, H, kind="ten", lam=lam_t, k=k_t, gate=True)
    r_main = U.order_in_race(rid, s_ten, hn, fill="back")
    pos_main = np.where(D["field_size"] > 1, (r_main - 1) / np.maximum(D["field_size"] - 1, 1), 0.5)
    pos_bl = np.where(D["field_size"] > 1, (rank - 1) / np.maximum(D["field_size"] - 1, 1), 0.5)
    X = pd.DataFrame({"race_id": D["race_id"], "horse_id": D["horse_id"]})
    X["u_ten_hat"] = s_ten
    X["u_trel_hat"] = s_r
    X["u_rank"] = r_main.astype("float32")
    X["u_pos"] = pos_main.astype("float32")
    X["u_class"] = U.style_class(pos_main).astype("float32")
    X["u_blend_pos"] = pos_bl.astype("float32")
    X["u_blend_class"] = U.style_class(pos_bl).astype("float32")
    lead_ten = pd.Series(s_ten).groupby(D["race_id"].to_numpy()).transform("min")
    X["u_gap_to_leader"] = s_ten - lead_ten.to_numpy()
    cols = ["u_leader_zf", "u_n_close03", "u_n_close05", "u_pressure", "u_gap2"]
    X = X.merge(T[["race_id"] + cols], on="race_id", how="left")
    ph = np.full(len(T), np.nan)
    names = FEATS["ユーザー手順: 先頭馬の前半ラップ+先行争い"]
    for y in sorted(T["year"].unique()):
        if y > 2025:
            continue
        m = (T["year"] == y).to_numpy()
        ph[m] = _fit_predict(T, names, int(y))
    X = X.merge(pd.DataFrame({"race_id": T["race_id"], "u_pace_hat": ph}), on="race_id", how="left")
    X = X.rename(columns={"u_leader_zf": "u_leader_pace_z"})
    for c in X.columns:
        if c not in ("race_id", "horse_id"):
            X[c] = X[c].astype("float32")
    p = ROOT / "data/local/modeling/ptrain"
    p.mkdir(parents=True, exist_ok=True)
    X.to_parquet(p / "usr_rows.parquet", index=False)
    _log(f"usr_rows.parquet {X.shape}")


if __name__ == "__main__":
    {"position": cmd_position, "pace": cmd_pace, "export": cmd_export}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
