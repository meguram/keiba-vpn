"""ダートの牝系 5 代の候補の深掘り（統括役の 11/26 13:40 の依頼。開発の行・探す側・未確認・買い方に使わない）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=4 nice -n 19 python -m src.research.pedigree_lab.dirt_deep <anc|v5bag|scenes|popsire>

- anc: 祖先ごとの寄与（ダートの行での「血量の割合 × その祖先の適性」の和）の上位・下位と、祖先の名前・最後の適性・子孫の走りの数。
- v5bag: めぐ AI の v5bag（年ごとに前へ進めて学んだ予測。時間の順）と市場の両方に足して残るか。
- scenes: 効く場面の地図（距離帯・レースの年齢・ダートの経験・季節・馬場）。場面ごとに別々に当てた市場ありの γ と t（記述・同時の区間なし）。
  列は場面ごとに標準化し直すので、場面どうしで比べられるのは t だけ。ダートの経験の場面は 2014 年以降の生まれに限る（2016 年より前の走りが無いため）。
- popsire: 父の適性（--aspect の観点。既定はダート）の、人気の帯ごとの市場ありの γ（人気薄で織り込みが弱いか）。
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import aptitude as ap_
from src.research.pedigree_lab import aptitude_checks as ck
from src.research.pedigree_lab import data
from src.research.pedigree_lab.explore2 import fit_boot
from src.research.pedigree_lab.explore5 import sim_ci
from src.research.pedigree_lab.aptitude_period import attach_part

V5BAG = "/home/jovyan/work/megu_cycles_out/rich_pred_v5bag.parquet"
PED_JSON = "/home/jovyan/work/keiba-vpn/data/local/horse_pedigree_5gen"


def setup(aspect: str = "dirt", null_seed: int | None = None):
    F = ap_.base_frame()
    if aspect == "turf":                       # 芝の側の対（aptitude.py は凍結なので、ここで群を作る。dirt と同じく芝ダで分けない）
        A, keys = (F["surf"] == "T").to_numpy(), []
    else:
        A, keys = ap_.group_a(F, aspect)
    L = ap_.maternal_long(F, drop_common=False, null_seed=null_seed)
    F["z_sire"] = ap_.apt_by(F, A, ["sire"] + keys)
    F["z_bms"] = ap_.apt_by(F, A, ["ds"] + keys)
    F["z_mat5"], F["z_lv5"] = ck.mat_cols(F, A, keys, L)
    F["_A"] = A
    F["_nsurf"] = F.sort_values("date").groupby(["horse_id", "surf"]).cumcount().reindex(F.index).to_numpy()
    return F, A, keys, L


def names(ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(glob.glob(f"{PED_JSON}/*/*.json"), reverse=True):
        if len(out) >= len(ids):
            break
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for a in d.get("ancestors", []):
            i = a.get("horse_id")
            if i in ids and i not in out:
                out[i] = a.get("name", "")
    return out


def run_anc(F, A, keys, L) -> dict:
    row = L["row"].to_numpy()
    mA0, mT0 = ap_.base_means(F, A, keys)
    d = ap_.apt_by(L, A[row], ["anc"] + keys, base=(mA0[row], mT0[row]))
    dev = F["split"].isin([0, 1]).to_numpy() & A
    wsum = pd.Series(L["bw"].to_numpy()).groupby(row).transform("sum").to_numpy()
    share = L["bw"].to_numpy() / wsum
    T = pd.DataFrame({"anc": L["anc"], "c": share * d, "share": share, "d": d, "date": L["date"]})[dev[row]]
    g = T.groupby("anc").agg(contrib=("c", "sum"), rows=("c", "size"), share=("share", "mean"))
    last = T.sort_values("date").groupby("anc")["d"].last()
    g["apt_last"] = last
    g["abs"] = g["contrib"].abs()
    top = g.sort_values("contrib", ascending=False).head(15)
    bot = g.sort_values("contrib").head(10)
    nm = names(set(top.index) | set(bot.index))
    f = lambda t: [{"id": i, "name": nm.get(i, ""), "contrib": round(float(r.contrib), 2), "rows": int(r.rows),
                    "share": round(float(r.share), 3), "apt_last": round(float(r.apt_last), 4)} for i, r in t.iterrows()]
    return {"top": f(top), "bottom": f(bot), "n_anc": int(len(g))}


def dirt_rows(F) -> pd.DataFrame:
    """観点の群 A（_A）の開発の行（名前はダートの前提のままだが、--aspect の観点の行を返す）。"""
    X = attach_part(F, "dev")
    S = X[X["_A"].astype(bool)].reset_index(drop=True)
    rid = S["race_id"].to_numpy()
    S["_lqc"] = S["lq"].to_numpy() - pd.Series(S["lq"].to_numpy()).groupby(rid).transform("mean").to_numpy()
    S["_rk"] = S.groupby("race_id")["odds"].rank(method="first").to_numpy()
    return S


def zs(S: pd.DataFrame) -> dict[str, np.ndarray]:
    Z = {k: ap_.zA(S, S[f"z_{k}"].to_numpy()) for k in ("sire", "bms", "mat5", "lv5")}
    Z["ctrl"] = ap_.zA(S, S["sire_all_fpct"].fillna(0).to_numpy())
    return Z


def run_v5bag(F, n_boot: int) -> dict:
    S = dirt_rows(F)
    V = pd.read_parquet(V5BAG)[["race_id", "horse_id", "pr_win"]].dropna()
    n0 = S.groupby("race_id").size()
    S = S.merge(V, on=["race_id", "horse_id"], how="inner")
    n1 = S.groupby("race_id").size()
    S = S[S["race_id"].map(n1 == n0.reindex(n1.index)).to_numpy()]          # v5bag の予測が全頭にあるレースだけ
    S = S.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
    S["lv"] = np.log(np.clip(S["pr_win"] / S.groupby("race_id")["pr_win"].transform("sum"), 1e-5, 1))
    Z = zs(S)
    res = {"rows": int(len(S)), "races": int(S["race_id"].nunique())}
    for tag, base in {"v5bag": [S["lv"]], "v5bag_market": [S["lv"], S["lq"]], "v5bag_market_lp": [S["lv"], S["lq"], S["lp"]]}.items():
        M = np.c_[tuple(b.to_numpy() for b in base) + (Z["ctrl"], Z["lv5"], Z["sire"], Z["bms"], Z["mat5"])]
        nb = len(base)
        idx = {"sire": nb + 2, "bms": nb + 3, "mat5": nb + 4}
        b, bs = fit_boot(S, M, n_boot)
        c, r = sim_ci(b, bs, idx)
        res[tag] = {"c": round(c, 3), **{k: {"gamma": v["gamma"], "sim_ci": v["sim_ci"]} for k, v in r.items()}}
        print(tag, json.dumps(res[tag], ensure_ascii=False), flush=True)
    return res


CALIB = {"on": False}


def calib_cols(S: pd.DataFrame) -> list[np.ndarray]:
    """市場の確率の曲がり（較正）を足す列: 中心化した lq の 2 乗・人気の帯の印・lq × 帯・道悪 × lq（--calib のときだけ）。
    lq の中心化と人気の順位は、場面で絞る前の全頭で作った列（_lqc・_rk。dirt_rows）を使う（critic 4 回目の 4）。
    道悪の印がどのレースでも同じ値（道悪だけ・良だけの場面や観点）なら、道悪の 2 列は lq・lq の 2 乗と共線になるので入れない（critic 4 回目の 1〜3）。"""
    if not CALIB["on"]:
        return []
    lqc, rk = S["_lqc"].to_numpy(), S["_rk"].to_numpy()
    out = [lqc ** 2]
    for lo, hi in ((4, 6), (7, 9), (10, 99)):
        mk = ((rk >= lo) & (rk <= hi)).astype(float)
        out += [mk, lqc * mk]
    wet = S["going"].isin(["S", "W"]).to_numpy().astype(float)
    if 0 < wet.mean() < 1:
        out += [lqc * wet, lqc ** 2 * wet]          # 道悪で市場の曲がりが違う分（統括役の (4)）
    return out


def run_basic(F, n_boot: int) -> dict:
    S = dirt_rows(F)
    Z = zs(S)
    res = {"rows": int(len(S))}
    for market in (False, True):
        base = [S["lp"].to_numpy(), S["lq"].to_numpy()] if market else [S["lp"].to_numpy()]
        M = np.c_[tuple(base) + (Z["ctrl"], Z["lv5"], Z["sire"], Z["bms"], Z["mat5"]) + tuple(calib_cols(S) if market else [])]
        nb = len(base)
        idx = {"sire": nb + 2, "bms": nb + 3, "mat5": nb + 4}
        b, bs = fit_boot(S, M, n_boot)
        c, r = sim_ci(b, bs, idx)
        res["mkt" if market else "nomkt"] = {"c": round(c, 3), **{k: {"gamma": v["gamma"], "sim_ci": v["sim_ci"]} for k, v in r.items()}}
        print("mkt" if market else "nomkt", res["mkt" if market else "nomkt"], flush=True)
    return res


def scene_masks(S: pd.DataFrame) -> dict[str, np.ndarray]:
    d, m = S["distance"].to_numpy(), S["date"].dt.month.to_numpy()
    young = (pd.to_numeric(S["horse_id"].str[:4]) >= 2014).to_numpy()
    return {
        "距離 〜1400": d <= 1400, "距離 1500〜1800": (d >= 1500) & (d <= 1800), "距離 1900〜": d >= 1900,
        "2 歳戦": S["rage"].eq("A2").to_numpy(), "3 歳戦": S["rage"].eq("A3").to_numpy(), "古馬混合": S["rage"].isin(["A3U", "A4U"]).to_numpy(),
        "初ダート": young & (S["_nsurf"] == 0).to_numpy(), "ダート 1〜4 走": young & ((S["_nsurf"] >= 1) & (S["_nsurf"] <= 4)).to_numpy(),
        "ダート 5 走〜": young & (S["_nsurf"] >= 5).to_numpy(),
        "冬（12〜2 月）": np.isin(m, [12, 1, 2]), "夏（6〜8 月）": np.isin(m, [6, 7, 8]),
        "良": S["going"].eq("G").to_numpy(), "稍重〜不良": S["going"].isin(["S", "W"]).to_numpy(),
    }


def run_scenes(F, n_boot: int) -> dict:
    S = dirt_rows(F)
    res = {}
    for nm, mk in scene_masks(S).items():
        T = S[mk].reset_index(drop=True)
        if len(T) < 3000:
            res[nm] = {"rows": int(len(T))}
            continue
        Zt = zs(T)
        M = np.c_[tuple([T["lp"].to_numpy(), T["lq"].to_numpy(), Zt["ctrl"], Zt["lv5"], Zt["sire"], Zt["bms"], Zt["mat5"]] + calib_cols(T))]
        b, bs = fit_boot(T, M, n_boot)
        res[nm] = {"rows": int(len(T)), "mat5": round(float(b[6]), 4), "ci": np.percentile(bs[:, 6], [2.5, 97.5]).round(4).tolist(),
                   "se": round(float(bs[:, 6].std()), 4), "sire": round(float(b[4]), 4)}
        print(nm, res[nm], flush=True)
    ts = {k: v["mat5"] / v["se"] for k, v in res.items() if "se" in v}
    res["note"] = "場面ごとの別々の当てはめ（記述）。場面どうしは重なる。13 場面を見たので、1 本ごとの区間は割り引いて読む"
    res["t"] = {k: round(v, 2) for k, v in ts.items()}
    return res


def run_popsire(F, n_boot: int) -> dict:
    S = dirt_rows(F)
    Z = zs(S)
    rk = S.groupby("race_id")["odds"].rank(method="first").to_numpy()
    bands = {"1〜3 番人気": rk <= 3, "4〜6 番人気": (rk >= 4) & (rk <= 6), "7〜9 番人気": (rk >= 7) & (rk <= 9), "10 番人気〜": rk >= 10}
    cols, names_ = [S["lp"].to_numpy(), S["lq"].to_numpy(), Z["ctrl"], Z["lv5"], Z["bms"], Z["mat5"]], []
    rid = S["race_id"].to_numpy()

    def inband(z, mk):
        # 帯の中で中心化し直す（レースの中の帯の行の平均を引く。帯の間の比べを混ぜない。critic 3 回目の 4）
        v = np.where(mk, z, np.nan)
        mu = pd.Series(v).groupby(rid).transform("mean").to_numpy()
        return np.where(mk, z - mu, 0.0)
    for nm, mk in bands.items():
        cols.append(inband(Z["sire"], mk))
        names_.append(nm)
    M = np.c_[tuple(cols + calib_cols(S))]
    idx = {n: 6 + i for i, n in enumerate(names_)}
    b, bs = fit_boot(S, M, n_boot)
    c, r = sim_ci(b, bs, idx)
    # 牝系も同じ帯で
    cols2 = [S["lp"].to_numpy(), S["lq"].to_numpy(), Z["ctrl"], Z["lv5"], Z["sire"], Z["bms"]] + [inband(Z["mat5"], mk) for mk in bands.values()]
    b2, bs2 = fit_boot(S, np.c_[tuple(cols2 + calib_cols(S))], n_boot)
    c2, r2 = sim_ci(b2, bs2, idx)
    return {"sire_by_band": {"c": round(c, 3), **{k: {"gamma": v["gamma"], "sim_ci": v["sim_ci"]} for k, v in r.items()}},
            "mat5_by_band": {"c": round(c2, 3), **{k: {"gamma": v["gamma"], "sim_ci": v["sim_ci"]} for k, v in r2.items()}},
            "note": "人気の帯は確定オッズの順位（kf_v4 の予測がある行の中の順位・同じオッズは馬番の順）。列は帯の中で中心化（帯の中の効き）"}


def run_gens(F, A, keys, L, n_boot: int) -> dict:
    """代ごとの牝系の列（2 代＝母父・3 代・4 代・5 代）を同時に当てる（どの深さが効くか。記述）。
    aptitude.py は前向きの記録で凍結しているので触らず、まとめた後の表を代で絞る（同じ祖先が複数の代に出るときは一番近い代に入り、
    その重みには遠い代の血量も足されている）。"""
    cols = {}
    for g in (2, 3, 4, 5):
        Lg = L[L["gen"] == g].reset_index(drop=True)
        cols[f"g{g}"], _ = ck.mat_cols(F, A, keys, Lg)
    for k, v in cols.items():
        F[f"z_{k}"] = v
    S = dirt_rows(F)
    Z = zs(S)
    G = {k: ap_.zA(S, S[f"z_{k}"].to_numpy()) for k in cols}
    res = {"rows": int(len(S))}
    for market in (False, True):
        base = [S["lp"].to_numpy(), S["lq"].to_numpy()] if market else [S["lp"].to_numpy()]
        M = np.c_[tuple(base) + (Z["ctrl"], Z["lv5"], Z["sire"]) + tuple(G.values())]
        idx = {k: len(base) + 3 + i for i, k in enumerate(G)}
        b, bs = fit_boot(S, M, n_boot)
        c, r = sim_ci(b, bs, idx)
        res["mkt" if market else "nomkt"] = {"c": round(c, 3), **{k: {"gamma": v["gamma"], "sim_ci": v["sim_ci"]} for k, v in r.items()}}
        print("mkt" if market else "nomkt", res["mkt" if market else "nomkt"], flush=True)
    res["note"] = "g2 は母父（2 代）。母父の列（z_bms）は入れず、代ごとの列で置き換えた"
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["anc", "v5bag", "scenes", "popsire", "gens", "basic"])
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--calib", action="store_true", help="市場の確率の曲がり（lq の 2 乗・人気の帯・道悪）を基準に足す（統括役の (3)(4)）")
    p.add_argument("--null-seed", type=int, default=None, help="牝系の祖先の札の帰無（aptitude の作り）")
    p.add_argument("--aspect", default="dirt", choices=["rand", "early", "pace_h", "pace_s", "wet", "ext", "short", "steep", "straight", "yoshiba", "dirt", "long", "turf"], help="popsire だけ: 8 観点の型（early・ext・straight・wet など。統括役の (d)）")
    a = p.parse_args()
    CALIB["on"] = a.calib
    if a.calib and a.what not in ("popsire", "scenes", "basic"):
        raise SystemExit("--calib は popsire・scenes・basic だけ")
    if a.aspect != "dirt" and a.what not in ("popsire", "gens", "basic"):
        raise SystemExit("--aspect は popsire・gens・basic だけ")
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", f"dirt_deep_{a.what}" + ("" if a.aspect == "dirt" else f"_{a.aspect}") + ("_calib" if a.calib else "") + ("" if a.null_seed is None else f"_null{a.null_seed}"),
                                   f"ダートの牝系の候補の深掘り: {a.what} {a.aspect}（尤度だけ・記述）"))
    F, A, keys, L = setup(a.aspect, a.null_seed)
    if a.what == "gens":
        res = run_gens(F, A, keys, L, a.boot)
    elif a.what == "basic":
        res = run_basic(F, a.boot)
    else:
        res = run_anc(F, A, keys, L) if a.what == "anc" else {"v5bag": run_v5bag, "scenes": run_scenes, "popsire": run_popsire}[a.what](F, a.boot)
    res["aspect"], res["calib"], res["null_seed"] = a.aspect, a.calib, a.null_seed
    tag = ("" if a.aspect == "dirt" else f"_{a.aspect}") + ("_calib" if a.calib else "") + ("" if a.null_seed is None else f"_null{a.null_seed}")
    path = data.OUT / f"dirt_deep_{a.what}{tag}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False)[:4000], flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
