"""tp-60 血統別の勝率・複勝率・単勝/複勝回収率と、翌年の再現性・市場の織り込み。

python -m src.scripts.data.tp60_roi cells         # 血統キー別（全期間・年別）の成績、上位/下位、系統・新種牡馬・近交・外国産の回収率と区間
python -m src.scripts.data.tp60_roi persist       # キー × 条件ごとに『その年より前の成績 → その年』の再現性（順位相関・上位10%を買った回収率 − 全体）
python -m src.scripts.data.tp60_roi popular       # 種牡馬の人気（産駒数・平均人気）と回収率

入力: tp-59 の master.parquet（確定オッズ・払戻・V4 の確率つき）と tp-60 の血統表。選択は 2020〜2024 の過去だけで行い、評価年は 2021〜2025（2026 は独立確認のテストセット②として別掲）。
回収率は 100 円あたり（1.0 = 100%）。区間は 95%・レース単位ブートストラップ。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.scripts.data import tp60_lib as L

EVAL_YEARS = [2021, 2022, 2023, 2024, 2025]
TEST2 = 2026
K_SHRINK = 150.0         # 回収率の縮約の疑似件数（persist の中で最適値も報告）


def _save(name: str, obj) -> None:
    L.RES.mkdir(parents=True, exist_ok=True)
    (L.RES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding="utf-8")


def cmd_cells(a) -> None:
    m = L.load_master()
    L.log(f"master {m.shape}、年別行数 {m.groupby('year').size().to_dict()}")
    out: dict = {"baseline": {}, "keys": {}}
    for y, g in m.groupby("year"):
        out["baseline"][int(y)] = {"horses": int(len(g)), "win_rate": float(g["win"].mean()), "top3_rate": float(g["top3"].mean()),
                                     "roi_win": float(g["ret_win"].mean() / 100), "roi_place": float(g["ret_place"].mean() / 100)}
    train = m["year"] <= 2025
    base_all = {"roi_win": L.roi_ci(m, train.to_numpy(), "ret_win"), "roi_place": L.roi_ci(m, train.to_numpy(), "ret_place")}
    out["baseline"]["pooled_2020_2025"] = base_all
    for key in L.KEYS:
        c = L.collapse(m[train], key, "all")
        r = L.agg_cells(c)
        r["name"] = r["key"]
        nmin = 300 if key in ("sire_id", "dsire_id", "ssire_id") else (150 if key == "sire_dsire" else 1)
        keep = r[r["horses"] >= nmin].sort_values("horses", ascending=False)
        ent = {"n_cells": int(len(r)), "n_cells_eligible": int(len(keep)), "nmin": nmin}
        # 回収率に区間を付けた表（産駒数が多い順に上位 40）
        rows = []
        for _, row in keep.head(40 if key in ("sire_id", "dsire_id", "ssire_id", "sire_dsire") else 20).iterrows():
            mask = (train & (m[key] == row["key"])).to_numpy()
            cw, cp = L.roi_ci(m, mask, "ret_win", B=600), L.roi_ci(m, mask, "ret_place", B=600)
            rows.append({"key": row["key"], "horses": int(row["horses"]), "win_rate": row["win_rate"], "top3_rate": row["top3_rate"],
                         "roi_win": cw["roi"], "roi_win_lo": cw["lo"], "roi_win_hi": cw["hi"], "roi_place": cp["roi"], "roi_place_lo": cp["lo"], "roi_place_hi": cp["hi"],
                         "ex_mkt": row["ex_mkt"]})
        ent["by_size"] = rows
        # 区間の下限が 100% を超えるセル（全期間・件数下限あり）
        sig_w = [x for x in rows if x["roi_win_lo"] > 1.0]
        sig_p = [x for x in rows if x["roi_place_lo"] > 1.0]
        ent["lo_gt_100_win"], ent["lo_gt_100_place"] = len(sig_w), len(sig_p)
        out["keys"][key] = ent
        L.log(f"{key}: セル {len(r)}（下限 {nmin} 以上 {len(keep)}）、区間の下限>100%: 単勝 {len(sig_w)} 複勝 {len(sig_p)}")
    # 年別の特殊セル（新種牡馬・外国産・近交）
    spec = {}
    for key, vals in (("crop", ["crop0", "crop1", "crop2", "crop3+", "established"]), ("foreign", ["foreign_bred", "jp_bred"]),
                      ("xF_band", ["no_cross", "F<0.8%", "F<1.6%", "F<3.1%", "F>=3.1%"]),
                      ("sire_line", ["SS系", "ロベルト系", "MrP系", "ND系", "ナスルーラ系", "ネイティヴダンサー系", "ヘイルトゥリーズン系", "ブランドフォード系", "その他"]),
                      ("dsire_line", ["SS系", "ロベルト系", "MrP系", "ND系", "ナスルーラ系", "ネイティヴダンサー系", "ヘイルトゥリーズン系", "ブランドフォード系", "その他"])):
        spec[key] = {}
        for v in vals:
            e = {}
            for label, msk in (("2020-2025", m["year"] <= 2025), ("2026", m["year"] == 2026)) + tuple((str(y), m["year"] == y) for y in EVAL_YEARS):
                mk = (msk & (m[key] == v)).to_numpy()
                if mk.sum() < 30:
                    continue
                cw, cp = L.roi_ci(m, mk, "ret_win", B=500), L.roi_ci(m, mk, "ret_place", B=500)
                d = m[mk]
                e[label] = {"horses": int(mk.sum()), "win_rate": float(d["win"].mean()), "top3_rate": float(d["top3"].mean()), "roi_win": cw["roi"], "roi_win_lo": cw["lo"], "roi_win_hi": cw["hi"],
                            "roi_place": cp["roi"], "roi_place_lo": cp["lo"], "roi_place_hi": cp["hi"], "ex_mkt": float((d["win"] - d["q_win"]).mean())}
            spec[key][v] = e
    out["special"] = spec
    _save("cells.json", out)


def _cum_cells(c: pd.DataFrame, y: int) -> pd.DataFrame:
    return L.agg_cells(c[c["year"] < y])


def persist_one(m: pd.DataFrame, key: str, cond: str, nmin: int, years: list[int], metric_years_min: int = 2020) -> dict:
    d = m.dropna(subset=[key]).copy()
    c = L.collapse(d, key, cond)
    d["cell"] = d[key].astype(str) + "#" + d[cond].astype(str)
    c["cell"] = c["key"].astype(str) + "#" + c["cond"].astype(str)
    res = {"key": key, "cond": cond, "nmin": nmin, "years": {}}
    pool_sel_w, pool_sel_p = [], []
    for y in years:
        pri = _cum_cells(c, y)
        pri["cell"] = pri["key"].astype(str) + "#" + pri["cond"].astype(str)
        pri = pri[pri["horses"] >= nmin]
        if len(pri) < 20:
            continue
        mu_w, mu_p = pri["rw"].sum() / pri["horses"].sum(), pri["rp"].sum() / pri["horses"].sum()
        pri["s_w"] = L.eb_shrink(pri["rw"].to_numpy(), pri["horses"].to_numpy(), mu_w, K_SHRINK) / 100.0
        pri["s_p"] = L.eb_shrink(pri["rp"].to_numpy(), pri["horses"].to_numpy(), mu_p, K_SHRINK) / 100.0
        cur = L.agg_cells(c[c["year"] == y])
        cur["cell"] = cur["key"].astype(str) + "#" + cur["cond"].astype(str)
        j = pri.merge(cur[["cell", "horses", "roi_win", "roi_place", "win_rate", "top3_rate", "ex_mkt", "ex_model"]], on="cell", how="inner", suffixes=("", "_next"))
        j = j[j["horses_next"] >= max(nmin // 3, 10)]
        ent: dict = {"n_cells_prior": int(len(pri)), "n_cells_pairs": int(len(j))}
        if len(j) >= 20:
            for nm, a_, b_ in (("roi_win", "roi_win", "roi_win_next"), ("roi_place", "roi_place", "roi_place_next"), ("win_rate", "win_rate", "win_rate_next"),
                               ("top3_rate", "top3_rate", "top3_rate_next"), ("ex_mkt", "ex_mkt", "ex_mkt_next"), ("shrunk_win", "s_w", "roi_win_next"), ("shrunk_place", "s_p", "roi_place_next")):
                ent[f"rho_{nm}"] = float(j[a_].corr(j[b_], method="spearman"))
        # 上位 10% を買った回収率 − 全体（選ぶのは過去の縮約回収率）
        yr = (d["year"] == y).to_numpy()
        for bt, scol, rcol in (("win", "s_w", "ret_win"), ("place", "s_p", "ret_place")):
            thr = pri[scol].quantile(0.9)
            topc, botc = set(pri.loc[pri[scol] >= thr, "cell"]), set(pri.loc[pri[scol] <= pri[scol].quantile(0.1), "cell"])
            elig = yr & d["cell"].isin(set(pri["cell"])).to_numpy()
            sel_top = d["cell"].isin(topc).to_numpy()
            sel_bot = d["cell"].isin(botc).to_numpy()
            # boot_diff は行位置で動くので、d のインデックスを振り直す
            dd = d.reset_index(drop=True)
            rt = L.boot_diff(dd, sel_top, elig, rcol, B=800)
            rb = L.boot_diff(dd, sel_bot, elig, rcol, B=800)
            ent[f"top_{bt}"], ent[f"bottom_{bt}"] = rt, rb
        res["years"][str(y)] = ent
    # 年をまたいだ平均
    ys = [v for v in res["years"].values() if "rho_roi_win" in v]
    if ys:
        res["mean_rho"] = {k: float(np.nanmean([v[k] for v in ys if k in v])) for k in ys[0] if k.startswith("rho_")}
    for bt in ("win", "place"):
        tops = [v[f"top_{bt}"] for k, v in res["years"].items() if f"top_{bt}" in v and k != str(L_TEST2) and np.isfinite(v[f"top_{bt}"].get("diff", np.nan))]
        if tops:
            res[f"top_{bt}_mean_diff"] = float(np.mean([t["diff"] for t in tops]))
            res[f"top_{bt}_mean_sel_roi"] = float(np.mean([t["sel_roi"] for t in tops]))
            res[f"top_{bt}_mean_all_roi"] = float(np.mean([t["all_roi"] for t in tops]))
            res[f"top_{bt}_n_pos"] = int(sum(t["diff"] > 0 for t in tops))
            res[f"top_{bt}_n_years"] = len(tops)
    return res


L_TEST2 = TEST2

PERSIST_SPECS = [("sire_id", "all", 100), ("dsire_id", "all", 100), ("ssire_id", "all", 100), ("sire_dsire", "all", 40),
                 ("sire_id", "sd", 40), ("dsire_id", "sd", 40), ("sire_id", "going", 40), ("sire_id", "venue", 40), ("sire_id", "cls", 40), ("sire_id", "age_band", 40),
                 ("sire_id", "pop_band", 40), ("dsire_id", "pop_band", 40), ("sire_line", "sd", 150), ("sire_line", "going", 150), ("sire_line", "venue", 150),
                 ("line_pair", "sd", 150), ("dsire_line", "sd", 150), ("crop", "sd", 150), ("xF_band", "sd", 150)]


def cmd_persist(a) -> None:
    m = L.load_master()
    out = {}
    only = a.only.split(",") if a.only else None
    for key, cond, nmin in PERSIST_SPECS:
        nm = f"{key}@{cond}@{nmin}"
        if only and nm not in only:
            continue
        L.log(f"persist {nm}")
        r = persist_one(m, key, cond, nmin, EVAL_YEARS + [TEST2])
        out[nm] = r
        mr = r.get("mean_rho", {})
        L.log(f"  rho(roi_win)={mr.get('rho_roi_win', np.nan):+.3f} rho(roi_place)={mr.get('rho_roi_place', np.nan):+.3f} rho(win_rate)={mr.get('rho_win_rate', np.nan):+.3f} rho(top3)={mr.get('rho_top3_rate', np.nan):+.3f} "
              f"top10%買い(単勝) 差の平均={r.get('top_win_mean_diff', np.nan):+.3f} 複勝={r.get('top_place_mean_diff', np.nan):+.3f}")
    _save("persist.json" if not only else "persist_part.json", out)


def cmd_popular(a) -> None:
    m = L.load_master()
    tr = m[m["year"] <= 2025]
    out = {}
    for key in ("sire_id", "dsire_id"):
        g = tr.groupby(key).agg(n=("win", "size"), pop=("popularity", "mean"), q=("q_win", "mean"), rw=("ret_win", "mean"), rp=("ret_place", "mean"), win=("win", "mean"), top3=("top3", "mean"))
        g = g[g["n"] >= 100].sort_values("n", ascending=False)
        g["rw"], g["rp"] = g["rw"] / 100, g["rp"] / 100
        ent = {"n_sires": int(len(g)), "corr_q_roi_win": float(g["q"].corr(g["rw"], method="spearman")), "corr_q_roi_place": float(g["q"].corr(g["rp"], method="spearman")),
               "corr_n_roi_win": float(g["n"].corr(g["rw"], method="spearman")), "corr_n_roi_place": float(g["n"].corr(g["rp"], method="spearman"))}
        grp = {}
        order = g.index.tolist()
        for lab, ids in (("産駒数 上位20", order[:20]), ("21〜60位", order[20:60]), ("61位以下（100頭以上）", order[60:])):
            mk = (tr[key].isin(ids)).to_numpy()
            grp[lab] = {"sires": len(ids), "horses": int(mk.sum()), "roi_win": L.roi_ci(tr.reset_index(drop=True), mk, "ret_win", B=500), "roi_place": L.roi_ci(tr.reset_index(drop=True), mk, "ret_place", B=500),
                        "win_rate": float(tr.loc[mk, "win"].mean()), "top3_rate": float(tr.loc[mk, "top3"].mean()), "mean_q": float(tr.loc[mk, "q_win"].mean())}
        # 市場確率の 4 分位（人気の度合い）ごとの回収率
        qs = pd.qcut(g["q"], 4, labels=["人気薄の血統", "やや人気薄", "やや人気", "人気の血統"])
        ent["by_q_quartile"] = {k: {"sires": int(v.shape[0]), "roi_win": float(v["rw"].mean()), "roi_place": float(v["rp"].mean())} for k, v in g.groupby(qs, observed=True)}
        ent["groups"] = grp
        out[key] = ent
    _save("popular.json", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cells")
    p = sub.add_parser("persist")
    p.add_argument("--only")
    sub.add_parser("popular")
    a = ap.parse_args()
    {"cells": cmd_cells, "persist": cmd_persist, "popular": cmd_popular}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
