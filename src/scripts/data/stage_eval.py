"""舞台（競馬場×コース×クラス群）の定義検証と、舞台別の重み付け（部品モデルの凸結合）の LOYO 評価。

python -m src.scripts.data.stage_eval counts   # 舞台の件数表（年別・最小サンプル基準・縮約の階層）
python -m src.scripts.data.stage_eval hetero   # 部品モデル間の LogLoss 差が舞台群で変わるか（クラスの分割ラインの走査）
python -m src.scripts.data.stage_eval weights  # 舞台別の重み（全体の重みへ縮約）の LOYO 評価
python -m src.scripts.data.stage_eval all

入力は既存の OOF（2022〜2024 年）。2026 年は使わない。障害は平地と分けて除外（件数だけ報告）。
出力: data/local/modeling/stage_eval/{counts,hetero,weights}.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "data/local/modeling/training_pipeline_v4/v6"
TABLE = ROOT / "data/local/modeling/training_pipeline_fixed/features/train_table.parquet"
OUT = ROOT / "data/local/modeling/stage_eval"
KEY = ["race_id", "horse_id"]
YEARS = [2022, 2023, 2024]
MODELS = {
    "A_RB_e": V / "eval_rev/oof_z:RB@e.parquet",
    "B_CY_BEST": V / "eval/oof_d_CY_BEST.parquet",
    "C_REV": V / "eval_rev/oof_z:RB+REV_tp32_FIRST+REV_tp39_GR_comp+REV_tp41_SIM_apt+REV_tp42_RL_A@e.parquet",
    "D_CB_NJ_RS59": V / "eval/oof_z:RB+CB+NJ+RS59.parquet",
    "E_S76": V / "eval/oof_z:RB+S76_D2+S76_D5+S76_D5r@e.parquet",
    "F_NJ": V / "eval/oof_x_NJ.parquet",
}
LADDER = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP"]
EPS = 1e-6


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────── データ ─────────
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s or ""))


def ladder_of(rc: str, name: str, grade: str) -> str:
    t = _norm(rc) + " " + _norm(name)
    if "新馬" in t:
        return "新馬"
    if "未勝利" in t:
        return "未勝利"
    if re.search(r"1勝", t):
        return "1勝"
    if re.search(r"2勝", t):
        return "2勝"
    if re.search(r"3勝", t):
        return "3勝"
    if re.search(r"オープン|OP", t) or str(grade) == "OP":
        return "OP"
    return "その他"


def age_of(rc: str, name: str) -> str:
    t = _norm(rc) or _norm(name)
    return "限定戦" if re.search(r"[23]歳(?!以上)", t) else "それ以外"


def band_of(d: int) -> str:
    return "短距離" if d <= 1400 else "マイル" if d <= 1800 else "中距離" if d <= 2200 else "長距離"


def load() -> tuple[pd.DataFrame, list[str]]:
    t = pd.read_parquet(TABLE, columns=["race_id", "horse_id", "date", "venue", "surface", "distance", "grade", "race_class", "race_name", "finish_position"])
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    t["year"] = t["date"].astype(str).str[:4].astype(int)
    for c in ("venue", "surface", "grade", "race_class", "race_name"):
        t[c] = t[c].astype(str)
    t = t[t["year"].isin(YEARS)].drop_duplicates(KEY)
    base = None
    names = []
    for n, p in MODELS.items():
        if not p.exists():
            _log(f"スキップ（OOF なし）: {n}")
            continue
        o = pd.read_parquet(p)[KEY + ["p_win", "p_top3"]]
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        o = o.drop_duplicates(KEY).rename(columns={"p_win": f"w_{n}", "p_top3": f"t_{n}"})
        base = o if base is None else base.merge(o, on=KEY, how="inner")
        names.append(n)
    df = base.merge(t, on=KEY, how="inner")
    df["y_win"] = (df["finish_position"] == 1).astype(float)
    df["y_top3"] = (df["finish_position"] <= 3).astype(float)
    r = df.drop_duplicates("race_id")[["race_id", "race_class", "race_name", "grade", "distance"]].copy()
    r["ladder"] = [ladder_of(a, b, c) for a, b, c in zip(r.race_class, r.race_name, r.grade)]
    r["age"] = [age_of(a, b) for a, b in zip(r.race_class, r.race_name)]
    r["band"] = r["distance"].astype(int).map(band_of)
    df = df.merge(r[["race_id", "ladder", "age", "band"]], on="race_id", how="left")
    df["flat"] = df["surface"] != "障"
    df["course"] = df["venue"] + df["surface"] + df["distance"].astype(int).astype(str)
    df["level2"] = np.where(df["ladder"].isin(["新馬", "未勝利", "1勝"]), "低", np.where(df["ladder"].isin(["2勝", "3勝", "OP"]), "高", "他"))
    df["level3"] = np.where(df["ladder"].isin(["新馬", "未勝利", "1勝"]), "低", np.where(df["ladder"].isin(["2勝", "3勝"]), "中", np.where(df["ladder"] == "OP", "OP以上", "他")))
    df["al"] = df["age"] + "×" + df["level2"]
    df["sb"] = df["surface"] + df["band"]
    return df, names


def ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ───────── (1) 件数 ─────────
def cmd_counts(df: pd.DataFrame, names: list[str]) -> dict:
    res: dict = {}
    races = df.drop_duplicates("race_id")
    res["races_total"] = int(len(races))
    res["hurdle_races"] = int((~races["flat"]).sum())
    res["rows"] = int(len(df))
    f = races[races["flat"]]
    res["ladder_counts"] = f["ladder"].value_counts().to_dict()
    res["age_counts"] = f["age"].value_counts().to_dict()
    res["al_counts"] = f["al"].value_counts().to_dict()
    res["level3_counts"] = f["level3"].value_counts().to_dict()
    res["unread_other"] = int((f["ladder"] == "その他").sum())
    res["unread_other_examples"] = f.loc[f["ladder"] == "その他", ["race_class", "race_name"]].drop_duplicates().head(10).values.tolist()
    res["by_year_al"] = f.groupby(["year", "al"]).size().unstack(fill_value=0).to_dict()
    # 最小サンプル基準（年別レース数 >= N が 3 年とも）
    out = {}
    for tag, keys in {
        "競馬場×コース": ["course"],
        "競馬場×コース×年齢群": ["course", "age"],
        "競馬場×コース×レベル2": ["course", "level2"],
        "競馬場×コース×年齢×レベル": ["course", "al"],
        "芝ダ×距離帯×年齢×レベル": ["sb", "al"],
        "芝ダ×距離帯": ["sb"],
    }.items():
        g = f.groupby(keys + ["year"]).size().unstack("year", fill_value=0)
        total = len(f)
        row = {"cells": int(len(g))}
        for n_min in (3, 5, 10, 20):
            ok = (g[YEARS] >= n_min).all(axis=1)
            row[f"年別{n_min}R以上(3年とも)"] = {"cells": int(ok.sum()), "races_share": round(float(g.loc[ok, YEARS].sum().sum() / total), 3)}
        row["median_races_per_cell_year"] = float(np.median(g[YEARS].to_numpy()))
        out[tag] = row
    res["min_sample"] = out
    res["hierarchy"] = ["全体", "年齢×レベル(4群)", "年齢×レベル×芝ダ×距離帯", "年齢×レベル×競馬場×コース"]
    res["shrink_rule"] = "w_cell = λ·w_own + (1-λ)·w_parent、λ = n_races / (n_races + κ)。学習側で未見のセルは親の重み"
    _log(json.dumps({k: res[k] for k in ("races_total", "hurdle_races", "ladder_counts", "al_counts", "level3_counts", "unread_other")}, ensure_ascii=False))
    for k, v in out.items():
        _log(f"{k}: " + json.dumps(v, ensure_ascii=False))
    return res


# ───────── (2) 舞台でモデル差が変わるか ─────────
def _race_arrays(f: pd.DataFrame, target: str, names: list[str]):
    pref = "w_" if target == "win" else "t_"
    y = f["y_win" if target == "win" else "y_top3"].to_numpy()
    ids, inv = np.unique(f["race_id"].to_numpy(), return_inverse=True)
    n = np.bincount(inv, minlength=len(ids)).astype(float)
    L = {}
    for m in names:
        l = ll(f[pref + m].to_numpy(), y)
        L[m] = np.bincount(inv, weights=l, minlength=len(ids))
    first = f.groupby("race_id").first().reindex(ids)
    return ids, n, L, first


def cmd_hetero(df: pd.DataFrame, names: list[str], B: int = 500, seed: int = 0) -> dict:
    f = df[df["flat"] & (df["ladder"] != "その他")].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    res: dict = {"models": names}
    for target in ("win", "top3"):
        ids, n, L, first = _race_arrays(f, target, names)
        R = len(ids)
        yr = first["year"].to_numpy()
        W = np.stack([np.bincount(rng.integers(0, R, size=R), minlength=R) for _ in range(B)]).astype(np.float32)
        tot = {m: float(L[m].sum() / n.sum()) for m in names}
        res[f"{target}_logloss"] = {m: round(v, 5) for m, v in tot.items()}
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]

        def delta(mask, a, b, w=None):
            s = L[a] - L[b]
            if w is None:
                return float(s[mask].sum() / n[mask].sum())
            return (w[:, mask] @ s[mask]) / (w[:, mask] @ n[mask])

        def interaction(m1, m2, a, b):
            d_obs = delta(m1, a, b) - delta(m2, a, b)
            d_bs = delta(m1, a, b, W) - delta(m2, a, b, W)
            lo, hi = np.percentile(d_bs, [2.5, 97.5])
            by_year = []
            for y in YEARS:
                my1, my2 = m1 & (yr == y), m2 & (yr == y)
                by_year.append(float(delta(my1, a, b) - delta(my2, a, b)) if my1.sum() > 50 and my2.sum() > 50 else float("nan"))
            same = bool(np.all(np.sign(by_year) == np.sign(d_obs)))
            return {"inter": round(d_obs, 5), "ci": [round(float(lo), 5), round(float(hi), 5)], "excl0": bool(lo > 0 or hi < 0), "by_year": [round(v, 5) for v in by_year], "same_sign_all_years": same}

        scans: dict = {}
        lad = first["ladder"].to_numpy()
        # クラス階段の分割ライン: 低=ladder[:k], 高=ladder[k:]
        for k in range(1, len(LADDER)):
            low = np.isin(lad, LADDER[:k]); high = np.isin(lad, LADDER[k:])
            key = f"{'/'.join(LADDER[:k])} | {'/'.join(LADDER[k:])}"
            rs = [interaction(low, high, a, b) for a, b in pairs]
            scans[f"レベル分割 {key}"] = {"n_low": int(low.sum()), "n_high": int(high.sum()), "pairs_excl0": int(sum(r["excl0"] for r in rs)), "pairs_stable": int(sum(r["excl0"] and r["same_sign_all_years"] for r in rs)), "n_pairs": len(pairs), "mean_abs_inter": round(float(np.mean([abs(r["inter"]) for r in rs])), 5), "max_abs_inter": round(float(max(abs(r["inter"]) for r in rs)), 5)}
        age = first["age"].to_numpy()
        lim, opn = age == "限定戦", age == "それ以外"
        rs = [interaction(lim, opn, a, b) for a, b in pairs]
        scans["年齢 限定戦 | それ以外"] = {"n_low": int(lim.sum()), "n_high": int(opn.sum()), "pairs_excl0": int(sum(r["excl0"] for r in rs)), "pairs_stable": int(sum(r["excl0"] and r["same_sign_all_years"] for r in rs)), "n_pairs": len(pairs), "mean_abs_inter": round(float(np.mean([abs(r["inter"]) for r in rs])), 5), "max_abs_inter": round(float(max(abs(r["inter"]) for r in rs)), 5)}
        # 年齢を揃えた上でのレベル差（限定戦内 / それ以外内）
        for nm, am in (("限定戦内", lim), ("それ以外内", opn)):
            low = am & np.isin(lad, ["新馬", "未勝利", "1勝"]); high = am & np.isin(lad, ["2勝", "3勝", "OP"])
            if low.sum() > 100 and high.sum() > 100:
                rs = [interaction(low, high, a, b) for a, b in pairs]
                scans[f"レベル2 低|高（{nm}）"] = {"n_low": int(low.sum()), "n_high": int(high.sum()), "pairs_excl0": int(sum(r["excl0"] for r in rs)), "pairs_stable": int(sum(r["excl0"] and r["same_sign_all_years"] for r in rs)), "n_pairs": len(pairs), "mean_abs_inter": round(float(np.mean([abs(r["inter"]) for r in rs])), 5), "max_abs_inter": round(float(max(abs(r["inter"]) for r in rs)), 5)}
        # 芝/ダ（参考）
        sf = first["surface"].to_numpy()
        rs = [interaction(sf == "芝", sf == "ダート", a, b) for a, b in pairs]
        scans["芝 | ダート（参考）"] = {"n_low": int((sf == "芝").sum()), "n_high": int((sf == "ダート").sum()), "pairs_excl0": int(sum(r["excl0"] for r in rs)), "pairs_stable": int(sum(r["excl0"] and r["same_sign_all_years"] for r in rs)), "n_pairs": len(pairs), "mean_abs_inter": round(float(np.mean([abs(r["inter"]) for r in rs])), 5), "max_abs_inter": round(float(max(abs(r["inter"]) for r in rs)), 5)}
        res[f"{target}_scan"] = scans
        # 最大の差が出たペアの詳細（レベル2 低|高）
        low = np.isin(lad, ["新馬", "未勝利", "1勝"]); high = np.isin(lad, ["2勝", "3勝", "OP"])
        det = {f"{a} - {b}": {"低レベル": round(delta(low, a, b), 5), "ハイレベル": round(delta(high, a, b), 5), **interaction(low, high, a, b)} for a, b in pairs}
        res[f"{target}_pair_detail_level2"] = det
        for k, v in scans.items():
            _log(f"[{target}] {k}: 有意{v['pairs_excl0']}/{v['n_pairs']} 安定{v['pairs_stable']} 平均|差|={v['mean_abs_inter']}")
    return res


# ───────── (3) 舞台別の重み（凸結合）と LOYO ─────────
def _fit_w(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    K = P.shape[1]

    def f(th):
        w = np.exp(th - th.max()); w /= w.sum()
        p = np.clip(P @ w, EPS, 1 - EPS)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        g_p = -(y / p - (1 - y) / (1 - p)) / len(y)
        g_w = P.T @ g_p
        g_th = w * (g_w - (w * g_w).sum())
        return loss, g_th

    r = minimize(f, np.zeros(K), jac=True, method="L-BFGS-B")
    w = np.exp(r.x - r.x.max())
    return w / w.sum()


HIER = {
    "H0 全体のみ": [],
    "H1 全体→年齢": ["age"],
    "H2 全体→レベル2": ["level2"],
    "H3 全体→年齢×レベル": ["al"],
    "H4 全体→レベル3": ["level3"],
    "H5 全体→年齢×レベル→×芝ダ距離帯": ["al", ["al", "sb"]],
    "H6 全体→年齢×レベル→×芝ダ距離帯→×競馬場コース": ["al", ["al", "sb"], ["al", "course"]],
    "H7 全体→芝ダ距離帯→競馬場コース（クラス無し）": ["sb", ["sb", "course"]],
}


def _keys(f: pd.DataFrame, spec) -> np.ndarray:
    cols = [spec] if isinstance(spec, str) else spec
    return f[cols].astype(str).agg("|".join, axis=1).to_numpy() if len(cols) > 1 else f[cols[0]].astype(str).to_numpy()


def cmd_weights(df: pd.DataFrame, names: list[str], B: int = 500, seed: int = 1) -> dict:
    f = df[df["flat"] & (df["ladder"] != "その他")].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    res: dict = {"models": names, "kappas": [100, 500, 2000]}
    kappas = res["kappas"]
    for target in ("win", "top3"):
        pref = "w_" if target == "win" else "t_"
        P = np.stack([f[pref + m].to_numpy() for m in names], axis=1)
        y = f["y_win" if target == "win" else "y_top3"].to_numpy()
        yr = f["year"].to_numpy()
        rid = f["race_id"].to_numpy()
        ids, inv = np.unique(rid, return_inverse=True)
        R = len(ids)
        nrace = np.bincount(inv, minlength=R).astype(float)
        W = np.stack([np.bincount(rng.integers(0, R, size=R), minlength=R) for _ in range(B)]).astype(np.float32)
        # 単独モデルと等重みの参照
        l_single = {m: ll(P[:, i], y) for i, m in enumerate(names)}
        l_equal = ll(P.mean(axis=1), y)
        out_t: dict = {"single_logloss": {m: round(float(v.mean()), 5) for m, v in l_single.items()}, "equal_logloss": round(float(l_equal.mean()), 5)}
        # 各 fold・各階層・各セルの own 重み
        rows_ll: dict = {}   # (hier, kappa) -> row-level 損失（LOYO の held-out）
        rows_ll_global = np.zeros(len(f))
        for hname, levels in HIER.items():
            for kp in kappas:
                rows_ll[(hname, kp)] = np.zeros(len(f))
        glob_w = {}
        for y_out in YEARS:
            tr = yr != y_out
            te = ~tr
            Pt, yt = P[tr], y[tr]
            wg = _fit_w(Pt, yt)
            glob_w[y_out] = wg
            rows_ll_global[te] = ll(P[te] @ wg, y[te])
            for hname, levels in HIER.items():
                if not levels:
                    for kp in kappas:
                        rows_ll[(hname, kp)][te] = rows_ll_global[te]
                    continue
                # 各レベルの own 重み（学習側）
                keys_tr = [_keys(f[tr], s) for s in levels]
                keys_te = [_keys(f[te], s) for s in levels]
                own: list[dict] = []
                for kt in keys_tr:
                    d = {}
                    Rtr = f.loc[tr, "race_id"].to_numpy()
                    for c in np.unique(kt):
                        m = kt == c
                        nr = len(np.unique(Rtr[m]))
                        if m.sum() >= 30:
                            d[c] = (_fit_w(Pt[m], yt[m]), nr)
                    own.append(d)
                for kp in kappas:
                    # 学習側のセル別の縮約済み重み（親から順に）
                    shr: list[dict] = []
                    parent_of_cell = None
                    # 行ごとに親を引く必要があるため、held-out 行ごとに連鎖で計算（セルの組合せをキャッシュ）
                    cache: dict = {}
                    W_te = np.zeros((te.sum(), len(names)))
                    for i in range(te.sum()):
                        ck = tuple(k[i] for k in keys_te)
                        if ck not in cache:
                            w = wg
                            for lv, c in enumerate(ck):
                                if c in own[lv]:
                                    wo, nr = own[lv][c]
                                    lam = nr / (nr + kp)
                                    w = lam * wo + (1 - lam) * w
                            cache[ck] = w
                        W_te[i] = cache[ck]
                    rows_ll[(hname, kp)][te] = ll((P[te] * W_te).sum(axis=1), y[te])
            _log(f"[{target}] fold {y_out} 完了")
        base = rows_ll_global

        def cmp(l_new, l_old):
            d = np.bincount(inv, weights=l_new - l_old, minlength=R)
            pt = float(d.sum() / nrace.sum())
            bs = (W @ d) / (W @ nrace)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            by = {int(yy): round(float((l_new[yr == yy] - l_old[yr == yy]).mean()), 5) for yy in YEARS}
            return {"d_logloss": round(pt, 5), "ci": [round(float(lo), 5), round(float(hi), 5)], "by_year": by, "same_sign_all_years": bool(all(v < 0 for v in by.values()))}

        out_t["global_weights_loyo_logloss"] = round(float(base.mean()), 5)
        out_t["global_weights"] = {str(k): dict(zip(names, np.round(v, 3).tolist())) for k, v in glob_w.items()}
        out_t["global_vs_equal"] = cmp(base, l_equal)
        best_single = min(l_single, key=lambda m: l_single[m].mean())
        out_t["best_single"] = best_single
        out_t["global_vs_best_single"] = cmp(base, l_single[best_single])
        tab = {}
        for (hname, kp), l in rows_ll.items():
            tab[f"{hname} κ={kp}"] = {"logloss": round(float(l.mean()), 5), **cmp(l, base)}
            _log(f"[{target}] {hname} κ={kp}: Δ={tab[f'{hname} κ={kp}']['d_logloss']:+.5f} CI {tab[f'{hname} κ={kp}']['ci']}")
        out_t["vs_global"] = tab
        res[target] = out_t
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["counts", "hetero", "weights", "all"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, names = load()
    _log(f"行 {len(df)}、レース {df['race_id'].nunique()}、モデル {names}")
    todo = ["counts", "hetero", "weights"] if a.cmd == "all" else [a.cmd]
    for c in todo:
        r = {"counts": cmd_counts, "hetero": cmd_hetero, "weights": cmd_weights}[c](df, names)
        (OUT / f"{c}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        _log(f"{c}: 保存 {OUT / (c + '.json')}")


if __name__ == "__main__":
    main()
