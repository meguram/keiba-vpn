"""tp-43 の分析本体（tp43_marks.py の build が作る race_features.parquet / returns.npy を読む）。

回収率は『1 レースの賭け金を一定にして買い目で等分した戻り』のレース平均。外れ値（まれな高配当）に平均が引きずられるので、
1 レースの戻りを賭け金の 100 倍で頭打ちにした回収率（capped。全列で共通の上限なので、印方式どうしの比較を歪めない）を主指標にし、生の回収率（raw）も併記する。
荒れ度の区分の境目と『荒れ度で印を切り替える規則（M6b）』の選択は 2022〜2023 年だけで決め、2024 年で確認、2025 年は参考。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp43_marks as tm

CAP_X = 100.0
FOCUS_SCHEMES = ["tan1", "fuku1", "umaren_n3", "wide_n3", "umatan_n3", "sanpuku_a4", "sanpuku_b4"]
ALTS = ["M2_ptop3", "M3m_evwin_model", "M3b_evwin_blend", "M4_evplace", "M5b_t25", "M5b_t35", "M5b_t45", "M5m_t35", "M7_ev_x_top3"]
BUCKET_JP = {0: "堅い（下位1/3）", 1: "中", 2: "荒れやすい（上位1/3）"}


def cap_returns(R: np.ndarray, cap: float = CAP_X) -> tuple[np.ndarray, np.ndarray]:
    caps = np.full(R.shape[1], cap)
    return np.minimum(R, cap), caps


def _roi(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> list[float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def split_masks(rf: pd.DataFrame) -> dict[str, np.ndarray]:
    y = rf["year"].to_numpy()
    return {"design": np.isin(y, tm.DESIGN_YEARS), "confirm": y == tm.CONFIRM_YEAR, "ref": y == tm.REF_YEAR,
            "test": np.isin(y, (tm.CONFIRM_YEAR, tm.REF_YEAR)), "all": np.ones(len(y), bool)}


def col(mark: str, scheme: str) -> int:
    return tm.COMBO_IDX[(mark, scheme)]


def baseline_table(R, Rc, sm) -> list[dict]:
    rows = []
    for m in tm.MARKS:
        for s in tm.SCHEME_NAMES:
            i = col(m, s)
            row = {"mark": m, "scheme": s}
            for k, mask in sm.items():
                x, xc = R[mask, i], Rc[mask, i]
                row[k] = {"raw": _roi(x), "cap": _roi(xc), "n": int(np.isfinite(x).sum())}
            row["hit_all"] = float((R[:, i][np.isfinite(R[:, i])] > 0).mean())
            rows.append(row)
    return rows


def bucket_grid(Rc, bk: pd.DataFrame, sm) -> dict:
    """荒れ度の定義 × 区分 × 買い方 × 印 → 設計/確認/参考の回収率（capped）とレース数。"""
    out: dict = {}
    for d in tm.DEFS:
        b = bk[d].to_numpy()
        out[d] = {}
        for v in (0, 1, 2):
            inb = b == v
            cell = {}
            for s in tm.SCHEME_NAMES:
                cell[s] = {}
                for m in tm.MARKS:
                    i = col(m, s)
                    cell[s][m] = [_roi(Rc[sm[k] & inb, i]) for k in ("design", "confirm", "ref")] + [int((sm["all"] & inb & np.isfinite(Rc[:, i])).sum())]
            out[d][str(v)] = cell
    return out


def excess_vs_m1(Rc, bk: pd.DataFrame, sm) -> dict:
    """荒れ度の区分ごとに、各印の『M1（勝率順）との回収率の差』を全買い方で平均。荒れるほど M1 以外が有利になるかを見る。"""
    out: dict = {}
    for d in tm.DEFS:
        b = bk[d].to_numpy()
        out[d] = {}
        for alt in ALTS:
            out[d][alt] = {}
            for v in (0, 1, 2):
                inb = b == v
                row = {}
                for k in ("design", "confirm", "ref"):
                    diffs = []
                    for s in tm.SCHEME_NAMES:
                        a, base = Rc[sm[k] & inb, col(alt, s)], Rc[sm[k] & inb, col("M1_pwin", s)]
                        ok = np.isfinite(a) & np.isfinite(base)
                        if ok.sum() >= 30:
                            diffs.append(float(a[ok].mean() - base[ok].mean()))
                    row[k] = float(np.mean(diffs)) if diffs else float("nan")
                out[d][alt][str(v)] = row
    return out


def interaction(Rc, bk: pd.DataFrame, rf: pd.DataFrame, n_boot: int = 1500) -> list[dict]:
    """(M_alt − M1) の差が『荒れ』で『堅い』より大きいか。2023〜2025 年をまとめ、レース単位ブートストラップ。"""
    rows = []
    yrs = rf["year"].to_numpy()
    use = yrs >= 2023
    for d in tm.DEFS:
        b = bk[d].to_numpy()
        for alt in ALTS:
            for s in FOCUS_SCHEMES:
                a, base = Rc[:, col(alt, s)], Rc[:, col("M1_pwin", s)]
                diff = a - base
                ok = use & np.isfinite(diff)
                hi, lo = ok & (b == 2), ok & (b == 0)
                if hi.sum() < 100 or lo.sum() < 100:
                    continue
                rng = np.random.default_rng(0)
                dh, dl = diff[hi], diff[lo]
                bh = dh[rng.integers(0, len(dh), (n_boot, len(dh)))].mean(1)
                bl = dl[rng.integers(0, len(dl), (n_boot, len(dl)))].mean(1)
                inter = bh - bl
                rows.append({"def": d, "alt": alt, "scheme": s, "rough": float(dh.mean()), "solid": float(dl.mean()), "inter": float(dh.mean() - dl.mean()),
                             "lo": float(np.quantile(inter, 0.025)), "hi": float(np.quantile(inter, 0.975)), "n_rough": int(hi.sum()), "n_solid": int(lo.sum())})
    return rows


def select_m6b(Rc, bk: pd.DataFrame, sm) -> dict:
    """設計年（2022〜23）だけで、荒れ度の定義×区分×買い方ごとに回収率が最大の印を選ぶ。"""
    sel: dict = {}
    for d in tm.DEFS:
        b = bk[d].to_numpy()
        sel[d] = {}
        for v in (0, 1, 2):
            inb = (b == v) & sm["design"]
            for s in tm.SCHEME_NAMES:
                best, bv = None, -1e9
                for m in tm.MARKS:
                    x = Rc[inb, col(m, s)]
                    x = x[np.isfinite(x)]
                    if len(x) < 150:
                        continue
                    if x.mean() > bv:
                        best, bv = m, float(x.mean())
                sel[d][f"{v}|{s}"] = {"mark": best, "design_roi": bv}
    return sel


def eval_m6b(R, Rc, bk, sm, sel: dict, n_boot: int = 1500) -> list[dict]:
    rows = []
    for d in tm.DEFS:
        b = bk[d].to_numpy()
        for s in tm.SCHEME_NAMES:
            x, xr = np.full(len(b), np.nan), np.full(len(b), np.nan)
            for v in (0, 1, 2):
                pick = sel[d].get(f"{v}|{s}", {}).get("mark")
                if pick is None:
                    continue
                inb = b == v
                x[inb], xr[inb] = Rc[inb, col(pick, s)], R[inb, col(pick, s)]
            base, base_r = Rc[:, col("M1_pwin", s)], R[:, col("M1_pwin", s)]
            row = {"def": d, "scheme": s}
            for k in ("design", "confirm", "ref", "test"):
                mk = sm[k]
                row[k] = {"m6b_cap": _roi(x[mk]), "m6b_raw": _roi(xr[mk]), "m1_cap": _roi(base[mk]), "m1_raw": _roi(base_r[mk])}
            ci = boot_ci((x - base)[sm["test"]], n_boot)
            row["test_diff_vs_m1"] = ci
            rows.append(row)
    return rows


def global_best(Rc, sm) -> dict:
    """設計年だけで、買い方ごとに回収率が最大の印（荒れ度を使わない単一の印）。"""
    gb = {}
    for s in tm.SCHEME_NAMES:
        best, bv = "M1_pwin", -1e9
        for m in tm.MARKS:
            x = Rc[sm["design"], col(m, s)]
            x = x[np.isfinite(x)]
            if len(x) >= 500 and x.mean() > bv:
                best, bv = m, float(x.mean())
        gb[s] = best
    return gb


def switch_value(R, Rc, bk, sm, sel: dict, gb: dict, n_boot: int = 1000) -> list[dict]:
    """荒れ度で印を切り替える価値: 切り替え規則 − 『荒れ度を使わない単一の印』を、検証年（2024〜25）で比べる。"""
    rows = []
    for d in tm.DEFS:
        b = bk[d].to_numpy()
        for s in tm.SCHEME_NAMES:
            xb = np.full(len(b), np.nan)
            for v in (0, 1, 2):
                pick = sel[d].get(f"{v}|{s}", {}).get("mark")
                if pick:
                    xb[b == v] = Rc[b == v, col(pick, s)]
            m1, m5, g = Rc[:, col("M1_pwin", s)], Rc[:, col("M5b_t35", s)], Rc[:, col(gb[s], s)]
            xa = np.where(b == 2, m5, m1)
            row = {"def": d, "scheme": s, "global_best": gb[s]}
            for name, (u, w) in {"m6b_vs_global": (xb, g), "m6a_vs_m5": (xa, m5), "m6a_vs_m1": (xa, m1)}.items():
                diff = (u - w)
                row[name] = {"test": boot_ci(diff[sm["test"]], n_boot), "confirm": _roi(diff[sm["confirm"]]), "ref": _roi(diff[sm["ref"]])}
            rows.append(row)
    return rows


def run(out: Path, n_boot: int = 500) -> None:
    rf = pd.read_parquet(out / "race_features.parquet")
    R = np.load(out / "returns.npy")
    sm = split_masks(rf)
    Rc, caps = cap_returns(R)
    cuts = tm.cutoffs(rf)
    bk = tm.buckets(rf, cuts)
    res: dict = {"races": int(len(rf)), "by_year": rf.groupby("year").size().to_dict(), "cuts": cuts,
                 "cap_x": CAP_X, "frac_capped": float(np.nanmean(np.where(np.isfinite(R), R > caps[None, :], np.nan)))}
    res["bucket_share"] = {d: {str(v): float((bk[d] == v).mean()) for v in (0, 1, 2)} for d in tm.DEFS}
    res["baseline"] = baseline_table(R, Rc, sm)
    res["grid"] = bucket_grid(Rc, bk, sm)
    res["excess_vs_m1"] = excess_vs_m1(Rc, bk, sm)
    res["interaction"] = interaction(Rc, bk, rf)
    sel = select_m6b(Rc, bk, sm)
    res["m6b_selection"] = sel
    res["m6b_eval"] = eval_m6b(R, Rc, bk, sm, sel)
    gb = global_best(Rc, sm)
    res["global_best"] = gb
    res["switch_value"] = switch_value(R, Rc, bk, sm, sel, gb)
    # reality check（固定の戦略だけ。検証年 2024〜25 年）
    names, X = tm.strategy_matrix(Rc, bk, sm["test"])
    rc = tm.reality_check(X, n_boot=n_boot)
    kept = rc.pop("kept")
    mean, t = rc.pop("mean"), rc.pop("t")
    order = np.argsort(-np.array(t))[:40]
    rc["top"] = [{"strategy": list(names[kept[i]]), "excess_per_race": mean[i], "t": t[i], "bet_races": int((X[:, kept[i]] != 0).sum())} for i in order]
    rc["n_all_strategies"] = len(names)
    res["reality_check_test"] = rc
    # 全期間（2023〜2025）でも同じ（設計年を含むので参考）
    names2, X2 = tm.strategy_matrix(Rc, bk, rf["year"].to_numpy() >= 2023)
    rc2 = tm.reality_check(X2, n_boot=n_boot)
    k2 = rc2.pop("kept")
    m2, t2 = rc2.pop("mean"), rc2.pop("t")
    o2 = np.argsort(-np.array(t2))[:20]
    rc2["top"] = [{"strategy": list(names2[k2[i]]), "excess_per_race": m2[i], "t": t2[i], "bet_races": int((X2[:, k2[i]] != 0).sum())} for i in o2]
    res["reality_check_2023_25"] = rc2
    # 破産確率: 検証上位と、ユーザー案（荒れ区分で M5b_t35）
    ruin = []
    cand = [tuple(r["strategy"]) for r in rc["top"][:6]]
    for d in ("ent", "fav", "mkt"):
        for s in ("wide_n3", "umaren_n3", "sanpuku_a4"):
            cand.append((f"m6_{d}_M5b_t35", "M1/M5b_t35", s))
    seen = set()
    for c in cand:
        if c in seen:
            continue
        seen.add(c)
        j = names.index(c) if c in names else None
        if j is None:
            continue
        ex = X[:, j]
        bets = ex[ex != 0] + 1.0
        ruin.append({"strategy": list(c), "bet_races": int(len(bets)), "roi_cap_on_bets": float(bets.mean()) if len(bets) else None, "ruin": tm.ruin_prob(bets)})
    res["ruin"] = ruin
    (out / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", out / "analysis.json", "strategies", len(names), "rc p", rc["p"], "t_max", rc["t_max"])


def final2026(out: Path, rules: Path) -> None:
    """2026 年（未使用）で、設計年だけで決めた規則を一度だけ評価する。単勝オッズが壊れたレースは frame_all の作成時に除外済み。"""
    from src.scraper.payoff_repair import load_fix
    from src.scripts.data.tp36_final2026 import parse_payoff_json

    a = json.loads(rules.read_text(encoding="utf-8"))
    d = tm.load_frame((2026,))
    pj = pd.read_parquet(tm.ROOT / "data/page_reference/tables/2026/race_result_flat.parquet", columns=["race_id", "payoff"]).drop_duplicates("race_id")
    fix = load_fix(2026)
    pays, bad = {}, set()
    for r in pj.itertuples(index=False):
        rid = str(r.race_id)
        if rid not in set(d["race_id"]):
            continue
        p, broken = tm.payoffs_from_json(parse_payoff_json(r.payoff, fix.get(rid)))
        if broken:
            bad.add(rid)
        pays[rid] = p
    rf, R = tm.build(d, pays, bad)
    Rc, _ = cap_returns(R)
    cuts = {k: tuple(v) for k, v in a["cuts"].items()}
    bk = tm.buckets(rf, cuts)
    gb, sel = a["global_best"], a["m6b_selection"]
    res: dict = {"races": int(len(rf)), "excluded_broken_payoff": len(bad), "rows": []}
    for s in tm.SCHEME_NAMES:
        cand = {"M1_pwin": Rc[:, col("M1_pwin", s)], "M2_ptop3": Rc[:, col("M2_ptop3", s)], "M3b_evwin_blend": Rc[:, col("M3b_evwin_blend", s)],
                "M5b_t35": Rc[:, col("M5b_t35", s)], "M5b_t45": Rc[:, col("M5b_t45", s)], f"global_best({gb[s]})": Rc[:, col(gb[s], s)]}
        raw = {"M1_pwin": R[:, col("M1_pwin", s)], "M5b_t35": R[:, col("M5b_t35", s)], "M3b_evwin_blend": R[:, col("M3b_evwin_blend", s)]}
        for dname in tm.DEFS:
            b = bk[dname].to_numpy()
            cand[f"M6a_{dname}"] = np.where(b == 2, Rc[:, col("M5b_t35", s)], Rc[:, col("M1_pwin", s)])
            xb = np.full(len(b), np.nan)
            for v in (0, 1, 2):
                pick = sel[dname].get(f"{v}|{s}", {}).get("mark")
                if pick:
                    xb[b == v] = Rc[b == v, col(pick, s)]
            cand[f"M6b_{dname}"] = xb
        for name, x in cand.items():
            ci = boot_ci(x, 1000)
            rr = raw.get(name)
            res["rows"].append({"scheme": s, "rule": name, "roi_cap": ci[0], "lo": ci[1], "hi": ci[2], "roi_raw": _roi(rr) if rr is not None else None,
                                "hit": float((x[np.isfinite(x)] > 0).mean())})
    (out / "final2026.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    best = max(res["rows"], key=lambda r: r["lo"])
    print("saved", out / "final2026.json", "races", res["races"], "max lower bound", best["rule"], best["scheme"], round(best["lo"], 3))
