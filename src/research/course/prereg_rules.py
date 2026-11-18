"""事前登録した閾値のルールを、2017〜2024 年に 1 回だけ当てて確かめる（docs/ideas/course_class_handicapper.md §13）。

- ルール・向き・範囲・期間・層・効果の下限・ブートストラップは config/prereg_threshold_rules.json（commit 済みであること）。式はこのファイルの rule_masks。
- 通常の面は §12 と同じ単位（コース × クラスの群 × 良馬場）の行。重賞の面は G1〜G3 の全馬場の行。
- 比べ方: ルールに当たる馬（⭕ でない馬）と当たらない馬を、層（年 × 対数オッズの帯 × 頭数の帯 × 面の鍵）の中で比べた
  Mantel–Haenszel のリスク差（3 着内）。区間と p は、レースで束ねたポアソンのブートストラップ。
- 採る条件: BH（登録の組の数）の後に q ≤ q_max・向きが登録どおり・|差| ≥ 効果の下限・2022〜24 で同じ向きかつ 2017〜21 の半分以上・
  2024 年の区間がすべて逆向きの側に無い・（通常のみ）似た単位の 3/4 以上で同じ向き。
- 負の対照（N1〜N3）が q ≤ q_max なら、帰無が壊れているとして全体を使わない。
- 1 回だけ: config とこのファイルが commit 済みで変更が無いこと（git）を確かめ、印を計算の前に立てる。2025 年以降は読まない。
使い方: python -m src.research.course.prereg_rules run   # <COURSE_OUT>/prereg/result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import threshold as TH

CONFIG = CR.REPO / "config/prereg_threshold_rules.json"
OUTP = CR.OUT / "prereg"
MARK = OUTP / "RAN_PREREG_RULES.json"
OPPLUS = {"OP", "L", "G3", "G2", "G1"}


def prev_lookup(d: pd.DataFrame) -> pd.DataFrame:
    """前走の行（p_race_id × horse_id）の wt・jockey_id と、前走の勝ち馬の斤量（同着は軽い方）。引けなければ NaN。"""
    key = d[["race_id", "horse_id"]].astype(str)
    prev = pd.DataFrame({"race_id": d["p_race_id"].astype("string").fillna("").to_numpy(), "horse_id": key["horse_id"].to_numpy()})
    tab = pd.DataFrame({"race_id": key["race_id"].to_numpy(), "horse_id": key["horse_id"].to_numpy(),
                        "prev_wt": pd.to_numeric(d["wt"], errors="coerce").to_numpy(), "prev_jockey": d["jockey_id"].astype("string").to_numpy()})
    m = prev.merge(tab.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    win = d[pd.to_numeric(d["fin"], errors="coerce") == 1]
    wjw = pd.to_numeric(win["jw"], errors="coerce").groupby(win["race_id"].astype(str)).min()
    m["prev_win_jw"] = prev["race_id"].map(wjw).to_numpy()
    return m


def rule_masks(d: pd.DataFrame, pv: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """ID → (判定できる馬, ⭕ でない馬)。範囲（今回の格）も判定できる馬に入れる。"""
    num = lambda c: pd.to_numeric(d[c], errors="coerce").to_numpy(float)
    g = d["grade"].astype(str).to_numpy()
    pg = d["p_grade"].astype("string").fillna("").to_numpy()
    has_prev = d["p_race_id"].notna().to_numpy()
    pf, pm, l3, p1, p4 = num("p_fin"), num("p_margin"), num("p_l3rank"), num("p_pos1"), num("p_pos4")
    jw, pjw, age, iv, dist, pdist = num("jw"), num("p_jw"), num("age"), num("interval"), num("dist"), num("p_dist")
    pwin = pv["prev_win_jw"].to_numpy(float)
    out = {}
    # U01（G1）
    eff = pm - 0.1 * (pjw - pwin)
    ok_g3 = (pf == 1) | (eff <= 0.1)
    known = has_prev & ~np.isnan(pf) & ~((pg == "G3") & np.isnan(pwin) & (pf != 1))
    good = (pg == "G1") | ((pg == "G2") & (pf <= 3)) | ((pg == "G3") & ok_g3)
    out["U01"] = ((g == "G1") & known, ~good)
    # U04（G1）
    lead = np.where(np.isnan(p1), p4 == 1, (p1 == 1) & (p4 == 1))
    pos_known = ~np.isnan(p4) & (p4 > 0)                                          # 通過順位が欠ける馬は、逃げを判定できないので外す
    out["U04"] = ((g == "G1") & ~np.isnan(l3) & pos_known, (l3 >= 7) & ~lead)
    # U06（G3）
    out["U06"] = ((g == "G3") & ~np.isnan(pf) & (p4 > 0), (pf >= 5) & (p4 >= 1) & (p4 <= 4))
    # U04c（G3）
    lplus = np.isin(pg, ["L", "G3", "G2", "G1"])
    out["U04c"] = ((g == "G3") & ~np.isnan(l3), (l3 >= 10) | ((l3 >= 4) & (l3 <= 9) & ~lplus))
    # U20（G3）
    pw = pv["prev_wt"].to_numpy(float)
    pw = np.where(pw > 0, pw, np.nan)                                             # 体重 0 は不明
    out["U20"] = ((g == "G3") & ~np.isnan(pw), pw <= 469)
    # U15（G3）
    out["U15"] = ((g == "G3") & ~np.isnan(age), age <= 5)
    # U18（G3）
    out["U18"] = ((g == "G3") & ~np.isnan(iv), (iv <= 29) & ~(np.isin(pg, list(OPPLUS)) & (pf <= 3)))
    # U09b（G2）
    out["U09b"] = ((g == "G2") & ~np.isnan(pdist), dist <= pdist)
    # U19（G2）
    out["U19"] = ((g == "G2") & ~np.isnan(jw), jw >= 59)
    # G2a（範囲は呼ぶ側で面ごとに切る）
    pj = pv["prev_jockey"].astype("string")
    kn = pj.notna().to_numpy()
    out["G2a"] = (kn, kn & (d["jockey_id"].astype("string").to_numpy() != pj.fillna("").to_numpy()))
    # 負の対照: ゲートの順・枠の内外と関係せず、同じ馬でも走ごとに変わる印（馬の ID とレースの ID に塩を混ぜたハッシュの偶奇）
    base = d["horse_id"].astype(str) + "#" + d["race_id"].astype(str)
    for nid, salt in (("N1", "a"), ("N2", "b"), ("N3", "c")):
        hh = pd.util.hash_pandas_object(base + "#" + salt, index=False).to_numpy()
        out[nid] = (np.ones(len(d), bool), (hh % 2) == 1)
    return out


def strata(d: pd.DataFrame, key: np.ndarray, edges: list[float]) -> np.ndarray:
    o = pd.to_numeric(d["odds_final"], errors="coerce").to_numpy(float)
    ob = np.digitize(np.nan_to_num(o, nan=9999), edges)
    fb = np.digitize(pd.to_numeric(d["field"], errors="coerce").fillna(12).to_numpy(), [10.5, 14.5])
    s = pd.Series(d["yr"].astype(str).to_numpy()) + "|" + pd.Series(ob).astype(str) + "|" + pd.Series(fb).astype(str) + "|" + pd.Series(key)
    return pd.factorize(s)[0]


def mh(st: np.ndarray, flag: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """MH のリスク差（flag 側 − 他方）。w は行の重み（ブートストラップ）。"""
    w = np.ones(len(st)) if w is None else w
    K = st.max() + 1 if len(st) else 0
    n1 = np.bincount(st, weights=w * flag, minlength=K)
    n0 = np.bincount(st, weights=w * ~flag, minlength=K)
    k1 = np.bincount(st, weights=w * flag * y, minlength=K)
    k0 = np.bincount(st, weights=w * ~flag * y, minlength=K)
    ok = (n1 > 0) & (n0 > 0)
    if not ok.any():
        return None
    wt = n1[ok] * n0[ok] / (n1[ok] + n0[ok])
    return float((wt * (k1[ok] / n1[ok] - k0[ok] / n0[ok])).sum() / wt.sum())


def boot(st, flag, y, race, reps: int, seed: int) -> np.ndarray:
    """レースで束ねたポアソンのブートストラップ（各レースに Poisson(1) の重み）。"""
    rng = np.random.default_rng(seed)
    if not len(st):
        return np.array([])
    rc, _ = pd.factorize(race)
    out = []
    for _ in range(reps):
        wr = rng.poisson(1.0, rc.max() + 1)
        v = mh(st, flag, y, wr[rc].astype(float))
        if v is not None:
            out.append(v)
    return np.array(out)


def _stat(Y: np.ndarray) -> dict:
    return TH._stat(Y.sum(0)) if len(Y) else TH._stat(np.zeros(9))


def _roi_trim(Y: np.ndarray) -> dict:
    """複勝の回収率と、払戻の上位 1%・上位 1 本を除いた値（高配当への頼りを見る）。"""
    pay = Y[Y[:, 4] > 0]
    if not len(pay):
        return {}
    rf = np.sort(pay[:, 6])[::-1]
    k = max(1, int(len(rf) * 0.01))
    return {"roi_f": float(rf.sum() / len(pay)), "roi_f_trim1pct": float(rf[k:].sum() / len(pay)), "roi_f_trim1": float(rf[1:].sum() / len(pay))}


def evaluate(full: pd.DataFrame, C: dict, log=print) -> dict:
    """登録の組をすべて確かめる。full は 2024 年までの frame（TH.load の形。2016 年を含む）。前走の行は full 全体から引き、
    その後で期間（C["period"]）と前走が地方・海外の馬の除外を当てる。"""
    if (full["yr"] > C["period"][1]).any():
        raise ValueError("登録の期間の後の行がある")
    pv_all = prev_lookup(full)
    pr = full["prev_region"].astype("string").fillna("").to_numpy()
    keep = (full["yr"].to_numpy() >= C["period"][0]) & ~np.isin(pr, ["地方", "海外"])
    d, pv = full[keep].reset_index(drop=True), pv_all[keep].reset_index(drop=True)
    M = rule_masks(d, pv)
    yr = d["yr"].to_numpy()
    race = d["race_id"].astype(str).to_numpy()
    popf = pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float)
    Pc = TH.pools(d, "course")
    side_rows = {"course": Pc["target"] != "", "graded": np.isin(d["grade"].astype(str).to_numpy(), ["G1", "G2", "G3"])}
    side_key = {"course": Pc["fam"], "graded": np.where(side_rows["graded"], (d["surf"].astype(str) + "|" + d["dband"]).to_numpy(), "")}
    # オッズから見込む 3 着内（§12 と同じ作り。並べるための値で、検定は MH で行う）
    e = np.zeros(len(d))
    for sd in ("course", "graded"):
        e = np.where(side_rows[sd], TH.expected(d, side_rows[sd], side_key[sd]), e)
    Y = TH.ymat(d, e)
    y = (Y[:, 1] + Y[:, 2] + Y[:, 3]) > 0
    oddsf = pd.to_numeric(d["odds_final"], errors="coerce").to_numpy(float)
    layers = {ly: TH.in_layer(TH.LAYERS[ly], popf, oddsf) for ly in C["layers"]}
    pop6 = TH.in_layer(("popmin", 6), popf, oddsf)
    B = C["bootstrap"]
    rows = []
    groups = [(r["id"], s) for r in C["rules"] for s in (["course", "graded"] if r["side"] == "both" else [r["side"]])]
    ctrls = [(n["id"], s) for n in C["negative_controls"] for s in ("course", "graded")]
    for (rid, side) in groups + ctrls:
        for ly, lm in layers.items():
            t0 = time.time()
            app, flag = M[rid]
            m = app & side_rows[side] & lm
            st = strata(d.loc[m].reset_index(drop=True), side_key[side][m], C["odds_edges"])
            f, yy, rr, yy_r = flag[m], y[m], race[m], yr[m]
            est = mh(st, f, yy)
            bs = boot(st, f, yy, rr, B["reps"], B["seed"])
            se = float(bs.std()) if len(bs) > 1 else None
            per = {}
            for lab, sel in (("2017〜21", (yy_r >= 2017) & (yy_r <= 2021)), ("2022〜24", (yy_r >= 2022) & (yy_r <= 2024)),
                             ("2020〜24", yy_r >= 2020)):
                per[lab] = mh(st[sel], f[sel], yy[sel])
            s24 = yy_r == 2024
            b24 = boot(st[s24], f[s24], yy[s24], rr[s24], B["reps"], B["seed"] + 1)
            br = {}
            keys = side_key[side][m]
            for kk in pd.unique(keys):        # 似た単位（通常は採否に使う。重賞は並べるだけ）
                km = keys == kk
                if km.sum() >= C["breadth"]["min_runs"] and f[km].any() and (~f[km]).any():
                    br[kk] = mh(st[km], f[km], yy[km])
            Ym = Y[m]
            p6 = pop6[m]
            extra = {"pop6_rd": mh(st[p6], f[p6], yy[p6]) if p6.any() else None,                      # 参考: 6 番人気以下（BH の外）
                     "n_flag_by_period": {"2017〜22": int(f[yy_r <= 2022].sum()), "2023〜24": int(f[yy_r >= 2023].sum())}}
            rows.append({"id": rid, "side": side, "layer": ly, "control": rid.startswith("N"), "n": int(m.sum()), "n_flag": int(f.sum()),
                         "rd": est, "se": se, "lo": float(np.percentile(bs, 2.5)) if len(bs) else None, "hi": float(np.percentile(bs, 97.5)) if len(bs) else None,
                         "periods": per, "rd2024_lo": float(np.percentile(b24, 2.5)) if len(b24) else None,
                         "rd2024_hi": float(np.percentile(b24, 97.5)) if len(b24) else None, "breadth": br,
                         "flag": _stat(Ym[f]) | _roi_trim(Ym[f]), "other": _stat(Ym[~f]) | _roi_trim(Ym[~f]),
                         "base": _stat(Ym) | _roi_trim(Ym)} | extra)
            log(f"{rid}/{side}/{ly}: n {m.sum():,}・当たる {f.sum():,}・{time.time() - t0:.0f}s", flush=True)   # 差は出さない（途中で結果を見ない）
    return judge(rows, C)


def judge(rows: list[dict], C: dict) -> dict:
    """p（向きのある片側。負の対照は両側）→ BH（登録の組・負の対照は別に）→ 採る条件。"""
    from scipy.stats import norm
    for r in rows:
        z = (r["rd"] / r["se"]) if (r["rd"] is not None and r["se"]) else 0.0
        r["p"] = float(2 * norm.sf(abs(z))) if r["control"] else float(norm.cdf(z))   # 登録はすべて −（差が負なら小さい p）
    for ctl in (False, True):
        sub = [r for r in rows if r["control"] == ctl]
        q = TH.bh(np.array([r["p"] for r in sub])) if sub else []
        for r, qq in zip(sub, q):
            r["q"] = float(qq)
    broken = any(r["q"] <= C["q_max"] for r in rows if r["control"])
    for r in rows:
        if r["control"]:
            continue
        a, b = r["periods"]["2017〜21"], r["periods"]["2022〜24"]
        c = {"1 BH": r["q"] <= C["q_max"] and (r["rd"] or 0) < 0,
             "2 効果の下限": r["rd"] is not None and abs(r["rd"]) * 100 >= C["effect_min_pt"][r["layer"]],
             "3 最近の持続": a is not None and b is not None and b < 0 and abs(b) >= 0.5 * abs(a),
             "4 2024 年": r["rd2024_lo"] is not None and not (r["rd2024_lo"] > 0)}
        if r["side"] == "course":
            v = [x for x in r["breadth"].values() if x is not None]
            c["5 似た単位"] = bool(v) and sum(x < 0 for x in v) >= C["breadth"]["share"] * len(v)
        r["checks"] = c
        r["adopted"] = bool(all(c.values()) and not broken)
    return {"rows": rows, "null_broken": broken, "n_tests": sum(1 for r in rows if not r["control"]),
            "adopted": [f"{r['id']}/{r['side']}/{r['layer']}" for r in rows if r.get("adopted")]}


def _git_clean(paths: list[Path]) -> bool:
    for p in paths:
        if subprocess.run(["git", "-C", str(CR.REPO), "ls-files", "--error-unmatch", str(p)], capture_output=True, check=False).returncode != 0:
            return False
        if subprocess.run(["git", "-C", str(CR.REPO), "diff", "--quiet", "HEAD", "--", str(p)], check=False).returncode != 0:
            return False
    return True


def cmd_run() -> None:
    """1 回だけ: config とコード（このファイル・threshold・v2・run）が commit 済みで変更が無いときだけ流す。印は計算の前に立てる。
    印があれば、途中で落ちたときも流し直さない（流し直すかは、統括役に返して決める）。途中のログに差は出さない。"""
    from src.research.course import v2 as V2
    if MARK.exists() or (OUTP / "result.json").exists():
        sys.exit("事前登録のルールは流し済み（1 回だけ）")
    code = Path(__file__)
    deps = [CONFIG, code, Path(TH.__file__), Path(V2.__file__), Path(CR.__file__)]
    if not _git_clean(deps):
        sys.exit("config・コードが commit されていないか、変更がある")
    C = json.loads(CONFIG.read_text(encoding="utf-8"))
    CR.wait_resources()
    OUTP.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(CR.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deps + [TH.FRAME]}   # 層の作り（frame・コード）の sha
    MARK.write_text(json.dumps({"started_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "git_head": head, "sha256": sha}), encoding="utf-8")
    d = TH.load(C["period"][1])
    res = evaluate(d, C)
    res |= {"ran_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "git_head": head, "sha256": sha, "period": C["period"]}
    (OUTP / "result.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    MARK.write_text(json.dumps(json.loads(MARK.read_text(encoding="utf-8")) | {"finished_at": res["ran_at"]}), encoding="utf-8")
    print("result", OUTP / "result.json", "採った", res["adopted"], "帰無が壊れている" if res["null_broken"] else "", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run()


if __name__ == "__main__":
    main()
