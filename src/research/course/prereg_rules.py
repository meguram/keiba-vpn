"""事前登録した閾値のルールを、2018〜2024 年に 1 回だけ当てて確かめる（docs/ideas/course_class_handicapper.md §13）。

- ルール・向き・範囲・期間・層・効果の下限・ブートストラップは config/prereg_threshold_rules.json（commit 済みであること）。式はこのファイルの rule_masks。
- 通常の面は §12 と同じ単位（コース × クラスの群 × 良馬場）の行。重賞の面は G1〜G3 の全馬場の行。
- 比べ方: ルールに当たる馬（⭕ でない馬）と当たらない馬を、層（年 × 対数オッズの帯 × 頭数の帯 × 面の鍵）の中で比べた
  Mantel–Haenszel のリスク差（3 着内）。区間と p は、開催日 × 場で束ねたポアソンのブートストラップ。
- 期間（戦略 C2・C7）: 採否 2018〜21（BH・効果の下限・件数・2020〜21 も同じ向き）・確かめ 2022〜24（持続: 同じ向きで半分以上・2024 年が逆向きでない）。
  （通常のみ）似た単位は採否の期間で見る。通ったものは「候補」で、採否は凍結の後の前向きの追跡だけ（C1）。2025・2026 年は評価だけ。
- 負の対照（N1〜N3）が q ≤ q_max なら、帰無が壊れているとして全体を使わない。
- 1 回だけ: config・このファイル・§14 の登録とコードが commit 済みで変更が無いこと（git）を確かめ、読み込みを済ませてから印を立てる。2025 年以降は読まない。
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
    その後で期間（C["discovery"] の始め〜C["confirm"] の終わり）と、前走が地方・海外の馬の除外を当てる。"""
    if (full["yr"] > C["confirm"][1]).any():
        raise ValueError("登録の期間の後の行がある")
    pv_all = prev_lookup(full)
    pr = full["prev_region"].astype("string").fillna("").to_numpy()
    keep = (full["yr"].to_numpy() >= C["discovery"][0]) & ~np.isin(pr, ["地方", "海外"])
    d, pv = full[keep].reset_index(drop=True), pv_all[keep].reset_index(drop=True)
    M = rule_masks(d, pv)
    yr = d["yr"].to_numpy()
    clus = (d["date"].astype(str).str[:10] + "|" + d["venue"].astype(str)).to_numpy()   # ブートストラップで束ねる単位（開催日 × 場）
    popf = pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float)
    D0, D1 = C["discovery"]
    F0, F1 = C["confirm"]
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
            f, yy, rr, yy_r = flag[m], y[m], clus[m], yr[m]
            sd_ = (yy_r >= D0) & (yy_r <= D1)                    # 採否の期間（C2: 2018〜21）
            sc_ = (yy_r >= F0) & (yy_r <= F1)                    # 確かめの期間（C2: 2022〜24）
            est = mh(st[sd_], f[sd_], yy[sd_])
            bs = boot(st[sd_], f[sd_], yy[sd_], rr[sd_], B["reps"], B["seed"])
            se = float(bs.std()) if len(bs) > 1 else None
            bc = boot(st[sc_], f[sc_], yy[sc_], rr[sc_], B["reps"], B["seed"] + 2)
            s2021 = (yy_r >= 2020) & (yy_r <= D1)            # 前走が地方・海外の馬を外せる採否の期間（2020〜21）
            per = {f"{D0}〜{D1 % 100:02d}": est, f"{F0}〜{F1 % 100:02d}": mh(st[sc_], f[sc_], yy[sc_]),
                   "2020〜21": mh(st[s2021], f[s2021], yy[s2021])}
            Ym_ = Y[m]
            shrink = {}
            for lab, sel in ((f"{D0}〜{D1 % 100:02d}", sd_), (f"{F0}〜{F1 % 100:02d}", sc_)):   # 縮小（案 C）: (k − 見込み) ÷ (n + κ)
                sf = Ym_[sel & f].sum(0)
                shrink[lab] = {"n": int(sf[0]), "excess": float((sf[1] + sf[2] + sf[3] - sf[7]) / (sf[0] + C["shrink_kappa"])) if sf[0] else None}
            s24 = yy_r == 2024
            b24 = boot(st[s24], f[s24], yy[s24], rr[s24], B["reps"], B["seed"] + 1)
            br = {}
            keys = side_key[side][m]
            for kk in pd.unique(keys):        # 似た単位（採否の期間で見る。通常は採否に使い、重賞は並べるだけ）
                km = (keys == kk) & sd_
                if km.sum() >= C["breadth"]["min_runs"] and f[km].any() and (~f[km]).any():
                    br[kk] = mh(st[km], f[km], yy[km])
            Ym = Y[m]
            p6 = pop6[m]
            extra = {"pop6_rd": mh(st[p6 & sd_], f[p6 & sd_], yy[p6 & sd_]) if (p6 & sd_).any() else None,   # 参考: 6 番人気以下（BH の外）
                     "n_flag_by_period": {"〜2022": int(f[yy_r <= 2022].sum()), "2023〜24": int(f[yy_r >= 2023].sum())}}
            rows.append({"id": rid, "side": side, "layer": ly, "control": rid.startswith("N"), "n": int(m.sum()), "n_flag": int(f.sum()),
                         "n_flag_disc": int((f & sd_).sum()), "n_flag_conf": int((f & sc_).sum()), "shrink": shrink,
                         "rd": est, "se": se, "lo": float(np.percentile(bs, 2.5)) if len(bs) else None, "hi": float(np.percentile(bs, 97.5)) if len(bs) else None,
                         "conf_lo": float(np.percentile(bc, 2.5)) if len(bc) else None, "conf_hi": float(np.percentile(bc, 97.5)) if len(bc) else None,
                         "periods": per, "rd2024_lo": float(np.percentile(b24, 2.5)) if len(b24) else None,
                         "rd2024_hi": float(np.percentile(b24, 97.5)) if len(b24) else None, "breadth": br,
                         "flag": _stat(Ym[f]) | _roi_trim(Ym[f]), "other": _stat(Ym[~f]) | _roi_trim(Ym[~f]),
                         "base": _stat(Ym) | _roi_trim(Ym)} | extra)
            log(f"{rid}/{side}/{ly}: n {m.sum():,}・当たる {f.sum():,}・{time.time() - t0:.0f}s", flush=True)   # 差は出さない（途中で結果を見ない）
    res = judge(rows, C)
    res["roi_note"] = "回収率は確定オッズで数えた値（楽観）。2 歳戦の単勝の上乗せ（2016 夏〜2023）を含む。採否には使わない（C5）"
    try:                                      # C4 は評価だけ。落ちても本体の結果は残す
        res["honmei_eval"] = honmei_eval(d, M, y, C)
    except Exception as exc:                  # noqa: BLE001
        res["honmei_eval"] = {"error": repr(exc)}
    return res


def honmei_eval(d: pd.DataFrame, M: dict, y: np.ndarray, C: dict) -> dict:
    """C4: 実際に使う凍結の ◎（重賞は好走データ予想のタブの B。history.json）で、ルールの ⭕・⭕ でないごとの ◎ の成績を並べる（評価だけ。採否に使わない）。
    通常レースには、本番で使っている凍結の ◎ が無い（第 3 版はどのタブにも載せていない）ので並べない。"""
    hp = CR.REPO / "data/local/graded_picks/history.json"
    if not hp.exists():
        return {"note": "history.json が無い（並べない）"}
    H = json.loads(hp.read_text(encoding="utf-8"))
    pick = {(x["race_id"], int(x["B"]["honmei"]["hn"])) for v in H["rkeys"].values() for x in v["races"] if x.get("B") and x["B"].get("honmei")}
    key = list(zip(d["race_id"].astype(str), pd.to_numeric(d["hn"], errors="coerce").fillna(-1).astype(int)))
    hm = np.array([k in pick for k in key])
    yr = d["yr"].to_numpy()
    out = {"note": "評価だけ（採否に使わない）。B の ◎ は 2018〜2024 年で選んだ規則なので、この期間は当てはめ", "n_honmei": int(hm.sum())}
    for r in C["rules"]:
        app, flag = M[r["id"]]
        m = hm & app & (yr >= C["discovery"][0])
        if not m.any():
            continue
        out[r["id"]] = {"⭕ でない": {"n": int((m & flag).sum()), "place": float(y[m & flag].mean()) if (m & flag).any() else None},
                        "⭕": {"n": int((m & ~flag).sum()), "place": float(y[m & ~flag].mean()) if (m & ~flag).any() else None}}
    return out


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
    n_reg = sum(1 for r in rows if not r["control"])
    for r in rows:   # C6: 試した数による割引き（Bonferroni の p と、t ≥ 3 の目安）。報告に並べる（採否は BH）
        if not r["control"]:
            z = (r["rd"] / r["se"]) if (r["rd"] is not None and r["se"]) else 0.0
            r["discount"] = {"tests": n_reg, "p_bonf": float(min(1.0, r["p"] * n_reg)), "t": float(z), "t_ge_3": bool(abs(z) >= 3 and z < 0)}
    for r in rows:
        if r["control"]:
            continue
        D0, D1 = C["discovery"]
        F0, F1 = C["confirm"]
        a, b = r["periods"][f"{D0}〜{D1 % 100:02d}"], r["periods"][f"{F0}〜{F1 % 100:02d}"]
        c = {"1 BH": r["q"] <= C["q_max"] and (r["rd"] or 0) < 0,
             "2 効果の下限": r["rd"] is not None and abs(r["rd"]) * 100 >= C["effect_min_pt"][r["layer"]],
             "3 件数": r["n_flag_disc"] >= C["min_flag"]["discovery"] and r["n_flag_conf"] >= C["min_flag"]["confirm"],
             "3 2020〜21 も同じ向き": r["periods"].get("2020〜21") is not None and r["periods"]["2020〜21"] < 0,
             "3 最近の持続": a is not None and b is not None and b < 0 and abs(b) >= 0.5 * abs(a),
             "4 2024 年": r["rd2024_lo"] is not None and not (r["rd2024_lo"] > 0)}
        if r["side"] == "course":
            v = [x for x in r["breadth"].values() if x is not None]
            c["5 似た単位"] = bool(v) and sum(x < 0 for x in v) >= C["breadth"]["share"] * len(v)
        r["checks"] = c
        r["adopted"] = bool(all(c.values()) and not broken)
    return {"rows": rows, "null_broken": broken, "n_tests": n_reg,
            "candidates": [f"{r['id']}/{r['side']}/{r['layer']}" for r in rows if r.get("adopted")],
            "note": "C1: ここで通ったものは「候補」。採否は凍結の後の前向きの追跡だけで決める。2025・2026 年は評価だけ"}


def _git_clean(paths: list[Path]) -> bool:
    for p in paths:
        if subprocess.run(["git", "-C", str(CR.REPO), "ls-files", "--error-unmatch", str(p)], capture_output=True, check=False).returncode != 0:
            return False
        if subprocess.run(["git", "-C", str(CR.REPO), "diff", "--quiet", "HEAD", "--", str(p)], check=False).returncode != 0:
            return False
    return True


def preflight(C: dict, need_history: bool = True) -> None:
    """印の前の確かめ: 要る表がそろっているか・重い計算を流さない時刻でないか。
    11/21・22 の 07:30〜17:00 は流さない。計算は 2 時間かからないので、05:30 からは始めない（禁止の時間帯に入らないように）。"""
    need = [TH.FRAME] + ([CR.REPO / "data/local/graded_picks/history.json"] if need_history else []) + \
        [CR.REPO / "data/page_reference/tables" / str(y) / "race_result_flat.parquet" for y in range(2016, C["confirm"][1] + 1)]
    miss = [str(x) for x in need if not x.exists()]
    if miss:
        sys.exit(f"要る表が無い（データのつながった作業場所で流す）: {miss[:3]}")
    now = datetime.now(CR.JST)
    if now.strftime("%m-%d") in ("11-21", "11-22") and "05:30" <= now.strftime("%H:%M") < "17:00":
        sys.exit("11/21・22 の 07:30〜17:00 は重い計算を流さない")


def cmd_run() -> None:
    """1 回だけ: config とコード（このファイル・threshold・v2・run・§14 の config とコード）が commit 済みで変更が無いときだけ流す。
    要る表と時刻を確かめ、表を読んでから、計算の前に印を立てる。印があれば、途中で落ちたときも流し直さない
    （流し直すかは、統括役に返して決める）。途中のログに差は出さない。"""
    from src.research.course import v2 as V2
    if MARK.exists() or (OUTP / "result.json").exists():
        sys.exit("事前登録のルールは流し済み（1 回だけ）")
    code = Path(__file__)
    deps = [CONFIG, code, Path(TH.__file__), Path(V2.__file__), Path(CR.__file__),
            CR.REPO / "config/prereg_upset_types.json", CR.REPO / "src/research/course/upset_types.py"]   # §14 の凍結が先（§14.5 の 10）
    if not _git_clean(deps):
        sys.exit("config・コード（§14 の登録を含む）が commit されていないか、変更がある")
    C = json.loads(CONFIG.read_text(encoding="utf-8"))
    preflight(C)
    CR.wait_resources()
    preflight(C)                              # 待っている間に時刻の決まりに入ったら止める
    d = TH.load(C["confirm"][1])              # 読み込みは印の前（ここで落ちても 1 回を使わない）
    OUTP.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(CR.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deps + [TH.FRAME]}   # 層の作り（frame・コード）の sha
    MARK.write_text(json.dumps({"started_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "git_head": head, "sha256": sha}), encoding="utf-8")
    res = evaluate(d, C)
    res |= {"ran_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "git_head": head, "sha256": sha,
            "discovery": C["discovery"], "confirm": C["confirm"]}
    (OUTP / "result.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    MARK.write_text(json.dumps(json.loads(MARK.read_text(encoding="utf-8")) | {"finished_at": res["ran_at"]}), encoding="utf-8")
    print("result", OUTP / "result.json", "候補", res["candidates"], "帰無が壊れている" if res["null_broken"] else "", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run()


if __name__ == "__main__":
    main()
