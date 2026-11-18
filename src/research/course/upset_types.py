"""穴馬が勝つレースの型（段 1）と、穴馬の中の型（段 2）。計画書: docs/ideas/course_class_handicapper.md §14。

- 登録: config/prereg_upset_types.json（流す前に commit する。git で確かめる）。期間は採否 2018〜21・確かめ 2022〜24（戦略 C2・C7）。
- 見込み: 馬ごとの勝つ確率 = 確定の単勝オッズの帯ごとの勝率（採否の期間の母集団で当てはめ）を、レースの中で和 1 に直したもの。
  1/オッズの比例だけでは favourite-longshot bias が残り、頭数・クラスと交絡するので、帯ごとの実の勝率で較正する（レビューの指摘 2）。
- 段 1: レースごとに「4 番人気以下が 1 着か」。型のレースと型でないレースを、層（年 × 見込みの帯 × 頭数の帯 × クラスの群。
  検定する変数は層から外す）の中で比べた Mantel–Haenszel のリスク差。区間と p は、開催日 × 場で束ねたポアソンのブートストラップ。
- 段 2: 4 番人気以下が勝ったレースの、4 番人気以下の馬だけで、勝ち（0/1）− 見込みの取り分（p_i ÷ Σ_{4 番人気以下} p_j）の平均を、
  登録したルール（§13 の rule_masks）に当たる馬で見る（条件付きロジットの考え。勝った穴馬が必ずよく見える偏りは、取り分を引くので入らない）。
- 採否: BH（段ごと）・効果の下限・件数・持続（2022〜24 で同じ向きかつ半分以上）・2024 年が逆向きでない。負の対照が通れば、その段を使わない。
  通ったものは「候補」で、採否は凍結の後の前向きの追跡だけ（C1）。2025・2026 年は評価だけ。
- 件数の表（counts）は、結果を見ずに数えられる（型のレース数・4 番人気以下の頭数だけ）。凍結の前に検出力の目安を出すために使う。
使い方: python -m src.research.course.upset_types counts   # 件数と検出力の目安（結果は見ない）
       python -m src.research.course.upset_types run      # 1 回だけ
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.course import prereg_rules as PR
from src.research.course import run as CR
from src.research.course import threshold as TH

CONFIG = CR.REPO / "config/prereg_upset_types.json"
OUTU = CR.OUT / "upset"
MARK = OUTU / "RAN_UPSET_TYPES.json"


def prepare(full: pd.DataFrame, C: dict) -> pd.DataFrame:
    """期間・面・見込み・人気の列を足した表（取消は frame に無い）。オッズの無い馬のいるレースは外す（見込みを作れない）。"""
    d = full[(full["yr"] >= C["discovery"][0]) & (full["yr"] <= C["confirm"][1])].copy()
    d["face"] = np.where(d["clsn"] >= 6, "graded", np.where(d["clsn"].between(0, 5), "course", ""))
    d = d[d["face"] != ""]
    o = pd.to_numeric(d["odds_final"], errors="coerce")
    bad = o.isna() | (o <= 0)
    d = d[~d["race_id"].isin(d.loc[bad, "race_id"])].reset_index(drop=True)
    d["odds"] = pd.to_numeric(d["odds_final"], errors="coerce")
    d["pop"] = pd.to_numeric(d["pop_final"], errors="coerce")
    d["win"] = (pd.to_numeric(d["fin"], errors="coerce") == 1).astype(float)   # 同着の 1 着は両方 1
    d["long"] = d["pop"] >= C["longshot_pop"]
    pr = d["prev_region"].astype("string").fillna("").to_numpy() if "prev_region" in d.columns else np.full(len(d), "")
    d["prev_away"] = np.isin(pr, ["地方", "海外"])                              # 段 2 では §13 と同じく外す（2020 年以降だけ判定できる）
    rc = d["race_class"].astype(str) if "race_class" in d.columns else pd.Series("", index=d.index)
    d["two"] = rc.str.contains("[2２]歳", regex=True).to_numpy()            # frame の race_class は全角（「サラ系２歳 未勝利」）
    disc = (d["yr"] <= C["discovery"][1]).to_numpy()
    band = np.digitize(d["odds"].to_numpy(), C["odds_edges"])
    rate = {}
    for fc in ("course", "graded"):
        m = disc & (d["face"] == fc).to_numpy()
        n = np.bincount(band[m], minlength=len(C["odds_edges"]) + 1)
        k = np.bincount(band[m], weights=d["win"].to_numpy()[m], minlength=len(C["odds_edges"]) + 1)
        rate[fc] = np.where(n >= 30, k / np.maximum(n, 1), np.nan)
        rate[fc] = pd.Series(rate[fc]).interpolate(limit_direction="both").to_numpy()   # 件数の少ない帯は隣から
    raw = np.where(d["face"] == "course", rate["course"][band], rate["graded"][band])
    d["p"] = raw / pd.Series(raw).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    inv = 1.0 / d["odds"]
    d["p_prop"] = inv / inv.groupby(d["race_id"]).transform("sum")              # 感度: 1/オッズの比例（並べるだけ）
    d["clus"] = d["date"].astype(str).str[:10] + "|" + d["venue"].astype(str)
    return d


def race_table(d: pd.DataFrame, C: dict) -> pd.DataFrame:
    lw = (d["long"] & (d["win"] > 0)).astype(float)
    d2 = d.assign(lw=lw, pl=d["p"].where(d["long"], 0.0), ppl=d["p_prop"].where(d["long"], 0.0))
    fcol = "field_decl" if "field_decl" in d.columns else "field"
    g = d2.groupby("race_id", sort=False)
    r = g.agg(face=("face", "first"), yr=("yr", "first"), clus=("clus", "first"), clsn=("clsn", "first"), field=(fcol, "first"),
              two=("two", "first"), y=("lw", "max"), e=("pl", "sum"), e_prop=("ppl", "sum"))
    r["field"] = pd.to_numeric(r["field"], errors="coerce")
    return r.reset_index()


def type_flags(r: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"R2": (r["field"] >= 16).to_numpy(), "R3": (r["clsn"] <= 1).to_numpy(), "R4": (r["clsn"] == 5).to_numpy()}


def strata_r(r: pd.DataFrame, drop: str) -> np.ndarray:
    eb = np.digitize(r["e"].to_numpy(), np.arange(0.05, 1.0, 0.05))       # 見込みの帯は 0.05 刻み（レビューの指摘 2）
    fb = np.digitize(r["field"].fillna(12).to_numpy(), [10.5, 14.5])
    cg = np.select([r["clsn"] <= 1, r["clsn"] <= 3, r["clsn"] <= 5], ["a", "b", "c"], "g")
    parts = [r["yr"].astype(str), pd.Series(eb).astype(str)]
    if drop != "class":                                                    # 2 歳戦か（単勝の上乗せ）。R3 は 2 歳戦とほぼ重なるので、クラスと一緒に外す
        parts.append(r["two"].astype(str))
    if drop != "field":
        parts.append(pd.Series(fb).astype(str))
    if drop != "class":
        parts.append(pd.Series(cg))
    s = parts[0].reset_index(drop=True)
    for x in parts[1:]:
        s = s + "|" + x.reset_index(drop=True)
    return pd.factorize(s)[0]


def _ci(bs: np.ndarray) -> tuple[float | None, float | None]:
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) else (None, None)


def stage1(r: pd.DataFrame, C: dict) -> list[dict]:
    T = type_flags(r)
    D0, D1 = C["discovery"]
    F0, F1 = C["confirm"]
    B = C["bootstrap"]
    rows = []
    hh = pd.util.hash_pandas_object(r["race_id"].astype(str), index=False).to_numpy()
    ctrls = {f"N{i + 1}": ((hh >> (i + 1)) % 2 == 1) for i in range(3)}
    tests = [(t["id"], t["face"], t["direction"], t["drop"], t.get("in_bh", True)) for t in C["stage1"]] + \
        [(k, fc, "±", "", True) for k in ctrls for fc in ("course", "graded")]
    for tid, fc, dirn, drop, in_bh in tests:
        fm = (r["face"] == fc).to_numpy()
        flag = (T.get(tid) if tid in T else ctrls[tid])
        st = strata_r(r, drop)
        yr = r["yr"].to_numpy()
        y = r["y"].to_numpy()
        cl = r["clus"].to_numpy()
        sd, sc = fm & (yr >= D0) & (yr <= D1), fm & (yr >= F0) & (yr <= F1)
        yres = y - r["e"].to_numpy()                       # 残差（観測 − 見込み）で比べる。層の中の見込みのずれを引く（レビューの指摘 2）
        est = PR.mh(st[sd], flag[sd], yres[sd]) if sd.any() else None
        bs = PR.boot(st[sd], flag[sd], yres[sd], cl[sd], B["reps"], B["seed"])
        conf = PR.mh(st[sc], flag[sc], yres[sc]) if sc.any() else None
        s24 = fm & (yr == 2024)
        b24 = PR.boot(st[s24], flag[s24], yres[s24], cl[s24], B["reps"], B["seed"] + 1)
        balance = PR.mh(st[sd], flag[sd], r["e"].to_numpy()[sd]) if sd.any() else None   # 層の中の型の内外の見込みの差（釣り合いの確かめ）
        two = r["two"].to_numpy()
        no2 = sd & ~two
        sens_no2 = PR.mh(st[no2], flag[no2], yres[no2]) if no2.any() else None          # 感度: 2 歳戦を除く
        lv = {}
        for lab, sel in (("採否", sd), ("確かめ", sc)):
            m = sel & flag
            lv[lab] = {"races": int(m.sum()), "obs": float(y[m].mean()) if m.any() else None, "exp": float(r["e"].to_numpy()[m].mean()) if m.any() else None,
                       "exp_prop": float(r["e_prop"].to_numpy()[m].mean()) if m.any() else None}
        rows.append({"id": tid, "face": fc, "direction": dirn, "control": tid.startswith("N"), "in_bh": in_bh, "rd": est, "se": float(bs.std()) if len(bs) > 1 else None,
                     "ci": _ci(bs), "rd_conf": conf, "ci2024": _ci(b24), "n_type_disc": int((sd & flag).sum()), "n_type_conf": int((sc & flag).sum()),
                     "levels": lv, "balance": balance, "sens_no_2yo": sens_no2})
    return rows


def stage2(d: pd.DataFrame, C: dict, M: dict) -> list[dict]:
    """4 番人気以下が勝ったレースの、4 番人気以下の馬だけ。勝ち − 取り分 を、ルールに当たる馬（⭕ でない）で平均。"""
    lw = d["long"] & (d["win"] > 0)
    races = set(d.loc[lw, "race_id"])
    m0 = d["race_id"].isin(races).to_numpy() & d["long"].to_numpy() & ~d["prev_away"].to_numpy()
    share = (d["p"] / d["p"].where(d["long"]).groupby(d["race_id"]).transform("sum")).to_numpy()
    resid = d["win"].to_numpy() - share
    yr = d["yr"].to_numpy()
    cl = d["clus"].to_numpy()
    D0, D1 = C["discovery"]
    F0, F1 = C["confirm"]
    B = C["bootstrap"]
    rng_seed = B["seed"] + 10
    out = []
    band = np.digitize(d["odds"].to_numpy(), C["odds_edges"])
    for rid, fc, in_bh in [(t["id"], t["face"], t.get("in_bh", True)) for t in C["stage2"]]:
        app, flag = M[rid]
        base = m0 & app & (d["face"] == fc).to_numpy()
        res = {}
        sdd = base & (yr >= D0) & (yr <= D1)
        vs = PR.mh(band[sdd], flag[sdd], resid[sdd]) if (sdd & flag).any() and (sdd & ~flag).any() else None   # 並べるだけ: 当たる馬 − 当たらない馬（オッズの帯の中）
        for lab, sel in (("採否", (yr >= D0) & (yr <= D1)), ("確かめ", (yr >= F0) & (yr <= F1)), ("2024", yr == 2024)):
            m = base & sel & flag
            v = resid[m]
            if not len(v):
                res[lab] = {"n": 0, "mean": None, "ci": (None, None), "se": None}
                continue
            cc, _ = pd.factorize(cl[m])
            rng = np.random.default_rng(rng_seed)
            bs = []
            for _ in range(B["reps"]):
                w = rng.poisson(1.0, cc.max() + 1)[cc]
                if w.sum():
                    bs.append(float((w * v).sum() / w.sum()))
            bs = np.array(bs)
            res[lab] = {"n": int(m.sum()), "mean": float(v.mean()), "ci": _ci(bs), "se": float(bs.std()) if len(bs) > 1 else None,
                        "wins": int(d["win"].to_numpy()[m].sum()), "exp_wins": float(share[m].sum())}
        out.append({"id": rid, "face": fc, "control": rid.startswith("N"), "in_bh": in_bh, "res": res, "vs_unflagged": vs})
    return out


def judge(rows: list[dict], C: dict, stage: str, broken_from: bool = False) -> dict:
    from scipy.stats import norm
    for r in rows:
        if stage == "1":
            est, se, dirn = r["rd"], r["se"], r["direction"]
        else:
            est, se, dirn = r["res"]["採否"]["mean"], r["res"]["採否"]["se"], "-"
        z = (est / se) if (est is not None and se) else 0.0
        r["z"] = float(z)
        r["p"] = float(2 * norm.sf(abs(z))) if r["control"] else float(norm.sf(z) if dirn == "+" else norm.cdf(z))
    for ctl in (False, True):
        sub = [r for r in rows if r["control"] == ctl and r.get("in_bh", True)]   # 検出力の足りない組は BH に入れない（並べるだけ）
        q = TH.bh(np.array([r["p"] for r in sub])) if sub else []
        for r, qq in zip(sub, q):
            r["q"] = float(qq)
    for r in rows:
        r.setdefault("q", None)
    broken = broken_from or any(r["q"] is not None and r["q"] <= C["q_max"] for r in rows if r["control"])
    n_reg = sum(1 for r in rows if not r["control"] and r.get("in_bh", True))
    cands = []
    for r in rows:
        if r["control"]:
            continue
        if not r.get("in_bh", True):
            r["checks"], r["candidate"] = {"検出力が足りない（並べるだけ）": False}, False
            r["discount"] = None
            continue
        sgn = 1 if (stage == "1" and r["direction"] == "+") else -1
        if stage == "1":
            a, b, lo24, hi24 = r["rd"], r["rd_conf"], *r["ci2024"]
            n_ok = r["n_type_disc"] >= C["min_n"]["stage1"]["discovery"] and r["n_type_conf"] >= C["min_n"]["stage1"]["confirm"]
            lim = C["effect_min_pt"]["stage1"]
        else:
            a, b = r["res"]["採否"]["mean"], r["res"]["確かめ"]["mean"]
            lo24, hi24 = r["res"]["2024"]["ci"]
            n_ok = r["res"]["採否"]["n"] >= C["min_n"]["stage2"]["discovery"] and r["res"]["確かめ"]["n"] >= C["min_n"]["stage2"]["confirm"]
            lim = C["effect_min_pt"]["stage2"]
        c = {"1 BH": r["q"] <= C["q_max"] and a is not None and a * sgn > 0,
             "2 効果の下限": a is not None and abs(a) * 100 >= lim,
             "3 件数": bool(n_ok),
             "4 最近の持続": a is not None and b is not None and b * sgn > 0 and abs(b) >= 0.5 * abs(a),
             "5 2024 年": lo24 is not None and not ((lo24 > 0) if sgn < 0 else (hi24 < 0))}
        r["checks"] = c
        r["discount"] = {"tests": n_reg, "p_bonf": float(min(1.0, r["p"] * n_reg)), "t_ge_3": bool(r["z"] * sgn >= 3)}
        r["candidate"] = bool(all(c.values()) and not broken)
        if r["candidate"]:
            cands.append(f"{r['id']}/{r['face']}")
    return {"rows": rows, "null_broken": broken, "n_tests": n_reg, "candidates": cands}


def evaluate(full: pd.DataFrame, C: dict) -> dict:
    d = prepare(full, C)
    r = race_table(d, C)
    s1 = judge(stage1(r, C), C, "1")
    M = PR.rule_masks(d, _prev_for(full, d))
    s2 = judge(stage2(d, C, M), C, "2", broken_from=s1["null_broken"])   # 段 1 の負の対照が通れば、段 2 も使わない
    return {"stage1": s1, "stage2": s2, "note": "C1: 候補は前向きの追跡で採否を決める。2025・2026 年は評価だけ。段 1 の型の内外の交互作用は行わない（§14.6）"}


def counts(full: pd.DataFrame, C: dict) -> dict:
    """結果（着順）を使わずに数える: 面 × 型 × 期間のレース数と、段 2 の当てはまる穴馬の数の目安。検出力の目安（最小の検出できる差）も。"""
    d = prepare(full, C)
    r = race_table(d, C)
    T = type_flags(r)
    out = {}
    for t in C["stage1"]:
        fm = (r["face"] == t["face"]).to_numpy()
        f = T[t["id"]]
        row = {}
        for lab, (a, b) in (("採否", C["discovery"]), ("確かめ", C["confirm"])):
            sel = fm & (r["yr"] >= a).to_numpy() & (r["yr"] <= b).to_numpy()
            n1, n0 = int((sel & f).sum()), int((sel & ~f).sum())
            pbar = float(r["e_prop"].to_numpy()[sel].mean()) if sel.any() else 0.0   # 1/オッズの比例の見込みから作る（着順は使わない）
            se = float(np.sqrt(pbar * (1 - pbar) * (1 / max(n1, 1) + 1 / max(n0, 1))))
            row[lab] = {"type_races": n1, "other_races": n0, "mde_pt": round(2.8 * se * 100, 2)}   # 両側 5%・検出力 80% の目安
        out[f"{t['id']}/{t['face']}"] = row
    # 段 2: 当たる穴馬が「穴が勝ったレース」に入る数の見込み = Σ_レース（4 番人気以下の勝つ見込み e_prop）×（当たる穴馬の頭数）
    pv = _prev_for(full, d)
    M = PR.rule_masks(d, pv)
    rid = d["race_id"].to_numpy()
    ep = pd.Series(r["e_prop"].to_numpy(), index=r["race_id"]).reindex(rid).to_numpy()
    yr = d["yr"].to_numpy()
    for t in C["stage2"]:
        app, flag = M[t["id"]]
        m = app & flag & d["long"].to_numpy() & ~d["prev_away"].to_numpy() & (d["face"] == t["face"]).to_numpy()
        row = {}
        for lab, (a, b) in (("採否", C["discovery"]), ("確かめ", C["confirm"])):
            sel = m & (yr >= a) & (yr <= b)
            row[lab] = {"exp_n": round(float(ep[sel].sum()), 1)}
        out[f"段2 {t['id']}/{t['face']}"] = row
    return out


def _prev_for(full: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """前走の表を full 全体から引き、d の行の順にそろえる（race_id × horse_id の鍵で引き直す）。"""
    pv = PR.prev_lookup(full)
    pv.index = full[["race_id", "horse_id"]].astype(str).agg("#".join, axis=1).to_numpy()
    keyd = d[["race_id", "horse_id"]].astype(str).agg("#".join, axis=1)
    return pv.loc[~pv.index.duplicated()].reindex(keyd.to_numpy()).reset_index(drop=True)


def _deps() -> list[Path]:
    from src.research.course import v2 as V2
    return [CONFIG, Path(__file__), Path(PR.__file__), Path(PR.CONFIG), Path(TH.__file__), Path(V2.__file__), Path(CR.__file__)]


def cmd_run() -> None:
    if MARK.exists() or (OUTU / "result.json").exists():
        sys.exit("段 1・段 2 は流し済み（1 回だけ）")
    deps = _deps()
    if not PR._git_clean(deps):
        sys.exit("config・コードが commit されていないか、変更がある")
    C = json.loads(CONFIG.read_text(encoding="utf-8"))
    if PR.MARK.exists():                      # §13 を先に流したなら、§14 は §13 の印に記録した sha のまま（§13 の結果を見て直していない）
        rec = json.loads(PR.MARK.read_text(encoding="utf-8")).get("sha256", {})
        for p in (CONFIG, Path(__file__)):
            if rec.get(p.name) != hashlib.sha256(p.read_bytes()).hexdigest():
                sys.exit(f"§13 を流した後に {p.name} が変わっている（§14 は §13 の前に凍結する）")
    PR.preflight(C, need_history=False)
    CR.wait_resources()
    PR.preflight(C, need_history=False)
    full = TH.load(C["confirm"][1])           # 読み込みは印の前
    OUTU.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "-C", str(CR.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deps + [TH.FRAME]}
    MARK.write_text(json.dumps({"started_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "git_head": head, "sha256": sha}), encoding="utf-8")
    res = evaluate(full, C)
    res |= {"ran_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "git_head": head, "sha256": sha}
    (OUTU / "result.json").write_text(json.dumps(res, ensure_ascii=False, default=float), encoding="utf-8")
    MARK.write_text(json.dumps(json.loads(MARK.read_text(encoding="utf-8")) | {"finished_at": res["ran_at"]}), encoding="utf-8")
    print("result", OUTU / "result.json", "段 1 の候補", res["stage1"]["candidates"], "段 2 の候補", res["stage2"]["candidates"], flush=True)


def cmd_counts() -> None:
    C = json.loads(CONFIG.read_text(encoding="utf-8"))
    print(json.dumps(counts(TH.load(C["confirm"][1]), C), ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "counts"])
    a = ap.parse_args()
    cmd_run() if a.cmd == "run" else cmd_counts()


if __name__ == "__main__":
    main()
