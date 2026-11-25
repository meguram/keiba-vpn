"""RC-2 の段 S' と確かめ（本物の着順。各 1 回だけ）。事前登録 config/rules_council/prereg_rc2.json（版 3）。

版 3（統括役の指示・1 号の了承）: 2016〜24 の開催を 70% の探す側・30% の確かめの側に分ける（engine.search70）。
手順（prereg の順に、段ごとに分けて流す。各段の前に prereg と道具のコードが commit 済みで変更が無いことを確かめる）:
  null    : 探す側の本物の着順のまま、水準の行を層の中で入れ替えた帰無 200 回（種 20261124+1〜200）の max-T（方向ごと）を、
            再現の種 1000〜1004 のそれぞれで出す（種ごとに検定する水準の集合が違うため）。<COURSE_OUT>/rules_council/rc2_null.json。
            担当がこの値を prereg の stage_S.gate1_line に写して commit してから次へ。
  search  : 種ごとに 1 段目（z_band ≥ その種の線）→ 2 段目（z_be が Holm）。候補の条件（再現 5 本のうち 4 本・5 本すべて内・前向きの量）。参考の表。
            gate1_line が rc2_null.json と一致し、commit 済みでなければ止める。
  confirm : 探した候補（z_be の上位 3 つまで。rc2_search の結果を commit してから）を、確かめの側 30% に 1 回だけ当てる。
z は、モデルの分散と馬で束ねた分散 Σ_馬(Σx)² の大きい方で割る（同じ馬の繰り返しを入れる）。
台帳: family rules_council。null・search は purpose select、confirm は confirm（期間は 2016〜24。2022〜24 は record の期間）。
使い方: KEIBA_SESSION=rules-council python -m src.research.rules_council.rc2_real null|search|confirm（nice -n 19）
費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E
from src.research.rules_council import rc2 as RC
from src.research.rules_council import rc_frame as F
from src.research.rules_council.rc2s import ALPHA2, FWD_MIN, holm_pass

REPO = F.CR.REPO
PREREG = REPO / "config/rules_council/prereg_rc2.json"
OUT_REPO = REPO / "docs/operations/retrain_trials/rules_council_rc2.json"
FROZEN = ([PREREG] + [REPO / f"src/research/rules_council/{x}.py" for x in ("engine", "rc_frame", "rc2", "rc2s", "rc2_real")]
          + [REPO / "src/research/graded/model.py", REPO / "src/research/graded/catalog.py", REPO / "src/research/course/longshot_rules.py",
             REPO / "config/longshot_rules/prereg.json"])      # 水準の定義（NO_INFO・型の規則）に効くファイルも凍結する
NULL_OUT = F.OUTW / "rc2_null.json"
SEARCH_OUT = F.OUTW / "rc2_search.json"
EMIN = 1000.0
N_PERM, PERM_SEED = 200, 20261124
N_SEEDS = 5
F_ALPHA, F_N_MIN, F_MAX_CAND = 0.01, 300, 3
T60_NOTE = {"H06": "頭数（取消の後に変わりうる。T-60 では出走表の頭数）", "H10": "クッション値（当日の朝に発表）",
            "I03": "継続騎乗・乗り替わり（出馬表で確定）", "I04": "乗り替わりの格（同上）", "H08": "開催の何日目（前日に確定）"}


def check_split() -> None:
    """分け方が prereg の sha256 と同じか（全走の表を作り直して開催の集合が変わったら止める）。"""
    import hashlib
    keys = np.array(sorted(pd.read_parquet(F.OUTW / "full.parquet", columns=["kaisai"])["kaisai"].unique()))
    sha = hashlib.sha256(json.dumps(sorted(keys[E.search70(keys)].tolist())).encode()).hexdigest()
    if sha != prereg()["split"]["search_kaisai_sha256"]:
        raise SystemExit("開催の 70/30 の分け方が prereg の sha256 と違う")


def world(pre_seed: int) -> RC.RC2:
    R = RC.RC2(pre_seed, mode="k70")
    R.elig = np.flatnonzero(R.Eq >= EMIN)
    return R


def clean_or_stop(paths) -> None:
    st = subprocess.run(["git", "status", "--porcelain", "--", *map(str, paths)], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if st:
        raise SystemExit(f"commit していない変更がある（prereg・道具のコード）。commit してから流す:\n{st}")


def guard(purpose: str, reason: str):
    from src.ops import data_ledger as DL
    return DL.guard(tool="rules_council.rc2_real", family="rules_council", version="rc2_v3", period=("2016-01-01", "2024-12-31"),
                    purpose=purpose, reads=("着順", "払戻"), reason=reason, session="rules-council", frozen_paths=FROZEN)


def prereg() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def cmd_null() -> None:
    clean_or_stop(FROZEN)
    check_split()
    P = prereg()
    t0 = time.time()
    with guard("select", "RC-2 段 S' の 1 段目の線（探す側 70% の本物の着順のまま水準を入れ替えた帰無 200 回 × 再現 5 本）") as tok:
        out = {"n_perm": N_PERM, "perm_seeds": [PERM_SEED + 1, PERM_SEED + N_PERM], "emin": EMIN, "seeds": {}}
        for s in range(N_SEEDS):
            R = world(s)
            if s == 0:
                assert len(R.elig) == int(P["stage_B"]["result"]["n_elig"]["1000"]), ("検定する水準の数が prereg と違う", len(R.elig))
            d = R.P.d
            win, rt = d["win"].to_numpy(float), d["rt"].to_numpy(float)
            assert not np.isnan(win).any(), "探す側に着順の欠けがある"
            allr = np.arange(len(d))
            rho = R.rho(allr, win)
            mu, md = [], []
            for i in range(N_PERM):
                E.check_stop()
                st = R.stats(R.permute(PERM_SEED + 1 + i), allr, win, rt, rho, robust=True)
                z = st["z_band"][R.elig]
                mu.append(float(z.max())); md.append(float((-z).max()))
            out["seeds"][str(s)] = {"n_elig": int(len(R.elig)), "t_up": float(np.quantile(mu, 0.95)), "t_down": float(np.quantile(md, 0.95)),
                                    "max_up": mu, "max_down": md}
            print("seed", s, out["seeds"][str(s)]["n_elig"], out["seeds"][str(s)]["t_up"], out["seeds"][str(s)]["t_down"], round(time.time() - t0), flush=True)
        NULL_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=NULL_OUT)
    print({s: {k: v for k, v in x.items() if not k.startswith("max_")} for s, x in out["seeds"].items()})


def committed_lines() -> dict:
    g = prereg()["stage_S"].get("gate1_line")
    if not isinstance(g, dict) or "seeds" not in g:
        raise SystemExit("prereg の stage_S.gate1_line がまだ書かれていない（null を流して写し、commit してから）")
    nul = json.loads(NULL_OUT.read_text(encoding="utf-8"))
    for s in map(str, range(N_SEEDS)):
        a, b = g["seeds"][s], nul["seeds"][s]
        for k in ("t_up", "t_down", "n_elig"):
            if abs(float(a[k]) - float(b[k])) > 1e-9:
                raise SystemExit(f"prereg の線が rc2_null.json と違う（種 {s} の {k}）")
    for k in ("n_perm", "perm_seeds", "emin"):
        if g.get(k) != nul[k]:
            raise SystemExit(f"prereg の {k} が rc2_null.json と違う")
    return g


def two_stage(R: RC.RC2, win, rt, rows, t_up, t_dn) -> dict:
    rho = R.rho(rows, win)
    st = R.stats(R.flat, rows, win, rt, rho, robust=True)
    el = R.elig
    up1 = el[st["z_band"][el] >= t_up]
    dn1 = el[-st["z_band"][el] >= t_dn]
    s2 = up1[holm_pass(st["z_be"][up1], ALPHA2)]
    return {"st": st, "up1": up1, "down1": dn1, "s2": s2, "rho": rho}


def lv_rows(R: RC.RC2, j: int) -> np.ndarray:
    return np.flatnonzero((R.flat == j).any(1))


def summary_rows(d: pd.DataFrame, rows: np.ndarray, e: np.ndarray) -> dict:
    x = d.iloc[rows]
    n = len(x)
    if n == 0:
        return {"n": 0}
    k = x["kidx"].to_numpy()
    rt = x["rt"].to_numpy(float)
    roi, lo, hi = E.boot_ci(k, rt, np.ones(n))
    mr, mlo, mhi = E.boot_ci(k, x["win"].to_numpy(float), e)
    cap = np.quantile(rt[rt > 0], 0.99) if (rt > 0).any() else 0
    return {"n": n, "wins": int(x["win"].sum()), "roi": roi, "roi_ci95": [lo, hi], "mr_band": mr, "mr_band_ci95": [mlo, mhi],
            "roi_trim_top1pct": float(np.minimum(rt, cap).sum() / n), "roi_x0.86": roi * 0.86,
            "by_year": {int(y): {"n": int(len(g)), "roi": float(g["rt"].sum() / len(g))} for y, g in x.groupby("year")}}


def level_row(R, r, j) -> dict:
    st = r["st"]
    return {"level": R.P.level_name(int(j)), "family": R.P.level_name(int(j))[0], "z_band": float(st["z_band"][j]), "z_be": float(st["z_be"][j]),
            "E": float(R.Eq[j]), "n": int(st["N"][j]), "roi": float(st["RT"][j] / max(st["N"][j], 1)),
            "mr_band": float(st["W"][j] / max(st["Eb"][j], 1e-9)), "fwd_n_2024": int(R.fwd_n[j]),
            "t60_note": T60_NOTE.get(R.P.level_name(int(j)).split("=")[0])}


def cmd_search() -> None:
    clean_or_stop(FROZEN)
    check_split()
    g = committed_lines()
    if SEARCH_OUT.exists() or (OUT_REPO.exists() and "search" in json.loads(OUT_REPO.read_text(encoding="utf-8"))):
        raise SystemExit("段 S' はもう流した（1 回だけ）")
    t0 = time.time()
    with guard("select", "RC-2 段 S'（探す側 70% の本物の着順で 1 回だけ。1 段目・2 段目・候補の条件・参考の表）") as tok:
        mains = {}
        for s in range(N_SEEDS):
            E.check_stop()
            R = world(s)
            d = R.P.d
            win, rt = d["win"].to_numpy(float), d["rt"].to_numpy(float)
            ln = g["seeds"][str(s)]
            assert len(R.elig) == int(ln["n_elig"]), ("検定する水準の数が線と違う", s)
            mains[s] = (R, two_stage(R, win, rt, np.arange(len(d)), float(ln["t_up"]), float(ln["t_down"])))
            print("seed", s, "up1", len(mains[s][1]["up1"]), "down1", len(mains[s][1]["down1"]), "s2", len(mains[s][1]["s2"]), round(time.time() - t0), flush=True)
        R0, r0 = mains[0]
        d0 = R0.P.d
        pre_all = np.load(F.OUTW / "full_pre.npy")
        assert pre_all.shape[0] >= N_SEEDS
        all_in = (pre_all[:N_SEEDS][:, R0.P.full_rows] < RC.PRE_HI).all(0)
        pass_by_seed = {s: {mains[s][0].P.level_name(int(j)) for j in mains[s][1]["s2"]} for s in mains}
        stage2, cands = [], []
        for j in r0["s2"]:
            rec = level_row(R0, r0, j)
            n_rep = sum(rec["level"] in pass_by_seed[s] for s in range(N_SEEDS))
            rows = lv_rows(R0, int(j))
            ai = rows[all_in[rows]]
            rec |= {"replicates_pass": n_rep, "n_all_five_in": int(len(ai)),
                    "roi_all_five_in": float(d0["rt"].to_numpy()[ai].sum() / max(len(ai), 1))}
            rec["candidate"] = bool(n_rep >= 4 and rec["roi_all_five_in"] >= 1.0 and rec["fwd_n_2024"] >= FWD_MIN)
            stage2.append(rec)
            if rec["candidate"]:
                cands.append(rec)
        cands = sorted(cands, key=lambda x: -x["z_be"])
        e0 = r0["rho"] * R0.q
        ref = {"baseline_all": summary_rows(d0, np.arange(len(d0)), e0)}
        if cands:
            idx = [i for i in r0["s2"] if R0.P.level_name(int(i)) in {c["level"] for c in cands[:F_MAX_CAND]}]
            ur = np.unique(np.concatenate([lv_rows(R0, int(i)) for i in idx]))
            ref["frozen_union"] = summary_rows(d0, ur, e0)
        # 参考: 探す側の中の開催 5 分割で、各側で 2 段の門を選び直した外側（合否に使わない）
        win0, rt0 = d0["win"].to_numpy(float), d0["rt"].to_numpy(float)
        N_ = RT_ = W_ = Eb_ = 0.0
        ln0 = g["seeds"]["0"]
        for k in range(5):
            trr, hor = np.flatnonzero(R0.fold != k), np.flatnonzero(R0.fold == k)
            rr = two_stage(R0, win0, rt0, trr, float(ln0["t_up"]), float(ln0["t_down"]))
            if not len(rr["s2"]):
                continue
            sel = np.zeros(R0.P.n_levels, bool)
            sel[rr["s2"]] = True
            f = R0.flat[hor]
            b = hor[np.where(f >= 0, sel[np.maximum(f, 0)], False).any(1)]
            N_ += len(b); RT_ += rt0[b].sum(); W_ += win0[b].sum(); Eb_ += (rr["rho"][b] * R0.q[b]).sum()
        ref["outer_cv_in_search"] = {"n": N_, "roi": RT_ / N_ if N_ else None, "mr_band": W_ / Eb_ if Eb_ else None}
        out = {"version": "rc2_v3", "lines": {s: {k: g["seeds"][s][k] for k in ("t_up", "t_down", "n_elig")} for s in g["seeds"]},
               "n_elig": int(len(R0.elig)), "n_up1": int(len(r0["up1"])), "n_down1": int(len(r0["down1"])), "n_stage2": int(len(r0["s2"])),
               "n_candidates": len(cands), "frozen": [c["level"] for c in cands[:F_MAX_CAND]], "stage2": stage2,
               "up1": [level_row(R0, r0, j) for j in r0["up1"]], "down1": [level_row(R0, r0, j) for j in r0["down1"]],
               "replicates": {str(s): {"n_elig": int(len(mains[s][0].elig)), "up1": int(len(mains[s][1]["up1"])), "s2": sorted(pass_by_seed[s])} for s in mains},
               "reference": ref, "seconds": round(time.time() - t0)}
        SEARCH_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        OUT_REPO.write_text(json.dumps({"search": out}, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=OUT_REPO)
    print(json.dumps({k: v for k, v in out.items() if k not in ("up1", "down1", "stage2", "replicates")}, ensure_ascii=False, indent=1)[:4000])


def cmd_confirm() -> None:
    clean_or_stop(FROZEN + [OUT_REPO])
    check_split()
    res = json.loads(OUT_REPO.read_text(encoding="utf-8"))
    if "confirm" in res:
        raise SystemExit("確かめの側はもう開いた（1 回だけ）")
    frozen = res["search"]["frozen"]
    if not frozen:
        raise SystemExit("候補が無いので確かめは流さない（prereg の打ち切りの規則）")
    with guard("confirm", f"RC-2 の確かめの側 30% に、凍結した候補 {len(frozen)} つを 1 回だけ当てる") as tok:
        out = {"frozen": frozen, "seeds": {}}
        for s in range(N_SEEDS):
            P = E.load_pop(0.0, RC.PRE_HI, pre_seed=s, exclude_types=RC.EXCLUDE_T60)
            d = P.d
            srch = E.search70(d["kaisai"].to_numpy())
            win = d["win"].to_numpy(float)
            rho = E.band_ratio(d, srch, np.where(srch, win, 0.0))      # 帯の比は探す側だけで出す（1 号の条件 4）
            flat = P.flat()
            names = {P.level_name(j): j for j in range(P.n_levels)}
            sel = np.zeros(P.n_levels, bool)
            for nm in frozen:
                if nm not in names:
                    raise SystemExit(f"凍結した水準 {nm} が種 {s} の母集団に無い")
                sel[names[nm]] = True
            hit = np.where(flat >= 0, sel[np.maximum(flat, 0)], False).any(1)
            b = np.flatnonzero(hit & ~srch)
            x = d.iloc[b]
            e = rho[b] * x["q"].to_numpy()
            n = len(b)
            roi = float(x["rt"].sum() / max(n, 1))
            k = pd.factorize(x["kidx"])[0]
            nk = int(k.max()) + 1 if n else 0
            p = 1.0
            if nk:
                A = np.bincount(k, weights=x["win"].to_numpy(float), minlength=nk)
                B = np.bincount(k, weights=e, minlength=nk)
                idx = np.random.default_rng(20261124).integers(0, nk, size=(2000, nk))
                r = A[idx].sum(1) / B[idx].sum(1)
                p = float((np.sum(r <= 1.0) + 1) / (len(r) + 1))     # 百分位の片側 p（ブートストラップの比が 1 以下の割合）
            by_period = {nm_: {"n": int(m.sum()), "roi": float(x["rt"].to_numpy()[m].sum() / max(m.sum(), 1)),
                               "mr_band": float(x["win"].to_numpy()[m].sum() / max(e[m].sum(), 1e-9))}
                         for nm_, m in {"2016〜21": x["year"].to_numpy() <= 2021, "2022〜24": x["year"].to_numpy() >= 2022}.items()}
            out["seeds"][str(s)] = {"n": n, "roi": roi, "mr_band": float(x["win"].sum() / max(e.sum(), 1e-9)), "p_boot": p,
                                    "pass": bool(roi >= 1.0 and p <= F_ALPHA and n >= F_N_MIN), "by_period": by_period}
        out["n_seeds_pass"] = sum(v["pass"] for v in out["seeds"].values())
        out["pass"] = bool(out["seeds"]["0"]["pass"] and out["n_seeds_pass"] >= 4)
        res["confirm"] = out
        OUT_REPO.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=OUT_REPO)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["null", "search", "confirm"])
    a = ap.parse_args()
    {"null": cmd_null, "search": cmd_search, "confirm": cmd_confirm}[a.cmd]()


if __name__ == "__main__":
    main()
