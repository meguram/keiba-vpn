"""tp-58 交互作用の候補発見（データ駆動）。V5 の OOF 確率に対するスコア検定で、再学習なしに『この掛け合わせ列を足すと上乗せがありそうか』を総当たりで見る。

発見期間 = 2022–2023、確認期間 = 2024（OOF が存在する年）。test 2025・2026 は使わない。市場オッズ・人気の列は除く。
基の列 = V5 の gain 重要度（2021 年までで学習）上位 K 列 + その上位 R 列のレース内偏差。操作 = 積・差（標準化後）と、正の列の対数比。

python -m src.scripts.data.tp58_discover run [--k 150] [--rel 60]
出力: $TP58_DIR/out/{candidates.parquet,summary.json,partial.json}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.tp58_score import ScoreTest

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("TP58_DIR", ROOT / "data/local/modeling/shared_cache/tp58"))
CACHE = WORKDIR / "cache"
V = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = WORKDIR / "out"
OOF = V / "v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
BAN = ("odds", "ninki", "popular", "pop_rank", "fukusho_o", "tansho")
DISC, CONF = (2022, 2023), (2024,)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load():
    F = pd.read_parquet(CACHE / "v5_frame_num.parquet")
    F["race_id"], F["horse_id"] = F["race_id"].astype(str), F["horse_id"].astype(str)
    O = pd.read_parquet(OOF)
    O["race_id"], O["horse_id"] = O["race_id"].astype(str), O["horse_id"].astype(str)
    D = F[F["year"].isin(DISC + CONF)].merge(O[KEY + ["p_win", "p_top2", "p_top3"]], on=KEY, how="inner")
    D = D.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    return D


def _bases(D: pd.DataFrame, k: int, n_rel: int):
    imp = json.loads((CACHE / "importance_gain.json").read_text(encoding="utf-8"))
    names, miss = [], {}
    for c, _ in imp:
        if any(b in c.lower() for b in BAN) or c not in D.columns:
            continue
        s = D[c]
        if s.isna().mean() > 0.3 or s.nunique(dropna=True) < 2 or float(s.std()) == 0.0:
            continue
        names.append(c)
        if len(names) >= k:
            break
    X = D[names].to_numpy(dtype="float32")
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    S = np.clip((X - mu) / sd, -5, 5)
    S = np.nan_to_num(S, nan=0.0).astype("float32")
    pos = [i for i, c in enumerate(names) if np.nanmin(X[:, i]) > 0 and np.unique(X[:100000, i]).size > 10]
    L = {}
    for i in pos:
        lg = np.log(X[:, i])
        m, s = np.nanmean(lg), np.nanstd(lg)
        L[i] = np.nan_to_num(np.clip((lg - m) / s, -5, 5), nan=0.0).astype("float32")
    # レース内偏差（上位 n_rel 列）
    races = D["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, races[1:] != races[:-1]])
    cnt = np.diff(np.r_[starts, len(races)])
    rel_names, rel_cols = [], []
    for i in range(min(n_rel, len(names))):
        mean = np.add.reduceat(S[:, i], starts) / cnt
        rel_cols.append(S[:, i] - np.repeat(mean, cnt))
        rel_names.append(f"rel({names[i]})")
    R = np.column_stack(rel_cols).astype("float32") if rel_cols else np.zeros((len(D), 0), "float32")
    return names + rel_names, np.hstack([S, R]), pos, L


def cmd_run(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = _load()
    _log(f"行={len(D)} レース={D['race_id'].nunique()} 年={sorted(D['year'].unique())}")
    names, S, pos, L = _bases(D, a.k, a.rel)
    nb = len(names)
    _log(f"基の列 {nb}（うち対数比に使える正の列 {len(pos)}）")
    year = D["year"].to_numpy()
    periods = {"disc": np.isin(year, DISC), "conf": np.isin(year, CONF)}
    fp = pd.to_numeric(D["finish_position"], errors="coerce").to_numpy()
    races = D["race_id"].to_numpy()
    tests = {
        "win": ScoreTest((fp == 1).astype(float), D["p_win"].to_numpy(), races, periods, "win"),
        "top2": ScoreTest((fp <= 2).astype(float), D["p_top2"].to_numpy(), races, periods, "binary"),
        "top3": ScoreTest((fp <= 3).astype(float), D["p_top3"].to_numpy(), races, periods, "binary"),
    }
    rows = []

    def add(block, label_a, labels_b, op):
        zz = {t: st.z(block) for t, st in tests.items()}
        for j, lb in enumerate(labels_b):
            r = {"a": label_a, "b": lb, "op": op}
            for t in tests:
                for p in ("disc", "conf"):
                    r[f"z_{t}_{p}"] = float(zz[t][p][j])
            rows.append(r)

    t0 = time.time()
    for ia in range(nb - 1):
        if ia % 20 == 0:
            _log(f"a={ia}/{nb} 候補={len(rows)} {time.time() - t0:.0f}s")
        rest = [ib for ib in range(ia + 1, nb) if not (names[ib] == f"rel({names[ia]})" or names[ia] == f"rel({names[ib]})")]
        if not rest:
            continue
        B = S[:, rest]
        add(S[:, [ia]] * B, names[ia], [names[i] for i in rest], "prod")
        add(S[:, [ia]] - B, names[ia], [names[i] for i in rest], "diff")
    pl = sorted(L)
    for x, ia in enumerate(pl[:-1]):
        rest = pl[x + 1 :]
        B = np.column_stack([L[i] for i in rest])
        add(L[ia][:, None] - B, names[ia], [names[i] for i in rest], "logratio")
    C = pd.DataFrame(rows)
    # 単独列（基の列そのもの）の z。掛け合わせが単独列の言い換えに過ぎないものを除くため
    sing = {}
    for t, st in tests.items():
        zz = st.z(S)
        for p_ in ("disc", "conf"):
            sing[(t, p_)] = pd.Series(zz[p_], index=names)
    for t in tests:
        for p_ in ("disc", "conf"):
            C[f"za_{t}_{p_}"] = C["a"].map(sing[(t, p_)])
            C[f"zb_{t}_{p_}"] = C["b"].map(sing[(t, p_)])
    C.to_parquet(OUT / "candidates.parquet", index=False)
    pd.DataFrame({f"{t}_{p_}": v for (t, p_), v in sing.items()}).to_csv(OUT / "singles.csv")
    n = len(C)
    _log(f"候補総数 {n}（3 目的で検定 → 検定数 {3 * n}）")

    # 帰無のばらつき（ほぼ全候補が無効という前提で MAD から）と選別
    summ = {"n_candidates": n, "n_tests": 3 * n, "k": a.k, "rel": a.rel, "bases": nb, "disc_years": list(DISC), "conf_years": list(CONF)}
    from scipy import stats

    sdv = {}
    for t in tests:
        for p_ in ("disc", "conf"):
            x = C[f"z_{t}_{p_}"].to_numpy()
            sdv[f"{t}_{p_}"] = float(np.median(np.abs(x - np.median(x))) * 1.4826)
    summ["null_sd_mad"] = sdv
    thr = stats.norm.isf(0.05 / (2 * 3 * n))
    summ["bonferroni_z"] = float(thr)
    sel = []
    for t in tests:
        zd = C[f"z_{t}_disc"] / sdv[f"{t}_disc"]
        zc = C[f"z_{t}_conf"] / sdv[f"{t}_conf"]
        over = C[f"z_{t}_disc"].abs() > 1.25 * np.maximum(C[f"za_{t}_disc"].abs(), C[f"zb_{t}_disc"].abs())
        ok = (zd.abs() > thr) & (np.sign(zd) == np.sign(zc)) & (zc.abs() > 2.58) & over
        summ[f"n_pass_{t}_disc_bonf"] = int((zd.abs() > thr).sum())
        summ[f"n_pass_{t}_disc_conf"] = int(((zd.abs() > thr) & (np.sign(zd) == np.sign(zc)) & (zc.abs() > 2.58)).sum())
        summ[f"n_pass_{t}_disc_conf_over_singles"] = int(ok.sum())
        sel.append(C[ok].assign(target=t, zd=zd[ok], zc=zc[ok]))
    S_sel = pd.concat(sel) if sel else pd.DataFrame()
    S_sel.to_parquet(OUT / "shortlist.parquet", index=False)
    if len(S_sel):
        summ["shortlist_by_op"] = S_sel.groupby(["target", "op"]).size().rename("n").reset_index().to_dict("records")
        cols = pd.concat([S_sel["a"], S_sel["b"]]).value_counts().head(25)
        summ["shortlist_top_columns"] = cols.to_dict()
    top = {}
    for t in tests:
        zs = (C[f"z_{t}_disc"] / sdv[f"{t}_disc"]).abs().sort_values(ascending=False).index[:30]
        top[t] = [
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in C.loc[i, ["a", "b", "op", f"z_{t}_disc", f"z_{t}_conf", f"za_{t}_disc", f"zb_{t}_disc"]].to_dict().items()}
            for i in zs
        ]
    summ["top30_by_disc"] = top
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    _log("summary: " + json.dumps({k: v for k, v in summ.items() if k != "top30_by_disc"}, ensure_ascii=False))
    for t in tests:
        _log(f"[{t}] 発見期間上位: " + "; ".join(f"{r['a']}|{r['op']}|{r['b']} d={r['z_%s_disc' % t]} c={r['z_%s_conf' % t]} (a={r['za_%s_disc' % t]},b={r['zb_%s_disc' % t]})" for r in top[t][:6]))


def cmd_select(a) -> None:
    """発見期間で上位・確認期間で同符号・単独列より強い候補を、同じ列は最大 3 回までで n_win/n_top2/n_top3 個ずつ選ぶ。"""
    C = pd.read_parquet(OUT / "candidates.parquet")
    summ = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    chosen, used = [], {}
    seen = set()
    for t, n in (("win", a.n_win), ("top3", a.n_top3), ("top2", a.n_top2)):
        sd_d, sd_c = summ["null_sd_mad"][f"{t}_disc"], summ["null_sd_mad"][f"{t}_conf"]
        zd, zc = C[f"z_{t}_disc"], C[f"z_{t}_conf"]
        ok = (np.sign(zd) == np.sign(zc)) & (zc.abs() / sd_c > 1.0) & (zd.abs() > 1.25 * np.maximum(C[f"za_{t}_disc"].abs(), C[f"zb_{t}_disc"].abs()))
        order = (zd.abs() / sd_d)[ok].sort_values(ascending=False).index
        k = 0
        for i in order:
            r = C.loc[i]
            if (r["op"], r["a"], r["b"]) in seen or used.get(r["a"], 0) >= 3 or used.get(r["b"], 0) >= 3:
                continue
            chosen.append({"op": r["op"], "a": r["a"], "b": r["b"], "target": t, "z_disc": float(zd[i]), "z_conf": float(zc[i])})
            seen.add((r["op"], r["a"], r["b"]))
            used[r["a"]] = used.get(r["a"], 0) + 1
            used[r["b"]] = used.get(r["b"], 0) + 1
            k += 1
            if k >= n:
                break
    (OUT / "pairs_top.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"選択 {len(chosen)} 列 -> pairs_top.json")
    for c in chosen:
        print(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--k", type=int, default=150)
    r.add_argument("--rel", type=int, default=60)
    sl = sub.add_parser("select")
    sl.add_argument("--n-win", type=int, default=20)
    sl.add_argument("--n-top3", type=int, default=10)
    sl.add_argument("--n-top2", type=int, default=10)
    a = ap.parse_args()
    {"run": cmd_run, "select": cmd_select}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
