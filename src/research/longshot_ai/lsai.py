"""穴馬 AI（担当 longshot-ai、2026-11-27）: 単勝 10 倍以上の馬だけで、1 馬身の拡張ラベル（y_p1）を市場と独立の列で学ぶ。

- 計画: docs/operations/longshot_ai/plan.md。事前登録: config/longshot_ai/prereg.json（la-4）。会議の決定: roi_plus_council/decisions.md。
- 列: めぐ AI の v5（megu_cycles の _rich_matrix）から追い切り評価の 3 列と course_type を外した MF 版（market_free_columns.md）。
  v6 は FIT_END の族と s_* を外した漏れの確かめ、v7 は s_*（leave-one-year-out の上流）だけを外した版（相関で外す版の基準）。
- ラベル: model-relabel の出力（labels_convert_A.parquet の y_p1・y3）を読むだけ。sha256 が事前登録と違えば止める。
- 腕: L1（確定 10 倍以上の行・y_p1）と C0（全馬・y3。主の比べ先）。木は megu_cycles の BAG_SETS × XGB_PARAMS × 800 本。
- 分け: 年 t は t より前の年で学ぶ。いまの段で予測するのは 2018〜2021 だけ（2022〜24 は確かめの 1 回の後）。
- 2025 年以降の行は読み込みの直後に落とす。費用 0 円（手元のファイルだけ。GCS を読まない）。
  重い学習: flock /tmp/claude-1000/heavy.lock nice -n 19 env KEIBA_SESSION=longshot-ai python -m src.research.longshot_ai.lsai train --arm L1 --fs v5
  着順なしの表: python -m src.research.longshot_ai.lsai census
- 本番の推論の経路（src/pipeline/・src/api/）からは import しない。
- 注意: megu_cycles の _rich_matrix は meta（cycles16_fix3/meta.parquet）を全列で読むので、2024 年までの finish_position が
  いったんメモリに載る。load_matrix で直後に捨て、使わない（他の担当の道具なので読む列は変えていない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PREREG = ROOT / "config/longshot_ai/prereg.json"
OUT = Path(os.environ.get("LONGSHOT_AI_OUT") or "/home/jovyan/work/longshot_ai_out")
RUNNERS = Path(os.environ.get("MEGU_CYCLES_OUT") or "/home/jovyan/work/megu_cycles_out") / "runners.parquet"
V5BAG = RUNNERS.parent / "rich_pred_v5bag.parquet"
RICH_V5 = RUNNERS.parent / "rich_pred_v5.parquet"
KEY = ["race_id", "horse_id"]
YEAR_END = 2024
N_PRE = 5
THRESH = 10.0
DROP_DIR = ROOT / "config/longshot_ai"                                   # 外す列の一覧（drop_rho50.json・drop_rho30.json。凍結）
# 方針で外す列: 追い切り評価の 3 列（公開の専門家の意見）と course_type（推論の経路で 100% 空。点検役の指摘、11/27）
DROP_POLICY = ("oikiri_eval", "oikiri_eval_rank", "oikiri_eval_gap", "course_type")
DUP_RACES = ("202408060108",)      # 取り込みの誤りの写し（202408060101 と同じ馬が 7 頭）。runners から外す
PRED_YEARS = (2018, 2019, 2020, 2021)                                    # いまの段の予測の年（2022〜24 は確かめの後）
NTHREAD = int(os.environ.get("LONGSHOT_AI_NTHREAD") or 2)
ODDS_BANDS = (10.0, 15.0, 30.0, 50.0, np.inf)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def prereg() -> dict:
    return json.loads(PREREG.read_text(encoding="utf-8"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 純粋関数（テスト対象）
def mf_columns(names: list[str]) -> list[str]:
    """市場と独立の列（MF）: 方針で外す 4 列（追い切り評価の 3 列・course_type）を除く。名前に市場の語が残っていないことも確かめる。"""
    import re
    out = [c for c in names if c not in DROP_POLICY]
    bad = [c for c in out if re.search(r"odds|pop|ninki|tansho|fukusho|payoff|market|ret_", c, re.IGNORECASE)]
    assert not bad, f"市場の語を含む列が残っている: {bad[:5]}"
    return out


def population(odds: np.ndarray, thresh: float = THRESH) -> np.ndarray:
    """母集団（単勝 thresh 倍以上）。オッズの無い馬は外す。"""
    o = np.asarray(odds, float)
    return np.isfinite(o) & (o >= thresh)


def race_top1(race: np.ndarray, score: np.ndarray, mask: np.ndarray, tiebreak: np.ndarray | None = None) -> np.ndarray:
    """レースごとに、mask の中で score が最大の 1 頭の印（bool）。同じ値は tiebreak（小さい方。既定は行の順）。mask が空のレースは印なし。"""
    n = len(score)
    tb = np.arange(n, dtype=float) if tiebreak is None else np.asarray(tiebreak, float)
    d = pd.DataFrame({"r": race, "s": np.where(mask, score, -np.inf), "t": tb, "m": mask, "i": np.arange(n)})
    d = d[d["m"]].sort_values(["r", "s", "t"], ascending=[True, False, True], kind="mergesort")
    pick = d.drop_duplicates("r")["i"].to_numpy()
    out = np.zeros(n, bool)
    out[pick] = True
    return out


def q_from_odds(race: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """1/オッズをレースの中で和 1 に。オッズの無い馬は 0。"""
    o = np.asarray(odds, float)
    inv = np.where(np.isfinite(o) & (o > 0), 1.0 / np.where(o > 0, o, 1.0), 0.0)
    s = pd.Series(inv).groupby(race).transform("sum").to_numpy()
    return np.where(s > 0, inv / np.where(s > 0, s, 1.0), 0.0)


def odds_band(odds: np.ndarray) -> np.ndarray:
    """0: 10〜15・1: 15〜30・2: 30〜50・3: 50 以上（10 未満は −1）。"""
    o = np.asarray(odds, float)
    b = np.searchsorted(np.asarray(ODDS_BANDS), o, side="right") - 1
    return np.where(o >= ODDS_BANDS[0], np.minimum(b, len(ODDS_BANDS) - 2), -1)


def cluster_of(race_id: pd.Series, date: pd.Series) -> np.ndarray:
    """開催日 × 場（race_id の 5〜6 桁目が場）。"""
    return date.astype(str).to_numpy() + "_" + race_id.astype(str).str[4:6].to_numpy()


def power_sim(p_true: np.ndarray, star: np.ndarray, ref: np.ndarray, cluster_r: np.ndarray, target: float = 0.01,
              n_sim: int = 1000, seed: int = 20261127) -> dict:
    """検出力の模擬（la-2 の決定 7、着順を使わない）。

    - p_true: 馬ごとの真の 3 着内の確率（較正した市場の q3 の 5 通りの平均）。
    - star・ref: (K, R) の馬の番号（行の位置）。再現 k・レース r の ★ と比べ先の 1 位。無いときは −1。
    - 判定の物差しと同じく、レースの差 d_r = 再現の平均の (y[★] − y[比べ先])。
    - 上乗せ: 「どれかの再現で ★ で、その再現の比べ先と違う馬」に同じ L を足し、d の期待値の増分がちょうど target
      （+1.0pt）になる L を解く（同じ馬のレースは差 0 なので、全馬に足すと真の差が target より小さくなる）。
    - 馬ごとの y は独立に引く（同じレースの 3 枠の取り合いの負の相関は入れない。差の分散を少し大きく見る向き）。
    - 区間は束ねた（開催日 × 場）正規の区間に小標本の補正 G/(G−1) を掛けたもの。判定のブートストラップ 2,000 回の近似。
    - レースごとに再現の平均をとってからレースで平均する（判定の「再現ごとの物差しの 5 通りの平均」とは、母集団 0 頭の
      レースが再現で違うときだけ重みが少しずれる近似）。
    """
    rng = np.random.default_rng(seed)
    ok = (star >= 0) & (ref >= 0)
    w = ok.sum(0).astype(float)                                   # レースごとの使える再現の数
    use = w > 0
    lift_set = np.zeros(len(p_true))
    sel = ok & (star != ref)
    lift_set[star[sel]] = 1.0
    st = np.where(ok, star, 0)
    rf = np.where(ok, ref, 0)
    inc = (np.where(ok, lift_set[st] - lift_set[rf], 0.0).sum(0)[use] / w[use]).mean()
    if inc <= 0:
        return {"power": 0.0, "L": None, "agree": float((star == ref)[ok].mean()), "n_races": int(use.sum()), "n_sim": n_sim}
    L = target / inc
    p = np.clip(p_true + L * lift_set, 0.0, 1.0)
    codes, cl = np.unique(cluster_r[use], return_inverse=True)
    G = len(codes)
    n_c = np.bincount(cl, minlength=G).astype(float)
    n = int(use.sum())
    hit = 0
    for _ in range(n_sim):
        y = (rng.random(len(p)) < p).astype(float)
        d = (np.where(ok, y[st] - y[rf], 0.0).sum(0)[use] / w[use])
        m = d.mean()
        s_c = np.bincount(cl, weights=d, minlength=G)
        se = np.sqrt(((s_c - n_c * m) ** 2).sum() * G / max(G - 1, 1)) / n
        hit += (m - 1.96 * se) > 0
    return {"power": hit / n_sim, "L": float(L), "agree": float((star == ref)[ok].mean()), "n_races": n, "n_clusters": int(G),
            "n_lifted": int(lift_set.sum()), "n_sim": n_sim, "target": target}


# ---------------------------------------------------------------- 読み込み
def load_labels(max_year: int = YEAR_END) -> pd.DataFrame:
    """ラベル（y3・y_p1）を max_year までの行だけ読む（台帳に記す期間と読む期間をそろえる）。"""
    pr = prereg()["labels"]
    p = Path(pr["source"])
    got = sha256(p)
    assert got == pr["source_sha256"], f"ラベルのファイルが事前登録と違う: {got}"
    lab = pd.read_parquet(p, columns=KEY + ["y3", "y_p1", "year", "grade", "race_class"], filters=[("year", "<=", max_year)])
    assert lab["year"].max() <= max_year
    lab["race_id"], lab["horse_id"] = lab["race_id"].astype(str), lab["horse_id"].astype(str)
    return lab.drop_duplicates(KEY)


def load_runners() -> pd.DataFrame:
    cols = KEY + ["date", "year", "venue", "field", "hn", "odds_final", "pop_final", "cls"] + [f"odds_pre{k}" for k in range(N_PRE)]
    r = pd.read_parquet(RUNNERS, columns=cols, filters=[("year", "<=", YEAR_END)])
    assert r["year"].max() <= YEAR_END
    r["race_id"], r["horse_id"] = r["race_id"].astype(str), r["horse_id"].astype(str)
    r = r[~r["race_id"].isin(DUP_RACES)]
    return r.drop_duplicates(KEY)


def load_matrix(fs: str):
    """X（MF の列・float32）と meta（race_id・horse_id・year）を year<=2024 の行だけ。"""
    from src.research.megu_cycles import mc_data as MC
    X, meta, names = MC._rich_matrix(fs)
    cols = mf_columns(list(names))
    X = X[cols]
    meta = meta[["race_id", "horse_id", "year"]].copy().reset_index(drop=True)      # 着順（finish_position）はここで捨てる
    meta["race_id"], meta["horse_id"] = meta["race_id"].astype(str), meta["horse_id"].astype(str)
    meta["year"] = pd.to_numeric(meta["year"]).astype(int)
    assert meta["year"].max() <= YEAR_END
    return X, meta, cols


def join_frame(meta: pd.DataFrame, label_max_year: int | None) -> tuple[pd.DataFrame, dict]:
    """meta の行の順に、runners（オッズ）と、label_max_year までのラベルを結ぶ（None ならラベルを読まない）。
    結べた数を返す。ラベルの年の行で、母集団・全馬のどちらでもラベルの欠けが 0.5% を超えたら止める。"""
    r = load_runners()
    j = meta.merge(r.drop(columns=["year"]), on=KEY, how="left")
    assert len(j) == len(meta)
    stat = {"rows_meta": len(meta), "rows_with_odds": int(j["odds_final"].notna().sum())}
    if label_max_year is not None:
        lab = load_labels(label_max_year)
        j = j.merge(lab.drop(columns=["year"]), on=KEY, how="left")
        assert len(j) == len(meta)
        in_years = (meta["year"] <= label_max_year).to_numpy()
        j["has_label"] = j["y3"].notna().to_numpy() & in_years
        pop = population(j["odds_final"].to_numpy(float)) & in_years
        miss_pop = pop & ~j["has_label"].to_numpy()
        miss_all = in_years & ~j["has_label"].to_numpy()
        stat.update(label_max_year=label_max_year, rows_pop_final=int(pop.sum()), pop_missing_label=int(miss_pop.sum()),
                    all_missing_label=int(miss_all.sum()), rows_label_years=int(in_years.sum()))
        assert miss_pop.sum() <= 0.005 * max(pop.sum(), 1), f"母集団でラベルの無い行が多い: {stat}"
        assert miss_all.sum() <= 0.005 * max(in_years.sum(), 1), f"ラベルの無い行が多い: {stat}"
        for c in ("y3", "y_p1"):
            j[c] = j[c].fillna(False).astype(bool)
    return j, stat


# ---------------------------------------------------------------- 学習
def arm_rows(j: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """腕の訓練の行の印と目標（ラベルのある行だけ）。L1: 確定 10 倍以上・y_p1。C0: オッズのある全馬（オッズの帯を問わない）・y3。
    X の meta にあって runners に無い行（毎年約 1,500 頭）は、全部が障害レース（runners は平地だけ）。L1 と C0 の差を
    「母集団とラベル」だけにするため、C0 も平地（オッズのある行）で学ぶ（dataset_check.md）。"""
    has = j["has_label"].to_numpy() & j["odds_final"].notna().to_numpy()
    if arm == "L1":
        return has & population(j["odds_final"].to_numpy(float)), j["y_p1"].to_numpy(np.float32)
    if arm == "C0":
        return has, j["y3"].to_numpy(np.float32)
    raise ValueError(arm)


def cmd_train(a) -> None:
    import xgboost as xgb

    from src.ops import data_ledger as DL
    from src.research.megu_cycles import mc_data as MC
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = OUT / "parts"
    parts.mkdir(exist_ok=True)
    years = [int(y) for y in a.years.split(",")]
    assert max(years) <= 2021, "2022〜24 の予測は確かめの 1 回まで作らない"
    pr = prereg()
    label_max = max(years) - 1
    with DL.guard(tool="src/research/longshot_ai/lsai.py:train", family="longshot_ai", version=pr["version"],
                  period=("2016-01-01", f"{label_max}-12-31"), purpose="reproduce", reads=("着順",),
                  frozen_paths=[PREREG, Path(__file__)] + ([DROP_DIR / f"drop_{a.drop}.json"] if a.drop else []),
                  reason="学習の年の着順を教師に使うだけ。予測の年の成績は数えない") as tok:
        X, meta, cols = load_matrix(a.fs)
        drop_sha = ""
        if a.drop:
            drop_path = DROP_DIR / f"drop_{a.drop}.json"
            drop = set(json.loads(drop_path.read_text(encoding="utf-8"))["columns"])
            drop_sha = sha256(drop_path)[:8]
            cols = [c for c in cols if c not in drop]
            X = X[cols]
        j, stat = join_frame(meta, label_max_year=label_max)
        rows, y = arm_rows(j, a.arm)
        yr = meta["year"].to_numpy()
        log(f"{a.arm} {a.fs}: 列 {len(cols)}・結び {stat}")
        out = meta[KEY + ["year"]].copy()
        P = np.full(len(meta), np.nan)
        sets = [dict(MC.XGB_PARAMS, **bs, nthread=NTHREAD) for bs in MC.BAG_SETS]
        # 部品の名前に、ラベルの sha・列の数・外す列の一覧の sha を入れる（替えた後に古い部品を黙って使わない）。
        # 事前登録の版は入れない（版を上げても腕の中身が同じなら部品を使い回す）
        tag = f"{pr['labels']['source_sha256'][:8]}_{len(cols)}{'_' + a.drop + drop_sha if a.drop else ''}"
        for t in years:
            te = yr == t
            acc = np.zeros(int(te.sum()))
            for i, ps in enumerate(sets):
                part = parts / f"{a.arm}_{a.fs}_{t}_b{i}_{tag}.npy"
                if part.exists():
                    acc += np.load(part)
                    continue
                tr = rows & (yr < t)
                m = xgb.train(ps, xgb.DMatrix(X[tr], label=y[tr]), MC.XGB_ROUNDS)
                p = m.predict(xgb.DMatrix(X[te]))
                np.save(part, p)
                acc += p
                log(f"  {a.arm} {a.fs} {t} bag{i}: 学習 {int(tr.sum())} → 予測 {int(te.sum())}（{time.time() - t0:.0f} 秒）")
            P[te] = acc / len(sets)
        out["p"] = P
        name = f"{a.arm}_{a.fs}{'_' + a.drop if a.drop else ''}"
        dest = OUT / f"pred_{name}.parquet"
        out[np.isin(yr, years)].to_parquet(dest, index=False)
        (OUT / f"pred_{name}.json").write_text(json.dumps(
            {"arm": a.arm, "fs": a.fs, "drop": a.drop, "drop_sha8": drop_sha, "years": years, "n_cols": len(cols), "join": stat, "prereg": pr["version"], "tag": tag,
             "label_sha256": pr["labels"]["source_sha256"], "params": sets, "rounds": MC.XGB_ROUNDS, "sec": round(time.time() - t0)},
            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tok.finish(result_path=dest)
    log(f"保存 {dest}（{time.time() - t0:.0f} 秒）")


# ---------------------------------------------------------------- 市場の logit との順位相関（la-3。着順は読まない）
CORR = OUT / "market_corr_v5mf.parquet"


def spearman_cols(X: pd.DataFrame, target: np.ndarray, rows: np.ndarray) -> pd.DataFrame:
    """rows の行で、各列と target の Spearman の順位相関（列ごとに欠けを落とす）。"""
    out = []
    t_all = target[rows]
    for c in X.columns:
        v = X[c].to_numpy(float)[rows]
        ok = np.isfinite(v) & np.isfinite(t_all)
        if ok.sum() < 100 or np.nanstd(v[ok]) == 0:
            out.append((c, np.nan, int(ok.sum())))
            continue
        a = pd.Series(v[ok]).rank().to_numpy()
        b = pd.Series(t_all[ok]).rank().to_numpy()
        out.append((c, float(np.corrcoef(a, b)[0, 1]), int(ok.sum())))
    return pd.DataFrame(out, columns=["column", "rho", "n"])


def cmd_corr(a) -> None:
    """v5 の MF の列と logit(q_final) の順位相関（2016〜2021、全馬と確定 10 倍以上の中）。全馬の値は model-relabel の表と突き合わせる。
    外す列の一覧を config/longshot_ai/drop_rho{50,30}.json に書く（凍結し、train --drop はそれだけを読む）。"""
    t0 = time.time()
    X, meta, cols = load_matrix("v5")
    j, _ = join_frame(meta, label_max_year=None)
    yr = meta["year"].to_numpy()
    race = meta["race_id"].to_numpy()
    q = q_from_odds(race, j["odds_final"].to_numpy(float))
    lq = np.where(q > 0, np.log(np.clip(q, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(q, 1e-6, 1 - 1e-6)), np.nan)
    base = (yr <= 2021) & j["odds_final"].notna().to_numpy()
    pop = base & population(j["odds_final"].to_numpy(float))
    ra = spearman_cols(X, lq, base).rename(columns={"rho": "rho_all", "n": "n_all"})
    log(f"全馬 {int(base.sum())} 行（{time.time() - t0:.0f} 秒）")
    rp = spearman_cols(X, lq, pop).rename(columns={"rho": "rho_pop", "n": "n_pop"})
    t = ra.merge(rp, on="column")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_parquet(CORR, index=False)
    summ = {"rows_all": int(base.sum()), "rows_pop": int(pop.sum()), "n_cols": len(cols), "years": [2016, 2021],
            "counts": {str(th): {"all": int((t["rho_all"].abs() >= th).sum()), "pop": int((t["rho_pop"].abs() >= th).sum()),
                                 "union": len(corr_drop_list(th, t))} for th in (0.5, 0.3)},
            "top20_all": t.reindex(t["rho_all"].abs().sort_values(ascending=False).index).head(20)[["column", "rho_all", "rho_pop"]].round(3).values.tolist(),
            "top20_pop": t.reindex(t["rho_pop"].abs().sort_values(ascending=False).index).head(20)[["column", "rho_all", "rho_pop"]].round(3).values.tolist()}
    for th in (0.5, 0.3):
        lst = corr_drop_list(th, t)
        summ[f"drop_{int(th * 100)}"] = lst
        (DROP_DIR / f"drop_rho{int(th * 100)}.json").write_text(json.dumps(
            {"threshold": th, "rule": "全馬・確定 10 倍以上の中のどちらかで |ρ| ≥ 閾値（2016〜21、logit(q_final) との Spearman）",
             "n": len(lst), "columns": lst}, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "market_corr_v5mf.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if not k.startswith("drop_")}, ensure_ascii=False, indent=1))


def corr_drop_list(th: float, t: pd.DataFrame | None = None) -> list[str]:
    """全馬・母集団のどちらかで |ρ| ≥ th の列（和集合）。"""
    t = pd.read_parquet(CORR) if t is None else t
    m = (t["rho_all"].abs() >= th) | (t["rho_pop"].abs() >= th)
    return sorted(t.loc[m, "column"].tolist())


# ---------------------------------------------------------------- 着順なしの表（la-2 の指示 5）と検出力（指示 6）
def eval_frame(fs: str = "v5", years=(2019, 2020, 2021), star: str = "") -> pd.DataFrame:
    """予測の年の行に、オッズ・腕の予測・めぐ AI の予測を結ぶ（着順は読まない）。star は ★ の腕の名前
    （既定は L1_{fs}。drop の版は L1_v5_rho50 など）。C0 はいつも C0_{fs}。"""
    r = load_runners()
    r = r[r["year"].isin(years)].copy()
    for arm, name in (("L1", star or f"L1_{fs}"), ("C0", f"C0_{fs}")):
        p = pd.read_parquet(OUT / f"pred_{name}.parquet")
        p["race_id"], p["horse_id"] = p["race_id"].astype(str), p["horse_id"].astype(str)
        r = r.merge(p[KEY + ["p"]].rename(columns={"p": f"p_{arm}"}), on=KEY, how="left")
    for path, col, name in ((V5BAG, "pr_win", "megu_win"), (RICH_V5, "pr_top3", "megu_top3")):
        m = pd.read_parquet(path, columns=KEY + [col])
        m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
        r = r.merge(m.rename(columns={col: name}).drop_duplicates(KEY), on=KEY, how="left")
    return r.sort_values(["race_id", "hn"]).reset_index(drop=True)


def picks(r: pd.DataFrame, k: int) -> dict[str, np.ndarray]:
    """再現 k の母集団の中の 1 位（★・C0・市場・めぐ）。同じ値は馬番の小さい方。
    母集団は「再現で 10 倍以上」かつ ★ と C0 の予測がある馬（比べ先ごとに選べる馬の集合をそろえる。外した数は census に出す）。"""
    race = r["race_id"].to_numpy()
    pop = population(r[f"odds_pre{k}"].to_numpy(float)) & r["p_L1"].notna().to_numpy() & r["p_C0"].notna().to_numpy()
    hn = r["hn"].to_numpy(float)
    o = r[f"odds_pre{k}"].to_numpy(float)
    out = {"pop": pop}
    for name, s in (("star", r["p_L1"]), ("c0", r["p_C0"]), ("megu_win", r["megu_win"]), ("megu_top3", r["megu_top3"])):
        v = s.to_numpy(float)
        out[name] = race_top1(race, np.where(np.isfinite(v), v, -np.inf), pop, hn)
    out["market"] = race_top1(race, -o, pop, hn)
    return out


def cmd_census(a) -> None:
    """着順を読まずに: 一致率・オッズの帯・5 通りでの入れ替わり・母集団の形・skew の雑音での入れ替わり・母集団 0 頭のレース。"""
    r = eval_frame(a.fs, star=a.star)
    race = r["race_id"].to_numpy()
    nr = r["race_id"].nunique()
    res = {"fs": a.fs, "star": a.star or f"L1_{a.fs}", "years": [2019, 2020, 2021], "races": int(nr), "rows": len(r),
           "missing_pred": {c: int(r[c].isna().sum()) for c in ("p_L1", "p_C0", "megu_win", "megu_top3")},
           "pop_dropped_no_pred": [int((population(r[f"odds_pre{k}"].to_numpy(float)) & (r["p_L1"].isna() | r["p_C0"].isna()).to_numpy()).sum())
                                   for k in range(N_PRE)]}
    P = [picks(r, k) for k in range(N_PRE)]
    names = ("star", "c0", "market", "megu_win", "megu_top3")
    # 一致率（★ と各 1 位が同じ馬のレースの割合、5 通りの平均）
    res["agree_with_star"] = {n: round(float(np.mean([(P[k]["star"] & P[k][n]).sum() / max(P[k]["star"].sum(), 1) for k in range(N_PRE)])), 4)
                              for n in names[1:]}
    # オッズの帯（再現 k のオッズで、選んだ馬の帯の割合。5 通りの平均）と中央値
    band = {}
    for n in names:
        rows = []
        for k in range(N_PRE):
            o = r[f"odds_pre{k}"].to_numpy(float)[P[k][n]]
            rows.append(np.bincount(odds_band(o)[odds_band(o) >= 0], minlength=4) / max(len(o), 1))
        med = float(np.median(np.concatenate([r[f"odds_pre{k}"].to_numpy(float)[P[k][n]] for k in range(N_PRE)])))
        band[n] = {"10-15": round(float(np.mean([x[0] for x in rows])), 4), "15-30": round(float(np.mean([x[1] for x in rows])), 4),
                   "30-50": round(float(np.mean([x[2] for x in rows])), 4), "50+": round(float(np.mean([x[3] for x in rows])), 4),
                   "median_odds_pre": round(med, 2)}
    res["odds_band"] = band
    # 5 通りでの入れ替わり: 再現 0 の ★ と同じ馬が再現 k でも ★ のレースの割合
    res["star_same_as_pre0"] = [round(float((P[0]["star"] & P[k]["star"]).sum() / max(P[0]["star"].sum(), 1)), 4) for k in range(1, N_PRE)]
    # 確定で選んだ ★ との一致（母集団を確定で決めた場合）
    popf = population(r["odds_final"].to_numpy(float))
    star_f = race_top1(race, r["p_L1"].fillna(-np.inf).to_numpy(float), popf, r["hn"].to_numpy(float))
    res["star_same_as_final"] = round(float(np.mean([(star_f & P[k]["star"]).sum() / max(star_f.sum(), 1) for k in range(N_PRE)])), 4)
    # ★ が「再現で 10 倍以上・確定で一桁」に入る割合（母集団全体の割合と並べる）
    res["star_in_pre_ge10_final_lt10"] = round(float(np.mean([(P[k]["star"] & (r["odds_final"].to_numpy(float) < THRESH)).sum()
                                                               / max(P[k]["star"].sum(), 1) for k in range(N_PRE)])), 4)
    res["pop_in_pre_ge10_final_lt10"] = round(float(np.mean([(P[k]["pop"] & (r["odds_final"].to_numpy(float) < THRESH)).sum()
                                                              / max(P[k]["pop"].sum(), 1) for k in range(N_PRE)])), 4)
    # 母集団 0 頭のレース・1 レースの頭数
    res["races_pop0"] = [int(nr - pd.Series(race[P[k]["pop"]]).nunique()) for k in range(N_PRE)]
    # 年ごとの母集団の形（再現 0）
    r["_pop0"] = P[0]["pop"]
    g = r.groupby("year")
    res["year_shape"] = {int(y): {"races": int(d["race_id"].nunique()), "pop_share": round(float(d["_pop0"].mean()), 4),
                                  "pop_per_race": round(float(d.groupby("race_id")["_pop0"].sum().mean()), 2),
                                  "field_mean": round(float(d.groupby("race_id")["field"].first().mean()), 2)} for y, d in g}
    # train-serve-skew の雑音（logit に N(0, s)）で ★ が変わるレースの割合（再現 0）
    rng = np.random.default_rng(20261127)
    pv = np.clip(r["p_L1"].to_numpy(float), 1e-6, 1 - 1e-6)
    lp = np.log(pv) - np.log1p(-pv)                                   # 予測の無い馬は NaN のまま（母集団から外してある）
    flips = {}
    for s in (0.10, 0.23):
        v = []
        for _ in range(20):
            alt = race_top1(race, lp + rng.normal(0, s, len(lp)), P[0]["pop"], r["hn"].to_numpy(float))
            v.append(1 - (alt & P[0]["star"]).sum() / max(P[0]["star"].sum(), 1))
        flips[str(s)] = round(float(np.mean(v)), 4)
    res["skew_flip_rate"] = flips
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"census_{a.star or 'L1_' + a.fs}.json"
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def raw_q3(race: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """オッズからの生の Harville の 3 着内の確率（レース内で和 3、上限 1）。"""
    from src.pipeline.models import zoo
    from src.scripts.data import longshot_model as LM
    q = q_from_odds(race, odds)
    q = np.where(q > 0, q, 1e-5)
    q = q / pd.Series(q).groupby(race).transform("sum").to_numpy()
    _, q3 = zoo.harville_topk(race, q, 1.0, 1.0)
    return LM.norm_race_cap(race, q3, 3.0)


def calibrated_market_q3(eval_years=(2019, 2020, 2021)) -> pd.DataFrame:
    """較正した市場の 3 着内の確率（再現 k ごと）。前の試み（longshot_model）の式: logit(q)・2 乗・logit(Harville q3)・
    頭数の帯・クラスの群のダミーと積のロジスティック回帰。年 t は t より前の年（2016〜）の全馬で当てる（la-2 の決定 5 (iii)）。
    当てる側の着順は 2016〜(max(eval_years)−1) 年だけ。結果は OUT/market_cq3.parquet（A/E の分母にも使う）。"""
    from src.ops import data_ledger as DL
    from src.scripts.data import longshot_model as LM
    dest = OUT / "market_cq3.parquet"
    pr = prereg()
    stamp = {"eval_years": list(eval_years), "label_sha256": pr["labels"]["source_sha256"], "code_sha256": sha256(Path(__file__))[:16]}
    side = dest.with_suffix(".json")
    if dest.exists() and side.exists():
        old = json.loads(side.read_text(encoding="utf-8"))
        keys = ("eval_years", "label_sha256")
        if {k: old.get(k) for k in keys} == {k: stamp[k] for k in keys}:
            return pd.read_parquet(dest)
        raise SystemExit(f"market_cq3 の作りが違う（{old} ≠ {stamp}）。消してから作り直す")
    last = max(eval_years) - 1
    with DL.guard(tool="src/research/longshot_ai/lsai.py:calibrated_market_q3", family="longshot_ai", version=pr["version"],
                  period=("2016-01-01", f"{last}-12-31"), purpose="fit", reads=("着順",), frozen_paths=[PREREG, Path(__file__)],
                  reason="市場の 3 着内の確率の較正（favourite-longshot の直し）を、評価の年より前の年の着順で当てるだけ。モデルの成績は数えない") as tok:
        r = load_runners()
        r = r[r["year"] <= max(eval_years)].sort_values(["race_id", "hn"]).reset_index(drop=True)
        lab = load_labels(last)
        cls = pd.read_parquet(Path(pr["labels"]["source"]), columns=KEY + ["grade", "race_class", "year"],
                              filters=[("year", "<=", max(eval_years))]).drop_duplicates(KEY)       # 着順の列は読まない
        cls["race_id"], cls["horse_id"] = cls["race_id"].astype(str), cls["horse_id"].astype(str)
        r = r.merge(cls.drop(columns=["year"]), on=KEY, how="left").merge(lab[KEY + ["y3"]], on=KEY, how="left")
        race = r["race_id"].to_numpy()
        fband = LM.field_band(r["field"].to_numpy(float))
        cgrp = LM.class_group(r["grade"].fillna(""), r["race_class"].fillna(""))
        yr = r["year"].to_numpy()
        out = r[KEY + ["year"]].copy()
        info = {}
        for k in range(N_PRE):
            q = q_from_odds(race, r[f"odds_pre{k}"].to_numpy(float))
            q = np.clip(np.where(q > 0, q, 1e-5), 1e-5, 1.0)
            q3 = raw_q3(race, r[f"odds_pre{k}"].to_numpy(float))
            des = LM.calib_design(LM.mm.logit(q), LM.mm.logit(q3), fband, cgrp)
            cq = np.full(len(r), np.nan)
            for t in eval_years:
                tr = (yr < t) & r["y3"].notna().to_numpy()
                m = LM.calibrate_fit(des[tr], r["y3"].to_numpy()[tr].astype(int))
                te = yr == t
                cq[te] = LM.sigmoid(LM.calibrate_logit(m, des[te]))
            ok = np.isin(yr, eval_years)
            cq[ok] = LM.norm_race_cap(race[ok], cq[ok], 3.0)
            out[f"cq3_{k}"] = cq
            out[f"rq3_{k}"] = q3
            info[k] = {"rows_fit_max": int(((yr < max(eval_years)) & r["y3"].notna().to_numpy()).sum())}
        out = out[np.isin(yr, eval_years)].reset_index(drop=True)
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dest, index=False)
        side.write_text(json.dumps(stamp, ensure_ascii=False), encoding="utf-8")
        tok.finish(result_path=dest)
    return out


def cmd_power(a) -> None:
    """検出力の模擬（la-2 の決定 7）。真の 3 着内の確率は較正した市場の q3 の 5 通りの平均。評価の年の着順は読まない。"""
    r = eval_frame(a.fs, star=a.star)
    cq = calibrated_market_q3()
    r = r.merge(cq.drop(columns=["year"]), on=KEY, how="left")
    race_codes, race_idx = np.unique(r["race_id"].to_numpy(), return_inverse=True)
    R = len(race_codes)
    p_true = np.nanmean(np.column_stack([r[f"cq3_{k}"].to_numpy(float) for k in range(N_PRE)]), axis=1)
    p_true = np.where(np.isfinite(p_true), p_true, 0.0)
    star = np.full((N_PRE, R), -1)
    ref = np.full((N_PRE, R), -1)
    for k in range(N_PRE):
        P = picks(r, k)
        star[k, race_idx[P["star"]]] = np.flatnonzero(P["star"])
        ref[k, race_idx[P["c0"]]] = np.flatnonzero(P["c0"])
    clus_r = pd.Series(cluster_of(r["race_id"], r["date"])).groupby(race_idx).first().reindex(range(R)).to_numpy()
    res = power_sim(p_true, star, ref, clus_r, target=a.lift, n_sim=a.n_sim)
    res.update(fs=a.fs, star=a.star or f"L1_{a.fs}", true_p="較正した市場の q3（再現 5 通りの平均）", ci="束ねた正規の区間 × G/(G−1)（判定のブートストラップの近似）",
               independence="馬ごとの y は独立に引く（3 枠の取り合いの負の相関は入れない）",
               mean_p_star=round(float(np.mean([p_true[star[k][star[k] >= 0]].mean() for k in range(N_PRE)])), 4),
               mean_p_c0=round(float(np.mean([p_true[ref[k][ref[k] >= 0]].mean() for k in range(N_PRE)])), 4))
    res["gate"] = "通る" if res["power"] >= 0.5 else "閉じる（0.5 未満）"
    dest = OUT / f"power_{a.star or 'L1_' + a.fs}.json"
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=["L1", "C0"], required=True)
    t.add_argument("--fs", choices=["v5", "v6", "v7"], default="v5")
    t.add_argument("--years", default=",".join(str(y) for y in PRED_YEARS))
    t.add_argument("--drop", choices=["rho50", "rho30"], default=None, help="市場の logit と相関の高い列を外す（la-3）")
    sub.add_parser("corr")
    c = sub.add_parser("census")
    c.add_argument("--fs", choices=["v5", "v6", "v7"], default="v5")
    c.add_argument("--star", default="", help="★ の腕（既定 L1_{fs}。例: L1_v5_rho50）")
    p = sub.add_parser("power")
    p.add_argument("--fs", choices=["v5", "v6", "v7"], default="v5")
    p.add_argument("--star", default="")
    p.add_argument("--lift", type=float, default=0.01)
    p.add_argument("--n-sim", type=int, default=1000)
    a = ap.parse_args()
    {"train": cmd_train, "census": cmd_census, "power": cmd_power, "corr": cmd_corr}[a.cmd](a)


if __name__ == "__main__":
    main()
