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
    - 上乗せ: 「どれかの再現で ★ で、その再現の比べ先と違う馬」に同じ L を足し、d の期待値（上乗せの前の差 base を含む）が
      ちょうど target（+1.0pt）になる L を解く。市場を真とすると ★ はもともと比べ先より低いことがある（base < 0）ので、
      増分だけを target にすると真の差が target より小さくなる（11/27 に直した。直す前の 0.063 は真の差 約 +0.13pt での値）。
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
    # 上乗せの前の真の差（市場を真としたときの ★ − 比べ先）。真の差の全体が target になるように L を解く（la-2 の決定 7）
    base = (np.where(ok, p_true[st] - p_true[rf], 0.0).sum(0)[use] / w[use]).mean()
    if inc <= 0:
        return {"power": 0.0, "L": None, "base_diff": float(base), "agree": float((star == ref)[ok].mean()), "n_races": int(use.sum()), "n_sim": n_sim}
    L = (target - base) / inc
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
    return {"power": hit / n_sim, "L": float(L), "base_diff": float(base), "agree": float((star == ref)[ok].mean()), "n_races": n, "n_clusters": int(G),
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


def load_matrix(fs: str, raw: bool = False):
    """X（MF の列・float32）と meta（race_id・horse_id・year）を year<=2024 の行だけ。raw=True は M の版（方針で外す列も残す）。"""
    from src.research.megu_cycles import mc_data as MC
    X, meta, names = MC._rich_matrix(fs)
    cols = list(names) if raw else mf_columns(list(names))
    if raw:
        extra = set(cols) - set(mf_columns(list(names)))
        assert extra <= set(DROP_POLICY), f"M の版と MF の差が方針の列だけではない: {sorted(extra - set(DROP_POLICY))}"
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
def chalk_races(race: np.ndarray, y3: np.ndarray, pop_final: np.ndarray) -> np.ndarray:
    """行ごとに、そのレースの 3 着内の馬が全部 3 番人気以内（確定の人気）なら True（LC の版。訓練の行にだけ使う）。"""
    d = pd.DataFrame({"r": race, "y": y3.astype(bool), "pf": np.nan_to_num(np.asarray(pop_final, float), nan=99.0)})
    t = d[d["y"]].groupby("r")["pf"].max()
    ch = (t <= 3).reindex(pd.unique(race)).fillna(False)
    return d["r"].map(ch).fillna(False).to_numpy(bool)


def arm_rows(j: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """腕の訓練の行の印と目標（ラベルのある行だけ）。L1: 確定 10 倍以上・y_p1。C0: オッズのある全馬（オッズの帯を問わない）・y3。
    記述の腕: L0 は L1 のラベルを y3 に。LA は全馬・「y_p1 かつ確定 10 倍以上」（一桁の馬は全部 0）。
    LC は L1 から人気どおりのレース（3 着内が全部 3 番人気以内）を除く。L1T は母集団を再現 0（odds_pre0）の 10 倍以上で決める。
    X の meta にあって runners に無い行（毎年約 1,500 頭）は、全部が障害レース（runners は平地だけ）。L1 と C0 の差を
    「母集団とラベル」だけにするため、C0 も平地（オッズのある行）で学ぶ（dataset_check.md）。"""
    has = j["has_label"].to_numpy() & j["odds_final"].notna().to_numpy()
    if arm == "L1":
        return has & population(j["odds_final"].to_numpy(float)), j["y_p1"].to_numpy(np.float32)
    if arm == "C0":
        return has, j["y3"].to_numpy(np.float32)
    pf = population(j["odds_final"].to_numpy(float))
    if arm == "L0":
        return has & pf, j["y3"].to_numpy(np.float32)
    if arm == "LA":
        return has, (j["y_p1"].to_numpy() & pf).astype(np.float32)
    if arm == "LC":
        ch = chalk_races(j["race_id"].to_numpy(), j["y3"].to_numpy() & has, j["pop_final"].to_numpy(float))
        return has & pf & ~ch, j["y_p1"].to_numpy(np.float32)
    if arm == "L1T":
        return has & population(j["odds_pre0"].to_numpy(float)), j["y_p1"].to_numpy(np.float32)
    raise ValueError(arm)


def cmd_train(a) -> None:
    import xgboost as xgb

    from src.ops import data_ledger as DL
    from src.research.megu_cycles import mc_data as MC
    from src.scripts.data import longshot_model as LM
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
        X, meta, cols = load_matrix(a.fs, raw=a.raw)
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
        if a.model == "lgb":
            sets = [dict(MC.PARAMS, num_threads=NTHREAD)]
        else:
            sets = [dict(MC.XGB_PARAMS, **bs, nthread=NTHREAD) for bs in MC.BAG_SETS]
        margin = None
        if a.offset:
            # 市場の 3 着内の確率（再現 0 の生の Harville の q3）の logit を offset に（記述の腕。評価の再現 k でも同じ offset）
            # q3 は runners の全馬（中止の馬も含む。発走前に分かる頭数）で作ってから結ぶ。オッズの無い行（障害）は 0 で、
            # L1 の訓練にも評価にも入らない
            rr = load_runners()
            rr["q3m"] = LM.mm.logit(raw_q3(rr["race_id"].to_numpy(), rr["odds_pre0"].to_numpy(float)))
            mg = j[KEY].merge(rr[KEY + ["q3m"]], on=KEY, how="left")["q3m"].to_numpy(float)
            margin = np.nan_to_num(mg, nan=0.0).astype(np.float32)
        # 部品の名前に、ラベルの sha・列の数・外す列の一覧の sha を入れる（替えた後に古い部品を黙って使わない）。
        # 事前登録の版は入れない（版を上げても腕の中身が同じなら部品を使い回す）
        suffix = (f"_{a.drop}" if a.drop else "") + ("_lgb" if a.model == "lgb" else "") + ("_off" if a.offset else "") + ("_raw" if a.raw else "")
        tag = f"{pr['labels']['source_sha256'][:8]}_{len(cols)}{suffix}{'_' + drop_sha if drop_sha else ''}"
        for t in years:
            te = yr == t
            acc = np.zeros(int(te.sum()))
            for i, ps in enumerate(sets):
                part = parts / f"{a.arm}_{a.fs}_{t}_b{i}_{tag}.npy"
                if part.exists():
                    acc += np.load(part)
                    continue
                tr = rows & (yr < t)
                if a.model == "lgb":
                    import lightgbm as lgb
                    m = lgb.train(ps, lgb.Dataset(X[tr], y[tr], init_score=None if margin is None else margin[tr]), MC.N_ROUNDS)
                    z = m.predict(X[te], raw_score=True) + (0.0 if margin is None else margin[te])
                    p = 1.0 / (1.0 + np.exp(-z))
                else:
                    dtr = xgb.DMatrix(X[tr], label=y[tr], base_margin=None if margin is None else margin[tr])
                    m = xgb.train(ps, dtr, MC.XGB_ROUNDS)
                    p = m.predict(xgb.DMatrix(X[te], base_margin=None if margin is None else margin[te]))
                np.save(part, p)
                acc += p
                log(f"  {a.arm} {a.fs} {t} bag{i}: 学習 {int(tr.sum())} → 予測 {int(te.sum())}（{time.time() - t0:.0f} 秒）")
            P[te] = acc / len(sets)
        out["p"] = P
        name = f"{a.arm}_{a.fs}{suffix}"
        dest = OUT / f"pred_{name}.parquet"
        out[np.isin(yr, years)].to_parquet(dest, index=False)
        (OUT / f"pred_{name}.json").write_text(json.dumps(
            {"arm": a.arm, "fs": a.fs, "drop": a.drop, "model": a.model, "offset": a.offset, "raw": a.raw, "drop_sha8": drop_sha, "years": years, "n_cols": len(cols), "join": stat, "prereg": pr["version"], "tag": tag,
             "label_sha256": pr["labels"]["source_sha256"], "params": sets, "rounds": MC.N_ROUNDS if a.model == "lgb" else MC.XGB_ROUNDS, "sec": round(time.time() - t0)},
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


def calibrated_market_q3(eval_years=(2018, 2019, 2020, 2021)) -> pd.DataFrame:
    """較正した市場の 3 着内の確率（再現 k ごと）。前の試み（longshot_model）の式: logit(q)・2 乗・logit(Harville q3)・
    頭数の帯・クラスの群のダミーと積のロジスティック回帰。年 t は t より前の年（2016〜）の全馬で当てる（la-2 の決定 5 (iii)）。
    当てる側の着順は 2016〜(max(eval_years)−1) 年だけ。結果は OUT/market_cq3.parquet（A/E の分母にも使う）。"""
    from src.ops import data_ledger as DL
    from src.scripts.data import longshot_model as LM
    dest = OUT / "market_cq3.parquet"
    pr = prereg()
    stamp = {"eval_years": list(eval_years), "label_sha256": pr["labels"]["source_sha256"], "cols": "cq3_0..4+rq3_0..4+fq3",
             "code_sha256": sha256(Path(__file__))[:16]}
    side = dest.with_suffix(".json")
    if dest.exists() and side.exists():
        old = json.loads(side.read_text(encoding="utf-8"))
        keys = ("eval_years", "label_sha256", "cols")
        if {k: old.get(k) for k in keys} == {k: stamp[k] for k in keys}:
            return pd.read_parquet(dest)
        raise RuntimeError(f"market_cq3 の作りが違う（{old} ≠ {stamp}）。消してから作り直す")
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
        for k in list(range(N_PRE)) + ["final"]:
            oc = "odds_final" if k == "final" else f"odds_pre{k}"
            q = q_from_odds(race, r[oc].to_numpy(float))
            q = np.clip(np.where(q > 0, q, 1e-5), 1e-5, 1.0)
            q3 = raw_q3(race, r[oc].to_numpy(float))
            des = LM.calib_design(LM.mm.logit(q), LM.mm.logit(q3), fband, cgrp)
            cq = np.full(len(r), np.nan)
            for t in eval_years:
                tr = (yr < t) & r["y3"].notna().to_numpy()
                m = LM.calibrate_fit(des[tr], r["y3"].to_numpy()[tr].astype(int))
                te = yr == t
                cq[te] = LM.sigmoid(LM.calibrate_logit(m, des[te]))
            ok = np.isin(yr, eval_years)
            cq[ok] = LM.norm_race_cap(race[ok], cq[ok], 3.0)
            if k == "final":
                out["fq3"] = cq                                  # 確定オッズの較正した q3（la-6 の A/E の分母）
            else:
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


# ---------------------------------------------------------------- 決める段（2019〜21、la-6 の合否・la-5 の中止の馬の 3 通り）
MAIN_REPO = Path(os.environ.get("KEIBA_MAIN_REPO") or "/home/jovyan/work/keiba-vpn")      # worktree には data/ が無いので本体の置き場を読む
PAYOFF_PATH = MAIN_REPO / "data/local/modeling/betting/payoff.parquet"
TABLES_MAIN = MAIN_REPO / "data/page_reference/tables"


def cluster_boot(cl: np.ndarray, num: dict, den: dict | None = None, n_boot: int = 2000, seed: int = 20261127) -> dict:
    """レースの値を束（開催日 × 場）で引き直す対のブートストラップ。num[name] はレースの値、den[name] があれば比 Σnum/Σden、
    無ければ平均 Σnum/レース数。全部の name で同じ引き直しを使う。返すのは点・2.5%・97.5%・5%（片側用）。"""
    codes, ci = np.unique(cl, return_inverse=True)
    G = len(codes)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    cnt = np.bincount(ci, minlength=G).astype(float)
    out = {}
    for k, v in num.items():
        s = np.bincount(ci, weights=v, minlength=G)
        if den is not None and k in den:
            d = np.bincount(ci, weights=den[k], minlength=G)
            pt, bs = s.sum() / d.sum(), s[idx].sum(1) / d[idx].sum(1)
        else:
            pt, bs = s.sum() / cnt.sum(), s[idx].sum(1) / cnt[idx].sum(1)
        out[k] = {"point": float(pt), "lo95": float(np.quantile(bs, 0.025)), "hi95": float(np.quantile(bs, 0.975)),
                  "lo90": float(np.quantile(bs, 0.05))}
    return out


def pick_rows(race_idx: np.ndarray, R: int, mask: np.ndarray) -> np.ndarray:
    """bool の印（1 レース 1 頭）を、レースの添字 → 行の位置（無ければ −1）に。"""
    out = np.full(R, -1)
    rows = np.flatnonzero(mask)
    out[race_idx[rows]] = rows
    return out


def race_avg(vals: np.ndarray, sel: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(K, R) の行の位置 sel について、ok の再現だけで値を平均したレースの値（ok が無いレースは NaN）。
    vals は行の値（1 次元）か、行 × 再現（2 次元。再現 k の列を使う）。"""
    s = np.where(sel >= 0, sel, 0)
    if vals.ndim == 1:
        v = vals[s]
    else:
        v = np.take_along_axis(vals.T, s, axis=1)
    v = np.where(ok, v, 0.0)
    n = ok.sum(0)
    return np.where(n > 0, v.sum(0) / np.maximum(n, 1), np.nan)


SEL_YEARS = (2018, 2019, 2020, 2021)       # 2018 は Platt・2 段の当てはめ（t−1 年）にだけ使う。物差しは 2019〜21


def select_frame(fs: str, star: str = "", dry: bool = False) -> pd.DataFrame:
    """評価の行（runners の 2018〜21・重複を除く）に、予測・較正した市場の q3（再現 cq3_k・確定 fq3）・本物のラベル（y3）・
    単勝と複勝の払戻（ret_tan・ret_fuku。1 円あたり）を結ぶ。予測の無い馬（中止など）は y3 = 0 で、fin ≤ 0 であることを確かめる。
    dry=True は着順・払戻を読まず、y3 を Bernoulli(fq3)、払戻を作り物にする（道具の空運転）。"""
    r = eval_frame(fs, years=SEL_YEARS, star=star)
    cq = calibrated_market_q3()
    r = r.merge(cq.drop(columns=["year"]), on=KEY, how="left")
    nopred = r["p_L1"].isna() | r["p_C0"].isna()
    if dry:
        rng = np.random.default_rng(1)
        r["y3"] = (rng.random(len(r)) < r["fq3"].to_numpy(float)) & ~nopred.to_numpy()
        r["ret_tan"] = np.where(rng.random(len(r)) < r["fq3"].to_numpy(float) / 3, r["odds_final"].to_numpy(float), 0.0)
        r["ret_fuku"] = np.where(r["y3"], r["odds_final"].to_numpy(float) / 4 + 1.0, 0.0)
        r["dnf"] = nopred.to_numpy()
    else:
        lab = load_labels(2021)
        r = r.merge(lab[KEY + ["y3"]], on=KEY, how="left")
        fin = pd.read_parquet(RUNNERS, columns=KEY + ["fin", "ret_tan", "ret_fuku"], filters=[("year", "in", list(SEL_YEARS))])
        fin["race_id"], fin["horse_id"] = fin["race_id"].astype(str), fin["horse_id"].astype(str)
        r = r.merge(fin.drop_duplicates(KEY), on=KEY, how="left")
        bad = nopred & (pd.to_numeric(r["fin"], errors="coerce").fillna(0) > 0)
        assert not bad.any(), f"予測が無いのに着順のある馬がいる: {int(bad.sum())}"
        r["dnf"] = nopred.to_numpy()
        r["y3"] = r["y3"].fillna(False).astype(bool) & ~r["dnf"]
    assert r[[f"cq3_{k}" for k in range(N_PRE)] + ["fq3"]].notna().all().all(), "較正した q3 の欠け"
    return r.reset_index(drop=True)


def case_picks(r: pd.DataFrame, case: str, R: int, race_idx: np.ndarray, odds_cols: list[str]) -> dict[str, np.ndarray]:
    """中止の馬の 3 通り（la-5）ごとの、(K, R) の行の位置: star・c0・market。odds_cols は母集団を決めるオッズ（再現 5 通りか確定）。"""
    race = r["race_id"].to_numpy()
    hn = r["hn"].to_numpy(float)
    dnf = r["dnf"].to_numpy()
    K = len(odds_cols)
    out = {n: np.full((K, R), -1) for n in ("star", "c0", "market")}
    for k, oc in enumerate(odds_cols):
        o = r[oc].to_numpy(float)
        pop_all = population(o)
        pop_pred = pop_all & ~dnf
        st = race_top1(race, r["p_L1"].fillna(-np.inf).to_numpy(float), pop_pred, hn)
        c0 = race_top1(race, r["p_C0"].fillna(-np.inf).to_numpy(float), pop_pred, hn)
        mk = race_top1(race, -o, pop_pred if case == "a_drop" else pop_all, hn)
        out["star"][k] = pick_rows(race_idx, R, st)
        out["c0"][k] = pick_rows(race_idx, R, c0)
        out["market"][k] = pick_rows(race_idx, R, mk)
        if case == "c_worst":
            # 母集団に中止の馬がいるレースは、★ がその馬（確定の較正した q3 の最も高い 1 頭）を選んだとする
            w = race_top1(race, r["fq3"].to_numpy(float), pop_all & dnf, hn)
            sel = pick_rows(race_idx, R, w)
            out["star"][k] = np.where(sel >= 0, sel, out["star"][k])
    return out


def select_metrics(r: pd.DataFrame, case: str, odds_cols: list[str] | None = None, n_boot: int = 2000) -> dict:
    """1 つの扱い（case）・1 つの母集団の決め方（odds_cols）の物差し: ★ − C0・★ − 市場の 3 着内率の差、
    ★・C0・市場の A/E（主は確定の q3 = fq3、記述は再現の q3 = cq3_k）、年ごと、帯ごとの A/E、確定で一桁かの 2 層。"""
    odds_cols = odds_cols or [f"odds_pre{k}" for k in range(N_PRE)]
    codes, race_idx = np.unique(r["race_id"].to_numpy(), return_inverse=True)
    R = len(codes)
    y = r["y3"].to_numpy(float)
    fq = r["fq3"].to_numpy(float)
    cq = np.column_stack([r[f"cq3_{k}"].to_numpy(float) for k in range(N_PRE)]) if len(odds_cols) == N_PRE else fq[:, None]
    P = case_picks(r, case, R, race_idx, odds_cols)
    ok = {n: P[n] >= 0 for n in P}
    rate = {n: race_avg(y, P[n], ok[n]) for n in P}
    e_f = {n: race_avg(fq, P[n], ok[n]) for n in P}
    e_c = {n: race_avg(cq, P[n], ok[n]) for n in P}
    both_c = ok["star"] & ok["c0"]
    both_m = ok["star"] & ok["market"]
    d_sc = race_avg(y, P["star"], both_c) - race_avg(y, P["c0"], both_c)
    d_sm = race_avg(y, P["star"], both_m) - race_avg(y, P["market"], both_m)
    clus = pd.Series(cluster_of(r["race_id"], r["date"])).groupby(race_idx).first().reindex(range(R)).to_numpy()
    use = np.isfinite(d_sc) & np.isfinite(d_sm) & np.isfinite(e_f["star"]) & np.isfinite(e_f["c0"]) & np.isfinite(e_f["market"])
    num = {"d_star_c0": d_sc[use], "d_star_market": d_sm[use]}
    den = {}
    for n in ("star", "c0", "market"):
        num[f"rate_{n}"] = rate[n][use]
        num[f"ae_{n}"] = rate[n][use]
        den[f"ae_{n}"] = e_f[n][use]
        num[f"ae_pre_{n}"] = rate[n][use]
        den[f"ae_pre_{n}"] = e_c[n][use]
    b = cluster_boot(clus[use], num, den=den, n_boot=n_boot)
    yr = pd.Series(r["year"].to_numpy()).groupby(race_idx).first().reindex(range(R)).to_numpy()
    pop0 = population(r[odds_cols[0]].to_numpy(float))
    by_year = {int(t): {"races": int((use & (yr == t)).sum()), "d_star_c0": round(float(np.nanmean(d_sc[use & (yr == t)])), 4),
                        "d_star_market": round(float(np.nanmean(d_sm[use & (yr == t)])), 4),
                        "rate_star": round(float(np.nanmean(rate["star"][use & (yr == t)])), 4),
                        "pop_mean_y3": round(float(r.loc[pop0 & (r["year"] == t).to_numpy(), "y3"].mean()), 4)}
               for t in (2019, 2020, 2021)}
    # 再現 × レースの組ごとの値（帯ごとの A/E・2 層）
    of = r["odds_final"].to_numpy(float)
    band, layers = {}, {}
    for n in ("star", "c0", "market"):
        sel = np.concatenate([P[n][k][(P[n][k] >= 0) & use] for k in range(len(odds_cols))])
        opre = np.concatenate([r[odds_cols[k]].to_numpy(float)[P[n][k][(P[n][k] >= 0) & use]] for k in range(len(odds_cols))])
        for name, lo, hi in (("10-15", 10, 15), ("15-50", 15, 50), ("50+", 50, np.inf)):
            mm = (opre >= lo) & (opre < hi)
            band.setdefault(n, {})[name] = {"n_pairs": int(mm.sum()), "rate": round(float(y[sel][mm].mean()), 4) if mm.any() else None,
                                            "ae": round(float(y[sel][mm].sum() / fq[sel][mm].sum()), 4) if mm.any() else None}
        for name, mm in (("final_lt10", of[sel] < THRESH), ("final_ge10", of[sel] >= THRESH)):
            layers.setdefault(n, {})[name] = {"share": round(float(mm.mean()), 4), "rate": round(float(y[sel][mm].mean()), 4) if mm.any() else None,
                                              "ae": round(float(y[sel][mm].sum() / fq[sel][mm].sum()), 4) if mm.any() else None}
    dnf_pop = r["dnf"].to_numpy() & pop0
    return {"case": case, "odds": odds_cols, "races_used": int(use.sum()), "races_dnf_in_pop": int(np.unique(race_idx[dnf_pop]).size),
            "boot": b, "by_year": by_year, "ae_by_band": band, "layers": layers,
            "agree_star_c0": float((P["star"] == P["c0"])[both_c].mean()),
            "_d_sc": d_sc, "_use": use, "_clus": clus, "_ys": rate["star"], "_P": P, "_race_idx": race_idx}


def judge_select(m: dict, m_final: dict, m_v6: dict | None) -> dict:
    """決める段の合否（la-6）。m は主の母集団（再現 5 通り）、m_final は確定の母集団、m_v6 は v6 の同じ扱い。"""
    b = m["boot"]
    i = b["d_star_c0"]["lo95"] > 0
    ii = b["d_star_market"]["lo95"] > 0
    iii = b["ae_star"]["lo95"] > 1.0 and b["ae_star"]["point"] >= b["ae_c0"]["point"]
    iv = None if m_v6 is None else m_v6["boot"]["d_star_c0"]["point"] > 0
    v = m_final["boot"]["d_star_c0"]["point"] > 0 and m_final["boot"]["d_star_market"]["point"] > 0
    return {"i": bool(i), "ii": bool(ii), "iii": bool(iii), "iv": iv, "v": bool(v), "pass": bool(i and ii and iii and bool(iv) and v)}



def _platt_fit(z: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1) if z.ndim == 1 else z, y.astype(int))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def secondary_metrics(r_all: pd.DataFrame, n_boot: int = 2000) -> dict:
    """副の物差し（prereg の secondary。判定に使わない）。母集団は確定 10 倍以上で予測のある馬（学習の母集団と同じ定義）。
    - AI（★ の腕 p_L1）と C0 は、年 t を t−1 年の予測で当てた Platt で本物の 3 着内の確率に直す。市場は確定の較正した q3（fq3）。
    - 対数損失・AUC（全体・レース内の平均）・ECE（十分位）・上位 10% の 3 着内率。
    - Benter の 2 段: logit(AI の較正後) と logit(fq3) のロジスティック回帰（t−1 年で当てる）と、logit(fq3) だけの回帰の対数損失の差
      （市場だけ − 2 段。正が上乗せ）。レース内の条件付きロジットではなく 2 値の回帰（母集団だけを学んだ AI に合わせた近似）。"""
    from sklearn.metrics import roc_auc_score
    yr = r_all["year"].to_numpy()
    pop = population(r_all["odds_final"].to_numpy(float)) & ~r_all["dnf"].to_numpy()
    y = r_all["y3"].to_numpy(int)
    zA, zC, zM = _logit(r_all["p_L1"].fillna(0.5)), _logit(r_all["p_C0"].fillna(0.5)), _logit(r_all["fq3"])
    pa, pc, b2, bm = (np.full(len(r_all), np.nan) for _ in range(4))
    for t in (2019, 2020, 2021):
        tr, te = pop & (yr == t - 1), pop & (yr == t)
        pa[te] = _platt_fit(zA[tr], y[tr]).predict_proba(zA[te].reshape(-1, 1))[:, 1]
        pc[te] = _platt_fit(zC[tr], y[tr]).predict_proba(zC[te].reshape(-1, 1))[:, 1]
        zAc_tr = _logit(_platt_fit(zA[tr], y[tr]).predict_proba(zA[tr].reshape(-1, 1))[:, 1])
        X2tr, X2te = np.column_stack([zAc_tr, zM[tr]]), np.column_stack([_logit(pa[te]), zM[te]])
        b2[te] = _platt_fit(X2tr, y[tr]).predict_proba(X2te)[:, 1]
        bm[te] = _platt_fit(zM[tr], y[tr]).predict_proba(zM[te].reshape(-1, 1))[:, 1]
    ev = pop & np.isin(yr, (2019, 2020, 2021))
    yy = y[ev]
    race = r_all["race_id"].to_numpy()[ev]

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

    def auc_in_race(p):
        d = pd.DataFrame({"r": race, "y": yy, "p": p})
        v = [roc_auc_score(g["y"], g["p"]) for _, g in d.groupby("r") if 0 < g["y"].sum() < len(g)]
        return float(np.mean(v))

    def ece(p):
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        d = pd.DataFrame({"q": q, "y": yy, "p": p}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
        return float((d["n"] * (d["y"] - d["p"]).abs()).sum() / d["n"].sum()), d.round(4).to_dict("index")

    out = {"pop": "確定 10 倍以上・予測のある馬（2019〜21）", "rows": int(ev.sum())}
    for name, p in (("ai", pa[ev]), ("c0", pc[ev]), ("market", r_all["fq3"].to_numpy(float)[ev]), ("benter", b2[ev]), ("market_refit", bm[ev])):
        e, tab = ece(p)
        top = p >= np.quantile(p, 0.9)
        out[name] = {"logloss": round(float(ll(p).mean()), 5), "auc": round(float(roc_auc_score(yy, p)), 4), "auc_in_race": round(auc_in_race(p), 4),
                     "ece": round(e, 5), "top10_rate": round(float(yy[top].mean()), 4), "deciles": tab}
    clus = cluster_of(r_all["race_id"][ev], r_all["date"][ev])
    cnt = np.ones(int(ev.sum()))
    gain = ll(bm[ev]) - ll(b2[ev])
    ai_vs_mkt = ll(r_all["fq3"].to_numpy(float)[ev]) - ll(pa[ev])
    b = cluster_boot(clus, {"benter_gain": gain, "ai_vs_market": ai_vs_mkt}, den={"benter_gain": cnt, "ai_vs_market": cnt}, n_boot=n_boot)
    out["benter_gain_ll"] = b["benter_gain"]
    out["ai_minus_market_ll"] = b["ai_vs_market"]
    return out


def layer_describe(r: pd.DataFrame, m: dict) -> dict:
    """層の記述（会議の決定 3。判定に使わない）: 新馬（cls 0）・7 頭以下・未勝利（1）・条件戦（2〜5）・OP と重賞（6〜8）ごとの
    ★ − C0 の差の平均と ★ の 3 着内率（主の母集団・中止の馬は外したまま）。"""
    ri = m["_race_idx"]
    R = int(ri.max() + 1)
    cls = pd.Series(r["cls"].to_numpy()).groupby(ri).first().reindex(range(R)).to_numpy(float)
    fld = pd.Series(r["field"].to_numpy()).groupby(ri).first().reindex(range(R)).to_numpy(float)
    use = m["_use"]
    out = {}
    for name, mk in (("新馬", cls == 0), ("7 頭以下", fld <= 7), ("未勝利", cls == 1), ("条件戦", (cls >= 2) & (cls <= 5)),
                     ("OP・重賞", cls >= 6)):
        u = use & mk
        out[name] = {"races": int(u.sum()), "d_star_c0": round(float(np.nanmean(m["_d_sc"][u])), 4) if u.any() else None,
                     "rate_star": round(float(np.nanmean(m["_ys"][u])), 4) if u.any() else None}
    return out


def single_describe(r: pd.DataFrame, m: dict) -> dict:
    """★・C0・市場・母集団の無作為の単勝・複勝の回収率（再現 0・確定の払戻・記述だけ）。束ねた区間と上位 5 件を除いた値。"""
    P = m["_P"]
    ri = m["_race_idx"]
    R = int(ri.max() + 1)
    pop0 = population(r["odds_pre0"].to_numpy(float)) & ~r["dnf"].to_numpy()
    ok = m["_use"] & (P["star"][0] >= 0) & (P["c0"][0] >= 0) & (P["market"][0] >= 0)
    out = {}
    for col in ("ret_tan", "ret_fuku"):
        v = r[col].to_numpy(float)
        vals = {n: v[P[n][0][ok]] for n in ("star", "c0", "market")}
        rnd = pd.Series(np.where(pop0, v, np.nan)).groupby(ri).mean().reindex(range(R)).to_numpy()
        vals["random"] = rnd[ok]
        b = cluster_boot(m["_clus"][ok], {f"roi_{n}": vals[n] for n in vals}, n_boot=2000)
        top5 = {n: round(float(np.sort(vals[n])[:-5].sum() / (len(vals[n]) - 5)), 4) for n in vals}
        out[col] = {"races": int(ok.sum()), "boot": b, "hit_rate": {n: round(float((vals[n] > 0).mean()), 4) for n in vals}, "roi_drop_top5": top5}
    out["note"] = "記述だけ。再現 0。確定の払戻（runners の ret_tan・ret_fuku）"
    return out


def positive_control(m: dict, lift: float = 0.015, reps: int = 200, n_boot: int = 2000, seed: int = 20261128) -> dict:
    """正の対照（la-2 の決定 8）: 帰無を中心にした差に、平均で lift の ★ だけの上乗せを植え、(i) の下端が 0 を超える割合を数える。

    - 観測の差 d をレースの平均 0 に寄せ（d − 平均）、そこに「★ が 3 着外で C0 と違う馬の再現」の分だけの当たりを、平均 lift に
      なる割合で植える（合成の列を学び直す代わり。観測の差の符号に左右されないように中心を 0 に置く）。
    - 植えられる量（room の平均）が lift に届かないときは、届いた量を記録する。"""
    use = m["_use"]
    d, cl, ys = m["_d_sc"][use], m["_clus"][use], m["_ys"][use]
    P = m["_P"]
    both = (P["star"] >= 0) & (P["c0"] >= 0)
    differ = (((P["star"] != P["c0"]) & both).sum(0) / np.maximum(both.sum(0), 1))[use]      # d と同じ分母（両方ある再現）
    room = np.clip(1.0 - ys, 0, 1) * differ
    pi = min(1.0, lift / max(room.mean(), 1e-9))
    planted = float(room.mean() * pi)
    rng = np.random.default_rng(seed)
    base = d - d.mean()
    hit = 0
    for i in range(reps):
        dd = base + room * (rng.random(len(d)) < pi)
        hit += cluster_boot(cl, {"d": dd}, n_boot=n_boot, seed=seed + i)["d"]["lo95"] > 0
    return {"lift": lift, "planted_mean": round(planted, 5), "reached": planted >= lift - 1e-9, "reps": reps,
            "detected": hit / reps, "pi": float(pi), "blunt": (hit / reps) < 0.7,
            "how": "帰無を中心にした観測の差 + ★ だけの当たりの植え込み（合成の列を学び直す代わりの近似）"}


def load_pair_payoffs(race_ids: set[str]) -> dict:
    """馬連・ワイドの確定の払戻（100 円あたり）。2020〜21 は払戻の表を race_id で絞って読み、無いレース（2019 年など）は
    結果の表の払戻の列で補う。2022 年以降は読まない。使うレースがすべて馬連・ワイドを持つことを確かめる。"""
    from src.scripts.data import bet_cycles as BC
    ys = sorted({int(x[:4]) for x in race_ids})
    assert max(ys) <= 2021, "払戻は 2021 年までしか読まない"
    ids = sorted(race_ids)
    t = pd.read_parquet(PAYOFF_PATH, columns=["race_id", "bet_type", "combo", "payout"], filters=[("race_id", "in", ids)])
    t["race_id"] = t["race_id"].astype(str)
    assert t["race_id"].str[:4].astype(int).max() <= 2021 if len(t) else True
    pay: dict = {}
    for rid, bt, combo, v in zip(t["race_id"], t["bet_type"], t["combo"].astype(str), t["payout"]):
        if bt not in ("umaren", "wide"):
            continue
        sep = ">" if ">" in combo else "-"
        pay.setdefault(rid, {}).setdefault(bt, {})[BC.combo_key(bt, combo.split(sep))] = int(v)
    for y in ys:
        f = pd.read_parquet(TABLES_MAIN / str(y) / "race_result_flat.parquet", columns=["race_id", "payoff"]).drop_duplicates("race_id")
        for rid, raw in zip(f["race_id"].astype(str), f["payoff"]):
            if rid in race_ids and not ("umaren" in pay.get(rid, {}) and "wide" in pay.get(rid, {})):
                x = BC.parse_flat_payoff(raw)
                if x:
                    pay[rid] = {bt: x[bt] for bt in ("umaren", "wide") if bt in x}
    miss = [rid for rid in race_ids if not ("umaren" in pay.get(rid, {}) and "wide" in pay.get(rid, {}))]
    assert len(miss) <= 0.005 * len(race_ids), f"馬連・ワイドの払戻の無いレースが多い: {len(miss)}"
    return pay, miss


def fake_pair_payoffs(r: pd.DataFrame) -> tuple[dict, list]:
    """空運転の作り物の払戻（y3 の作り物の 3 着内の組に 500 円・1・2 着の組に 1,000 円）。"""
    from src.scripts.data import bet_cycles as BC
    pay = {}
    for rid, g in r[r["y3"]].groupby("race_id"):
        h = g.sort_values("fq3", ascending=False)["hn"].astype(int).tolist()[:3]
        pay[rid] = {"wide": {BC.combo_key("wide", [a, b]): 500 for i, a in enumerate(h) for b in h[i + 1:]},
                    "umaren": {BC.combo_key("umaren", h[:2]): 1000} if len(h) >= 2 else {}}
    for rid in set(r["race_id"]) - set(pay):
        pay[rid] = {"wide": {}, "umaren": {}}
    return pay, []


def pair_describe(r: pd.DataFrame, m: dict, n_null: int = 1000, seed: int = 20261129, dry: bool = False) -> dict:
    """組の記述（la-2 の決定 7 の型・買い方にしない）。物差しをそろえるため、すべて再現 0 の母集団と再現 0 の ★ で数える。

    - ★ を軸の片方、相手はめぐ（v5bag pr_win）のレースの 1 位（★ と同じ馬なら 2 位）の馬連・ワイド 1 点。
    - 比べは同じ相手で、軸を C0・市場（再現 0 の最低オッズ）・母集団の無作為（期待値）に替えた並び。
    - 束ねた区間・上位 5 件を除いた値・帰無（★ を母集団の中の無作為に替えて n_null 回、★ 以上の回収率になる割合。同じレースの集合）。
    - 払戻の無いレース（0.5% まで）は外して数を記録する。返還は数えない（a_drop では軸・相手が中止の馬にならない）。"""
    from src.scripts.data import bet_cycles as BC
    race = r["race_id"].to_numpy()
    race_idx = m["_race_idx"]
    R = int(race_idx.max() + 1)
    pay, miss = fake_pair_payoffs(r) if dry else load_pair_payoffs(set(race))
    hn = r["hn"].to_numpy(int)
    mw = np.where(r["dnf"].to_numpy(), np.nan, r["megu_win"].to_numpy(float))       # 相手に中止の馬を選ばない
    order = np.lexsort((hn, -np.nan_to_num(mw, nan=-1.0), race_idx))
    first, second = np.full(R, -1), np.full(R, -1)
    for i in order:
        ri = race_idx[i]
        if first[ri] < 0:
            first[ri] = i
        elif second[ri] < 0:
            second[ri] = i
    rid_of = np.empty(R, dtype=object)
    rid_of[race_idx] = race
    has_pay = np.array([rid_of[ri] not in set(miss) for ri in range(R)])
    P = m["_P"]
    pop0 = population(r["odds_pre0"].to_numpy(float)) & ~r["dnf"].to_numpy()
    ok = m["_use"] & has_pay & (P["star"][0] >= 0) & (P["c0"][0] >= 0) & (P["market"][0] >= 0)
    cand = [np.flatnonzero((race_idx == ri) & pop0) for ri in range(R)]
    clus = m["_clus"]

    dnf = r["dnf"].to_numpy()

    def partner(ri, a):
        b = first[ri] if first[ri] != a else second[ri]
        assert b < 0 or not dnf[b], "相手が中止の馬になった"
        return b

    def pay_of(bt, ri, a):
        b = partner(ri, a)
        if a < 0 or b < 0:
            return np.nan
        return float(pay[rid_of[ri]][bt].get(BC.combo_key(bt, [hn[a], hn[b]]), 0))

    out = {"races_no_payoff": len(miss)}
    rows_ok = np.flatnonzero(ok)
    for bt in ("umaren", "wide"):
        vals = {n: np.array([pay_of(bt, ri, P[n][0][ri]) for ri in rows_ok]) for n in ("star", "c0", "market")}
        vals["random"] = np.array([np.mean([pay_of(bt, ri, a) for a in cand[ri]]) for ri in rows_ok])
        b = cluster_boot(clus[ok], {f"roi_{n}": vals[n] / 100.0 for n in vals}, n_boot=2000)
        hit = {n: round(float((vals[n] > 0).mean()), 4) for n in vals}
        top5 = {n: round(float(np.sort(vals[n])[:-5].sum() / (100.0 * (len(vals[n]) - 5))), 4) for n in vals}
        rng = np.random.default_rng(seed)
        pays = [np.array([pay_of(bt, ri, a) for a in cand[ri]]) for ri in rows_ok]
        base = vals["star"].mean()
        ge = 0
        for _ in range(n_null):
            s = np.array([p[rng.integers(len(p))] for p in pays])
            ge += s.mean() >= base
        out[bt] = {"races": int(ok.sum()), "boot": b, "hit_rate": hit, "roi_drop_top5": top5, "null_p_ge_star": ge / n_null}
    out["note"] = ("記述だけ。買い方にしない（pair-bets・rules-council の材料）。再現 0 だけで数える。"
                   "払戻は確定（2020〜21 は払戻の表、2019 年は結果の表の列）")
    return out


DESCRIBE_ARMS = ["L0_v5", "LA_v5", "LC_v5", "L1T_v5", "L1_v5_raw", "L1_v5_lgb", "L1_v5_off", "L1_v7", "L1_v7_rho50", "L1_v7_rho30"]


def describe_fs(name: str) -> str:
    fs = name.split("_")[1]
    assert fs in ("v5", "v6", "v7"), name
    return fs


def cmd_select(a) -> None:
    """決める段を 1 回だけ流す（台帳 select・2019〜21）。合否は la-5 の 3 通りすべてで la-6 の線を越えたときだけ合格。

    - 開ける前に全部の腕（主・C0・v6・記述の腕と、その C0）の予測と、較正した q3 の表がそろっていることを確かめる。
    - 落ちうる準備（全部の評価の表）を合否の計算より前に済ませる。合否は json に書いた後にだけ表示する。
    - 区間は、1 回の select_metrics の中では全部の物差しで同じ引き直し。確定の母集団・v6・記述の腕は使うレースの集合が違うので
      別の引き直し（腕どうしの差の対の区間は出さない。記述の限界）。
    - A/E は Σ（レースの再現の平均の y）÷ Σ（再現の平均の q3）で、「再現ごとの比の平均」の近似。
    - --dry は着順・払戻を読まない空運転（y3 と払戻は作り物。台帳を使わず select_dry.json に書く）。"""
    from contextlib import nullcontext

    from src.ops import data_ledger as DL
    pr = prereg()
    assert pr["version"] == "la-7", pr["version"]
    dest = OUT / ("select_dry.json" if a.dry else "select_main.json")
    assert a.dry or not dest.exists(), f"{dest} はもうある（決める段は 1 回だけ）"
    describe = DESCRIBE_ARMS if not a.describe else [x for x in a.describe.split(",") if x]
    need = ["L1_v5", "C0_v5", "L1_v6", "C0_v6"] + describe + sorted({f"C0_{describe_fs(n)}" for n in describe})
    for n in need:
        assert (OUT / f"pred_{n}.parquet").exists(), f"pred_{n} が無い（開ける前に全部の腕を学ぶ）"
    calibrated_market_q3()                                   # 較正の表は guard の外で先に作る（作りが違えば例外で止まる）
    t0 = time.time()
    guard = nullcontext() if a.dry else DL.guard(
        tool="src/research/longshot_ai/lsai.py:select", family="longshot_ai", version=pr["version"],
        period=("2016-01-01", "2021-12-31"), purpose="select", reads=("着順", "払戻"), frozen_paths=[PREREG, Path(__file__)],
        reason="穴馬 AI の決める段（2019〜21）を事前登録 la-7 のとおり 1 回だけ流す")
    with guard as tok:
        def ev(f):
            return f[f["year"] >= 2019].reset_index(drop=True)
        r_all = select_frame("v5", dry=a.dry)
        r6_all = select_frame("v6", dry=a.dry)
        rd_all = {n: select_frame(describe_fs(n), star=n, dry=a.dry) for n in describe}
        r, r6 = ev(r_all), ev(r6_all)
        res = {"prereg": pr["version"], "main": "L1_v5", "dry": bool(a.dry), "cases": {}, "describe": {}}
        for case in ("a_drop", "b_zero", "c_worst"):
            m = select_metrics(r, case)
            mf = select_metrics(r, case, odds_cols=["odds_final"])
            m6 = select_metrics(r6, case)
            out = {k: v for k, v in m.items() if not k.startswith("_")}
            out["final_pop"] = {k: v for k, v in mf.items() if not k.startswith("_")}
            out["v6"] = {k: v for k, v in m6.items() if not k.startswith("_")}
            out["judge"] = judge_select(m, mf, m6)
            if case == "a_drop":
                out["positive_control"] = positive_control(m)
                out["pairs"] = pair_describe(r, m, dry=a.dry)
                out["single"] = single_describe(r, m)
                out["layers_race"] = layer_describe(r, m)
            res["cases"][case] = out
            log(f"{case} を数えた（{time.time() - t0:.0f} 秒）")
        res["secondary"] = {"L1_v5": secondary_metrics(r_all), "L1_v6": secondary_metrics(r6_all)}
        for n, f in rd_all.items():
            md = select_metrics(ev(f), "a_drop")
            res["describe"][n] = {"boot": md["boot"], "by_year": md["by_year"], "layers": md["layers"], "ae_by_band": md["ae_by_band"],
                                  "agree_star_c0": md["agree_star_c0"], "races_used": md["races_used"], "secondary": secondary_metrics(f)}
        log(f"記述の腕を数えた（{time.time() - t0:.0f} 秒）")
        res["pass"] = all(res["cases"][c]["judge"]["pass"] for c in res["cases"])
        only_i = all(all(res["cases"][c]["judge"][x] for x in ("ii", "iii", "iv", "v")) for c in res["cases"])
        pc = res["cases"]["a_drop"]["positive_control"]
        if res["pass"]:
            res["verdict"] = "確かめの段へ"
        elif pc["blunt"] and only_i:
            res["verdict"] = "判定できない（(i) だけで落ち、物差しが鈍い）"
        else:
            res["verdict"] = "不合格（閉じる）"
        res["sec"] = round(time.time() - t0)
        dest.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        if tok is not None:
            tok.finish(result_path=dest)
    if a.dry:
        print("空運転", dest, f"{res['sec']} 秒")                # 作り物の数は表示しない
    else:
        print(json.dumps({c: res["cases"][c]["judge"] for c in res["cases"]}, ensure_ascii=False), res["verdict"])


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=["L1", "C0", "L0", "LA", "LC", "L1T"], required=True)
    t.add_argument("--model", choices=["xgb", "lgb"], default="xgb")
    t.add_argument("--offset", action="store_true", help="市場の q3 の logit を offset（記述の腕 L1_off）")
    t.add_argument("--raw", action="store_true", help="方針で外す列も残す（記述の腕 L1_M）")
    t.add_argument("--fs", choices=["v5", "v6", "v7"], default="v5")
    t.add_argument("--years", default=",".join(str(y) for y in PRED_YEARS))
    t.add_argument("--drop", choices=["rho50", "rho30"], default=None, help="市場の logit と相関の高い列を外す（la-3）")
    sub.add_parser("corr")
    s_ = sub.add_parser("select")
    s_.add_argument("--describe", default="", help="記述の腕の pred の名前（カンマ区切り。例: L0_v5,LA_v5）")
    s_.add_argument("--dry", action="store_true", help="着順・払戻を読まない空運転（作り物の y3・払戻）")
    c = sub.add_parser("census")
    c.add_argument("--fs", choices=["v5", "v6", "v7"], default="v5")
    c.add_argument("--star", default="", help="★ の腕（既定 L1_{fs}。例: L1_v5_rho50）")
    p = sub.add_parser("power")
    p.add_argument("--fs", choices=["v5", "v6", "v7"], default="v5")
    p.add_argument("--star", default="")
    p.add_argument("--lift", type=float, default=0.01)
    p.add_argument("--n-sim", type=int, default=1000)
    a = ap.parse_args()
    {"train": cmd_train, "census": cmd_census, "power": cmd_power, "corr": cmd_corr, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
