"""親モデルの条件別の弱み・強み: 回収率を下げる条件・上げる条件を、的中率の差と的中時の払戻の差に分けて出す（読み取りのみ・再学習しない）。

python -m src.scripts.data.rank_condition_analysis [--boot 1000] [--out DIR]

対象（ユーザーの依頼 2026-11-14）:
- v6e_prod: 本番の v6e（2020 年起点）。valid は training_pipeline_v4/tp27_v6e/ens/v6.parquet、2025 年は training_pipeline/final_eval_v6e
- v6e_2016: s18 の候補（2016 年起点の v6e）。valid は base_2016/tp27_v6e/ens/v6.parquet、2025 年は base_2016/final_eval_v6e16
- B3・A6: ランキング比較の最終候補（3 seed）。valid は tp_rank/oof_s3、2025 年は tp_rank/test2025
- market: 市場の基準（最終オッズの 1 番人気 → 2・3 番人気）。4 モデルに共通の、控除率・人気馬の配当の構造を分けるため

買い方（rank_bakeoff_eval と同じ。ユーザー指定）: 予測 1 番手の単勝・複勝（各 100 円）、1 番手 → 2・3 番手の馬連・ワイドの流し（各 2 点 200 円）。
- 的中 = 払戻が 1 つ以上あったレース（流しは 2 点のどちらか。ワイドは 2 点とも当たることがあり、そのときは払戻を足す）。
- 回収率 = 的中率 × 的中時の平均払戻（的中したレースの払戻の合計 ÷ 的中したレース数）÷ 購入額。
- 差の分解（対称）: Δ回収率 = (h1−h0)(a1+a0)/2/c ［的中率の差の分］ + (a1−a0)(h1+h0)/2/c ［的中時の払戻の差の分］。2 つの和はちょうど差になる。

期間: valid 2022〜2024 の OOF（選ぶのに使ってきた期間）が主。2025 年は、valid で決めた区切り・条件のまま、向きの再現の確認だけに使う（選び直さない）。
レース: 5 つ（4 モデル＋市場）すべてに予測があり、4 券種の払戻がそろったレースだけ。

区間: レース単位のブートストラップ（Poisson(1) の重み、全モデル・全条件で同じ重みを使うので、モデル間の差も対になる）。
縮小推定: 条件の軸ごとに、セルの回収率を全体へ寄せる経験ベイズ（τ² はモーメント法）。区間の端も同じ係数で寄せる。
多重比較: Holm（族 = valid の セル × 券種 × モデル）。的中率は二項で安定しているので、主の目印は的中率の区間。
出力: <out>/cells.parquet（全セル）、summary.json（区切り・族の大きさ・上位）、report.md（上位の表）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.rank_bakeoff_eval import BETS, COST, _pair, load_payoff

ROOT = Path(__file__).resolve().parents[3]
B16 = Path(os.environ.get("REBUILD_BASE") or "/home/jovyan/work/retrain_out/base_2016")
MOD = ROOT / "data/local/modeling"
MODELS = {
    "v6e_prod": (MOD / "training_pipeline_v4/tp27_v6e/ens/v6.parquet", MOD / "training_pipeline/final_eval_v6e/test2025_predictions.parquet"),
    "v6e_2016": (B16 / "tp27_v6e/ens/v6.parquet", B16 / "final_eval_v6e16/test2025_predictions.parquet"),
    "B3": (B16 / "tp_rank/oof_s3/B3_xgb_bin_top3.parquet", B16 / "tp_rank/test2025/B3_xgb_bin_top3.parquet"),
    "A6": (B16 / "tp_rank/oof_s3/A6_lgb_xendcg_top5.parquet", B16 / "tp_rank/test2025/A6_lgb_xendcg_top5.parquet"),
}
REF = "v6e_prod"
MARKET = "market"
CONF_SRC = "v6e_2016"           # 自信の度合いの軸（1 番手の確率・1〜2 番手の差・平たさ）は、同じレースの v6e_2016 の確率で決める（全モデル共通の条件）
VALID_YEARS = (2022, 2023, 2024)
TEST_YEARS = (2025,)
MIN_N_LIST = 100                # 上位の一覧に載せる最小のレース数（valid）

# ── 条件（レースの性質。全モデル共通） ──────────────────────────────────────
_Z2H = str.maketrans("０１２３４５６７８９", "0123456789")


def class_level(grade: str) -> str:
    g = str(grade)
    for k in ("G1", "G2", "G3"):
        if k in g:
            return k
    if g in ("OP", "L"):
        return "OP・L"
    return {"新馬": "新馬", "未勝利": "未勝利", "1勝": "1勝", "2勝": "2勝", "3勝": "3勝"}.get(g, "不明")


def age_condition(race_class: str) -> str:
    s = str(race_class).translate(_Z2H)
    for k in ("2歳", "3歳以上", "4歳以上", "3歳"):
        if k in s:
            return k
    return "不明"


def season(month: int) -> str:
    return {12: "冬(12-2月)", 1: "冬(12-2月)", 2: "冬(12-2月)", 3: "春(3-5月)", 4: "春(3-5月)", 5: "春(3-5月)",
            6: "夏(6-8月)", 7: "夏(6-8月)", 8: "夏(6-8月)"}.get(int(month), "秋(9-11月)")


def band(x, edges, labels) -> pd.Series:
    return pd.cut(pd.Series(x), edges, labels=labels, right=False).astype(str)


def race_conditions(flat: pd.DataFrame) -> pd.DataFrame:
    """race_result_flat（出走馬の行）→ レースごとの条件。"""
    f = flat[~flat["scratched"]].copy() if "scratched" in flat else flat.copy()     # 頭数・性別は実際に走った馬で数える
    f["sex"] = f["sex_age"].astype(str).str[0]
    g = f.groupby("race_id")
    r = g.agg(date=("date", "first"), venue=("venue", "first"), surface=("surface", "first"), distance=("distance", "first"),
              grade=("grade", "first"), race_class=("race_class", "first"), going=("track_condition", "first"),
              n_run=("horse_number", "size"), n_mare=("sex", lambda s: int((s == "牝").sum()))).reset_index()
    r["surface"] = r["surface"].astype(str)
    r["dist_band"] = band(r["distance"].astype(float), [0, 1400, 1700, 2000, 2400, 9999], ["~1300m", "1400-1600m", "1700-1900m", "2000-2300m", "2400m~"]).to_numpy()
    r["surf_dist"] = r["surface"] + "×" + r["dist_band"]
    r["class"] = r["grade"].map(class_level)
    r["field"] = band(r["n_run"].astype(float), [0, 10, 14, 16, 99], ["~9頭", "10-13頭", "14-15頭", "16頭~"]).to_numpy()
    r["going"] = r["going"].astype(str).replace({"稍": "稍重", "不": "不良"})
    r["season"] = [season(pd.Timestamp(d).month) for d in r["date"]]
    r["age_cond"] = r["race_class"].map(age_condition)
    r["sex_cond"] = np.where(r["n_mare"] == r["n_run"], "牝馬限定", "混合")
    r["year"] = pd.to_datetime(r["date"]).dt.year
    return r[["race_id", "year", "surface", "dist_band", "surf_dist", "venue", "class", "field", "going", "season", "age_cond", "sex_cond"]]


RACE_DIMS = ["surface", "dist_band", "surf_dist", "venue", "class", "field", "going", "season", "age_cond", "sex_cond",
             "conf_p1", "conf_gap", "conf_flat"]
MODEL_DIMS = ["top1_pop", "top1_odds"]           # モデルごとに違う（そのモデルの 1 番手の人気・オッズ）
DIM_LABEL = {"surface": "芝・ダート", "dist_band": "距離帯", "surf_dist": "芝ダ×距離", "venue": "会場", "class": "クラス", "field": "頭数",
             "going": "馬場状態", "season": "季節", "age_cond": "年齢の条件", "sex_cond": "性別の条件",
             "conf_p1": "1番手の確率の帯（v6e_2016）", "conf_gap": "1・2番手の確率の差（v6e_2016）", "conf_flat": "確率の平たさ（v6e_2016）",
             "top1_pop": "1番手の人気", "top1_odds": "1番手の単勝オッズ"}
BET_LABEL = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連流し", "wide": "ワイド流し"}


def quantile_cuts(x: np.ndarray, q=(0.2, 0.4, 0.6, 0.8)) -> list[float]:
    """valid で決めて凍結する区切り（5 分位）。"""
    return [float(v) for v in np.quantile(np.asarray(x, dtype=float), q)]


def apply_cuts(x: np.ndarray, cuts: list[float], fmt: str = "{:.3f}") -> np.ndarray:
    """凍結した区切りで帯に分ける（2025 年にも同じ区切りを使う）。帯の名前は区切りの値。"""
    edges = [-np.inf] + list(cuts) + [np.inf]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"Q{i + 1}(<{fmt.format(hi)})" if np.isinf(lo) else (f"Q{i + 1}(≥{fmt.format(lo)})" if np.isinf(hi) else f"Q{i + 1}({fmt.format(lo)}-{fmt.format(hi)})"))
    idx = np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(x, dtype=float), side="right")
    return np.asarray(labels, dtype=object)[idx]


def pop_band(p) -> str:
    if not np.isfinite(p):
        return "不明"
    p = int(p)
    return "1番人気" if p == 1 else "2番人気" if p == 2 else "3番人気" if p == 3 else "4-5番人気" if p <= 5 else "6番人気~"


def odds_band(o) -> str:
    if not np.isfinite(o):
        return "不明"
    return "~1.9倍" if o < 2 else "2.0-2.9倍" if o < 3 else "3.0-4.9倍" if o < 5 else "5.0-9.9倍" if o < 10 else "10倍~"


# ── レースごとの買い目と払戻 ──────────────────────────────────────────────


def race_bets(pred: pd.DataFrame, runners: pd.DataFrame, pay: dict, score: str = "p_win") -> pd.DataFrame:
    """予測（race_id・horse_id・score）→ レースごとの 1〜3 番手と、4 券種の払戻（ret_*）・的中（hit_*）。
    並びは score の大きい順（同じ値は馬番の小さい順）。runners: race_id・horse_id・horse_number・odds・popularity。
    4 券種の払戻がそろわないレース・3 頭未満のレースは入れない。"""
    d = pred[["race_id", "horse_id", score]].merge(runners, on=["race_id", "horse_id"], how="inner")
    d = d.sort_values(["race_id", score, "horse_number"], ascending=[True, False, True], kind="stable")
    rows = []
    for rid, g in d.groupby("race_id", sort=False):
        if len(g) < 3 or not all((rid, bt) in pay for bt in BETS):
            continue
        hn = g["horse_number"].to_numpy().astype(int)
        p = g[score].to_numpy(dtype=float)
        ps = p / p.sum() if p.sum() > 0 else p
        r = {"race_id": rid, "top1_hn": hn[0], "top1_pop": float(g["popularity"].iloc[0]), "top1_odds_v": float(g["odds"].iloc[0]),
             "p1": ps[0], "gap": ps[0] - ps[1], "flat": float(-(ps[ps > 0] * np.log(ps[ps > 0])).sum() / np.log(len(ps)))}
        r["ret_tansho"] = pay[(rid, "tansho")].get(str(hn[0]), 0)
        r["ret_fukusho"] = pay[(rid, "fukusho")].get(str(hn[0]), 0)
        for bt in ("umaren", "wide"):
            tbl = pay[(rid, bt)]
            r[f"ret_{bt}"] = tbl.get(_pair(hn[0], hn[1]), 0) + tbl.get(_pair(hn[0], hn[2]), 0)
        rows.append(r)
    t = pd.DataFrame(rows)
    for bt in BETS:
        t[f"hit_{bt}"] = (t[f"ret_{bt}"] > 0).astype(float)
    return t


def market_pred(runners: pd.DataFrame) -> pd.DataFrame:
    """市場の基準: 最終オッズの人気順（1 番人気が 1 番手）。人気が無ければオッズの小さい順。取消・除外は呼び出し側で外しておく。"""
    m = runners[["race_id", "horse_id", "popularity", "odds"]].copy()
    key = m["popularity"].where(m["popularity"].notna(), 100 + m["odds"].fillna(9999))
    m["score_mkt"] = -key.astype(float)
    return m[["race_id", "horse_id", "score_mkt"]]


# ── 分解・ブートストラップ・縮小 ─────────────────────────────────────────


def decompose(h1: float, a1: float, h0: float, a0: float, cost: float) -> tuple[float, float]:
    """回収率の差（h1·a1 − h0·a0）/c を、的中率の分と的中時の払戻の分に対称に分ける。和はちょうど差になる。"""
    return (h1 - h0) * (a1 + a0) / 2 / cost, (a1 - a0) * (h1 + h0) / 2 / cost


def poisson_weights(n: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """レース単位のブートストラップの重み（Poisson(1)）。全モデル・全セルで同じ重みを使い、差を対にする。"""
    return np.random.default_rng(seed).poisson(1.0, size=(n_boot, n)).astype(np.float32)


def eb_shrink(dev: np.ndarray, se: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """セルの偏差（セル − 全体）を 0 に寄せる係数 B = τ²/(τ²+se²)。τ² はモーメント法（ok のセルで推定）。"""
    d, s = dev[ok], se[ok]
    tau2 = max(0.0, float(np.mean(d ** 2 - s ** 2))) if ok.any() else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(se > 0, tau2 / (tau2 + se ** 2), 0.0)
    return np.nan_to_num(b)


def holm(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Holm の手順で有意かどうか（族全体の誤りの確率を alpha 以下に）。"""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    sig = np.zeros(m, dtype=bool)
    for k, i in enumerate(order):
        if p[i] <= alpha / (m - k):
            sig[i] = True
        else:
            break
    return sig


def cell_stats(t: dict[str, pd.DataFrame], labels: dict[str, np.ndarray], W: np.ndarray, period: str, dim: str) -> list[dict]:
    """1 つの軸について、各モデル・各水準・各券種の指標と区間。t[model] は共通レースに揃えて同じ順に並んだ表。"""
    from math import erf, sqrt

    out = []
    tot_w = W.sum(1)
    ref_cache: dict = {}
    for model, tb in t.items():
        lab = labels[model]
        levels = sorted(set(lab))
        M = np.stack([(lab == lv) for lv in levels], axis=1).astype(np.float32)        # N × K
        nb = W @ M                                                                       # 重みつきのレース数（boot × K）
        for bt in BETS:
            c = COST[bt]
            ret, hit = tb[f"ret_{bt}"].to_numpy(np.float32), tb[f"hit_{bt}"].to_numpy(np.float32)
            R, H = W @ (M * ret[:, None]), W @ (M * hit[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                roi_b = R / (nb * c)
                hr_b = H / nb
            roi0_b = (W @ ret) / (tot_w * c)
            hr0_b = (W @ hit) / tot_w
            dev_b, hdev_b = roi_b - roi0_b[:, None], hr_b - hr0_b[:, None]
            n = M.sum(0)
            h = (M * hit[:, None]).sum(0) / n
            rsum = (M * ret[:, None]).sum(0)
            hits = (M * hit[:, None]).sum(0)
            a = np.where(hits > 0, rsum / np.maximum(hits, 1), 0.0)
            roi = rsum / (n * c)
            h0, roi0 = float(hit.mean()), float(ret.sum() / (len(ret) * c))
            a0 = float(ret.sum() / max(hit.sum(), 1))
            dev = roi - roi0
            se = np.nanstd(dev_b, axis=0)
            okc = n >= 30
            bsh = eb_shrink(dev, se, okc)
            lo, hi = np.nanquantile(dev_b, 0.025, axis=0), np.nanquantile(dev_b, 0.975, axis=0)
            hlo, hhi = np.nanquantile(hdev_b, 0.025, axis=0), np.nanquantile(hdev_b, 0.975, axis=0)
            hse = np.nanstd(hdev_b, axis=0)
            if model == REF:
                ref_cache[bt] = (dict(zip(levels, range(len(levels)))), roi_b, hr_b, roi, h)
            for k, lv in enumerate(levels):
                hp, ap = decompose(h[k], a[k], h0, a0, c)
                z = dev[k] / se[k] if se[k] > 0 else 0.0
                hz = (h[k] - h0) / hse[k] if hse[k] > 0 else 0.0
                row = {"period": period, "dim": dim, "level": lv, "model": model, "bet": bt, "n": int(n[k]),
                       "hit_rate": float(h[k]), "avg_pay": float(a[k]), "roi": float(roi[k]), "roi_all": roi0, "hit_all": h0, "pay_all": a0,
                       "dev": float(dev[k]), "dev_lo": float(lo[k]), "dev_hi": float(hi[k]), "dev_hit_part": float(hp), "dev_pay_part": float(ap),
                       "shrink_b": float(bsh[k]), "dev_shrunk": float(bsh[k] * dev[k]), "dev_shrunk_lo": float(bsh[k] * lo[k]), "dev_shrunk_hi": float(bsh[k] * hi[k]),
                       "p_dev": float(2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))),
                       "hit_dev": float(h[k] - h0), "hit_dev_lo": float(hlo[k]), "hit_dev_hi": float(hhi[k]),
                       "p_hit": float(2 * (1 - 0.5 * (1 + erf(abs(hz) / sqrt(2)))))}
                out.append(row)
            # v6e_prod との差。レースの性質の軸では同じレースどうしの対の差。1 番手の人気・オッズの軸では、
            # 「そのモデルの 1 番手がその帯のレース」と「v6e_prod の 1 番手がその帯のレース」という別の集合どうしの差（重みは共通）
            if model != REF and bt in ref_cache:
                idx, rroi_b, rhr_b, rroi, rh = ref_cache[bt]
                for k, lv in enumerate(levels):
                    if lv not in idx:
                        continue
                    j = idx[lv]
                    d_b = roi_b[:, k] - rroi_b[:, j]
                    dh_b = hr_b[:, k] - rhr_b[:, j]
                    row = out[-len(levels) + k]
                    row.update({"vs_ref": float(roi[k] - rroi[j]), "vs_ref_lo": float(np.nanquantile(d_b, 0.025)), "vs_ref_hi": float(np.nanquantile(d_b, 0.975)),
                                "vs_ref_hit": float(h[k] - rh[j]), "vs_ref_hit_lo": float(np.nanquantile(dh_b, 0.025)), "vs_ref_hit_hi": float(np.nanquantile(dh_b, 0.975))})
    return out


# ── 読み込み ─────────────────────────────────────────────────────────────


def load_runners(years) -> pd.DataFrame:
    from src.config.data_paths import TABLES_DIR

    cols = ["race_id", "date", "venue", "surface", "distance", "grade", "race_class", "track_condition", "horse_number", "horse_id",
            "sex_age", "odds", "popularity"]
    fs = [pd.read_parquet(Path(TABLES_DIR) / str(y) / "race_result_flat.parquet", columns=cols) for y in years]
    f = pd.concat(fs, ignore_index=True)
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    f["horse_number"] = pd.to_numeric(f["horse_number"], errors="coerce")
    f["odds"] = pd.to_numeric(f["odds"], errors="coerce")
    f["popularity"] = pd.to_numeric(f["popularity"], errors="coerce")
    f.loc[f["odds"] <= 0, "odds"] = np.nan
    f.loc[f["popularity"] <= 0, "popularity"] = np.nan
    f["scratched"] = f["odds"].isna() & f["popularity"].isna()       # 取消・除外は人気 0・オッズ 0 で入っている（両方とも無い馬を取消とみなす）
    return f[f["horse_number"].notna()]


def load_pred(path: Path) -> pd.DataFrame:
    o = pd.read_parquet(path, columns=["race_id", "horse_id", "p_win"])
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    return o


def build_period(years, which: int, pay: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """which: 0=valid の予測、1=2025 年の予測。共通レースに揃えた {model: レースの表} と、レースの条件を返す。"""
    flat = load_runners(years)
    runners = flat.loc[~flat["scratched"], ["race_id", "horse_id", "horse_number", "odds", "popularity"]]   # 取消の馬は予測からも外す（買えない）
    tabs = {m: race_bets(load_pred(paths[which]), runners, pay) for m, paths in MODELS.items()}
    tabs[MARKET] = race_bets(market_pred(runners), runners, pay, score="score_mkt")
    common = sorted(set.intersection(*(set(t["race_id"]) for t in tabs.values())))
    tabs = {m: t.set_index("race_id").loc[common].reset_index() for m, t in tabs.items()}
    cond = race_conditions(flat[flat["race_id"].isin(common)]).set_index("race_id").loc[common].reset_index()
    return tabs, cond


def labels_for(tabs: dict[str, pd.DataFrame], cond: pd.DataFrame, cuts: dict) -> dict[str, dict[str, np.ndarray]]:
    """軸 → {model: 水準の配列}。"""
    out: dict[str, dict[str, np.ndarray]] = {}
    src = tabs[CONF_SRC]
    conf = {"conf_p1": apply_cuts(src["p1"].to_numpy(), cuts["conf_p1"]), "conf_gap": apply_cuts(src["gap"].to_numpy(), cuts["conf_gap"]),
            "conf_flat": apply_cuts(src["flat"].to_numpy(), cuts["conf_flat"])}
    for d in RACE_DIMS:
        lab = conf[d] if d in conf else cond[d].astype(str).to_numpy()
        out[d] = {m: lab for m in tabs}
    out["top1_pop"] = {m: np.array([pop_band(x) for x in t["top1_pop"]], dtype=object) for m, t in tabs.items()}
    out["top1_odds"] = {m: np.array([odds_band(x) for x in t["top1_odds_v"]], dtype=object) for m, t in tabs.items()}
    return out


def overall(tabs: dict[str, pd.DataFrame]) -> dict:
    res = {}
    for m, t in tabs.items():
        r = {}
        for bt in BETS:
            hit, ret = t[f"hit_{bt}"], t[f"ret_{bt}"]
            r[bt] = {"hit_rate": float(hit.mean()), "avg_pay": float(ret.sum() / max(hit.sum(), 1)), "roi": float(ret.sum() / (len(t) * COST[bt]))}
        res[m] = r
    return res


def run(n_boot: int, out: Path) -> dict:
    pay = load_payoff()
    tv, cv = build_period(VALID_YEARS, 0, pay)
    tt, ct = build_period(TEST_YEARS, 1, pay)
    cuts = {k: quantile_cuts(tv[CONF_SRC][c].to_numpy()) for k, c in (("conf_p1", "p1"), ("conf_gap", "gap"), ("conf_flat", "flat"))}   # valid で凍結
    rows = []
    for period, tabs, cond, seed in (("valid", tv, cv, 0), ("test2025", tt, ct, 1)):
        W = poisson_weights(len(cond), n_boot, seed)
        labs = labels_for(tabs, cond, cuts)
        for dim, lab in labs.items():
            rows += cell_stats(tabs, lab, W, period, dim)
    cells = pd.DataFrame(rows)
    v = cells["period"] == "valid"
    fam = v & (cells["model"] != MARKET) & (cells["n"] >= 30)  # 多重比較の族: valid の 30 レース以上のセル × 券種 × モデル（市場は除く。的中 0 の小さいセルで z が壊れるのを避ける）
    cells["holm_roi"] = False
    cells["holm_hit"] = False
    cells.loc[fam, "holm_roi"] = holm(cells.loc[fam, "p_dev"].to_numpy())
    cells.loc[fam, "holm_hit"] = holm(cells.loc[fam, "p_hit"].to_numpy())
    # 市場の同じセルの偏差（レースの性質の軸だけ。モデルの 1 番手で決まる軸は市場と同じセルにならない）
    mk = cells[cells["model"] == MARKET].set_index(["period", "dim", "level", "bet"])[["roi", "dev", "hit_rate", "hit_dev"]]
    mk.columns = ["mkt_roi", "mkt_dev", "mkt_hit", "mkt_hit_dev"]
    cells = cells.join(mk, on=["period", "dim", "level", "bet"])
    cells.loc[cells["dim"].isin(MODEL_DIMS), ["mkt_roi", "mkt_dev", "mkt_hit", "mkt_hit_dev"]] = np.nan
    cells["dev_ex_mkt"] = cells["dev"] - cells["mkt_dev"]     # 市場の同じ条件の偏差を引いた差（モデル固有の分の目安。区間は付けない）
    # 2025 年の同じセル（向きの再現）
    te = cells[cells["period"] == "test2025"].set_index(["dim", "level", "model", "bet"])[["n", "roi", "dev", "hit_rate", "hit_dev", "dev_lo", "dev_hi"]]
    te.columns = [f"t25_{c}" for c in te.columns]
    out.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out / "cells.parquet", index=False)
    vc = cells[v].join(te, on=["dim", "level", "model", "bet"])
    summ = {"n_races": {"valid": int(len(cv)), "test2025": int(len(ct))}, "cuts": cuts, "family_size": int(fam.sum()), "n_boot": n_boot,
            "holm_roi": int(cells["holm_roi"].sum()), "holm_hit": int(cells["holm_hit"].sum()),
            "overall": {"valid": overall(tv), "test2025": overall(tt)}, "replication": replication(vc)}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    vc.to_parquet(out / "cells_valid_with_2025.parquet", index=False)
    (out / "report.md").write_text(render(vc, summ), encoding="utf-8")
    return summ


def replication(vc: pd.DataFrame) -> dict:
    """valid の偏差と 2025 年の偏差の順位相関（モデル × 券種、valid・2025 年とも 100 レース以上のセル）と、
    valid で区間が 0 をまたがないセルの向きが 2025 年でも同じだった割合。"""
    res = {}
    for (m, bt), g in vc.groupby(["model", "bet"]):
        g = g[(g["n"] >= 100) & (g["t25_n"] >= 100)]
        sig = g[(g["dev_lo"] > 0) | (g["dev_hi"] < 0)]
        res[f"{m}|{bt}"] = {"n_cells": int(len(g)), "spearman": float(g["dev"].rank().corr(g["t25_dev"].rank())) if len(g) > 2 else None,
                            "n_sig": int(len(sig)), "same_sign": int((np.sign(sig["dev"]) == np.sign(sig["t25_dev"])).sum())}
    return res


def _fmt_row(r) -> str:
    mk = "" if pd.isna(r.get("mkt_dev")) else f"{r['mkt_roi']:.3f}（{r['mkt_dev']:+.3f}）／{r['dev_ex_mkt']:+.3f}"
    vs = "" if pd.isna(r.get("vs_ref")) else f"{r['vs_ref']:+.3f}（{r['vs_ref_lo']:+.3f}〜{r['vs_ref_hi']:+.3f}）"
    t25 = "" if pd.isna(r.get("t25_dev")) else f"{r['t25_dev']:+.3f}（n={int(r['t25_n'])}）"
    flag = ("※" if r["holm_roi"] else "") + ("†" if r["holm_hit"] else "")
    return (f"| {DIM_LABEL.get(r['dim'], r['dim'])}: {r['level']} | {BET_LABEL[r['bet']]} | {r['n']} | {r['hit_rate']:.3f}（{r['hit_dev']:+.3f}） | "
            f"{r['avg_pay']:.0f} | {r['roi']:.3f} | {r['dev']:+.3f}（{r['dev_lo']:+.3f}〜{r['dev_hi']:+.3f}） | {r['dev_shrunk']:+.3f} | "
            f"{r['dev_hit_part']:+.3f} / {r['dev_pay_part']:+.3f} | {mk} | {vs} | {t25} | {flag} |")


HEAD = ("| 条件 | 券種 | 件数 | 的中率（全体との差） | 的中時の平均払戻 | 回収率 | 全体との差（95%区間） | 縮小後の差 | 差の内訳 的中率 / 払戻 | 市場の同じ条件の回収率（全体との差）／市場を引いた差 | v6e_prod との差（区間） | 2025 年の差 | 印 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def top_cells(vc: pd.DataFrame, model: str, weak: bool, k: int = 12) -> pd.DataFrame:
    g = vc[(vc["model"] == model) & (vc["n"] >= MIN_N_LIST) & (vc["dim"] != "surf_dist")]
    g = g[g["dev_hi"] < 0] if weak else g[g["dev_lo"] > 0]
    return g.sort_values("dev_shrunk", ascending=weak).head(k)


def render(vc: pd.DataFrame, summ: dict) -> str:
    L = [f"# 条件別の弱み・強み（valid {summ['n_races']['valid']} レース、2025 年 {summ['n_races']['test2025']} レース）", "",
         f"族の大きさ {summ['family_size']}（valid のセル × 券種 × モデル）。Holm で残ったセル: 回収率 {summ['holm_roi']}、的中率 {summ['holm_hit']}。",
         "印: ※ = 回収率の差が Holm で有意、† = 的中率の差が Holm で有意。上位は 95% 区間が 0 をまたがないセルを、縮小後の差の順に。", ""]
    ov = summ["overall"]["valid"]
    L += ["## 全体（valid）", "", "| モデル | " + " | ".join(f"{BET_LABEL[b]} 的中率・払戻・回収率" for b in BETS) + " |", "|---|" + "---|" * len(BETS)]
    for m, r in ov.items():
        L.append(f"| {m} | " + " | ".join(f"{r[b]['hit_rate']:.3f}・{r[b]['avg_pay']:.0f}・{r[b]['roi']:.3f}" for b in BETS) + " |")
    L.append("")
    for m in list(MODELS) + [MARKET]:
        for weak in (True, False):
            g = top_cells(vc, m, weak)
            L += [f"## {m} の{'弱み' if weak else '強み'}（回収率が全体より{'低い' if weak else '高い'}条件）", "", HEAD]
            L += [_fmt_row(r) for _, r in g.iterrows()] or ["| （区間が 0 をまたがないセルなし） |" + " |" * 12]
            L.append("")
    L += ["## 2025 年での再現（valid の偏差と 2025 年の偏差、100 レース以上のセル）", "", "| モデル・券種 | セル数 | 順位相関 | valid で有意 | 2025 年も同じ向き |", "|---|---|---|---|---|"]
    for k, r in summ["replication"].items():
        sp = "－" if r["spearman"] is None else f"{r['spearman']:.2f}"
        L.append(f"| {k} | {r['n_cells']} | {sp} | {r['n_sig']} | {r['same_sign']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(B16 / "tp_rank/cond"))
    a = ap.parse_args()
    s = run(a.boot, Path(a.out))
    print(json.dumps({k: s[k] for k in ("n_races", "family_size", "holm_roi", "holm_hit")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
