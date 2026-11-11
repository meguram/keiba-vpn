"""ペルソナ 1（機械的に買う人）: 予測値で回収率 100% 超の戦略があるかを、先に決めた手順で検証する。

python -m src.scripts.data.persona1_search [--frame PATH] [--first-test 2023] [--last-test 2025] [--out DIR]

入力の表（1 行 = 1 頭）に必要な列:
  race_id, date, year, horse_number, win_odds, place_odds_min, place_odds_max（確定オッズ）, p_win, p_top3（モデルの確率）,
  ret_win, ret_place（100 円あたりの払戻 / 100。外れは 0）。市場の確率 q_win は win_odds から作る。
既定は tp-36 の frame_all（V4 の予測、2022〜2026 年）。v6 系の予測ができたら --frame で差し替える（同じ列にそろえる）。

手順（事前に固定。結果を見て変えない）:
  1. 候補の格子: 券種{単勝, 複勝} × 市場とのブレンド{モデルのみ, Benter 混合} × 期待値の下限 θ{1.0, 1.05, 1.1, 1.2, 1.3, 1.5}
     × オッズの上限{なし, 20 倍以下, 10 倍以下}。格子の大きさ K を多重比較に数える。
  2. ブレンド p ∝ p_model^a · q_market^b の a, b は『テスト年より前の年』の勝ち馬の対数尤度で選ぶ（回収率では選ばない）。
  3. テスト年 Y ごとに、Y より前の年だけで各候補の回収率（レース単位クラスタの下側 90% 信頼限界）を出し、
     下限が最大で 100% を超える候補を 1 つ選んで Y に適用する（超える候補が無ければ買わない）。
  4. テスト年を通した回収率と 95% 区間（レース単位のブートストラップ）、買ったレース数を出す。
  5. White の Reality Check: テスト年の全候補のうち、標準化した最良の平均超過収益が偶然で出る確率（開催日単位のブロック・ブートストラップ）。
2026 年はテストセット②。--last-test を 2025 にしておき、最後に 1 回だけ 2026 を足す。
GCS は読まない（手元の表だけ）。費用 0 円。

留保（結果の JSON にも書く）:
  ・期待値・オッズ上限・市場の確率は『確定オッズ』で計算している。購入時点では分からない値で、締切直前の資金は情報を持つため、
    回収率は実際より楽観側に出る。自分の賭けでオッズが動く分も入っていない。100% を超えても『確定オッズを使った上限の値』と読む。
  ・p_win・p_top3 は、テスト年を学習に含まないモデルの予測（leave-one-year-out または expanding の OOF）であることが前提。
    表に train_end_year 列があれば、テスト年より前であることを確かめる。無い場合は呼び出し側の責任。
  ・複勝は出走 8 頭以上のレースだけ（7 頭以下は 2 着までしか払われない）。選ぶときの期待値は払戻の下限（place_odds_min）で見て、楽観を避ける。
  ・候補の選択は片側 99% の下限（格子 K=72 の選択に伴う偶然の当たりを減らすため）。多重比較の判定は Reality Check（開催日単位のブロック・標準化）。
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FRAME = ROOT / "data/local/modeling/betting/tp36/frame_all.parquet"
OUT = ROOT / "data/local/research/persona1"
THETAS = (1.0, 1.05, 1.1, 1.2, 1.3, 1.5)
ODDS_CAPS = (None, 20.0, 10.0)
BLENDS = ("model", "benter")
BETS = ("win", "place")
AB_GRID = [(a, b) for a in (0.0, 0.2, 0.3, 0.4, 0.5, 1.0) for b in (0.0, 0.7, 0.8, 1.0)]
MIN_N = 300
BOOT = 1000
Z_SELECT = 2.3263          # 片側 99%（候補選びの下限。格子の選択による偶然の当たりを減らす）
MIN_FIELD_PLACE = 8        # 複勝が 3 着まで払われる出走頭数


def softmax_by_race(logit: np.ndarray, rid: np.ndarray) -> np.ndarray:
    s = pd.Series(logit)
    m = s.groupby(rid).transform("max")
    e = np.exp(s - m)
    return (e / e.groupby(rid).transform("sum")).to_numpy()


def fit_ab(d: pd.DataFrame) -> tuple[float, float]:
    """勝ち馬の対数尤度が最大の (a, b)。回収率は見ない。"""
    rid = d["race_id"].to_numpy()
    lp, lq = np.log(np.clip(d["p_win"], 1e-6, 1)), np.log(np.clip(d["q_win"], 1e-6, 1))
    best, ab = -1e18, (1.0, 0.0)
    for a, b in AB_GRID:
        if a == 0 and b == 0:
            continue
        p = softmax_by_race(a * lp.to_numpy() + b * lq.to_numpy(), rid)
        ll = float(np.log(np.clip(p[d["win"].to_numpy() == 1], 1e-9, 1)).sum())
        if ll > best:
            best, ab = ll, (a, b)
    return ab


def add_blend(d: pd.DataFrame, ab: tuple[float, float]) -> pd.DataFrame:
    d = d.copy()
    a, b = ab
    d["pw_b"] = softmax_by_race(a * np.log(np.clip(d["p_win"], 1e-6, 1)).to_numpy() + b * np.log(np.clip(d["q_win"], 1e-6, 1)).to_numpy(), d["race_id"].to_numpy())
    ratio = (d["pw_b"] / d["p_win"].clip(lower=1e-6)).clip(0.2, 5.0)
    p3 = d["p_top3"] * ratio
    n_run = d.groupby("race_id")["race_id"].transform("size")
    p3 = p3 * (np.minimum(3, n_run) / p3.groupby(d["race_id"]).transform("sum"))      # レース内で合計 3（3 着以内の頭数）にそろえる
    d["p3_b"] = np.clip(p3, 1e-6, 0.99)
    return d


def candidates() -> list[tuple[str, str, float, float | None]]:
    return list(itertools.product(BETS, BLENDS, THETAS, ODDS_CAPS))


def select(d: pd.DataFrame, cand: tuple[str, str, float, float | None]) -> np.ndarray:
    bet, blend, theta, cap = cand
    if bet == "win":
        p = d["pw_b"] if blend == "benter" else d["p_win"]
        ev = p * d["win_odds"]
        ok = (ev >= theta) & ((d["win_odds"] <= cap) if cap else True)
    else:
        p = d["p3_b"] if blend == "benter" else d["p_top3"]
        ev = p * d["place_odds_min"]                 # 払戻の下限で見る（中間値だと人気馬の払戻を過大に見る）
        ok = (ev >= theta) & (d["place_odds_min"] > 0) & (d["n_run"] >= MIN_FIELD_PLACE) & ((d["win_odds"] <= cap) if cap else True)
    return ok.to_numpy()


def race_profit(d: pd.DataFrame, cand) -> pd.DataFrame:
    """レースごとの (損益, 賭け数)。1 点 100 円。"""
    ok = select(d, cand)
    ret = d["ret_win"] if cand[0] == "win" else d["ret_place"]
    x = pd.DataFrame({"race_id": d["race_id"].to_numpy(), "pl": np.where(ok, ret.to_numpy() - 1.0, 0.0), "n": ok.astype(int)})
    return x.groupby("race_id", sort=False).sum()


def roi_lower(pr: pd.DataFrame, z: float = Z_SELECT) -> tuple[float, float, int]:
    n = int(pr["n"].sum())
    if n < MIN_N:
        return float("nan"), 0.0, n
    # ROI = 1 + sum(pl)/sum(n)。レース単位の比の分散（デルタ法）
    r = pr["pl"].sum() / n
    resid = pr["pl"] - r * pr["n"]
    se = float(np.sqrt((resid**2).sum()) / n)
    return 1 + r, 1 + r - z * se, n


def boot_ci(pr: pd.DataFrame, rng: np.random.Generator, reps: int = BOOT) -> tuple[float, float, float]:
    pl, n = pr["pl"].to_numpy(), pr["n"].to_numpy()
    if n.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(pl), size=(reps, len(pl)))
    rois = 1 + pl[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1)
    return 1 + pl.sum() / n.sum(), float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


PRE_ODDS = ROOT / "data/local/research/persona1/pre_odds.parquet"
BUCKETS = [0, 3, 6, 12, 25, 50, 1e9]


def decision_odds_noise(d: pd.DataFrame, phase: str, rng: np.random.Generator, pre_path: Path = PRE_ODDS, final_path: Path | None = None) -> pd.DataFrame:
    """購入時点（phase: t60/t15/t10/t05/t02）のオッズの不確かさを再現する。判断に使うオッズ（期待値・オッズ上限・市場の確率）だけを
    『確定オッズ × exp(実測のずれ)』に置き換え、払戻（ret_win・ret_place）は確定のままにする（払戻は確定オッズで決まるため）。
    ずれは pre_odds_table.py の実測 log(購入時点のオッズ / 確定オッズ) を、確定オッズの帯ごとに、単勝と複勝の組（同じ馬の実測）で復元抽出する。
    レース内の 1/オッズ の合計（控除率）は元の確定オッズのレースの値に戻す（馬ごとに独立に引くと実際には起きない控除率になるため）。
    留保: 再現したずれは結果と独立。実際のずれは終盤の資金（情報）と相関する。偏りの向きは一つに決まらない
    （情報が残る分は楽観側、余分なずれで買い目が誤って増える分は悲観側）ので、『上限』とは読まず、実測が貯まったら実際の t02/t05 のオッズで置き換える。
    実測は 2026-10-17 以降の少数のレース（t05 は約 130 レース）。標本が 30 未満の帯は、全帯をまとめた分布で代用する（その数を結果に出す）。"""
    p = pd.read_parquet(pre_path)
    p = p[(p["phase"] == phase) & (p["win_odds"] >= 1) & (p["place_odds_min"] > 0)]
    fin = pd.read_parquet(final_path or ROOT / "data/page_reference/tables/2026/race_odds_flat.parquet",
                          columns=["race_id", "horse_number", "win_odds", "place_odds_min"]).rename(columns={"win_odds": "final", "place_odds_min": "final_pl"})
    fin["race_id"] = fin["race_id"].astype(str)
    p = p.assign(race_id=p["race_id"].astype(str)).merge(fin, on=["race_id", "horse_number"])
    p = p[(p["final"] >= 1) & (p["final_pl"] > 0)]
    p["lr"] = np.log(p["win_odds"] / p["final"])
    p["lrp"] = np.log(p["place_odds_min"] / p["final_pl"])
    p["b"] = pd.cut(p["final"], BUCKETS, labels=False)
    pools = {int(b): g[["lr", "lrp"]].to_numpy() for b, g in p.groupby("b") if len(g) >= 30}
    allv = p[["lr", "lrp"]].to_numpy()
    d = d.copy()
    b = pd.cut(d["win_odds"], BUCKETS, labels=False).to_numpy()
    lr = np.zeros(len(d))
    lrp = np.zeros(len(d))
    substituted = 0
    for k in np.unique(b[~pd.isna(b)]):
        m = b == k
        pool = pools.get(int(k))
        if pool is None:
            pool, substituted = allv, substituted + int(m.sum())
        draw = pool[rng.integers(0, len(pool), size=int(m.sum()))]
        lr[m], lrp[m] = draw[:, 0], draw[:, 1]
    inv_old = 1.0 / d["win_odds"]
    inv_new = inv_old / np.exp(lr)
    # レース内の 1/オッズ の合計を元の値に戻す（控除率を保つ）
    inv_new = inv_new * (inv_old.groupby(d["race_id"]).transform("sum") / inv_new.groupby(d["race_id"]).transform("sum"))
    d["win_odds"] = np.maximum(1.0 / inv_new, 1.0)
    d["place_odds_min"] = np.where(d["place_odds_min"] > 0, np.maximum(d["place_odds_min"] * np.exp(lrp), 1.0), 0)
    d["place_odds_max"] = np.where(d["place_odds_max"] > 0, np.maximum(np.maximum(d["place_odds_max"] * np.exp(lrp), d["place_odds_min"]), 1.0), 0)
    q = 1.0 / d["win_odds"]
    d["q_win"] = q / q.groupby(d["race_id"]).transform("sum")
    d.attrs["noise_substituted_rows"] = substituted
    return d


def reality_check(series: dict[int, list[pd.DataFrame]], day_of: pd.Series, rng: np.random.Generator) -> dict | None:
    """Reality Check: テスト年の全候補を『買い続けた』ときの標準化した平均超過収益（比の t 値）の最大値が、偶然で出るか。
    開催日単位のブロックで引き（同じ日のレースは馬場・展開を共有して相関する）、各候補を自分の平均で中心化して帰無を作る。
    series: 候補の番号 → テスト年ごとの race_profit（index = race_id, 列 pl・n）の一覧。day_of: race_id → 開催日。"""
    mats = [pd.concat(series[k]) for k in sorted(series)]
    if not mats:
        return None
    races = sorted(set().union(*[set(m.index) for m in mats]))
    pos = {r: i for i, r in enumerate(races)}
    PL = np.zeros((len(races), len(mats)))
    NN = np.zeros_like(PL)
    for j, m in enumerate(mats):
        ii = [pos[r] for r in m.index]
        PL[ii, j] = m["pl"].to_numpy()
        NN[ii, j] = m["n"].to_numpy()
    keep = NN.sum(axis=0) >= MIN_N
    PL, NN = PL[:, keep], NN[:, keep]
    days = pd.Series([day_of.get(r, r) for r in races]).astype("category").cat.codes.to_numpy()
    nd = int(days.max()) + 1
    PLd, NNd = np.zeros((nd, PL.shape[1])), np.zeros((nd, PL.shape[1]))
    np.add.at(PLd, days, PL)
    np.add.at(NNd, days, NN)
    mean = PLd.sum(axis=0) / NNd.sum(axis=0)
    resid = PLd - mean * NNd                                           # 日ごとの残差（比のデルタ法）

    def tmax(num: np.ndarray, sq: np.ndarray) -> float:
        """候補ごとの t 値（比のデルタ法）の最大。標本に買いが 1 件も入らない候補（分母 0）は t=0 とする（NaN にすると最大値の比較が常に偽になる）。"""
        den = np.sqrt(sq)
        t = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
        return float(t.max())

    t_obs = tmax(PLd.sum(axis=0), (resid**2).sum(axis=0))
    cnt = 0
    for _ in range(BOOT):
        idx = rng.integers(0, nd, nd)
        rb = resid[idx]
        if tmax(rb.sum(axis=0), (rb**2).sum(axis=0)) >= t_obs:
            cnt += 1
    if nd < 30:      # 開催日のブロックが少なすぎると、ブートストラップが退化して p が信用できない
        return {"candidates_used": int(keep.sum()), "blocks": nd, "p": None, "note": "開催日が 30 日未満のため判定しない"}
    else:
        return {"candidates_used": int(keep.sum()), "blocks": nd, "best_mean_excess": round(float(mean.max()), 4), "t_max": round(t_obs, 3), "p": round((cnt + 1) / (BOOT + 1), 4)}


def run(frame_path: Path, first_test: int, last_test: int, out: Path, noise_phase: str | None = None, noise_seed: int = 7) -> dict:
    d = pd.read_parquet(frame_path)
    d = d[(d["win_odds"] > 0) & d["ret_win"].notna() & d["ret_place"].notna()].copy()      # 複勝オッズが無い馬は残す（市場の確率の正規化を壊さない）
    d["place_odds_min"] = d["place_odds_min"].fillna(0)
    d["place_odds_max"] = d["place_odds_max"].fillna(0)
    d["n_run"] = d.groupby("race_id")["race_id"].transform("size")
    d["year"] = pd.to_datetime(d["date"]).dt.year
    if "train_end_year" in d.columns:
        assert (d["train_end_year"] < d["year"]).all(), "予測にテスト年以降のデータで学習したものが混ざっている（先読み）"
    if "q_win" not in d.columns:
        q = 1.0 / d["win_odds"]
        d["q_win"] = q / q.groupby(d["race_id"]).transform("sum")
    for c in ("ret_win", "ret_place"):      # 100 円あたりの円（例 450）で入っている表は、倍率（4.5）に直す
        pos = d[c][d[c] > 0]
        if len(pos) and pos.median() > 20:
            d[c] = d[c] / 100.0
    d["win"] = (d["ret_win"] > 0).astype(int)
    rng = np.random.default_rng(20261109)
    if noise_phase:
        d = decision_odds_noise(d, noise_phase, np.random.default_rng(noise_seed))
    cands = candidates()
    picks, oos_parts, grid_rows = [], [], []
    rc_series: dict[int, list[pd.DataFrame]] = {}
    for Y in range(first_test, last_test + 1):
        train, test = d[d["year"] < Y], d[d["year"] == Y]
        if train.empty or test.empty:
            continue
        ab = fit_ab(train)
        train, test = add_blend(train, ab), add_blend(test, ab)
        best, best_lb, best_row = None, 1.0, None
        for k, c in enumerate(cands):
            roi, lb, n = roi_lower(race_profit(train, c))
            tp = race_profit(test, c)
            grid_rows.append({"test_year": Y, "cand": str(c), "train_roi": (None if roi != roi else round(roi, 4)), "train_lb99": round(lb, 4), "train_n": n, "test_roi": round(float(tp["pl"].sum() / max(tp["n"].sum(), 1) + 1), 4), "test_n": int(tp["n"].sum())})
            rc_series.setdefault(k, []).append(tp)
            if roi == roi and lb > best_lb:
                best, best_lb, best_row = c, lb, (roi, lb, n)
        if best is None:
            picks.append({"test_year": Y, "ab": ab, "pick": None, "note": "前の年の下限が 100% を超える候補が無い → 買わない"})
            continue
        tp = race_profit(test, best)
        oos_parts.append(tp)
        picks.append({"test_year": Y, "ab": ab, "pick": str(best), "train_roi": round(best_row[0], 4), "train_lb99": round(best_row[1], 4), "train_n": best_row[2],
                      "test_roi": round(float(1 + tp["pl"].sum() / max(tp["n"].sum(), 1)), 4), "test_n": int(tp["n"].sum())})
    result: dict = {"frame": str(frame_path), "K": len(cands), "decision_odds": noise_phase or "final（確定オッズ。先読みで楽観）", "noise_seed": noise_seed, "picks": picks}
    if oos_parts:
        allp = pd.concat(oos_parts)
        roi, lo, hi = boot_ci(allp, rng)
        result["oos"] = {"roi": round(roi, 4), "ci95": [round(lo, 4), round(hi, 4)], "bets": int(allp["n"].sum()), "races_bought": int((allp["n"] > 0).sum())}
    else:
        result["oos"] = {"roi": None, "note": "どの年も買わなかった"}
    day_of = d.drop_duplicates("race_id").set_index("race_id")["date"].astype(str)
    result["reality_check"] = reality_check(rc_series, day_of, rng)
    result["caveat"] = "確定オッズで期待値・オッズ上限・市場の確率を作っているため、回収率は実際より楽観側（上限の値）。自分の賭けでオッズが動く分も入っていない。"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"search_{first_test}_{last_test}_s{noise_seed}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(grid_rows).to_csv(out / f"grid_{first_test}_{last_test}_s{noise_seed}.csv", index=False)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=str(FRAME))
    ap.add_argument("--first-test", type=int, default=2023)
    ap.add_argument("--last-test", type=int, default=2025)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--noise-seeds", type=int, default=5, help="--decision-odds のとき、乱数（ずれの抽出）を変えて繰り返す回数。選ばれる候補と回収率の幅を出す")
    ap.add_argument("--decision-odds", choices=["t60", "t15", "t10", "t05", "t02"], default=None, help="判断に使うオッズを、購入時点のずれを再現したものにする（払戻は確定のまま）")
    a = ap.parse_args()
    seeds = range(7, 7 + (a.noise_seeds if a.decision_odds else 1))
    res = [run(Path(a.frame), a.first_test, a.last_test, Path(a.out), a.decision_odds, sd) for sd in seeds]
    if len(res) == 1:
        print(json.dumps(res[0], ensure_ascii=False, indent=1, default=str))
        return
    summ = [{"seed": r["noise_seed"], "oos": r["oos"], "rc_p": (r.get("reality_check") or {}).get("p"), "picks": [(x["test_year"], x.get("pick")) for x in r["picks"]]} for r in res]
    print(json.dumps({"decision_odds": a.decision_odds, "runs": summ, "caveat": res[0]["caveat"]}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
