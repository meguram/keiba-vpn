"""事前登録した候補 C1・C2 を、登録したとおりに判定する（docs/html/design/persona_status.html の 3.2）。

python -m src.scripts.data.persona1_confirm --frame-prior PATH --frame-test PATH --years 2025 [2026] [--out DIR]

  C1 = 単勝・Benter 混合（a, b はテスト年より前の年の勝ち馬の対数尤度で選ぶ）の期待値 1.3 以上・オッズ上限なし・1 頭 100 円。
  C2 = ワイド・モデルの確率のみ・期待値 1.3 以上・的中確率 8% 以上・1 レース最大 2 点・100 円ずつ（期待値 = p × 払戻下限オッズ）。
入力の表（1 頭 1 行）: race_id, date, horse_number, horse_id, win_odds, place_odds_min, place_odds_max, p_win, p_top2, p_top3, ret_win, ret_place。
  --frame-prior: a, b を選ぶための、テスト年より前の年の予測（OOF）。--frame-test: 判定する年の予測（学習がその年より前のモデルによるもの）。
判定（段階）:
  強い合格 = 対象の年をまとめた回収率の 95% 区間（レース単位のブートストラップ）の下限が 100% 超、かつ上位 3 的中を除いた回収率が 100% 以上、買った点数 300 点以上
             （C1 は上位 3 除く条件なし。初版の条件のとおり）。
  弱い合格 = まとめた回収率が 100% 超で P(回収率 > 100%) が 90% 以上、買った点数 300 点以上。『確証』とは呼ばない。前向き追跡へ。
  不合格   = 上のどちらでもない、または個別の年の回収率が 80% 未満。
条件（θ・点数・確率の下限・オッズ上限）は変えない。変えたい場合は新しい候補として別に事前登録する。
GCS は読まない。費用 0 円。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import persona1_exotic as X
from src.scripts.data import persona1_search as P

OUT = P.OUT
C1 = ("win", "benter", 1.3, None)
C2 = {"bet": "wide", "blend": "model", "theta": 1.3, "kmax": 2, "pmin": 0.08}
MIN_BETS = 300
YEAR_FLOOR = 0.8


def verdict(per_year: dict[int, float], roi: float, lo: float, p_gt1: float, n: int, ex3: float | None, need_ex3: bool) -> str:
    if n < MIN_BETS:
        return f"判定不能（買った点数 {n} が {MIN_BETS} 未満）"
    if any(v < YEAR_FLOOR for v in per_year.values()):
        return "不合格（個別の年の回収率が 80% 未満）"
    strong = lo > 1.0 and (not need_ex3 or (ex3 is not None and ex3 >= 1.0))
    if strong:
        return "強い合格（採用の候補。それでも購入時点のオッズの前向き追跡が貯まるまで『確証』とは書かない）"
    if roi > 1.0 and p_gt1 >= 0.90:
        return "弱い合格（前向き追跡へ。『確証』とは呼ばない）"
    return "不合格"


def pooled(parts: list[pd.DataFrame], rng: np.random.Generator) -> dict:
    allp = pd.concat(parts)
    roi, lo, hi = P.boot_ci(allp, rng, reps=P.BOOT)
    pl, n = allp["pl"].to_numpy(), allp["n"].to_numpy()
    idx = rng.integers(0, len(pl), size=(P.BOOT, len(pl)))
    rois = 1 + pl[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1)
    return {"roi": roi, "lo": lo, "hi": hi, "p_gt1": float((rois > 1).mean()), "bets": int(n.sum())}


def run_c1(prior: pd.DataFrame, test: pd.DataFrame, years: list[int], rng: np.random.Generator) -> dict:
    parts, per_year = [], {}
    for Y in years:
        pr = prior[prior["year"] < Y]
        ab = P.fit_ab(pr)
        te = P.add_blend(test[test["year"] == Y], ab)
        tp = P.race_profit(te, C1)
        parts.append(tp)
        per_year[Y] = float(1 + tp["pl"].sum() / max(tp["n"].sum(), 1))
    s = pooled(parts, rng)
    return {"candidate": "C1", "spec": str(C1), "per_year": per_year, **s, "ex_top3": None,
            "verdict": verdict(per_year, s["roi"], s["lo"], s["p_gt1"], s["bets"], None, False)}


def run_c2(test: pd.DataFrame, years: list[int], rng: np.random.Generator) -> dict:
    pair, pay = X.load_pair_odds(set(years)), X.load_pay(set(years))
    tick = pd.concat([X.tickets_for_year(test, y, X.DEFAULT_AB, pair, pay) for y in years], ignore_index=True)
    c = (C2["bet"], C2["blend"], C2["theta"], C2["kmax"], C2["pmin"])
    parts, per_year = [], {}
    for Y in years:
        tp = X.race_profit(tick[tick["year"] == Y], c)
        parts.append(tp)
        per_year[Y] = float(1 + tp["pl"].sum() / max(tp["n"].sum(), 1))
    s = pooled(parts, rng)
    ex3 = X.ex_top3(tick[tick["year"].isin(years)], c)
    return {"candidate": "C2", "spec": str(C2), "per_year": per_year, **s, "ex_top3": ex3,
            "verdict": verdict(per_year, s["roi"], s["lo"], s["p_gt1"], s["bets"], ex3, True)}


def prep(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[(d["win_odds"] > 0) & d["ret_win"].notna() & d["ret_place"].notna()].copy()
    d["race_id"] = d["race_id"].astype(str)
    d["year"] = pd.to_datetime(d["date"]).dt.year
    for c in ("ret_win", "ret_place"):
        pos = d[c][d[c] > 0]
        if len(pos) and pos.median() > 20:
            d[c] = d[c] / 100.0
    d["place_odds_min"] = d["place_odds_min"].fillna(0)
    d["place_odds_max"] = d["place_odds_max"].fillna(0)
    d["n_run"] = d.groupby("race_id")["race_id"].transform("size")
    q = 1.0 / d["win_odds"]
    d["q_win"] = q / q.groupby(d["race_id"]).transform("sum")
    d["win"] = (d["ret_win"] > 0).astype(int)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-prior", required=True)
    ap.add_argument("--frame-test", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--label", default="")
    ap.add_argument("--decision-odds", choices=["t60", "t15", "t10", "t05", "t02"], default=None,
                    help="C1 の判断に使うオッズを、購入時点のずれを再現したものにする（既定は確定オッズ。登録では t05 または貯まった実測を使う）")
    ap.add_argument("--noise-seeds", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(20261109)
    prior, test = prep(Path(a.frame_prior)), prep(Path(a.frame_test))
    c1s = []
    for sd in range(7, 7 + (a.noise_seeds if a.decision_odds else 1)):
        pr_, te_ = prior, test
        if a.decision_odds:
            pr_ = P.decision_odds_noise(prior, a.decision_odds, np.random.default_rng(sd))
            te_ = P.decision_odds_noise(test, a.decision_odds, np.random.default_rng(sd))
        r1 = run_c1(pr_, te_, a.years, np.random.default_rng(20261109))
        r1["noise_seed"] = sd if a.decision_odds else None
        c1s.append(r1)
    res = {"years": a.years, "label": a.label, "decision_odds_c1": a.decision_odds or "final", "c1": c1s[0] if len(c1s) == 1 else c1s,
           "c2": run_c2(test, a.years, rng),
           "caveat": "確定オッズでの評価。組合せ券の購入時点のオッズの実測は未取得。条件は事前登録のとおりで、変えていない。"}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / f"confirm_{'_'.join(map(str, a.years))}{('_' + a.label) if a.label else ''}.json"
    f.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
