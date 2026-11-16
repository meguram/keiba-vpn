"""弱み・強みの条件を「買う・買わない」の決まりにしたときに、的中率・回収率がどう変わるか（読み取りのみ）。

python -m src.scripts.data.rank_rule_backtest [--boot 1000] [--out DIR]

依頼（ユーザー、統括役経由 2026-11-14）: 弱みを除いた場合・強みだけで集計した場合に、的中率・回収率に差が出るか。
条件は rank_condition_analysis（docs/operations/retrain_trials/model_condition_strength_weakness_20261114.md）で、
valid 2022〜24 から見つけたもの。閾値（自信の度合いの 5 分位）は valid で凍結し、2025 年にそのまま当てはめる。
- valid の値は「条件を見つけた期間の値」で、良く出て当然（選んだ期間）。判断には 2025 年の値を使う。
- 2025 年の向きは、前の分析ですでにセルごとに見ている（ここで決まりを作るときに見えていた）。そのため 2025 年も
  完全に手つかずの検証ではない。決まりは前の分析の valid の結果だけから機械的に作り、2025 年を見て足し引きはしていない。

決まり（券種ごと。RULES）:
- 単勝: 弱み = 1 番手が 1 番人気、または単勝 2.0 倍未満。強み = 1 番手が 2〜5 番人気。
- 複勝: 弱み = 混戦（v6e_2016 の 1 番手の確率が下位 20%・確率の平たさが上位 20%・1〜2 番手の差が下位 20% のどれか）。
        強み = 自信あり（1 番手の確率が上位 20%、または平たさが下位 20%）。
- 馬連流し: 弱み = 1〜2 番手の差が下位 20%、9 頭以下、（B3・A6 だけ）馬場が不良。強み = 16 頭以上。
- ワイド流し: 弱み = 1〜2 番手の差が下位 20%、1 番手の確率が下位 20%、（B3・A6 だけ）馬場が不良。強み = 16 頭以上。
絞り込み: all（全体）、no_weak（弱みを全部外す）、strong（強みだけ）、no_weak_strong（弱みを外したうえで強みだけ）、
          drop:<条件>（その弱みの条件だけを外す。どの条件が効いているか）。

オッズ・人気は race_result_flat の最終オッズ（確定）。実際に買う時点（締切前）のオッズとはずれる。
出す値: レース数（買う割合）・的中率・回収率・全体との差（レース単位の Poisson ブートストラップの 95% 区間。全体と同じ重みで対にする）・
        1 日ごとの損益（100 円単位の購入額あたり）の標準偏差・最悪の日・プラスの日の割合・累積損益の最大の落ち込み。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import rank_condition_analysis as rc
from src.scripts.data.rank_bakeoff_eval import BETS, COST, load_payoff

RANK_MODELS = ("B3", "A6")       # ランキング学習の 2 つ（馬連・ワイドで不良馬場を弱みに足す）


def conditions(tabs: dict[str, pd.DataFrame], cond: pd.DataFrame, cuts: dict) -> dict[str, dict[str, np.ndarray]]:
    """モデル → {条件名: そのレースが条件に当たるか（bool の配列）}。自信の度合いは v6e_2016 の確率（全モデル共通）。"""
    src = tabs[rc.CONF_SRC]
    p1 = rc.apply_cuts(src["p1"].to_numpy(), cuts["conf_p1"])
    gap = rc.apply_cuts(src["gap"].to_numpy(), cuts["conf_gap"])
    flat = rc.apply_cuts(src["flat"].to_numpy(), cuts["conf_flat"])
    q = lambda lab, k: np.array([x.startswith(f"Q{k}(") for x in lab])   # noqa: E731
    race = {"p1_low": q(p1, 1), "p1_high": q(p1, 5), "gap_low": q(gap, 1), "flat_high": q(flat, 5), "flat_low": q(flat, 1),
            "field_small": (cond["field"] == "~9頭").to_numpy(), "field_large": (cond["field"] == "16頭~").to_numpy(),
            "going_bad": (cond["going"] == "不良").to_numpy()}
    out = {}
    for m, t in tabs.items():
        pop, odds = t["top1_pop"].to_numpy(dtype=float), t["top1_odds_v"].to_numpy(dtype=float)
        c = dict(race)
        c["fav"] = pop == 1
        c["odds_lt2"] = odds < 2.0
        c["pop2to5"] = (pop >= 2) & (pop <= 5)
        out[m] = c
    return out


def rules(model: str) -> dict[str, dict[str, list[str]]]:
    """券種 → {"weak": 弱みの条件名, "strong": 強みの条件名}（強みは「どれかに当たる」）。"""
    r = {"tansho": {"weak": ["fav", "odds_lt2"], "strong": ["pop2to5"]},
         "fukusho": {"weak": ["p1_low", "flat_high", "gap_low"], "strong": ["p1_high", "flat_low"]},
         "umaren": {"weak": ["gap_low", "field_small"], "strong": ["field_large"]},
         "wide": {"weak": ["gap_low", "p1_low"], "strong": ["field_large"]}}
    if model in RANK_MODELS:
        r["umaren"]["weak"] = r["umaren"]["weak"] + ["going_bad"]
        r["wide"]["weak"] = r["wide"]["weak"] + ["going_bad"]
    return r


def selections(c: dict[str, np.ndarray], rule: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """絞り込みの名前 → 買うレースの bool。"""
    n = len(next(iter(c.values())))
    weak = np.zeros(n, bool)
    for k in rule["weak"]:
        weak |= c[k]
    strong = np.zeros(n, bool)
    for k in rule["strong"]:
        strong |= c[k]
    sel = {"all": np.ones(n, bool), "no_weak": ~weak, "strong": strong, "no_weak_strong": ~weak & strong}
    for k in rule["weak"]:
        sel[f"drop:{k}"] = ~c[k]
    return sel


def daily_stats(dates: np.ndarray, ret: np.ndarray, cost: float, mask: np.ndarray) -> dict:
    """1 日ごとの損益（その日に買った全レースの「払戻 − 購入額」の合計、円。単勝・複勝は 1 レース 100 円、流しは 200 円）。
    買った日だけを数える（日数 days は絞り込みごとに違う）。絞り込みどうしを比べられるように、購入額で割った値も出す:
    day_roi_sd = 1 日の回収率（払戻 ÷ 購入額）の標準偏差、dd_days = 最大の落ち込み ÷ 1 日の平均購入額（何日分の購入額を失ったか）。"""
    d = pd.DataFrame({"date": dates, "pl": np.where(mask, ret - cost, 0.0), "ret": np.where(mask, ret, 0.0), "buy": mask.astype(int)})
    g = d.groupby("date").agg(pl=("pl", "sum"), ret=("ret", "sum"), buy=("buy", "sum"))
    g = g[g["buy"] > 0]
    if g.empty:
        return {"days": 0}
    cum = g["pl"].cumsum().to_numpy()
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum))
    stake = g["buy"] * cost
    day_roi = g["ret"] / stake
    return {"days": int(len(g)), "day_mean": float(g["pl"].mean()), "day_sd": float(g["pl"].std(ddof=1)) if len(g) > 1 else 0.0,
            "worst_day": float(g["pl"].min()), "pos_day_share": float((g["pl"] > 0).mean()), "max_drawdown": dd, "total_pl": float(cum[-1]),
            "day_stake": float(stake.mean()), "day_roi_sd": float(day_roi.std(ddof=1)) if len(g) > 1 else 0.0,
            "dd_days": dd / float(stake.mean())}


def evaluate(tabs: dict[str, pd.DataFrame], cond: pd.DataFrame, cuts: dict, W: np.ndarray, period: str) -> list[dict]:
    conds = conditions(tabs, cond, cuts)
    mkt = tabs[rc.MARKET]
    dates = cond["date"].to_numpy()
    tot_w = W.sum(1)
    rows = []
    for m, t in tabs.items():
        if m == rc.MARKET:                   # 市場（◎ は常に 1 番人気）には決まりを当てない。各モデルの行に「同じレースを市場で買ったら」を付ける
            continue
        rl = rules(m)
        for bt in BETS:
            c = COST[bt]
            ret, hit = t[f"ret_{bt}"].to_numpy(np.float32), t[f"hit_{bt}"].to_numpy(np.float32)
            mret, mhit = mkt[f"ret_{bt}"].to_numpy(np.float32), mkt[f"hit_{bt}"].to_numpy(np.float32)
            roi_all_b = (W @ ret) / (tot_w * c)
            hit_all_b = (W @ hit) / tot_w
            roi_all, hit_all = float(ret.sum() / (len(ret) * c)), float(hit.mean())
            for name, mask in selections(conds[m], rl[bt]).items():
                n = int(mask.sum())
                row = {"period": period, "model": m, "bet": bt, "subset": name, "n": n, "share": n / len(mask),
                       "rule_weak": "+".join(rl[bt]["weak"]), "rule_strong": "+".join(rl[bt]["strong"])}
                if n == 0:
                    rows.append(row)
                    continue
                mk = mask.astype(np.float32)
                nb = W @ mk
                with np.errstate(divide="ignore", invalid="ignore"):
                    roi_b = (W @ (ret * mk)) / (nb * c)
                    hit_b = (W @ (hit * mk)) / nb
                roi, h = float(ret[mask].sum() / (n * c)), float(hit[mask].mean())
                mroi_b = (W @ (mret * mk)) / (nb * c)
                row.update({"mkt_roi": float(mret[mask].sum() / (n * c)), "mkt_hit": float(mhit[mask].mean()),   # 同じレースを市場の 1 番人気で買ったら
                            "vs_mkt": roi - float(mret[mask].sum() / (n * c)),
                            "vs_mkt_lo": float(np.nanquantile(roi_b - mroi_b, 0.025)), "vs_mkt_hi": float(np.nanquantile(roi_b - mroi_b, 0.975)),
                            "boot_nan": int(np.isnan(roi_b).sum())})                                            # 重みが全部 0 になって捨てた回数
                hits = float(hit[mask].sum())
                row.update({"hit_rate": h, "avg_pay": float(ret[mask].sum() / hits) if hits else 0.0, "roi": roi,
                            "roi_all": roi_all, "hit_all": hit_all, "d_roi": roi - roi_all,
                            "d_roi_lo": float(np.nanquantile(roi_b - roi_all_b, 0.025)), "d_roi_hi": float(np.nanquantile(roi_b - roi_all_b, 0.975)),
                            "roi_lo": float(np.nanquantile(roi_b, 0.025)), "roi_hi": float(np.nanquantile(roi_b, 0.975)),
                            "d_hit": h - hit_all, "d_hit_lo": float(np.nanquantile(hit_b - hit_all_b, 0.025)),
                            "d_hit_hi": float(np.nanquantile(hit_b - hit_all_b, 0.975))})
                row.update(daily_stats(dates, ret, c, mask))
                rows.append(row)
    return rows


SUB_LABEL = {"all": "全体", "no_weak": "弱みを除く", "strong": "強みだけ", "no_weak_strong": "弱みを除き強みだけ"}
COND_LABEL = {"fav": "1番人気", "odds_lt2": "2.0倍未満", "pop2to5": "2〜5番人気", "p1_low": "1番手の確率が下位20%", "p1_high": "1番手の確率が上位20%",
              "gap_low": "1〜2番手の差が下位20%", "flat_high": "平たさが上位20%", "flat_low": "平たさが下位20%",
              "field_small": "9頭以下", "field_large": "16頭以上", "going_bad": "不良馬場"}


def sub_label(s: str) -> str:
    return f"{COND_LABEL.get(s[5:], s[5:])}だけ外す" if s.startswith("drop:") else SUB_LABEL.get(s, s)


def render(df: pd.DataFrame) -> str:
    L = ["# 弱みを除く・強みだけの集計（rank_rule_backtest）", "",
         "回収率の括弧は全体との差の 95% 区間（レース単位のブートストラップ、全体と対）。",
         "市場（同じレース）= そのモデルが買うのと同じレースを、市場の 1 番人気（→ 2・3 番人気）で買った回収率。",
         "1 日の値は買った日だけ（日数の列）。1 日の回収率の標準偏差と、最大の落ち込み ÷ 1 日の平均購入額（何日分）は、絞り込みどうしで比べられる。", ""]
    for period in ("test2025", "valid"):
        L += [f"## {'2025 年（判断に使う）' if period == 'test2025' else 'valid 2022〜24（条件を見つけた期間。良く出て当然）'}", ""]
        for bt in BETS:
            L += [f"### {rc.BET_LABEL[bt]}", "", "| モデル | 絞り込み | レース数（割合） | 的中率（差） | 平均払戻 | 回収率 | 全体との差（区間） | 市場（同じレース） | 市場との差（区間） | 日数 | 1日の回収率の標準偏差 | プラスの日 | 最大の落ち込み（円・何日分） |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            g = df[(df["period"] == period) & (df["bet"] == bt)]
            for _, r in g.iterrows():
                if r["n"] == 0:
                    L.append(f"| {r['model']} | {sub_label(r['subset'])} | 0 | | | | | | | | | | |")
                    continue
                L.append(f"| {r['model']} | {sub_label(r['subset'])} | {r['n']}（{r['share']:.0%}） | {r['hit_rate']:.3f}（{r['d_hit']:+.3f}） | {r['avg_pay']:.0f} | "
                         f"{r['roi']:.3f} | {r['d_roi']:+.3f}（{r['d_roi_lo']:+.3f}〜{r['d_roi_hi']:+.3f}） | {r['mkt_roi']:.3f} | "
                         f"{r['vs_mkt']:+.3f}（{r['vs_mkt_lo']:+.3f}〜{r['vs_mkt_hi']:+.3f}） | {r['days']} | {r['day_roi_sd']:.2f} | "
                         f"{r['pos_day_share']:.0%} | {r['max_drawdown']:.0f}・{r['dd_days']:.1f} |")
            L.append("")
    return "\n".join(L) + "\n"


def run(n_boot: int, out: Path) -> pd.DataFrame:
    pay = load_payoff()
    tv, cv = rc.build_period(rc.VALID_YEARS, 0, pay)
    tt, ct = rc.build_period(rc.TEST_YEARS, 1, pay)
    cuts = {k: rc.quantile_cuts(tv[rc.CONF_SRC][c].to_numpy()) for k, c in (("conf_p1", "p1"), ("conf_gap", "gap"), ("conf_flat", "flat"))}   # valid で凍結
    rows = evaluate(tv, cv, cuts, rc.poisson_weights(len(cv), n_boot, 0), "valid")
    rows += evaluate(tt, ct, cuts, rc.poisson_weights(len(ct), n_boot, 1), "test2025")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "rule_backtest.parquet", index=False)
    (out / "rule_backtest.json").write_text(json.dumps({"cuts": cuts, "n_races": {"valid": len(cv), "test2025": len(ct)},
                                                        "rules": {m: rules(m) for m in tabs_names(tv)}}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "rule_backtest.md").write_text(render(df), encoding="utf-8")
    return df


def tabs_names(tabs: dict) -> list[str]:
    return list(tabs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=str(rc.B16 / "tp_rank/cond"))
    a = ap.parse_args()
    df = run(a.boot, Path(a.out))
    t = df[df["period"] == "test2025"]
    print(t.pivot_table(index=["bet", "subset"], columns="model", values="roi").round(3).to_string())


if __name__ == "__main__":
    main()
