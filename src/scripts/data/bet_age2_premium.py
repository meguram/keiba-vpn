"""2 歳戦の単勝の上乗せ（JRA プレミアム「2歳単勝」、2016 年夏・2017〜2023 年）の印と補正を、凍結した買い方の記録に付けて読み直す（A1）。

python -m src.scripts.data.bet_age2_premium [--frozen config/bet_cycles/frozen_20261117_r2.json]

- 買い方は選び直さない（凍結のまま流し、記録の読み直しだけ）。新しい期間は見ない（訓練・評価・test 2025 は記録のある期間）。
- 印: 上乗せのあった期間（2016-06-04〜09-04、2017〜2023 年の 6〜12 月）の 2 歳戦で、確定の単勝オッズの Σ(1/オッズ) < 1.245 のレース
  （期間と Σ の両方。期間の外の Σ の低いレースは誤検出、期間の中で Σ が高いレースは人気の集中で上乗せが無かったレース）。
- 補正: 印のレースの単勝の払戻を × Σ_レース ÷ 1.264（今の制度の控除にそろえる）。判定（◎・見送り）は凍結のまま変えない。
- r の確かめ: 訓練の期間の全馬で、第 2 段の勝率 p に対する実際の勝率（勝率 ÷ p）と、市場 q に対する比（勝率 ÷ q）を、r の帯ごとに、
  印のある 2 歳戦と 3 歳以上で比べる。2 歳戦で r が高い馬ほど勝率 ÷ p が下がるなら、上乗せで q が低く出て r が甘くなっている。
出力: docs/operations/retrain_trials/bet_age2_premium_<hash>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FLAG_SIGMA = 1.245
NOW_SIGMA = 1.264


def in_premium_period(dates) -> np.ndarray:
    """上乗せのあった期間（公式の企画の期間。data_trend_20261118.md §3）: 2016-06-04〜09-04、2017〜2023 年の 6〜12 月。"""
    import pandas as pd

    d = pd.to_datetime(pd.Series(dates))
    y, md = d.dt.year.to_numpy(), d.dt.strftime("%m-%d").to_numpy()
    return ((y == 2016) & (md >= "06-04") & (md <= "09-04")) | ((y >= 2017) & (y <= 2023) & (d.dt.month.to_numpy() >= 6))


def run(a) -> None:
    from src.scripts.data import bet_cycles as bc

    fz = Path(a.frozen)
    body = json.loads(fz.read_text(encoding="utf-8"))
    spec, cuts = body["spec"], body["qcuts"]
    if any(t.get("type") != "tansho" for t in spec.get("tickets") or [{"type": "tansho"}]):
        raise SystemExit("この読み直しは単勝だけの買い方を前提にしています")
    names = [n for n in ("train", "eval", "test") if n in body["inputs"]["periods"]]
    from src.ops import data_ledger as L

    tok = L.guard(tool="src/scripts/data/bet_age2_premium.py", family="bet_cycles", version=body["hash"], period=("2019-01-01", "2025-12-31"),
                  purpose="reproduce", frozen_paths=[fz], reason="2 歳戦の上乗せの印と補正で、凍結の買い方の記録を読み直す（選び直さない）")
    per, _ = bc.prepare_periods(body["inputs"], tuple(names), cuts=cuts)
    out = {"frozen": str(fz.resolve().relative_to(ROOT)), "hash": body["hash"], "flag": f"上乗せのあった期間（2016-06-04〜09-04、2017〜2023 年の 6〜12 月）の 2 歳戦で Σ(1/確定の単勝オッズ) < {FLAG_SIGMA}",
           "adjust": f"印のレースの単勝の払戻 × Σ_レース ÷ {NOW_SIGMA}", "note": "買い方は選び直さない。記録の読み直しだけ。", "periods": {}}
    for nm in names:
        p = per[nm]
        n0 = len(p.base.races)
        age = p.base.feats["c_age_cond"].astype(str).to_numpy()
        sig = np.array([float(np.nansum(1.0 / r.odds)) for r in p.base.races])
        flag = (age == "2歳") & (sig < FLAG_SIGMA) & in_premium_period([r.date for r in p.base.races])
        fac = np.where(flag, sig / NOW_SIGMA, 1.0)
        r = bc.run_period(p, spec)
        k = p.k
        flagk, fack = np.tile(flag, k), np.tile(fac, k)
        ret_adj = r["ret"] * fack
        m_raw = bc.metrics_run(r, p, p.W)
        m_adj = bc.metrics(r["cost"], ret_adj, p.dates, p.W)
        bet = r["cost"] > 0
        out["periods"][nm] = {
            "years": body["inputs"]["periods"][nm]["years"], "races": n0, "age2_races": int((age == "2歳").sum()), "flag_races": int(flag.sum()),
            "bet_races": m_raw.get("bet_races"), "bet_in_flag": float((bet & flagk).sum() / k),
            "roi_raw": m_raw.get("roi"), "roi_raw_lo": m_raw.get("roi_lo"), "roi_raw_hi": m_raw.get("roi_hi"),
            "roi_adj": m_adj.get("roi"), "roi_adj_lo": m_adj.get("roi_lo"), "roi_adj_hi": m_adj.get("roi_hi"),
            "roi_ex_top5_raw": m_raw.get("roi_ex_top5"), "roi_ex_top5_adj": m_adj.get("roi_ex_top5"),
            "diff_pt": (m_raw.get("roi", 0) - m_adj.get("roi", 0)) * 100,
            "roi_in_flag_raw": float(r["ret"][bet & flagk].sum() / max(r["cost"][bet & flagk].sum(), 1)),
            "roi_in_flag_adj": float(ret_adj[bet & flagk].sum() / max(r["cost"][bet & flagk].sum(), 1)),
            "decision_odds": m_raw.get("decision_odds"),
        }
        print(nm, json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in out["periods"][nm].items()}, ensure_ascii=False), flush=True)
    # r の確かめ（訓練の期間、確定の時点の p・q・r。全馬）
    p = per["train"].base
    age = p.feats["c_age_cond"].astype(str).to_numpy()
    sig = np.array([float(np.nansum(1.0 / r.odds)) for r in p.races])
    grp = np.where((age == "2歳") & (sig < FLAG_SIGMA) & in_premium_period([r.date for r in p.races]), "2歳（印あり）", np.where(age == "2歳", "2歳（印なし）", "3歳以上"))
    edges = [-np.inf, 0.0, 0.3, 0.6, 1.0, np.inf]
    labels = ["r<0", "0〜0.3", "0.3〜0.6", "0.6〜1.0", "1.0〜"]
    rows = []
    for g in ("2歳（印あり）", "3歳以上"):
        idx = np.flatnonzero(grp == g)
        W_, P_, Q_, R_ = [], [], [], []
        for i in idx:
            rc = p.races[i]
            if rc.r is None or rc.q is None:
                continue
            W_.append((rc.fin == 1).astype(float))
            P_.append(rc.pw)
            Q_.append(rc.q / np.nansum(rc.q))
            R_.append(rc.r)
        w, pp, qq, rr = (np.concatenate(x) for x in (W_, P_, Q_, R_))
        b = np.searchsorted(np.asarray(edges[1:-1]), rr, side="right")
        for j, lab in enumerate(labels):
            mk = (b == j) & np.isfinite(rr)
            if mk.sum() < 50:
                continue
            rows.append({"群": g, "r の帯": lab, "頭数": int(mk.sum()), "勝率": float(w[mk].mean()), "勝率÷p": float(w[mk].sum() / pp[mk].sum()),
                         "勝率÷q": float(w[mk].sum() / qq[mk].sum())})
    out["r_check_train"] = rows
    dst = ROOT / f"docs/operations/retrain_trials/bet_age2_premium_{body['hash']}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    tok.finish(result_path=str(dst))
    print("書いた", dst.relative_to(ROOT))
    for x in rows:
        print(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=str(ROOT / "config/bet_cycles/frozen_20261117_r2.json"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
