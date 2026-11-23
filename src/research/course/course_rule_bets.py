"""めぐ分析の「過去データの条件」（凍結した全レース版の第 3 版）で、実際に馬券を買ったらどうなるかを数える（表示用・評価だけ。条件は選び直さない）。

買い方（ユーザーの依頼、2026-11-22）:
- レースの単位（コース × クラスの群）で、めぐ分析に出る買いの条件（rule_stats_v3.json の各単位の上位 5 つ）を見る。
- 訓練（抽出の期間 2016〜2024 年）のその条件の単勝の回収率が 100% 以上なら、当てはまる馬の単勝を 100 円買う。複勝も同じく、訓練の複勝の回収率で別に判定する。
- 1 頭が複数の条件に当てはまっても 1 点（馬ごとに重複を除く）。
- 判定に使う値は訓練の値だけ（2025・2026 年の値では選ばない）。条件の水準の一部（人気など）は確定オッズで決まるので、判定は楽観。

使い方: python -m src.research.course.course_rule_bets（子の処理で 2026 年の表を作る。出力 <COURSE_OUT>/v3/rule_bets_v3.json と
docs/operations/retrain_trials/strategy_tables_b.json）。
2025・2026 年の着順・払戻を読むので、流す前に台帳へ eval_only を申告する（family course）。費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import search as SE
from src.research.course import v2 as V2
from src.research.course.v3 import C3, EVAL_MARK3, OUT3, PREREG3, _frame_2026, load3, ordered_conds, units_of

STATS = CR.REPO / "data/meta/course/rule_stats_v3.json"
OUT = OUT3 / "rule_bets_v3.json"
OUT_REPO = CR.REPO / "docs/operations/retrain_trials/strategy_tables_b.json"


def chosen_rules(stats: dict, kept: dict) -> dict:
    """単位ごとに、めぐ分析に出る買いの条件（表示の番号 i）と、訓練の単・複の回収率・走数。"""
    out = {}
    for u, r in stats["rules"].items():
        tv = (stats["values"]["train"].get(u) or {}).get("buy") or []
        rows = []
        for j, b in enumerate(r.get("buy") or []):
            t = tv[j] if j < len(tv) else {}
            rows.append({"i": b["i"], "text": b["text"], "n": t.get("n"), "roi_t": t.get("roi_t"), "roi_f": t.get("roi_f")})
        if u in kept:
            out[u] = rows
    return out


def period_bets(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: C3, rules: dict) -> dict:
    """期間の表に、買う条件（訓練の回収率 ≥ 100%）を当て、単勝・複勝それぞれ馬ごとに重複を除いて数える。"""
    X, types, code = V2._codes(L, kept)
    Y = SE.y_matrix(d)
    unit = units_of(d, c)
    race = d["race_id"].astype(str).to_numpy()
    buy = {"tansho": np.zeros(len(d), bool), "fukusho": np.zeros(len(d), bool)}
    shown_hit = np.zeros(len(d), bool)              # 表示の条件のどれかに当てはまる馬（回収率で絞る前）
    n_rules = {"tansho": 0, "fukusho": 0, "shown": 0}
    for u, cu in kept.items():
        idx = np.flatnonzero(unit == u)
        if not len(idx) or u not in rules:
            continue
        conds, _ = ordered_conds(cu)
        Xu = X[idx]
        for r in rules[u]:
            m = V2._mask_names(Xu, types, code, conds[r["i"]])
            shown_hit[idx[m]] = True
            n_rules["shown"] += 1
            for kind, key in (("tansho", "roi_t"), ("fukusho", "roi_f")):
                if r.get(key) is not None and r[key] >= 1.0:
                    buy[kind][idx[m]] = True
                    n_rules[kind] += 1
    res = {"races": int(len(set(race))), "runners": int(len(d)), "rules_used": n_rules,
           "date_min": str(pd.to_datetime(d["date"]).min())[:10] if "date" in d.columns else None,
           "date_max": str(pd.to_datetime(d["date"]).max())[:10] if "date" in d.columns else None,
           "shown_hit_runners": int(shown_hit.sum())}
    for kind, col in (("tansho", 6), ("fukusho", 7)):
        m = buy[kind]
        Yb = Y[m]
        pay = Yb[:, 5] > 0
        hit = (Yb[:, 2] > 0) if kind == "tansho" else (Yb[:, 7] > 0)   # 複勝は払戻が出たか（7 頭以下の 3 着は外れ）
        n = int(m.sum())
        npay = int(pay.sum())
        rets = Yb[pay, col]
        rng = np.random.default_rng(20261122)
        rr = pd.Series(race[m][pay])                 # レース単位のブートストラップ（同じレースの馬は一緒に引く）
        g = pd.DataFrame({"r": rr.to_numpy(), "ret": rets}).groupby("r")["ret"].agg(["sum", "count"])
        if len(g):
            w = rng.poisson(1.0, size=(1000, len(g)))
            bs = (w @ g["sum"].to_numpy()) / np.maximum(w @ g["count"].to_numpy(), 1e-9)
            lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        else:
            lo = hi = None
        c1, c2, c3 = (int(Yb[:, i].sum()) for i in (2, 3, 4))
        res[kind] = {"bets": n, "bet_races": int(len(set(race[m]))), "npay": npay, "hit": int(hit.sum()),
                     "hit_rate": float(hit.mean()) if n else None, "roi": float(rets.sum() / npay) if npay else None,
                     "roi_lo": lo, "roi_hi": hi, "rec": [c1, c2, c3, n - c1 - c2 - c3],
                     "win": c1 / n if n else None, "place": (c1 + c2 + c3) / n if n else None}
    return res


def cmd_run() -> None:
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    if CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    mark = json.loads(EVAL_MARK3.read_text(encoding="utf-8")) if EVAL_MARK3.exists() else {}
    if "evaluated_at" not in mark:
        sys.exit("2026 年の 1 回だけの評価が済んでいない")
    CR.wait_resources()
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval")}
    subprocess.run([sys.executable, "-m", "src.research.course.course_rule_bets", "_inner"], env=e, cwd=CR.REPO, check=True)


def _inner() -> None:
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if stats["sha256_cells"] != P["sha256_cells"] or CR._sha_file(OUT3 / P["cells_file"]) != P["sha256_cells"]:
        sys.exit("めぐ分析の条件の表・凍結の sha が合わない")
    c = C3(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    rules = chosen_rules(stats, kept)
    out = {}
    d, L, tr, ev = load3()
    out["train"] = period_bets(d[tr].reset_index(drop=True), L[tr].reset_index(drop=True), kept, c, rules)
    out["2025"] = period_bets(d[ev].reset_index(drop=True), L[ev].reset_index(drop=True), kept, c, rules)
    del d, L
    d6, L6 = _frame_2026()
    out["2026"] = period_bets(d6, L6, kept, c, rules)
    n_t = sum(1 for rs in rules.values() for r in rs if (r.get("roi_t") or 0) >= 1.0)
    n_f = sum(1 for rs in rules.values() for r in rs if (r.get("roi_f") or 0) >= 1.0)
    n_small = sum(1 for rs in rules.values() for r in rs if ((r.get("roi_t") or 0) >= 1.0 or (r.get("roi_f") or 0) >= 1.0) and (r.get("n") or 0) < 30)
    res = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cycle": P["cycle"], "sha256_cells": P["sha256_cells"],
           "stats_made_at": stats.get("made_at"), "periods": {"train": [2016, 2024], "2025": [2025, 2025], "2026": [2026, 2026]},
           "rules_total": sum(len(v) for v in rules.values()), "rules_buy_tansho": n_t, "rules_buy_fukusho": n_f, "rules_buy_small_n": n_small,
           "units": len(rules), "values": out,
           "note": ("凍結した第 3 版の条件のうち、めぐ分析に出る買いの条件（各単位の上位 5）で、訓練の回収率 100% 以上のものに当てはまる馬を買う。"
                    "対象は良馬場・芝ダ・重賞以外のレース。訓練の値は条件を選んだ期間の当てはめの値（楽観）。条件の水準は確定オッズの人気などを含む（楽観）")}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    OUT_REPO.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print("rule_bets", OUT, {k: (v["tansho"]["bets"], v["tansho"]["roi"], v["fukusho"]["roi"]) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["_inner"]:
        _inner()
    else:
        cmd_run()
