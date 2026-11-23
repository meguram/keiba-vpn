"""最新の買い方の成績表（docs/html/operations/final/latest_strategy_tables.html）の数字を、記録と道具から作る。

- a2: 第 2 回の凍結（config/bet_cycles/frozen_20261117_r2.json）の買い方。
  訓練 2019〜22・評価 2023〜24 は bet_cycles の道具で流し直し、記録（final_candidate.json）の回収率と合うかを照合する。
  test 2025・2026 年の主は、凍結の道具が書いたレースごとの記録（final_test_races_*・prereg_main_races_*）を読み、回収率を照合する。
  どの期間も ◎ の着順（1-2-3-着外）を、結果の表（race_result_flat）に当てて数える。本番の再現は 5 通りの版を並べた形なので、版で割る。
  2025・2026 年の着順を読むので、流す前に台帳へ reproduce を申告する（python -m src.ops.data_ledger declare）。
- a1: 本番の推奨馬券（target・sec77）の 2026 年の日次の記録（data/local/prediction_results/*.json）を集計する。
- b: めぐ分析の過去データの条件（src/research/course/course_rule_bets.py の出力）を読む（ここでは数えない）。

出力は JSON（既定 docs/operations/retrain_trials/strategy_tables_<部>.json）。HTML は src/scripts/docs/build_latest_strategy_tables.py が作る。
費用 0 円（手元のファイルだけ。GCS を読まない）。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
FROZEN_R2 = ROOT / "config/bet_cycles/frozen_20261117_r2.json"
R2_OUT = Path(os.environ.get("BET_CYCLES_R2_RECORDS") or "/home/jovyan/work/retrain_out/bet_cycles/r2")
OUT_DIR = ROOT / "docs/operations/retrain_trials"


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float | None, float | None]:
    """率の 95% 区間（Wilson）。件数が版の平均（小数）でも使う。"""
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def finish_table(years) -> pd.DataFrame:
    """(race_id, 馬番) → 着順（数値。取消・除外・中止は NaN）。"""
    from src.scripts.data import bet_cycles as bc

    f = bc.load_runners(years)
    return f[["race_id", "horse_number", "finish_position", "date"]].rename(columns={"horse_number": "H_num"})


def honmei_record(rec: pd.DataFrame, fin: pd.DataFrame, k: int) -> dict:
    """買った行（cost>0）の ◎ の着順を数え、版の数 k で割る。着外には取消・中止（着順なし）も入れ、別にも数える。"""
    b = rec[rec["cost"] > 0].copy()
    b["race_id"] = b["race_id"].astype(str)
    b["H_num"] = pd.to_numeric(b["H_num"], errors="coerce")
    m = b.merge(fin[["race_id", "H_num", "finish_position"]], on=["race_id", "H_num"], how="left")
    fp = m["finish_position"]
    n = len(m) / k
    c1, c2, c3 = ((fp == i).sum() / k for i in (1, 2, 3))
    nofin = fp.isna().sum() / k
    out = n - c1 - c2 - c3
    lo, hi = wilson(c1, n)
    plo, phi = wilson(c1 + c2 + c3, n)
    return {"n": n, "rec": [c1, c2, c3, out], "no_finish": nofin, "win": c1 / n if n else None, "win_lo": lo, "win_hi": hi,
            "place": (c1 + c2 + c3) / n if n else None, "place_lo": plo, "place_hi": phi, "versions": k}


def _pick(m: dict) -> dict:
    keys = ("races", "bet_races", "bet_rate", "hit_rate", "hit_lo", "hit_hi", "roi", "roi_lo", "roi_hi", "roi_ex_top5", "avg_return",
            "median_return", "cost_total", "return_total", "days", "decision_odds", "sim_roi_min", "sim_roi_max", "n_versions")
    return {k: m.get(k) for k in keys if k in m}


def cmd_a2(a) -> None:
    from src.scripts.data import bet_cycles as bc

    body = json.loads(FROZEN_R2.read_text(encoding="utf-8"))
    if bc.strategy_hash({k: v for k, v in body.items() if k not in ("frozen_at", "hash")}) != body["hash"]:
        raise SystemExit("凍結ファイルのハッシュが合いません")
    want = body["pred_sha256"].get(str(bc.PRE_ODDS.relative_to(bc.ROOT)))
    cand = [bc.PRE_ODDS] + sorted(R2_OUT.glob("pre_odds_frozen_*.parquet"))
    bc.DRIFT_SRC = next((c for c in cand if c.exists() and bc._sha(c) == want), None)   # 再現のずれの元は凍結のときの中身
    if bc.DRIFT_SRC is None:
        raise SystemExit("凍結のときの発走前オッズが見つかりません")
    cand_rec = json.loads((R2_OUT / "final_candidate.json").read_text(encoding="utf-8"))
    if cand_rec["spec"] != body["spec"]:
        raise SystemExit("final_candidate.json の買い方が凍結と違います")
    res = {"made_at": datetime.now(JST).isoformat(timespec="seconds"), "frozen": str(FROZEN_R2.relative_to(ROOT)), "hash": body["hash"],
           "spec_name": body["spec"]["name"], "tickets": [t["type"] for t in body["spec"]["tickets"]], "periods": {}}
    pred_now = bc._sha(Path(body["inputs"]["periods"]["train"]["pred"][0]["path"]))
    res["pred_sha_now"], res["pred_sha_frozen"] = pred_now, body["pred_sha256"].get(body["inputs"]["periods"]["train"]["pred"][0]["path"])

    # 訓練・評価: 道具で流し直す（記録の回収率と照合）
    per, _ = bc.prepare_periods(body["inputs"], ("train", "eval"), cuts=body["qcuts"])
    for nm in ("train", "eval"):
        p = per[nm]
        r = bc.run_period(p, body["spec"], detail=True)
        m = bc.metrics_run(r, p, p.W)
        recorded = cand_rec["check"][nm]["metrics"]
        ok = abs(m["roi"] - recorded["roi"]) < 1e-6 and m["bet_races"] == recorded["bet_races"]
        fin = finish_table(body["inputs"]["periods"][nm]["years"])
        res["periods"][nm] = {"years": body["inputs"]["periods"][nm]["years"], "metrics": _pick(m),
                              "recorded": {"roi": recorded["roi"], "bet_races": recorded["bet_races"], "source": "final_candidate.json"},
                              "reproduced": bool(ok), "honmei": honmei_record(r["records"], fin, p.k),
                              "role": body["inputs"]["periods"][nm].get("role")}
        print(nm, "再現" if ok else "再現せず", m["roi"], recorded["roi"], m["bet_races"], recorded["bet_races"], flush=True)

    # test 2025・2026 年の主: 凍結の道具の記録を読む（流し直さない）
    t25 = json.loads((OUT_DIR / f"bet_test2025_r2_test_{body['hash']}.json").read_text(encoding="utf-8"))["periods"]["test"]
    m26 = json.loads((OUT_DIR / f"bet_2026_r2_main_{body['hash']}.json").read_text(encoding="utf-8"))
    for nm, fn, recm, years, extra in (
            ("test2025", f"final_test_races_{body['hash']}.parquet", t25["metrics"], [2025], {"role": t25.get("role")}),
            ("main2026", f"prereg_main_races_{body['hash']}.parquet", m26["metrics"], [2026],
             {"role": "2026 年の主（事前登録の群 main、推論の経路の予測・評価だけ）", "verdict": m26.get("verdict"),
              "dates": (m26.get("group") or {}).get("dates")})):
        rec = pd.read_parquet(R2_OUT / fn)
        k = int(rec["ver"].max()) + 1
        cost, ret = rec["cost"].to_numpy(float), rec["ret"].to_numpy(float)
        roi = ret.sum() / cost.sum()
        nb = (cost > 0).sum() / k
        ok = abs(roi - recm["roi"]) < 1e-6 and round(nb) == recm["bet_races"]
        fin = finish_table(years)
        res["periods"][nm] = {"years": years, "metrics": _pick(recm), "reproduced": bool(ok),
                              "recorded": {"roi": recm["roi"], "bet_races": recm["bet_races"], "source": fn},
                              "honmei": honmei_record(rec, fin, k)} | extra
        print(nm, "再現" if ok else "再現せず", roi, recm["roi"], nb, recm["bet_races"], flush=True)
    fo = cand_rec["check"].get("final_odds") or {}        # 楽観（確定オッズで判定）の記録の値
    res["optimistic_final_odds"] = {"train": (fo.get("train") or {}).get("roi"), "eval": (fo.get("eval") or {}).get("roi"),
                                    "test2025": (t25.get("metrics_final_odds") or {}).get("roi")}
    dst = Path(a.out or OUT_DIR / "strategy_tables_a2.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print("書いた", dst)


def _boot_roi(cost: np.ndarray, ret: np.ndarray, n: int = 1000, seed: int = 20261122) -> tuple[float | None, float | None]:
    """レース単位のブートストラップ（Poisson の重み）で回収率の 95% 区間。"""
    b = cost > 0
    if not b.any():
        return None, None
    w = np.random.default_rng(seed).poisson(1.0, size=(n, int(b.sum())))
    bs = (w @ ret[b]) / np.maximum(w @ cost[b], 1e-9)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _stat(cost, ret, hit) -> dict:
    cost, ret, hit = (np.asarray(x, float) for x in (cost, ret, hit))
    b = cost > 0
    lo, hi = _boot_roi(cost, ret)
    hl, hh = wilson(float(hit[b].sum()), float(b.sum()))
    return {"bet_races": int(b.sum()), "cost": float(cost.sum()), "ret": float(ret.sum()),
            "hit_rate": float(hit[b].mean()) if b.any() else None, "hit_lo": hl, "hit_hi": hh,
            "roi": float(ret.sum() / cost.sum()) if cost.sum() else None, "roi_lo": lo, "roi_hi": hi}


def _marks(rows: list[dict]) -> dict:
    fin = [int(x["finish"]) if isinstance(x.get("finish"), (int, float)) and x["finish"] == x["finish"] else None for x in rows]
    n = len(fin)
    c = [sum(1 for f in fin if f == i) for i in (1, 2, 3)]
    lo, hi = wilson(c[0], n)
    plo, phi = wilson(sum(c), n)
    return {"n": n, "rec": c + [n - sum(c)], "win": c[0] / n if n else None, "win_lo": lo, "win_hi": hi,
            "place": sum(c) / n if n else None, "place_lo": plo, "place_hi": phi}


A1_PERIODS = (("sealed2026", "2026-01-01", "2026-11-20", "2026 年"), ("forward", "2026-11-21", "2026-11-21", "前向き"))   # 前向きの終わりは台帳の申告（〜11/21）に合わせる。延ばすときは申告し直す


def cmd_a1(a) -> None:
    """本番の予測成績の日次の記録（prediction_results）から、推奨馬券（target と従来方式）と ◎ の着順を期間ごとに集計する。"""
    from src.pipeline.inference.prediction_results import TARGET_ODDS_BASIS as PR_BASIS, TARGET_SELECTION as PR_SELECTION

    src = Path(a.dir)
    rows = []
    for f in sorted(src.glob("2026*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        day = f"{f.stem[:4]}-{f.stem[4:6]}-{f.stem[6:8]}"
        for r in d.get("races") or []:
            if r.get("result_status") not in ("final", "provisional") or not r.get("included"):     # 本番の集計と同じ（included が真のレースだけ）
                continue
            bt, bl = r.get("bets_t") or {}, r.get("bets") or {}
            cands = bl.get("candidates") or []
            rows.append({"date": day, "day_status": d.get("status"), "race_id": r["race_id"], "model": r.get("model_version"),
                         "source": r.get("prediction_source"),
                         "t_status": ("ok" if (bt.get("status") == "ok" and bt.get("basis") == PR_BASIS and bt.get("selection") == PR_SELECTION)
                                      else (bt.get("status") or "none")),       # 本番の record_level と同じ基準（sec77・actual_ge_est）
                         "t_level": bt.get("level"), "t_all_bet": (bt.get("all") or {}).get("total_bet") or 0,
                         "t_all_ret": (bt.get("all") or {}).get("total_return") or 0, "t_sel_bet": bt.get("total_bet") or 0, "t_sel_ret": bt.get("total_return") or 0,
                         "l_status": bl.get("status"), "l_level": bl.get("level"), "l_odds_phase": bl.get("odds_phase"), "l_est": bool(bl.get("estimated")),
                         "l_bet": bl.get("total_bet", 0) or 0, "l_ret": bl.get("total_return", 0) or 0,
                         "l_hit": any(c.get("hit") for c in cands),
                         "l_types": {t: v for t, v in (bl.get("by_type") or {}).items()},
                         "honmei": (r.get("marks") or {}).get("honmei") or [], "top1": [r["top1"]] if r.get("top1") else []})
    df = pd.DataFrame(rows)
    res = {"made_at": datetime.now(JST).isoformat(timespec="seconds"), "source_dir": str(src), "days": int(df["date"].nunique()),
           "models": df["model"].value_counts().to_dict(), "sources": df["source"].value_counts().to_dict(), "periods": {}}
    for key, lo, hi, label in A1_PERIODS:
        p = df[(df["date"] >= lo) & (df["date"] <= hi)]
        if p.empty:
            continue
        t = p[p["t_status"] == "ok"]
        legacy = p[p["l_status"] == "ok"]
        by_level = {}
        for lv, g in legacy.groupby("l_level"):
            by_level[str(int(lv))] = _stat(g["l_bet"], g["l_ret"], g["l_hit"])
        by_type = {}
        for typ in sorted({k for x in legacy["l_types"] for k in x}):
            inv = np.array([x.get(typ, {}).get("invested", 0) for x in legacy["l_types"]], float)
            ret = np.array([x.get(typ, {}).get("returned", 0) for x in legacy["l_types"]], float)
            hit = np.array([x.get(typ, {}).get("hit", 0) > 0 for x in legacy["l_types"]], float)
            by_type[typ] = _stat(inv, ret, hit)
        res["periods"][key] = {
            "label": f'{label}（{p["date"].min()[5:].replace("-", "/")}〜{p["date"].max()[5:].replace("-", "/")}' + ("・速報を含む）" if (p["day_status"] != "final").any() else "）"),
            "races": int(len(p)), "days": int(p["date"].nunique()), "date_min": p["date"].min(), "date_max": p["date"].max(),
            "sources": p["source"].value_counts().to_dict(), "target_sources": t["source"].value_counts().to_dict(),
            "provisional_days": int(p.loc[p["day_status"] != "final", "date"].nunique()),
            "target": {"races_ok": int(len(t)), "races_basis_missing": int((p["t_status"] != "ok").sum()),
                       "level_counts": t["t_level"].value_counts().to_dict(), "bought_races": int((t["t_all_bet"] > 0).sum()),
                       "all": _stat(t["t_all_bet"], t["t_all_ret"], t["t_all_ret"] > 0)},
            "legacy": {"races_ok": int(len(legacy)), "odds_phase": legacy["l_odds_phase"].value_counts().to_dict(),
                       "estimated_races": int(legacy["l_est"].sum()),
                       "all": _stat(legacy["l_bet"], legacy["l_ret"], legacy["l_hit"]), "by_level": by_level, "by_type": by_type},
            "honmei": _marks([x for xs in p["honmei"] for x in xs]), "top1": _marks([x for xs in p["top1"] for x in xs])}
    rep = ROOT / "data/local/modeling/portfolio_target/pattern_report.json"     # 既定方策を決めたときの検証（2026-10-24、v6e より前のモデル）
    if rep.exists():
        sel = json.loads(rep.read_text(encoding="utf-8"))["selection"]["k2.0_tau1.0"]["frac"]["30%"]
        res["validation"] = {"source": "data/local/modeling/portfolio_target/pattern_report.json（selection.k2.0_tau1.0.frac.30%）",
                             "made_at": datetime.fromtimestamp(rep.stat().st_mtime, JST).isoformat(timespec="seconds"),
                             "ev_threshold": sel["ev_threshold"], "pooled_2022_2024": sel["pooled_2022_2024"], "2025": sel["2025"],
                             "note": "v6e より前のモデルの予測で、設定（k・τ・レースの区切り）を選んだ期間の値。◎ の着順は記録に無い"}
    dst = Path(a.out or OUT_DIR / "strategy_tables_a1.json")
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: {"races": v["races"], "target_bought": v["target"]["bought_races"], "legacy_roi": v["legacy"]["all"]["roi"],
                          "honmei": v["honmei"]["rec"], "top1": v["top1"]["rec"]} for k, v in res["periods"].items()}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("a2", help="第 2 回の凍結の買い方（A2）")
    s.add_argument("--out", default="")
    s = sub.add_parser("a1", help="本番の推奨（A1）の 2026 年の日次の記録")
    s.add_argument("--dir", default=str(ROOT / "data/local/prediction_results"), help="日次の記録の置き場（root 専用のときは写しを指す）")
    s.add_argument("--out", default="")
    a = ap.parse_args()
    {"a2": cmd_a2, "a1": cmd_a1}[a.cmd](a)


if __name__ == "__main__":
    main()
