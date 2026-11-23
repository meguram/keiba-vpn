"""◎ を固定した馬券のパターンの道具（A2 = 第 2 回の凍結 frozen_20261117_r2 の ◎）。

  python -m src.research.bet_patterns.bet_patterns_run build                    # 訓練の行を作る（台帳に select を申告してから着順・払戻を読む）
  python -m src.research.bet_patterns.bet_patterns_run select                   # 訓練だけで想定オッズの換算・組み方・閾値を選ぶ → config/bet_patterns/selection.json
  python -m src.research.bet_patterns.bet_patterns_run prereg                   # 評価の前に主の規則・基準・合格の条件を書く → config/bet_patterns/prereg.json
  （selection.json・prereg.json・道具を commit してから）
  python -m src.research.bet_patterns.bet_patterns_run evaluate                 # 選んだ規則を評価・2025・2026 に 1 回だけ当てる → docs/operations/retrain_trials/bet_patterns.json

期間と台帳の目的: train（2019〜22）は select、eval（2023〜24）は confirm（1 回だけ）、test（2025）・main2026・pre2026（本物の T-60）は eval_only（選び直しに使わない）。
重い計算なので nice -n 19 で流す。費用 0 円（手元のファイルだけ。GCS を読まない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.research.bet_patterns import data as D
from src.research.bet_patterns import odds_map as OM
from src.research.bet_patterns import search as S

JST = timezone(timedelta(hours=9))
PERIODS = {"train": ("2019-01-01", "2022-12-31"), "eval": ("2023-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31"), "pre2026": D.PRE2026,
           "main2026": D.MAIN2026}
PURPOSE = {"train": "select", "eval": "confirm", "test": "eval_only", "main2026": "eval_only", "pre2026": "eval_only"}
FAMILY = "bet_patterns"
SELECTION = D.ROOT / "config/bet_patterns/selection.json"
PREREG = D.ROOT / "config/bet_patterns/prereg.json"
RESULT = D.ROOT / "docs/operations/retrain_trials/bet_patterns.json"
A2_TABLE = D.ROOT / "docs/operations/retrain_trials/strategy_tables_a2.json"
TRAIN_YEARS = (2019, 2020, 2021, 2022)


def rows_path(period: str, setting: str) -> Path:
    return D.CACHE / f"rows_{period}_{setting}.npz"


def guard(period: str, version: str, reason: str, frozen=(), purpose: str | None = None):
    from src.ops import data_ledger as L

    return L.guard(tool="src/research/bet_patterns/bet_patterns_run.py", family=FAMILY, version=version, period=PERIODS[period],
                   purpose=purpose or PURPOSE[period], frozen_paths=list(frozen), reason=reason, target_name=f"bet_patterns {period}")


def seen_after_train() -> list[str]:
    """台帳に、この系統で訓練の外（評価・2025・2026）を読んだ行があるか（選び直し・評価の流し直しを止める。RESULT を消しても効く）。"""
    from src.ops import data_ledger as L

    out = []
    for r in L.load_rows():
        t = r.get("target") or {}
        if r.get("kind") == "start" and t.get("family") == FAMILY and (t.get("name") or "").split(" ")[-1] in ("eval", "test", "main2026", "pre2026"):
            fin = [x for x in L.load_rows() if x.get("ref") == r.get("id") and x.get("kind") == "finish"]
            if not any(x.get("status") == "error" for x in fin):
                out.append(f'{t.get("name")}（{r.get("at")}）')
    return out


def _now() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=list).encode()).hexdigest()[:12]


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def cmd_build(a) -> None:
    body = D.load_frozen()
    with guard("train", "search", "◎ を固定した馬券のパターンの閾値・券種を訓練（2019〜22）だけで選ぶ"):
        out = D.build_period("train", body)
        for s, rows in out.items():
            D.save_rows(rows, rows_path("train", s))
            print(s, rows.counts, len(rows.meta), flush=True)


def _select_setting(rows: D.Rows, coefs: dict, workers: int, with_cv: bool = True) -> dict:
    pp = S.Prep(rows, coefs)
    A = S.stage_a(pp, workers)
    se = S.Searcher(pp, A, workers)
    sel = se.run(TRAIN_YEARS)
    out = {"counts": rows.counts, "n_tried": sel["n_tried"], "best": sel["best"], "pairs": sel["pairs"][:20],
           "per": {sid: (None if v is None else {**v, "ys": se.ev(v["final"])["ys"]}) for sid, v in sel["per"].items()}}
    if with_cv:
        out["rc"] = S.reality_check(A)
        c = S.cv(se, TRAIN_YEARS)
        out["cv"] = c
    return out


def cmd_select(a) -> None:
    body = D.load_frozen()
    if SELECTION.exists() and not a.force:
        raise SystemExit(f"{SELECTION} がすでにあります（選び直すなら --force。評価を流した後は選び直さない）")
    if RESULT.exists() or seen_after_train():
        raise SystemExit(f"評価の結果がすでにあります（{seen_after_train()}）。評価を見た後に選び直さない")
    with guard("train", "search", "◎ を固定した馬券のパターンの閾値・券種を訓練（2019〜22）だけで選ぶ（想定オッズの換算の当てはめを含む）"):
        frames = D.frames_of(TRAIN_YEARS)
        rows_all = {s: D.load_rows(rows_path("train", s)) for s in D.SETTINGS}
        ids = set(rows_all["all"].meta["race_id"]) | set(rows_all["a2"].meta["race_id"])
        fl_ids = set()
        from src.scripts.data import bet_cycles as bc

        fl = bc.load_runners(TRAIN_YEARS)
        fl_ids = set(fl["race_id"])
        waku = D.wakuren_payoffs(fl_ids | ids, TRAIN_YEARS)
        wt = OM.winners_table(TRAIN_YEARS, frames, waku)
        coefs = OM.fit(wt)
        cmp_ = OM.compare_original(wt)
        print("換算", json.dumps({k: (v["n"], round(v["rmse_log"], 3)) for k, v in coefs.items()}, ensure_ascii=False), flush=True)
        orig = OM.original_coefs()
        orig.setdefault("fukusho", coefs["fukusho"])
        orig.setdefault("wakuren", coefs["wakuren"])
        res = {"frozen_r2": str(D.FROZEN.relative_to(D.ROOT)), "frozen_r2_hash": body["hash"], "train_years": list(TRAIN_YEARS),
               "odds_map": {"coefs": coefs, "original_on_train": cmp_,
                            "note": "P_hat = 確定の単勝オッズからの Harville。log(払戻/100) = poly2(log(1/P_hat))、当たった組だけで当てはめ。判定は T-60 の再現の単勝オッズからの P_hat に当てる"},
               "grid": {"P_MIN": S.P_MIN, "RATIO_MIN": S.RATIO_MIN, "DEV_MIN": S.DEV_MIN, "ODDS_MIN": S.ODDS_MIN, "ODDS_MAX": S.ODDS_MAX,
                        "EV_MIN": S.EV_MIN, "STAKES": S.STAKES, "CAPS": S.CAPS, "SYNTH_MIN": S.SYNTH_MIN, "REV_MIN": S.REV_MIN,
                        "KELLY": [S.KELLY_BANK, S.KELLY_FRAC, S.KELLY_CAP], "MIN_BETS_PER_YEAR": S.MIN_BETS_PER_YEAR,
                        "structs": [{k: v for k, v in s.items()} for s in S.STRUCTS], "top_pairs": S.TOP_PAIRS,
                        "mandated_pairs": S.MANDATED_PAIRS}, "settings": {}}
        for s in D.SETTINGS:
            print("設定", s, _now(), flush=True)
            res["settings"][s] = _select_setting(rows_all[s], coefs, a.workers)
            print("元の換算で", s, _now(), flush=True)
            alt = _select_setting(rows_all[s], orig, a.workers, with_cv=False)
            res["settings"][s]["with_original_odds_map"] = {
                sid: {"same_final": (v or {}).get("final") == (res["settings"][s]["per"].get(sid) or {}).get("final"),
                      "final": (v or {}).get("final"), "score": (v or {}).get("score")} for sid, v in alt["per"].items()}
            res["settings"][s]["with_original_odds_map_best"] = alt["best"]
    res = json.loads(json.dumps(res, default=_js))
    res["hash"] = _hash(res)
    res["made_at"] = _now()
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_js) + "\n", encoding="utf-8")
    print("書いた", SELECTION, res["hash"])


BASE_TAN = {"s": "tan", "f": None, "sh": [0, 0], "ev": None, "stake": "equal", "skip": {}}      # 素の A2（◎ の単勝 1 点・均等）。比べる基準
BASE_FUKU = {"s": "fuku", "f": None, "sh": [0, 0], "ev": None, "stake": "equal", "skip": {}}
CAP_UI = 20     # UI の既定の上限（1 レース 2,000 円）


def _final_specs(st: dict) -> dict[str, list[dict]]:
    """表に出す規則: 基準（素の単勝・複勝）、組み方ごとの最良（段 B 後）、組み合わせの最良と決めておいた組、
    フォーメーションの最良に UI の上限（20 点）の見送りを足した副の版（選び直しではなく、決まった変換）。"""
    out = {"base_tan": [BASE_TAN], "base_fuku": [BASE_FUKU]}
    out.update({sid: [v["final"]] for sid, v in st["per"].items() if v})
    if st["pairs"]:
        p = st["pairs"][0]["pair"]
        out["pair:" + "+".join(p)] = [st["per"][p[0]]["final"], st["per"][p[1]]["final"]]
    for p in S.MANDATED_PAIRS:
        key = "pair:" + "+".join(p)
        if key not in out and st["per"].get(p[0]) and st["per"].get(p[1]):
            out[key] = [st["per"][p[0]]["final"], st["per"][p[1]]["final"]]
    for sid in ("trio_f", "tri_f"):
        v = st["per"].get(sid)
        if v and max_points(v["final"]) > CAP_UI:          # 選んだ形で上限が効くときだけ（効かなければ同じ規則の重複になる）
            sk = dict(v["final"].get("skip") or {})
            if sk.get("cap") is None or sk["cap"] > CAP_UI:
                sk["cap"] = CAP_UI
                out[f"{sid}_cap{CAP_UI}"] = [dict(v["final"], skip=sk)]
    return out


def max_points(spec: dict) -> int:
    """フォーメーションの最大の点数（相手がそろったとき）。"""
    n2, n3 = spec["sh"]
    if S.SBYID[spec["s"]]["kind"] == "t1_form":
        return n2 * (n3 - 1)
    return n2 * (n3 - n2) + n2 * (n2 - 1) // 2


def primary_key(st: dict) -> str:
    b = st["best"]
    return ("pair:" + "+".join(b["pair"])) if b["kind"] == "pair" else b["sid"]


def _committed(path: Path) -> bool:
    rel = str(path.relative_to(D.ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=D.ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=D.ROOT).returncode == 0
    return tracked and clean


def cmd_prereg(a) -> None:
    """評価の前に、主の規則・基準・合格の条件を書いて固定する（評価の値を見て主を選び直さないため）。"""
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    if RESULT.exists() or seen_after_train() or PREREG.exists():
        raise SystemExit("評価の結果・事前登録がすでにあります（事前登録は評価の前に 1 回だけ）")
    reg = {"selection_hash": sel["hash"], "registered_at": _now(),
           "primary": {s: primary_key(sel["settings"][s]) for s in D.SETTINGS},
           "baseline": "base_tan（素の A2: ◎ の単勝 1 点・均等）",
           "criteria": {"pass": "評価 2023〜24 で、主の規則の回収率 − 基準の回収率（同じレース・同じブートストラップの重みの対の差）の 95% 区間の下限 > 0",
                        "direction": "差の点の値 > 0 だが区間の下限 ≤ 0",
                        "fail": "差の点の値 ≤ 0"},
           "rules": ["主は設定ごとに 1 つ（訓練の物差しの最良）。ほかの規則の評価の値は参考で、評価を見て主を替えない。",
                     "2025・2026 は結果を出すだけで、合否に使わない。",
                     "評価は 1 回だけ流す（台帳の confirm）。"]}
    PREREG.write_text(json.dumps(reg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("書いた", PREREG, json.dumps(reg["primary"], ensure_ascii=False))


def verdict(m: dict) -> str:
    if m.get("diff_lo") is not None and m["diff_lo"] > 0:
        return "合格（基準の単勝を上回ることを確かめた）"
    if (m.get("diff_roi") or 0) > 0:
        return "向きだけ（区間は 0 をまたぐ）"
    return "不合格（基準の単勝以下）"


def cmd_evaluate(a) -> None:
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    body_ = {k: v for k, v in sel.items() if k not in ("hash", "made_at")}
    if _hash(body_) != sel["hash"]:
        raise SystemExit("selection.json のハッシュが合いません（選んだ後に書き換えた）")
    reg = json.loads(PREREG.read_text(encoding="utf-8"))
    if reg["selection_hash"] != sel["hash"]:
        raise SystemExit("事前登録が今の selection.json と違います")
    if sel["frozen_r2_hash"] != D.load_frozen()["hash"]:
        raise SystemExit("A2 の凍結が選んだときと違います")
    for f in (SELECTION, PREREG, Path(__file__).resolve(), Path(S.__file__).resolve(), Path(D.__file__).resolve(), Path(OM.__file__).resolve()):
        if not _committed(f):
            raise SystemExit(f"{f.relative_to(D.ROOT)} を commit してから評価する（結果を見る前に固定する）")
    if (RESULT.exists() or seen_after_train()) and not a.periods:
        raise SystemExit(f"評価はすでに流しています（1 回だけ）: {seen_after_train()}")
    body = D.load_frozen()
    coefs = sel["odds_map"]["coefs"]
    specs = {s: _final_specs(sel["settings"][s]) for s in D.SETTINGS}
    out = {"made_at": _now(), "selection_hash": sel["hash"], "prereg": reg, "periods": {}, "ledger": {}}
    if RESULT.exists():
        out = json.loads(RESULT.read_text(encoding="utf-8"))
        if out.get("selection_hash") != sel["hash"]:
            raise SystemExit("前回の結果が別の版の選択のものです")
    names = a.periods.split(",") if a.periods else ["train", "eval", "test", "main2026", "pre2026"]
    reasons = {"train": "選んだ規則の訓練の当てはめの値（選んだ期間）",
               "eval": "選んだ規則の評価（2023〜24）。事前登録の主の規則の合否を 1 回だけ",
               "test": "結果を出すだけ。選び直しに使わない（2025 は A2 の test。第 1 回・第 2 回で既に見た期間）",
               "main2026": "結果を出すだけ。選び直しに使わない（2026 の主の窓。A2 の事前登録で eval_only）",
               "pre2026": "結果を出すだけ。本物の T-60 のオッズで判定した参考（再現の楽観の大きさを見る）"}
    for nm in names:
        done = [x for x in seen_after_train() if x.startswith(f"bet_patterns {nm}（")]
        if nm in out["periods"] or done:
            raise SystemExit(f"期間 {nm} はすでに流しています（1 回だけ）: {done}")
        ver = "search" if nm == "train" else sel["hash"]
        with guard(nm, ver, reasons[nm], frozen=[SELECTION, PREREG], purpose="reproduce" if nm == "train" else None) as tok:
            if nm == "train":
                rows = {s: D.load_rows(rows_path("train", s)) for s in D.SETTINGS}
            else:
                rows = D.build_period(nm, body)
            tok.check_race_ids(set().union(*[set(r.meta["race_id"]) for r in rows.values()]), f"bet_patterns {nm}")
            per = {}
            for s in D.SETTINGS:
                pp = S.Prep(rows[s], coefs)
                per[s] = {"counts": rows[s].counts, "rules": {k: S.period_metrics(pp, v, base=[BASE_TAN]) for k, v in specs[s].items()}}
                pk = reg["primary"][s]
                per[s]["primary"] = {"key": pk, "verdict": verdict(per[s]["rules"][pk]) if nm == "eval" else "参考（合否は評価だけ）"}
            out["periods"][nm] = per
            out["ledger"][nm] = tok.summary()
        print(nm, json.dumps({s: {"primary": per[s]["primary"], "roi": {k: round(v["roi"] or 0, 3) for k, v in per[s]["rules"].items()}} for s in per},
                             ensure_ascii=False), flush=True)
        out["made_at"] = _now()
        RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_js) + "\n", encoding="utf-8")
    # A2（素の単勝 ◎・設定 a2）の記録との照合（行の作りの検算）
    a2 = json.loads(A2_TABLE.read_text(encoding="utf-8"))["periods"]
    chk = {}
    for nm, key in (("train", "train"), ("eval", "eval"), ("test", "test2025"), ("main2026", "main2026")):
        if nm in out["periods"] and key in a2:
            m = out["periods"][nm]["a2"]["rules"]["base_tan"]
            r = a2[key]["metrics"]
            chk[nm] = {"roi": m["roi"], "roi_record": r["roi"], "bet_races": m["bet_races"], "bet_races_record": r["bet_races"],
                       "same": bool(abs(m["roi"] - r["roi"]) < 1e-6 and round(m["bet_races"]) == r["bet_races"])}
    out["a2_reproduction"] = chk
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_js) + "\n", encoding="utf-8")
    print("A2 の照合", json.dumps(chk, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("prereg")
    s = sub.add_parser("select")
    s.add_argument("--workers", type=int, default=S.cpu_workers())
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("evaluate")
    s.add_argument("--periods", default="", help="一部の期間だけ（落ちた後の続き。流した期間は流し直さない）")
    a = ap.parse_args()
    {"build": cmd_build, "select": cmd_select, "prereg": cmd_prereg, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
