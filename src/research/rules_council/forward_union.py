"""RC-2 の和集合（config/rules_council/forward_union_rc2.json）の前向きの台帳（1 号の決定「RC-2 の結果の判断と RC-3 の決定」A・RC-6 の 4）。

- build  : 2026 年の表で graded/catalog の水準を作る（course/v3 の _frame_2026 と同じ作り: folklore_build → graded.features.build → CR.frame_and_levels）。
           置き場は <COURSE_OUT>/rules_council/fwd（v3 の表は触らない）。2026 年の結果は、前の走の履歴の特徴量としてだけ使う（台帳 purpose eval_only）。
- check  : 作りの確かめ（凍結の日より前の 2026 年の行で、欠けの率・水準の分布。着順・払戻は読まない）。
- ledger : 凍結の後（2026-11-28〜）のレースの判定と精算を、bet_forward_ledger と揃えた列で書く（台帳 purpose forward）。
           判定: 本物の T-60（pre_odds.parquet の t60）で 10 倍未満の馬のうち、和集合の水準のどれかに当たるか。精算: 確定の単勝の払戻（load_payoffs）。
           主の物差し: 同じ期間の一桁の中での『和集合の中の市場比 ÷ 外の市場比』（市場比 = 勝ち ÷ Σ 確定の q。凍結の定義）。副: 中の市場比 ÷ 帯の比。
           線（凍結のとおり）: 1,500 点で主の比 < 0.98・4,000 点で < 1.00 なら止める。満期 7,000 点で 主の比 > 1.00 かつ片側 p ≤ 0.05。回収率の主張はしない。
費用 0 円（手元の表だけ。GCS は読まない）。土日の 07:30〜17:00 は止める（重い計算）。
使い方: KEIBA_SESSION=rules-council python -m src.research.rules_council.forward_union build|check|ledger（nice -n 19）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E
from src.research.rules_council import rc_frame as F

ROOT = F.CR.REPO
FWD = F.OUTW / "fwd"
UNION = ROOT / "config/rules_council/forward_union_rc2.json"
PRE = ROOT / "data/local/research/persona1/pre_odds.parquet"
LEDGER_OUT = ROOT / "docs/operations/retrain_trials/rules_council_forward_union.json"
COUNT_FROM = "2026-11-28"      # 凍結の JSON は 2026-11-24 23:24 JST に commit。翌日から数えるが、最初の JRA の開催日が 11/28


def env26() -> dict:
    return F.CR.env(2026) | {"FOLKLORE_OUT": str(FWD / "master"), "GRADED_OUT": str(FWD)}


def cmd_build() -> None:
    E.check_stop()
    _frozen_ok()
    from src.ops import data_ledger as DL
    FWD.mkdir(parents=True, exist_ok=True)
    with DL.guard(tool="rules_council.forward_union:build", family="rules_council", version="fwd_union_build", period=("2026-01-01", None),
                  purpose="eval_only", reads=("着順（前の走の履歴の特徴量として）", "払戻（with_returns が読む。出す列には使わない）"),
                  reason="和集合の前向きの台帳のための 2026 年の水準の作り（着順は前の走の履歴としてだけ。台帳の判定・精算はしない）",
                  session="rules-council") as tok:
        e = env26()
        subprocess.run([sys.executable, "-m", "src.scripts.data.folklore_build"], env=e, cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "src.research.rules_council.forward_union", "_build_inner"], env=e, cwd=ROOT, check=True)
        tok.finish(result_path=FWD / "levels_2026.parquet")


def _build_inner() -> None:
    from src.research.graded import features as GF
    from src.research.graded.run import with_returns
    d, L = F.CR.frame_and_levels(with_returns(GF.build(out=None)))
    keep = (pd.to_numeric(d["year"]) == 2026).to_numpy()
    cols = ["race_id", "horse_id", "date", "year", "cls", "surf", "hn"]
    d = d.loc[keep, [c for c in cols if c in d.columns]].reset_index(drop=True)
    L = L[keep].reset_index(drop=True)
    d.to_parquet(FWD / "frame_2026.parquet", index=False)
    L.to_parquet(FWD / "levels_2026.parquet", index=False)
    print("2026 rows", len(d), flush=True)


def _union_mask(L: pd.DataFrame) -> np.ndarray:
    U = json.loads(UNION.read_text(encoding="utf-8"))
    m = np.zeros(len(L), bool)
    for lv in U["levels"]:
        t, v = lv.split("=", 1)
        if t not in L.columns:
            raise SystemExit(f"水準の型 {t} が 2026 年の表に無い")
        m |= (L[t].astype("string") == v).fillna(False).to_numpy(dtype=bool)
    return m


def cmd_check() -> dict:
    """作りの確かめ: 凍結の日より前の 2026 年の行で、和集合の水準に当たる割合と、型ごとの欠けの率（2024 年の行と比べる）。着順・払戻は読まない。"""
    from src.research.graded import model as M
    d = pd.read_parquet(FWD / "frame_2026.parquet")
    L = pd.read_parquet(FWD / "levels_2026.parquet")
    pre_freeze = (d["date"].astype(str) < COUNT_FROM).to_numpy()
    U = json.loads(UNION.read_text(encoding="utf-8"))
    types = sorted({lv.split("=", 1)[0] for lv in U["levels"]})
    L24 = pd.read_parquet(F.OUTW / "full_levels.parquet", columns=types)
    y24 = pd.read_parquet(F.OUTW / "full.parquet", columns=["year"])["year"].astype(int).to_numpy() == 2024
    out = {"rows_2026_pre_freeze": int(pre_freeze.sum()), "union_share_2026": float(_union_mask(L[pre_freeze]).mean()), "missing_rate": {}}
    no = set(M.NO_INFO)
    for t in types:
        a = L.loc[pre_freeze, t].astype("string").isin(no).mean()
        b = L24.loc[y24, t].astype("string").isin(no).mean()
        out["missing_rate"][t] = {"2026": float(a), "2024": float(b)}
    (FWD / "check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


STATE = FWD / "ledger_state.json"
LOOK_DAYS = 28


def _frozen_ok() -> None:
    """凍結の JSON の frozen_sha256 と、この道具が commit 済みで変わっていないことを確かめる（rc2_real の clean_or_stop に当たる守り）。"""
    import hashlib
    U = json.loads(UNION.read_text(encoding="utf-8"))
    bad = [f for f, h in (U.get("frozen_sha256") or {}).items() if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    if bad:
        raise SystemExit(f"凍結の後に変わったファイルがある（水準の定義がずれる）: {bad}")
    st = subprocess.run(["git", "status", "--porcelain", "--", "src/research/rules_council/forward_union.py", str(UNION)],
                        capture_output=True, text=True, cwd=ROOT).stdout.strip()
    if st:
        raise SystemExit(f"道具か凍結の JSON に commit していない変更がある:\n{st}")


def _band_table() -> pd.DataFrame:
    """副の物差しの帯の比: 探す側 70%（2016〜24）・再現 T-60 < 10 の、確定オッズの帯 × クラス群 の 勝ち ÷ q（rc2_real と同じ engine.band_ratio）。"""
    P = E.load_pop(0.0, 10.0, outcome_until=None)
    d = P.d
    s = E.search70(d["kaisai"].to_numpy())
    rho = E.band_ratio(d, s, np.where(s, d["win"].to_numpy(float), 0.0))
    return pd.DataFrame({"obin": d["obin"].to_numpy(), "cgrp": d["cgrp"].to_numpy(), "rho": rho}).drop_duplicates(["obin", "cgrp"])


def cmd_ledger(force_look: bool = False) -> dict:
    """凍結の後のレースの判定と精算（列は bet_forward_ledger と揃える: race_id・date・horse_number・as_of・odds_phase・オッズの出どころ・判定・精算）。
    主の物差し: 同じ期間の一桁（本物の T-60 < 10 倍）の中での『和集合の中の市場比 ÷ 外の市場比』。市場比 = 勝ち ÷ Σ 確定の q（凍結の定義）。
    副: 中の市場比 ÷ 帯の比（探す側 70% の 確定オッズの帯 × クラス群）。
    止める線は、累計が 1,500 点目・4,000 点目に達した時点の比で 1 回だけ決め、状態に残す。満期（7,000 点）は 主の比 > 1.00 かつ
    片側 p ≤ 0.05（開催で束ねたブートストラップ 2,000 回で 主の比 ≤ 1 の割合）。比を見るのは月 1 回まで（--force-look は満期・線の時点だけ）。"""
    E.check_stop()
    _frozen_ok()
    from src.config.data_paths import TABLES_DIR
    from src.ops import data_ledger as DL
    from src.scripts.data.bet_cycles import combo_key, load_payoffs
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    with DL.guard(tool="rules_council.forward_union:ledger", family="rules_council", version="fwd_union_v1", period=(COUNT_FROM, None),
                  purpose="forward", reads=("着順", "払戻", "T-60 のオッズ", "確定オッズ"), frozen_paths=[UNION, ROOT / "src/research/rules_council/forward_union.py"],
                  reason="RC-2 の和集合の前向きの記録（凍結の後のレースだけ。回収率の主張はしない）", session="rules-council") as tok:
        d = pd.read_parquet(FWD / "frame_2026.parquet")
        L = pd.read_parquet(FWD / "levels_2026.parquet")
        keep = (d["date"].astype(str) >= COUNT_FROM).to_numpy()
        last_date = str(d["date"].max())
        d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True)
        d["in_union"] = _union_mask(L)
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
        rids = set(d["race_id"])
        pre = pd.read_parquet(PRE)
        pre = pre[(pre["phase"] == "t60") & pre["race_id"].astype(str).isin(rids)][["race_id", "horse_number", "win_odds", "snapshot_at"]].copy()
        pre["race_id"] = pre["race_id"].astype(str)
        pre["horse_number"] = pd.to_numeric(pre["horse_number"], errors="coerce")
        fl = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id", "finish_position", "odds", "date"])
        fl["race_id"], fl["horse_id"] = fl["race_id"].astype(str), fl["horse_id"].astype(str)
        fl = fl[fl["race_id"].isin(rids)].copy()
        fl["date"] = pd.to_datetime(fl["date"]).dt.strftime("%Y-%m-%d")
        tok.check_dates(fl["date"])
        x = d.drop(columns=["date"]).merge(fl, on=["race_id", "horse_id"], how="inner").merge(pre, on=["race_id", "horse_number"], how="left")
        n_races0 = int(x["race_id"].nunique())
        if n_races0 == 0:                                                   # まだ数え始めの後の結果が無い
            res = {"count_from": COUNT_FROM, "races_with_results": 0, "bets_in_union": 0, "status": "満期前（まだ対象のレースが無い）",
                   "frame_last_date": last_date, "note": "2026 年の水準の表（build）を、開催日の後に作り直してから流す"}
            LEDGER_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            tok.finish(result_path=LEDGER_OUT)
            print(json.dumps(res, ensure_ascii=False, indent=1))
            return res
        ok = x.groupby("race_id")["win_odds"].transform(lambda s: bool(s.notna().all() and (s > 0).all()))
        dropped = int(x.loc[~ok, "race_id"].nunique())
        x = x[ok].copy()
        x["odds"] = pd.to_numeric(x["odds"], errors="coerce")
        inv = 1.0 / x["odds"].where(x["odds"] > 0)
        x["q"] = inv / inv.groupby(x["race_id"]).transform("sum")          # 確定の q（凍結の定義）
        x = x[x["win_odds"] < 10].copy()                                   # 判定: 本物の T-60 < 10 倍
        fin = pd.to_numeric(x["finish_position"], errors="coerce")
        x["win"] = (fin == 1).astype(float)
        pay = load_payoffs(set(x["race_id"]))
        x["payout"] = [float((pay.get(r, {}).get("tansho", {}) or {}).get(combo_key("tansho", [str(int(h))]), 0)) / 100 if w else 0.0
                       for r, h, w in zip(x["race_id"], x["horse_number"], x["win"])]
        x["kaisai"] = x["race_id"].str[:8]
        cls = pd.to_numeric(x["cls"], errors="coerce").fillna(-1).astype(int).to_numpy()
        x["cgrp"] = np.select([cls <= 1, cls == 2, cls == 3, cls == 4], [0, 1, 2, 3], 4)
        x["obin"] = F.odds_bin(x["odds"].to_numpy(float))
        x = x.merge(_band_table(), on=["obin", "cgrp"], how="left")
        x = x.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)
        rows = pd.DataFrame({"race_id": x["race_id"], "date": x["date"], "horse_number": x["horse_number"], "horse_id": x["horse_id"],
                             "as_of": x["snapshot_at"], "odds_phase": "t60", "odds_source": "data/local/research/persona1/pre_odds.parquet",
                             "decision": np.where(x["in_union"], "in_union", "out"), "q_final": x["q"], "odds_decision": x["win_odds"],
                             "odds_final": x["odds"], "settle_win": x["win"], "settle_payout_per_1": x["payout"],
                             "payout_drift": x["odds"] / x["win_odds"]})
        rows.to_parquet(FWD / "ledger_rows.parquet", index=False)
        inn = x[x["in_union"]]
        n = int(len(inn))

        def ratio(df):
            a, b = df[df["in_union"]], df[~df["in_union"]]
            mi = a["win"].sum() / max(a["q"].sum(), 1e-9)
            mo = b["win"].sum() / max(b["q"].sum(), 1e-9)
            return float(mi / mo) if mo > 0 else None
        # 止める線: 1,500 点目・4,000 点目に達した時点の比を 1 回だけ記録する（状態に残す）
        cum = np.cumsum(x["in_union"].to_numpy())
        for k, line in ((1500, 0.98), (4000, 1.00)):
            key = f"at_{k}"
            if key not in state and n >= k:
                cut = int(np.searchsorted(cum, k)) + 1
                r = ratio(x.iloc[:cut])
                state[key] = {"ratio": r, "date": str(x["date"].iloc[cut - 1]), "stop": bool(r is not None and r < line)}
        res = {"count_from": COUNT_FROM, "races_with_results": n_races0, "races_dropped_no_t60": dropped, "races": int(x["race_id"].nunique()),
               "fav_runs": int(len(x)), "bets_in_union": n, "lines_state": {k: v for k, v in state.items() if k.startswith("at_")}}
        last = state.get("last_look")
        look_due = (last is None) or ((pd.Timestamp.now(tz="Asia/Tokyo") - pd.Timestamp(last)).days >= LOOK_DAYS)
        if look_due or force_look:
            mi = float(inn["win"].sum() / max(inn["q"].sum(), 1e-9))
            res |= {"main_ratio": ratio(x), "mr_in": mi, "sub_ratio_vs_band": float(inn["win"].sum() / max((inn["rho"] * inn["q"]).sum(), 1e-9))}
            if n >= 7000:
                ks = x["kaisai"].unique()
                rb = np.random.default_rng(20261124)
                g = {k: x[x["kaisai"] == k] for k in ks}
                v = [ratio(pd.concat([g[k] for k in rb.choice(ks, size=len(ks))])) for _ in range(2000)]
                v = [t for t in v if t is not None]
                p = (np.sum(np.array(v) <= 1.0) + 1) / (len(v) + 1)
                res["mature"] = {"p_one_sided": float(p), "pass": bool(res["main_ratio"] > 1.0 and p <= 0.05)}
            state["last_look"] = pd.Timestamp.now(tz="Asia/Tokyo").isoformat()
        else:
            res["note_look"] = f"比は月 1 回まで（前回 {last}）。点数だけ記録した"
        res["status"] = ("止めた" if any(v.get("stop") for v in res["lines_state"].values()) else ("満期" if n >= 7000 else "満期前"))
        res["note"] = "回収率の主張はしない。比を見るのは月 1 回まで"
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        LEDGER_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=LEDGER_OUT)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "ledger", "_build_inner"])
    ap.add_argument("--force-look", action="store_true", help="線の時点・満期だけ。月 1 回の決まりの外で比を見る")
    a = ap.parse_args()
    if a.cmd == "ledger":
        cmd_ledger(a.force_look)
    else:
        {"build": cmd_build, "check": cmd_check, "_build_inner": _build_inner}[a.cmd]()


if __name__ == "__main__":
    main()
