"""払戻・オッズデータの異常監査（読み取り専用。データは書き換えない）。

馬券最適化の前に、手元の払戻（race_result_flat.payoff を整形した betting/payoff.parquet と payoff_fix）、
確定オッズ（odds_win_place / odds_pair / race_trifecta_odds キャッシュ）、race_result_flat の odds を検査する。

    python -m src.scripts.data.audit_payoff_odds [--years 2020-2026] [--out data/local/meta/odds_audit]

出力: <out>/audit.json（検査の全数値）, <out>/summary.md（年×券種の表と判定）, <out>/races_<検査名>.txt（異常レースの一覧）
検査: (1) キーと網羅 (2) 払戻の形式 (3) 単勝/複勝払戻とオッズの突き合わせ (4) 着順との整合（払戻の組合せが着順から決まる集合と一致するか）
(5) 単勝・複勝オッズの妥当性（Σ1/odds・人気の単調性・極端値）(6) 馬連/ワイド/馬単のオッズ（網羅・Σ1/odds・券種間）
(7) 3連単キャッシュ (8) モック混入 (9) 全組合せ購入の回収率が払戻率に一致するか。2026 年は検証専用（選択に使わない）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
BET = ROOT / "data/local/modeling/betting"
BET26 = ROOT / "data/local/modeling/betting_2026"
FIXDIR = ROOT / "data/local/meta/payoff_fix"
ODDS_FIX = ROOT / "data/local/meta/odds_fix/win_odds_fix.parquet"
CACHE = ROOT / "data/cache"
PAYBACK = {"tansho": 0.80, "fukusho": 0.80, "wakuren": 0.775, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrenpuku": 0.75, "sanrentan": 0.725}
BETS = list(PAYBACK)
ARITY_ORD = {"umatan": 2, "sanrentan": 3}
SAMPLE = 25


def log(msg: str) -> None:
    print(msg, flush=True)


# ───────── 着順から決まる「払戻が出るはずの組合せ」
def _perms_choose(group: list[int], r: int) -> list[tuple[int, ...]]:
    out = []
    for c in itertools.combinations(group, r):
        out.extend(itertools.permutations(c))
    return out


def expected_ordered(groups: list[list[int]], k: int) -> set[tuple[int, ...]]:
    """着順の同着グループ（上位から）に従って、上位 k 頭の並び（同着は並び替え自由）を全て返す。"""
    res: list[tuple[int, ...]] = [()]
    need = k
    for g in groups:
        if need <= 0:
            break
        take = min(len(g), need)
        opts = _perms_choose(g, take)
        res = [a + b for a in res for b in opts]
        need -= take
    return {t for t in res if len(t) == k}


def expected_sets(groups: list[list[int]], k: int) -> set[frozenset]:
    return {frozenset(t) for t in expected_ordered(groups, k)}


def place_set(groups: list[list[int]], k: int) -> set[int]:
    got: list[int] = []
    for g in groups:
        if len(got) >= k:
            break
        got.extend(g)
    return set(got)


def parse_combo(bt: str, s: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(x) for x in re.split(r"[>\-]", str(s)))
    except ValueError:
        return None


# ───────── データの読み込み
def load_year(y: int) -> dict:
    d = TABLES / str(y)
    rr = pd.read_parquet(d / "race_result_flat.parquet", columns=["race_id", "date", "horse_number", "bracket_number", "finish_position", "field_size", "odds", "popularity", "finish_time", "payoff"])
    ro = pd.read_parquet(d / "race_odds_flat.parquet", columns=["race_id", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "popularity"])
    pf = pd.read_parquet(d / "race_pair_odds_flat.parquet", columns=["race_id", "odds_type", "pair", "odds", "odds_min", "odds_max"])
    b = BET if y <= 2025 else BET26
    pre = str(y)
    wp = pd.read_parquet(b / "odds_win_place.parquet")
    wp = wp[wp.race_id.str.startswith(pre)]
    pair = pd.read_parquet(b / "odds_pair.parquet")
    pair = pair[pair.race_id.str.startswith(pre)]
    pay = pd.read_parquet(b / "payoff.parquet")
    pay = pay[pay.race_id.str.startswith(pre)]
    fx = FIXDIR / f"payoff_fix_{y}.parquet"
    fix = pd.read_parquet(fx) if fx.exists() else pd.DataFrame(columns=["race_id", "bet_type", "combo", "payout", "n_rows", "ambiguous", "resolved_by"])
    return {"rr": rr, "ro": ro, "pf": pf, "wp": wp, "pair": pair, "pay": pay, "fix": fix}


def apply_fix(pay: pd.DataFrame, fix: pd.DataFrame) -> pd.DataFrame:
    if not fix.empty:
        fix = fix[~fix["ambiguous"].astype(bool)]  # 曖昧な補正行は当てない（再取得・焼き込み済みの正しい行を上書きしないため）
    if fix.empty:
        return pay
    keys = fix[["race_id", "bet_type"]].drop_duplicates()
    m = pay.merge(keys.assign(_f=1), on=["race_id", "bet_type"], how="left")
    keep = pay[m["_f"].isna().to_numpy()]
    add = fix.copy()
    for c in ("popularity", "parsed"):
        if c not in add.columns:
            add[c] = np.nan if c == "popularity" else True
    return pd.concat([keep, add[keep.columns.intersection(add.columns)]], ignore_index=True)


# ───────── 検査
def check_keys(y: int, D: dict) -> dict:
    rr, wp, pay, pair = D["rr"], D["wp"], D["pay"], D["pair"]
    races = set(rr.race_id.unique())
    out: dict = {"races_result": len(races), "races_flat_odds": D["ro"].race_id.nunique(), "races_odds_table": wp.race_id.nunique(),
                 "races_payoff": pay.race_id.nunique()}
    out["bad_race_id_format"] = int((~pd.Series(list(races)).str.fullmatch(r"\d{12}")).sum())
    out["race_id_year_vs_date"] = int((rr.drop_duplicates("race_id").assign(_y=lambda d: d.race_id.str[:4], _d=lambda d: d.date.astype(str).str[:4]).query("_y != _d")).shape[0])
    out["bad_venue_code"] = int((~pd.Series(list(races)).str[4:6].isin([f"{i:02d}" for i in range(1, 11)])).sum())
    out["dup_result_rows"] = int(rr.duplicated(["race_id", "horse_number"]).sum())
    out["dup_odds_rows"] = int(wp.duplicated(["race_id", "horse_number"]).sum())
    out["dup_pair_rows"] = int(pair.duplicated(["race_id", "bet_type", "combo"]).sum())
    out["dup_payoff_rows"] = int(pay.duplicated(["race_id", "bet_type", "combo"]).sum())
    out["odds_races_not_in_result"] = len(set(wp.race_id) - races)
    out["result_races_without_odds"] = len(races - set(wp.race_id))
    out["payoff_races_not_in_result"] = len(set(pay.race_id) - races)
    out["result_races_without_payoff"] = len(races - set(pay.race_id))
    r1 = rr.drop_duplicates("race_id")
    nullpay = r1[r1.payoff.isna() | (r1.payoff.astype(str).str.strip().isin(["", "None", "{}", "null"]))].race_id
    out["result_races_with_null_payoff"] = int(len(nullpay))
    out["_null_payoff_races"] = sorted(nullpay)
    ft = rr.finish_time.astype(str).str.strip()
    dnf_t = rr[(rr.finish_position <= 0) & ~ft.isin(["", "None", "nan", "NaN"])]
    out["rows_finish_position_le0_with_finish_time"] = int(len(dnf_t))
    out["races_finish_position_le0_with_finish_time"] = int(dnf_t.race_id.nunique())
    out["_pos_le0_with_time_races"] = sorted(dnf_t.race_id.unique())
    out["rows_finish_position_le0"] = int((rr.finish_position <= 0).sum())
    big = pay[pay.payout > 10_000_000]
    out["_big_payouts"] = [f"{a}:{b}:{c}:{d}" for a, b, c, d in zip(big.race_id, big.bet_type, big.combo, big.payout)]
    out["_dup_payoff_races"] = sorted(pay[pay.duplicated(["race_id", "bet_type", "combo"], keep=False)].race_id.unique())
    cov = {}
    for bt in BETS:
        cov[bt] = {"payoff_races": int(pay[pay.bet_type == bt].race_id.nunique())}
    for bt in ("umaren", "wide", "umatan"):
        cov[bt]["odds_races"] = int(pair[pair.bet_type == bt].race_id.nunique())
    pft = D["pf"].groupby("odds_type").race_id.nunique().to_dict()
    cov["_flat_pair_races_by_type"] = {k: int(v) for k, v in pft.items()}
    out["coverage"] = cov
    return out


def check_payoff_format(y: int, D: dict) -> dict:
    pay, fix = D["pay"], D["fix"]
    out: dict = {"rows": len(pay)}
    out["not_multiple_of_10"] = int((pay.payout % 10 != 0).sum())
    out["lt_100"] = int((pay.payout < 100).sum())
    out["eq_70_special"] = int((pay.payout == 70).sum())
    out["unparsed_races"] = int(pay[~pay.parsed].race_id.nunique())
    out["ambiguous_races"] = int(pay[pay.ambiguous].race_id.nunique())
    out["by_bet_ambiguous_races"] = {bt: int(pay[(pay.bet_type == bt) & pay.ambiguous].race_id.nunique()) for bt in BETS}
    out["payout_gt_10m"] = int((pay.payout > 10_000_000).sum())
    if not fix.empty:
        a = pay.assign(k=lambda d: d.race_id + "|" + d.bet_type).groupby("k").apply(lambda g: tuple(sorted(zip(g.combo, g.payout))))
        b = fix.assign(k=lambda d: d.race_id + "|" + d.bet_type).groupby("k").apply(lambda g: tuple(sorted(zip(g.combo, g.payout))))
        common = a.index.intersection(b.index)
        diff = [k for k in common if a[k] != b[k]]
        out["fix_keys"] = int(len(b))
        out["fix_keys_in_payoff"] = int(len(common))
        out["fix_differs_from_payoff"] = len(diff)
        out["fix_differs_by_bet"] = dict(Counter(k.split("|")[1] for k in diff))
        out["fix_races_ambiguous_after_fix"] = int(fix[fix.ambiguous].race_id.nunique())
        out["_fix_diff_races"] = sorted({k.split("|")[0] for k in diff})
    return out


def check_payoff_vs_odds(y: int, D: dict, pay: pd.DataFrame) -> dict:
    wp = D["wp"]
    out: dict = {}
    t = pay[pay.bet_type == "tansho"].copy()
    t["hn"] = pd.to_numeric(t.combo, errors="coerce")
    m = t.merge(wp[["race_id", "horse_number", "win_odds"]], left_on=["race_id", "hn"], right_on=["race_id", "horse_number"], how="left")
    ok = m.win_odds.notna()
    diff = (m.payout / 100.0 - m.win_odds).abs()
    bad = ok & (diff > 0.1)
    out["tansho_rows"] = len(m)
    out["tansho_no_odds"] = int((~ok).sum())
    out["tansho_payout_vs_odds_mismatch"] = int(bad.sum())
    out["tansho_mismatch_races"] = sorted(m[bad].race_id.unique())[:SAMPLE]
    out["_tansho_mismatch_all"] = sorted(m[bad].race_id.unique())
    f = pay[pay.bet_type == "fukusho"].copy()
    f["hn"] = pd.to_numeric(f.combo, errors="coerce")
    mf = f.merge(wp[["race_id", "horse_number", "place_odds_min", "place_odds_max"]], left_on=["race_id", "hn"], right_on=["race_id", "horse_number"], how="left")
    okf = mf.place_odds_min.notna()
    p = mf.payout / 100.0
    badf = okf & ((p < mf.place_odds_min - 0.11) | (p > mf.place_odds_max + 0.11))
    out["fukusho_rows"] = len(mf)
    out["fukusho_no_odds"] = int((~okf).sum())
    out["fukusho_payout_outside_odds_range"] = int(badf.sum())
    out["_fukusho_mismatch_all"] = sorted(mf[badf].race_id.unique())
    # 複勝オッズ帯が 1 点に潰れた（min==max）行の割合（狭すぎるのも不自然）
    out["place_range_min_eq_max_rows"] = int((wp.place_odds_min == wp.place_odds_max).sum())
    return out


def check_finish_consistency(y: int, D: dict, pay: pd.DataFrame, label: str) -> dict:
    rr, wp = D["rr"], D["wp"]
    runners = wp[wp.win_odds > 0].groupby("race_id").size()
    rrp = rr[rr.finish_position > 0]
    groups_by_race: dict[str, list[list[int]]] = {}
    brk: dict[str, dict[int, int]] = {}
    for rid, g in rrp.groupby("race_id"):
        gg = g.sort_values("finish_position")
        groups_by_race[rid] = [list(x.horse_number) for _, x in gg.groupby("finish_position", sort=True)]
    for rid, g in rr.groupby("race_id"):
        brk[rid] = dict(zip(g.horse_number, g.bracket_number))
    nonpos = {rid: set(g.horse_number) for rid, g in rr[rr.finish_position <= 0].groupby("race_id")}
    fsize = rr.groupby("race_id").field_size.max().to_dict()
    obs: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    bad_parse = 0
    flag = pay.groupby(["race_id", "bet_type"]).agg(amb=("ambiguous", "max"), unp=("parsed", lambda s: (~s).any()))
    for rid, bt, combo in zip(pay.race_id.to_numpy(), pay.bet_type.to_numpy(), pay.combo.to_numpy()):
        c = parse_combo(bt, combo)
        if c is None:
            bad_parse += 1
            continue
        obs[(rid, bt)].append(c)
    res: dict = {b: Counter() for b in BETS}
    bad_races: dict[str, list[str]] = {b: [] for b in BETS}
    no_result = 0
    for rid, groups in groups_by_race.items():
        n = int(runners.get(rid, 0)) or len(brk.get(rid, {}))
        if not groups:
            no_result += 1
            continue
        exp: dict[str, set] = {}
        exp["tansho"] = {t[0] for t in expected_ordered(groups, 1)}
        kk = 3 if n >= 8 else (2 if n >= 5 else 0)
        exp["fukusho"] = place_set(groups, kk) if kk else set()
        nf = int(fsize.get(rid, n) or n)
        kk_alt = 3 if nf >= 8 else (2 if nf >= 5 else 0)
        alt_fuku = place_set(groups, kk_alt) if kk_alt else set()
        sets2 = expected_sets(groups, 2) if sum(len(g) for g in groups) >= 2 else set()
        exp["umaren"] = sets2
        exp["umatan"] = expected_ordered(groups, 2) if sets2 else set()
        # ワイドは『3 連複で当たる組の中の 2 頭』（3 着同着の A-B は当たらない）。共通関数 winning_combos に揃える
        from src.pipeline.betting import winning_combos as _W
        exp["wide"] = {frozenset(c) for c in _W.winning_combos(groups, "wide")}
        exp["sanrenpuku"] = expected_sets(groups, 3) if sum(len(g) for g in groups) >= 3 else set()
        exp["sanrentan"] = expected_ordered(groups, 3) if sum(len(g) for g in groups) >= 3 else set()
        bm = brk.get(rid, {})
        exp["wakuren"] = {tuple(sorted((bm.get(a, -1), bm.get(b, -1)))) for a, b in expected_ordered(groups, 2)} if sets2 else set()
        for bt in BETS:
            o = obs.get((rid, bt), [])
            if bt in ("tansho", "fukusho"):
                oset = {c[0] for c in o if len(c) == 1}
            elif bt in ("umatan", "sanrentan"):
                oset = {c for c in o}
            elif bt == "wakuren":
                oset = {tuple(sorted(c)) for c in o}
            else:
                oset = {frozenset(c) for c in o}
            e = exp[bt]
            fl = flag.loc[(rid, bt)] if (rid, bt) in flag.index else None
            if not oset and not e:
                res[bt]["ok_none"] += 1
            elif not oset and e:
                if bt == "wakuren" and n <= 8 or bt == "fukusho" and kk == 0:
                    res[bt]["ok_not_sold"] += 1
                else:
                    res[bt]["missing"] += 1
                    bad_races[bt].append(rid)
            elif oset == e:
                res[bt]["ok"] += 1
            elif bt == "fukusho" and oset == alt_fuku:
                res[bt]["ok_by_registered_field_size"] += 1
            else:
                horses_o = {h for c in o for h in (c if bt != "wakuren" else ())}
                horses_e = {h for c in e for h in (c if isinstance(c, (tuple, frozenset)) else (c,))} if bt != "wakuren" else set()
                extra = horses_o - horses_e
                if bt != "wakuren" and extra and extra <= nonpos.get(rid, set()):
                    tag = "mismatch_rr_position"
                else:
                    tag = "mismatch_ambiguous" if (fl is not None and fl.amb) else ("mismatch_unparsed" if (fl is not None and fl.unp) else "mismatch")
                res[bt][tag] += 1
                bad_races[bt].append(rid)
    out = {"label": label, "races_checked": len(groups_by_race), "races_without_result": no_result, "unparseable_combo_rows": bad_parse,
           "by_bet": {b: dict(res[b]) for b in BETS}, "_bad_races": {b: sorted(set(v)) for b, v in bad_races.items()}}
    return out


def check_win_place_odds(y: int, D: dict) -> dict:
    wp, rr = D["wp"], D["rr"]
    out: dict = {"rows": len(wp)}
    w = wp[wp.win_odds > 0].copy()
    s = w.groupby("race_id").win_odds.apply(lambda x: float((1.0 / np.clip(x, 1.0, None)).sum()))
    n = w.groupby("race_id").size()
    out["races"] = int(len(s))
    out["sum_inv_median"] = round(float(s.median()), 4)
    out["sum_inv_p01_p99"] = [round(float(s.quantile(0.01)), 3), round(float(s.quantile(0.99)), 3)]
    big = s[n.reindex(s.index) >= 4]
    out["sum_inv_outside_1.0_1.6"] = int(((big < 1.0) | (big > 1.6)).sum())
    out["sum_inv_outside_1.1_1.45"] = int(((big < 1.1) | (big > 1.45)).sum())
    out["_sum_inv_outside_races"] = sorted(big[(big < 1.0) | (big > 1.6)].index)
    out["win_odds_lt_1"] = int(((wp.win_odds > 0) & (wp.win_odds < 1.0)).sum())
    out["win_odds_ge_1000"] = int((wp.win_odds >= 1000).sum())
    out["win_odds_nan_or_zero"] = int((wp.win_odds.isna() | (wp.win_odds <= 0)).sum())
    ww = w.sort_values(["race_id", "popularity", "horse_number"])
    dd = ww.groupby("race_id").win_odds.diff()
    viol = ww[dd < -0.051]
    out["popularity_odds_nonmonotonic_races"] = int(viol.race_id.nunique())
    out["_nonmonotonic_races"] = sorted(viol.race_id.unique())
    out["popularity_min_not_1_races"] = int((w.groupby("race_id").popularity.min() != 1).sum())
    out["popularity_gt_n_rows"] = int((w.popularity > w.groupby("race_id").horse_number.transform("size")).sum())
    # 同じ値が 4 頭以上: 列のずれ・定数埋めの兆候
    rep = w.groupby(["race_id", "win_odds"]).size()
    out["races_with_4plus_identical_odds"] = int(rep[rep >= 4].reset_index().race_id.nunique())
    pm = wp[wp.place_odds_min.notna() & (wp.win_odds > 0)]
    out["place_min_gt_max"] = int((pm.place_odds_min > pm.place_odds_max).sum())
    out["place_min_lt_1"] = int((pm.place_odds_min < 1.0).sum())
    out["place_max_gt_win_plus_tol"] = int((pm.place_odds_max > pm.win_odds + 0.11).sum())
    out["place_missing_for_runner"] = int(((wp.win_odds > 0) & wp.place_odds_min.isna()).sum())
    kk = np.where(n.reindex(ps_idx := pm.groupby("race_id").size().index) >= 8, 3, np.where(n.reindex(ps_idx) >= 5, 2, 0))
    ps = pm.assign(m=lambda d: 1.0 / ((d.place_odds_min + d.place_odds_max) / 2)).groupby("race_id").m.sum().reindex(ps_idx)
    ratio = ps / (pd.Series(kk, index=ps_idx).replace(0, np.nan) / 0.8)
    out["place_sum_inv_ratio_median"] = round(float(ratio.median()), 3)
    out["place_sum_inv_ratio_outside_0.8_1.2"] = int(((ratio < 0.8) | (ratio > 1.2)).sum())
    # 確定オッズ表と race_result_flat.odds / 元の race_odds_flat の一致
    ro = D["ro"][["race_id", "horse_number", "win_odds"]].rename(columns={"win_odds": "flat_win"})
    m = wp.merge(ro, on=["race_id", "horse_number"], how="outer", indicator=True)
    out["built_vs_flat_only_built"] = int((m._merge == "left_only").sum())
    out["built_vs_flat_only_flat"] = int((m._merge == "right_only").sum())
    both = m[m._merge == "both"]
    out["built_vs_flat_win_mismatch"] = int(((both.win_odds - both.flat_win).abs() > 0.051).sum())
    q = rr[["race_id", "horse_number", "odds"]].merge(wp[["race_id", "horse_number", "win_odds"]], on=["race_id", "horse_number"], how="inner")
    q = q[q.win_odds > 0]
    bad = (q.odds - q.win_odds).abs() > (0.05 + 0.01 * q.win_odds)
    out["result_odds_vs_win_odds_mismatch_rows"] = int(bad.sum())
    badr = q[bad].race_id.unique()
    out["result_odds_mismatch_races"] = int(len(badr))
    fx = pd.read_parquet(ODDS_FIX) if ODDS_FIX.exists() else pd.DataFrame(columns=["race_id"])
    out["result_odds_mismatch_races_not_in_odds_fix"] = int(len(set(badr) - set(fx.race_id)))
    out["_result_odds_mismatch_races"] = sorted(badr)
    return out


def check_pair_odds(y: int, D: dict, pay: pd.DataFrame) -> dict:
    pair, wp = D["pair"], D["wp"]
    runners = wp[wp.win_odds > 0].groupby("race_id").horse_number.apply(set)
    n = runners.map(len)
    out: dict = {}
    for bt in ("umaren", "wide", "umatan"):
        p = pair[pair.bet_type == bt].copy()
        info: dict = {"rows": len(p), "races": int(p.race_id.nunique())}
        if p.empty:
            out[bt] = info
            continue
        parts = p.combo.str.split(r"[>\-]", regex=True)
        p["a"] = parts.str[0].astype(int)
        p["b"] = parts.str[1].astype(int)
        val = p.odds if bt != "wide" else (p.odds_min + p.odds_max) / 2
        p["v"] = val
        info["combo_not_ascending"] = int((p.a >= p.b).sum()) if bt != "umatan" else 0
        info["self_pair"] = int((p.a == p.b).sum())
        info["odds_nan"] = int(val.isna().sum())
        info["wide_min_gt_max"] = int((p.odds_min > p.odds_max).sum()) if bt == "wide" else 0
        info["odds_ge_100000"] = int((p.odds.fillna(0) >= 100000).sum())
        # 取消馬を含む組合せ（odds が 0 で保存される）と、出走馬だけの組合せを分ける
        rs = p.race_id.map(runners)
        has_nonrunner = pd.Series([isinstance(r, set) and (a not in r or b not in r) for r, a, b in zip(rs, p.a, p.b)], index=p.index)
        zero = p.v.fillna(0) <= 0
        info["rows_zero_odds"] = int(zero.sum())
        info["rows_nonrunner_pairs"] = int(has_nonrunner.sum())
        info["zero_odds_rows_not_explained_by_nonrunner"] = int((zero & ~has_nonrunner).sum())
        info["nonrunner_rows_with_positive_odds"] = int((has_nonrunner & ~zero).sum())
        info["races_with_zero_odds_rows"] = int(p[zero].race_id.nunique())
        q = p[~zero & ~has_nonrunner]
        info["runner_pair_odds_lt_1"] = int((q.v < 1.0).sum())
        cnt = q.groupby("race_id").size()
        nn = n.reindex(cnt.index)
        exp = nn * (nn - 1) / (1 if bt == "umatan" else 2)
        short = cnt[(cnt != exp) & nn.notna()]
        info["races_incomplete_pairs"] = int(len(short))
        info["races_without_runner_info"] = int(nn.isna().sum())
        info["_incomplete_races"] = sorted(short.index)
        s = (1.0 / q.v.clip(lower=1.0)).groupby(q.race_id).sum()
        s = s[nn.reindex(s.index) >= 5]
        med = float(s.median()) if len(s) else float("nan")
        info["sum_inv_median"] = round(med, 3)
        info["expected_sum_inv"] = round((3 if bt == "wide" else 1) / PAYBACK[bt], 3)
        out_idx = s[(abs(s - med) > 0.25 * med)].index if len(s) else []
        info["sum_inv_outlier_races_gt25pct"] = int(len(out_idx))
        info["_sum_inv_outlier_races"] = sorted(out_idx)
        if bt != "wide":
            qq = q.sort_values(["race_id", "popularity"])
            d = qq.groupby("race_id").odds.diff()
            info["popularity_odds_nonmonotonic_races"] = int(qq[d < -0.051].race_id.nunique())
        # 払戻（的中した組合せ）の金額 / 100 がオッズ表と一致するか
        pb = pay[pay.bet_type == bt]
        if len(pb):
            m = pb[["race_id", "combo", "payout"]].merge(p[["race_id", "combo", "odds", "odds_min", "odds_max"]], on=["race_id", "combo"], how="left")
            if bt == "wide":
                ok = m.odds_min.notna()
                lo, hi = m.odds_min - 0.11, m.odds_max + 0.11
                bad = ok & ((m.payout / 100.0 < lo) | (m.payout / 100.0 > hi))
            else:
                ok = m.odds.notna()
                bad = ok & ((m.payout / 100.0 - m.odds).abs() > 0.1 + 0.001 * m.odds)
            info["hit_rows"] = len(m)
            info["hit_rows_without_odds"] = int((~ok).sum())
            info["hit_payout_vs_odds_mismatch_rows"] = int(bad.sum())
            info["_hit_mismatch_races"] = sorted(m[bad].race_id.unique())
        out[bt] = info
    um, wi = pair[pair.bet_type == "umaren"], pair[pair.bet_type == "wide"]
    m = um[["race_id", "combo", "odds"]].merge(wi[["race_id", "combo", "odds_min", "odds_max"]], on=["race_id", "combo"])
    m = m[(m.odds > 0) & (m.odds_min > 0)]
    out["wide_min_gt_umaren_pairs"] = int((m.odds_min > m.odds + 0.11).sum())
    out["_wide_gt_umaren_races"] = sorted(m[m.odds_min > m.odds + 0.11].race_id.unique())
    ut = pair[(pair.bet_type == "umatan") & (pair.odds > 0)].copy()
    if len(ut):
        ab = ut.combo.str.split(">", regex=False)
        a1, a2 = ab.str[0].astype(int), ab.str[1].astype(int)
        ut["k"] = a1.clip(upper=a2).astype(str) + "-" + a1.clip(lower=a2).astype(str)
        um2 = um[um.odds > 0]
        mm = um2[["race_id", "combo", "odds"]].rename(columns={"combo": "k", "odds": "um"}).merge(ut[["race_id", "k", "odds"]], on=["race_id", "k"])
        out["umatan_lt_0.7_umaren_pairs"] = int((mm.odds < mm.um * 0.7).sum())
        out["umatan_pairs_compared"] = len(mm)
    return out


def check_roi(y: int, D: dict, pay: pd.DataFrame) -> dict:
    wp, rr = D["wp"], D["rr"]
    runners = wp[wp.win_odds > 0].groupby("race_id").size()
    brk = rr[rr.horse_number.isin([]) | True].merge(wp[wp.win_odds > 0][["race_id", "horse_number"]], on=["race_id", "horse_number"])
    bsz = brk.groupby(["race_id", "bracket_number"]).size().groupby("race_id").apply(lambda s: (len(s), int((s >= 2).sum())))
    bad_amb = set(pay[pay.ambiguous | ~pay.parsed].race_id)
    out: dict = {}
    for bt in BETS:
        pb = pay[(pay.bet_type == bt) & ~pay.race_id.isin(bad_amb)]
        rids = pb.race_id.unique()
        nn = runners.reindex(rids).dropna()
        if bt == "tansho":
            cost = nn.sum()
        elif bt == "fukusho":
            nn = nn[nn >= 5]
            cost = nn.sum()
            pb = pb[pb.race_id.isin(nn.index)]
        elif bt == "wakuren":
            nb = bsz.reindex(rids).dropna()
            cost = sum(b * (b - 1) / 2 + s for b, s in nb)
            pb = pb[pb.race_id.isin(nb.index)]
        elif bt in ("umaren", "wide"):
            cost = (nn * (nn - 1) / 2).sum()
        elif bt == "umatan":
            cost = (nn * (nn - 1)).sum()
        elif bt == "sanrenpuku":
            cost = (nn * (nn - 1) * (nn - 2) / 6).sum()
        else:
            cost = (nn * (nn - 1) * (nn - 2)).sum()
        keep_ids = set(nn.index) if bt not in ("wakuren",) else set(nb.index)
        pb = pb[pb.race_id.isin(keep_ids)]
        roi = float(pb.payout.sum() / (100.0 * cost)) if cost else float("nan")
        out[bt] = {"races": int(pb.race_id.nunique()), "roi_all_combos": round(roi, 4), "payback_ref": PAYBACK[bt], "diff": round(roi - PAYBACK[bt], 4)}
    # 1 番人気の単勝・複勝
    fav = wp[wp.popularity == 1][["race_id", "horse_number"]]
    for bt in ("tansho", "fukusho"):
        pb = pay[(pay.bet_type == bt) & ~pay.race_id.isin(bad_amb)].assign(hn=lambda d: pd.to_numeric(d.combo, errors="coerce"))
        fr = fav[fav.race_id.isin(set(pb.race_id) | set(pay[pay.bet_type == bt].race_id))]
        if bt == "fukusho":
            ok_r = runners[runners >= 5].index
            fr = fr[fr.race_id.isin(ok_r)]
        hit = fr.merge(pb, left_on=["race_id", "horse_number"], right_on=["race_id", "hn"], how="left")
        out[f"favorite_{bt}"] = {"bets": len(fr), "roi": round(float(hit.payout.fillna(0).sum() / (100.0 * max(len(fr), 1))), 4)}
    return out


def check_cache(runners_by_race: dict[str, int]) -> dict:
    out: dict = {}
    for cat in ("race_odds", "race_trifecta_odds"):
        d = CACHE / cat
        files = sorted(d.rglob("*.json")) if d.exists() else []
        out[cat] = {"files": len(files)}
        try:
            r = subprocess.run(["grep", "-rl", "dev_mock", str(d)], capture_output=True, text=True, timeout=300)
            out[cat]["files_with_dev_mock"] = len([x for x in r.stdout.splitlines() if x])
        except Exception as e:  # noqa: BLE001
            out[cat]["dev_mock_scan_error"] = str(e)[:80]
    # 3 連単キャッシュ
    d = CACHE / "race_trifecta_odds"
    stats: dict = defaultdict(lambda: Counter())
    sums: dict = defaultdict(list)
    bad: list[str] = []
    for f in sorted(d.rglob("*.json")) if d.exists() else []:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stats["_all"]["unreadable"] += 1
            continue
        rows = j.get("trifecta") or []
        y = f.parent.name
        if not rows:
            stats[y]["empty"] += 1
            continue
        stats[y]["races"] += 1
        keys = [(r[0], r[1], r[2]) for r in rows]
        if len(set(keys)) != len(keys):
            stats[y]["dup_rows"] += 1
            bad.append(f.stem)
        horses = sorted({h for k in keys for h in k})
        n = runners_by_race.get(f.stem) or len(horses)
        exp_rows = n * (n - 1) * (n - 2)
        if len(rows) != exp_rows:
            stats[y]["incomplete"] += 1
            stats[y]["incomplete_more_than_expected" if len(rows) > exp_rows else "incomplete_fewer_than_expected"] += 1
            bad.append(f.stem)
        if f.stem in runners_by_race and not set(horses) <= set(range(1, 19)):
            stats[y]["horse_number_out_of_range"] += 1
        odds = np.array([r[3] for r in rows], dtype=float)
        zero = odds <= 0
        stats[y]["rows_zero_odds"] += int(zero.sum())
        stats[y]["rows_odds_cap_999999"] += int((odds >= 999999).sum())
        if (odds[~zero] < 1.0).any():
            stats[y]["odds_lt_1"] += 1
        v = odds[~zero]
        sums[y].append(float((1.0 / np.clip(v, 1.0, None)).sum()))
        pr = sorted(((r[4], r[3]) for r in rows if r[3] > 0))
        po = np.array([o for _, o in pr])
        if len(po) > 1 and (np.diff(po) < -0.051 * np.maximum(po[:-1], 1) / 10).any():
            stats[y]["popularity_nonmonotonic"] += 1
    out["trifecta"] = {y: {**dict(c), "sum_inv_median": round(float(np.median(sums[y])), 3) if sums[y] else None,
                            "expected_sum_inv": round(1 / PAYBACK["sanrentan"], 3)} for y, c in stats.items()}
    out["_trifecta_bad_races"] = sorted(set(bad))
    return out


def trifecta_vs_payoff(pay26: pd.DataFrame) -> dict:
    d = CACHE / "race_trifecta_odds"
    if not d.exists():
        return {}
    want = pay26[pay26.bet_type == "sanrentan"].set_index("race_id")
    n_ok = n_bad = n_missing = 0
    bad_ids: list[str] = []
    for f in sorted(d.rglob("*.json")):
        rid = f.stem
        if rid not in want.index:
            continue
        try:
            rows = json.loads(f.read_text(encoding="utf-8")).get("trifecta") or []
        except (OSError, ValueError):
            continue
        tab = {(r[0], r[1], r[2]): r[3] for r in rows}
        w = want.loc[[rid]]
        for combo, pay in zip(w.combo, w.payout):
            c = parse_combo("sanrentan", combo)
            if c is None or c not in tab:
                n_missing += 1
                continue
            if abs(pay / 100.0 - tab[c]) <= 0.15 + 0.001 * tab[c]:
                n_ok += 1
            else:
                n_bad += 1
                bad_ids.append(rid)
    return {"hit_combos_match_payoff": n_ok, "hit_combos_mismatch": n_bad, "hit_combo_missing_in_cache": n_missing, "_mismatch_races": sorted(set(bad_ids))}


def verdict(rep: dict, years: list[int]) -> dict:
    v: dict = {}
    base = [y for y in years if y <= 2025]
    med = {bt: float(np.nanmedian([rep["years"][str(y)]["roi_fixed"][bt]["roi_all_combos"] for y in base])) if base else float("nan") for bt in BETS}
    for y in years:
        yr = rep["years"][str(y)]
        v[str(y)] = {}
        nres = max(yr["keys"]["races_result"], 1)
        for bt in BETS:
            fc = yr["finish_fixed"]["by_bet"][bt]
            checked = sum(fc.values()) or 1
            cov = sum(v for k, v in fc.items() if k.startswith("ok")) / checked
            bad = fc.get("mismatch", 0) + fc.get("missing", 0)
            rr_pos = fc.get("mismatch_rr_position", 0)
            bad_amb = fc.get("mismatch_ambiguous", 0) + fc.get("mismatch_unparsed", 0)
            roi = dict(yr["roi_fixed"][bt])
            roi["diff"] = round(roi["roi_all_combos"] - med[bt], 4)
            dev = abs(roi["diff"]) if roi["diff"] == roi["diff"] else 9
            raw_bad = yr["finish_raw"]["by_bet"][bt]
            raw_bad_n = raw_bad.get("mismatch", 0) + raw_bad.get("missing", 0) + raw_bad.get("mismatch_ambiguous", 0) + raw_bad.get("mismatch_unparsed", 0)
            fixed_bad_n = bad + bad_amb
            if cov < 0.97 or fixed_bad_n / checked > 0.02 or dev > 0.08:
                pj = "使えない"
            elif fixed_bad_n / checked > 0.003 or dev > 0.04 or cov < 0.995:
                pj = "要確認（除外すれば使える）"
            elif raw_bad_n > fixed_bad_n:
                pj = "修正後に使える（payoff_fix を当てる）"
            else:
                pj = "使える"
            if bt in ("tansho", "fukusho"):
                n_ = yr["keys"]["races_odds_table"] / nres
                oj = "使える" if n_ >= 0.995 and yr["win_place"]["sum_inv_outside_1.0_1.6"] / max(yr["win_place"]["races"], 1) < 0.005 else "要確認"
            elif bt in ("umaren", "wide", "umatan"):
                pr = yr["pair"][bt]
                incompl = pr.get("races_incomplete_pairs", 0) / max(pr.get("races", 1), 1)
                oj = "使える" if pr.get("races", 0) / nres >= 0.995 and incompl < 0.01 else "要確認"
            else:
                oj = "なし（オッズ未取得）"
            v[str(y)][bt] = {"payoff": pj, "odds": oj, "payoff_ok_share": round(cov, 4), "finish_mismatch_after_fix": fixed_bad_n, "finish_mismatch_raw": raw_bad_n, "mismatch_due_to_result_position": rr_pos, "roi_vs_yearmedian": roi["diff"]}
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2020-2026")
    ap.add_argument("--out", type=Path, default=ROOT / "data/local/meta/odds_audit")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-"))
    years = list(range(lo, hi + 1))
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"years": {}, "note": "2026 年は検証専用（選択に使わない）。読み取り専用の監査。"}
    lists: dict[str, set] = defaultdict(set)
    pay26 = None
    runners26: dict[str, int] = {}
    for y in years:
        log(f"== {y}")
        D = load_year(y)
        pay_raw = D["pay"]
        pay_fix = apply_fix(pay_raw, D["fix"])
        yr: dict = {}
        yr["keys"] = check_keys(y, D)
        yr["payoff_format"] = check_payoff_format(y, D)
        yr["payoff_vs_odds_raw"] = check_payoff_vs_odds(y, D, pay_raw)
        yr["payoff_vs_odds_fixed"] = check_payoff_vs_odds(y, D, pay_fix)
        yr["finish_raw"] = check_finish_consistency(y, D, pay_raw, "raw")
        yr["finish_fixed"] = check_finish_consistency(y, D, pay_fix, "payoff_fix 適用後")
        yr["win_place"] = check_win_place_odds(y, D)
        yr["pair"] = check_pair_odds(y, D, pay_fix)
        yr["roi_raw"] = check_roi(y, D, pay_raw)
        yr["roi_fixed"] = check_roi(y, D, pay_fix)
        for bt in BETS:
            lists[f"finish_fixed_{bt}"].update(yr["finish_fixed"]["_bad_races"][bt])
            lists[f"finish_raw_{bt}"].update(yr["finish_raw"]["_bad_races"][bt])
        lists["null_payoff_races"].update(yr["keys"]["_null_payoff_races"])
        lists["finish_position_le0_with_finish_time"].update(yr["keys"]["_pos_le0_with_time_races"])
        lists["big_payouts_over_10m_yen"].update(yr["keys"]["_big_payouts"])
        lists["dup_payoff_rows"].update(yr["keys"]["_dup_payoff_races"])
        lists["tansho_payout_vs_odds"].update(yr["payoff_vs_odds_fixed"]["_tansho_mismatch_all"])
        lists["fukusho_payout_vs_odds"].update(yr["payoff_vs_odds_fixed"]["_fukusho_mismatch_all"])
        lists["win_sum_inv_outside"].update(yr["win_place"]["_sum_inv_outside_races"])
        lists["win_popularity_nonmonotonic"].update(yr["win_place"]["_nonmonotonic_races"])
        lists["result_odds_mismatch"].update(yr["win_place"]["_result_odds_mismatch_races"])
        for bt in ("umaren", "wide", "umatan"):
            lists[f"pair_incomplete_{bt}"].update(yr["pair"][bt].get("_incomplete_races", []))
            lists[f"pair_sum_inv_outlier_{bt}"].update(yr["pair"][bt].get("_sum_inv_outlier_races", []))
            lists[f"pair_hit_payout_vs_odds_{bt}"].update(yr["pair"][bt].get("_hit_mismatch_races", []))
        lists["wide_gt_umaren"].update(yr["pair"].get("_wide_gt_umaren_races", []))
        lists["payoff_fix_differs"].update(yr["payoff_format"].get("_fix_diff_races", []))
        for sec in yr.values():
            if isinstance(sec, dict):
                for k in [k for k in sec if k.startswith("_")]:
                    sec.pop(k, None)
        for k in ("finish_raw", "finish_fixed"):
            yr[k].pop("_bad_races", None)
        for sub in ("payoff_vs_odds_raw", "payoff_vs_odds_fixed", "win_place", "payoff_format"):
            for k in [k for k in yr[sub] if k.startswith("_")]:
                yr[sub].pop(k, None)
        for bt in ("umaren", "wide", "umatan"):
            for k in [k for k in yr["pair"][bt] if k.startswith("_")]:
                yr["pair"][bt].pop(k, None)
        yr["pair"].pop("_wide_gt_umaren_races", None)
        rep["years"][str(y)] = yr
        if y == 2026:
            pay26 = pay_fix
            runners26 = D["wp"][D["wp"].win_odds > 0].groupby("race_id").size().to_dict()
    rep["cache"] = check_cache(runners26)
    rep["cache"]["trifecta_vs_payoff_2026"] = trifecta_vs_payoff(pay26) if pay26 is not None else {}
    for k in [k for k in rep["cache"] if k.startswith("_")]:
        lists["trifecta_cache_bad"].update(rep["cache"].pop(k))
    for k in [k for k in rep["cache"]["trifecta_vs_payoff_2026"] if k.startswith("_")]:
        lists["trifecta_vs_payoff_mismatch"].update(rep["cache"]["trifecta_vs_payoff_2026"].pop(k))
    rep["verdict"] = verdict(rep, years)
    (a.out / "audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k, v in lists.items():
        (a.out / f"races_{k}.txt").write_text("\n".join(sorted(v)) + ("\n" if v else ""), encoding="utf-8")
    (a.out / "summary.md").write_text(render_summary(rep, years, lists), encoding="utf-8")
    log(f"完了: {a.out}")
    return 0


def render_summary(rep: dict, years: list[int], lists: dict) -> str:
    L = ["# 払戻・オッズ監査の要約", ""]
    L.append("## 判定（払戻 / オッズ）")
    L.append("| 年 | " + " | ".join(BETS) + " |")
    L.append("|---|" + "---|" * len(BETS))
    for y in years:
        L.append(f"| {y} | " + " | ".join(f"{rep['verdict'][str(y)][b]['payoff']} / {rep['verdict'][str(y)][b]['odds']}" for b in BETS) + " |")
    L.append("")
    L.append("## 全組合せを等額で買ったときの回収率（払戻 fix 適用後）と、2020〜2025 年の中央値との差（払戻率そのものとは一致しない: 等額買いは穴馬の比重が大きく、払戻率より低くなる）")
    L.append("| 年 | " + " | ".join(BETS) + " |")
    L.append("|---|" + "---|" * len(BETS))
    for y in years:
        L.append(f"| {y} | " + " | ".join(f"{rep['years'][str(y)]['roi_fixed'][b]['roi_all_combos']:.3f}({rep['verdict'][str(y)][b]['roi_vs_yearmedian']:+.3f})" for b in BETS) + " |")
    L.append("")
    L.append("## 異常レース数（一覧は races_*.txt）")
    for k, v in sorted(lists.items()):
        L.append(f"- {k}: {len(v)}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
