"""馬券シミュレーション（バックテスト・強化学習の環境）用に、過去のオッズと払戻を整形して保存する。

入力は、既にエクスポート済みの flat テーブル（``data/page_reference/tables/<年>/``）:
  race_odds_flat（単勝・複勝）, race_pair_odds_flat（馬連・ワイド・馬単）, race_result_flat（着順と払戻 ``payoff`` の JSON 文字列）
出力（``data/local/modeling/betting/``）:
  odds_win_place.parquet / odds_pair.parquet / payoff.parquet / coverage.json

払戻は HTML の表を文字列にしたもので、複数行の券種（複勝・ワイド・同着）は数字と金額が区切り無しで連結されている
（例: 複勝 numbers=``458`` payout=``100230150``、ワイド payout=``4002401,020``）。金額は「3 桁、または桁区切りカンマ付き」の
トークンに分けられ、組み合わせは着順上位の馬番（候補）で分割できるので、候補と件数の両方が合う分割を採る。複数あれば ambiguous を立てる。

ここで作る「オッズ」は、2020〜2025 年については締切後に取得した確定オッズ（スナップショットなし）で、購入判断の時点の値ではない。
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

BET_KEYS = {"単勝": "tansho", "複勝": "fukusho", "枠連": "wakuren", "馬連": "umaren", "ワイド": "wide", "馬単": "umatan", "三連複": "sanrenpuku", "三連単": "sanrentan"}
ARITY = {"tansho": 1, "fukusho": 1, "wakuren": 2, "umaren": 2, "wide": 2, "umatan": 2, "sanrenpuku": 3, "sanrentan": 3}
ORDERED = {"umatan", "sanrentan"}
# JRA の払戻率（控除率の裏返し）。単勝・複勝 80%、枠連・馬連・ワイド 77.5%、馬単・三連複 75%、三連単 72.5%
PAYBACK = {"tansho": 0.80, "fukusho": 0.80, "wakuren": 0.775, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrenpuku": 0.75, "sanrentan": 0.725}


def _to_int(x) -> int | None:
    try:
        return int(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# ── 払戻の金額の分割
def split_payouts(s: str, k: int | None = None) -> list[list[int]]:
    """連結された金額の文字列を、有効なトークン（3 桁、または桁区切りカンマ付き）に分ける全ての分割。k を渡せばその個数のものだけ。"""
    s = re.sub(r"\s+", "", str(s or "")).replace("円", "")
    if not s:
        return []
    out: list[list[int]] = []
    tok = re.compile(r"(\d{3}|\d{1,3}(?:,\d{3})+)")

    def rec(i: int, acc: list[int]) -> None:
        if i == len(s):
            if k is None or len(acc) == k:
                out.append(list(acc))
            return
        if k is not None and len(acc) >= k:
            return
        # 取りうるトークンの長さをすべて試す（3 桁、カンマ付き "1,020"・"12,340" など）
        for end in range(i + 3, min(len(s), i + 12) + 1):
            piece = s[i:end]
            if tok.fullmatch(piece):
                v = int(piece.replace(",", ""))
                if v >= 100 and v % 10 == 0:
                    acc.append(v)
                    rec(end, acc)
                    acc.pop()

    rec(0, [])
    return out


# ── 馬番の組の分割
def _segment_run(run: str, candidates: set[int]) -> list[list[int]]:
    """数字だけの連なりを、候補の馬番（1〜2 桁）に分ける全ての分割。"""
    res: list[list[int]] = []

    def rec(i: int, acc: list[int]) -> None:
        if i == len(run):
            res.append(list(acc))
            return
        for ln in (1, 2):
            if i + ln <= len(run):
                v = int(run[i:i + ln])
                if str(v) == run[i:i + ln] and v in candidates:
                    acc.append(v)
                    rec(i + ln, acc)
                    acc.pop()

    rec(0, [])
    return res


def parse_rows(numbers: str, arity: int, k: int, candidates: set[int], ordered: bool, allow_dup: bool = False) -> list[list[tuple[int, ...]]]:
    """馬番の文字列を k 行・arity 個ずつの組に分ける全ての解釈。組は（順序なしなら昇順、順序ありなら表示順）。"""
    runs = [r for r in re.split(r"[^\d]+", str(numbers or "")) if r]
    seg_options = [_segment_run(r, candidates) for r in runs]
    if any(not o for o in seg_options):
        return []
    sols: list[list[tuple[int, ...]]] = []
    for combo in itertools.product(*seg_options):
        toks = [x for part in combo for x in part]
        if len(toks) != arity * k:
            continue
        rows = [tuple(toks[i * arity:(i + 1) * arity]) for i in range(k)]
        if not allow_dup and any(len(set(r)) != arity for r in rows):
            continue
        rows = [r if ordered else tuple(sorted(r)) for r in rows]
        sols.append(rows)
    return sols


def combo_str(row: Sequence[int], ordered: bool) -> str:
    return (">" if ordered else "-").join(str(x) for x in row)


def parse_payoff(payoff: dict, finishers: dict[int, int], field: Iterable[int]) -> list[dict]:
    """1 レース分の payoff（券種 → {numbers, payout}）を行に展開する。
    finishers: 馬番 → 着順（1 以上のもの）。戻り値の各行: bet_type, combo, payout, n_rows, ambiguous, parsed。"""
    top3 = {h for h, p in finishers.items() if 1 <= p <= 3}
    field_set = set(field) | set(finishers)
    out: list[dict] = []
    for jp, v in (payoff or {}).items():
        bt = BET_KEYS.get(jp)
        if bt is None or not isinstance(v, (dict, list)):
            continue
        ordered = bt in ORDERED
        n = ARITY[bt]
        if isinstance(v, list):      # 旧形式: 1 行 1 組の辞書のリスト（人気順位つき）
            for it in v:
                nums = [int(x) for x in re.findall(r"\d+", str(it.get("numbers", "")))]
                pay = _to_int(it.get("payout"))
                if len(nums) == n and (len(set(nums)) == n or bt == "wakuren") and pay:
                    out.append({"bet_type": bt, "combo": combo_str(nums if ordered else sorted(nums), ordered), "payout": pay, "n_rows": len(v), "ambiguous": False,
                                "parsed": True, "popularity": _to_int(it.get("popularity"))})
                else:
                    out.append({"bet_type": bt, "combo": None, "payout": None, "n_rows": len(v), "ambiguous": False, "parsed": False, "popularity": None})
            continue
        if isinstance(v.get("numbers"), str) and v.get("popularity") not in (None, "") and len(split_payouts(v.get("payout", ""))) == 1 and \
                len(re.findall(r"\d+", v["numbers"])) == n and _to_int(v.get("payout")):
            nums = [int(x) for x in re.findall(r"\d+", v["numbers"])]       # 旧形式の単一行（人気順位つき）
            out.append({"bet_type": bt, "combo": combo_str(nums if ordered else sorted(nums), ordered), "payout": _to_int(v["payout"]), "n_rows": 1, "ambiguous": False,
                        "parsed": True, "popularity": _to_int(v.get("popularity"))})
            continue
        pays = split_payouts(v.get("payout", ""))
        cand_sets = [set(range(1, 9))] if bt == "wakuren" else [top3, field_set]
        best = None
        for cands in cand_sets:
            hits = []
            for p in pays:
                rows = parse_rows(v.get("numbers", ""), n, len(p), cands, ordered, allow_dup=(bt == "wakuren"))   # 枠連は同じ枠の組もある
                for r in rows:
                    hits.append((p, r))
            if hits:
                best = (hits, cands is cand_sets[0])
                break
        if not best:
            out.append({"bet_type": bt, "combo": None, "payout": None, "n_rows": len(pays[0]) if pays else 0, "ambiguous": False, "parsed": False, "popularity": None})
            continue
        hits, _ = best
        ambiguous = len({(tuple(p), tuple(r)) for p, r in hits}) > 1
        p, r = hits[0]
        for payout, row in zip(p, r):
            out.append({"bet_type": bt, "combo": combo_str(row, ordered), "payout": payout, "n_rows": len(p), "ambiguous": ambiguous, "parsed": True, "popularity": None})
    return out


def build_payoff_table(result_flat: pd.DataFrame) -> pd.DataFrame:
    """race_result_flat（race_id, horse_number, finish_position, payoff）から払戻の整形テーブルを作る。"""
    rows = []
    for rid, g in result_flat.groupby("race_id", sort=False):
        payoff = g["payoff"].iloc[0]
        if isinstance(payoff, str) and payoff:
            try:
                payoff = json.loads(payoff)
            except ValueError:
                payoff = {}
        fin = {int(h): int(p) for h, p in zip(g["horse_number"], g["finish_position"]) if pd.notna(h) and pd.notna(p) and p > 0}
        for r in parse_payoff(payoff if isinstance(payoff, dict) else {}, fin, [int(h) for h in g["horse_number"].dropna()]):
            r["race_id"] = rid
            rows.append(r)
    return pd.DataFrame(rows, columns=["race_id", "bet_type", "combo", "payout", "popularity", "n_rows", "ambiguous", "parsed"])


# ── オッズ
def pair_combo(pair: str, bet_type: str) -> str:
    nums = [int(x) for x in re.findall(r"\d+", str(pair))]
    return combo_str(nums if bet_type in ORDERED else sorted(nums), bet_type in ORDERED)


def build_pair_table(pair_flat: pd.DataFrame) -> pd.DataFrame:
    d = pair_flat[["race_id", "odds_type", "pair", "odds", "odds_min", "odds_max", "popularity"]].copy()
    d["bet_type"] = d["odds_type"]
    d["combo"] = [pair_combo(p, t) for p, t in zip(d["pair"], d["odds_type"])]
    return d[["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max", "popularity"]]


def build_win_place_table(odds_flat: pd.DataFrame, result_flat: pd.DataFrame) -> pd.DataFrame:
    ids = result_flat[["race_id", "horse_number", "horse_id"]].drop_duplicates(["race_id", "horse_number"])
    d = odds_flat[["race_id", "date", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "popularity"]].merge(ids, on=["race_id", "horse_number"], how="left")
    d["phase"] = "unknown(final)"       # 2020〜2025 年は締切後に取得した確定オッズ。スナップショット（T-60・T-15）は無い
    d["has_snapshot"] = False
    return d


# ── 整合の検証
def consistency_report(payoff: pd.DataFrame, win_place: pd.DataFrame, pair: pd.DataFrame, result_flat: pd.DataFrame) -> dict:
    """払戻とオッズの整合、確定オッズの控除率を検証する。"""
    out: dict = {}
    ok = payoff[payoff["parsed"] & payoff["combo"].notna()]
    out["payoff_parse_rate"] = float(payoff.groupby("race_id")["parsed"].all().mean()) if len(payoff) else None
    out["payoff_ambiguous_share"] = float(payoff["ambiguous"].mean()) if len(payoff) else None
    # 単勝: 払戻 = 単勝オッズ × 100（10 円未満切り捨て）
    t = ok[ok.bet_type == "tansho"].copy()
    t["horse_number"] = t["combo"].astype(int)
    m = t.merge(win_place[["race_id", "horse_number", "win_odds"]], on=["race_id", "horse_number"], how="inner")
    exp = np.floor(m["win_odds"] * 10 + 1e-6) * 10
    out["tansho_checked"] = int(len(m))
    out["tansho_exact_share"] = float((m["payout"] == exp).mean()) if len(m) else None
    out["tansho_within_10yen_share"] = float(((m["payout"] - exp).abs() <= 10).mean()) if len(m) else None
    # 複勝: 払戻/100 が複勝オッズの幅に入る
    f = ok[ok.bet_type == "fukusho"].copy()
    f["horse_number"] = f["combo"].astype(int)
    m = f.merge(win_place[["race_id", "horse_number", "place_odds_min", "place_odds_max"]], on=["race_id", "horse_number"], how="inner")
    inside = (m["payout"] / 100 >= m["place_odds_min"] - 0.051) & (m["payout"] / 100 <= m["place_odds_max"] + 0.051)
    out["fukusho_checked"] = int(len(m))
    out["fukusho_inside_range_share"] = float(inside.mean()) if len(m) else None
    # 馬連・馬単: 払戻 = オッズ × 100
    for bt in ("umaren", "umatan"):
        p = ok[ok.bet_type == bt].merge(pair[pair.bet_type == bt][["race_id", "combo", "odds"]], on=["race_id", "combo"], how="inner")
        e = np.floor(p["odds"] * 10 + 1e-6) * 10
        out[f"{bt}_checked"] = int(len(p))
        out[f"{bt}_exact_share"] = float((p["payout"] == e).mean()) if len(p) else None
        out[f"{bt}_within_10yen_share"] = float(((p["payout"] - e).abs() <= 10).mean()) if len(p) else None
    # ワイド: 払戻/100 がワイドオッズの幅に入る
    p = ok[ok.bet_type == "wide"].merge(pair[pair.bet_type == "wide"][["race_id", "combo", "odds_min", "odds_max"]], on=["race_id", "combo"], how="inner")
    ins = (p["payout"] / 100 >= p["odds_min"] - 0.051) & (p["payout"] / 100 <= p["odds_max"] + 0.051)
    out["wide_checked"] = int(len(p))
    out["wide_inside_range_share"] = float(ins.mean()) if len(p) else None
    return out


def takeout_by_year(win_place: pd.DataFrame, pair: pd.DataFrame) -> dict:
    """確定オッズの Σ(1/オッズ) の年別の中央値と、そこから逆算した払戻率（JRA の公称は PAYBACK）。"""
    res: dict = {}
    w = win_place.dropna(subset=["win_odds"]).copy()
    w["year"] = w["date"].astype(str).str[:4]
    s = w.groupby(["year", "race_id"])["win_odds"].apply(lambda x: float((1.0 / x).sum())).reset_index(name="s")
    res["tansho"] = {y: {"median_sum_inv_odds": float(g["s"].median()), "implied_payback": float(1.0 / g["s"].median())} for y, g in s.groupby("year")}
    pr = pair.copy()
    pr["year"] = pr["race_id"].astype(str).str[:4]    # race_id の先頭 4 桁は開催年
    for bt in ("umaren", "umatan"):
        d = pr[(pr.bet_type == bt) & pr["odds"].notna()]
        s = d.groupby(["year", "race_id"])["odds"].apply(lambda x: float((1.0 / x).sum())).reset_index(name="s")
        res[bt] = {y: {"median_sum_inv_odds": float(g["s"].median()), "implied_payback": float(1.0 / g["s"].median())} for y, g in s.groupby("year")}
    res["nominal_payback"] = {k: PAYBACK[k] for k in ("tansho", "umaren", "umatan")}
    return res


def coverage_report(years: Sequence[int], races: dict[int, set], win_place: pd.DataFrame, pair: pd.DataFrame, payoff: pd.DataFrame) -> dict:
    cov: dict = {}
    for y in years:
        rs = races[y]
        n = len(rs)
        row = {"races": n, "win_place_odds": len(rs & set(win_place["race_id"])) , "payoff_all_parsed": 0}
        for bt in ("umaren", "wide", "umatan", "sanrenpuku"):
            row[f"pair_{bt}"] = int(pair[pair.bet_type == bt]["race_id"].isin(rs).groupby(pair[pair.bet_type == bt]["race_id"]).first().sum()) if (pair.bet_type == bt).any() else 0
        pj = payoff[payoff.race_id.isin(rs)]
        for bt in BET_KEYS.values():
            row[f"payoff_{bt}"] = int(pj[(pj.bet_type == bt)]["race_id"].nunique())
        row["payoff_all_parsed"] = int(pj.groupby("race_id")["parsed"].all().sum())
        cov[str(y)] = row
    return cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path, default=Path("data/page_reference/tables"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/local/modeling/betting"))
    ap.add_argument("--years", default="2020-2025")
    ap.add_argument("--allow-2026", action="store_true", help="2026 年（最終評価用の未使用年）を含める。既定の出力先は上書きさせないので --out-dir の明示が必須")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-"))
    years = list(range(lo, hi + 1))
    if max(years) >= 2026:
        assert a.allow_2026 and a.out_dir != Path("data/local/modeling/betting"), "2026 年は学習・選択に使わない。最終評価用は --allow-2026 と別の --out-dir を指定する"
    a.out_dir.mkdir(parents=True, exist_ok=True)
    wp_all, pair_all, pay_all, races = [], [], [], {}
    for y in years:
        d = a.tables_dir / str(y)
        rr = pd.read_parquet(d / "race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id", "finish_position", "payoff", "date"])
        races[y] = set(rr["race_id"].unique())
        wp = build_win_place_table(pd.read_parquet(d / "race_odds_flat.parquet", columns=["race_id", "date", "horse_number", "win_odds", "place_odds_min", "place_odds_max", "popularity"]), rr)
        pr = build_pair_table(pd.read_parquet(d / "race_pair_odds_flat.parquet", columns=["race_id", "odds_type", "pair", "odds", "odds_min", "odds_max", "popularity"]))
        py = build_payoff_table(rr)
        wp_all.append(wp); pair_all.append(pr); pay_all.append(py)
        print(y, "odds", len(wp), "pair", len(pr), "payoff", len(py), "parse率", round(float(py.groupby("race_id")["parsed"].all().mean()), 4), flush=True)
    wp, pair, pay = pd.concat(wp_all, ignore_index=True), pd.concat(pair_all, ignore_index=True), pd.concat(pay_all, ignore_index=True)
    wp.to_parquet(a.out_dir / "odds_win_place.parquet", index=False)
    pair.to_parquet(a.out_dir / "odds_pair.parquet", index=False)
    pay.to_parquet(a.out_dir / "payoff.parquet", index=False)
    rr_all = pd.concat([pd.read_parquet(a.tables_dir / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "finish_position"]) for y in years])
    report = {"years": years, "coverage": coverage_report(years, races, wp, pair, pay), "consistency": consistency_report(pay, wp, pair, rr_all),
              "takeout": takeout_by_year(wp, pair),
              "notes": ["2020〜2025 年のオッズは締切後に取得した確定オッズ（phase なし・スナップショットなし）。購入判断の時点の値ではない",
                        "三連複・三連単のオッズは取得できていない（払戻のみ）", "複勝のオッズは幅（min/max）、払戻は確定値"]}
    (a.out_dir / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report["consistency"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
