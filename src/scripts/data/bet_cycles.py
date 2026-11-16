"""推奨馬券の最適化サイクル（読み取りのみ・手元のデータだけ・費用 0 円）。

python -m src.scripts.data.bet_cycles run --cycle config/bet_cycles/cycle_01.json [--threads 4] [--inputs config/bet_cycles/inputs.json]
出力先: 環境変数 BET_CYCLES_OUT（既定 data/local/research/bet_cycles。root 所有で書けない PC では retrain_out などを指す）の <tag>/ に
cycle_XX.json（記録）・state.json（系統ごとの最良）・qcuts_train.json（訓練で凍結した分位の区切り）・cycle_XX_selected_train_races.parquet。

依頼（ユーザー、統括役経由 2026-11-16。指示書 /home/jovyan/work/progress/bet_optimizer_spec.md）:
勝率・複勝率の分布の型ごとに一番良い買い方（◎ 起点）を 20 サイクルで決め、得意なレースだけを推奨度の段に分けて推奨する。

作法（時間の漏れを防ぐ）:
- 予測は OOF（その年を学習に含まないモデルの予測）だけ。期間は config/bet_cycles/inputs.json の periods。
  訓練（2022〜23）で探す・選ぶ・回顧する。評価（2024）は「変わらず上向きか」の確かめだけ。test（2025）は凍結の後に 1 回だけ。
- 分位の区切り（混戦度など）は訓練の期間だけで決めて凍結し、他の期間にそのまま当てる。
- オッズ・人気は確定（race_result_flat の最終オッズ）。実際に買う時点（締切前）のオッズとはずれる。
  三連複・三連単はオッズが無いので、確率だけで組む（期待値は単勝でだけ使う）。
- 取消・除外の馬（人気・オッズとも無い）は買えないので予測から外し、勝率を出走馬で合計 1、複勝率を合計 min(3, 頭数) に正規化し直す。

買い方（strategy。JSON で書ける dict）:
  {"name": str,
   "honmei": {"rule": "top"|"fav_maxodds"|"ana_ev"|"ana_p3"|"ev_max", ...},   # ◎ の決め方（主な調整の対象）
   "partners": {"mode": "p3"|"pw"|"fav"|"mix", "n": 2, "fav_k": 3, "n_ana": 0}, # 相手（○▲△ の順）
   "tickets": [{"type": "tansho"|"fukusho"|"umaren"|"wide"|"umatan"|"umatan_multi"|"trio1"|"trio2"|"trio_form"
                         |"tri_first"|"tri_form"|"box_umaren"|"box_wide"|"box_trio", "k": 2, "m": 2, "units": 1}],
   "main_w": 1,                                                                 # 本線（◎-○ を含む目）の倍率
   "skip": {"min_field": 5, "h_ev_min": null, "h_pw_min": null, "when": {特徴: [下限, 上限)}}}
  政策（policy）: {"name": str, "policy": {"by": 特徴の名前, "map": {水準: strategy | "skip"}}}
1 点 100 円、1 レースの上限 2,000 円（20 点を超えたら印の下位の目から削る）。

印と買い目の整合（docs/ideas/betting_patterns.md §4 の R1〜R8）を、どの買い方にも機械的に当てる。違反した買い方は採らない。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CFG_DIR = ROOT / "config/bet_cycles"
OUT_DIR = Path(os.environ.get("BET_CYCLES_OUT") or ROOT / "data/local/research/bet_cycles")
PAYOFF_PATHS = [ROOT / "data/local/modeling/betting/payoff.parquet", ROOT / "data/local/modeling/betting_2026/payoff.parquet"]
UNIT = 100
MAX_UNITS = 20                    # 1 レース 2,000 円（UI の既定の上限 delegates.DEFAULT_MAX_TOTAL_BET）
MIN_BET_RACES = 300               # 訓練でこれ未満しか参戦しない買い方は採らない
BET_TYPES = ("tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan")
ORDERED = ("umatan", "sanrentan")
MARK_NAMES = ("◎", "○", "▲", "△", "△", "△", "☆", "☆", "☆", "☆")

# ── 払戻 ─────────────────────────────────────────────────────────────────


def combo_key(bet_type: str, horses) -> str:
    """払戻の表の組の書き方にそろえる。順序の無い券種は馬番の昇順、馬単・三連単は着順のまま。"""
    h = [int(x) for x in horses]
    if bet_type in ("tansho", "fukusho"):
        return str(h[0])
    if bet_type in ORDERED:
        return ">".join(map(str, h))
    return "-".join(map(str, sorted(h)))


def load_payoffs(race_ids: set[str] | None = None, paths=PAYOFF_PATHS) -> dict[str, dict[str, dict[str, int]]]:
    """{race_id: {bet_type: {組: 払戻（100 円あたり）}}}。同着は組ごとの行なので、当たった組をすべて足せば入る。"""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for p in paths:
        if not Path(p).exists():
            continue
        pay = pd.read_parquet(p, columns=["race_id", "bet_type", "combo", "payout"])
        pay["race_id"] = pay["race_id"].astype(str)
        pay = pay[pay["bet_type"].isin(BET_TYPES)]
        if race_ids is not None:
            pay = pay[pay["race_id"].isin(race_ids)]
        for rid, bt, combo, v in zip(pay["race_id"], pay["bet_type"], pay["combo"].astype(str), pay["payout"]):
            sep = ">" if ">" in combo else "-"
            key = combo_key(bt, combo.split(sep))
            out.setdefault(rid, {}).setdefault(bt, {})[key] = int(v)
    return out


# ── レースの読み込み ───────────────────────────────────────────────────────


@dataclass
class Race:
    """1 レース。馬の配列はモデルの勝率の降順（同じ値は馬番の小さい順）。index 0 = 予測 1 位。"""
    race_id: str
    date: str
    hn: np.ndarray         # 馬番
    pw: np.ndarray         # 勝率（出走馬で合計 1）
    p3: np.ndarray         # 複勝率（合計 min(3, 頭数)、各 1 以下）
    odds: np.ndarray       # 確定の単勝オッズ（無ければ nan）
    pop: np.ndarray        # 人気（無ければ 99）
    fin: np.ndarray        # 着順（無ければ 99）
    pay: dict = field(default_factory=dict)
    p2: np.ndarray | None = None   # 連対率（合計 min(2, 頭数)。予測に p_top2 が無ければ Harville で補う）


def _p3_from_win(pw: np.ndarray) -> np.ndarray:
    from src.pipeline.inference.exotic_probabilities import RankDistribution

    return RankDistribution(win=pw).top3()


def p2_from(pw: np.ndarray, p3: np.ndarray, p2: np.ndarray | None) -> np.ndarray:
    """連対率。予測にあればそれを、無ければ Harville の top2。勝率 ≤ 連対率 ≤ 複勝率・合計 2 にそろえる。"""
    from src.pipeline.inference.exotic_probabilities import RankDistribution, normalize_marginal

    if p2 is None or not np.all(np.isfinite(p2)):
        p2 = RankDistribution(win=pw).top2()
    v = normalize_marginal(np.minimum(np.maximum(np.asarray(p2, float), pw), p3), 2.0)
    return np.minimum(np.maximum(v, pw), p3)          # 合計 2 に戻した後でも 勝率 ≤ 連対率 ≤ 複勝率 を守る（合計はわずかにずれうる）


def normalize_probs(pw: np.ndarray, p3: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    pw = np.clip(np.asarray(pw, dtype=float), 1e-9, None)
    pw = pw / pw.sum()
    n = len(pw)
    if p3 is None or not np.all(np.isfinite(p3)):
        p3 = _p3_from_win(pw)
    p3 = np.clip(np.asarray(p3, dtype=float), 1e-9, None)
    tgt = min(3, n)
    for _ in range(20):                                  # 合計を頭数に合わせつつ各 1 以下に（数回で収まる）
        p3 = np.minimum(p3 * tgt / p3.sum(), 1.0)
        if abs(p3.sum() - tgt) < 1e-9:
            break
    return pw, p3


def load_runners(years) -> pd.DataFrame:
    from src.config.data_paths import TABLES_DIR

    cols = ["race_id", "date", "venue", "surface", "distance", "grade", "race_class", "track_condition", "horse_number", "horse_id",
            "sex_age", "odds", "popularity", "finish_position"]
    fs = [pd.read_parquet(Path(TABLES_DIR) / str(y) / "race_result_flat.parquet", columns=cols) for y in years]
    f = pd.concat(fs, ignore_index=True)
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    for c in ("horse_number", "odds", "popularity", "finish_position"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f.loc[f["odds"] <= 0, "odds"] = np.nan
    f.loc[f["popularity"] <= 0, "popularity"] = np.nan
    f["scratched"] = f["odds"].isna() & f["popularity"].isna()   # rank_condition_analysis.load_runners と同じ決まり
    return f[f["horse_number"].notna()]


def load_pred(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    cols = ["race_id", "horse_id", "p_win"] + [c for c in ("p_top2", "p_top3") if c in names]
    o = pd.read_parquet(path, columns=cols)
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    return o.drop_duplicates(["race_id", "horse_id"])


def build_races(pred: pd.DataFrame, flat: pd.DataFrame, pay: dict, years=None, odds_override: pd.DataFrame | None = None
                ) -> tuple[list[Race], pd.DataFrame, dict]:
    """予測と出走表から Race の一覧を作る。全頭の予測と 7 券種の払戻がそろったレースだけ（母数 = 全レース）。
    odds_override: race_id, horse_number, odds, popularity（発走前のオッズで決める版）。"""
    run = flat[~flat["scratched"]]
    if years is not None:
        run = run[run["race_id"].str[:4].astype(int).isin(list(years))]
    if odds_override is not None:
        ov = odds_override.set_index(["race_id", "horse_number"])
        idx = pd.MultiIndex.from_arrays([run["race_id"], run["horse_number"]])
        hit = idx.isin(ov.index)
        run = run.copy()
        run = run[run["race_id"].isin(set(odds_override["race_id"]))]
        idx = pd.MultiIndex.from_arrays([run["race_id"], run["horse_number"]])
        run["odds"] = ov.reindex(idx)["odds"].to_numpy()
        run["popularity"] = ov.reindex(idx)["popularity"].to_numpy()
        del hit
    has_p3 = "p_top3" in pred.columns
    d = run.merge(pred, on=["race_id", "horse_id"], how="left")
    races, drop = [], {"no_pred": 0, "no_pay": 0, "small": 0}
    for rid, g in d.groupby("race_id", sort=True):
        if g["p_win"].isna().any():
            drop["no_pred"] += 1
            continue
        if len(g) < 3:
            drop["small"] += 1
            continue
        pr = pay.get(rid)
        if not pr or any(bt not in pr for bt in BET_TYPES):
            drop["no_pay"] += 1
            continue
        g = g.sort_values(["p_win", "horse_number"], ascending=[False, True], kind="stable")
        pw, p3 = normalize_probs(g["p_win"].to_numpy(), g["p_top3"].to_numpy(dtype=float) if has_p3 else None)
        p2 = p2_from(pw, p3, g["p_top2"].to_numpy(dtype=float) if "p_top2" in g else None)
        races.append(Race(rid, str(pd.Timestamp(g["date"].iloc[0]).date()), g["horse_number"].to_numpy(int), pw, p3,
                          g["odds"].to_numpy(float), g["popularity"].fillna(99).to_numpy(float),
                          g["finish_position"].fillna(99).to_numpy(float), pr, p2))
    from src.scripts.data.rank_condition_analysis import race_conditions

    cond = race_conditions(flat[flat["race_id"].isin({r.race_id for r in races})]).set_index("race_id")
    return races, cond, drop


# ── レースの分布の特徴 ─────────────────────────────────────────────────────


def race_features(r: Race) -> dict[str, float]:
    pw, p3, odds, pop = r.pw, r.p3, r.odds, r.pop
    n = len(pw)
    s = np.sort(pw)[::-1]
    p1, p2, p3_ = s[0], s[1], (s[2] if n > 2 else 0.0)
    ent = float(-(pw * np.log(pw)).sum() / np.log(n)) if n > 1 else 0.0
    srt = np.sort(pw)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ srt / (n * srt.sum())) if n > 1 else 0.0
    ev = pw * odds
    ev_ok = np.where(np.isfinite(ev), ev, -1)
    j = int(np.argmax(ev_ok))
    fav = int(np.argmin(pop))
    mk = 1.0 / odds
    mk = mk / np.nansum(mk) if np.isfinite(mk).any() else mk
    s3 = np.sort(p3)[::-1]
    return {
        "field": n, "p1": p1, "gap12": p1 - p2, "ratio12": p1 / max(p2, 1e-9), "ratio23": p2 / max(p3_, 1e-9),
        "top3sum": float(s[:3].sum()), "top5sum": float(s[:5].sum()), "var5": float(np.var(s[:5])), "ent": ent, "gini": gini,
        "p3_top1": float(s3[0]), "p3_top3sum": float(s3[:3].sum()), "p3_n50": int((p3 >= 0.5).sum()), "p3_n30": int((p3 >= 0.3).sum()),
        "ev_top": float(ev[0]) if np.isfinite(ev[0]) else np.nan, "ev_max": float(ev_ok[j]), "ev_max_pop": float(pop[j]),
        "ev_max_rank": j, "top_pop": float(pop[0]), "top_odds": float(odds[0]), "fav_rank": fav,
        "mk_gap_top": float(pw[0] - mk[0]) if np.isfinite(mk[0]) else np.nan,
        "fav1_p2": float(r.p2[fav]) if r.p2 is not None else np.nan, "fav1_pw": float(pw[fav]), "fav1_is_top": int(fav == 0),
        "n_mk_under": int(np.nansum(pw - mk > 0.03)),
    }


def hand_shape(f: dict, r_one: float = 2.0, r_two: float = 1.3, r_close: float = 1.3, p_chaos: float = 0.2) -> str:
    """手で決めた区切り（betting_patterns.md §6.2）: one（一強: 1 位が 2 位の r_one 倍以上）/ chaos（混戦: 1 位の勝率が p_chaos 未満）/
    two（二強: 1・2 位が近く〈比 r_close 未満〉、2 位が 3 位の r_two 倍以上）/ mid（それ以外）。"""
    if f["ratio12"] >= r_one:
        return "one"
    if f["p1"] < p_chaos:
        return "chaos"
    if f["ratio23"] >= r_two and f["ratio12"] < r_close:
        return "two"
    return "mid"


QCUT_FEATURES = ("p1", "gap12", "ratio12", "ent", "gini", "var5", "top3sum", "p3_top3sum", "ev_max", "ev_top", "mk_gap_top")


def fit_qcuts(feats: pd.DataFrame, cols=QCUT_FEATURES, q=(0.2, 0.4, 0.6, 0.8)) -> dict[str, list[float]]:
    """訓練の期間だけで 5 分位の区切りを決める（凍結して他の期間にそのまま当てる）。"""
    return {c: [float(x) for x in np.nanquantile(feats[c].to_numpy(float), q)] for c in cols if c in feats}


def apply_qcuts(feats: pd.DataFrame, cuts: dict[str, list[float]]) -> pd.DataFrame:
    out = feats.copy()
    for c, e in cuts.items():
        v = out[c].to_numpy(float)
        out[f"q_{c}"] = np.where(np.isfinite(v), np.searchsorted(np.asarray(e), v, side="right") + 1, 0).astype(int)
    return out


def feature_table(races: list[Race], cond: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame([race_features(r) for r in races])
    f.insert(0, "race_id", [r.race_id for r in races])
    f.insert(1, "date", [r.date for r in races])
    f["year"] = f["race_id"].str[:4].astype(int)
    f["month"] = pd.to_datetime(f["date"]).dt.month
    f["shape"] = [hand_shape(x) for x in f.to_dict("records")]
    c = cond.reindex(f["race_id"])
    for k in ("surface", "dist_band", "venue", "class", "field", "going", "season", "age_cond"):
        f[f"c_{k}"] = c[k].to_numpy()
    f["fieldband"] = f["c_field"]
    return f


# ── ◎ の決め方・相手・印 ────────────────────────────────────────────────────


def pick_honmei(r: Race, f: dict, h: dict) -> tuple[int | None, str]:
    """◎ の index（Race の配列の位置）と理由。None は見送り。予測 1 位（index 0）以外なら理由を返す。"""
    rule = h.get("rule", "top")
    if rule == "switch":                     # 特徴の水準で決め方を切り替える（例: 一強なら予測 1 位、混戦なら人気の中の最高オッズ）
        lvl = str(f.get(h["by"]))
        sub = h["map"].get(lvl, h.get("default", {"rule": "top"}))
        if sub == "skip":
            return None, f"見送り（{h['by']}={lvl}）"
        return pick_honmei(r, f, sub)
    if rule == "top":
        return 0, "予測1位"
    if rule == "fav1_penalty":               # 1 番人気へのペナルティ λ: 1 番人気の点 = 勝率 × (1 − λ)。λ=1 は「2 番人気以下で勝率最大」
        lam = float(h.get("lam", 0.5))
        sc = np.where(r.pop == 1, r.pw * (1.0 - lam), r.pw)
        i = int(np.argmax(sc))
        return i, ("予測1位（1番人気でない）" if (i == 0 and r.pop[i] != 1) else ("1番人気（ペナルティ後も最大）" if r.pop[i] == 1
                                                                        else f"1番人気にペナルティ（λ={lam:g}）"))
    if rule == "fav1_exc":                   # 基本は 2 番人気以下から ◎。1 番人気が「ほぼ確実に勝ち負け」のときだけ例外で ◎ にする
        fav = int(np.argmin(r.pop))
        ok = r.pop[fav] == 1 and r.p2 is not None and r.p2[fav] >= float(h.get("p2_min", 0.85))
        if ok and h.get("gap_min") is not None:
            others = np.delete(r.pw, fav)
            ok = r.pw[fav] - (others.max() if len(others) else 0.0) >= float(h["gap_min"])
        if ok:
            return fav, "1番人気の例外（連対率の予測が高い）"
        sub = dict(h.get("base") or {"rule": "fav1_penalty", "lam": 1.0})
        i, why = pick_honmei(r, f, sub)
        if i is not None and r.pop[i] == 1:  # 下の決め方が 1 番人気を選んだら、2 番人気以下の勝率最大に替える
            i, why = pick_honmei(r, f, {"rule": "fav1_penalty", "lam": 1.0})
        return i, why
    if rule == "market_fav":                 # 市場の 1 番人気（基準の比べる相手。印の決め方の候補ではない）
        i = int(np.argmin(r.pop))
        return i, ("予測1位" if i == 0 else "市場の1番人気")
    pw, p3, odds, pop = r.pw, r.p3, r.odds, r.pop
    rank = np.arange(len(pw))
    ev = np.where(np.isfinite(odds), pw * odds, -1.0)
    k, m = int(h.get("fav_k", 3)), int(h.get("max_rank", 99))
    ok = (rank < m) & (pw >= float(h.get("pw_min", 0.0))) & (p3 >= float(h.get("p3_min", 0.0)))
    if rule == "fav_maxodds":                # 上位人気の中で一番オッズが高い馬
        c = ok & (pop <= k) & np.isfinite(odds)
        key, why = np.where(c, odds, -1.0), f"{k}番人気以内で最高オッズ"
    elif rule == "ana_ev":                   # 人気の外（穴）で期待値が最大
        c = ok & (pop > k) & (ev >= float(h.get("ev_min", 0.0)))
        key, why = np.where(c, ev, -1.0), f"{k}番人気より下で期待値最大"
    elif rule == "ana_p3":                   # 人気の外（穴）で複勝率が最大
        c = ok & (pop > k)
        key, why = np.where(c, p3, -1.0), f"{k}番人気より下で複勝率最大"
    elif rule == "ev_max":                   # 期待値が最大
        c = ok & (ev >= float(h.get("ev_min", 0.0)))
        key, why = np.where(c, ev, -1.0), "期待値最大"
    else:
        raise ValueError(f"◎ の決め方が不明: {rule}")
    if not c.any():
        fb = h.get("fallback", "top")
        return (None, f"見送り（{why}の候補なし）") if fb == "skip" else (0, "予測1位（候補なし）")
    i = int(np.argmax(key))
    return i, ("予測1位" if i == 0 else why)


def pick_partners(r: Race, H: int, p: dict) -> list[int]:
    """相手（印の順 ○▲△…）。mode: p3=複勝率順、pw=勝率順、fav=人気（fav_k 以内）だけを複勝率順（型 A）、
    mix=人気から n 頭 + 穴から n_ana 頭（型 B。それぞれ複勝率順）。"""
    mode, n = p.get("mode", "p3"), int(p.get("n", 2))
    k = int(p.get("fav_k", 3))
    idx = [i for i in range(len(r.pw)) if i != H]
    by3 = sorted(idx, key=lambda i: (-r.p3[i], -r.pw[i], r.hn[i]))
    if mode == "p3":
        return by3[:n]
    if mode == "pw":
        return [i for i in range(len(r.pw)) if i != H][:n]
    if mode == "fav":
        return [i for i in by3 if r.pop[i] <= k][:n]
    if mode == "mix":
        fav = [i for i in by3 if r.pop[i] <= k][:n]
        ana = [i for i in by3 if r.pop[i] > k][: int(p.get("n_ana", 2))]
        return fav + ana
    raise ValueError(f"相手の選び方が不明: {mode}")


# ── 買い目 ───────────────────────────────────────────────────────────────


def make_tickets(H: int, O: list[int], t: dict) -> list[tuple[str, tuple[int, ...], int]]:
    """印（◎=H、○▲△=O の順）から買い目を組む。返り値は (券種, 位置の組, 口数)。◎ は必ず軸・1 着。"""
    typ, u = t["type"], int(t.get("units", 1))
    k = min(int(t.get("k", len(O))), len(O))
    P = O[:k]
    out: list[tuple[str, tuple[int, ...], int]] = []
    if typ in ("tansho", "fukusho"):
        out = [(typ, (H,), u)]
    elif typ in ("umaren", "wide"):
        out = [(typ, (H, x), u) for x in P]
    elif typ == "umatan":
        out = [("umatan", (H, x), u) for x in P]
    elif typ == "umatan_multi":                     # ◎ 1 着を厚く（裏は口数を減らして R6 を守る）
        out = [("umatan", (H, x), u + 1) for x in P] + [("umatan", (x, H), u) for x in P]
    elif typ == "trio1":                            # 三連複 ◎ 1 頭軸流し
        out = [("sanrenpuku", (H, a, b), u) for a, b in itertools.combinations(P, 2)]
    elif typ == "trio2":                            # 三連複 ◎○ 2 頭軸流し
        if len(O) >= 2:
            out = [("sanrenpuku", (H, O[0], x), u) for x in O[1:k + 1]]
    elif typ == "trio_form":                        # 三連複 ◎ - ○▲ - ○▲△(m)
        m = int(t.get("m", 2))
        c2, c3 = O[:2], O[: 2 + m]
        seen = set()
        for a in c2:
            for b in c3:
                if a != b and frozenset((a, b)) not in seen:
                    seen.add(frozenset((a, b)))
                    out.append(("sanrenpuku", (H, a, b), u))
    elif typ == "tri_first":                        # 三連単 ◎ 1 着固定流し
        out = [("sanrentan", (H, a, b), u) for a, b in itertools.permutations(P, 2)]
    elif typ == "tri_form":                         # 三連単 ◎ → ○▲ → ○▲△(m)
        m = int(t.get("m", 2))
        out = [("sanrentan", (H, a, b), u) for a in O[:2] for b in O[: 2 + m] if a != b]
    elif typ.startswith("box_"):                    # ◎ なし（参考）。H と O を同列に
        bt = {"box_umaren": "umaren", "box_wide": "wide", "box_trio": "sanrenpuku"}[typ]
        hs = [H] + P
        out = [(bt, c, u) for c in itertools.combinations(hs, 3 if bt == "sanrenpuku" else 2)]
    else:
        raise ValueError(f"券種の型が不明: {typ}")
    return out


def apply_main_weight(tk, H: int, O: list[int], w: int):
    """本線（◎ と ○ を両方含む目、単複は ◎）を w 倍にする。"""
    if w <= 1 or not O:
        return tk
    P = O[0]
    return [(bt, c, u * w if (H in c and (P in c or bt in ("tansho", "fukusho"))) else u) for bt, c, u in tk]


def cap_units(tk, max_units: int = MAX_UNITS):
    """1 レースの上限を超えたら後ろ（印の下位）の目から削る。"""
    out, tot = [], 0
    for x in tk:
        if tot + x[2] > max_units:
            break                                   # 印の上位から順に積み、入らなくなったらそこで止める（下位だけ残すことはしない）
        out.append(x)
        tot += x[2]
    return out


# ── 印と買い目の整合（betting_patterns.md §4） ─────────────────────────────


def horse_stakes(bets: list[dict]) -> dict[int, float]:
    from src.pipeline.inference.bet_marks import horse_stakes as hs

    return hs(bets)


def check_marks(bets: list[dict], marks: dict[int, str], kesi: set[int] | frozenset = frozenset(), c_min: float = 0.5,
                nohonmei: bool = False) -> list[str]:
    """R1〜R8 の違反の一覧（空なら合格）。bets: [{bet_type, combo(馬番), yen}]、marks: 馬番 → 印（◎○▲△☆）。"""
    v: list[str] = []
    if not bets:
        return v
    for b in bets:                                                   # R1 消の馬を入れない
        if kesi & set(b["combo"]):
            v.append("R1")
            break
    if nohonmei:
        return v
    H = next((h for h, m in marks.items() if m == "◎"), None)
    if H is None:
        return v + ["R0"]                                            # ◎ が無いのに ◎ 起点の買い方
    P = next((h for h, m in marks.items() if m == "○"), None)
    S = next((h for h, m in marks.items() if m == "▲"), None)
    st = horse_stakes(bets)
    mx = max(st.values())
    if st.get(H, 0.0) + 1e-9 < mx:                                  # R2 ◎ の持ち分が最大
        v.append("R2")
    if P is not None and st.get(P, 0.0) > st.get(H, 0.0) + 1e-9:    # R3 印の順
        v.append("R3")
    if P is not None and S is not None and st.get(S, 0.0) > st.get(P, 0.0) + 1e-9:
        v.append("R3")
    top = max(b["yen"] for b in bets)                                # R4 本線は ◎ を含む
    if any(b["yen"] == top and H not in b["combo"] for b in bets):
        v.append("R4")
    tot = sum(b["yen"] for b in bets)                                # R5 ◎ の被覆
    if sum(b["yen"] for b in bets if H in b["combo"]) / tot < c_min:
        v.append("R5")
    first: dict[int, float] = {}
    for b in bets:                                                   # R6 1 着の向き
        if b["bet_type"] in ORDERED:
            first[b["combo"][0]] = first.get(b["combo"][0], 0.0) + b["yen"]
    if first and max(first.values()) > first.get(H, 0.0) + 1e-9:
        v.append("R6")
    win = {b["combo"][0]: b["yen"] for b in bets if b["bet_type"] == "tansho"}   # R7 単勝の向き
    if win and (H not in win or max(win.values()) > win[H]):
        v.append("R7")
    multi = {}
    for b in bets:                                                   # R8 1 着固定の向き（裏表を除く）
        if b["bet_type"] in ORDERED:
            multi.setdefault(b["bet_type"], set()).add(b["combo"][0])
    for bt, s in multi.items():                                      # 裏表（マルチ）は、その券種の目がすべて ◎ を含むときだけ免除
        rev = all(H in b["combo"] for b in bets if b["bet_type"] == bt)
        if len(s) > 1 and not rev:
            v.append("R8")
    return sorted(set(v))


# ── 1 レースの買い方の実行と精算 ───────────────────────────────────────────


def strategy_hash(s: Any) -> str:
    return hashlib.sha1(json.dumps(s, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def _skip(f: dict, sk: dict) -> bool:
    if f["field"] < int(sk.get("min_field", 5)):
        return True
    for c, (lo, hi) in (sk.get("when") or {}).items():            # when: この範囲のレースだけ買う
        x = f.get(c)
        if x is None or not (lo is None or x >= lo) or not (hi is None or x < hi):
            return True
    for c, lst in (sk.get("in") or {}).items():                      # in: この水準のレースだけ買う
        if str(f.get(c)) not in [str(y) for y in lst]:
            return True
    for c, lst in (sk.get("not_in") or {}).items():
        if str(f.get(c)) in [str(y) for y in lst]:
            return True
    return False


def play(r: Race, f: dict, s: dict) -> dict:
    """1 レースで買い方 s を実行する。返り値: cost・ret・H・reason・marks・bets・violations（見送りなら cost=0）。"""
    if "policy" in s:
        pol = s["policy"]
        lvl = typing_label(f, pol["typing"]) if "typing" in pol else str(f.get(pol["by"]))
        sub = pol["map"].get(lvl, pol.get("default", "skip"))
        if sub == "skip":
            return {"cost": 0, "ret": 0, "H": None, "reason": f"見送り（型 {lvl}）"}
        return play(r, f, sub)
    sk = s.get("skip") or {}
    if _skip(f, sk):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（条件）"}
    H, why = pick_honmei(r, f, s.get("honmei") or {"rule": "top"})
    if H is None:
        return {"cost": 0, "ret": 0, "H": None, "reason": why}
    ev_h = r.pw[H] * r.odds[H] if np.isfinite(r.odds[H]) else np.nan
    if sk.get("h_ev_min") is not None and not (ev_h >= float(sk["h_ev_min"])):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（◎ の期待値）"}
    if sk.get("h_ev_max") is not None and not (ev_h < float(sk["h_ev_max"])):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（◎ の期待値が高すぎ）"}
    if sk.get("h_pw_min") is not None and r.pw[H] < float(sk["h_pw_min"]):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（◎ の勝率）"}
    if sk.get("h_pop_max") is not None and r.pop[H] > float(sk["h_pop_max"]):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（◎ が人気の外）"}
    if sk.get("h_pop_min") is not None and r.pop[H] < float(sk["h_pop_min"]):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（◎ が人気馬）"}
    if sk.get("h_odds_max") is not None and not (r.odds[H] < float(sk["h_odds_max"])):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（◎ のオッズが高い）"}
    if sk.get("h_odds_min") is not None and not (r.odds[H] >= float(sk["h_odds_min"])):
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（◎ のオッズ）"}
    O = pick_partners(r, H, s.get("partners") or {})
    tk = []
    for t in s.get("tickets") or [{"type": "tansho"}]:
        tk += make_tickets(H, O, t)
    tk = cap_units(apply_main_weight(tk, H, O, int(s.get("main_w", 1))))
    if not tk:
        return {"cost": 0, "ret": 0, "H": None, "reason": "見送り（買い目なし）"}
    bets = [{"bet_type": bt, "combo": [int(r.hn[i]) for i in c], "yen": u * UNIT} for bt, c, u in tk]
    nohonmei = all(t["type"].startswith("box_") for t in s.get("tickets") or [{"type": "tansho"}])   # ボックスだけの買い方（◎ なしの参考）
    marks = {} if nohonmei else {int(r.hn[i]): MARK_NAMES[min(j, len(MARK_NAMES) - 1)] for j, i in enumerate([H] + O)}
    viol = check_marks(bets, marks, nohonmei=nohonmei)
    ret = sum(r.pay[b["bet_type"]].get(combo_key(b["bet_type"], b["combo"]), 0) * (b["yen"] // UNIT) for b in bets)
    return {"cost": sum(b["yen"] for b in bets), "ret": int(ret), "H": H, "reason": why, "O": O, "bets": bets, "violations": viol}


def run_strategy(races: list[Race], feats: list[dict], s: dict, detail: bool = False) -> dict:
    """全レースで実行し、レースごとの cost・ret の配列と、違反の数・◎ の内訳を返す。"""
    n = len(races)
    cost, ret = np.zeros(n), np.zeros(n)
    hrank = np.full(n, -1)                   # ◎ の予測の順位（0 = 予測 1 位、-1 = 見送り）
    hfin = np.full(n, 99.0)
    hpop = np.full(n, 99.0)
    nviol, vkinds, recs = 0, {}, []
    o_in3 = np.zeros(n)                      # 相手のうち 3 着以内に来た頭数
    for i, (r, f) in enumerate(zip(races, feats)):
        x = play(r, f, s)
        cost[i], ret[i] = x["cost"], x["ret"]
        if x.get("H") is not None:
            hrank[i], hfin[i], hpop[i] = x["H"], r.fin[x["H"]], r.pop[x["H"]]
            o_in3[i] = sum(r.fin[j] <= 3 for j in x.get("O", []))
            if x.get("violations"):
                nviol += 1
                for k in x["violations"]:
                    vkinds[k] = vkinds.get(k, 0) + 1
        if detail:
            recs.append({"race_id": r.race_id, "H_num": int(r.hn[x["H"]]) if x.get("H") is not None else None,
                         "H_rank": int(hrank[i]), "reason": x["reason"], "cost": int(cost[i]), "ret": int(ret[i]),
                         "bets": json.dumps(x.get("bets", []), ensure_ascii=False), "violations": ",".join(x.get("violations", []))})
    out = {"cost": cost, "ret": ret, "hrank": hrank, "hfin": hfin, "hpop": hpop, "o_in3": o_in3, "n_violation_races": nviol,
           "violations": vkinds}
    if detail:
        out["records"] = pd.DataFrame(recs)
    return out


# ── 指標 ─────────────────────────────────────────────────────────────────


def poisson_weights(n: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """レース単位のブートストラップの重み（Poisson(1)）。同じ重みを全部の買い方に使うので、差は対の区間になる。"""
    return np.random.default_rng(seed).poisson(1.0, size=(n_boot, n)).astype(np.float32)


def metrics(cost: np.ndarray, ret: np.ndarray, dates: np.ndarray, W: np.ndarray | None = None) -> dict:
    """必須の指標（参戦数・率、的中率、回収率、平均・中央値の回収額）と、区間・1 日の振れ・落ち込み・上位を除いた回収率。"""
    n_all = len(cost)
    b = cost > 0
    nb = int(b.sum())
    m: dict[str, Any] = {"races": n_all, "bet_races": nb, "bet_rate": nb / n_all if n_all else 0.0}
    if nb == 0:
        return m
    c, r = cost[b], ret[b]
    hit = r > 0
    m.update({"hit_rate": float(hit.mean()), "roi": float(r.sum() / c.sum()), "avg_return": float(r.mean()),
              "median_return": float(np.median(r)), "median_hit_return": float(np.median(r[hit])) if hit.any() else 0.0,
              "avg_cost": float(c.mean()), "cost_total": float(c.sum()), "return_total": float(r.sum()),
              "avg_hit_return": float(r[hit].mean()) if hit.any() else 0.0})
    srt = np.sort(r)[::-1]
    for k in (1, 5):
        m[f"roi_ex_top{k}"] = float((r.sum() - srt[:k].sum()) / c.sum())
    if W is not None:
        Wb = W[:, b]
        bs = (Wb @ r) / np.maximum(Wb @ c, 1e-9)
        m["roi_lo"], m["roi_hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
        hb = (Wb @ hit.astype(np.float32)) / np.maximum(Wb.sum(1), 1e-9)
        m["hit_lo"], m["hit_hi"] = float(np.quantile(hb, 0.025)), float(np.quantile(hb, 0.975))
    d = pd.DataFrame({"date": dates[b], "c": c, "r": r}).groupby("date").sum()
    dr = d["r"] / d["c"]
    pl = (d["r"] - d["c"]).cumsum().to_numpy()
    dd = float(np.max(np.maximum.accumulate(np.r_[0.0, pl]) - np.r_[0.0, pl]))
    m.update({"days": int(len(d)), "daily_roi_sd": float(dr.std(ddof=1)) if len(d) > 1 else 0.0,
              "max_drawdown": dd, "max_drawdown_days": dd / float(d["c"].mean()), "plus_day_rate": float((d["r"] > d["c"]).mean()),
              "profit": float(r.sum() - c.sum())})
    return m


def paired_diff(a: dict, b: dict, W: np.ndarray) -> tuple[float, float, float]:
    """回収率の差 a − b と、対のブートストラップの 95% 区間（同じレースの集合・同じ重み）。"""
    ra = (W @ a["ret"]) / np.maximum(W @ a["cost"], 1e-9)
    rb = (W @ b["ret"]) / np.maximum(W @ b["cost"], 1e-9)
    pa = a["ret"].sum() / max(a["cost"].sum(), 1e-9)
    pb = b["ret"].sum() / max(b["cost"].sum(), 1e-9)
    d = ra - rb
    return float(pa - pb), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def decompose(sub: dict, ref: dict) -> dict:
    """回収率の比 = 的中率の比 × 的中時の平均払戻の比 ÷ 1 レースの購入額の比（対数で足し算に分ける）。"""
    try:
        lh = np.log(sub["hit_rate"] / ref["hit_rate"])
        la = np.log(sub["avg_hit_return"] / ref["avg_hit_return"])
        lc = -np.log(sub["avg_cost"] / ref["avg_cost"])
    except (KeyError, ZeroDivisionError, FloatingPointError):
        return {}
    if not all(np.isfinite([lh, la, lc])):
        return {}
    return {"log_hit": float(lh), "log_payout": float(la), "log_cost": float(lc), "log_total": float(lh + la + lc)}


# ── 回顧 ─────────────────────────────────────────────────────────────────

RETRO_AXES = {
    "shape": "分布の型（手の区切り）", "q_p1": "1 位の勝率の 5 分位", "q_gap12": "1・2 位の差の 5 分位", "q_ent": "エントロピーの 5 分位",
    "q_ev_max": "期待値の最大の 5 分位", "fieldband": "頭数", "c_class": "クラス", "c_surface": "芝・ダート", "c_dist_band": "距離帯",
    "c_going": "馬場", "c_season": "季節", "c_venue": "会場", "h_popband": "◎ の人気", "h_is_top": "◎ が予測 1 位か",
}


def pop_band(p: float) -> str:
    if p <= 1:
        return "1番人気"
    if p <= 3:
        return "2-3番人気"
    if p <= 5:
        return "4-5番人気"
    if p <= 9:
        return "6-9番人気"
    return "10番人気~" if p < 99 else "見送り"


def miss_types(res: dict) -> dict[str, int]:
    """不的中の典型を数える（参戦して外れたレース）。"""
    b = (res["cost"] > 0) & (res["ret"] == 0)
    fin, oin = res["hfin"][b], res["o_in3"][b]
    return {"◎ が 4 着以下（◎ が飛んだ）": int((fin > 3).sum()),
            "◎ は 3 着内・相手が 1 頭も来ない": int(((fin <= 3) & (oin == 0)).sum()),
            "◎ は 3 着内・相手も来たが組・着順が合わない": int(((fin <= 3) & (oin > 0)).sum())}


def retrospective(res: dict, ft: pd.DataFrame, W: np.ndarray, axes=RETRO_AXES, min_n: int = 30) -> list[dict]:
    """条件ごとに、的中率・回収率・区間・全体との差の分解。区間が全体の回収率を含むセルは「保留」。"""
    ft = ft.copy()
    ft["h_popband"] = [pop_band(p) for p in res["hpop"]]
    ft["h_is_top"] = np.where(res["hrank"] == 0, "予測1位", np.where(res["hrank"] > 0, "予測1位以外", "見送り"))
    dates = ft["date"].to_numpy()
    allm = metrics(res["cost"], res["ret"], dates, W)
    rows = []
    for ax, lab in axes.items():
        if ax not in ft:
            continue
        for lvl, idx in ft.groupby(ax).indices.items():
            mask = np.zeros(len(ft), bool)
            mask[idx] = True
            c, r = np.where(mask, res["cost"], 0), np.where(mask, res["ret"], 0)
            if (c > 0).sum() < min_n:
                continue
            m = metrics(c, r, dates, W)
            m.pop("races", None)
            row = {"axis": ax, "axis_label": lab, "level": str(lvl), **m, "races_in_level": int(mask.sum())}
            row.update(decompose(m, allm))
            row["verdict"] = ("保留" if m["roi_lo"] <= allm["roi"] <= m["roi_hi"] else ("良い" if m["roi"] > allm["roi"] else "悪い"))
            rows.append(row)
    return rows


# ── 方策（分布の型 → 買い方の対応表） ────────────────────────────────────


def typing_label(f: dict, ty: dict) -> str:
    """レースの型の名前。kind: feat（特徴の水準そのまま）/ grid（複数の特徴の水準の組）/ kmeans（凍結した中心に最も近い）/
    tree（凍結した決定木の葉）。どれも訓練の期間だけで当てはめた値を持つ（他の期間にはそのまま当てる）。"""
    k = ty["kind"]
    if k == "feat":
        return str(f.get(ty["col"]))
    if k == "grid":
        return "|".join(f"{c}={f.get(c)}" for c in ty["cols"])
    x = np.array([float(f.get(c, np.nan)) for c in ty["cols"]])
    x = np.where(np.isfinite(x), x, np.asarray(ty["fill"]))
    if k == "kmeans":
        z = (x - np.asarray(ty["mean"])) / np.asarray(ty["std"])
        return f"k{int(np.argmin(((np.asarray(ty['centers']) - z) ** 2).sum(1)))}"
    if k == "tree":
        node = 0
        nodes = ty["nodes"]
        while nodes[node]["leaf"] is False:
            nd = nodes[node]
            node = nd["left"] if x[nd["f"]] <= nd["t"] else nd["right"]
        return f"t{node}"
    raise ValueError(f"型の作り方が不明: {k}")


def fit_typing(ty: dict, ft: pd.DataFrame, R: np.ndarray | None = None, C: np.ndarray | None = None, seed: int = 0) -> dict:
    """型の作り方を訓練の期間で当てはめ、凍結できる形（JSON）にする。
    kmeans: 特徴を標準化して k 個の中心。tree: 上位の買い方のレースごとの損益率（R/C−1）を多出力で当てる回帰木の葉。"""
    k = ty["kind"]
    if k in ("feat", "grid"):
        return dict(ty)
    cols = ty["cols"]
    X = ft[cols].to_numpy(float)
    fill = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, fill)
    out = dict(ty, fill=[float(v) for v in fill])
    if k == "kmeans":
        from sklearn.cluster import KMeans

        mu, sd = X.mean(0), X.std(0) + 1e-9
        km = KMeans(n_clusters=int(ty["k"]), n_init=10, random_state=seed).fit((X - mu) / sd)
        out.update(mean=mu.tolist(), std=sd.tolist(), centers=km.cluster_centers_.tolist())
        return out
    if k == "tree":
        from sklearn.tree import DecisionTreeRegressor

        Y = np.where(C > 0, R / np.maximum(C, 1) - 1.0, 0.0).T             # レース × 買い方
        tr = DecisionTreeRegressor(max_depth=int(ty.get("depth", 3)), min_samples_leaf=int(ty.get("min_leaf", 400)), random_state=seed)
        tr.fit(X, Y)
        t = tr.tree_
        nodes = []
        for i in range(t.node_count):
            leaf = bool(t.children_left[i] == t.children_right[i])
            nodes.append({"leaf": leaf, "f": int(t.feature[i]) if not leaf else -1, "t": float(t.threshold[i]) if not leaf else 0.0,
                          "left": int(t.children_left[i]), "right": int(t.children_right[i])})
        out.update(nodes=nodes)
        return out
    raise ValueError(k)


def choose_cells(labels: np.ndarray, R: np.ndarray, C: np.ndarray, mask: np.ndarray, min_cell: int, skip_below: float | None
                 ) -> dict[str, int]:
    """型ごとに、mask のレースで回収率が一番高い買い方の番号（-1 = 見送り）。参戦が min_cell 未満の買い方は候補にしない。"""
    out = {}
    for lv in np.unique(labels[mask]):
        m = mask & (labels == lv)
        c, r = C[:, m].sum(1), R[:, m].sum(1)
        nb = (C[:, m] > 0).sum(1)
        roi = np.where((nb >= min_cell) & (c > 0), r / np.maximum(c, 1), -1.0)
        j = int(np.argmax(roi))
        out[str(lv)] = -1 if roi[j] < 0 or (skip_below is not None and roi[j] < skip_below) else j
    return out


def apply_cells(labels: np.ndarray, cells: dict[str, int], R: np.ndarray, C: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost, ret = np.zeros(C.shape[1]), np.zeros(C.shape[1])
    for i in np.where(mask)[0]:
        j = cells.get(str(labels[i]), -1)
        if j >= 0:
            cost[i], ret[i] = C[j, i], R[j, i]
    return cost, ret


def cv_policy(ty: dict, ft: pd.DataFrame, frecs: list[dict], years: np.ndarray, R: np.ndarray, C: np.ndarray, min_cell: int,
              skip_below: float | None) -> tuple[np.ndarray, np.ndarray]:
    """訓練の中の年の交差検証: ある年で型の作り方（kmeans・木）を当てはめ、型ごとの最良を選び、もう一方の年に当てる。
    型の当てはめも fold ごとにやり直す（木は損益を見るので、見た年に当てると有利に出る）。返り値は全レースの（当てた側の）cost・ret。"""
    cost, ret = np.zeros(C.shape[1]), np.zeros(C.shape[1])
    for y in np.unique(years):
        fit, hold = years != y, years == y
        fty = fit_typing(ty, ft[fit], R[:, fit], C[:, fit])
        labels = np.array([typing_label(f, fty) for f in frecs])
        cells = choose_cells(labels, R, C, fit, min_cell, skip_below)
        c, r = apply_cells(labels, cells, R, C, hold)
        cost += c
        ret += r
    return cost, ret


# ── 期間の用意 ───────────────────────────────────────────────────────────


@dataclass
class Period:
    name: str
    races: list[Race]
    feats: pd.DataFrame
    W: np.ndarray
    drop: dict

    @property
    def frecs(self) -> list[dict]:
        if not hasattr(self, "_frecs"):
            self._frecs = self.feats.to_dict("records")
        return self._frecs

    @property
    def dates(self) -> np.ndarray:
        return self.feats["date"].to_numpy()


def load_inputs(path: Path | None = None) -> dict:
    return json.loads((path or CFG_DIR / "inputs.json").read_text(encoding="utf-8"))


def prepare_periods(inputs: dict, names=("train", "eval"), cuts: dict | None = None, n_boot: int = 1000) -> tuple[dict[str, Period], dict]:
    """期間ごとの Race・特徴を作る。分位の区切りは訓練の期間で決め（cuts が無ければ）、他の期間にそのまま当てる。"""
    per = {}
    for nm in names:
        spec = inputs["periods"][nm]
        years = spec["years"]
        pred = pd.concat([load_pred(ROOT / p if not str(p).startswith("/") else Path(p)) for p in spec["pred"]], ignore_index=True)
        pred = pred[pred["race_id"].str[:4].astype(int).isin(years)]
        flat = load_runners(years)
        pay = load_payoffs(set(flat["race_id"]))
        oo = None
        if spec.get("pre_odds"):
            oo = pre_odds_override(Path(spec["pre_odds"]) if str(spec["pre_odds"]).startswith("/") else ROOT / spec["pre_odds"],
                                   spec.get("pre_phase", ["t02", "t05", "t10", "t15", "t60"]))
        races, cond, drop = build_races(pred, flat, pay, years, odds_override=oo)
        drop["all_races_in_period"] = int(flat["race_id"].nunique())      # 期間の全レース（予測・払戻がそろわないものを含む）
        drop["computed_races"] = len(races)
        ft = feature_table(races, cond)
        per[nm] = (races, ft, drop)
    if cuts is None:
        cuts = fit_qcuts(per["train"][1]) if "train" in per else {}
    out = {}
    for nm, (races, ft, drop) in per.items():
        ft = apply_qcuts(ft, cuts)
        out[nm] = Period(nm, races, ft, poisson_weights(len(races), n_boot, seed=zlib.crc32(nm.encode()) % 1000), drop)
    return out, cuts


def pre_odds_override(path: Path, phases) -> pd.DataFrame:
    """発走前のオッズ（pre_odds_table の出力）から、レースごとに一番遅い時点（phases の順）の単勝オッズ・人気を取る。"""
    p = pd.read_parquet(path)
    p["race_id"] = p["race_id"].astype(str)
    p["horse_number"] = pd.to_numeric(p["horse_number"], errors="coerce")
    p["win_odds"] = pd.to_numeric(p["win_odds"], errors="coerce")
    p["rk"] = p["phase"].map({ph: i for i, ph in enumerate(phases)})
    p = p[p["rk"].notna() & (p["win_odds"] > 0)]
    best = p.groupby("race_id")["rk"].min().rename("best")
    p = p.join(best, on="race_id")
    p = p[p["rk"] == p["best"]]
    p["popularity"] = p.groupby("race_id")["win_odds"].rank(method="min")
    return p.rename(columns={"win_odds": "odds"})[["race_id", "horse_number", "odds", "popularity"]]


# ── サイクル ─────────────────────────────────────────────────────────────


def expand_variants(cyc: dict) -> list[dict]:
    """サイクルの定義から試す買い方の一覧を作る。variants（そのまま）と grid（base × 軸の総当たり）。"""
    out = list(cyc.get("variants") or [])
    grids = list(cyc.get("grids") or [])
    for inc in cyc.get("include") or []:                       # 候補の束（config/bet_cycles/pool_*.json）を読み込む
        sub = json.loads((CFG_DIR / inc).read_text(encoding="utf-8"))
        out += list(sub.get("variants") or [])
        grids += list(sub.get("grids") or [])
    for g in grids:
        keys = list(g["axes"])
        for vals in itertools.product(*(g["axes"][k] for k in keys)):
            s = json.loads(json.dumps(g["base"]))
            for k, v in zip(keys, vals):
                _set(s, k, v)
            s["name"] = g.get("prefix", "g") + "|" + "|".join(f"{k.split('.')[-1]}={_short(v)}" for k, v in zip(keys, vals))
            out.append(s)
    seen, uniq = set(), []
    for s in out:
        h = strategy_hash({k: v for k, v in s.items() if k != "name"})
        if h not in seen:
            seen.add(h)
            uniq.append(s)
    return uniq


def _short(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def _set(d: dict, path: str, v) -> None:
    ks = path.split(".")
    for k in ks[:-1]:
        if k.isdigit():
            d = d[int(k)]
        else:
            d = d.setdefault(k, {})
    last = ks[-1]
    if last.isdigit():
        d[int(last)] = v
    else:
        d[last] = v


def summarize_variant(s: dict, per: dict[str, Period], res: dict[str, dict]) -> dict:
    row = {"name": s["name"], "hash": strategy_hash({k: v for k, v in s.items() if k != "name"}), "spec": s}
    for nm, p in per.items():
        r = res[nm]
        row[nm] = metrics(r["cost"], r["ret"], p.dates, p.W)
        row[nm]["violation_races"] = r["n_violation_races"]
        row[nm]["violations"] = r["violations"]
        row[nm]["h_not_top_rate"] = float(((r["hrank"] > 0).sum()) / max((r["hrank"] >= 0).sum(), 1))
        row[nm]["h_fav1_rate"] = float(((r["hpop"] == 1) & (r["hrank"] >= 0)).sum() / max((r["hrank"] >= 0).sum(), 1))   # ◎ が 1 番人気（確定オッズの順）
        if nm == "train":                                             # 訓練の中の fold（年）ごとの回収率
            yrs = p.feats["year"].to_numpy()
            row["train_folds"] = {}
            for y in sorted(set(yrs)):
                mk = yrs == y
                c, rr = r["cost"][mk], r["ret"][mk]
                row["train_folds"][str(y)] = float(rr.sum() / c.sum()) if c.sum() > 0 else None
    return row


def _eval_one(args):
    s, per_names = args
    per = _GLOBAL_PER
    return s, {nm: run_strategy(per[nm].races, per[nm].frecs, s) for nm in per_names}


_GLOBAL_PER: dict[str, Period] = {}


def run_many(strats: list[dict], per: dict[str, Period], threads: int = 4) -> list[tuple[dict, dict]]:
    """多くの買い方をまとめて実行する（fork のプロセスで並べる。スレッドは 4 以下）。"""
    global _GLOBAL_PER
    _GLOBAL_PER = per
    names = list(per)
    if threads <= 1 or len(strats) < 4:
        return [_eval_one((s, names)) for s in strats]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(min(threads, 4)) as pool:
        return pool.map(_eval_one, [(s, names) for s in strats], chunksize=max(1, len(strats) // (threads * 4)))


def eligible(row: dict, base: dict | None = None, min_hit: float = 0.0) -> bool:
    """候補に残す条件（cycle 1 の結果を見た後に決めた。記録に書く）:
    参戦 300R 以上・印の違反なし・訓練の 2 年とも基準（B0）の回収率より上・払戻の上位 5 件を除いた回収率も基準以上。
    順位は訓練の回収率の区間の下限（roi_lo）で付ける（振れの大きい買い方が偶然で勝つのを抑える）。"""
    t = row["train"]
    if t.get("bet_races", 0) < MIN_BET_RACES or t.get("violation_races", 0) != 0:
        return False
    if t.get("hit_rate", 0) < min_hit:          # cycle 8〜: 的中率の下限（的中 3% 前後の 1 点買いは評価で崩れやすいので主にしない）
        return False
    if base is None:
        return True
    b = base["train"]
    if t.get("roi_ex_top5", 0) < b.get("roi_ex_top5", 0):
        return False
    bf, rf = base.get("train_folds") or {}, row.get("train_folds") or {}
    return all(rf.get(y) is not None and rf[y] > v for y, v in bf.items() if v is not None)


RANK_BY = {"roi_lo": "訓練の回収率の 95% 区間の下限", "robust": "上位 5 件を除いた回収率と、訓練の年ごとの回収率の小さい方（の小さい方）"}


def rank_key(row: dict, by: str = "roi_lo") -> float:
    t = row["train"]
    if by == "robust":
        folds = [v for v in (row.get("train_folds") or {}).values() if v is not None]
        return float(min([t.get("roi_ex_top5", -1.0)] + folds))
    return float(t.get("roi_lo", -1.0))


def family(s: dict) -> str:
    """買い方の系統（前の最良は系統ごとに持ち、改善は同じ系統の中で判定する）。"""
    if s.get("family"):
        return s["family"]
    if "policy" in s:
        return "方策（型 → 買い方）"
    if any(t["type"].startswith("box_") for t in s.get("tickets") or []):
        return "参考: ◎ なし（ボックス）"
    h = s.get("honmei") or {}
    rule = h.get("rule", "top")
    if rule == "switch":
        return "◎ の切り替え（型ごと）"
    if rule in ("fav1_penalty", "fav1_exc"):          # 2 番人気以下の ◎（ユーザー 2026-11-16 の方針。主な系統）
        if rule == "fav1_exc":
            b = h.get("base") or {"rule": "fav1_penalty"}
            if b.get("rule") in ("ana_ev", "ana_p3", "ev_max") and not any(k in b for k in ("max_rank", "p3_min", "pw_min")):
                return "参考: 制約なしの穴・期待値の ◎"
        mode = (s.get("partners") or {}).get("mode", "p3")
        tail = {"fav": "人気の相手", "mix": "相手を広く"}.get(mode, "複勝率上位の相手")
        head = "2番人気以下の ◎（1番人気は例外のみ）" if rule == "fav1_exc" else "1番人気にペナルティ"
        return f"{head}・{tail}"
    if rule in ("top", "fav_maxodds", "market_fav") and not (s.get("skip") or {}).get("h_pop_min"):
        return "型1 人気の ◎（点数を絞る）"
    if rule in ("ana_ev", "ana_p3", "ev_max") and not any(k in h for k in ("max_rank", "p3_min", "pw_min")):
        return "参考: 制約なしの穴・期待値の ◎"
    return "型A 穴の ◎・人気の相手" if (s.get("partners") or {}).get("mode") == "fav" else "型B 穴の ◎・相手を広く"


def perturb_period(p: Period, sigma: float, seed: int, cuts: dict) -> Period:
    """◎ の決め方・見送りに使う単勝オッズ（と人気）にだけ対数正規の揺れ σ を足した期間（精算は確定の払戻のまま）。
    買う時点のオッズと確定オッズのずれの代わり（tp-50 §6 と同じ方法）。"""
    rng = np.random.default_rng(seed)
    races = []
    for r in p.races:
        o = r.odds * np.exp(rng.normal(0.0, sigma, len(r.odds)))
        pop = np.where(np.isfinite(o), pd.Series(o).rank(method="min").to_numpy(), 99.0)
        races.append(Race(r.race_id, r.date, r.hn, r.pw, r.p3, o, pop, r.fin, r.pay, r.p2))
    f = pd.DataFrame([race_features(r) for r in races])
    keep = [c for c in p.feats.columns if c not in f.columns and not c.startswith("q_")]
    ft = pd.concat([p.feats[keep].reset_index(drop=True), f], axis=1)
    return Period(f"{p.name}_s{sigma}", races, apply_qcuts(ft, cuts), p.W, p.drop)


def odds_robustness(spec: dict, p: Period, cuts: dict, sigmas=(0.1, 0.2), seeds=(0, 1, 2)) -> dict:
    out = {}
    for sg in sigmas:
        rois = []
        for sd in seeds:
            q = perturb_period(p, sg, sd, cuts)
            r = run_strategy(q.races, q.frecs, spec)
            rois.append(float(r["ret"].sum() / max(r["cost"].sum(), 1)))
        out[str(sg)] = {"mean": float(np.mean(rois)), "min": float(np.min(rois)), "rois": rois}
    return out


def judge_eval(sel: dict, base: dict, per_e: Period, res_sel: dict, res_base: dict) -> dict:
    """評価の期間での「上向き」の確かめ（順位付けには使わない）。"""
    d, lo, hi = paired_diff(res_sel, res_base, per_e.W)
    roi = sel["eval"].get("roi")
    if roi is None:
        verdict = "確かめられず（参戦なし）"
    elif d > 0 and lo > 0:
        verdict = "上向き（基準との差の区間も 0 より上）"
    elif d > 0:
        verdict = "上向き（点推定のみ。区間は 0 をまたぐ）"
    else:
        verdict = "確かめられず（基準より下）"
    return {"diff_vs_base": d, "diff_lo": lo, "diff_hi": hi, "verdict": verdict, "over_100": bool(roi is not None and roi > 1.0)}


def _arrays(res: dict) -> dict:
    return {"cost": res["cost"], "ret": res["ret"]}


def policy_search(ps: dict, var_rows: list[dict], res_list: list[dict], per: dict[str, Period]) -> list[dict]:
    """型 × 買い方の格子。候補（このサイクルの買い方のうち違反の無いもの）から、型ごとに訓練で最良を選ぶ方策を作る。
    選ぶ物差しは訓練の中の年の交差検証の回収率（ある年で選んで、もう一方の年に当てた値）。"""
    ok = [i for i, r in enumerate(var_rows) if r["train"]["violation_races"] == 0 and r["train"].get("bet_races", 0) > 0]
    Ct = np.stack([res_list[i]["train"]["cost"] for i in ok])
    Rt = np.stack([res_list[i]["train"]["ret"] for i in ok])
    Ce = np.stack([res_list[i]["eval"]["cost"] for i in ok])
    Re = np.stack([res_list[i]["eval"]["ret"] for i in ok])
    tr, ev = per["train"], per["eval"]
    years = tr.feats["year"].to_numpy()
    allmask_t, allmask_e = np.ones(len(tr.races), bool), np.ones(len(ev.races), bool)
    rows = []
    for ty in ps["typings"]:
        fty = fit_typing(ty, tr.feats, Rt, Ct)
        lt = np.array([typing_label(f, fty) for f in tr.frecs])
        le = np.array([typing_label(f, fty) for f in ev.frecs])
        for mc, sb in itertools.product(ps.get("min_cell", [150]), ps.get("skip_below", [None])):
            cc, cr = cv_policy(ty, tr.feats, tr.frecs, years, Rt, Ct, max(1, mc // 2), sb)
            cells = choose_cells(lt, Rt, Ct, allmask_t, mc, sb)
            it_c, it_r = apply_cells(lt, cells, Rt, Ct, allmask_t)
            ev_c, ev_r = apply_cells(le, cells, Re, Ce, allmask_e)
            name = f"policy|{ty.get('name', ty['kind'])}|min_cell={mc}|skip_below={sb}"
            pmap = {lv: (var_rows[ok[j]]["spec"] if j >= 0 else "skip") for lv, j in cells.items()}
            spec = {"name": name, "policy": {"typing": fty, "map": pmap, "default": "skip"}}
            cell_tab = []
            for lv in sorted(set(lt)):
                mt, me = lt == lv, le == lv
                j = cells.get(str(lv), -1)
                cm = metrics(np.where(mt, cc, 0), np.where(mt, cr, 0), tr.dates)
                cell_tab.append({"type": str(lv), "races_train": int(mt.sum()), "races_eval": int(me.sum()),
                                 "chosen": var_rows[ok[j]]["name"] if j >= 0 else "見送り",
                                 "train_insample": metrics(np.where(mt, it_c, 0), np.where(mt, it_r, 0), tr.dates),
                                 "train_cv": {k: cm.get(k) for k in ("bet_races", "hit_rate", "roi")},
                                 "eval": {k: metrics(np.where(me, ev_c, 0), np.where(me, ev_r, 0), ev.dates).get(k)
                                          for k in ("bet_races", "hit_rate", "roi")}})
            row = {"name": name, "hash": strategy_hash(spec["policy"]), "spec": spec, "kind": "policy",
                   "train": metrics(cc, cr, tr.dates, tr.W), "train_insample": metrics(it_c, it_r, tr.dates, tr.W),
                   "eval": metrics(ev_c, ev_r, ev.dates, ev.W), "cells": cell_tab, "n_types": int(len(set(lt))),
                   "train_folds": {str(y): (float(cr[years == y].sum() / cc[years == y].sum()) if cc[years == y].sum() > 0 else None)
                                   for y in sorted(set(years))}}
            row["train"]["violation_races"] = 0
            row["_arr"] = {"train": {"cost": cc, "ret": cr}, "eval": {"cost": ev_c, "ret": ev_r}}
            rows.append(row)
    return rows


def _prev_selected(state: dict, n: int) -> dict | None:
    prev = [int(k) for k in state.get("cycles", {}) if int(k) < n and state["cycles"][k].get("selected")]
    return state["cycles"][str(max(prev))]["selected"] if prev else None


def _prev_best(state: dict, n: int) -> dict | None:
    prev = [int(k) for k in state.get("cycles", {}) if int(k) < n and state["cycles"][k].get("best_after")]
    return state["cycles"][str(max(prev))]["best_after"] if prev else None


def _group_best(var_rows: list[dict], key: str, base: dict | None = None, min_hit: float = 0.0) -> list[dict]:
    """key（例: honmei.rule）ごとに、訓練で一番良い買い方（参戦 300R 以上・違反なし）。◎ の決め方ごとの回顧に使う。"""
    def get(spec, path):
        d = spec
        for k in path.split("."):
            if isinstance(d, list):
                d = d[int(k)] if k.isdigit() else None
            elif isinstance(d, dict):
                d = d.get(k)
            if d is None:
                return "（なし）"
        return d if isinstance(d, str) else json.dumps(d, ensure_ascii=False)
    best: dict[str, dict] = {}
    for r in var_rows:                                   # 回顧用: 系統の条件（eligible）は課さず、参戦 300R 以上・違反なしの中の訓練の最良
        if not eligible(r):
            continue
        g = get(r["spec"], key)
        if g not in best or r["train"]["roi"] > best[g]["train"]["roi"]:
            best[g] = r
    return [{"group": g, "passes_filter": bool(eligible(r, base, min_hit)), **_slim([r], full=False)[0]}
            for g, r in sorted(best.items(), key=lambda x: -x[1]["train"]["roi"])]


def cmd_run(a) -> None:
    cyc = json.loads(Path(a.cycle).read_text(encoding="utf-8"))
    n = int(cyc["n"])
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out = OUT_DIR / inputs.get("tag", "v6e")
    out.mkdir(parents=True, exist_ok=True)
    tr_spec = inputs["periods"]["train"]
    cuts_path = out / f"qcuts_train_{strategy_hash({'years': tr_spec['years'], 'pred': tr_spec['pred']})}.json"   # 訓練の年・予測が変われば別の区切り
    cuts = json.loads(cuts_path.read_text()) if cuts_path.exists() else None
    per, cuts = prepare_periods(inputs, ("train", "eval"), cuts)
    if not cuts_path.exists():
        cuts_path.write_text(json.dumps(cuts, indent=1))
    strats = expand_variants(cyc)
    baselines = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"]
    print(f"cycle {n}: 買い方 {len(strats)} 本（基準 {len(baselines)} 本）、訓練 {len(per['train'].races)}R・評価 {len(per['eval'].races)}R", flush=True)
    allres = run_many(baselines + strats, per, threads=a.threads)
    rows = [summarize_variant(s, per, r) for s, r in allres]
    base_rows, var_rows = rows[: len(baselines)], rows[len(baselines):]
    for r, (_, res) in zip(rows, allres):
        r["_arr"] = {nm: _arrays(res[nm]) for nm in res}
    var_res = [res for _, res in allres[len(baselines):]]
    state_path = out / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"cycles": {}}
    state.setdefault("cycles", {})
    base0 = base_rows[0]
    min_hit = float(cyc.get("min_hit", 0.0))
    rank_by = cyc.get("rank_by", "roi_lo")
    cands = [r for r in var_rows if eligible(r, base0, min_hit)]
    pol_rows = []
    if cyc.get("policy_search"):
        pol_rows = policy_search(cyc["policy_search"], var_rows, var_res, per)
        for r in pol_rows:
            r["train"]["violation_races"] = 0
        cands = [r for r in pol_rows if eligible(r, base0, min_hit)]
        if cyc["policy_search"].get("include_fixed", True):
            cands += [r for r in var_rows if eligible(r, base0, min_hit)]
    for r in var_rows + pol_rows:
        r["family"] = family(r["spec"])
    cands.sort(key=lambda r: -rank_key(r, rank_by))
    main_cands = [r for r in cands if not r["family"].startswith("参考")]   # 参考の系統（◎ なし・制約なしの穴）は並べるだけで選ばない
    sel = main_cands[0] if main_cands else None
    n_tried = len(var_rows) + len(pol_rows)
    prev_fam = _prev_best(state, n) or {}
    state["cycles"][str(n)] = {"n_tried": n_tried}
    rec: dict[str, Any] = {"n": n, "title": cyc.get("title", ""), "hypothesis": cyc.get("hypothesis", ""), "change": cyc.get("change", ""),
                           "n_variants": len(var_rows), "n_policies": len(pol_rows), "n_eligible": len(main_cands), "n_eligible_ref": len(cands) - len(main_cands),
                           "baselines": _slim(base_rows, with_eval=True), "drop": {k: p.drop for k, p in per.items()}, "qcuts": cuts,
                           "inputs_tag": inputs.get("tag"), "notes": cyc.get("notes", {}),
                           "selection_rule": f"候補: 参戦300R以上・違反なし・訓練の2年とも B0 より上・上位5件を除いた回収率も B0 以上・的中率 {min_hit:.0%} 以上。"
                                             f"順位: {RANK_BY.get(rank_by, rank_by)}。改善: 同じ系統の前の最良との対の差の区間が 0 より上。"}
    if cyc.get("group_by"):
        rec["groups"] = {g: _group_best(var_rows, g, base0, min_hit) for g in cyc["group_by"]}
    # 系統ごとの最良（訓練の区間の下限で順位）と、同じ系統の前の最良との比べ
    fam_best: dict[str, dict] = {}
    for r in cands:
        fam_best.setdefault(r["family"], r)
    best_after = dict(prev_fam)
    rec["families"] = []
    for fam, r in fam_best.items():
        x = {"family": fam, **_slim([r])[0]}
        pv = prev_fam.get(fam)
        if pv is not None and float(pv.get("train_hit", 1.0)) < min_hit:   # 前の最良が今の的中率の下限を満たさないなら比べない
            x["prev_dropped"] = f"前の最良（cycle {pv['cycle']}）は的中率 {float(pv.get('train_hit', 0)):.1%} で下限 {min_hit:.0%} 未満のため比べない"
            pv = None
        if pv is not None:
            pr = _load_best_arrays(out, pv) or run_many([pv["spec"]], {"train": per["train"]}, threads=1)[0][1]["train"]
            d = paired_diff(r["_arr"]["train"], pr, per["train"].W)
            x["prev"] = {"cycle": pv["cycle"], "name": pv["name"], "train_roi": float(pr["ret"].sum() / max(pr["cost"].sum(), 1))}
            x["vs_prev"] = d
            x["improved"] = bool(d[1] > 0)
        else:
            x["improved"] = True
        if x["improved"]:
            best_after[fam] = {"cycle": n, "name": r["name"], "spec": r["spec"], "train_roi": r["train"]["roi"], "train_roi_lo": rank_key(r, "roi_lo"), "train_hit": r["train"].get("hit_rate"),
                               "arrays": _save_best_arrays(out, n, r)}
        rec["families"].append(x)
    state["cycles"][str(n)]["best_after"] = best_after
    if sel is not None:
        rs, rb = sel["_arr"], base0["_arr"]
        rec["selected"] = _slim([sel], with_eval=True)[0]
        rec["selected"]["family"] = sel["family"]
        if sel.get("kind") == "policy":
            rec["selected"].update(kind="policy", cells=sel["cells"], train_insample=sel["train_insample"], n_types=sel["n_types"])
        rec["selected_vs_base_train"] = paired_diff(rs["train"], rb["train"], per["train"].W)
        rec["eval_check"] = judge_eval(sel, base0, per["eval"], rs["eval"], rb["eval"])
        rec["improved"] = next((x["improved"] for x in rec["families"] if x["family"] == sel["family"]), False)
        ps = _prev_selected(state, n)                     # 系統を乗り換えたときも、前のサイクルで選んだ 1 本との差を残す
        if ps is not None:
            pr = _load_best_arrays(out, ps) or run_many([ps["spec"]], {"train": per["train"]}, threads=1)[0][1]["train"]
            rec["vs_prev_selected"] = {"cycle": ps["cycle"], "name": ps["name"], "family": ps.get("family"),
                                       "train_roi": float(pr["ret"].sum() / max(pr["cost"].sum(), 1)),
                                       "diff": paired_diff(rs["train"], pr, per["train"].W)}
        state["cycles"][str(n)]["selected"] = {"cycle": n, "name": sel["name"], "family": sel["family"], "spec": sel["spec"],
                                               "arrays": _save_best_arrays(out, n, sel)}
        rec["odds_robustness_train"] = odds_robustness(sel["spec"], per["train"], cuts)
        # 回顧（訓練の期間だけ。方策は訓練全体で当てはめた版＝標本内）
        rr = run_strategy(per["train"].races, per["train"].frecs, sel["spec"], detail=True)
        rec["retro"] = retrospective(rr, per["train"].feats, per["train"].W)
        rec["miss_types"] = miss_types(rr)
        rec["h_reasons"] = rr["records"]["reason"].value_counts().to_dict()
        rr["records"].to_parquet(out / f"cycle_{n:02d}_selected_train_races.parquet", index=False)
    rec["n_tried_cum"] = int(sum(v.get("n_tried", 0) for k, v in state["cycles"].items() if int(k) <= n))
    rec["top"] = _slim(cands[:15])
    for t, c in zip(rec["top"], cands[:15]):
        t["family"] = c["family"]
        if c.get("kind") == "policy":
            t.update(kind="policy", cells=_cells_no_eval(c["cells"]), train_insample=c["train_insample"])
    rec["all"] = _slim(var_rows, full=False) + _slim(pol_rows, full=False)
    for t, c in zip(rec["all"], var_rows + pol_rows):
        t["family"] = c["family"]
        t["eligible"] = bool(eligible(c, base0, min_hit))
    (out / f"cycle_{n:02d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    _print_rec(rec)


def _save_best_arrays(out: Path, n: int, r: dict) -> str:
    """系統の最良の訓練の cost・ret（方策は交差検証の値）を保存する。次のサイクルの比べを、同じ物差し（標本外どうし）で行うため。"""
    f = out / f"best_c{n:02d}_{r['hash']}.npz"
    np.savez_compressed(f, cost=r["_arr"]["train"]["cost"], ret=r["_arr"]["train"]["ret"])
    return f.name


def _load_best_arrays(out: Path, pv: dict) -> dict | None:
    f = out / str(pv.get("arrays") or "")
    if not pv.get("arrays") or not f.exists():
        return None
    z = np.load(f)
    return {"cost": z["cost"], "ret": z["ret"]}


def _cells_no_eval(cells: list[dict]) -> list[dict]:
    return [{k: v for k, v in c.items() if k != "eval"} for c in cells]


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def _slim(rows: list[dict], full: bool = True, with_eval: bool = False) -> list[dict]:
    """記録用に軽くする。評価の期間の値は、選んだ 1 本と基準にだけ付ける（with_eval）。評価で選び直さない作法を記録の作りで守る。"""
    out = []
    for r in rows:
        if full:
            x = {"name": r["name"], "hash": r["hash"], "train": r["train"], "train_folds": r.get("train_folds"), "spec": r["spec"]}
        else:
            x = {"name": r["name"], "hash": r["hash"], "train_folds": r.get("train_folds"),
                 "train": {k: r["train"].get(k) for k in ("bet_races", "bet_rate", "hit_rate", "roi", "roi_lo", "roi_hi", "avg_return",
                                                         "median_return", "median_hit_return", "roi_ex_top5", "violation_races", "h_fav1_rate")}}
        if with_eval:
            x["eval"] = r.get("eval", {})
        out.append(x)
    return out


def _fmt(m: dict) -> str:
    if not m or not m.get("bet_races"):
        return "参戦なし"
    return (f"参戦 {m['bet_races']}/{m['races']}（{m['bet_rate']:.0%}） 的中 {m['hit_rate']:.1%} 回収 {m['roi']:.3f}"
            f"［{m.get('roi_lo', float('nan')):.3f}〜{m.get('roi_hi', float('nan')):.3f}］ 平均 {m['avg_return']:.0f} 中央 {m['median_return']:.0f}"
            f"（的中時 {m['median_hit_return']:.0f}） 購入 {m['avg_cost']:.0f} 上位5除く {m['roi_ex_top5']:.3f}")


def _print_rec(rec: dict) -> None:
    print("基準:")
    for b in rec["baselines"]:
        print(f"  {b['name']}: 訓練 {_fmt(b['train'])}\n      評価 {_fmt(b['eval'])}")
    print("系統ごとの最良（訓練の区間の下限で順位）:")
    for x in rec.get("families", []):
        print(f"  [{x['family']}] {x['name']}: 訓練 {_fmt(x['train'])} fold {x['train_folds']} 改善 {x['improved']} {x.get('vs_prev', '')}")
    if "selected" in rec:
        s = rec["selected"]
        print(f"選んだ: [{s['family']}] {s['name']}  fold {s['train_folds']}")
        print(f"  訓練 {_fmt(s['train'])}\n  評価 {_fmt(s['eval'])}")
        print(f"  評価の確かめ: {rec['eval_check']}  改善: {rec['improved']}")
        print(f"  オッズの揺れ（訓練）: {rec['odds_robustness_train']}")
        print(f"  ◎ の理由: {rec['h_reasons']}  不的中: {rec['miss_types']}")
    print(f"上位（訓練の区間の下限）: 試した {rec['n_variants']} 本・方策 {rec['n_policies']} 本・条件を満たす {rec['n_eligible']} 本・累計 {rec['n_tried_cum']} 本")
    for t in rec["top"][:10]:
        print(f"  [{t['family']}] {t['name']}: 訓練 {t['train']['roi']:.3f}（下限 {t['train'].get('roi_lo', 0):.3f}・{t['train']['bet_races']}R・"
              f"的中 {t['train']['hit_rate']:.1%}） fold {t['train_folds']}")


# ── 1 番人気を ◎ にしてよい例外（ユーザー 2026-11-16: 「どうしても 1 人気を 1 番手に推奨するときには、ほぼすべてが勝ち負けに匹敵する成績であるべき」） ──

FAV1_TARGET = 0.85        # 例外のレースで、1 番人気の連対率が「ほぼすべて」の目安（推奨で決めた。80〜90% の中ほど）
FAV1_MIN_N = 30


def choose_fav1_threshold(grid: list[dict], target: float = FAV1_TARGET, min_n: int = FAV1_MIN_N) -> dict | None:
    """しきい値の選び方: そのしきい値と、それより厳しい（件数が min_n 以上の）すべてのしきい値で連対率が target 以上になる、最も緩いもの。
    1 点だけ振れて満たしたしきい値を選ばないため（グリッドの連対率は単調でない）。"""
    g = sorted([x for x in grid if x.get("n", 0) >= min_n], key=lambda x: x["p2_min"])
    for i, x in enumerate(g):
        if all(y.get("rentai", 0) >= target for y in g[i:]):
            return x
    return None


def fav1_rows(p: Period) -> pd.DataFrame:
    """レースごとの 1 番人気（確定オッズの順）の予測の連対率・勝率と、着順・単複の払戻。"""
    rows = []
    for r, f in zip(p.races, p.frecs):
        j = int(np.argmin(r.pop))
        if r.pop[j] != 1:
            continue
        others = np.delete(r.pw, j)
        h = str(int(r.hn[j]))
        rows.append({"race_id": r.race_id, "date": r.date, "p2": float(r.p2[j]), "pw": float(r.pw[j]), "gap": float(r.pw[j] - others.max()),
                     "is_top": int(j == 0), "fin": float(r.fin[j]), "tansho": r.pay["tansho"].get(h, 0), "fukusho": r.pay["fukusho"].get(h, 0)})
    return pd.DataFrame(rows)


def fav1_stats(d: pd.DataFrame) -> dict:
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "rentai": float((d["fin"] <= 2).mean()), "win": float((d["fin"] == 1).mean()), "place": float((d["fin"] <= 3).mean()),
            "roi_tansho": float(d["tansho"].sum() / (100 * len(d))), "roi_fukusho": float(d["fukusho"].sum() / (100 * len(d)))}


def cmd_fav1(a) -> None:
    """例外のしきい値（1 番人気の予測の連対率 p2 の下限）を訓練で決め、評価で連対率が保たれるかを確かめる。"""
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out = OUT_DIR / inputs.get("tag", "v6e")
    out.mkdir(parents=True, exist_ok=True)
    per, _ = prepare_periods(inputs, ("train", "eval"), cuts={}, n_boot=10)
    tr, ev = fav1_rows(per["train"]), fav1_rows(per["eval"])
    grid = []
    for t in np.round(np.arange(0.40, 0.951, 0.025), 3):
        st = fav1_stats(tr[tr["p2"] >= t])
        st["p2_min"] = float(t)
        grid.append(st)
    chosen = choose_fav1_threshold(grid)
    res = {"target_rentai": FAV1_TARGET, "min_n": FAV1_MIN_N, "train_all_fav1": fav1_stats(tr), "eval_all_fav1": fav1_stats(ev), "grid_train": grid,
           "chosen": chosen}
    if chosen:
        t = chosen["p2_min"]
        res["train_exception"] = fav1_stats(tr[tr["p2"] >= t])
        res["eval_exception"] = fav1_stats(ev[ev["p2"] >= t])
        res["train_rest"] = fav1_stats(tr[tr["p2"] < t])
        res["eval_rest"] = fav1_stats(ev[ev["p2"] < t])
        res["train_races"], res["eval_races"] = len(per["train"].races), len(per["eval"].races)
    fmt = lambda x: "－" if x is None else f"{x * 100:.1f}%"   # noqa: E731
    row = lambda lab, s_: [lab, f"{s_.get('n', 0):,}", fmt(s_.get("rentai")), fmt(s_.get("win")), fmt(s_.get("place")),   # noqa: E731
                           fmt(s_.get("roi_tansho")), fmt(s_.get("roi_fukusho"))]
    head = ["区分", "レース", "1番人気の連対率", "勝率", "複勝率", "単勝の回収率", "複勝の回収率"]
    tables = [{"title": "しきい値ごと（訓練 2022〜23。1 番人気の予測の連対率 ≥ しきい値のレース）", "head": ["しきい値"] + head[1:],
               "rows": [[f"{g['p2_min']:.3f}"] + row("", g)[1:] for g in grid if g["n"] > 0]}]
    if chosen:
        tables.insert(0, {"title": f"選んだしきい値 {chosen['p2_min']:.3f}（連対率 {FAV1_TARGET:.0%} 以上・{FAV1_MIN_N} 件以上で最も緩いもの）", "head": head,
                          "rows": [row("訓練: 例外（1 番人気を ◎ にしてよい）", res["train_exception"]), row("訓練: それ以外の 1 番人気", res["train_rest"]),
                                   row("訓練: 1 番人気すべて", res["train_all_fav1"]), row("評価: 例外（確かめ）", res["eval_exception"]),
                                   row("評価: それ以外の 1 番人気", res["eval_rest"]), row("評価: 1 番人気すべて", res["eval_all_fav1"])]})
    text = [f"ユーザー（2026-11-16）: 「1 人気を 1 番手に推奨するのはペナルティを設けたい。2 人気以降の馬から本命馬を選びたい。」「どうしても 1 人気を 1 番手に推奨するときには、"
            f"そのほぼすべてが勝ち負けに匹敵する成績であるべき。市場の注目度の穴をねらうべき。」",
            f"例外の規則: 1 番人気（確定オッズの順。発走前のオッズの人気とはずれうる）の予測の連対率が、しきい値以上のときだけ、その馬を ◎ にしてよい。"
            f"しきい値は訓練 2022〜23 で『そのしきい値と、それより厳しいすべてのしきい値（{FAV1_MIN_N} 件以上）で、例外にしたレースの 1 番人気の連対率が {FAV1_TARGET:.0%} 以上（目安 80〜90% の中ほどを推奨で決めた）』を満たす最も緩い値。"
            f"評価 2024 は、そのしきい値で連対率が保たれるかを見るだけ。"]
    if not chosen:
        text.append(f"訓練では、連対率 {FAV1_TARGET:.0%} 以上を {FAV1_MIN_N} 件以上で満たすしきい値が無かった。例外は作らず、◎ はすべて 2 番人気以下から選ぶ。")
    res.update(title="1 番人気を ◎ にしてよい例外（連対率のしきい値）", text=text, tables=tables)
    (out / "stage_fav1_exception.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("chosen", "train_all_fav1", "eval_all_fav1") if k in res}, ensure_ascii=False, indent=1))
    for k in ("train_exception", "eval_exception", "train_rest", "eval_rest"):
        if k in res:
            print(k, res[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cycle", required=True)
    r.add_argument("--inputs", default=None)
    r.add_argument("--threads", type=int, default=4)
    r.set_defaults(func=cmd_run)
    f1 = sub.add_parser("fav1", help="1 番人気を ◎ にしてよい例外のしきい値を訓練で決める")
    f1.add_argument("--inputs", default=None)
    f1.set_defaults(func=cmd_fav1)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
