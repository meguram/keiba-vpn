"""推奨馬券の最適化サイクル（読み取りのみ・手元のデータだけ・費用 0 円）。

python -m src.scripts.data.bet_cycles run --cycle config/bet_cycles/cycle_01.json [--threads 4] [--inputs config/bet_cycles/inputs.json]
出力先: 環境変数 BET_CYCLES_OUT（既定 data/local/research/bet_cycles。root 所有で書けない PC では retrain_out などを指す）の <tag>/ に
cycle_XX.json（記録）・state.json（系統ごとの最良）・qcuts_train.json（訓練で凍結した分位の区切り）・cycle_XX_selected_train_races.parquet。

依頼（ユーザー、統括役経由 2026-11-16。指示書 /home/jovyan/work/progress/bet_optimizer_spec.md）:
勝率・複勝率の分布の型ごとに一番良い買い方（◎ 起点）を 20 サイクルで決め、得意なレースだけを推奨度の段に分けて推奨する。

作法（時間の漏れを防ぐ）:
- 予測は OOF（その年を学習に含まないモデルの予測）だけ。期間は config/bet_cycles/inputs.json の periods。
  訓練で探す・選ぶ・回顧する（最終モデル・第 2 回は 2019〜22、試しの回し v6e は 2022〜23）。評価（2023〜24、v6e は 2024）は「変わらず上向きか」の確かめだけ。test（2025）は凍結の後に 1 回だけ。
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
CYC_DIR = Path(os.environ.get("BET_CYCLES_CYC_DIR") or CFG_DIR)    # サイクルの設定の置き場（第 2 回は config/bet_cycles/r2）
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


import re as _re

_PAYOUT_RE = _re.compile(r"^\d{1,3}(,\d{3})*$|^\d+$")
JP_BET = {"単勝": "tansho", "複勝": "fukusho", "馬連": "umaren", "ワイド": "wide", "馬単": "umatan", "3連複": "sanrenpuku", "3連単": "sanrentan"}
ARITY = {"tansho": 1, "fukusho": 1, "umaren": 2, "wide": 2, "umatan": 2, "sanrenpuku": 3, "sanrentan": 3}


def parse_flat_payoff(raw) -> dict[str, dict[str, int]] | None:
    """race_result_flat.payoff（JSON の文字列）→ {bet_type: {組: 払戻}}。払戻の表（betting/payoff.parquet）に無い新しいレース用。
    7 券種のどれかが読めない、または複勝・ワイドが 1 行に連結された保存（rule_tracking.parse_payoffs と同じ判定）なら None。"""
    try:
        js = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError):
        return None
    out: dict[str, dict[str, int]] = {}
    for jp, bt in JP_BET.items():
        v = js.get(jp, js.get(jp.replace("3連", "三連")))   # 実データは「三連複・三連単」、馬単・三連単は着順どおりに "-" でつなぐ
        if v is None:
            return None
        if jp in ("複勝", "ワイド") and isinstance(v, dict) and "popularity" not in v:
            return None
        for e in (v if isinstance(v, list) else [v]):
            if not isinstance(e, dict):
                return None
            try:
                nums = [int(x) for x in str(e["numbers"]).replace("→", "-").replace(">", "-").replace(" ", "").split("-")]
                ps = str(e["payout"]).strip()
                if not _PAYOUT_RE.match(ps):                 # 金額の連結（例 "3,1903,470"）を弾く: カンマは 3 桁ごと
                    return None
                pay = int(ps.replace(",", ""))
            except (KeyError, ValueError, TypeError):
                return None
            if len(nums) != ARITY[bt] or max(nums) > 18:
                return None
            k = combo_key(bt, nums)
            out.setdefault(bt, {})[k] = pay          # 同じ組の重複保存は上書き（払戻の表の読みと同じ。同着は組が違うので別の行）
    return out


def fill_payoffs_from_flat(pay: dict, years) -> int:
    """払戻の表に無いレースを、race_result_flat の払戻の列で補う（在れば上書きしない）。補った件数を返す。"""
    from src.config.data_paths import TABLES_DIR

    n = 0
    for y in years:
        f = Path(TABLES_DIR) / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        t = pd.read_parquet(f, columns=["race_id", "payoff"]).drop_duplicates("race_id")
        for rid, raw in zip(t["race_id"].astype(str), t["payoff"]):
            if rid in pay and all(bt in pay[rid] for bt in BET_TYPES):
                continue
            x = parse_flat_payoff(raw)
            if x:
                pay[rid] = x
                n += 1
    return n


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
    q: np.ndarray | None = None    # 第 2 段のモデルの市場の確率（予測にあれば）
    r: np.ndarray | None = None    # 市場とのずれ r = log(p/q)（予測にあれば。第 2 段のモデル）


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


def pred_path(x) -> Path:
    """inputs の pred の 1 つ（文字列のパス、または {"path": …, "where": {列: 値 or [値]}}）のファイルのパス。"""
    p = x["path"] if isinstance(x, dict) else x
    return Path(p) if str(p).startswith("/") else ROOT / p


def load_pred(path, where: dict | None = None) -> pd.DataFrame:
    """予測を読む。where があれば、その列の値で絞る（第 2 段の 1 ファイルに part・odds_phase が並んでいる形）。"""
    import pyarrow.parquet as pq

    if isinstance(path, dict):
        where, path = path.get("where"), pred_path(path)
    names = pq.read_schema(path).names
    cols = ["race_id", "horse_id", "p_win"] + [c for c in ("p_top2", "p_top3", "q", "r") if c in names]
    if "part" in names and not where:
        raise ValueError(f"{path} は part・odds_phase の混ざった 1 ファイルです。inputs で where を指定してください")
    o = pd.read_parquet(path, columns=cols + [c for c in (where or {}) if c not in cols])
    for c, v in (where or {}).items():
        o = o[o[c].isin(v if isinstance(v, list) else [v])]
    o = o[cols]
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
        qv = g["q"].to_numpy(dtype=float) if "q" in g else None
        rv = g["r"].to_numpy(dtype=float) if "r" in g else (np.log(np.clip(pw, 1e-9, None) / np.clip(qv / np.nansum(qv), 1e-9, None)) if qv is not None else None)
        races.append(Race(rid, str(pd.Timestamp(g["date"].iloc[0]).date()), g["horse_number"].to_numpy(int), pw, p3,
                          g["odds"].to_numpy(float), g["popularity"].fillna(99).to_numpy(float),
                          g["finish_position"].fillna(99).to_numpy(float), pr, p2, qv, rv))
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
        **_r_features(r, pw, pop),
        "n_mk_under": int(np.nansum(pw - mk > 0.03)),
    }


def _r_features(r: "Race", pw: np.ndarray, pop: np.ndarray, p_min: float = 0.05) -> dict:
    """第 2 段のモデルの、市場とのずれ r の特徴（r が無ければ空）。r_max = 2 番人気以下・勝率 p_min 以上の中の r の最大（◎ の r_max の既定と同じ範囲）。"""
    if r.r is None:
        return {}
    rr = np.where(np.isfinite(r.r), r.r, -9.0)
    ok = (pw >= p_min) & (pop > 1) & (pop < 99) & np.isfinite(r.r)    # ◎ の r_max の既定（2 番人気以下・勝率 5% 以上）と同じ範囲
    j = int(np.argmax(np.where(ok, rr, -9.0))) if ok.any() else 0
    fav = int(np.argmin(pop))
    return {"r_max": float(rr[j]) if ok.any() else np.nan, "r_max_pop": float(pop[j]), "r_top": float(rr[0]), "r_fav1": float(rr[fav]),
            "n_r_pos": int(((rr > 0) & ok).sum())}


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


QCUT_FEATURES = ("p1", "gap12", "ratio12", "ent", "gini", "var5", "top3sum", "p3_top3sum", "ev_max", "ev_top", "mk_gap_top", "r_max", "r_top")


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
        if h.get("exc_by") == "r":                         # 第 2 段: 1 番人気の r（ずれ）がしきい値以上なら例外
            ok = r.pop[fav] == 1 and r.r is not None and np.isfinite(r.r[fav]) and r.r[fav] >= float(h.get("r_min", 9.0))
        else:
            ok = r.pop[fav] == 1 and r.p2 is not None and r.p2[fav] >= float(h.get("p2_min", 0.85))
        if ok and h.get("gap_min") is not None:
            others = np.delete(r.pw, fav)
            ok = r.pw[fav] - (others.max() if len(others) else 0.0) >= float(h["gap_min"])
        if ok:
            return fav, ("1番人気の例外（r が高い）" if h.get("exc_by") == "r" else "1番人気の例外（連対率の予測が高い）")
        sub = dict(h.get("base") or {"rule": "fav1_penalty", "lam": 1.0})
        i, why = pick_honmei(r, f, sub)
        if i is not None and r.pop[i] == 1:  # 下の決め方が 1 番人気を選んだら、2 番人気以下の勝率最大に替える
            i, why = pick_honmei(r, f, {"rule": "fav1_penalty", "lam": 1.0})
        return i, why
    if rule == "r_max":                      # 第 2 段: 勝率 p_min 以上の中で、市場とのずれ r = log(p/q) が最大の馬（fav_k 番人気以内は除く）
        if r.r is None:
            raise ValueError("予測に r（ずれ）がありません（第 2 段のモデルの予測を使う）")
        rr = np.where(np.isfinite(r.r), r.r, -9.0)
        k = int(h.get("fav_k", 1))
        c = (r.pw >= float(h.get("p_min", 0.05))) & (r.pop > k) & (r.pop < 99) & np.isfinite(r.r) & np.isfinite(r.odds) \
            & (rr >= float(h.get("r_min", -9.0))) & (np.arange(len(rr)) < int(h.get("max_rank", 99)))
        if not c.any():
            fb = h.get("fallback", "skip")
            return (None, "見送り（r の候補なし）") if fb == "skip" else (0, "予測1位（r の候補なし）")
        i = int(np.argmax(np.where(c, rr, -99.0)))
        return i, ("予測1位" if i == 0 else f"{k}番人気より下で r 最大")
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
    if sk.get("any_of"):                                             # any_of: どれかの条件（when・in・not_in の組）に当たるレースだけ買う
        bad = {k for cond in sk["any_of"] for k in cond} - {"when", "in", "not_in", "min_field", "any_of"}
        if bad:
            raise ValueError(f"any_of の中では使えない条件です: {sorted(bad)}（◎ に関わる条件は外側に書く）")
        if all(_skip(f, {"min_field": 0, **cond}) for cond in sk["any_of"]):
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
    top_pop = np.array([float(rc.pop[0]) for rc in races]) if races else np.zeros(0)   # 予測 1 位（p）の人気
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
    out = {"cost": cost, "ret": ret, "hrank": hrank, "hfin": hfin, "hpop": hpop, "o_in3": o_in3, "top_pop": top_pop, "n_violation_races": nviol,
           "violations": vkinds}
    if detail:
        out["records"] = pd.DataFrame(recs)
    return out


# ── 指標 ─────────────────────────────────────────────────────────────────


def poisson_weights(n: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """レース単位のブートストラップの重み（Poisson(1)）。同じ重みを全部の買い方に使うので、差は対の区間になる。"""
    return np.random.default_rng(seed).poisson(1.0, size=(n_boot, n)).astype(np.float32)


_REP: dict[int, int] = {}     # 本番の再現で「レース × 版」を並べた期間の長さ → 版の数（件数を版の数で割るため）


def rep_of(n: int) -> int:
    return _REP.get(int(n), 1)


def metrics(cost: np.ndarray, ret: np.ndarray, dates: np.ndarray, W: np.ndarray | None = None) -> dict:
    """必須の指標（参戦数・率、的中率、回収率、平均・中央値の回収額）と、区間・1 日の振れ・落ち込み・上位を除いた回収率。
    本番の再現（レース × 版を並べた期間）では、件数・合計・落ち込みを版の数で割り、上位を除く件数は版の数を掛ける。率は版を通した平均。"""
    k_ = rep_of(len(cost))
    n_all = len(cost)
    b = cost > 0
    nb = int(b.sum())
    m: dict[str, Any] = {"races": n_all // k_, "bet_races": int(round(nb / k_)), "bet_rate": nb / n_all if n_all else 0.0}
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
        m[f"roi_ex_top{k}"] = float((r.sum() - srt[:k * k_].sum()) / c.sum())
    m["cost_total"], m["return_total"] = m["cost_total"] / k_, m["return_total"] / k_
    if k_ > 1:                                       # 版ごとの回収率の幅（並べた期間は版ごとに同じ長さで続く）
        nn = n_all // k_
        rois = [float(ret[i * nn:(i + 1) * nn].sum() / max(cost[i * nn:(i + 1) * nn].sum(), 1e-9)) for i in range(k_)]
        m["sim_roi_min"], m["sim_roi_max"], m["n_versions"] = float(min(rois)), float(max(rois)), k_
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
              "max_drawdown": dd / k_, "max_drawdown_days": dd / float(d["c"].mean()), "plus_day_rate": float((d["r"] > d["c"]).mean()),
              "profit": float(r.sum() - c.sum()) / k_})
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
            if (c > 0).sum() < min_n * rep_of(len(c)):
                continue
            m = metrics(c, r, dates, W)
            m.pop("races", None)
            row = {"axis": ax, "axis_label": lab, "level": str(lvl), **m, "races_in_level": int(mask.sum()) // rep_of(len(mask))}
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
        tr = DecisionTreeRegressor(max_depth=int(ty.get("depth", 3)), min_samples_leaf=int(ty.get("min_leaf", 400)) * rep_of(len(X)), random_state=seed)
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
        roi = np.where((nb >= min_cell * rep_of(len(labels))) & (c > 0), r / np.maximum(c, 1), -1.0)
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
    k: int = 1                                   # 本番の再現: 発走前（T-60）のオッズを再現した版の数（races は「版 × レース」を並べたもの）
    base: Any = None                             # 本番の再現のとき、確定オッズで判定する元の期間（楽観の値を並べるため）

    @property
    def sims(self) -> list:                      # 互換: 版を並べた期間なら真（中身は使わない）
        return [self] if self.k > 1 else []

    @property
    def frecs(self) -> list[dict]:
        if not hasattr(self, "_frecs"):
            self._frecs = self.feats.to_dict("records")
        return self._frecs

    @property
    def dates(self) -> np.ndarray:
        return self.feats["date"].to_numpy()


# ── 本番の再現: 発走前（T-60）のオッズを、確定オッズから逆向きに作る ─────────────────────────────

DRIFT_SRC: Path | None = None                   # 再現のずれの元を差し替えるとき（凍結のときの写し）。None なら PRE_ODDS
POP_EDGES = [0, 1, 3, 6, 9, 99]                  # 人気帯（確定の人気）: 1・2〜3・4〜6・7〜9・10〜
PRE_ODDS = ROOT / "data/local/research/persona1/pre_odds.parquet"


def _pop_band(pop: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(POP_EDGES[1:-1]), np.asarray(pop, float), side="left")


def load_drift(phase: str = "t60", path: Path = PRE_ODDS, years=(2026,)) -> dict[int, np.ndarray]:
    """2026 年の発走前オッズ（phase の時点）と確定オッズの、対数の比 d = log(確定 / 発走前) を、確定の人気帯ごとに集める。
    T-60 から確定にかけて、人気馬は下がり（d<0）、人気薄は大きく上がる（d>0）。"""
    from src.config.data_paths import TABLES_DIR

    pre = pd.read_parquet(path)
    pre["race_id"] = pre["race_id"].astype(str)
    pre = pre[(pre["phase"] == phase) & (pd.to_numeric(pre["win_odds"], errors="coerce") > 0)][["race_id", "horse_number", "win_odds"]]
    fl = pd.concat([pd.read_parquet(Path(TABLES_DIR) / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_number", "odds", "popularity"]) for y in years])
    fl["race_id"] = fl["race_id"].astype(str)
    for c in ("horse_number", "odds", "popularity"):
        fl[c] = pd.to_numeric(fl[c], errors="coerce")
    pre["horse_number"] = pd.to_numeric(pre["horse_number"], errors="coerce")
    m = pre.merge(fl, on=["race_id", "horse_number"])
    m = m[(m["odds"] > 0) & (m["popularity"] > 0)].copy()
    m["d"] = np.log(m["odds"].to_numpy(float) / pd.to_numeric(m["win_odds"]).to_numpy(float))
    d = m["d"].to_numpy()
    b = _pop_band(m["popularity"].to_numpy(float))
    out = {int(k): d[b == k] for k in np.unique(b)}
    # レースごとの、確定の人気の順に並べたずれ（レースの中の並び＝人気の入れ替わり方を保って借りる）
    out["races"] = [g.sort_values("popularity")["d"].to_numpy() for _, g in m.groupby("race_id") if len(g) >= 5]
    return out


def simulate_pre_period(p: "Period", drift: dict, seed: int, cuts: dict) -> "Period":
    """確定オッズに、人気帯ごとの実測のずれ d を逆向きに当てて（発走前 = 確定 × exp(−d)）発走前のオッズを作った期間。
    人気・期待値・q・r（r は q の替わり分だけ動かす）・特徴・分位をその版で作り直す。精算は確定の払戻のまま。
    予測の p は確定の時点の offset のまま（過去の年に発走前の予測は無い。第 2 段の予測は T-60 の雑音つきで学習した版）。"""
    rng = np.random.default_rng(1000 + seed)
    races = []
    pool = drift.get("races") or []
    for r in p.races:
        if pool:                                          # 2026 年の 1 レースを丸ごと借り、確定の人気の順位ごとに、その順位のずれを当てる
            src = pool[int(rng.integers(len(pool)))]
            rk = np.clip(np.where(r.pop < 99, r.pop, len(src)).astype(int) - 1, 0, len(src) - 1)
            d = src[rk]
        else:
            b = _pop_band(r.pop)
            d = np.array([rng.choice(drift[k]) if k in drift and len(drift[k]) else 0.0 for k in b])
        o = r.odds * np.exp(-d)
        s0, s1 = np.nansum(1.0 / r.odds), np.nansum(1.0 / o)
        if s0 > 0 and s1 > 0:                             # 1/オッズの合計（控除の大きさ）を元のレースに合わせる
            o = o * (s1 / s0)
        pop = np.where(np.isfinite(o), pd.Series(o).rank(method="min").to_numpy(), 99.0)
        q, rr = r.q, r.r
        inv = np.where(np.isfinite(o), 1.0 / o, 0.0)
        qs = inv / inv.sum() if inv.sum() > 0 else inv
        if rr is not None:                                # r は「1/確定オッズの正規化 ÷ 1/再現オッズの正規化」の分だけ動かす（ずれ 0 なら動かない）
            inv0 = np.where(np.isfinite(r.odds), 1.0 / r.odds, 0.0)
            qf = inv0 / inv0.sum() if inv0.sum() > 0 else inv0
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = rr + np.log(np.clip(qf, 1e-12, None) / np.clip(qs, 1e-12, None))
            q = None if r.q is None else r.q * np.clip(qs, 1e-12, None) / np.clip(qf, 1e-12, None)
        races.append(Race(r.race_id, r.date, r.hn, r.pw, r.p3, o, pop, r.fin, r.pay, r.p2, q, rr))
    f = pd.DataFrame([race_features(r) for r in races])
    keep = [c for c in p.feats.columns if c not in f.columns and not c.startswith("q_")]
    ft = pd.concat([p.feats[keep].reset_index(drop=True), f], axis=1)
    return Period(f"{p.name}_sim{seed}", races, apply_qcuts(ft, cuts), p.W, p.drop)


def run_period(p: "Period", s: dict, detail: bool = False) -> dict:
    """期間で買い方を実行する（本番の再現のときは「版 × レース」を並べた期間のまま。指標は metrics が版の数で割る）。
    詳しい記録（records）には版の番号（ver）を付ける。"""
    res = run_strategy(p.races, p.frecs, s, detail)
    if detail and "records" in res:
        n0 = len(p.races) // max(p.k, 1)
        res["records"]["ver"] = np.arange(len(res["records"])) // max(n0, 1)
    return res


def metrics_run(res: dict, p: "Period", W=None) -> dict:
    m = metrics(res["cost"], res["ret"], p.dates, W)
    if p.k > 1 and m.get("bet_races"):
        m["decision_odds"] = f"発走前（再現 {p.k} 通りを並べた平均）"
    return m


def stack_versions(p: "Period", versions: list["Period"]) -> "Period":
    """発走前の再現の版を「版 × レース」に並べた期間。ブートストラップの重みは同じレースの版に同じ重み（レース単位の区間）。"""
    races = [r for v in versions for r in v.races]
    ft = pd.concat([v.feats for v in versions], ignore_index=True)
    k = len(versions)
    W = np.tile(p.W, (1, k))
    if _REP.get(len(races), k) != k or _REP.get(len(p.races), 1) != 1:
        raise RuntimeError("並べた期間の長さが別の期間と重なりました（件数の割り算を取り違える）")
    _REP[len(races)] = k
    return Period(p.name, races, ft, W, p.drop, k=k, base=p)


def load_inputs(path: Path | None = None) -> dict:
    return json.loads((path or CFG_DIR / "inputs.json").read_text(encoding="utf-8"))


_CACHE: dict = {}


def prepare_periods(inputs: dict, names=("train", "eval"), cuts: dict | None = None, n_boot: int = 1000) -> tuple[dict[str, Period], dict]:
    """期間ごとの Race・特徴を作る。分位の区切りは訓練の期間で決め（cuts が無ければ）、他の期間にそのまま当てる。"""
    per = {}
    for nm in names:
        spec = inputs["periods"][nm]
        years = spec["years"]
        pred = pd.concat([load_pred(p if isinstance(p, dict) else pred_path(p)) for p in spec["pred"]], ignore_index=True)
        pred = pred[pred["race_id"].str[:4].astype(int).isin(years)]
        if spec.get("ignore_ptop"):                 # 連対率・複勝率を勝率から Harville で補う側にそろえる（訓練が勝率だけのとき test も合わせる）
            pred = pred.drop(columns=[c for c in ("p_top2", "p_top3") if c in pred.columns])
        flat = load_runners(years)
        if spec.get("dates"):                       # 日付の窓（事前登録の群）: 出走表をその期間に絞る（両端を含む）
            lo, hi = spec["dates"]
            dd = pd.to_datetime(flat["date"]).dt.strftime("%Y-%m-%d")
            flat = flat[(dd >= lo) & (dd <= hi)]
        pay = load_payoffs(set(flat["race_id"]))
        n_fill = fill_payoffs_from_flat(pay, years) if spec.get("fill_payoff_from_flat") else 0
        oo = None
        if spec.get("pre_odds"):
            oo = pre_odds_override(Path(spec["pre_odds"]) if str(spec["pre_odds"]).startswith("/") else ROOT / spec["pre_odds"],
                                   spec.get("pre_phase", ["t02", "t05", "t10", "t15", "t60"]))
        races, cond, drop = build_races(pred, flat, pay, years, odds_override=oo)
        drop["all_races_in_period"] = int(flat["race_id"].nunique())      # 期間の全レース（予測・払戻がそろわないものを含む）
        drop["payoff_filled_from_flat"] = n_fill
        drop["computed_races"] = len(races)
        ft = feature_table(races, cond)
        per[nm] = (races, ft, drop)
    out = {}
    for nm, (races, ft, drop) in per.items():
        out[nm] = Period(nm, races, ft, poisson_weights(len(races), n_boot, seed=zlib.crc32(nm.encode()) % 1000), drop)
        if _REP.get(len(races), 1) != 1:              # 版を並べた別の期間と長さが重なると、件数を版の数で割り違える
            raise RuntimeError(f"期間 {nm} の長さ {len(races)} が、版を並べた別の期間と重なりました")
        _REP.setdefault(len(races), 1)
        do = inputs["periods"][nm].get("decision_odds")
        if do and do.get("kind") == "sim_pre":           # 本番の再現: 判定に使うオッズを発走前の再現にし、版を並べた期間にする
            src_ = DRIFT_SRC or PRE_ODDS                  # 再現のずれの元（凍結のときの中身に固定できる）
            key = "drift_" + do.get("phase", "t60") + "|" + str(src_)
            if key not in _CACHE:
                _CACHE[key] = load_drift(do.get("phase", "t60"), path=src_)
            vs = [simulate_pre_period(out[nm], _CACHE[key], k, {}) for k in range(int(do.get("n", 5)))]
            out[nm] = stack_versions(out[nm], vs)
    if cuts is None:                                     # 分位の区切りは訓練で決める（本番の再現なら、再現の特徴で）
        cuts = fit_qcuts(out["train"].feats) if "train" in out else {}
    for nm, p in out.items():
        p.feats = apply_qcuts(p.feats, cuts)
        p.__dict__.pop("_frecs", None)
        if p.base is not None:
            p.base.feats = apply_qcuts(p.base.feats, cuts)
            p.base.__dict__.pop("_frecs", None)
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
        sub = json.loads(((CYC_DIR / inc) if (CYC_DIR / inc).exists() else (CFG_DIR / inc)).read_text(encoding="utf-8"))
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
        row[nm] = metrics_run(r, p, p.W)
        if nm + "_fin" in res:                                        # 確定オッズで判定した値（楽観）
            rf = res[nm + "_fin"]
            row[nm + "_fin"] = metrics(rf["cost"], rf["ret"], p.base.dates, p.base.W)
        row[nm]["violation_races"] = r["n_violation_races"]
        row[nm]["violations"] = r["violations"]
        row[nm]["h_not_top_rate"] = float(((r["hrank"] > 0).sum()) / max((r["hrank"] >= 0).sum(), 1))
        row[nm]["h_fav1_rate"] = float(((r["hpop"] == 1) & (r["hrank"] >= 0)).sum() / max((r["hrank"] >= 0).sum(), 1))   # ◎ が 1 番人気（確定オッズの順）
        row[nm]["top_fav1_rate"] = float(((r["top_pop"] == 1) & (r["hrank"] >= 0)).sum() / max((r["hrank"] >= 0).sum(), 1))   # 予測 1 位（p）が 1 番人気
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
    out = {}
    for nm in per_names:
        out[nm] = run_period(per[nm], s)
        if per[nm].base is not None:                       # 確定オッズで判定した値（楽観）も並べる
            out[nm + "_fin"] = run_strategy(per[nm].base.races, per[nm].base.frecs, s)
    return s, out


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
    if rule == "r_max":
        mode = (s.get("partners") or {}).get("mode", "p3")
        return "r（ずれ）の ◎・" + {"fav": "人気の相手", "mix": "相手を広く"}.get(mode, "複勝率上位の相手")
    if rule in ("ana_ev", "ana_p3", "ev_max") and not any(k in h for k in ("max_rank", "p3_min", "pw_min")):
        return "参考: 制約なしの穴・期待値の ◎"
    return "型A 穴の ◎・人気の相手" if (s.get("partners") or {}).get("mode") == "fav" else "型B 穴の ◎・相手を広く"


def fav1_rule_ok(s: dict) -> bool:
    """ユーザーの方針（2026-11-16: ◎ は 2 番人気以下から、1 番人気は連対率の予測が高い例外のときだけ）に合う買い方か。
    合わない（1 番人気を例外なしに ◎ にしうる）買い方は、最後の候補では参考として順位から外す。方策は、型ごとの買い方がすべて合うときだけ。"""
    if "policy" in s:
        vals = list(s["policy"]["map"].values()) + [s["policy"].get("default", "skip")]
        return all(v == "skip" or fav1_rule_ok(v) for v in vals)
    h = s.get("honmei") or {}
    sk = s.get("skip") or {}
    if (sk.get("h_pop_min") or 0) >= 2:
        return True
    rule = h.get("rule", "top")
    if rule == "fav1_exc":                            # 例外のとき以外は 2 番人気以下（base が 1 番人気を選んだら替える作り）
        if h.get("exc_by") == "r":                    # 訓練で決めたしきい値（fav1 --by r）だけ。手書きの値は通さない（auto を置き換えた値は通す）
            return h.get("r_min") == "auto" or (h.get("r_min") is not None and float(h["r_min"]) in _RESOLVED_R)
        p2 = h.get("p2_min", 0.85)
        return p2 == "auto" or float(p2) >= FAV1_TARGET  # しきい値を手で低く書いた買い方は通さない
    if rule == "r_max":
        return int(h.get("fav_k", 1)) >= 1 and h.get("fallback", "skip") == "skip"
    if rule == "switch":
        subs = list((h.get("map") or {}).values()) + [h.get("default", {"rule": "top"})]
        return all(x == "skip" or fav1_rule_ok({**s, "honmei": x}) for x in subs)
    if rule == "fav1_penalty":
        return float(h.get("lam", 0.5)) >= 1.0
    if rule in ("ana_ev", "ana_p3"):
        return int(h.get("fav_k", 3)) >= 1 and h.get("fallback", "top") == "skip"
    return False


def perturb_period(p: Period, sigma: float, seed: int, cuts: dict) -> Period:
    """◎ の決め方・見送りに使う単勝オッズ（と人気）にだけ対数正規の揺れ σ を足した期間（精算は確定の払戻のまま）。
    買う時点のオッズと確定オッズのずれの代わり（tp-50 §6 と同じ方法）。"""
    rng = np.random.default_rng(seed)
    races = []
    for r in p.races:
        o = r.odds * np.exp(rng.normal(0.0, sigma, len(r.odds)))
        pop = np.where(np.isfinite(o), pd.Series(o).rank(method="min").to_numpy(), 99.0)
        races.append(Race(r.race_id, r.date, r.hn, r.pw, r.p3, o, pop, r.fin, r.pay, r.p2, r.q, r.r))
    f = pd.DataFrame([race_features(r) for r in races])
    keep = [c for c in p.feats.columns if c not in f.columns and not c.startswith("q_")]
    ft = pd.concat([p.feats[keep].reset_index(drop=True), f], axis=1)
    return Period(f"{p.name}_s{sigma}", races, apply_qcuts(ft, cuts), p.W, p.drop)


def odds_robustness(spec: dict, p: Period, cuts: dict, sigmas=(0.1, 0.2), seeds=(0, 1, 2)) -> dict:
    if p.k > 1:                                      # 本番の再現では、発走前の再現の版ごとの回収率の幅で代える（metrics の sim_roi_min・max）
        return {}
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
                cell_tab.append({"type": str(lv), "races_train": int(mt.sum()) // rep_of(len(mt)), "races_eval": int(me.sum()) // rep_of(len(me)),
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
    cuts_path = out / f"qcuts_train_{strategy_hash({'years': tr_spec['years'], 'pred': tr_spec['pred'], **({'decision_odds': tr_spec['decision_odds']} if tr_spec.get('decision_odds') else {})})}.json"   # 訓練の年・予測が変われば別の区切り
    cuts = json.loads(cuts_path.read_text()) if cuts_path.exists() else None
    per, cuts = prepare_periods(inputs, ("train", "eval"), cuts)
    if not cuts_path.exists():
        cuts_path.write_text(json.dumps(cuts, indent=1))
    strats = resolve_auto(expand_variants(cyc), out)
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
        rr = run_period(per["train"], sel["spec"], detail=True)
        rec["retro"] = retrospective(rr, per["train"].feats, per["train"].W)
        kk = per["train"].k
        rec["miss_types"] = {a_: int(round(b_ / kk)) for a_, b_ in miss_types(rr).items()}            # 版の数で割った件数
        rec["h_reasons"] = {a_: int(round(b_ / kk)) for a_, b_ in rr["records"]["reason"].value_counts().to_dict().items()}
        rr["records"].to_parquet(out / f"cycle_{n:02d}_selected_train_races.parquet", index=False)
    rec["n_tried_cum"] = int(sum(v.get("n_tried", 0) for k, v in state["cycles"].items() if int(k) <= n))
    rec["top"] = _slim(cands[:15])
    for t, c in zip(rec["top"], cands[:15]):
        t["family"] = c["family"]
        if c.get("kind") == "policy":
            t.update(kind="policy", cells=_cells_no_eval(c["cells"]), train_insample=c["train_insample"])
    rec["all"] = _slim(var_rows, full=False) + _slim(pol_rows, full=False)
    rec["policy_specs"] = {r["name"]: r["spec"] for r in pol_rows}          # pick-final が方策の中身を名前で引けるように
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


_RESOLVED_R: set = set()       # fav1 --by r が訓練で決めた r の例外のしきい値（auto を置き換えた値。方針の検査で通す）


def resolve_auto(strats: list[dict], out: Path) -> list[dict]:
    """p2_min が "auto" の値だけを、その予測で bet_cycles fav1 が訓練で決めたしきい値（stage_fav1_exception.json）に置き換える（再帰）。"""
    rp = out / "stage_fav1r_exception.json"              # r の例外のしきい値（訓練で決めた値）を、置き換えの有無にかかわらず読み直す（tag ごと）
    _RESOLVED_R.clear()
    if rp.exists():
        ch_ = json.loads(rp.read_text(encoding="utf-8")).get("chosen")
        _RESOLVED_R.add(float(ch_["p2_min"]) if ch_ else 9.0)

    def has(x):
        if isinstance(x, dict):
            return x.get("p2_min") == "auto" or x.get("r_min") == "auto" or any(has(v) for v in x.values())
        return isinstance(x, list) and any(has(v) for v in x)
    if not has(strats):
        return strats
    def _val(name):
        st = out / name
        if not st.exists():
            return None
        ch = json.loads(st.read_text(encoding="utf-8")).get("chosen")
        return float(ch["p2_min"]) if ch else 9.0          # しきい値が無ければ例外を作らない（9.0 は誰も満たさない）
    val, rval = _val("stage_fav1_exception.json"), _val("stage_fav1r_exception.json")
    txt = json.dumps(strats, ensure_ascii=False)
    if ('"p2_min": "auto"' in txt and val is None) or ('"r_min": "auto"' in txt and rval is None):
        raise SystemExit("auto には、先に `bet_cycles fav1`（r_min は `fav1 --by r`）で例外のしきい値を決める必要があります")

    def sub(x):
        if isinstance(x, dict):
            return {k: (val if (k == "p2_min" and v == "auto") else (rval if (k == "r_min" and v == "auto") else sub(v))) for k, v in x.items()}
        if isinstance(x, list):
            return [sub(v) for v in x]
        return x
    return sub(strats)


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
            if r.get("train_fin"):
                x["train_fin"] = r["train_fin"]
        else:
            x = {"name": r["name"], "hash": r["hash"], "train_folds": r.get("train_folds"),
                 "train": {k: r["train"].get(k) for k in ("bet_races", "bet_rate", "hit_rate", "roi", "roi_lo", "roi_hi", "avg_return",
                                                         "median_return", "median_hit_return", "roi_ex_top5", "violation_races", "h_fav1_rate", "top_fav1_rate")}}
        if with_eval:
            x["eval"] = r.get("eval", {})
            if r.get("eval_fin"):
                x["eval_fin"] = r["eval_fin"]
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
        if rec.get("odds_robustness_train"):
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
    return _fav1_rows(p)          # 本番の再現なら、並べた全部の版の発走前の人気で 1 番人気を決める（件数は版の数で割らずに数える）


def _fav1_rows(p: Period) -> pd.DataFrame:
    n0 = max(len(p.races) // max(p.k, 1), 1)
    """レースごとの 1 番人気（確定オッズの順）の予測の連対率・勝率と、着順・単複の払戻。"""
    rows = []
    for idx_, (r, f) in enumerate(zip(p.races, p.frecs)):
        j = int(np.argmin(r.pop))
        if r.pop[j] != 1:
            continue
        others = np.delete(r.pw, j)
        h = str(int(r.hn[j]))
        rows.append({"race_id": r.race_id, "ver": idx_ // n0, "date": r.date, "p2": float(r.p2[j]), "pw": float(r.pw[j]), "gap": float(r.pw[j] - others.max()),
                     "r": float(r.r[j]) if r.r is not None and np.isfinite(r.r[j]) else np.nan,
                     "is_top": int(j == 0), "fin": float(r.fin[j]), "tansho": r.pay["tansho"].get(h, 0), "fukusho": r.pay["fukusho"].get(h, 0)})
    return pd.DataFrame(rows)


def fav1_stats(d: pd.DataFrame, k: int = 1) -> dict:
    """k = 本番の再現の版の数（件数は版の数で割る。率は版を通した平均）。"""
    if not len(d):
        return {"n": 0}
    return {"n": int(round(len(d) / k)), "rentai": float((d["fin"] <= 2).mean()), "win": float((d["fin"] == 1).mean()), "place": float((d["fin"] <= 3).mean()),
            "roi_tansho": float(d["tansho"].sum() / (100 * len(d))), "roi_fukusho": float(d["fukusho"].sum() / (100 * len(d)))}


def cmd_fav1(a) -> None:
    """例外のしきい値（1 番人気の予測の連対率 p2 の下限）を訓練で決め、評価で連対率が保たれるかを確かめる。"""
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out = OUT_DIR / inputs.get("tag", "v6e")
    out.mkdir(parents=True, exist_ok=True)
    per, _ = prepare_periods(inputs, ("train", "eval"), cuts={}, n_boot=10)
    tr, ev = fav1_rows(per["train"]), fav1_rows(per["eval"])
    kt, ke = per["train"].k, per["eval"].k
    by = getattr(a, "by", "p2") or "p2"                     # p2 = 予測の連対率（第 1 回）、r = 市場とのずれ（第 2 回）
    col = "p2" if by == "p2" else "r"
    steps = np.round(np.arange(0.40, 0.951, 0.025), 3) if by == "p2" else np.round(np.arange(-0.5, 1.51, 0.05), 3)
    grid = []
    for t in steps:
        st = fav1_stats(tr[tr[col] >= t], kt)
        st["p2_min"] = float(t)                             # 名前は p2_min のまま（r のときは r のしきい値）
        st["by"] = by
        grid.append(st)
    chosen = choose_fav1_threshold(grid)
    res = {"target_rentai": FAV1_TARGET, "min_n": FAV1_MIN_N, "train_all_fav1": fav1_stats(tr, kt), "eval_all_fav1": fav1_stats(ev, ke), "grid_train": grid,
           "chosen": chosen}
    if chosen:
        t = chosen["p2_min"]
        res["train_exception"] = fav1_stats(tr[tr[col] >= t], kt)
        res["eval_exception"] = fav1_stats(ev[ev[col] >= t], ke)
        res["train_rest"] = fav1_stats(tr[~(tr[col] >= t)], kt)
        res["eval_rest"] = fav1_stats(ev[~(ev[col] >= t)], ke)
        res["by"] = by
        res["train_races"], res["eval_races"] = len(per["train"].races), len(per["eval"].races)
    fmt = lambda x: "－" if x is None else f"{x * 100:.1f}%"   # noqa: E731
    row = lambda lab, s_: [lab, f"{s_.get('n', 0):,}", fmt(s_.get("rentai")), fmt(s_.get("win")), fmt(s_.get("place")),   # noqa: E731
                           fmt(s_.get("roi_tansho")), fmt(s_.get("roi_fukusho"))]
    head = ["区分", "レース", "1番人気の連対率", "勝率", "複勝率", "単勝の回収率", "複勝の回収率"]
    tables = [{"title": "しきい値ごと（訓練の期間。1 番人気の予測の連対率 ≥ しきい値のレース）", "head": ["しきい値"] + head[1:],
               "rows": [[f"{g['p2_min']:.3f}"] + row("", g)[1:] for g in grid if g["n"] > 0]}]
    if chosen:
        tables.insert(0, {"title": f"選んだしきい値 {chosen['p2_min']:.3f}（連対率 {FAV1_TARGET:.0%} 以上・{FAV1_MIN_N} 件以上で最も緩いもの）", "head": head,
                          "rows": [row("訓練: 例外（1 番人気を ◎ にしてよい）", res["train_exception"]), row("訓練: それ以外の 1 番人気", res["train_rest"]),
                                   row("訓練: 1 番人気すべて", res["train_all_fav1"]), row("評価: 例外（確かめ）", res["eval_exception"]),
                                   row("評価: それ以外の 1 番人気", res["eval_rest"]), row("評価: 1 番人気すべて", res["eval_all_fav1"])]})
    text = [f"ユーザー（2026-11-16）: 「1 人気を 1 番手に推奨するのはペナルティを設けたい。2 人気以降の馬から本命馬を選びたい。」「どうしても 1 人気を 1 番手に推奨するときには、"
            f"そのほぼすべてが勝ち負けに匹敵する成績であるべき。市場の注目度の穴をねらうべき。」",
            f"例外の規則: 1 番人気（確定オッズの順。発走前のオッズの人気とはずれうる）の予測の連対率が、しきい値以上のときだけ、その馬を ◎ にしてよい。"
            f"しきい値は訓練の期間で『そのしきい値と、それより厳しいすべてのしきい値（{FAV1_MIN_N} 件以上）で、例外にしたレースの 1 番人気の連対率が {FAV1_TARGET:.0%} 以上（目安 80〜90% の中ほどを推奨で決めた）』を満たす最も緩い値。"
            f"評価の期間は、そのしきい値で連対率が保たれるかを見るだけ。"]
    if not chosen:
        text.append(f"訓練では、連対率 {FAV1_TARGET:.0%} 以上を {FAV1_MIN_N} 件以上で満たすしきい値が無かった。例外は作らず、◎ はすべて 2 番人気以下から選ぶ。")
    if by == "r":
        text[1] = text[1].replace("予測の連対率", "市場とのずれ r").replace("連対率が", "（実際の）連対率が", 1)
        for t_ in tables:
            t_["title"] = t_["title"].replace("予測の連対率", "r（市場とのずれ）")
    res.update(title="1 番人気を ◎ にしてよい例外（" + ("市場とのずれ r" if by == "r" else "連対率") + "のしきい値。列 p2_min は r のときは r のしきい値）", text=text, tables=tables)
    (out / ("stage_fav1_exception.json" if by == "p2" else "stage_fav1r_exception.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("chosen", "train_all_fav1", "eval_all_fav1") if k in res}, ensure_ascii=False, indent=1))
    for k in ("train_exception", "eval_exception", "train_rest", "eval_rest"):
        if k in res:
            print(k, res[k])


# ── 選び方そのものの交差検証（訓練の中の年: 2022 だけで格子の最良を選び 2023 で測る、その逆） ──────────


def _fold_pick(C: np.ndarray, R: np.ndarray, fit: np.ndarray, base: tuple[np.ndarray, np.ndarray], min_hit: float, min_n: int) -> int:
    """fit のレースだけで、cycle 8 以降と同じ考え方（的中率の下限・基準の上位 5 件除きより上）で最良の 1 本を選ぶ（-1 = なし）。
    1 年分しかないので、順位は上位 5 件を除いた回収率（年ごとの最小は使えない）。"""
    bc_, br_ = base[0][fit], base[1][fit]
    b5 = (br_.sum() - np.sort(br_[bc_ > 0])[::-1][:5 * rep_of(C.shape[1])].sum()) / max(bc_.sum(), 1)
    best, bj = -1.0, -1
    for j in range(C.shape[0]):
        c, r = C[j, fit], R[j, fit]
        b = c > 0
        kk = rep_of(C.shape[1])
        if b.sum() < min_n * kk or (r[b] > 0).mean() < min_hit:
            continue
        ex5 = (r.sum() - np.sort(r[b])[::-1][:5 * kk].sum()) / c.sum()
        if ex5 >= b5 and ex5 > best:
            best, bj = ex5, j
    return bj


def selection_cv(C: np.ndarray, R: np.ndarray, years: np.ndarray, base: tuple[np.ndarray, np.ndarray], min_hit: float, min_n: int = 150
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    cost, ret = np.zeros(C.shape[1]), np.zeros(C.shape[1])
    picks = {}
    for y in np.unique(years):
        fit, hold = years != y, years == y
        j = _fold_pick(C, R, fit, base, min_hit, min_n)
        picks[str(y)] = j
        if j >= 0:
            cost[hold], ret[hold] = C[j, hold], R[j, hold]
    return cost, ret, picks


def cmd_selcv(a) -> None:
    """過去のサイクルの格子（固定の買い方）を訓練で流し直し、全候補の cost・ret を保存し（多重比較の割り引き用）、
    サイクルごとに『選び方そのもの』の交差検証の回収率を出す。方策のサイクルは方策の行（すでに交差検証の値）を並べる。"""
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out = OUT_DIR / inputs.get("tag", "v6e")
    tr_spec = inputs["periods"]["train"]
    cuts = json.loads((out / f"qcuts_train_{strategy_hash({'years': tr_spec['years'], 'pred': tr_spec['pred'], **({'decision_odds': tr_spec['decision_odds']} if tr_spec.get('decision_odds') else {})})}.json").read_text())
    per, _ = prepare_periods(inputs, ("train",), cuts)
    tr = per["train"]
    years = tr.feats["year"].to_numpy()
    baselines = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"]
    b0 = run_period(tr, baselines[0])
    base = (b0["cost"], b0["ret"])
    rows = []
    for n in [int(x) for x in a.cycles.split(",")]:
        cyc = json.loads((CYC_DIR / f"cycle_{n:02d}.json").read_text(encoding="utf-8"))
        rec_p = out / f"cycle_{n:02d}.json"
        rec = json.loads(rec_p.read_text(encoding="utf-8")) if rec_p.exists() else {}
        strats = [s for s in resolve_auto(expand_variants(cyc), out) if not family(s).startswith("参考") and fav1_rule_ok(s)]
        if not strats:
            print(n, "固定の格子に、1 番人気の方針に合う候補なし", flush=True)
        if strats:
            key = strategy_hash({"pred": tr_spec["pred"], "years": tr_spec["years"], "cuts": cuts, "decision_odds": tr_spec.get("decision_odds"), "pre": (_sha(PRE_ODDS) if tr_spec.get("decision_odds") and PRE_ODDS.exists() else None),
                          "specs": [strategy_hash(x) for x in strats]})
            npz = out / f"cands_c{n:02d}_{key}.npz"           # 予測・区切り・買い方が変われば作り直す
            z = np.load(npz, allow_pickle=True) if npz.exists() else None
            if z is not None and list(z["names"]) == [x["name"] for x in strats]:
                C, R = z["cost"], z["ret"]
            else:
                res = run_many(strats, {"train": tr}, threads=a.threads)
                C = np.stack([r["train"]["cost"] for _, r in res]).astype(np.float32)
                R = np.stack([r["train"]["ret"] for _, r in res]).astype(np.float32)
                np.savez_compressed(npz, cost=C, ret=R, names=np.array([s["name"] for s in strats]))
            min_hit = float(cyc.get("min_hit", 0.08))
            cc, cr, picks = selection_cv(C, R, years, base, min_hit)
            m = metrics(cc, cr, tr.dates, tr.W)
            fold = {str(y): (float(cr[years == y].sum() / cc[years == y].sum()) if cc[years == y].sum() > 0 else None) for y in np.unique(years)}
            same = len(set(picks.values())) == 1 and -1 not in picks.values()
            complete = -1 not in picks.values()                  # どちらかの年で候補が無ければ、交差検証は 1 年分だけ（順位から外す）
            row = {"cycle": n, "kind": "固定の格子", "n_cands": int(C.shape[0]), "min_hit": min_hit, "cv": {k: m.get(k) for k in KEYS_CV}, "cv_folds": fold,
                   "picks": {y: (strats[j]["name"] if j >= 0 else "なし") for y, j in picks.items()}, "same_pick": bool(same), "cv_complete": bool(complete),
                   "score": robust_cv(m, fold) if complete else None}
            rows.append(row)
        psp = dict(rec.get("policy_specs") or {})
        for t in rec.get("top", []) + ([rec["selected"]] if rec.get("selected") else []):   # 古い記録（policy_specs が無い）は上位の中身で調べる
            if t.get("name", "").startswith("policy") and t.get("spec"):
                psp.setdefault(t["name"], t["spec"])
        pol = [x for x in rec.get("all", []) if x["name"].startswith("policy") and (x["train"].get("hit_rate") or 0) >= min_hit
               and (x["train"].get("bet_races") or 0) >= MIN_BET_RACES and x["name"] in psp and fav1_rule_ok(psp[x["name"]])]   # 中身の分からない方策は外す
        if pol:                                               # 同じ物差しで並べる。方策どうしの選び直しの楽観は残る（記録に書く）
            bestp = max(pol, key=lambda x: robust_cv(x["train"], x.get("train_folds") or {}))
            rows.append({"cycle": n, "kind": f"方策（交差検証、{len(pol)} 本から選んだ）", "n_cands": len(pol), "min_hit": min_hit, "cv": bestp["train"],
                         "cv_folds": bestp.get("train_folds"), "picks": {"": bestp["name"]}, "same_pick": None, "cv_complete": True,
                         "score": robust_cv(bestp["train"], bestp.get("train_folds") or {})})
        if strats:
            print(n, row["kind"], C.shape[0], {k: round(v, 3) if isinstance(v, float) else v for k, v in row["cv"].items()}, fold, row["picks"], same, flush=True)
    res = {"title": "選び方そのものの交差検証（訓練の中の年）", "rows": rows,
           "text": ["各サイクルの格子（固定の買い方）の中から、訓練の年を 1 年ずつ外し、残りの年だけで最良を選んで外した年に当てた回収率（年ごとの値を並べる）。"
                    "選ぶ物差しは cycle 8 以降と同じ考え方（的中率の下限・基準の上位 5 件除きより上・上位 5 件を除いた回収率で順位、各年 150R 以上）。"
                    "数百本の中から選んだ標本内の回収率（各サイクルの『選んだ』の訓練の値）より、選び方の楽観を除いた値になる。"
                    "方策の行は、型ごとの選び方は交差検証の値だが、同じサイクルの方策どうし（型の作り方・件数の下限・見送りの下限）の選び直しの楽観は残る。"
                    "どれかの年で条件を満たす候補が無かった行は、一部の年だけの値なので順位から外した（印『1 年分』）。",
                    "1 番人気の方針（◎ は 2 番人気以下、1 番人気は連対率の予測が高い例外のときだけ。ユーザー 2026-11-16）に合わない買い方（予測 1 位・人気の最高オッズ・期待値最大の ◎、候補なしで予測 1 位に戻す穴の ◎、λ<1 のペナルティ）は、比べる参考として候補から外した（2026-11-17、test を見る前に決めた）。"]}
    res["tables"] = [{"title": f"サイクルごとの選び方の交差検証（訓練 {min(years)}〜{max(years)}）",
                      "head": ["回", "種類", "候補", "参戦", "的中率", "回収率", "95% 区間", "上位5件を除く", "年ごと（その年を外して選び、その年に当てた値）", "どの年も同じ形"],
                      "rows": [[str(x["cycle"]), x["kind"], str(x["n_cands"]), f"{(x['cv'].get('bet_races') or 0):,}",
                                f"{(x['cv'].get('hit_rate') or 0) * 100:.1f}%", f"{(x['cv'].get('roi') or 0) * 100:.1f}%",
                                (f"{x['cv']['roi_lo'] * 100:.0f}〜{x['cv']['roi_hi'] * 100:.0f}%" if x["cv"].get("roi_lo") is not None else "－"),
                                f"{(x['cv'].get('roi_ex_top5') or 0) * 100:.1f}%",
                                " / ".join(f"{y} {v * 100:.1f}%" if v is not None else f"{y} －" for y, v in (x.get("cv_folds") or {}).items()),
                                {True: "同じ", False: "違う", None: "－"}[x["same_pick"]] + ("" if x.get("cv_complete", True) else "（1 年分・順位外）")] for x in rows], "left": 2}]
    (out / "stage_selcv.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")


def robust_cv(m: dict, folds: dict) -> float:
    """最後の候補の物差し: min（上位 5 件を除いた回収率, 年ごとの回収率）。"""
    vals = [m.get("roi_ex_top5")] + [v for v in (folds or {}).values()]
    vals = [v for v in vals if v is not None]
    return float(min(vals)) if vals else -1.0


KEYS_CV = ("bet_races", "bet_rate", "hit_rate", "roi", "roi_lo", "roi_hi", "roi_ex_top5", "avg_return", "median_return")


def cmd_check_periods(a) -> None:
    """test・2026 の入口の確かめ（結果を見ない）。レース数と落とした数だけを出す。回収率は出さない。"""
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    names = [x for x in a.periods.split(",")]
    per, _ = prepare_periods(inputs, names, cuts={}, n_boot=2)
    for nm, p in per.items():
        n_fav = sum(1 for r in p.races if (r.pop == 1).sum() == 1)
        print(nm, {"races": len(p.races), "drop": p.drop, "with_single_fav1": n_fav,
                   "dates": (p.feats["date"].min(), p.feats["date"].max()) if len(p.races) else None, "p2_from_pred": bool(p.races and p.races[0].p2 is not None)})


# ── 得意・不得意と推奨度の段（指示書 6）。判定は訓練だけ、評価は段の順の確かめだけ ─────────────────────

TIER_AXES = {"c_venue": "会場", "c_surface": "芝・ダート", "c_dist_band": "距離帯", "c_class": "クラス", "fieldband": "頭数", "c_going": "馬場",
             "c_season": "季節", "q_ev_max": "期待値の最大の5分位", "shape": "分布の型（手の区切り）"}
TIER_MIN_N = 100          # 判定する水準の参戦の下限（訓練）
TIER_PRIOR = 300          # 縮める強さ（参戦レース数の重み。件数の少ない水準ほど全体に寄せる）
TIER_NAMES = ("★★★", "★★", "★")


def level_stats(cost: np.ndarray, ret: np.ndarray, labels: np.ndarray, years: np.ndarray, dates: np.ndarray, W: np.ndarray, min_n: int = TIER_MIN_N) -> dict:
    """水準ごとの参戦・的中率・回収率・区間・上位5件を除いた回収率・年ごとの回収率。"""
    out = {}
    for lv in np.unique(labels):
        m = labels == lv
        c, r = np.where(m, cost, 0), np.where(m, ret, 0)
        if (c > 0).sum() < min_n * rep_of(len(cost)):
            continue
        st = metrics(c, r, dates, W)
        st["folds"] = {str(y): (float(r[years == y].sum() / c[years == y].sum()) if c[years == y].sum() > 0 else None) for y in np.unique(years)}
        out[str(lv)] = st
    return out


def judge_level(st: dict, base_roi: float) -> str:
    """得意 = 区間の下限が基準以上・年ごとがすべて 1 以上・上位 5 件を除いても 1 以上。不得意 = 区間の上限が基準（B0）未満。
    年の交差検証の中では、年ごとは fit の 1 年分だけになる（記録に書く）。"""
    folds = [v for v in st.get("folds", {}).values() if v is not None]
    if st.get("roi_hi", 9) < base_roi:            # 区間の上限が基準（B0）を下回る（指示書 6 の定義）
        return "不得意"
    if st.get("roi_lo", 0) >= base_roi and folds and min(folds) >= 1.0 and st.get("roi_ex_top5", 0) >= 1.0:
        return "得意"
    return "判定できない"


def shrunk(st: dict, overall: float, prior: int = TIER_PRIOR) -> float:
    n = st.get("bet_races", 0)
    return float((st.get("roi", overall) * n + overall * prior) / (n + prior))


def fit_tiers(cost, ret, feats: pd.DataFrame, mask_fit: np.ndarray, base_roi: float, dates, W, years, axes=None) -> dict:
    """mask_fit のレースで、軸ごとの水準の判定と縮めた回収率、段の区切り（点の 3 分位）を決める。"""
    model = {"axes": {}, "overall": float(ret[mask_fit].sum() / max(cost[mask_fit].sum(), 1))}
    for ax in (axes or list(TIER_AXES)):
        lab = feats[ax].astype(str).to_numpy()
        st = level_stats(np.where(mask_fit, cost, 0), np.where(mask_fit, ret, 0), lab, years, dates, W)
        model["axes"][ax] = {lv: {"judge": judge_level(x, base_roi), "shrunk": shrunk(x, model["overall"]), "n": x["bet_races"]} for lv, x in st.items()}
    sc = tier_scores(feats, model)
    keep = mask_fit & (cost > 0) & ~skip_mask(feats, model)
    model["cuts"] = [float(x) for x in np.quantile(sc[keep], [1 / 3, 2 / 3])] if keep.sum() > 10 else [float("inf"), float("inf")]
    return model


def skip_mask(feats: pd.DataFrame, model: dict) -> np.ndarray:
    m = np.zeros(len(feats), bool)
    for ax, lv in model["axes"].items():
        bad = {k for k, v in lv.items() if v["judge"] == "不得意"}
        if bad:
            m |= feats[ax].astype(str).isin(bad).to_numpy()
    return m


def tier_scores(feats: pd.DataFrame, model: dict) -> np.ndarray:
    """レースの点 = 軸ごとの、その水準の縮めた回収率の平均（判定に足りない水準は全体の回収率）。"""
    tot = np.zeros(len(feats))
    for ax, lv in model["axes"].items():
        tot += feats[ax].astype(str).map({k: v["shrunk"] for k, v in lv.items()}).fillna(model["overall"]).to_numpy(float)
    return tot / max(len(model["axes"]), 1)


def assign_tiers(feats: pd.DataFrame, model: dict) -> np.ndarray:
    """0 = 見送り（不得意）、1 = ★、2 = ★★、3 = ★★★。"""
    sc = tier_scores(feats, model)
    t = 1 + (sc > model["cuts"][0]).astype(int) + (sc > model["cuts"][1]).astype(int)   # 区切りと同じ点は下の段（上に寄せない）
    return np.where(skip_mask(feats, model), 0, t)


def cmd_tiers(a) -> None:
    """凍結の候補の買い方（--spec の JSON）で、得意・不得意の地図と推奨度の段を作る。訓練の値は年の交差検証、評価は段の順の確かめだけ。"""
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out = OUT_DIR / inputs.get("tag", "v6e")
    tr_spec = inputs["periods"]["train"]
    cuts = json.loads((out / f"qcuts_train_{strategy_hash({'years': tr_spec['years'], 'pred': tr_spec['pred'], **({'decision_odds': tr_spec['decision_odds']} if tr_spec.get('decision_odds') else {})})}.json").read_text())
    per, _ = prepare_periods(inputs, ("train", "eval"), cuts)
    spec = resolve_auto([json.loads(Path(a.spec).read_text(encoding="utf-8"))], out)[0]
    base = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"][0]
    tr, ev = per["train"], per["eval"]
    rt, re_ = run_period(tr, spec), run_period(ev, spec)
    b0 = run_period(tr, base)
    base_roi = float(b0["ret"].sum() / b0["cost"].sum())
    years = tr.feats["year"].to_numpy()
    allm = np.ones(len(tr.races), bool)
    axes = [x for x in a.axes.split(",")] if a.axes else list(TIER_AXES)
    full = fit_tiers(rt["cost"], rt["ret"], tr.feats, allm, base_roi, tr.dates, tr.W, years, axes)
    # 訓練の値は年の交差検証（ある年で判定・区切りを決め、もう一方の年に当てる）
    tcv = np.zeros(len(tr.races), int)
    for y in np.unique(years):
        fm = years != y
        base_fold = float(b0["ret"][fm].sum() / max(b0["cost"][fm].sum(), 1))      # 基準も fit の年だけで
        mdl = fit_tiers(rt["cost"], rt["ret"], tr.feats, fm, base_fold, tr.dates, tr.W, years, axes)
        tcv[years == y] = assign_tiers(tr.feats[years == y].reset_index(drop=True), mdl)
    tev = assign_tiers(ev.feats, full)
    rows_map = []
    for ax, lab in ((x, TIER_AXES[x]) for x in axes):
        st = level_stats(rt["cost"], rt["ret"], tr.feats[ax].astype(str).to_numpy(), years, tr.dates, tr.W)
        for lv, x in sorted(st.items(), key=lambda kv: -kv[1]["roi"]):
            rows_map.append([lab, lv, f"{x['bet_races']:,}", f"{x['hit_rate'] * 100:.1f}%", f"{x['roi'] * 100:.1f}%", f"{x['roi_lo'] * 100:.0f}〜{x['roi_hi'] * 100:.0f}%",
                             f"{x['roi_ex_top5'] * 100:.1f}%", " / ".join(f"{k} {v * 100:.0f}%" for k, v in x["folds"].items() if v is not None),
                             full["axes"][ax][lv]["judge"], f"{full['axes'][ax][lv]['shrunk'] * 100:.1f}%"])

    def tier_rows(cost, ret, tiers, dates, W, label):
        rows, res = [], {}
        for t in (3, 2, 1, 0):
            m = tiers == t
            c, r = np.where(m & (cost > 0), cost, 0), np.where(m & (cost > 0), ret, 0)
            st = metrics(c, r, dates, W)
            res[str(t)] = {k: st.get(k) for k in KEYS_CV}
            nm = TIER_NAMES[3 - t] if t > 0 else "見送り（不得意）"
            rows.append([label, nm, f"{st.get('bet_races', 0):,}", f"{(st.get('hit_rate') or 0) * 100:.1f}%", f"{(st.get('roi') or 0) * 100:.1f}%",
                         (f"{st['roi_lo'] * 100:.0f}〜{st['roi_hi'] * 100:.0f}%" if st.get("roi_lo") is not None else "－"),
                         f"{(st.get('roi_ex_top5') or 0) * 100:.1f}%", f"{(st.get('avg_return') or 0):,.0f}", f"{(st.get('median_return') or 0):,.0f}"])
        return rows, res
    r1, s_tr = tier_rows(rt["cost"], rt["ret"], tcv, tr.dates, tr.W, "訓練（年の交差検証）")
    if a.no_eval:                                     # 段の作り方を訓練だけで比べるとき（評価の段は出さない・返すだけ）
        tv = {k: {kk: v.get(kk) for kk in ("bet_races", "roi", "roi_ex_top5")} for k, v in tier_rows(rt["cost"], rt["ret"], tcv, tr.dates, tr.W, "")[1].items()}
        print(json.dumps({"axes": axes, "train_cv": tv}, ensure_ascii=False, default=_np))
        return {"axes": axes, "train_cv": tv}
    r2, s_ev = tier_rows(re_["cost"], re_["ret"], tev, ev.dates, ev.W, "評価（確かめ）")
    order_ok = all((s_ev[str(t)].get("roi") or 0) >= (s_ev[str(t - 1)].get("roi") or 0) for t in (3, 2))
    order_tr = all((s_tr[str(t)].get("roi") or 0) >= (s_tr[str(t - 1)].get("roi") or 0) for t in (3, 2))
    res = {"title": "得意・不得意の地図と推奨度の段（指示書 6）", "spec": spec, "model": full, "axes": axes, "base_roi_train": base_roi,
           "tiers_train_cv": s_tr, "tiers_eval": s_ev, "order_train_cv": order_tr, "order_eval": order_ok,
           "text": [f"買い方: {spec.get('name')}。判定と段の区切りは訓練の期間だけで決めた（評価の期間は段の順に回収率が並ぶかを見るだけ）。",
                    f"水準の判定（参戦 {TIER_MIN_N}R 以上の水準だけ）: 不得意 = 回収率の 95% 区間の上限が基準 B0 を下回る → 推奨しない（見送り）。"
                    f"得意 = 区間の下限が基準 B0（{base_roi * 100:.1f}%）以上・年ごとがすべて 100% 以上・上位 5 件を除いても 100% 以上。それ以外は判定できない。",
                    f"推奨度の点 = 軸ごとの、そのレースの水準の回収率を件数で全体に縮めた値（重み {TIER_PRIOR}R）の平均。不得意に当たらないレースを、点の 3 分位で ★★★・★★・★ に分けた。",
                    "注意: 段を付ける買い方そのものは訓練の期間で選んだもので、段の交差検証は買い方の選び直しまではしていない（買い方の選び方の楽観は残る）。"
                    "交差検証の中の判定は、年ごとの条件が fit の 1 年分だけになる。9 軸の多くの水準を 95% 区間で判定するので、偶然の不得意・得意も含みうる。",
                    f"訓練の段の値は年の交差検証（1 年を外し、残りの年で判定・区切りを決めて、外した年に当てた）。段の順（★★★ ≥ ★★ ≥ ★）: 訓練 {'並んだ' if order_tr else '並ばなかった'}・評価 {'並んだ' if order_ok else '並ばなかった'}。",
                    "画面との関係（案。この調査では画面は変えない）: 今の UI のレース選択は、推奨ポートフォリオの期待回収率で色を分けている（config/ev_calibration.json の levels、★オレンジ）。"
                    "ここの段（★★★・★★・★・見送り）を、その levels の段に当てる形が考えられる。"],
           "tables": [{"title": "推奨度の段", "head": ["期間", "段", "参戦", "的中率", "回収率", "95% 区間", "上位5件を除く", "平均回収額", "中央値回収額"], "rows": r1 + r2, "left": 2},
                      {"title": "得意・不得意の地図（訓練の期間、選んだ買い方の水準ごと）",
                       "head": ["軸", "水準", "参戦", "的中率", "回収率", "95% 区間", "上位5件を除く", "年ごと", "判定", "縮めた回収率"], "rows": rows_map, "left": 2}]}
    (out / f"stage_tiers{('_' + a.tag) if a.tag else ''}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    print(json.dumps({"train_cv": s_tr, "eval": s_ev, "order_train_cv": order_tr, "order_eval": order_ok}, ensure_ascii=False, indent=1, default=_np))


# ── 最後の候補・凍結・test を 1 回だけ ────────────────────────────────────────


def _period_out(inputs: dict) -> tuple[Path, dict, dict]:
    out = OUT_DIR / inputs.get("tag", "v6e")
    tr_spec = inputs["periods"]["train"]
    cuts = json.loads((out / f"qcuts_train_{strategy_hash({'years': tr_spec['years'], 'pred': tr_spec['pred'], **({'decision_odds': tr_spec['decision_odds']} if tr_spec.get('decision_odds') else {})})}.json").read_text())
    return out, tr_spec, cuts


def cmd_pick_final(a) -> None:
    """selcv の結果から、決めた決まり（min(上位5件除き, 年ごと) が最大・的中率の下限・参戦 300R 以上・どの年にも候補がある）で最後の候補を選び、
    その買い方（固定の格子なら、その選び方を訓練全体に当てて選んだ 1 本。方策なら訓練全体で当てはめた方策）を書き出す。"""
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out, tr_spec, cuts = _period_out(inputs)
    sel = json.loads((out / "stage_selcv.json").read_text(encoding="utf-8"))
    rows = [r for r in sel["rows"] if r.get("cv_complete") and r.get("score") is not None
            and (r["cv"].get("hit_rate") or 0) >= r["min_hit"] and (r["cv"].get("bet_races") or 0) >= MIN_BET_RACES]
    if not rows:
        raise SystemExit("条件を満たすサイクルがありません")
    rows.sort(key=lambda r: -r["score"])
    if getattr(a, "evidence", False):                       # 根拠の検査を通る中で一番（ユーザー 2026-11-17 の方針）
        best, ev_log = _pick_with_evidence(rows, inputs, out, cuts)
    else:
        best, ev_log = rows[0], []
    n = best["cycle"]
    rec = json.loads((out / f"cycle_{n:02d}.json").read_text(encoding="utf-8"))
    if best["kind"].startswith("方策"):
        name = best["picks"][""]
        spec = (rec.get("policy_specs") or {}).get(name)
        if spec is None:
            spec = next((t.get("spec") for t in rec.get("top", []) if t["name"] == name), None)
        if spec is None:
            cand = next((t for t in rec.get("top", []) if t["name"] == name), None) or (rec["selected"] if rec.get("selected", {}).get("name") == name else None)
            if cand is None:
                raise SystemExit(f"方策 {name} の中身が cycle_{n:02d}.json に見つかりません（サイクルを流し直す）")
            spec = cand["spec"]
    else:                                                    # 固定の格子: 同じ選び方を訓練全体に当てる
        per, _ = prepare_periods(inputs, ("train",), cuts, n_boot=10)
        tr = per["train"]
        years = tr.feats["year"].to_numpy()
        cyc = json.loads((CYC_DIR / f"cycle_{n:02d}.json").read_text(encoding="utf-8"))
        strats = [x for x in resolve_auto(expand_variants(cyc), out) if not family(x).startswith("参考") and fav1_rule_ok(x)]
        key = strategy_hash({"pred": tr_spec["pred"], "years": tr_spec["years"], "cuts": cuts, "decision_odds": tr_spec.get("decision_odds"), "pre": (_sha(PRE_ODDS) if tr_spec.get("decision_odds") and PRE_ODDS.exists() else None),
                          "specs": [strategy_hash(x) for x in strats]})
        zp = out / f"cands_c{n:02d}_{key}.npz"
        if not zp.exists():
            raise SystemExit(f"{zp.name} がありません（同じ予測・区切り・買い方で selcv をやり直す）")
        z = np.load(zp, allow_pickle=True)
        assert list(z["names"]) == [x["name"] for x in strats], "候補の配列と設定が合いません（selcv をやり直す）"
        base = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"][0]
        b0 = run_period(tr, base)
        bestj = _fold_pick(z["cost"], z["ret"], np.ones(len(tr.races), bool), (b0["cost"], b0["ret"]), best["min_hit"], MIN_BET_RACES)   # selcv と同じ選び方
        if bestj < 0:
            raise SystemExit("訓練全体で条件を満たす候補がありません")
        spec = strats[bestj]
    per2, _ = prepare_periods(inputs, ("train", "eval"), cuts)       # 選んだ 1 本だけ、訓練全体の値と評価の上向きの確かめ
    base = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"][0]
    chk = {}
    for nm in ("train", "eval"):
        pp = per2[nm]
        r1, r0 = run_period(pp, resolve_auto([spec], out)[0]), run_period(pp, base)
        m1 = metrics_run(r1, pp, pp.W)
        if pp.base is not None:                              # 確定オッズで判定した値（楽観）も
            rf = run_strategy(pp.base.races, pp.base.frecs, resolve_auto([spec], out)[0])
            chk.setdefault("final_odds", {})[nm] = metrics(rf["cost"], rf["ret"], pp.base.dates, pp.base.W)
        m1["h_fav1_rate"] = float(((r1["hpop"] == 1) & (r1["hrank"] >= 0)).sum() / max((r1["hrank"] >= 0).sum(), 1))
        m1["top_fav1_rate"] = float(((r1["top_pop"] == 1) & (r1["hrank"] >= 0)).sum() / max((r1["hrank"] >= 0).sum(), 1))
        chk[nm] = {"metrics": m1, "B0": metrics(r0["cost"], r0["ret"], pp.dates), "diff_vs_B0": paired_diff(r1, r0, pp.W)}
    d = chk["eval"]["diff_vs_B0"]
    chk["eval_verdict"] = ("上向き（基準との差の区間も 0 より上）" if d[1] > 0 else ("上向き（点推定のみ。区間は 0 をまたぐ）" if d[0] > 0 else "確かめられず（基準より下）"))
    if not fav1_rule_ok(spec):
        raise SystemExit("選んだ買い方が 1 番人気の方針に合いません（selcv をやり直す）")
    res = {"cycle": n, "kind": best["kind"], "name": spec.get("name"), "selcv_score": best["score"], "selcv_cv": best["cv"], "spec": spec, "check": chk,
           "evidence_log": ev_log,
           "rule_evidence": ("根拠の検査（年・似た単位は選び方の交差検証で、偶然の見込み、隣の水準、仕組みの説明）を通る中で、物差しが最大のもの" if ev_log else ""),
           "ranking": [{"cycle": r["cycle"], "kind": r["kind"], "score": r["score"], "roi_cv": r["cv"].get("roi")} for r in rows[:10]],
           "rule": "selcv の物差し min(上位5件を除いた回収率, 年ごとの回収率) が最大、的中率の下限・参戦 300R 以上・どの年にも候補がある・参考の系統を除く（cycle 13 の後に固定）"}
    (out / "final_candidate.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    pc = lambda x: "－" if x is None else f"{x * 100:.1f}%"           # noqa: E731
    rowm = lambda lab, m: [lab, f"{m.get('races', 0):,}", f"{m.get('bet_races', 0):,}（{pc(m.get('bet_rate'))}）", pc(m.get("h_fav1_rate")), pc(m.get("hit_rate")),   # noqa: E731
                           pc(m.get("roi")), f"{(m.get('avg_return') or 0):,.0f}", f"{(m.get('median_return') or 0):,.0f}", f"{(m.get('median_hit_return') or 0):,.0f}",
                           (f"{pc(m.get('roi_lo'))}〜{pc(m.get('roi_hi'))}" if m.get("roi_lo") is not None else "－"), pc(m.get("roi_ex_top5"))]
    stage = {"title": "最後の候補（決めた決まりで選んだ 1 本。凍結の前）", "text": [res["rule"], f"選んだ: cycle {n}・{best['kind']}・{spec.get('name')}",
             f"評価の確かめ: {chk['eval_verdict']}（B0 との差 {d[0] * 100:+.1f} 点、区間 {d[1] * 100:+.1f}〜{d[2] * 100:+.1f}）。評価は選び直しに使っていない。"],
             "tables": [{"title": "必須の指標", "head": ["期間", "全レース", "参戦（率）", "◎ が1番人気", "的中率", "回収率", "平均回収額", "中央値回収額", "的中時の中央値", "95% 区間", "上位5件を除く"],
                         "rows": [rowm("選び方の交差検証（訓練）", best["cv"] | {"races": chk["train"]["metrics"].get("races")}), rowm("訓練（全体で選んだ 1 本）", chk["train"]["metrics"]),
                                  rowm("評価（確かめ）", chk["eval"]["metrics"]), rowm("基準 B0・評価", chk["eval"]["B0"])], "left": 1},
                        {"title": "選び方の交差検証の順位（上位 10）", "head": ["回", "種類", "物差し", "交差検証の回収率"],
                         "rows": [[str(r["cycle"]), r["kind"], f"{r['score'] * 100:.1f}%", pc(r["roi_cv"])] for r in res["ranking"]], "left": 2}]}
    (out / "stage_candidate.json").write_text(json.dumps(stage, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    (out / "final_candidate_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("cycle", "kind", "name", "selcv_score", "ranking")}, ensure_ascii=False, indent=1, default=_np))
    print("訓練（全体）:", {k: chk["train"]["metrics"].get(k) for k in ("bet_races", "races", "hit_rate", "roi", "roi_ex_top5", "h_fav1_rate")})
    print("評価:", {k: chk["eval"]["metrics"].get(k) for k in ("bet_races", "races", "hit_rate", "roi", "roi_ex_top5", "h_fav1_rate")}, chk["eval_verdict"])


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _exc_r_values(x) -> list:
    """買い方の中の、1 番人気の r の例外のしきい値（fav1_exc で exc_by="r" の r_min）だけを集める。"""
    out = []
    if isinstance(x, dict):
        if x.get("rule") == "fav1_exc" and x.get("exc_by") == "r":
            out.append(x.get("r_min"))
        for v in x.values():
            out += _exc_r_values(v)
    elif isinstance(x, list):
        for v in x:
            out += _exc_r_values(v)
    return out


def _fwd_keys(inputs: dict) -> set:
    """2026 年の参考にかかわる入力（予測・その年の出走表・発走前オッズ）の、ハッシュの鍵の集合。"""
    sub = {"periods": {k: v for k, v in inputs["periods"].items() if k.startswith("fwd")}}
    return set(_input_hashes(sub)) - {str(Path(x)) for x in [str(CFG_DIR / "baselines.json")]} - set(_input_hashes({"periods": {}}))


def _input_hashes(inputs: dict) -> dict:
    """凍結に入れる入力のハッシュ: 予測・払戻の表・発走前オッズ・基準の設定・道具そのもの。"""
    paths = [str(pred_path(x)) for sp in inputs["periods"].values() for x in sp["pred"]]
    paths += [str(x) for x in PAYOFF_PATHS] + [x for sp in inputs["periods"].values() for x in ([sp["pre_odds"]] if sp.get("pre_odds") else [])]
    paths += [str(CFG_DIR / "baselines.json"), str(Path(__file__).resolve())]
    if any((sp.get("decision_odds") or {}).get("kind") == "sim_pre" for sp in inputs["periods"].values()):
        paths.append(str(PRE_ODDS))                          # 発走前の再現のずれの元
    from src.config.data_paths import TABLES_DIR

    yrs = sorted({y for sp in inputs["periods"].values() for y in sp["years"]})
    paths += [str(Path(TABLES_DIR) / str(y) / "race_result_flat.parquet") for y in yrs]   # 出走表・確定オッズ・払戻の列
    out = {}
    for x in dict.fromkeys(paths):
        pth = Path(x) if str(x).startswith("/") else ROOT / x
        key = str(x)
        try:
            key = str(Path(x).resolve().relative_to(ROOT.resolve())) if str(x).startswith("/") else str(x)   # 別の clone からでも同じ鍵
        except ValueError:
            pass
        out[key] = _sha(pth) if pth.exists() else None
    return out


def cmd_freeze(a) -> None:
    """最後の候補を凍結する: 買い方・分位の区切り・例外のしきい値・推奨度の段のモデル・予測ファイルのハッシュを 1 つにまとめ、
    config/bet_cycles/frozen_<日付>.json に書く（commit してから final を流す）。"""
    from datetime import date

    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out, tr_spec, cuts = _period_out(inputs)
    cand = json.loads((out / "final_candidate.json").read_text(encoding="utf-8"))
    for f in ("stage_tiers.json", "stage_fav1_exception.json"):
        if not (out / f).exists():
            raise SystemExit(f"{f} がありません（tiers-select・fav1 を先に流す）")
    tiers = json.loads((out / "stage_tiers.json").read_text(encoding="utf-8"))
    fav1 = json.loads((out / "stage_fav1_exception.json").read_text(encoding="utf-8"))
    fav1r = json.loads((out / "stage_fav1r_exception.json").read_text(encoding="utf-8")) if (out / "stage_fav1r_exception.json").exists() else {}
    spec = resolve_auto([cand["spec"]], out)[0]
    exc_r = _exc_r_values(spec)                              # r の例外のしきい値（exc_by="r" の階層だけ）が fav1 --by r の結果と同じか
    if exc_r:
        chr_ = fav1r.get("chosen")
        want = float(chr_["p2_min"]) if chr_ else 9.0
        if any(abs(float(v) - want) > 1e-9 for v in exc_r):
            raise SystemExit("買い方の r の例外のしきい値が fav1 --by r の結果と違います（pick-final をやり直す）")
    if strategy_hash(tiers.get("spec")) != strategy_hash(spec):
        raise SystemExit("stage_tiers.json の買い方が最後の候補と違います（tiers-select を最後の候補でやり直す）")
    if "selection" not in tiers:
        raise SystemExit("stage_tiers.json が tiers-select で作られていません（tiers-select をやり直す）")
    ch = fav1.get("chosen") or {}
    p2s = {x for x in json.dumps(spec).split('"p2_min": ')[1:]}
    if ch and any(not x.startswith(str(float(ch["p2_min"]))) for x in p2s):
        raise SystemExit("買い方の例外のしきい値が fav1 の結果と違います（pick-final をやり直す）")
    preds = _input_hashes(inputs)
    missing = [k for k, v in preds.items() if v is None]
    if missing:
        raise SystemExit(f"凍結に入れる入力が見つかりません: {missing}")
    body = {"frozen_at": date.today().isoformat(), "tag": inputs.get("tag"), "inputs": inputs, "pred_sha256": preds, "spec": spec,
            "selection": {k: cand[k] for k in ("cycle", "kind", "name", "selcv_score", "ranking", "rule")}, "qcuts": cuts,
            "fav1": fav1.get("chosen"), "fav1r": fav1r.get("chosen"), "tiers_model": (tiers or {}).get("model"), "tiers_eval_order": (tiers or {}).get("order_eval"),
            "note": "test 2025 と 2026 年は、このファイルを commit した後に bet_cycles final で 1 回だけ見る。見た後は変えない（変えたら別の凍結として始める）。"
                    + ((" " + a.note) if getattr(a, "note", "") else "")}
    body["hash"] = strategy_hash({k: v for k, v in body.items() if k != "frozen_at"})
    dst = Path(a.out) if a.out else CFG_DIR / f"frozen_{date.today().strftime('%Y%m%d')}_{inputs.get('tag')}.json"
    if dst.exists():
        raise SystemExit(f"{dst} はすでにあります（凍結は書き換えない）")
    dst.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=_np) + "\n", encoding="utf-8")
    print(f"凍結: {dst}（hash {body['hash']}）。commit してから final を流す。")


_MARK_RE = __import__("re").compile(r"^(?:TEST2025_EVALUATED_|bet_test2025_)(r(\d+)_)?(test_|fwd_)?([0-9a-f]{12})\.json$")


def _marker_names(out: Path) -> list[str]:
    return [x.name for x in sorted(OUT_DIR.glob("*/TEST2025_EVALUATED_*.json"))] + \
        [x.name for x in sorted((ROOT / "docs/operations/retrain_trials").glob("bet_test2025_*.json"))]


def test_markers_block(rnd: int, grp: str, names: list[str]) -> tuple[bool, str]:
    """test・2026 年を見る前の印の判定。回は 1・2 だけ。第 2 回は第 1 回の印があるときだけ。同じ回で、
    test を見る（grp="" か "test"）のは、その回の印（group なし・test）が無いときだけ。2026 年を見る（"fwd"）のは、その回の test が済み、
    fwd の印（group なし・fwd）が無いときだけ。"""
    if rnd not in (1, 2):
        return True, f"回は 1・2 だけです（{rnd}）"
    marks = []
    for n in names:
        m = _MARK_RE.match(n)
        if m:
            marks.append((int(m.group(2) or 1), (m.group(3) or "").rstrip("_")))
    if rnd == 2 and not any(r_ == 1 for r_, _ in marks):
        return True, "第 2 回は、第 1 回の test を見た後だけです"
    same = [g for r_, g in marks if r_ == rnd]
    if grp in ("", "test") and any(g in ("", "test") for g in same):
        return True, f"第 {rnd} 回の test 2025 はすでに見ています（1 回だけ）"
    if grp in ("", "fwd") and any(g in ("", "fwd") for g in same):
        return True, f"第 {rnd} 回の 2026 年の参考はすでに見ています（1 回だけ）"
    if grp == "fwd" and not any(g in ("", "test") for g in same):
        return True, f"第 {rnd} 回の 2026 年の参考は、test 2025 を見た後に見ます（順序）"
    return False, ""


def cmd_final(a) -> None:
    """凍結した買い方で、test 2025 と 2026 年の参考を 1 回だけ見る。凍結ファイルが git に commit 済みで変わっていないこと、
    ハッシュが合うこと、まだ一度も見ていないこと（出力が無いこと）を確かめてから流す。"""
    import subprocess

    fz = Path(a.frozen)
    body = json.loads(fz.read_text(encoding="utf-8"))
    if strategy_hash({k: v for k, v in body.items() if k not in ("frozen_at", "hash")}) != body["hash"]:
        raise SystemExit("凍結ファイルのハッシュが合いません（中身が変わった）")
    rel = str(fz.resolve().relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise SystemExit("凍結ファイルを commit してから流してください（test を見る前に固定する）")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/scripts/data/bet_cycles.py", "config/bet_cycles"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:                                                # 未追跡のファイルも含めて見る
        raise SystemExit(f"道具か設定に未コミットの変更があります（commit してから流す）: {dirty[:200]}")
    out = OUT_DIR / body["tag"]
    rnd = int(getattr(a, "round", 1) or 1)
    grp = getattr(a, "group", "") or ""
    blocked, why = test_markers_block(rnd, grp, _marker_names(out))
    if blocked:
        raise SystemExit(why)
    tagr = ("" if rnd == 1 else f"r{rnd}_") + (f"{grp}_" if grp else "")
    dst = out / f"TEST2025_EVALUATED_{tagr}{body['hash']}.json"
    inputs = body["inputs"]
    now = _input_hashes(inputs)
    changed = {}
    for k, v in body["pred_sha256"].items():                 # 予測・払戻・発走前オッズ・基準・道具が凍結の時と同じか
        if v is None or now.get(k) != v:
            if grp == "fwd" and now.get(k) is not None and k in _fwd_keys(inputs):
                changed[k] = {"frozen": v, "now": now.get(k)}   # 2026 年の参考は、凍結の後に差し替わった版で見てよい（統括役の判断）。記録に残す
                continue
            raise SystemExit(f"入力が凍結の時と違うか、見つかりません: {k}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dst.write_text(json.dumps({"status": "started", "hash": body["hash"]}), encoding="utf-8")   # 途中で落ちても 2 回目を流さない印
    (ROOT / f"docs/operations/retrain_trials/bet_test2025_{tagr}{body['hash']}.json").write_text(
        json.dumps({"status": "started", "hash": body["hash"], "frozen": rel}, ensure_ascii=False) + "\n", encoding="utf-8")   # git にも残す印（走り始めに書く）
    names = [x for x in inputs["periods"] if (x == "test" if grp == "test" else (x.startswith("fwd") if grp == "fwd" else x in ("test", "fwd2026", "fwd2026_pre")))]
    per, _ = prepare_periods(inputs, tuple(names), cuts=body["qcuts"])
    spec = body["spec"]
    bases = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"]
    res = {"hash": body["hash"], "frozen": rel, "git_head": head, "periods": {}, "round": rnd,
           "round_note": "" if rnd == 1 else f"第 {rnd} 回: 2025 年 test は前の回の最適化で 1 回使っている。今回の値は 2 回目の確かめ（別のモデル・別の買い方）。",
           "group": grp, "inputs_changed_after_freeze": changed,
           "note": "test は凍結を commit した後に 1 回だけ見た。結果は docs/operations/retrain_trials/bet_test2025_<hash>.json に写して commit する（2 回目を拒む印）。"}
    for nm, p in per.items():
        r = run_period(p, spec, detail=True)
        m = metrics_run(r, p, p.W)
        if p.base is not None:
            rf = run_strategy(p.base.races, p.base.frecs, spec)
            fin_m = metrics(rf["cost"], rf["ret"], p.base.dates, p.base.W)
        else:
            fin_m = None
        m["h_fav1_rate"] = float(((r["hpop"] == 1) & (r["hrank"] >= 0)).sum() / max((r["hrank"] >= 0).sum(), 1))
        m["top_fav1_rate"] = float(((r["top_pop"] == 1) & (r["hrank"] >= 0)).sum() / max((r["hrank"] >= 0).sum(), 1))
        bs = {}
        for b in bases[:2] + bases[-1:]:
            rb = run_period(p, b)
            bs[b["name"]] = {k: metrics(rb["cost"], rb["ret"], p.dates).get(k) for k in KEYS_CV}
        rb0 = run_period(p, bases[0])
        d = paired_diff(r, rb0, p.W)
        tiers = {}
        if body.get("tiers_model"):
            t = assign_tiers(p.feats, body["tiers_model"])
            for k in (3, 2, 1, 0):
                mk = (t == k) & (r["cost"] > 0)
                st = metrics(np.where(mk, r["cost"], 0), np.where(mk, r["ret"], 0), p.dates, p.W)
                tiers[str(k)] = {kk: st.get(kk) for kk in KEYS_CV}
        f1 = fav1_rows(p)
        rec_ = r["records"]
        if "ver" not in rec_:
            rec_["ver"] = 0
        mk_ = rec_["reason"].str.contains("1番人気の例外", na=False)
        used = set(zip(rec_.loc[mk_, "ver"], rec_.loc[mk_, "race_id"]))     # 買い方が実際に例外で 1 番人気を ◎ にした（版, レース）
        exc = fav1_stats(f1[[(v, x) in used for v, x in zip(f1["ver"], f1["race_id"])]], p.k) if len(f1) else {}
        sp_ = inputs["periods"][nm]
        res["periods"][nm] = {"metrics": m, "metrics_final_odds": fin_m, "baselines": bs, "diff_vs_B0": d, "tiers": tiers, "fav1_exception": exc,
                              "role": sp_.get("role"), "odds_source": ("発走前のオッズ（実際のスナップショット）で ◎・人気を決めた" if sp_.get("pre_odds") else
                                             (f"発走前のオッズの再現（{p.k} 通りを並べた平均）で判定" if p.k > 1 else "確定オッズで判定")),
                              "pred": sp_["pred"],
                              "drop": p.drop, "miss_types": miss_types(r), "h_reasons": r["records"]["reason"].value_counts().to_dict()}
        r["records"].to_parquet(out / f"final_{nm}_races_{body['hash']}.parquet", index=False)
    res["status"] = "done"
    dst.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    (ROOT / f"docs/operations/retrain_trials/bet_test2025_{tagr}{body['hash']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np) + "\n",
                                                                                        encoding="utf-8")
    print(f"結果を docs/operations/retrain_trials/bet_test2025_{tagr}{body['hash']}.json に書いた。commit すること（同じ回の 2 回目を拒む印になる）。")
    print(json.dumps({nm: {k: v.get(k) for k in ("bet_races", "races", "hit_rate", "roi", "roi_lo", "roi_hi", "avg_return", "median_return", "roi_ex_top5")}
                      for nm, v in ((n_, x["metrics"]) for n_, x in res["periods"].items())}, ensure_ascii=False, indent=1, default=_np))


PREREG_ORDER = ("否定", "100% 超えを確かめた", "100% 超えの向き（確かめは弱い）", "基準より上（100% は確かめられず）", "確かめられず")


def prereg_verdict(m: dict, d: tuple, levels: list[dict]) -> str:
    """事前登録の判定の基準（criteria.levels）を、事前登録の criteria.order の順に当て、最初に当たった段を返す（段が重なるときは先のもの）。
    levels の if の文は人が読むためのもの。式はここで固定し、ラベルと式の対応はテストで押さえる。"""
    roi, lo, hi, ex = (m.get(k) for k in ("roi", "roi_lo", "roi_hi", "roi_ex_top5"))
    d0, dlo, dhi = d
    if roi is None or lo is None or hi is None or ex is None:
        return "判定できない（参戦なし）"
    have = {x["label"] for x in levels}
    if set(PREREG_ORDER) != have:
        raise ValueError(f"事前登録の段のラベルが道具と違います: {sorted(have ^ set(PREREG_ORDER))}")
    if d0 <= 0 or hi < 1.0:
        return "否定"
    if dlo > 0 and lo >= 1.0 and ex >= 1.0:
        return "100% 超えを確かめた"
    if dlo > 0 and roi >= 1.0 and ex >= 1.0:
        return "100% 超えの向き（確かめは弱い）"
    if dlo > 0:
        return "基準より上（100% は確かめられず）"
    return "確かめられず"


PREREG_MIN_COVER = 0.95      # 窓の全レースのうち、予測がそろって計算できたレースの割合の下限（下回れば印を書く前に止める）


def cmd_prereg_run(a) -> None:
    """事前登録（config/bet_cycles/prereg_*.json）の 1 つの群を、凍結した買い方で 1 回だけ流す。
    結果を見る前に: 凍結・事前登録・道具・config/bet_cycles が commit 済みで変わっていないこと、凍結の入力（2026 年の分を除く）が凍結のときと同じこと、
    道具が test 2025 の記録を再現すること、その群の印が手元にも git の履歴にも無いこと、予測が窓をほぼ覆うことを確かめ、それから印を書いて流す。"""
    import subprocess

    fz, pr = Path(a.frozen), Path(a.prereg)
    body = json.loads(fz.read_text(encoding="utf-8"))
    reg = json.loads(pr.read_text(encoding="utf-8"))
    if strategy_hash({k: v for k, v in body.items() if k not in ("frozen_at", "hash")}) != body["hash"] or reg.get("frozen_hash") != body["hash"]:
        raise SystemExit("凍結のハッシュが合わないか、事前登録の凍結と違います")
    for f in (fz, pr, Path(__file__).resolve()):
        rel = str(f.resolve().relative_to(ROOT))
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
        if not (tracked and clean):
            raise SystemExit(f"{rel} を commit してから流してください（結果を見る前に固定する）")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "config/bet_cycles"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"config/bet_cycles に未コミットの変更があります: {dirty[:200]}")
    inputs = body["inputs"]
    now = _input_hashes(inputs)
    tool_key = str(Path(__file__).resolve().relative_to(ROOT))
    fwd = _fwd_keys(inputs)
    for k, v in body["pred_sha256"].items():                 # 2026 年の分（予測・出走表・発走前オッズ）と道具（下の再現で確かめる）以外は凍結のときと同じ
        if k == tool_key or k in fwd:
            continue
        if v is None or now.get(k) != v:
            raise SystemExit(f"入力が凍結の時と違います: {k}")
    grp = next((g for g in reg["groups"] if g["id"] == a.group), None)
    if grp is None:
        raise SystemExit(f"事前登録に群 {a.group} がありません")
    lo_, hi_ = grp["dates"]
    if not (_re.fullmatch(r"\d{4}-\d{2}-\d{2}", lo_) and _re.fullmatch(r"\d{4}-\d{2}-\d{2}", hi_) and lo_ <= hi_):
        raise SystemExit(f"事前登録の日付の窓が YYYY-MM-DD の 2 つ（lo ≤ hi）ではありません: {grp['dates']}")
    name = f"bet_2026_{body['tag']}_{grp['id']}_{body['hash']}.json"
    rel_mark = f"docs/operations/retrain_trials/{name}"
    mark = ROOT / rel_mark
    seen = subprocess.run(["git", "log", "--all", "--oneline", "--", rel_mark], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if mark.exists() or seen:
        raise SystemExit(f"群 {grp['id']} はすでに流しています（1 回だけ）: {name}")
    # 再現のずれの元（発走前オッズ）は週ごとに増えるので、凍結のときの中身の写しに固定する（sha を照合）
    global DRIFT_SRC
    want = body["pred_sha256"].get(str(PRE_ODDS.relative_to(ROOT))) or body["pred_sha256"].get(str(PRE_ODDS))
    cand_ = [PRE_ODDS] + sorted((OUT_DIR / body["tag"]).glob("pre_odds_frozen_*.parquet"))
    DRIFT_SRC = next((c for c in cand_ if c.exists() and _sha(c) == want), None)
    if DRIFT_SRC is None:
        raise SystemExit("凍結のときの発走前オッズ（再現のずれの元）が見つかりません（写し pre_odds_frozen_*.parquet を置く）")
    # 道具が凍結のときと同じ判定をすることを、2025 年 test の記録の再現で確かめる（回収率・区間・上位 5 件除き・B0 との差）
    tr = json.loads((ROOT / f"docs/operations/retrain_trials/bet_test2025_r2_test_{body['hash']}.json").read_text(encoding="utf-8"))
    if tr.get("status") != "done":
        raise SystemExit("test 2025 の記録が完了していません")
    rm = tr["periods"]["test"]["metrics"]
    bases = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"]
    pt, _ = prepare_periods({"periods": {"test": inputs["periods"]["test"]}}, ("test",), cuts=body["qcuts"])
    rt = run_period(pt["test"], body["spec"])
    mt = metrics_run(rt, pt["test"], pt["test"].W)
    dt = paired_diff(rt, run_period(pt["test"], bases[0]), pt["test"].W)
    bad = [k for k in ("races", "bet_races", "roi", "roi_lo", "roi_hi", "roi_ex_top5", "hit_rate") if abs(float(mt.get(k) or 0) - float(rm.get(k) or 0)) > 1e-9]
    bad += ["diff_vs_B0"] if any(abs(x - y) > 1e-9 for x, y in zip(dt, tr["periods"]["test"]["diff_vs_B0"])) else []
    if bad:
        raise SystemExit(f"道具が凍結のときと同じ判定をしません（test 2025 の記録と違う値: {bad}）")
    if a.check_only:                                         # 2026 年を読まずに、ここまでの確かめだけ
        print(f"確かめだけ: 凍結・事前登録・入力・道具の再現（test 2025）は合格。ずれの元 {DRIFT_SRC.name}。群 {grp['id']} は未実行")
        return
    base_nm = "fwd2026" if grp["id"] == "main" else "fwd2026_pre"     # 主は T-60 の再現、別枠・参考は本物の T-60
    spec = json.loads(json.dumps(inputs["periods"][base_nm]))
    spec["pred"] = [json.loads(a.pred)]
    spec["dates"] = grp["dates"]
    spec["role"] = f"事前登録の群 {grp['name']}（{lo_}〜{hi_}）"
    pp = pred_path(spec["pred"][0])
    pred_sha = _sha(pp) if pp.exists() else None
    if pred_sha is None:
        raise SystemExit(f"予測のファイルがありません: {pp}")
    per, _ = prepare_periods({"periods": {grp["id"]: spec}}, (grp["id"],), cuts=body["qcuts"])   # 着順・払戻の集計はまだしない
    p = per[grp["id"]]
    dr = dict(p.drop)
    n_all = int(dr.get("all_races_in_period") or 0)
    dr["no_pre_odds"] = n_all - int(dr.get("computed_races", 0)) - sum(int(dr.get(k, 0)) for k in ("no_pred", "no_pay", "small"))
    cover = dr.get("computed_races", 0) / max(n_all, 1)
    if cover < PREREG_MIN_COVER and not a.allow_low_cover:
        raise SystemExit(f"窓の全レース {n_all} のうち計算できたのは {dr.get('computed_races')}（{cover:.1%}）。内訳 {dr}。予測か発走前オッズの覆いが足りません"
                         "（印は書いていない。覆いの下限を下回っても流すなら、理由を記録して --allow-low-cover）")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    mark.write_text(json.dumps({"status": "started", "hash": body["hash"], "group": grp["id"], "git_head": head, "pred": spec["pred"], "pred_sha256": pred_sha,
                                "drop": dr}, ensure_ascii=False, default=_np) + "\n", encoding="utf-8")
    r = run_period(p, body["spec"], detail=True)
    m = metrics_run(r, p, p.W)
    rb0 = run_period(p, bases[0])
    d = paired_diff(r, rb0, p.W)
    m0 = metrics(rb0["cost"], rb0["ret"], p.dates, p.W)
    verdict = prereg_verdict(m, d, reg["criteria"]["levels"])
    res = {"status": "done", "hash": body["hash"], "prereg": str(pr.resolve().relative_to(ROOT)), "group": grp, "git_head": head, "pred": spec["pred"],
           "drift_src": str(DRIFT_SRC), "pre_odds_sha256_now": _sha(ROOT / spec["pre_odds"]) if spec.get("pre_odds") else None,
           "pred_sha256": pred_sha, "metrics": m, "B0": {k: m0.get(k) for k in KEYS_CV}, "diff_vs_B0": d, "verdict": verdict, "drop": dr,
           "cover": cover, "allow_low_cover": bool(a.allow_low_cover),
           "races_note": "指標の races は、窓の全レースのうち予測・払戻（別枠・参考は発走前オッズも）がそろって計算できたレース。内訳は drop。",
           "odds_source": "本物の T-60" if spec.get("pre_odds") else f"T-60 の再現（{p.k} 通り）", "miss_types": miss_types(r),
           "h_reasons": r["records"]["reason"].value_counts().to_dict()}
    mark.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_np) + "\n", encoding="utf-8")
    r["records"].to_parquet(OUT_DIR / body["tag"] / f"prereg_{grp['id']}_races_{body['hash']}.parquet", index=False)
    print(f"結果を {rel_mark} に書いた。commit すること。判定: {verdict}")


TIER_MIN_BET = 300        # 段の作り方を選ぶときの、各段の参戦の下限（訓練の交差検証）
TIER_AXES_CANDS = ["q_ev_max", "q_ev_max,c_class,fieldband", "q_ev_max,c_class", "c_class,c_venue,fieldband", "q_ev_max,shape"]


def cmd_tiers_select(a) -> None:
    """推奨度の段の作り方（点に使う軸の組）を、訓練の年の交差検証だけで選び、選んだ 1 つで stage_tiers.json を作る（評価は段の順を見るだけ）。
    決まり（cycle 15 で固定）: 回収率と上位 5 件を除いた回収率の両方で ★★★ ≥ ★★ ≥ ★ に並ぶ組のうち、★★★ の上位 5 件を除いた回収率が最大。"""
    import contextlib
    import io

    res = []
    for axes in TIER_AXES_CANDS:
        ns = argparse.Namespace(**{**vars(a), "axes": axes, "no_eval": True, "tag": ""})
        with contextlib.redirect_stdout(io.StringIO()):
            tv = cmd_tiers(ns)["train_cv"]
        g = lambda t, k: tv[str(t)].get(k) or 0.0                       # noqa: E731
        ok = all((tv[str(t)].get("bet_races") or 0) >= TIER_MIN_BET for t in (3, 2, 1)) and \
            all(g(t, "roi") >= g(t - 1, "roi") and g(t, "roi_ex_top5") >= g(t - 1, "roi_ex_top5") for t in (3, 2))
        res.append({"axes": axes, "train_cv": tv, "ordered": ok, "top_ex5": g(3, "roi_ex_top5")})
        print(axes, ok, round(g(3, "roi"), 3), round(g(2, "roi"), 3), round(g(1, "roi"), 3), flush=True)
    okr = [r for r in res if r["ordered"]]
    inputs0 = load_inputs(Path(a.inputs) if a.inputs else None)
    out0 = OUT_DIR / inputs0.get("tag", "v6e")
    if not okr:                                              # 段は作らない（買うレースはすべて同じ推奨度）。記録だけ残す
        spec0 = resolve_auto([json.loads(Path(a.spec).read_text(encoding="utf-8"))], out0)[0]
        none = {"title": "推奨度の段: 作らない（段の順に並ぶ作り方が無かった）", "spec": spec0, "model": None, "selection": {"candidates": res, "chosen": None},
                "text": [f"段の作り方の候補 {len(res)} 通りを訓練の年の交差検証で比べたが、各段 {TIER_MIN_BET}R 以上で回収率と上位 5 件を除いた回収率の両方が ★★★ ≥ ★★ ≥ ★ に並ぶものが無かった。"
                         "推奨度の段は作らず、買うレースはすべて同じ推奨度とする（不得意の見送りも行わない）。"],
                "tables": [{"title": "段の作り方の候補（訓練の年の交差検証）", "head": ["点に使う軸", "★★★", "★★", "★", "段の順に並ぶ"],
                            "rows": [[r["axes"]] + [f"{(r['train_cv'][str(t)].get('roi') or 0) * 100:.1f}%（{r['train_cv'][str(t)].get('bet_races') or 0:,}R）" for t in (3, 2, 1)]
                                     + ["×"] for r in res]}]}
        (out0 / "stage_tiers.json").write_text(json.dumps(none, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
        print(f"段の順に並ぶ作り方がありません（各段 {TIER_MIN_BET}R 以上）。推奨度の段は作らない（stage_tiers.json に記録）")
        return
    chosen = max(okr, key=lambda r: r["top_ex5"])
    ns = argparse.Namespace(**{**vars(a), "axes": chosen["axes"], "no_eval": False, "tag": ""})
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_tiers(ns)
    inputs = load_inputs(Path(a.inputs) if a.inputs else None)
    out = OUT_DIR / inputs.get("tag", "v6e")
    st = json.loads((out / "stage_tiers.json").read_text(encoding="utf-8"))
    st["title"] = f"推奨度の段（段の作り方は訓練だけで選んだ: {chosen['axes']}）"
    st["selection"] = {"candidates": res, "chosen": chosen["axes"], "any_ordered": bool(okr),
                       "rule": "回収率と上位 5 件を除いた回収率の両方で ★★★ ≥ ★★ ≥ ★ に並ぶ組のうち、★★★ の上位 5 件を除いた回収率が最大（cycle 15 で固定）"}
    st["tables"].insert(0, {"title": "段の作り方の候補（訓練の年の交差検証だけで比べた）", "head": ["点に使う軸", "★★★", "★★", "★", "段の順に並ぶ", "選んだ"],
                            "rows": [[r["axes"]] + [f"{(r['train_cv'][str(t)].get('roi') or 0) * 100:.1f}%（{r['train_cv'][str(t)].get('bet_races') or 0:,}R）" for t in (3, 2, 1)]
                                     + ["○" if r["ordered"] else "×", "◎" if r is chosen else ""] for r in res]})
    (out / "stage_tiers.json").write_text(json.dumps(st, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    print("選んだ:", chosen["axes"], "評価の段の順:", st.get("order_eval"))


# ── 事象と根拠（ユーザー 2026-11-17「わけもわからずたまたま成績が出るのは勘弁してください」） ──────────────

BENTER = ("Benter, W. (1994) Computer Based Horse Race Handicapping and Wagering Systems: A Report. "
          "In Hausch, Lo & Ziemba (eds.) Efficiency of Racetrack Betting Markets（市場の確率とファンダメンタルの予測を組み合わせると市場より良い確率になる）")
MECHANISM = {
    "r_max": ("市場の確率 q よりモデルの勝率 p が高い馬（ずれ r = log(p/q) が大きい馬）を ◎ にすると、市場が過小評価した馬の払戻を受け取る。",
              "パリミュチュエルのオッズは賭け手の合意で、全部の情報を反映しない。市場の確率とファンダメンタルの予測を組み合わせると、市場より良い確率が得られる（" + BENTER + "）。"),
    "ana_ev": ("勝率 × オッズ（期待値）が大きい 2 番人気以下の馬を ◎ にする。",
               "期待値が 1 を超える賭けだけが長期で利益になる。ただし期待値はオッズの時点に強く左右される（締切直前のオッズの動き: https://arxiv.org/abs/2509.14645）。本番の再現（T-60 で判定）で確かめた値を使う。"),
}
NO_REASON = {
    "fav1_penalty": "1 番人気を避けるだけでは根拠にならない。公開の知見（本命-大穴バイアス: Griffith 1949、Ali 1977）はむしろ人気薄ほど回収率が低いことを示す。ずれ r や期待値との組み合わせが要る。",
    "top": "予測 1 位の約 8 割は 1 番人気で、市場と同じ馬を買うので控除の分だけ負けやすい（参考）。",
}


def mechanism(spec: dict) -> tuple[str, str]:
    """買い方の仕組み（事象と、公開の知見による根拠）。説明の無いものは根拠を空で返す（検査で不合格）。"""
    if "policy" in spec:
        ty = spec["policy"].get("typing", {})
        subs = [v for v in spec["policy"]["map"].values() if v != "skip"]
        ok = bool(subs) and all(mechanism(v)[1] != "" for v in subs)
        return (f"レースを型（{ty.get('name', ty.get('kind'))}）に分け、型ごとに買い方と見送りを変える。",
                ("型ごとの買い方はどれも上の仕組み（ずれ・期待値）による。市場とのずれ・期待値が小さいレースは控除（JRA の払戻率 70〜80%）を超える余地が小さく、"
                 "ずれが大きいレースに絞るほど優位を払戻に変えられる。" if ok else ""))
    h = spec.get("honmei") or {}
    rule = h.get("rule", "top")
    if rule == "fav1_exc":
        rule = (h.get("base") or {}).get("rule", "fav1_penalty")
    if rule in MECHANISM:
        return MECHANISM[rule]
    return ("（この決め方の仕組みの説明なし）" + (f" {NO_REASON[rule]}" if rule in NO_REASON else ""), "")


def _neighbors(spec: dict) -> list[tuple[str, dict]]:
    """主な設定を 1 段ずらした買い方（隣の水準）。とがった 1 点（偶然）ではないかを見る。方策は別に扱う（同じ型の作り方の別の設定）。"""
    out = []

    def mod(path, fn, lab):
        s2 = json.loads(json.dumps(spec))
        d = s2
        ks = path.split(".")
        for k in ks[:-1]:
            if not isinstance(d, dict) or k not in d:
                return
            d = d[k]
        if not isinstance(d, dict) or ks[-1] not in d:
            return
        v = fn(d[ks[-1]])
        if v is None or v == d[ks[-1]]:
            return
        d[ks[-1]] = v
        out.append((lab, s2))
    base_h = "honmei.base" if (spec.get("honmei") or {}).get("rule") == "fav1_exc" else "honmei"
    for f_, lab in ((0.6, "×0.6"), (1.6, "×1.6")):
        mod(f"{base_h}.p_min", lambda v, f_=f_: round(float(v) * f_, 3), f"勝率の下限 {lab}")
    for dv in (-1, 1):
        mod(f"{base_h}.max_rank", lambda v, dv=dv: (max(3, int(v) + 2 * dv)) if int(v) < 50 else None, f"予測順位の上限 {dv * 2:+d}")
        mod("skip.h_pop_min", lambda v, dv=dv: max(2, int(v) + dv), f"◎ の人気の下限 {dv:+d}")
        mod("partners.n", lambda v, dv=dv: max(1, int(v) + dv), f"相手の数 {dv:+d}")
    for dv in (-0.2, 0.2):
        mod(f"{base_h}.ev_min", lambda v, dv=dv: round(max(0.0, float(v) + dv), 2), f"期待値の下限 {dv:+.1f}")
        mod(f"{base_h}.r_min", lambda v, dv=dv: round(float(v) + dv / 2, 3) if float(v) > -9 else None, f"r の下限 {dv / 2:+.1f}")
    for c, rng in ((spec.get("skip") or {}).get("when") or {}).items():
        if c.startswith("q_") and rng and rng[0] is not None:
            for dv in (-1, 1):
                lo = int(rng[0]) + dv
                hi = rng[1]
                if 1 <= lo <= 5 and lo != int(rng[0]) and (hi is None or lo < hi):
                    s2 = json.loads(json.dumps(spec))
                    s2["skip"]["when"][c] = [lo, hi]
                    out.append((f"見送り {c} の下限 {dv:+d}", s2))
    return out


def _sign_p(k: int, n: int) -> float:
    """片側の符号検定（帰無: 単位ごとに B0 より上か下かは五分五分）。"""
    from math import comb

    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else 1.0


def evidence_check(spec: dict, per: dict, base_spec: dict, cv_arrays: tuple | None = None, policy_neighbors: list | None = None,
                   cv_folds: dict | None = None) -> dict:
    """候補の買い方の根拠を、実データの統計で確かめる（訓練の期間だけ）。
    年と単位は、選び方の交差検証の配列（cv_arrays = (cost, ret)。1 年を外して選び、外した年に当てた値）で見る（選び方の楽観を除く）。
    1 年ごとに B0 より上、2 芝ダ×距離の単位（参戦 100R 以上が 5 つ以上）の 6 割以上で B0 より上かつ符号検定 p < 0.1、
    3 B0 との差の区間が 0 より上かつ上位 5 件を除いた回収率が B0 の上位 5 件除き以上、4 隣の水準（参戦 300R 以上のもの）の上位 5 件除きが B0 以上で、
    候補から 10 点以上は崩れない（隣が無ければ保留）、5 仕組みの説明（公開の知見）がある。すべて満たせば「根拠あり」。あわせて的中率の分と払戻の分の分解。"""
    tr = per["train"]
    r = run_period(tr, spec)
    b = run_period(tr, base_spec)
    m, mb = metrics_run(r, tr, tr.W), metrics_run(b, tr, tr.W)
    cc, cr = cv_arrays if cv_arrays is not None else (r["cost"], r["ret"])
    yrs = tr.feats["year"].to_numpy()
    folds = {int(y): float(cr[yrs == y].sum() / max(cc[yrs == y].sum(), 1e-9)) for y in np.unique(yrs)}
    if cv_folds:                                       # 方策: もとから交差検証の年ごとの値（selcv の行）を使う。欠けた年は不合格の扱い
        folds = {int(y): (float(v) if v is not None else -1.0) for y, v in cv_folds.items()}
    bfold = {int(y): float(b["ret"][yrs == y].sum() / max(b["cost"][yrs == y].sum(), 1e-9)) for y in np.unique(yrs)}
    years_ok = all(folds[y] > bfold[y] for y in folds)
    unit = (tr.feats["c_surface"].astype(str) + "×" + tr.feats["c_dist_band"].astype(str)).to_numpy()
    units = []
    for u in np.unique(unit):
        mk = unit == u
        mu = metrics(np.where(mk, cc, 0), np.where(mk, cr, 0), tr.dates)
        if (mu.get("bet_races") or 0) < 100:
            continue
        cb, rb = np.where(mk, b["cost"], 0), np.where(mk, b["ret"], 0)
        units.append({"unit": str(u), "bet_races": mu["bet_races"], "roi": mu["roi"], "b0": float(rb.sum() / max(cb.sum(), 1e-9))})
    k_up = sum(x["roi"] > x["b0"] for x in units)
    unit_share = float(k_up / len(units)) if units else 0.0
    unit_p = _sign_p(k_up, len(units))
    units_ok = len(units) >= 5 and unit_share >= 0.6 and unit_p < 0.1
    d = paired_diff(r, b, tr.W)
    chance_ok = d[1] > 0 and (m.get("roi_ex_top5") or 0) >= (mb.get("roi_ex_top5") or 0)
    if policy_neighbors is not None:
        nb = policy_neighbors
    else:
        nb = []
        for lab, s2 in _neighbors(spec):
            mn = metrics_run(run_period(tr, s2), tr)
            nb.append({"change": lab, "roi": mn.get("roi"), "roi_ex_top5": mn.get("roi_ex_top5"), "bet_races": mn.get("bet_races")})
    nb_used = [x for x in nb if (x.get("bet_races") or 0) >= MIN_BET_RACES]
    nb_ok = bool(nb_used) and all((x.get("roi_ex_top5") or 0) >= (mb.get("roi_ex_top5") or 0) and (x.get("roi") or 0) >= (m.get("roi") or 0) - 0.10
                                  for x in nb_used)
    ev, rs = mechanism(spec)
    mech_ok = bool(rs)
    dec = decompose(m, mb)
    ok = years_ok and units_ok and chance_ok and nb_ok and mech_ok
    why = []
    if not years_ok:
        why.append("選び方の交差検証で、訓練の年のどれかが B0 を下回る")
    if not units_ok:
        why.append(f"芝ダ×距離の単位（{len(units)} 個）で B0 を上回るのが {unit_share:.0%}・符号検定 p = {unit_p:.2f}（5 個以上・6 割以上・p<0.1 を満たさない）")
    if not chance_ok:
        why.append("B0 との差の区間が 0 をまたぐか、上位 5 件を除くと B0 の上位 5 件除きより下（偶然の幅）")
    if not nb_ok:
        why.append("隣の水準が無いか、隣で B0 を下回るか、10 点以上崩れる（とがった 1 点）")
    if not mech_ok:
        why.append("仕組みの説明（公開の知見）が無い")
    return {"verdict": "根拠あり" if ok else "保留（採らない）", "reasons_failed": why, "event": ev, "reason_public": rs,
            "years_cv": folds, "years_b0": bfold, "years_ok": years_ok, "units": units, "unit_share": unit_share, "unit_sign_p": unit_p, "units_ok": units_ok,
            "diff_vs_B0": d, "roi_ex_top5": m.get("roi_ex_top5"), "b0_ex_top5": mb.get("roi_ex_top5"), "chance_ok": chance_ok,
            "neighbors": nb, "neighbors_ok": nb_ok, "mechanism_ok": mech_ok, "decompose_vs_B0": dec, "roi": m.get("roi"), "b0_roi": mb.get("roi"),
            "years_basis": ("選び方の交差検証の配列" if cv_arrays is not None else
                            ("方策の交差検証の年ごとの値（単位は標本内の配列）" if cv_folds else "標本内の配列"))}


def _spec_for_row(row: dict, inputs: dict, out: Path, cuts: dict, per: dict) -> tuple[dict, tuple | None, list | None]:
    """selcv の行（固定の格子・方策）から、訓練全体で選ぶ買い方の spec（pick-final と同じ手順）と、
    根拠の検査に使う交差検証の配列（固定の格子）・方策の隣（同じ型の作り方の別の設定）を返す。"""
    n = row["cycle"]
    rec = json.loads((out / f"cycle_{n:02d}.json").read_text(encoding="utf-8"))
    tr = per["train"]
    if row["kind"].startswith("方策"):
        name = row["picks"][""]
        sp = (rec.get("policy_specs") or {}).get(name) or next((t.get("spec") for t in rec.get("top", []) if t["name"] == name), None)
        if sp is None and (rec.get("selected") or {}).get("name") == name:
            sp = rec["selected"].get("spec")
        if sp is None:
            raise SystemExit(f"方策 {name} の中身が見つかりません")
        stem = name.split("|min_cell=")[0]
        nbs = [{"change": x["name"].split("|", 2)[-1], "roi": x["train"].get("roi"), "roi_ex_top5": x["train"].get("roi_ex_top5"),
                "bet_races": x["train"].get("bet_races")} for x in rec.get("all", []) if x["name"].startswith(stem + "|") and x["name"] != name]
        return sp, None, nbs
    tr_spec = inputs["periods"]["train"]
    cyc = json.loads((CYC_DIR / f"cycle_{n:02d}.json").read_text(encoding="utf-8"))
    strats = [x for x in resolve_auto(expand_variants(cyc), out) if not family(x).startswith("参考") and fav1_rule_ok(x)]
    key = strategy_hash({"pred": tr_spec["pred"], "years": tr_spec["years"], "cuts": cuts, "decision_odds": tr_spec.get("decision_odds"),
                         "pre": (_sha(PRE_ODDS) if tr_spec.get("decision_odds") and PRE_ODDS.exists() else None), "specs": [strategy_hash(x) for x in strats]})
    zp = out / f"cands_c{n:02d}_{key}.npz"
    if not zp.exists():
        raise SystemExit(f"{zp.name} がありません（同じ予測・区切り・買い方で selcv をやり直す）")
    z = np.load(zp, allow_pickle=True)
    if list(z["names"]) != [x["name"] for x in strats]:
        raise SystemExit("候補の配列と設定が合いません（selcv をやり直す）")
    base = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"][0]
    b0 = run_period(tr, base)
    j = _fold_pick(z["cost"], z["ret"], np.ones(len(tr.races), bool), (b0["cost"], b0["ret"]), row["min_hit"], MIN_BET_RACES)
    if j < 0:
        raise SystemExit("訓練全体で条件を満たす候補がありません")
    cc, cr, _ = selection_cv(z["cost"], z["ret"], tr.feats["year"].to_numpy(), (b0["cost"], b0["ret"]), row["min_hit"])
    return strats[j], (cc, cr), None


def _pick_with_evidence(rows: list[dict], inputs: dict, out: Path, cuts: dict) -> tuple[dict, list]:
    """順位の上から、根拠の検査（evidence_check）を通る最初の行を選ぶ。通らなかった行と理由を記録する。"""
    per, _ = prepare_periods(inputs, ("train",), cuts)
    base = json.loads((CFG_DIR / "baselines.json").read_text(encoding="utf-8"))["baselines"][0]
    log = []
    for row in rows[:8]:
        try:
            sp, cva, pnb = _spec_for_row(row, inputs, out, cuts, per)
            sp = resolve_auto([sp], out)[0]
        except SystemExit as e:
            log.append({"cycle": row["cycle"], "kind": row["kind"], "verdict": f"保留（{e}）"})
            continue
        ev = evidence_check(sp, per, base, cv_arrays=cva, policy_neighbors=pnb, cv_folds=(row.get("cv_folds") if pnb is not None else None))
        log.append({"cycle": row["cycle"], "kind": row["kind"], "name": sp.get("name"), **{k: ev[k] for k in ("verdict", "reasons_failed", "roi", "unit_share")}})
        print(f"根拠の検査: cycle {row['cycle']} {row['kind']} → {ev['verdict']} {ev['reasons_failed']}", flush=True)
        if ev["verdict"] == "根拠あり":
            (out / "final_candidate_evidence.json").write_text(json.dumps({"row": row, "spec": sp, "evidence": ev, "log": log}, ensure_ascii=False, indent=1, default=_np),
                                                              encoding="utf-8")
            return row, log
    (out / "final_candidate_evidence.json").write_text(json.dumps({"row": None, "log": log}, ensure_ascii=False, indent=1, default=_np), encoding="utf-8")
    raise SystemExit("根拠の検査を通る候補がありません（上位 8 つ）。凍結しない（統括役に返す）")


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
    f1.add_argument("--by", default="p2", choices=["p2", "r"], help="例外のしきい値に使う値（p2 = 予測の連対率、r = 市場とのずれ）")
    f1.set_defaults(func=cmd_fav1)
    sc = sub.add_parser("selcv", help="選び方そのものの交差検証と、全候補の配列の保存")
    sc.add_argument("--cycles", required=True)
    sc.add_argument("--inputs", default=None)
    sc.add_argument("--threads", type=int, default=3)
    sc.set_defaults(func=cmd_selcv)
    cp = sub.add_parser("check-periods", help="test・2026 の入口を、結果を見ずに確かめる（件数だけ）")
    cp.add_argument("--periods", default="test,fwd2026,fwd2026_pre")
    cp.add_argument("--inputs", default=None)
    cp.set_defaults(func=cmd_check_periods)
    ti = sub.add_parser("tiers", help="得意・不得意の地図と推奨度の段（凍結の候補の買い方で）")
    ti.add_argument("--spec", required=True, help="買い方の JSON（cycle_XX.json の selected.spec を書き出したもの）")
    ti.add_argument("--tag", default="")
    ti.add_argument("--axes", default="", help="段の点に使う軸（カンマ区切り。既定は全 9 軸）")
    ti.add_argument("--no-eval", action="store_true", help="評価の期間の段を出さない（段の作り方を訓練だけで比べるとき）")
    ti.add_argument("--inputs", default=None)
    ti.set_defaults(func=cmd_tiers)
    tsx = sub.add_parser("tiers-select", help="推奨度の段の作り方を訓練だけで選び、その 1 つで段を作る")
    tsx.add_argument("--spec", required=True)
    tsx.add_argument("--inputs", default=None)
    tsx.set_defaults(func=cmd_tiers_select)
    pf = sub.add_parser("pick-final", help="selcv の結果から、決めた決まりで最後の候補を選ぶ")
    pf.add_argument("--inputs", default=None)
    pf.add_argument("--evidence", action="store_true", help="根拠の検査（年・似た単位・偶然・隣の水準）を通る中で一番を選ぶ")
    pf.set_defaults(func=cmd_pick_final)
    fz = sub.add_parser("freeze", help="最後の候補を凍結する（config/bet_cycles/frozen_<日付>_<tag>.json）")
    fz.add_argument("--inputs", default=None)
    fz.add_argument("--out", default=None)
    fz.add_argument("--note", default="", help="凍結ファイルに残す注意（例: 評価・test が手つかずでない事情）")
    fz.set_defaults(func=cmd_freeze)
    pg = sub.add_parser("prereg-run", help="事前登録の 1 つの群を、凍結した買い方で 1 回だけ流す")
    pg.add_argument("--frozen", required=True)
    pg.add_argument("--prereg", required=True)
    pg.add_argument("--group", required=True, help="事前登録の群の id（main・separate・reference）")
    pg.add_argument("--pred", required=True, help='予測の指定（JSON）。例: {"path": "...parquet", "where": {"part": "infer2026"}}')
    pg.add_argument("--allow-low-cover", action="store_true", help="予測の覆いが下限（95%）を下回っても流す（理由を記録に書く）")
    pg.add_argument("--check-only", action="store_true", help="2026 年を読まずに、事前の確かめ（commit・入力・test 2025 の再現）だけ行う")
    pg.set_defaults(func=cmd_prereg_run)
    fn = sub.add_parser("final", help="凍結した買い方で test 2025 と 2026 年の参考を 1 回だけ見る")
    fn.add_argument("--frozen", required=True)
    fn.add_argument("--group", default="", choices=["", "test", "fwd"], help="test と 2026 年の参考を別の時点で 1 回ずつ見る（印も別）")
    fn.add_argument("--round", type=int, default=1, choices=[1, 2], help="何回目の最適化か（2 なら、2025 年 test は 2 回目の確かめとして別の印で 1 回だけ）")
    fn.set_defaults(func=cmd_final)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
