"""重賞のルールベース予想家: 消しの版（好成績の点で ◎○ を選び、極端に成績の悪いルールに当てはまる馬を消して相手を絞る）。

- 消し: 馬 × 型（前日版の点に使う型 = 束ねの代表）の as-of の件数で、点 x が負、かつ版の層（主／主＋副／全部）のどれかで
  走数 ≥ n の下限・異なるレースの数 ≥ 下限（主は 3 回、副・全重賞は max(3, n の下限 / 2)）、かつ基準（3着内 0／3着内率の上側 95% が期待の半分以下）を満たす。
  回収率版で裏付けた「避ける」パターン（prereg_roi.json の主に見る版）を足す軸もある（層は主・副、x<0。export.roi_flag と同じ条件）。
  消しに当たる型の数が版の下限（1・2）以上の馬を「消し」とする。
- 印: 前日版の点の順（点 → 主の好成績パターンの数 → 馬番）。◎○ の扱いは版の軸（そのまま／消しは ◎○ にしない）。
  相手は ◎○ 以外のうち消しでない馬を点の順に並べ、足りなければ消した馬を点の順で補う。
- 買い方: ◎ の単勝・複勝、◎→○▲ の馬連・ワイド（2 点）、◎→消しでない馬すべての馬連・ワイド（点数は可変）、◎→相手 3・4・5 頭の馬連・ワイド。
  購入額は点数 × 100 円。0 点のレースは購入額 0。
主に見る版は払戻を使わず、2018〜2024 年の消しの精度で選ぶ（choose_primary）。オッズ・人気は切り口に使わない。
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import beta

from src.research.graded import model as M

CRITS = ("3着内0", "上限が期待の半分以下")
NMINS = (10, 20, 40)
LAYERS = {"主だけ": ("main",), "主＋副": ("main", "sub"), "全部": ("main", "sub", "glob")}
HITS = (1, 2)
ROI7 = ("入れない", "入れる")
HONMEI = ("そのまま", "消しは◎○にしない")
UPPER_Q, UPPER_RATIO = 0.95, 0.5
# 主に見る版の選び方（払戻を使わない）: 発見 2018〜2021・確認 2022〜2024 の両方で、消した馬の 3 着内率 ≤ 期待 × PRECISION_RATIO、
# かつ各期間で消した馬が MIN_KILLED 頭以上。そのうち 2 期間を合わせた 1 レースあたりの消しの数が最大の版。同数は n の下限が大きい → 層が狭い →
# 基準が 3着内0 → 当たる数の下限が大きい → 回収率の避けるを入れない の順。◎○ は「そのまま」（◎○ は好成績で選ぶ、が依頼の形）
PRECISION_RATIO, MIN_KILLED = 0.5, 100
PRIMARY_HONMEI = "そのまま"
PARTNERS = (3, 4, 5)
BETS = ["tansho", "fukusho", "umaren", "wide", "umaren_all", "wide_all"] + [f"{b}_{k}" for k in PARTNERS for b in ("umaren", "wide")]


def version_name(p: dict) -> str:
    if p.get("only_roi"):
        return f"消し版[回収率の避けるだけ・{p['honmei']}]"
    return f"消し版[{p['crit']}・{p['nmin']}走以上・{p['layer']}・{p['hits']}つ以上・回収率の避けるを{p['roi7']}・{p['honmei']}]"


def constants() -> dict:
    """消しの判定・選択に効く定数（prereg_kill.json に凍結し、評価の前に今のコードと同じか確かめる）。"""
    return {"UPPER_Q": UPPER_Q, "UPPER_RATIO": UPPER_RATIO, "ED_MIN": {"main": 3, "other": "max(3, ceil(nmin/2))"},
            "ed_min_check": [ed_min(ly, n) for ly in ("main", "sub") for n in NMINS], "SUB_FALLBACK_N": M.SUB_FALLBACK_N,
            "PRECISION_RATIO": PRECISION_RATIO, "MIN_KILLED": MIN_KILLED, "PRIMARY_HONMEI": PRIMARY_HONMEI, "PARTNERS": list(PARTNERS),
            "MARKS": list(M.MARKS)}


def all_versions() -> list[dict]:
    vs = [dict(crit=c, nmin=n, layer=ly, hits=h, roi7=r, honmei=ho)
          for c, n, ly, h, r, ho in itertools.product(CRITS, NMINS, LAYERS, HITS, ROI7, HONMEI)]
    vs += [dict(only_roi=True, hits=1, honmei=ho) for ho in HONMEI]
    return vs


def kill_key(p: dict) -> tuple:
    """消しの集合を決める軸だけ（◎○ の扱いは消しの集合に効かない）。"""
    if p.get("only_roi"):
        return ("only_roi",)
    return (p["crit"], p["nmin"], p["layer"], p["hits"], p["roi7"])


def _layer_counts(S: pd.DataFrame, ly: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """層の (走数, 3 着内の数, 期待 Σ3/頭数, 異なるレースの数)。副は shrink と同じく全体が少なければ 3 勝クラス以上（sub4）。"""
    if ly == "sub":
        use4 = (S["all_sub_n"] < M.SUB_FALLBACK_N).to_numpy()
        g = lambda c: np.where(use4, S[f"sub4_{c}"].to_numpy(float), S[f"sub_{c}"].to_numpy(float))   # noqa: E731
    else:
        g = lambda c: S[f"{ly}_{c}"].to_numpy(float)   # noqa: E731
    return g("n"), g("w1") + g("w2") + g("w3"), g("e3"), g("ed")


def ed_min(ly: str, nmin: int) -> int:
    """異なるレースの数の下限。主は同じレース名の過去の回で、2016 年からの集計では 2024 年でも 8 回までしかないので 3 回。
    副・全重賞は 1 レースに同じ水準の馬が固まるのを避けて max(3, n の下限 / 2)。"""
    return 3 if ly == "main" else max(3, math.ceil(nmin / 2))


def stat_hits(S: pd.DataFrame, crit: str, nmin: int, layer: str) -> np.ndarray:
    """S の各行（馬 × 型）が統計の消しに当たるか。x<0 の行だけ（NO_INFO・主の share の上限で 0 にした水準は外れる）。"""
    hit = np.zeros(len(S), bool)
    for ly in LAYERS[layer]:
        n, k, e3, ed = _layer_counts(S, ly)
        ok = (n >= nmin) & (ed >= ed_min(ly, nmin))
        if crit == "3着内0":
            ok &= k == 0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                up = beta.ppf(UPPER_Q, k + 0.5, np.maximum(n - k, 0) + 0.5)
                ok &= up <= UPPER_RATIO * e3 / n
        hit |= ok
    return hit & (S["x"].to_numpy() < 0)


def roi_hits(S: pd.DataFrame, prim_roi: dict) -> np.ndarray:
    """回収率版で裏付けた避けるパターン（型 × 水準 × 凡）に、その版の層（主＋副など）で x<0 として当たる行（export.roi_flag と同じ条件）。"""
    from src.research.graded.roi import LAYERS as ROI_LAYERS
    pats = {(t, lv) for t, lv, dr in prim_roi["patterns"] if dr == "凡" and t in prim_roi["used"]}
    key = pd.Series(list(zip(S["type"], S["level"].astype(str))), index=S.index)
    lay = ROI_LAYERS[prim_roi["params"]["layer"]]
    return key.isin(pats).to_numpy() & S["layer"].isin(lay).to_numpy() & (S["x"].to_numpy() < 0)


def killed_rows(S_kill: pd.DataFrame, S_roi: pd.DataFrame, p: dict, prim_roi: dict) -> pd.Series:
    """行（馬）→ 消しに当たる型の数。S_kill は前日版の使う型の行、S_roi は回収率版の型の行。"""
    parts = []
    if not p.get("only_roi"):
        h = stat_hits(S_kill, p["crit"], p["nmin"], p["layer"])
        parts.append(S_kill.loc[h, ["row", "type"]])
    if p.get("only_roi") or p.get("roi7") == "入れる":
        parts.append(S_roi.loc[roi_hits(S_roi, prim_roi), ["row", "type"]])
    if not parts:
        return pd.Series(dtype=int)
    x = pd.concat(parts).drop_duplicates()
    return x.groupby("row")["type"].nunique()


def order_marks(base: pd.DataFrame, killed: set, honmei: str) -> pd.DataFrame:
    """base: 前日版の印（race_id, row, hn, rank, y1, y3, field）。消しを反映した並び（rank）と、消しの印（kill）。"""
    b = base.sort_values(["race_id", "rank"]).copy()
    b["kill"] = b["row"].isin(killed)
    if honmei == "そのまま":
        top = b["rank"] <= 2
        key = np.where(top, 0, np.where(b["kill"], 2, 1))
    else:
        key = np.where(b["kill"], 1, 0)
    b["_k"] = key
    b = b.sort_values(["race_id", "_k", "rank"])
    b["rank"] = b.groupby("race_id").cumcount() + 1
    if "mark" in b.columns:   # 並べ替えた後の順位で印を付け直す
        b["mark"] = b["rank"].map(lambda k: M.MARKS[k - 1] if k <= len(M.MARKS) else "")
    return b.drop(columns="_k")


def race_metrics_kill(mk: pd.DataFrame, pay: dict) -> pd.DataFrame:
    """消しの版の、レースごとの指標・購入額・払戻（4 券種の払戻がそろうレースだけ。run.race_metrics と同じ定義のレース集合）。"""
    rows = []
    for rid, g in mk.groupby("race_id", sort=False):
        if not all((rid, bt) in pay for bt in ("tansho", "fukusho", "umaren", "wide")):
            continue
        g = g.sort_values("rank")
        hn = g["hn"].astype(int).tolist()
        kill = g["kill"].to_numpy(bool)
        h1 = hn[0]
        r = {"race_id": rid, "win": float(g["y1"].iloc[0]), "place": float(g["y3"].iloc[0]), "top3_in3": float(g["y3"].iloc[:3].sum())}
        r["ret_tansho"], r["cost_tansho"] = pay[(rid, "tansho")].get(str(h1), 0), 100
        r["ret_fukusho"], r["cost_fukusho"] = pay[(rid, "fukusho")].get(str(h1), 0), 100
        sets = {"": hn[1:3], "_all": [h for h, k in zip(hn[1:], kill[1:]) if not k]}
        sets |= {f"_{k}": hn[1:1 + k] for k in PARTNERS}
        for suf, part in sets.items():
            for bt in ("umaren", "wide"):
                r[f"ret_{bt}{suf}"] = sum(pay[(rid, bt)].get(f"{min(h1, h)}-{max(h1, h)}", 0) for h in part)
                r[f"cost_{bt}{suf}"] = 100 * len(part)
        e3 = (3.0 / pd.to_numeric(g["field"], errors="coerce")).clip(upper=1).to_numpy()
        y3 = g["y3"].to_numpy(float)
        r |= {"n_horses": len(g), "n_killed": int(kill.sum()), "n_left": int((~kill).sum()),
              "killed_top3": float(y3[kill].sum()), "killed_e3": float(e3[kill].sum()), "killed_win": float(g["y1"].to_numpy(float)[kill].sum()),
              "kill_miss_race": float((y3[kill] > 0).any()), "honmei_killed": float(kill[0])}
        rows.append(r)
    return pd.DataFrame(rows)


def kill_stat(rt: pd.DataFrame) -> dict:
    """消しの誤り: 馬単位（消した馬の 3 着内率・期待・1 着の数）とレース単位（消した馬が 3 着内に来たレースの割合）、頭数の平均。"""
    nk = rt["n_killed"].sum()
    return {"races": int(len(rt)), "killed_per_race": float(rt["n_killed"].mean()), "left_per_race": float(rt["n_left"].mean()),
            "horses_per_race": float(rt["n_horses"].mean()), "killed": int(nk),
            "killed_top3_rate": float(rt["killed_top3"].sum() / nk) if nk else None,
            "killed_e3_rate": float(rt["killed_e3"].sum() / nk) if nk else None,
            "killed_win_rate": float(rt["killed_win"].sum() / nk) if nk else None,
            "race_miss_rate": float(rt["kill_miss_race"].mean()), "honmei_killed_rate": float(rt["honmei_killed"].mean())}


def summarize_bets(rt: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """的中率と、券種ごとの回収率（払戻の和 ÷ 購入額の和）の点とレース単位のブートストラップ 95% 区間。"""
    cols = ["win", "place", "top3_in3"] + [f"ret_{b}" for b in BETS] + [f"cost_{b}" for b in BETS]
    A = rt[cols].to_numpy(float)
    W = np.random.default_rng(seed).poisson(1.0, size=(n_boot, len(rt))).astype(float)
    S = W @ A
    nb = W.sum(1)
    out = {"races": int(len(rt))}
    for j, k in enumerate(["win", "place", "top3_in3"]):
        with np.errstate(divide="ignore", invalid="ignore"):
            b = S[:, j] / nb
        b = b[np.isfinite(b)]
        out[k] = {"est": float(A[:, j].mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975))}
    for i, bt in enumerate(BETS):
        jr, jc = 3 + i, 3 + len(BETS) + i
        c = A[:, jc].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            b = S[:, jr] / S[:, jc]
        b = b[np.isfinite(b)]
        out[f"roi_{bt}"] = {"est": float(A[:, jr].sum() / c) if c else None, "lo": float(np.quantile(b, 0.025)) if len(b) else None,
                            "hi": float(np.quantile(b, 0.975)) if len(b) else None, "cost": float(c)}
    return out


def diff_bets(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 1) -> dict:
    """同じレースでの差（a − b）。的中率は平均の差、回収率は比の差。b に無い券種は a の値だけで比べない（出さない）。"""
    m = a.merge(b, on="race_id", suffixes=("_a", "_b"))
    if len(m) == 0:
        return {"races": 0}
    W = np.random.default_rng(seed).poisson(1.0, size=(n_boot, len(m))).astype(float)
    nb = W.sum(1)
    out = {"races": int(len(m))}
    for k in ("win", "place", "top3_in3"):
        d = (m[f"{k}_a"] - m[f"{k}_b"]).to_numpy(float)
        bs = W @ d / nb
        out[k] = {"est": float(d.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}
    for bt in BETS:
        if f"ret_{bt}_b" not in m.columns:
            continue
        ra, ca, rb, cb = (m[f"{c}_{bt}_{s}"].to_numpy(float) for s, c in (("a", "ret"), ("a", "cost"), ("b", "ret"), ("b", "cost")))
        if ca.sum() == 0 or cb.sum() == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            bs = W @ ra / (W @ ca) - W @ rb / (W @ cb)
        bs = bs[np.isfinite(bs)]
        out[f"roi_{bt}"] = {"est": float(ra.sum() / ca.sum() - rb.sum() / cb.sum()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}
    return out


def precision_by_period(killed: pd.Series, base: pd.DataFrame, spans: dict) -> dict:
    """払戻を使わない消しの精度（期間ごと）: 消した馬の数・3 着内率・期待（Σ3/頭数 の平均）・1 レースあたりの消しの数。"""
    b = base.copy()
    b["kill"] = b["row"].isin(set(killed.index))
    b["e3"] = (3.0 / pd.to_numeric(b["field"], errors="coerce")).clip(upper=1)
    out = {}
    for lab, (y0, y1) in spans.items():
        x = b[(b["year"] >= y0) & (b["year"] <= y1)]
        k = x[x["kill"]]
        out[lab] = {"races": int(x["race_id"].nunique()), "killed": int(len(k)), "killed_per_race": float(len(k) / max(x["race_id"].nunique(), 1)),
                    "top3_rate": float(k["y3"].mean()) if len(k) else None, "e3_rate": float(k["e3"].mean()) if len(k) else None,
                    "win_rate": float(k["y1"].mean()) if len(k) else None,
                    "race_miss_rate": float(k.groupby("race_id")["y3"].max().reindex(x["race_id"].unique()).fillna(0).mean())}
    return out


def passes(prec: dict) -> bool:
    return all(v["killed"] >= MIN_KILLED and v["top3_rate"] is not None and v["top3_rate"] <= PRECISION_RATIO * v["e3_rate"]
               for v in prec.values())


def choose_primary(vers: list[dict]) -> str:
    """主に見る版の名前（払戻を使わない。上の定数の説明のとおり）。どれも通らなければ、確認期間の 3 着内率 ÷ 期待が最小の版。"""
    cand = [v for v in vers if v["params"].get("honmei") == PRIMARY_HONMEI and not v["params"].get("only_roi")]
    ok = [v for v in cand if v["passes"]]
    ly_rank = {k: i for i, k in enumerate(LAYERS)}
    if ok:
        def key(v):
            p, pr = v["params"], v["precision"]
            kr = (pr["disc"]["killed"] + pr["conf"]["killed"]) / max(pr["disc"]["races"] + pr["conf"]["races"], 1)
            return (-kr, -p["nmin"], ly_rank[p["layer"]], CRITS.index(p["crit"]), -p["hits"], ROI7.index(p["roi7"]))
        return sorted(ok, key=key)[0]["name"]
    return min(cand, key=lambda v: (v["precision"]["conf"]["top3_rate"] or 9) / (v["precision"]["conf"]["e3_rate"] or 1))["name"]
