"""めぐコラム『血統で狙う』（sire-roi）を実データから集計し、frontend/content/columns/sire-roi.json を書く。

問い: 特定の種牡馬の産駒は、馬券で高い回収率を維持しているか。どの条件で高いか。

データ:
  レース: compare_series.load()（2016〜2025 年の JRA 平地、取消・除外を除く）。払戻は同じ race_result_flat の payoff（単勝・複勝）。
  父:     compare_series.SIRE_MAP_PATH（horse_sire_map.parquet）。対象の出走の 100% に父が付く。
集計:
  種牡馬ごとの全条件の成績: 着別度数（1着-2着-3着-着外）・勝率・複勝率・単勝回収率・複勝回収率（払戻から。複勝の発売がないレースは複勝回収率から除く）。
  回収率の 95% 区間は、出走を復元抽出するブートストラップ（パーセンタイル法）。
  大穴の外れ値の影響を見るため、単勝オッズ 50 倍以下の出走に絞った単勝回収率も並べる。
載せる種牡馬（N の根拠）:
  単勝回収率の 1 走あたりの標準偏差は約 590%（大半が 0、たまに数十倍）。95% 区間の幅は ±1.96×5.9/√n で、
  n=300 なら ±67pt、n=1,000 なら ±37pt。全体の平均（約 72〜80%）と 100% を見分けるには最低でも ±40pt 以内が要るので、
  産駒の出走 MIN_RUNS（1,000）以上を本表に載せ、300〜999 走は参考の表に回す。
条件（各種牡馬の折り畳みの中）:
  芝ダ×距離帯・芝ダ×競馬場・クラス・芝ダ×馬場・年齢・性別。条件の出走が MIN_COND（200）以上のものだけ。
  単勝回収率の上位 5 件と、100% 超えの全件を載せる。
  多数の条件から上位を選ぶと、偶然高かった条件が選ばれる（選択バイアス）。前半（2016〜2020 年）で 100% を超えた条件が、
  後半（2021〜2025 年）でも 100% を超えるかを全体で確かめ、各条件にも前半・後半の回収率を併記する。

使い方:
    python -m src.research.columns.sire_roi            # 集計して記事を書く
    python -m src.research.columns.sire_roi --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import FLAT_DIR, HALF_SPLIT, SIRE_MAP_PATH, YEARS, load
from src.research.columns.early_maturing_bloodline import dumps, short_name
from src.research.columns.wind_venue import pct

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/sire-roi.json"
MIN_RUNS = 1000          # 本表に載せる種牡馬の産駒の出走数
MIN_REF = 300            # 参考の表に載せる下限
MIN_COND = 200           # 条件の出走数の下限
MIN_HALF = 100           # 前半・後半の検証で使う、各期間の出走数の下限
LONGSHOT = 50            # 「単勝 50 倍以下」
SPLIT_YEAR = HALF_SPLIT   # 前半 = 2016〜2020 年、後半 = 2021 年〜（5 年ずつ。compare_series.HALF_SPLIT と共通）
B = 1000                 # ブートストラップの回数
SEED = 20261102
DIST = {"短距離": (0, 1400), "マイル": (1401, 1800), "中距離": (1801, 2200), "長距離": (2201, 9999)}
CLASS = {"新馬": ["新馬"], "未勝利": ["未勝利"], "1勝クラス": ["1勝"], "2勝クラス": ["2勝"], "3勝クラス": ["3勝"], "OP・重賞": ["OP", "L", "G3", "G2", "G1"]}


def _payouts(payoff: Any, key: str) -> dict[int, float]:
    """払戻の JSON から、馬番 → 100 円あたりの払戻 / 100。同着で複数のときは list。"""
    j = json.loads(payoff) if isinstance(payoff, str) else (payoff or {})
    v = j.get(key)
    if v is None:
        return {}
    out = {}
    for e in v if isinstance(v, list) else [v]:
        try:
            out[int(str(e["numbers"]).strip())] = float(str(e["payout"]).replace(",", "")) / 100
        except (KeyError, ValueError):
            continue
    return out


def frame() -> pd.DataFrame:
    df = load()
    cols = ["race_id", "horse_id", "horse_number", "track_condition", "payoff"]
    r = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    r["horse_id"] = r["horse_id"].astype(str)
    pay = r.drop_duplicates("race_id").set_index("race_id")["payoff"]
    win = {k: _payouts(v, "単勝") for k, v in pay.items()}
    plc = {k: _payouts(v, "複勝") for k, v in pay.items()}
    df = df.merge(r[["race_id", "horse_id", "horse_number", "track_condition"]].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    hn = df["horse_number"].astype(int)
    df["ret_win"] = [win[k].get(n, 0.0) for k, n in zip(df["race_id"], hn)]
    # 複勝の発売がないレース（4 頭立て以下など）は複勝回収率から除く
    df["ret_place"] = [plc[k].get(n, 0.0) if plc[k] else np.nan for k, n in zip(df["race_id"], hn)]
    m = pd.read_parquet(SIRE_MAP_PATH, columns=["horse_id", "sire_id", "sire_name"]).dropna(subset=["sire_id"]).drop_duplicates("horse_id")
    m["horse_id"] = m["horse_id"].astype(str)
    df = df.merge(m, on="horse_id", how="inner")
    df["sire"] = df["sire_name"].map(short_name)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["half"] = np.where(df["year"] <= SPLIT_YEAR, "前半", "後半")
    sf = df["surface"].map({"芝": "芝", "ダート": "ダ"})
    db = pd.cut(df["distance"], [0, 1400, 1800, 2200, 9999], labels=list(DIST)).astype(str)
    df["c_dist"] = sf + " " + db
    df["c_venue"] = df["venue"] + " " + sf
    cls = {g: k for k, gs in CLASS.items() for g in gs}
    df["c_class"] = df["grade"].map(cls)
    df["c_going"] = sf + " " + np.where(df["track_condition"] == "良", "良", "稍重〜不良")
    df["c_age"] = np.where(df["age"] >= 4, "4歳以上", df["age"].astype("Int64").astype(str) + "歳")
    df["c_sex"] = np.where(df["sex"] == "牝", "牝", "牡・セン")
    df["short"] = df["odds"] <= LONGSHOT
    return df


COND_COLS = {"c_dist": "芝ダ×距離", "c_venue": "競馬場", "c_class": "クラス", "c_going": "馬場", "c_age": "年齢", "c_sex": "性別"}


def boot_ci(v: np.ndarray, rng: np.random.Generator, b: int = B) -> tuple[float, float] | None:
    """平均のブートストラップ 95% 区間（パーセンタイル法）。"""
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return None
    means = np.empty(b)
    step = max(1, 2_000_000 // n)            # メモリを抑えるため、まとめて引く回数を区切る
    for s in range(0, b, step):
        k = min(step, b - s)
        means[s:s + k] = v[rng.integers(0, n, (k, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(x: pd.DataFrame, rng: np.random.Generator | None = None) -> dict[str, Any]:
    fp = x["finish_position"]
    out = {"n": len(x), "k1": int((fp == 1).sum()), "k2": int((fp == 2).sum()), "k3": int((fp == 3).sum()),
           "win": float(x["win"].mean()), "top3": float(x["top3"].mean()),
           "roi": float(x["ret_win"].mean()), "proi": float(x["ret_place"].mean()),
           "n50": int(x["short"].sum()), "roi50": float(x.loc[x["short"], "ret_win"].mean()) if x["short"].any() else float("nan")}
    out["k4"] = out["n"] - out["k1"] - out["k2"] - out["k3"]
    h = x.groupby("half")["ret_win"].agg(["mean", "size"])
    out["roi_a"] = float(h.loc["前半", "mean"]) if "前半" in h.index else float("nan")
    out["roi_b"] = float(h.loc["後半", "mean"]) if "後半" in h.index else float("nan")
    out["n_a"] = int(h.loc["前半", "size"]) if "前半" in h.index else 0
    out["n_b"] = int(h.loc["後半", "size"]) if "後半" in h.index else 0
    if rng is not None:
        out["roi_ci"] = boot_ci(x["ret_win"].to_numpy(float), rng)
        out["proi_ci"] = boot_ci(x["ret_place"].to_numpy(float), rng)
        out["roi50_ci"] = boot_ci(x.loc[x["short"], "ret_win"].to_numpy(float), rng)
    return out


def conditions(x: pd.DataFrame) -> pd.DataFrame:
    """その種牡馬の産駒の、条件ごとの成績（出走 MIN_COND 以上）。"""
    rows = []
    for col, kind in COND_COLS.items():
        for lv, g in x.groupby(col):
            if len(g) >= MIN_COND:
                rows.append({"kind": kind, "cond": lv, **summarize(g)})
    return pd.DataFrame(rows)


def validation(df: pd.DataFrame, sires: list[str]) -> dict[str, Any]:
    """前半で単勝回収率 100% 超えの条件は、後半でも 100% を超えるか（全種牡馬・全条件をまとめて）。"""
    d = df[df["sire"].isin(sires)]
    recs = []
    for col in COND_COLS:
        g = d.groupby(["sire", col, "half"])["ret_win"].agg(["sum", "size"]).unstack("half")
        g = g.dropna()
        g = g[(g[("size", "前半")] >= MIN_HALF) & (g[("size", "後半")] >= MIN_HALF)]
        recs.append(pd.DataFrame({"a": g[("sum", "前半")] / g[("size", "前半")], "b": g[("sum", "後半")] / g[("size", "後半")],
                                  "sb": g[("sum", "後半")], "nb": g[("size", "後半")]}))
    v = pd.concat(recs)
    sel = v[v["a"] > 1.0]
    rest = v[v["a"] <= 1.0]
    out = {"cells": len(v), "sel": len(sel), "sel_a_mean": float(sel["a"].mean()) if len(sel) else float("nan"),
           "sel_b": float(sel["sb"].sum() / sel["nb"].sum()) if len(sel) else float("nan"),
           "sel_b_over": float((sel["b"] > 1.0).mean()) if len(sel) else float("nan"),
           "rest_b": float(rest["sb"].sum() / rest["nb"].sum()), "rest_b_over": float((rest["b"] > 1.0).mean()),
           "all_b": float(v["sb"].sum() / v["nb"].sum()), "corr": float(np.corrcoef(v["a"], v["b"])[0, 1])}
    return out


def fmt_ci(c: tuple[float, float] | None) -> str:
    return "–" if c is None else f"{c[0] * 100:.0f}〜{c[1] * 100:.0f}%"


def roi_cell(v: float, c: tuple[float, float] | None) -> str:
    return f"{v * 100:.0f}%（{fmt_ci(c)}）"


def rec(s: dict[str, Any]) -> str:
    return f"{s['k1']}-{s['k2']}-{s['k3']}-{s['k4']}"


def r0(v: float) -> str:
    return "–" if v is None or not math.isfinite(v) else f"{v * 100:.0f}%"


def analyze(df: pd.DataFrame) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    a: dict[str, Any] = {"n_runs": len(df), "n_horses": df["horse_id"].nunique(), "n_sires": df["sire"].nunique()}
    a["all"] = summarize(df)
    a["sd_win"] = float(df["ret_win"].std())
    a["sd_win50"] = float(df.loc[df["short"], "ret_win"].std())
    runs = df.groupby("sire").size().sort_values(ascending=False)
    major = [s for s in runs.index if runs[s] >= MIN_RUNS]
    ref = [s for s in runs.index if MIN_REF <= runs[s] < MIN_RUNS]
    a["major"], a["ref"] = [], []
    for s in major:
        x = df[df["sire"] == s]
        st = summarize(x, rng)
        c = conditions(x)
        c = c.sort_values("roi", ascending=False)
        top = c.head(5)
        over = c[c["roi"] > 1.0]
        shown = pd.concat([top, over]).drop_duplicates(["kind", "cond"]).sort_values("roi", ascending=False)
        ci = []
        for _, r in shown.iterrows():
            g = x[x[{v: k for k, v in COND_COLS.items()}[r["kind"]]] == r["cond"]]
            ci.append(boot_ci(g["ret_win"].to_numpy(float), rng))
        shown = shown.assign(roi_ci=ci)
        a["major"].append({"sire": s, **st, "n_cond": len(c), "n_over": len(over), "shown": shown})
    for s in ref:
        a["ref"].append({"sire": s, **summarize(df[df["sire"] == s], rng)})
    a["val"] = validation(df, major)
    m = pd.DataFrame([{k: v for k, v in r.items() if k != "shown"} for r in a["major"]])
    a["m"] = m
    a["n_over_ci"] = int(sum(1 for r in a["major"] if r["roi_ci"] and r["roi_ci"][0] > 1.0))
    a["n_over_point"] = int((m["roi"] > 1.0).sum())
    a["n_over_ci50"] = int(sum(1 for r in a["major"] if r["roi50_ci"] and r["roi50_ci"][0] > 1.0))
    a["n_over_pci"] = int(sum(1 for r in a["major"] if r["proi_ci"] and r["proi_ci"][0] > 1.0))
    return a


def _num(v: Any) -> float | None:
    """NaN は null に（片方の期間に出走の無い種牡馬。JSON に NaN を書かない）。"""
    return None if v is None or not math.isfinite(float(v)) else round(float(v), 4)


def roi_chart(m: pd.DataFrame, n_over_ci: int, k: int = 12) -> dict[str, Any]:
    """単勝回収率の高い種牡馬 k 頭の 95% 区間つきの回収率。100% の線（これを下回れば買い続けると損）を点線で添える。"""
    top = m.sort_values("roi", ascending=False).head(k)
    cis = [[round(max(0.0, r["roi_ci"][0]), 4), round(r["roi_ci"][1], 4)] if r["roi_ci"] else None for _, r in top.iterrows()]
    return {"type": "chart", "title": "単勝回収率の高い種牡馬（95%区間つき）", "xLabel": "種牡馬（産駒の単勝回収率が高い順）", "yLabel": "単勝回収率",
            "xs": [r["sire"] for _, r in top.iterrows()],
            "series": [{"name": "単勝回収率", "values": [round(float(v), 4) for v in top["roi"]], "n": [int(v) for v in top["n"]], "ci": cis},
                       {"name": "100%（買い続けて損得なし）", "values": [1.0] * len(top), "baseline": True}],
            "caption": (f"産駒 {MIN_RUNS:,} 走以上の種牡馬のうち、単勝回収率が高い {k} 頭。縦線は 95% 区間。"
                        + ("区間の下限が点線（100%）を超える種牡馬は見つかりません。" if n_over_ci == 0 else f"区間の下限が点線（100%）を超える種牡馬は {n_over_ci} 頭です。"))}


def halves_chart(m: pd.DataFrame, k: int = 10) -> dict[str, Any]:
    """前半（2016〜2020 年）で単勝回収率が高かった種牡馬 k 頭の、前半と後半の回収率。高さが続くかを見る。"""
    top = m.sort_values("roi_a", ascending=False).head(k)
    return {"type": "chart", "title": "前半で回収率が高かった種牡馬の、後半の回収率", "xLabel": "種牡馬（前半の単勝回収率が高い順）", "yLabel": "単勝回収率",
            "xs": [r["sire"] for _, r in top.iterrows()],
            "series": [{"name": f"前半（2016〜{SPLIT_YEAR}年）", "values": [_num(v) for v in top["roi_a"]]},
                       {"name": f"後半（{SPLIT_YEAR + 1}〜2025年）", "values": [_num(v) for v in top["roi_b"]]},
                       {"name": "100%", "values": [1.0] * len(top), "baseline": True}],
            "initial": [f"前半（2016〜{SPLIT_YEAR}年）", f"後半（{SPLIT_YEAR + 1}〜2025年）"],
            "caption": "前半に高かった種牡馬ほど、後半は下がる傾向があるかを見る図。後半の値がバラバラに散るほど、前半の高さは偶然に近い。"}


def article(a: dict[str, Any]) -> dict[str, Any]:
    m: pd.DataFrame = a["m"]
    A, V = a["all"], a["val"]
    best = m.sort_values("roi", ascending=False).iloc[0]
    best50 = m.sort_values("roi50", ascending=False).iloc[0]
    bestp = m.sort_values("proi", ascending=False).iloc[0]
    hi = m[m["roi"] >= 0.9]
    drop = hi.assign(dd=hi["roi"] - hi["roi50"]).sort_values("dd", ascending=False).head(3)
    drop = drop[drop["dd"] > 0.1]
    half = lambda n: 1.96 * a["sd_win"] / math.sqrt(n) * 100
    main_rows = [[r["sire"], f"{r['n']:,}", rec(r), pct(r["win"]), pct(r["top3"]), roi_cell(r["roi"], r["roi_ci"]),
                  f"{roi_cell(r['roi50'], r['roi50_ci'])}", roi_cell(r["proi"], r["proi_ci"]), f"{r0(r['roi_a'])} → {r0(r['roi_b'])}"]
                 for r in a["major"]]
    ref_rows = [[r["sire"], f"{r['n']:,}", rec(r), pct(r["win"]), pct(r["top3"]), roi_cell(r["roi"], r["roi_ci"]),
                 roi_cell(r["roi50"], r["roi50_ci"]), roi_cell(r["proi"], r["proi_ci"])] for r in a["ref"]]
    accs = []
    for r in a["major"]:
        sh: pd.DataFrame = r["shown"]
        rows = [[f"{c['kind']}: {c['cond']}", f"{c['n']:,}", rec(c), pct(c["win"]), pct(c["top3"]), roi_cell(c["roi"], c["roi_ci"]),
                 r0(c["roi50"]), r0(c["proi"]), f"{r0(c['roi_a'])} → {r0(c['roi_b'])}"] for _, c in sh.iterrows()]
        n_keep = int(((sh["roi"] > 1.0) & (sh["roi_a"] > 1.0) & (sh["roi_b"] > 1.0)).sum())
        blocks: list[dict[str, Any]] = [
            {"type": "table", "columns": ["条件", "出走", "着別度数", "勝率", "複勝率", "単勝回収率（95%区間）", f"単勝{LONGSHOT}倍以下", "複勝回収率", "単勝 前半 → 後半"],
             "rows": rows,
             "caption": (f"単勝回収率の上位 5 件と、100% 超えの全件（出走 {MIN_COND} 以上の {r['n_cond']} 条件から）。"
                         f"100% 超えは {r['n_over']} 件、そのうち前半（2016〜2020）・後半（2021〜2025）の両方で 100% を超えたのは {n_keep} 件。"
                         "95% 区間が 100% をまたぐ条件は、偶然の高さと区別できない。")},
        ]
        title = (f"{r['sire']}（{r['n']:,}走）単勝 {r0(r['roi'])}・{LONGSHOT}倍以下 {r0(r['roi50'])}・複勝 {r0(r['proi'])} ／ 100%超え {r['n_over']} 条件")
        accs.append({"type": "accordion", "title": title, "blocks": blocks})

    val_txt = (f"前半（2016〜{SPLIT_YEAR}年）と後半（{SPLIT_YEAR + 1}〜2025年）の両方に {MIN_HALF} 走以上ある「種牡馬 × 条件」{V['cells']:,} 件で確かめると、"
               f"前半に単勝回収率が 100% を超えた {V['sel']:,} 件（前半の平均 {V['sel_a_mean'] * 100:.0f}%）は、後半には {V['sel_b'] * 100:.0f}% に下がり、"
               f"後半も 100% を超えたのは {V['sel_b_over'] * 100:.0f}% でした。前半 100% 以下だった条件の後半は {V['rest_b'] * 100:.0f}%（100% 超えは {V['rest_b_over'] * 100:.0f}%）で、"
               + ("前半に高かった条件は、後半も他の条件よりはやや高いものの、100% には届かず、ほとんどは全体並みに戻っています。" if V["sel_b"] > V["rest_b"] + 0.03 else "前半に高かった条件も、後半は他の条件とほとんど変わりません。")
               + f"前半と後半の回収率の相関は r={V['corr']:+.2f} と弱く、「過去に回収率が高かった条件」が次の期間も 100% を超えるとは期待しにくい、というのがデータの答えです。")
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "全産駒の単勝回収率", "value": r0(A["roi"]), "sub": f"複勝 {r0(A['proi'])}。{a['n_runs']:,} 走"},
            {"label": "単勝回収率の 95% 区間が 100% を超える種牡馬", "value": f"{a['n_over_ci']} / {len(m)} 頭", "sub": f"産駒 {MIN_RUNS:,} 走以上。点推定で 100% 超えは {a['n_over_point']} 頭"},
            {"label": "前半 100% 超えの条件の後半の回収率", "value": f"{V['sel_b'] * 100:.0f}%", "sub": f"後半も 100% 超えは {V['sel_b_over'] * 100:.0f}%（{V['sel']:,} 件中）"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"{min(YEARS)}〜{max(YEARS)}年の JRA 平地で、産駒の出走が {MIN_RUNS:,} 以上の {len(m)} 種牡馬の全条件の成績を並べました（下の表）。"
            f"単勝回収率が最も高いのは {best['sire']}（{r0(best['roi'])}、95% 区間 {fmt_ci(best['roi_ci'])}）、"
            f"単勝 {LONGSHOT} 倍以下に絞ると {best50['sire']}（{r0(best50['roi50'])}）、複勝回収率は {bestp['sire']}（{r0(bestp['proi'])}）です。"
            f"ただし、単勝回収率の 95% 区間の下限が 100% を超える種牡馬は {a['n_over_ci']} 頭、{LONGSHOT} 倍以下に絞っても {a['n_over_ci50']} 頭、複勝回収率では {a['n_over_pci']} 頭で、"
            "「父だけで買い続けて儲かる」と言える種牡馬は見つかりませんでした。\n\n"
            "種牡馬ごとの折り畳みには、芝ダ×距離・競馬場・クラス・馬場・年齢・性別の条件で、単勝回収率の上位 5 件と 100% 超えの条件を載せています。"
            "ただし、多数の条件の中から高いものを選ぶと、偶然高かった条件が必ず混ざります。" + val_txt)},
        roi_chart(m, a["n_over_ci"]),
        halves_chart(m),
        {"type": "heading", "text": "種牡馬別 全条件の成績"},
        {"type": "table",
         "columns": ["父", "出走", "着別度数（1-2-3-着外）", "勝率", "複勝率", "単勝回収率（95%区間）", f"単勝{LONGSHOT}倍以下の回収率（95%区間）", "複勝回収率（95%区間）", "単勝 前半 → 後半"],
         "rows": main_rows,
         "caption": (f"{min(YEARS)}〜{max(YEARS)}年 JRA 平地。産駒の出走 {MIN_RUNS:,} 以上、出走の多い順。回収率は払戻の実績（100 円ずつ買ったときの払戻 ÷ 購入額）。"
                     f"95% 区間は出走を復元抽出するブートストラップ（{B:,} 回）。{LONGSHOT} 倍以下は、単勝オッズ {LONGSHOT} 倍以下の出走だけの単勝回収率（大穴の 1 発の影響を除く）。"
                     f"前半 = 2016〜{SPLIT_YEAR}年、後半 = {SPLIT_YEAR + 1}〜2025年。")},
        {"type": "note", "text": (
            f"なぜ {MIN_RUNS:,} 走以上か: 単勝回収率は、ほとんどの出走が 0 で、たまに数十倍の払戻がある、ばらつきの大きい値です（1 走あたりの標準偏差は約 {a['sd_win'] * 100:.0f}%）。"
            f"95% 区間の幅はおよそ ±{half(300):.0f}pt（300 走）・±{half(1000):.0f}pt（1,000 走）・±{half(5000):.0f}pt（5,000 走）で、"
            f"300 走では全体の平均（{r0(A['roi'])}）と 140% を見分けられません。{MIN_RUNS:,} 走は、平均と 100% をかろうじて見分けられる最低限の数です。"
            f"300〜999 走の種牡馬は、記事の末尾の参考の表に載せています。")},
        {"type": "heading", "text": "種牡馬ごとの得意条件（上位 5 件と 100% 超え）"},
        {"type": "note", "text": (
            f"条件は出走 {MIN_COND} 以上のものだけ（{MIN_COND} 走でも単勝回収率の 95% 区間は ±{half(MIN_COND):.0f}pt ほどあります）。"
            "「単勝 前半 → 後半」で、前半・後半の両方で 100% を超えている条件のほうが、まだ信用できます。"
            "それでも将来も続く保証はなく、オッズはその後の人気で変わる点に注意してください。")},
        *accs,
        {"type": "heading", "text": "まとめ"},
        {"type": "text", "text": (
            f"・産駒 {MIN_RUNS:,} 走以上の {len(m)} 種牡馬で、単勝回収率の 95% 区間が 100% を上回るのは {a['n_over_ci']} 頭。父だけで回収率 100% 超えを狙うのは難しい。\n"
            f"・単勝 {LONGSHOT} 倍以下に絞ると、全体の回収率は {r0(A['roi50'])}、種牡馬の間のばらつき（標準偏差）は {m['roi'].std() * 100:.0f}pt から {m['roi50'].std() * 100:.0f}pt に縮む。"
            + ("、".join(f"{r['sire']}（{r0(r['roi'])} → {r0(r['roi50'])}）" for _, r in drop.iterrows()) + " のように、単勝回収率の高さが大穴の数回によるものもある。" if len(drop) else "")
            + "\n"
            f"・条件で絞ると 100% 超えの条件は多数見つかるが、前半に 100% 超えだった条件の後半の回収率は {V['sel_b'] * 100:.0f}%（他の条件は {V['rest_b'] * 100:.0f}%）で、選び出した高さはほとんど続かない。\n"
            "・折り畳みの条件は「過去にこうだった」という記録として見て、前半・後半の両方で高いか、95% 区間が 100% を超えるかを確かめてから使うのがよい。")},
        {"type": "accordion", "title": f"参考: 産駒 {MIN_REF}〜{MIN_RUNS - 1} 走の種牡馬（{len(a['ref'])} 頭）", "blocks": [
            {"type": "table", "columns": ["父", "出走", "着別度数（1-2-3-着外）", "勝率", "複勝率", "単勝回収率（95%区間）", f"単勝{LONGSHOT}倍以下（95%区間）", "複勝回収率（95%区間）"],
             "rows": ref_rows,
             "caption": "出走が少ないので 95% 区間が広い。点推定の回収率の順位は、ほぼ偶然で決まる。"},
        ]},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地（障害を除く）。取消・除外は除き、中止・失格は着外として数えています。父は血統表から付け、対象の出走のすべてに付いています。\n\n"
                "・回収率はレースの払戻（単勝・複勝）の実績です。同着は払戻どおり。複勝の発売がないレースは複勝回収率から除いています。\n\n"
                "・条件: 芝ダ×距離（短距離 〜1400m・マイル 1401〜1800m・中距離 1801〜2200m・長距離 2201m〜）、競馬場×芝ダ、クラス（OP・重賞は L・G を含む）、芝ダ×馬場（良 / 稍重〜不良）、年齢、性別。\n\n"
                f"・前半・後半の検証: 両期間に {MIN_HALF} 走以上ある「種牡馬 × 条件」で、前半の単勝回収率と後半の単勝回収率を比べました。"
                "前半が高い条件の後半が全体並みに戻るなら、前半の高さは偶然（選択バイアス）だったと考えられます。\n\n"
                "・以前の版で挙げた American Pharoah・グレーターロンドンなどは産駒が 1,000 走未満で、参考の表に移しました。")},
        ]},
    ]
    return {
        "slug": "sire-roi",
        "title": "血統で狙う——種牡馬の回収率は続くのか",
        "claim": "特定の血統は高回収率を維持している",
        "verdict": "not_supported" if a["n_over_ci"] == 0 and V["sel_b"] < 1.0 else "partial",
        "summary": (f"産駒 {MIN_RUNS:,} 走以上の {len(m)} 種牡馬で、単勝回収率の 95% 区間が 100% を上回るのは {a['n_over_ci']} 頭。"
                    f"条件で絞った 100% 超えも、前半（2016〜{SPLIT_YEAR}）で選んだ条件の後半の回収率は {V['sel_b'] * 100:.0f}% にとどまり、高い回収率は続きませんでした。"),
        "categories": ["bloodline/maturity"],
        "published": "2026-10-28",
        "updated": "2026-11-09",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": f"JRA全競馬場・芝ダート（障害除く）。本表は産駒の出走 {MIN_RUNS:,} 以上、参考は {MIN_REF} 以上",
                    "sample": f"{a['n_runs']:,} 走（{a['n_horses']:,} 頭・{a['n_sires']:,} 種牡馬）。本表 {len(m)} 種牡馬"},
        "body": body,
    }


def report(a: dict[str, Any]) -> None:
    print(f"runs={a['n_runs']} sd_win={a['sd_win']:.3f} sd_win50={a['sd_win50']:.3f} all={ {k: a['all'][k] for k in ('roi', 'roi50', 'proi')} }")
    print(f"major={len(a['major'])} ref={len(a['ref'])} over_ci={a['n_over_ci']} over_pt={a['n_over_point']} over_ci50={a['n_over_ci50']} over_pci={a['n_over_pci']}")
    print({k: v for k, v in a["val"].items() if not k.startswith("_")})
    print(a["m"][["sire", "n", "win", "top3", "roi", "roi_ci", "roi50", "proi", "roi_a", "roi_b", "n_over"]].round(3).head(80).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    df = frame()
    a = analyze(df)
    report(a)
    col = article(a)
    text = dumps(col)
    assert json.loads(text) == col
    if args.dry_run:
        print(f"(dry-run) {len(text):,} bytes")
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT} ({len(text):,} bytes)")


if __name__ == "__main__":
    main()
