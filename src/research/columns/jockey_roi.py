"""めぐコラム『勝率1位≠回収率1位』を実データから集計し、frontend/content/columns/jockey-roi.json を書く。

問い: 勝率の高い騎手（調教師）を買うと儲かるか。回収率の高い騎手・調教師は、次の期間も高いか（狙い目として使えるか）。

落とし穴: 単勝回収率は高配当 1 回で大きく動く（100 倍の 1 勝で、500 騎乗の回収率が 20pt 上がる）。
  同じ期間の回収率で上位を選ぶと、偶然の上振れを拾う（選択バイアス）。そこで期間を前半（2016〜2020）・後半（2021〜2025）に分け、
  前半で選んだ騎手・調教師の後半の成績を見る。両期間とも 500 騎乗（調教師は 300 出走）以上の人だけ。

データ: jockey_common.frame()（2016〜2025 年の JRA 平地、障害・取消・除外を除く。exp_win = 人気と頭数から見込まれる勝率）。
比較グラフ「勝率上位8名 vs それ以外の騎手 × 人気帯別 単勝回収率」は compare_series が書くので、既存の JSON から引き継ぐ。

使い方:
    python -m src.research.columns.jockey_roi            # 集計して記事を書く
    python -m src.research.columns.jockey_roi --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import FIRST_YEAR as cs_FIRST, FLAT_DIR, HALF_SPLIT as cs_SPLIT, YEARS
from src.research.columns.jockey_common import frame, jockey_names
from src.research.columns.race_context import series
from src.research.columns.wind_venue import mean_ci, pct

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/jockey-roi.json"
KEEP_CHART = "勝率上位8名 vs それ以外の騎手 × 人気帯別 単勝回収率"
QS = ["下位\n20%", "20〜40%", "40〜60%", "60〜80%", "上位\n20%"]
HALVES = {"A": f"前半（{cs_FIRST}〜{cs_SPLIT}年）", "B": f"後半（{cs_SPLIT + 1}〜2025年）"}
NAMED = ["川田将雅", "ルメール", "横山武史", "水沼元輝", "石田拓郎", "柴山雄一"]


def persistence(d: pd.DataFrame, key: str, min_n: int) -> dict[str, Any]:
    g = d.groupby([key, "half"]).agg(n=("ret", "size"), roi=("ret", "mean"), win=("win", "mean"), ew=("exp_win", "mean")).unstack()
    g = g[(g[("n", "A")] >= min_n) & (g[("n", "B")] >= min_n)]
    q = pd.qcut(g[("roi", "A")].rank(method="first"), 5, labels=QS)
    top = g[("roi", "A")].sort_values(ascending=False).head(10).index
    res = g["win"] - g["ew"]
    return {"n": len(g), "corr": float(np.corrcoef(g[("roi", "A")], g[("roi", "B")])[0, 1]),
            "corr_res": float(np.corrcoef(res["A"], res["B"])[0, 1]),
            "qA": [float(g.loc[q == x, ("roi", "A")].mean()) for x in QS], "qB": [float(g.loc[q == x, ("roi", "B")].mean()) for x in QS],
            "qN": [int((q == x).sum()) for x in QS],
            "top10A": float(g.loc[top, ("roi", "A")].mean()), "top10B": float(g.loc[top, ("roi", "B")].mean()), "ids": g.index, "q": q}


def pchart(p: dict[str, Any], d: pd.DataFrame, key: str, who: str, title: str) -> dict[str, Any]:
    """前半の回収率で 5 群に分け、群ごとの前半・後半の回収率（出走をまとめた平均。95% 区間つき）。"""
    def grp(h: str) -> list[pd.Series]:
        return [d.loc[(d["half"] == h) & d[key].isin(p["q"][p["q"] == x].index), "ret"] for x in QS]
    sa, sb = series(HALVES["A"], grp("A")), series(HALVES["B"], grp("B"))
    for s, h in ((sa, "A"), (sb, "B")):
        s["ci"] = [[max(0.0, c[0]), c[1]] if c else None for c in (mean_ci(v) for v in grp(h))]
    return {"type": "chart", "title": title, "xLabel": f"前半の単勝回収率で並べた{who}の 5 群", "yLabel": "単勝回収率", "xs": QS, "series": [sa, sb],
            "caption": f"両期間とも基準以上に乗った（出走した）{who} {p['n']} 人を、前半の回収率で 5 等分。縦線は 95% 信頼区間。"}


def analyze(d: pd.DataFrame) -> dict[str, Any]:
    d = d.copy()
    d["half"] = np.where(d["year"] <= cs_SPLIT, "A", "B")
    tr = pd.concat([pd.read_parquet(FLAT_DIR / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_id", "trainer_id"]) for y in YEARS])
    tr["horse_id"] = tr["horse_id"].astype(str)
    d = d.merge(tr.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    d["trainer_id"] = d["trainer_id"].astype(str)
    a: dict[str, Any] = {}
    # 1. 勝率で並べた騎手の回収率（全期間・500 騎乗以上）
    j = d.groupby("jockey_id").agg(n=("ret", "size"), win=("win", "mean"))
    j = j[j["n"] >= 500]
    jq = pd.qcut(j["win"].rank(method="first"), 5, labels=QS)
    parts = [d.loc[d["jockey_id"].isin(jq[jq == x].index), "ret"] for x in QS]
    s = series("単勝回収率", parts)
    s["ci"] = [mean_ci(v) for v in parts]
    a["c_win_q"] = {"type": "chart", "title": "勝率で並べた騎手 5 群の単勝回収率", "xLabel": "騎手の勝率（2016〜2025年、500騎乗以上）", "yLabel": "単勝回収率",
                    "xs": QS, "series": [s], "caption": f"{len(j)} 人を勝率で 5 等分し、群ごとに全騎乗の単勝回収率を出したもの。縦線は 95% 信頼区間。"}
    a["win_q"] = [float(v.mean()) for v in parts]
    a["win_q_range"] = [(float(j.loc[jq == x, "win"].min()), float(j.loc[jq == x, "win"].max())) for x in QS]
    # 2. 騎手・調教師の回収率は続くか
    a["pj"] = persistence(d, "jockey_id", 500)
    a["pt"] = persistence(d, "trainer_id", 300)
    a["c_pj"] = pchart(a["pj"], d, "jockey_id", "騎手", "前半の回収率で分けた騎手の、前半と後半の単勝回収率")
    a["c_pt"] = pchart(a["pt"], d, "trainer_id", "調教師", "前半の回収率で分けた調教師の、前半と後半の単勝回収率")
    # 3. 以前の版に載っていた騎手
    names = jockey_names(d)
    rows = []
    for nm in NAMED:
        ids = [k for k, v in names.items() if v == nm]
        x = d[d["jockey_id"].isin(ids)]
        if x.empty:
            continue
        ci = mean_ci(x["ret"])
        ci = [max(0.0, ci[0]), ci[1]]       # 回収率は 0 未満にならない
        big = x.loc[x["win"] == 1, "odds"].max()
        wo = (x["ret"].sum() - (big if isinstance(big, float) and not math.isnan(big) else 0)) / len(x)
        rows.append({"name": nm, "n": len(x), "win": x["win"].mean(), "exp": x["exp_win"].mean(), "roi": x["ret"].mean(), "ci": ci,
                     "big": big, "wo": wo, "A": d.loc[d["jockey_id"].isin(ids) & (d["half"] == "A"), "ret"].mean(),
                     "B": d.loc[d["jockey_id"].isin(ids) & (d["half"] == "B"), "ret"].mean()})
    a["named"] = rows
    a["n_runs"] = len(d)
    return a


def article(a: dict[str, Any], keep: dict[str, Any] | None) -> dict[str, Any]:
    pj, ptr = a["pj"], a["pt"]
    wq = a["win_q"]
    named = {r["name"]: r for r in a["named"]}
    mz = named.get("水沼元輝")

    def f(v: float) -> str:
        return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else pct(v, 0)

    trows = [[r["name"], f"{r['n']:,}", pct(r["win"]), pct(r["exp"]), f"{pct(r['roi'], 0)}（{pct(r['ci'][0], 0)}〜{pct(r['ci'][1], 0)}）",
              "–" if math.isnan(r["big"]) else f"{r['big']:.1f}倍", pct(r["wo"], 0), f(r["A"]), f(r["B"])] for r in a["named"]]
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "勝率上位 20% の騎手の単勝回収率", "value": pct(wq[-1]), "sub": f"下位 20% は {pct(wq[0])}"},
            {"label": "前半の回収率上位 10 人 → 後半", "value": f"{pct(pj['top10A'], 0)} → {pct(pj['top10B'], 0)}", "sub": "騎手。回収率の高さは続かない"},
            {"label": "前半と後半の回収率の相関", "value": f"{pj['corr']:.2f}", "sub": f"騎手 {pj['n']} 人（調教師 {ptr['corr']:.2f}）。0 なら無関係"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"「勝率の高い騎手ほど回収率も高い」は、相対的には正しい結果でした。勝率上位 20% の騎手の単勝回収率は {pct(wq[-1])}、下位 20% は {pct(wq[0])} です。"
            "人気薄の馬は実力以上に買われやすく（人気薄ほど回収率が低い）、勝率の低い騎手は人気薄に乗ることが多いためと考えられます。"
            "ただし、どの群でも 100% には届かず、勝率の高い騎手を買い続けて儲かるわけではありません。\n\n"
            "では「回収率の高い騎手（穴騎手）を狙えばよいか」というと、それも根拠がありません。"
            f"2016〜2020 年の回収率で上位 10 人を選ぶと、その期間は {pct(pj['top10A'], 0)} ですが、2021〜2025 年には {pct(pj['top10B'], 0)} に下がります。"
            f"騎手 {pj['n']} 人の前半と後半の回収率の相関は {pj['corr']:.2f}（調教師 {ptr['n']} 人では {ptr['corr']:.2f}）で、ある期間の回収率は次の期間をほとんど予想しません。"
            "高い回収率の多くは、少数の高配当による偶然の上振れです。")},
        a["c_win_q"],
        {"type": "heading", "text": "回収率の高い騎手・調教師は、次の期間も高いか"},
        a["c_pj"],
        a["c_pt"],
        {"type": "text", "text": (
            f"前半の回収率で騎手を 5 群に分けると、前半は {pct(pj['qA'][0], 0)}〜{pct(pj['qA'][-1], 0)} と大きな差がありますが、"
            f"後半は {pct(min(pj['qB']), 0)}〜{pct(max(pj['qB']), 0)} にそろい、前半の上位群が後半も上位とは限りません。調教師も同じです"
            f"（前半 {pct(ptr['qA'][0], 0)}〜{pct(ptr['qA'][-1], 0)} → 後半 {pct(min(ptr['qB']), 0)}〜{pct(max(ptr['qB']), 0)}）。\n\n"
            f"回収率ではなく「実際の勝率 − 人気から見込まれる勝率」で見ると、前半と後半の相関は騎手で {pj['corr_res']:.2f}、調教師で {ptr['corr_res']:.2f} と少しだけ続きます。"
            "人気より勝つ騎手・調教師はわずかにいるものの、その差は小さく、単勝の回収率を 100% 超にするほどではありません。")},
        {"type": "heading", "text": "以前の版に載せていた騎手"},
        {"type": "table", "columns": ["騎手", "騎乗数", "勝率", "人気から見込まれる勝率", "単勝回収率（95%区間）", "最高配当の勝ち", "その 1 勝を除いた回収率", "前半", "後半"],
         "rows": trows,
         "caption": "2016〜2025 年の JRA 平地（障害を除く）。前半 = 2016〜2020 年、後半 = 2021〜2025 年（騎乗の無い期間は –）。"},
        {"type": "text", "text": (
            (f"以前の版で「回収率 143%」としていた水沼騎手は、{mz['big']:.0f} 倍の 1 勝を除くと {pct(mz['wo'], 0)} で、95% 区間は {pct(mz['ci'][0], 0)}〜{pct(mz['ci'][1], 0)} と、"
             "100% を下回る可能性も十分にあります。" if mz else "")
            + "1 回の高配当で回収率は大きく変わるので、数百騎乗の回収率で騎手を「穴騎手」と呼ぶことはできません。"
            "なお以前の版の上野翔騎手は、平地の騎乗が 2016〜2025 年に十数回しかなく（主に障害の騎手）、500 騎乗以上の表に載せたのは誤りでした。")},
    ]
    if keep:
        body += [{"type": "heading", "text": "人気帯で分けると"}, keep,
                 {"type": "text", "text": "勝率上位 8 名とそれ以外の騎手で、人気帯ごとの回収率を比べても、差が有意なのは 4〜6 番人気だけで、上位騎手の人気薄が特に得とは言えません。"}]
    body += [
        {"type": "heading", "text": "馬券への活用"},
        {"type": "text", "text": (
            "勝率の低い騎手の人気薄は、回収率がとくに低いので避けるのが無難です。一方、勝率の高い騎手を買っても、回収率の高かった騎手を買っても、100% を超える得は続きません。"
            "騎手は「馬の力を出し切れるか」の材料として、人気（オッズ）と比べて評価してください。")},
        {"type": "accordion", "title": "データと方法・以前の版からの変更", "blocks": [
            {"type": "text", "text": (
                f"・対象: 2016〜2025 年の JRA 平地 {a['n_runs']:,} 出走（障害・取消・除外を除く）。単勝回収率は、単勝を 100 円ずつ買ったときの払戻の割合です。\n\n"
                "・人気から見込まれる勝率は、単勝オッズから求めた勝率を 50 段階に分け、頭数の帯ごとに実際の勝率をあてはめたもの（騎手コラム共通）。\n\n"
                "・以前の版は、同じ期間の回収率で騎手・調教師の上位を選んで載せていました。これは偶然の上振れを選ぶことになるので、"
                "期間を前後に分けて「前半で選び、後半で確かめる」形に直しました。調教師の表（200 出走で回収率 187% など）も同じ理由で外しました。")},
        ]},
    ]
    return {
        "slug": "jockey-roi",
        "title": "勝率の高い騎手ほど回収率は高い——それでも100%には届かず、「穴騎手」も続かない",
        "claim": "勝率の高い騎手が最も儲かる",
        "verdict": "partial",
        "summary": (f"勝率上位 20% の騎手の単勝回収率は {pct(wq[-1])}（下位 20% は {pct(wq[0])}）と高めですが、100% には届きません。回収率の高い「穴騎手」も続かず、"
                    f"前半の上位 10 人は後半 {pct(pj['top10B'], 0)} に下がります（前後半の相関 {pj['corr']:.2f}）。高い回収率の多くは高配当の偶然です。"),
        "categories": ["jockey/distance"],
        "published": "2026-10-28",
        "updated": "2026-11-03",
        "dataset": {"period": "2016〜2025年（前半 2016〜2020・後半 2021〜2025 で確かめる）",
                    "conditions": "JRA 平地（障害を除く）。前後半の比較は両期間とも 500 騎乗以上の騎手・300 出走以上の調教師",
                    "sample": f"{a['n_runs']:,} 出走"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    old = json.loads(OUT.read_text(encoding="utf-8"))
    keep = next((b for b in old["body"] if b.get("type") == "chart" and b.get("title") == KEEP_CHART), None)
    old_cats = old.get("categories")
    col = article(analyze(frame()), keep)
    if old_cats:
        col["categories"] = old_cats
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
