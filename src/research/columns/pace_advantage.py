"""めぐコラム『ハイペースになるほど、1番人気は信頼できない』を実データから集計し、frontend/content/columns/pace-advantage.json を書く。

問い: ハイペースのレースは荒れやすいか（1 番人気が崩れ、人気薄が来るか）。それはレース前に読めるか。オッズに織り込まれているか。

ペースの区分（data/local/modeling/shared_cache/race_quality/race_quality.parquet の pace_cls。src/scripts/data/tp61_analysis.py）:
  上がり3F のコース標準からのずれ − 前半3F のコース標準からのずれ（bal）の、基準年の上位 25% = ハイ、下位 25% = スロー、残り = ミドル。
  レースの結果として決まるので、レース前には分からない。
レース前の手がかり: 出走馬のうち「前寄り」（前走まで直近 3 走の初角の位置の平均が前から 15% 以内。race_context と同じ）の頭数。
市場との比較: オッズ（log とその 2 乗）・コース・頭数をそろえた線形確率モデル。

使い方:
    python -m src.research.columns.pace_advantage            # 集計して記事を書く
    python -m src.research.columns.pace_advantage --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import RACE_QUALITY_PATH, YEARS, _p, _stat, load, pace_charts
from src.research.columns.race_context import NOW_STYLES, ols_fit, series, with_context
from src.research.columns.wind_venue import pct, pfmt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/pace-advantage.json"
PACES = {"high": "ハイ", "mid": "ミドル", "slow": "スロー"}
NF_BINS = ([-1, 0, 1, 2, 3, 99], ["0頭", "1頭", "2頭", "3頭", "4頭以上"])


def frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = load()
    d = with_context(base)
    rq = pd.read_parquet(RACE_QUALITY_PATH, columns=["race_id", "pace_cls"]).dropna().drop_duplicates("race_id")
    d = d.merge(rq, on="race_id", how="left")
    d["lq"] = np.log(d["odds"])
    d["key"] = d["venue"] + d["surface"] + d["distance"].astype(str)
    d["front"] = (d["style_pre"] == "前寄り").astype(int)
    nf = d.groupby("race_id")["front"].sum().rename("nf")
    d = d.merge(nf, left_on="race_id", right_index=True)
    d["nfb"] = pd.cut(d["nf"], NF_BINS[0], labels=NF_BINS[1]).astype(str)
    return base, d


def analyze(base: pd.DataFrame, d: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    a["main"] = pace_charts(base)[0][1]                    # 人気帯 × ペース（compare_series と同じグラフ）
    ser = {s["name"]: s["values"] for s in a["main"]["series"]}
    a["fav"] = dict(zip(PACES, ser["1番人気"]))
    a["low"] = dict(zip(PACES, ser["7番人気以下"]))
    a["mid46"] = dict(zip(PACES, ser["4〜6番人気"]))
    a["avgpop"] = dict(zip(PACES, a["main"]["y2"]["series"][0]["values"]))
    p = d[d["pace_cls"].notna()]
    now = p[p["style_now"].notna()]
    a["c_style"] = {"type": "chart", "title": "ペース別 今走の脚質ごとの複勝率", "xLabel": "ペース（結果）", "yLabel": "複勝率", "xs": list(PACES.values()),
                    "series": [series(s, [now.loc[(now["style_now"] == s) & (now["pace_cls"] == k), "top3"] for k in PACES]) for s in NOW_STYLES],
                    "caption": "脚質はそのレースの初角の位置（逃げ=先頭、先行=前から 25% 以内、差し=65% 以内、追込=それより後ろ）。ペースも脚質もレースの結果。縦線は 95% 信頼区間。"}
    a["style"] = {(s, k): now.loc[(now["style_now"] == s) & (now["pace_cls"] == k), "top3"].mean() for s in NOW_STYLES for k in PACES}
    fp = p[p["popularity"] == 1]
    f = ols_fit("top3 ~ C(pace_cls, Treatment('mid')) + starters + C(grade) + C(key)", fp)
    a["pace_adj"] = {k: (float(f.params[f"C(pace_cls, Treatment('mid'))[T.{k}]"]), float(f.pvalues[f"C(pace_cls, Treatment('mid'))[T.{k}]"])) for k in ("high", "slow")}
    # レース前の手がかり: 前寄りの頭数
    races = p.drop_duplicates("race_id")
    xs = NF_BINS[1]
    a["c_nf_pace"] = {"type": "chart", "title": "前寄りの馬の頭数別 ハイペース・スローペースになった割合", "xLabel": "出走馬のうち前寄りの頭数（レース前に分かる）",
                      "yLabel": "レースの割合", "xs": xs,
                      "series": [series(f"{PACES[k]}ペース", [(races.loc[races["nfb"] == x, "pace_cls"] == k).astype(float) for x in xs]) for k in ("high", "slow")],
                      "caption": "前寄り=前走まで直近 3 走の初角の位置の平均が前から 15% 以内の馬。ハイ・スローはそれぞれ全レースの約 4 分の 1。前寄りが多いレースは頭数も多い（平均 13.4 頭 → 15.0 頭）ので、このグラフは頭数をそろえていない。1 番人気の回帰（本文）は頭数をそろえている。縦線は 95% 信頼区間。"}
    a["nf_pace"] = {x: ((races["nfb"] == x).sum(), (races.loc[races["nfb"] == x, "pace_cls"] == "high").mean(), (races.loc[races["nfb"] == x, "pace_cls"] == "slow").mean()) for x in xs}
    fav = d[d["popularity"] == 1]
    ps = []
    for x in xs:
        w = fav[fav["nfb"] == x]
        ma, _, _, va = _stat(w["win"])
        mb, _, _, vb = _stat(w["implied"])
        pv = _p(va, vb, ma, mb)
        ps.append(None if pv is None else float(f"{pv:.3g}"))
    a["c_nf_fav"] = {"type": "chart", "title": "前寄りの馬の頭数別 1番人気の成績", "xLabel": "出走馬のうち前寄りの頭数（レース前に分かる）", "yLabel": "率", "xs": xs,
                     "series": [series("複勝率", [fav.loc[fav["nfb"] == x, "top3"] for x in xs]),
                                series("勝率", [fav.loc[fav["nfb"] == x, "win"] for x in xs]),
                                series("オッズから見込まれる勝率", [fav.loc[fav["nfb"] == x, "implied"] for x in xs])],
                     "compare": {"a": "勝率", "b": "オッズから見込まれる勝率", "p": ps},
                     "caption": "1 番人気だけ。オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。＊は勝率と見込みの差が有意（p<0.05）。"}
    a["nf_fav"] = {x: fav.loc[fav["nfb"] == x, ["top3", "win", "implied"]].mean().to_dict() for x in xs}
    a["nf_adj_odds"] = ols_fit("top3 ~ nf + lq + I(lq**2) + starters + C(key)", fav)
    a["nf_adj_noodds"] = ols_fit("top3 ~ nf + starters + C(key)", fav)
    a["nf_adj_win"] = ols_fit("win ~ nf + lq + I(lq**2) + starters + C(key)", fav)
    a["n_races"] = int(races["race_id"].nunique())
    return a


def article(a: dict[str, Any]) -> dict[str, Any]:
    F, L, AP = a["fav"], a["low"], a["avgpop"]
    S = a["style"]
    hi, sl = a["pace_adj"]["high"], a["pace_adj"]["slow"]
    o, no, w = a["nf_adj_odds"], a["nf_adj_noodds"], a["nf_adj_win"]
    first, last = NF_BINS[1][0], NF_BINS[1][-1]
    NP, NFV = a["nf_pace"], a["nf_fav"]
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "1番人気の複勝率（スロー → ハイ）", "value": f"{pct(F['slow'])} → {pct(F['high'])}", "sub": f"{(F['high'] - F['slow']) * 100:+.1f}pt"},
            {"label": "7番人気以下の複勝率（スロー → ハイ）", "value": f"{pct(L['slow'])} → {pct(L['high'])}", "sub": f"{(L['high'] - L['slow']) * 100:+.1f}pt"},
            {"label": f"前寄りが{last}のとき 1番人気の複勝率", "value": pct(NFV[last]["top3"]), "sub": f"{first}なら {pct(NFV[first]['top3'])}（レース前に分かる）"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"ハイペースのレースは荒れます。1 番人気の複勝率はスロー {pct(F['slow'])}・ミドル {pct(F['mid'])}・ハイ {pct(F['high'])} と下がり、"
            f"7 番人気以下は {pct(L['slow'])} → {pct(L['high'])} と上がります。4〜6 番人気はほぼ横ばいです。"
            f"頭数・クラス・コースをそろえても、1 番人気の複勝率はミドルと比べてハイ {hi[0] * 100:+.1f}pt（{pfmt(hi[1])}）、スロー {sl[0] * 100:+.1f}pt（{pfmt(sl[1])}）です。\n\n"
            "ただしペースはレースの結果で、レース前には分かりません。予想に使えるのは、出走馬のうち前に行く馬（前寄り）の頭数です。"
            f"前寄りが{first}のレースはハイペースが {pct(NP[first][1], 0)}・スローが {pct(NP[first][2], 0)} ですが、{last}だとハイ {pct(NP[last][1], 0)}・スロー {pct(NP[last][2], 0)} になり、"
            f"1 番人気の複勝率も {pct(NFV[first]['top3'])} → {pct(NFV[last]['top3'])} と下がります。")},
        a["main"],
        {"type": "heading", "text": "ペースの区分と、脚質ごとの成績"},
        {"type": "text", "text": (
            "ペースは、前半 3F と上がり 3F が、それぞれ同じコースの標準よりどれだけ速かったか・遅かったかの差で決めています。"
            "前半が速く上がりがかかったレースの上位 25% をハイ、逆の下位 25% をスロー、残りをミドルとしました（コースの違いを除いた、レースの中での速さの配分です）。\n\n"
            f"ペースが速いほど、前にいた馬が苦しくなります。逃げた馬の複勝率はスロー {pct(S[('逃げ', 'slow')])} → ハイ {pct(S[('逃げ', 'high')])}、"
            f"追込は {pct(S[('追込', 'slow')])} → {pct(S[('追込', 'high')])} です。1 番人気が崩れるのも前で運ぶ人気馬が苦しくなるため、と考えられますが、このデータでは確かめていません（仮説）。")},
        a["c_style"],
        {"type": "heading", "text": "レース前に読む: 前寄りの馬の頭数"},
        a["c_nf_pace"],
        a["c_nf_fav"],
        {"type": "text", "text": (
            f"前寄りの馬が 1 頭増えるごとに、1 番人気の複勝率は {no.params['nf'] * 100:+.2f}pt（頭数・コースをそろえて、{pfmt(no.pvalues['nf'])}）下がります。"
            f"1 番人気のオッズもそろえると {o.params['nf'] * 100:+.2f}pt（{pfmt(o.pvalues['nf'])}）に縮みますが、0 にはなりません。市場は一部しか織り込んでいないと考えられます。\n\n"
            f"一方、勝率はオッズの見込みとほぼ同じです（オッズをそろえて 1 頭あたり {w.params['nf'] * 100:+.2f}pt、{pfmt(w.pvalues['nf'])}）。"
            "前に行く馬が多いレースの 1 番人気は、「勝つ」見込みは変わらないものの、「3 着も外す」ことが増えます。\n\n"
            "ただし、集計に使っていない 2026 年のデータで確かめると、このオッズをそろえた差は再現しませんでした（記事『ペースは読めるか』）。馬券の根拠にするには、まだ確かではありません。")},
        {"type": "heading", "text": "馬券への活用"},
        {"type": "text", "text": (
            "出走表で前寄りの馬（前走まで前々で運んできた馬）が多いレースは、ハイペースになりやすく、1 番人気が 3 着を外すことが増えます。"
            "単勝の評価は妥当でも、1 番人気を複勝・ワイド・3 連複の軸にするときは割り引いて見る余地があります（2026 年には再現していないので、強くは言えません）。"
            "人気薄との組み合わせを広げると回収率が上がるか（どの買い方が得か）は、このデータでは確かめていません。")},
        {"type": "link", "href": "/column/pace-scenario", "label": "ペースは読めるか——2026年のデータで確かめる"},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地（障害を除く）で、ペースの区分があるレース（{a['n_races']:,} レース）。取消・除外は除いています。\n\n"
                "・前寄りの頭数は、前走（JRA 平地、2019 年から）のある馬だけで数えています。デビュー戦や前走が地方の馬は数に入りません。\n\n"
                "・ペースの区分は、前半・上がりのラップから決まるため、レースの展開そのものです。「ハイペースだから荒れた」と「荒れる展開（人気馬が早めに動くなど）だったからハイペースになった」は、"
                "このデータでは区別できません。")},
        ]},
    ]
    return {
        "slug": "pace-advantage",
        "title": "ハイペースになるほど、1番人気は信頼できない",
        "claim": "ハイペースは荒れやすい",
        "verdict": "supported",
        "summary": (f"1 番人気の複勝率はスロー {pct(F['slow'])}、ハイ {pct(F['high'])}。ペースは結果ですが、前に行く馬が多いレースはハイペースになりやすく、"
                    f"1 番人気の複勝率も下がります（オッズをそろえても 1 頭あたり {o.params['nf'] * 100:+.1f}pt）。"),
        "categories": ["style/front-runner", "upset/conditions"],
        "published": "2026-10-28",
        "updated": "2026-11-03",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年", "conditions": "JRA全競馬場・芝ダート（障害除く）。ペースの区分があるレース",
                    "sample": f"{a['n_races']:,} レース"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    col = article(analyze(*frame()))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
