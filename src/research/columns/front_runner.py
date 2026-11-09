"""めぐコラム『逃げ・先行は本当に有利？』を実データから集計し、frontend/content/columns/front-runner.json を書く。

問い: 逃げ・先行の馬は有利か。予想（馬券）に使えるか。芝・ダート・距離で違うか。

脚質は src/research/columns/race_context.py と同じ定義:
  今走の脚質（逃げ/先行/差し/追込）= 今走の初角の位置。レースの結果なので予想には使えない（説明用）。
  事前の脚質（前寄り/好位/中団/後方）= 前走まで直近 3 走の初角の位置の平均。予想の時点で分かる。
市場との比較は、オッズ（log とその 2 乗）・コースをそろえた線形確率モデル（レース単位クラスタ SE）。区分を並べるときは Holm 補正。

以前の版は「騎手の平均の通過順」で脚質を分けていたが、騎手の腕・乗る馬の強さと区別できないので、馬の位置取りで集計し直した。

使い方:
    python -m src.research.columns.front_runner            # 集計して記事を書く
    python -m src.research.columns.front_runner --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import DIST_BANDS, YEARS, _p, _stat, load
from src.research.columns.race_context import NOW_STYLES, PRE_STYLES, ols_fit, series, with_context
from src.research.columns.wind_venue import holm, mean_ci, pct, pfmt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/front-runner.json"
SURFACES = ["芝", "ダート"]
DIST = ["全体"] + [b.split("\n")[0] for b in DIST_BANDS]


def frame() -> pd.DataFrame:
    d = with_context(load())
    d["lq"] = np.log(d["odds"])
    d["key"] = d["venue"] + d["surface"] + d["distance"].astype(str)
    d["db"] = d["dist_band"].astype(str).str.split("\n").str[0]
    return d


def _sub(d: pd.DataFrame, dist: str) -> pd.DataFrame:
    return d if dist == "全体" else d[d["db"] == dist]


def by_surface_chart(d: pd.DataFrame, col: str, xs: list[str], metric: str, title: str, x_label: str, y_label: str, caption: str) -> dict[str, Any]:
    """横軸 = 脚質、系列 = 芝・ダート。距離帯のプルダウンで切り替える。"""
    def ser(x: pd.DataFrame) -> list[dict[str, Any]]:
        return [series(sf, [x.loc[(x["surface"] == sf) & (x[col] == s), metric] for s in xs]) for sf in SURFACES]
    return {"type": "chart", "title": title, "xLabel": x_label, "yLabel": y_label, "xs": xs, "series": ser(d),
            "filters": [{"label": "距離", "options": DIST}], "variants": {k: {"series": ser(_sub(d, k))} for k in DIST[1:]}, "caption": caption}


def implied_chart(d: pd.DataFrame) -> dict[str, Any]:
    """事前の脚質ごとに、実際の勝率とオッズから見込まれる勝率。芝・ダートで切り替える。"""
    def build(x: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        a, b, ps = [], [], []
        for s in PRE_STYLES:
            w = x[x["style_pre"] == s]
            a.append(w["win"]); b.append(w["implied"])
            ma, _, _, va = _stat(w["win"])
            mb, _, _, vb = _stat(w["implied"])
            p = _p(va, vb, ma, mb)
            ps.append(None if p is None else float(f"{p:.3g}"))
        return [series("実際の勝率", a), series("オッズから見込まれる勝率", b)], {"a": "実際の勝率", "b": "オッズから見込まれる勝率", "p": ps}
    s0, c0 = build(d)
    variants = {}
    for sf in SURFACES:
        s, c = build(d[d["surface"] == sf])
        variants[sf] = {"series": s, "compare": c}
    return {"type": "chart", "title": "事前の脚質別 勝率（実際とオッズの見込み）", "xLabel": "前走までの位置取り", "yLabel": "勝率", "xs": PRE_STYLES,
            "series": s0, "compare": c0, "filters": [{"label": "芝・ダート", "options": ["全体"] + SURFACES}], "variants": variants,
            "caption": "前寄り=前走まで 3 走の初角の位置の平均が前から 15% 以内、好位=40% 以内、中団=70% 以内、後方=それより後ろ。"
                       "オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。＊は差が有意（p<0.05）。"}


def analyze(d: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    now = d[d["style_now"].notna()]
    pre = d[d["style_pre"].notna()]
    a["now"] = {(sf, s): now.loc[(now["surface"] == sf) & (now["style_now"] == s), "top3"].mean() for sf in SURFACES for s in NOW_STYLES}
    a["pre"] = {(sf, s): pre.loc[(pre["surface"] == sf) & (pre["style_pre"] == s), ["top3", "win", "implied", "ret", "popularity"]].mean().to_dict()
                | {"n": int(((pre["surface"] == sf) & (pre["style_pre"] == s)).sum()),
                   "roi_ci": mean_ci(pre.loc[(pre["surface"] == sf) & (pre["style_pre"] == s), "ret"])}
                for sf in SURFACES for s in PRE_STYLES}
    rows = []
    for sf in SURFACES:
        x = pre[pre["surface"] == sf]
        f3 = ols_fit("top3 ~ C(style_pre, Treatment('中団')) + lq + I(lq**2) + C(key)", x)
        fw = ols_fit("win ~ C(style_pre, Treatment('中団')) + lq + I(lq**2) + C(key)", x)
        for s in ("前寄り", "好位", "後方"):
            k = f"C(style_pre, Treatment('中団'))[T.{s}]"
            rows.append({"surface": sf, "style": s, "t3": float(f3.params[k]), "t3_se": float(f3.bse[k]), "t3_p": float(f3.pvalues[k]),
                         "w": float(fw.params[k]), "w_se": float(fw.bse[k]), "w_p": float(fw.pvalues[k])})
    for key in ("t3_p", "w_p"):
        for r, ph in zip(rows, holm([r[key] for r in rows])):
            r[key + "_holm"] = ph
    a["adj"] = rows
    a["n_runs"], a["n_pre"] = len(d), len(pre)
    a["c_now"] = by_surface_chart(now, "style_now", NOW_STYLES, "top3", "今走の脚質別 複勝率（芝 / ダート）", "今走の位置取り（結果）", "複勝率",
                                  "逃げ=初角で先頭、先行=前から 25% 以内、差し=65% 以内、追込=それより後ろ。レースの結果としての位置なので、予想には使えない。縦線は 95% 信頼区間。")
    a["c_pre"] = by_surface_chart(pre, "style_pre", PRE_STYLES, "top3", "事前の脚質別 複勝率（芝 / ダート）", "前走までの位置取り", "複勝率",
                                  "前走まで直近 3 走の初角の位置の平均で分けたもの（予想の時点で分かる）。縦線は 95% 信頼区間。")
    a["c_imp"] = implied_chart(pre)
    return a


def article(a: dict[str, Any]) -> dict[str, Any]:
    N, P = a["now"], a["pre"]
    adj = {(r["surface"], r["style"]): r for r in a["adj"]}
    dfront = adj[("ダート", "前寄り")]
    tfront = adj[("芝", "前寄り")]
    adj_rows = [[f"{r['surface']} {r['style']}", f"{P[(r['surface'], r['style'])]['n']:,}",
                 f"{r['t3'] * 100:+.2f}pt（±{1.96 * r['t3_se'] * 100:.2f}）", pfmt(r["t3_p_holm"]),
                 f"{r['w'] * 100:+.2f}pt（±{1.96 * r['w_se'] * 100:.2f}）", pfmt(r["w_p_holm"])] for r in a["adj"]]
    pre_rows = [[f"{sf} {s}", f"{P[(sf, s)]['n']:,}", f"{P[(sf, s)]['popularity']:.1f}", pct(P[(sf, s)]["top3"]), pct(P[(sf, s)]["win"]),
                 pct(P[(sf, s)]["implied"]), f"{pct(P[(sf, s)]['ret'])}（{pct(P[(sf, s)]['roi_ci'][0])}〜{pct(P[(sf, s)]['roi_ci'][1])}）"]
                for sf in SURFACES for s in PRE_STYLES]
    neg = [r for r in a["adj"] if r["t3"] < 0 and r["t3_p_holm"] < 0.05]
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "今走で逃げた馬の複勝率", "value": f"{pct(N[('芝', '逃げ')])} / {pct(N[('ダート', '逃げ')])}",
             "sub": f"芝 / ダート。追込は {pct(N[('芝', '追込')])} / {pct(N[('ダート', '追込')])}（結果としての位置）"},
            {"label": "前走まで前寄りの馬の複勝率", "value": f"{pct(P[('芝', '前寄り')]['top3'])} / {pct(P[('ダート', '前寄り')]['top3'])}",
             "sub": f"芝 / ダート。後方型は {pct(P[('芝', '後方')]['top3'])} / {pct(P[('ダート', '後方')]['top3'])}"},
            {"label": "オッズをそろえたダートの前寄り", "value": f"{dfront['t3'] * 100:+.1f}pt", "sub": f"同じ単勝オッズの中団型と比べた複勝率（補正後 {pfmt(dfront['t3_p_holm'])}）。勝つか崩れるか"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"逃げ・先行は、はっきり有利です。今走で逃げた馬の複勝率は芝 {pct(N[('芝', '逃げ')])}・ダート {pct(N[('ダート', '逃げ')])} で、追込（{pct(N[('芝', '追込')])}・{pct(N[('ダート', '追込')])}）の 3〜4 倍あります。"
            f"前走までの位置取りで分けても、前寄りの馬は芝 {pct(P[('芝', '前寄り')]['top3'])}・ダート {pct(P[('ダート', '前寄り')]['top3'])} と、後方型（{pct(P[('芝', '後方')]['top3'])}・{pct(P[('ダート', '後方')]['top3'])}）を大きく上回ります。\n\n"
            "ただし、馬券で「前に行く馬を買えば得」とは言えません。前に行けそうな馬はすでに人気で、オッズから見込まれる勝率は実際の勝率とほぼ同じです。"
            f"同じ単勝オッズの馬どうしで比べると、ダートの前寄りは中団型より複勝率が {dfront['t3'] * 100:+.1f}pt（補正後 {pfmt(dfront['t3_p_holm'])}）低く、"
            "「勝つか、崩れて 3 着も外すか」になりやすい型です。単勝の評価に比べて複勝圏には残りにくいので、複勝・ワイドでは割り引いて見る必要があります。"
            f"芝の前寄りは勝率で {tfront['w'] * 100:+.1f}pt（補正後 {pfmt(tfront['w_p_holm'])}、有意ではない）、複勝率で {tfront['t3'] * 100:+.1f}pt です。")},
        a["c_now"],
        {"type": "text", "text": (
            "今走の位置取りで分けた差は大きいものの、これはレースの結果です。強い馬は楽に前に行け、行き脚がつかなかった馬は後ろになるので、"
            "「前にいたから勝った」と「勝つ力があったから前にいた」を区別できません。予想に使えるのは、次の「前走までの位置取り」です。")},
        {"type": "heading", "text": "予想に使える脚質（前走までの位置取り）"},
        a["c_pre"],
        a["c_imp"],
        {"type": "table", "columns": ["条件", "出走数", "平均人気", "複勝率", "勝率", "オッズから見込まれる勝率", "単勝回収率（95%区間）"], "rows": pre_rows,
         "caption": "前走まで直近 3 走の初角の位置で分けたもの。単勝回収率は人気馬ほど高くなりやすい（人気薄が買われすぎる傾向）ので、脚質の比較は下のオッズをそろえた差で見る。"},
        {"type": "table", "columns": ["条件", "出走数", "複勝率の差", "補正後の p", "勝率の差", "補正後の p"], "rows": adj_rows,
         "caption": "中団型との差。オッズ（とその 2 乗）・コース（場・芝ダ・距離）をそろえた線形確率モデル。括弧は 95% 区間の幅。6 つの比較を指標ごとに Holm 法で補正。"},
        {"type": "text", "text": (
            "オッズをそろえた差は、どれも数 pt 以内です。前寄りの単勝回収率が高めに見えるのは、前寄りの馬に人気馬が多く、人気馬ほど回収率が高い（人気薄が買われすぎる）ためで、"
            "オッズをそろえると勝率の差はほぼなくなります。"
            + (("同じ単勝オッズでも 3 着以内に残りにくいのは " + "、".join(f"{r['surface']}の{r['style']}（{r['t3'] * 100:+.1f}pt）" for r in neg) + " です。") if neg else "")
            + "後方型は芝・ダートとも、同じオッズの中団型より勝率がわずかに低く（" + "、".join(f"{r['surface']} {r['w'] * 100:+.1f}pt・補正後 {pfmt(r['w_p_holm'])}" for r in a["adj"] if r["style"] == "後方")
            + "）、単勝ではやや買われすぎです。")},
        {"type": "heading", "text": "馬券への活用"},
        {"type": "text", "text": (
            "「逃げ・先行が有利」は正しいものの、オッズにほぼ織り込まれています。差がつくのは「どの馬が本当に前に行けるか」の読みで、"
            "前走までの位置取りだけで前寄りの馬を買っても、回収率は上がりません。ダートの前寄りは単勝の評価のわりに 3 着以内に残りにくいので、複勝・ワイドの軸には向きません。\n\n"
            "ペースによる違い（ハイペースで前が崩れるか）は、ペースのコラムで扱っています。")},
        {"type": "link", "href": "/column/pace-scenario", "label": "ペースで変わる展開シナリオ"},
        {"type": "accordion", "title": "データと方法・以前の版からの変更", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地（障害を除く）。取消・除外は除き、中止・失格は 3 着以内でないとして数えています。"
                f"事前の脚質は、前走（JRA 平地、2016 年から）のある出走だけです（{a['n_pre']:,} 走）。\n\n"
                "・以前の版は、騎手の過去の平均の通過順で「先行型騎手」「差し型騎手」に分けていました。上位の騎手ほど強い馬に乗り、強い馬は前に行けるので、"
                "騎手の腕・馬の強さと脚質の効果が区別できず、「逃げ・先行が有利」の根拠になりません。馬ごとの位置取りで集計し直しました。\n\n"
                "・「内回りが多い」「スローが多い」といった理由の説明は、このデータでは確かめていないので外しました。")},
        ]},
    ]
    return {
        "slug": "front-runner",
        "title": "逃げ・先行は本当に有利——ただし、前に行けそうな馬はすでに人気",
        "claim": "逃げ馬が有利",
        "verdict": "supported",
        "summary": (f"今走で逃げた馬の複勝率は芝 {pct(N[('芝', '逃げ')])}・ダート {pct(N[('ダート', '逃げ')])} で、追込の 3〜4 倍。前走までの位置取りで見ても前寄りが有利です。"
                    f"ただしオッズにほぼ織り込まれていて、ダートの前寄りは同じ単勝オッズの馬より 3 着以内に残りにくい（{dfront['t3'] * 100:+.1f}pt）です。"),
        "categories": ["style/front-runner"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年", "conditions": "JRA全競馬場・芝ダート（障害除く）。脚質は馬ごとの初角の位置取り",
                    "sample": f"{a['n_runs']:,} 出走（事前の脚質があるのは {a['n_pre']:,} 走）"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    col = article(analyze(frame()))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
