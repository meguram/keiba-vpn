"""めぐコラム『初ダート』を実データから集計し、frontend/content/columns/first-dirt.json を書く。

問い: 芝からダートに替わった初戦（初ダート）は、どんな条件なら走るのか。人気・枠・脚質で違うか。市場は織り込んでいるか。

データ: compare_series.load()（2016〜2025 年の JRA 平地。取消・除外を除く）と race_context（枠・脚質）。
定義:
  初ダート = 今走ダートで、前走（JRA 平地の直前の出走）が芝。比較の「通常」= 今走ダートで、前走もダート。
  ※ 2015 年以前や地方でダートを走った馬も、前走が芝なら「初ダート」に入る（厳密には「芝からのダート替わり」）。
  脚質 = 前走まで直近 3 走の初角の位置の平均（race_context.PRE_STYLES）。初ダートの馬では、ほぼ芝のレースでの位置取り。
  枠 = 枠番、および馬番の相対位置（内・中・外）。
比べ方: 生の複勝率は「初ダートの馬は人気薄が多い」ことが混ざるので、オッズ（log とその 2 乗）とコース（場・距離）をそろえた差も出す。

使い方:
    python -m src.research.columns.first_dirt            # 集計して記事を書く
    python -m src.research.columns.first_dirt --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import POP_BANDS, YEARS, load, pairs
from src.research.columns.race_context import GATE_POS, PRE_STYLES, WAKU, joint_p, ols_fit, pair_chart, with_context
from src.research.columns.wind_venue import holm, pct, pfmt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/first-dirt.json"
A, B = "初ダート", "通常（前走もダート）"
POP_FILTER = {"人気帯": {"全体": None, **{b: (lambda d, b=b: d["pop_band"].astype(str) == b) for b in POP_BANDS}}}
ADJ = "lq + I(lq**2) + C(key)"


def frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    d = with_context(load() if df is None else df)
    x = d[(d["surface"] == "ダート") & d["prev_surface"].notna()].copy()
    x["fd"] = (x["prev_surface"] == "芝").astype(float)
    x["lq"] = np.log(x["odds"])
    x["key"] = x["venue"] + x["distance"].astype(str)
    return x


def adj_rows(x: pd.DataFrame, col: str, levels: list[str]) -> list[dict[str, Any]]:
    """区分ごとに、オッズ・コースをそろえた 初ダート − 通常 の複勝率の差（Holm 補正つき）。"""
    rows = []
    for lv in levels:
        w = x[x[col].astype(str) == lv]
        f = ols_fit(f"top3 ~ fd + {ADJ}", w)
        rows.append({"lv": lv, "n_a": int(w["fd"].sum()), "n_b": int((1 - w["fd"]).sum()), "a": w.loc[w["fd"] == 1, "top3"].mean(),
                     "b": w.loc[w["fd"] == 0, "top3"].mean(), "e": float(f.params["fd"]), "se": float(f.bse["fd"]), "p": float(f.pvalues["fd"])})
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
    return rows


def analyze(x: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    fd, nm = x[x["fd"] == 1], x[x["fd"] == 0]
    a["n"] = (len(fd), len(nm))
    a["all"] = {k: (fd[k].mean(), nm[k].mean()) for k in ("top3", "win", "implied", "ret")}
    f = ols_fit(f"top3 ~ fd + {ADJ}", x)
    a["adj_top3"] = (float(f.params["fd"]), float(f.bse["fd"]), float(f.pvalues["fd"]))
    f = ols_fit(f"win ~ fd + {ADJ}", x)
    a["adj_win"] = (float(f.params["fd"]), float(f.bse["fd"]), float(f.pvalues["fd"]))
    a["pop"] = [{"band": b, "n": int((fd["pop_band"].astype(str) == b).sum()),
                 **{k: fd.loc[fd["pop_band"].astype(str) == b, k].mean() for k in ("top3", "win", "implied", "ret")},
                 "top3_b": nm.loc[nm["pop_band"].astype(str) == b, "top3"].mean()} for b in POP_BANDS]
    a["c_pop"] = next(p for p in pairs(df) if p.slug == "first-dirt").compute(df)
    a["c_waku"] = pair_chart(x, x["fd"] == 1, "waku", WAKU, "top3", A, B, title="枠番別 複勝率（初ダート vs 通常）", x_label="枠番", y_label="複勝率",
                             caption="ダートの出走。人気帯を切り替えると、同じ人気帯どうしで比べられる。縦線は 95% 信頼区間、＊は差が有意（p<0.05）。",
                             filters=POP_FILTER)
    a["c_style"] = pair_chart(x[x["style_pre"].notna()], (x["fd"] == 1)[x["style_pre"].notna()], "style_pre", PRE_STYLES, "top3", A, B,
                              title="脚質（前走までの位置取り）別 複勝率（初ダート vs 通常）", x_label="前走まで直近3走の初角の位置", y_label="複勝率",
                              caption="脚質は、前走まで直近 3 走の最初のコーナーの位置の平均（今回の位置取りは使わない）。初ダートの馬は芝での位置取り。人気帯を切り替えられる。＊は差が有意（p<0.05）。",
                              filters=POP_FILTER)
    a["gate"] = adj_rows(x, "gate_pos", GATE_POS)
    xs = x[x["style_pre"].notna()]
    a["style"] = adj_rows(xs, "style_pre", PRE_STYLES)
    a["gate_int"] = joint_p(ols_fit(f"top3 ~ fd*C(gate_pos) + {ADJ}", x), "fd:")
    a["style_int"] = joint_p(ols_fit(f"top3 ~ fd*C(style_pre) + {ADJ}", xs), "fd:")
    a["no_style"] = (int(x.loc[x["fd"] == 1, "style_pre"].isna().sum()), int(x.loc[x["fd"] == 0, "style_pre"].isna().sum()))
    # 結果としての脚質（今走の初角の位置）。予想には使えないが、初ダートの馬が実際どこを走ったか
    now = x[x["style_now"].notna()]
    a["now"] = [{"s": s, "share_a": (now.loc[now["fd"] == 1, "style_now"] == s).mean(), "share_b": (now.loc[now["fd"] == 0, "style_now"] == s).mean(),
                 "top3_a": now.loc[(now["fd"] == 1) & (now["style_now"] == s), "top3"].mean(),
                 "top3_b": now.loc[(now["fd"] == 0) & (now["style_now"] == s), "top3"].mean()} for s in ("逃げ", "先行", "差し", "追込")]
    return a


def _adj_table(rows: list[dict[str, Any]], head: str) -> list[list[str]]:
    return [[r["lv"].replace("\n", ""), f"{r['n_a']:,}", pct(r["a"]), pct(r["b"]), f"{r['e'] * 100:+.1f}pt", f"±{1.96 * r['se'] * 100:.1f}pt",
             pfmt(r["p"])[2:] if r["p"] >= 0.001 else "<0.001", pfmt(r["p_holm"])[2:] if r["p_holm"] >= 0.001 else "<0.001"] for r in rows]


def article(a: dict[str, Any]) -> dict[str, Any]:
    t3a, t3b = a["all"]["top3"]
    wa, wb = a["all"]["win"]
    ia, _ = a["all"]["implied"]
    ra, rb = a["all"]["ret"]
    e3, _, p3 = a["adj_top3"]
    ew, _, pw = a["adj_win"]
    P = {r["band"]: r for r in a["pop"]}
    top, hole = P[POP_BANDS[0]], P[POP_BANDS[-1]]
    G = {r["lv"]: r for r in a["gate"]}
    S = {r["lv"]: r for r in a["style"]}
    outer, inner = G[GATE_POS[2]], G[GATE_POS[0]]
    cols = ["区分", "初ダートの出走数", "初ダート 複勝率", "通常 複勝率", "差（オッズ・コース調整）", "95%区間の幅", "p 値", "補正後の p 値"]
    W = {s["name"]: s["values"] for s in a["c_waku"]["series"]}
    W1, W8 = (W[A][0], W[B][0]), (W[A][-1], W[B][-1])
    pop_rows = [[r["band"], f"{r['n']:,}", pct(r["top3"]), pct(r["top3_b"]), pct(r["win"]), pct(r["implied"]), pct(r["ret"])] for r in a["pop"]]
    now_rows = [[r["s"], pct(r["share_a"]), pct(r["share_b"]), pct(r["top3_a"]), pct(r["top3_b"])] for r in a["now"]]
    nowd = {r["s"]: r for r in a["now"]}
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "初ダートの複勝率", "value": pct(t3a), "sub": f"通常（前走もダート）{pct(t3b)}。人気薄が多いため"},
            {"label": "1〜3番人気の初ダート 複勝率", "value": pct(top["top3"]), "sub": f"通常の 1〜3番人気は {pct(top['top3_b'])}"},
            {"label": "オッズ・コースをそろえた差", "value": f"{e3 * 100:+.1f}pt", "sub": f"複勝率（{pfmt(p3)}）。勝率は {ew * 100:+.1f}pt（{pfmt(pw)}）"},
        ]},
        {"type": "heading", "text": "結論"},
        {"type": "text", "text": (
            f"初ダートの馬の複勝率は {pct(t3a)} で、前走もダートだった馬（{pct(t3b)}）より低くなります。ただし初ダートの馬は人気薄が多く、"
            f"オッズから見込まれる勝率も {pct(ia)} と低めです。オッズとコースをそろえると、複勝率の差は {e3 * 100:+.1f}pt（{pfmt(p3)}）に縮み、"
            f"勝率ではむしろ {ew * 100:+.1f}pt（{pfmt(pw)}）とわずかに上回ります。2〜3 着に入る力より、勝ち切るかどうかの振れが大きい条件なのかもしれません。\n\n"
            f"人気別では、1〜3番人気の初ダートは複勝率 {pct(top['top3'])} と高いものの、同じ人気帯の通常の馬（{pct(top['top3_b'])}）には届きません。"
            f"7番人気以下は {pct(hole['top3'])}（通常 {pct(hole['top3_b'])}）です。単勝回収率は初ダート全体で {pct(ra)}（通常 {pct(rb)}）で、"
            "市場はおおむね正しく評価していると考えられます。")},
        a["c_pop"],
        {"type": "heading", "text": "人気帯ごとの成績とオッズの見込み"},
        {"type": "table", "columns": ["人気帯", "初ダートの出走数", "初ダート 複勝率", "通常 複勝率", "初ダート 勝率", "オッズから見込まれる勝率", "単勝回収率"], "rows": pop_rows,
         "caption": "オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。単勝回収率は 1 走あたりの払戻 ÷ 100 円。"},
        {"type": "heading", "text": "枠番で違うか"},
        {"type": "text", "text": (
            "「初ダートは砂をかぶらない外枠がよい」とよく言われます。枠番ごとに見ると、初ダートの複勝率は 1枠 "
            f"{pct(W1[0])} から 8枠 {pct(W8[0])} へと外ほど高くなりますが、通常の馬も {pct(W1[1])} から {pct(W8[1])} へ同じように上がっていて、"
            "初ダートだけ外枠が特によいという形にはなっていません。\n\n"
            f"頭数の影響を除くため、馬番を頭数で割った位置（内・中・外）に分け、オッズとコースをそろえて比べると、初ダート − 通常 の複勝率の差は"
            f"内 {inner['e'] * 100:+.1f}pt、外 {outer['e'] * 100:+.1f}pt でした（枠の位置で差が変わるかの検定は {pfmt(a['gate_int'])}）。"
            + ("外のほうが初ダートの割引がやや大きく、外枠の初ダートは「砂をかぶらない」と評価されやすい分、オッズほどは走っていない可能性があります。"
               if outer["e"] < inner["e"] and a["gate_int"] < 0.05 else "枠の位置による違いははっきりしません。"))},
        a["c_waku"],
        {"type": "table", "columns": cols, "rows": _adj_table(a["gate"], "枠"),
         "caption": "馬番の位置 =（馬番−1）÷（登録頭数−1）。内は前 1/3、外は後 1/3。差は オッズ（とその 2 乗）・コース（場・距離）をそろえた線形確率モデルの係数で、3 区分を同時に見ているので Holm 法で補正した p 値も示す。"},
        {"type": "heading", "text": "脚質で違うか"},
        {"type": "text", "text": (
            "脚質は、予想の時点で分かるよう、前走まで直近 3 走の最初のコーナーの位置から作りました（初ダートの馬では、ほとんどが芝での位置取りです）。"
            "生の複勝率はどの脚質でも初ダートのほうが低く、前に行く馬ほど成績がよいのは両者に共通です。\n\n"
            "オッズとコースをそろえた差は、"
            + "、".join(f"{r['lv']} {r['e'] * 100:+.1f}pt" for r in a["style"])
            + f"で、脚質で差が変わるかの検定は {pfmt(a['style_int'])} です。"
            + (f"これまで前で競馬をしてきた馬ほど、初ダートではオッズの見込みより走れていない傾向があります（後方は {S['後方']['e'] * 100:+.1f}pt と小さい）。"
               "ただ、脚質による違いは境目の有意性で、確かな差とまでは言えません。"
               if a["style_int"] < 0.05 else "脚質による違いははっきりせず、脚質で初ダートの評価を変える根拠は弱いと考えています。"))},
        a["c_style"],
        {"type": "table", "columns": cols, "rows": _adj_table(a["style"], "脚質"),
         "caption": f"脚質は前走まで直近 3 走の初角の位置の平均（≤0.15 前寄り・≤0.4 好位・≤0.7 中団・それより後ろ 後方）。通過順の記録がない出走（初ダート {a['no_style'][0]:,} 走）は除く。"},
        {"type": "accordion", "title": "参考: 今回の位置取り（結果としての脚質）", "blocks": [
            {"type": "text", "text": (
                "今回のレースで実際に最初のコーナーをどこで回ったか（結果としての脚質）で分けた参考値です。レースが終わってから分かる値なので、予想にはそのまま使えません。\n\n"
                f"初ダートの馬は、逃げ・先行の割合が {pct(nowd['逃げ']['share_a'] + nowd['先行']['share_a'])}（通常 {pct(nowd['逃げ']['share_b'] + nowd['先行']['share_b'])}）と少なく、"
                f"後方になりやすい傾向です（追込 {pct(nowd['追込']['share_a'])}、通常 {pct(nowd['追込']['share_b'])}）。ダートのスピードについていけないと、前に行けません。")},
            {"type": "table", "columns": ["今回の位置", "初ダートの割合", "通常の割合", "初ダート 複勝率", "通常 複勝率"], "rows": now_rows,
             "caption": "今回の初角の位置（（順位−1）÷（頭数−1））で、先頭を逃げ、0.25 までを先行、0.65 までを差し、それより後ろを追込とした。"},
        ]},
        {"type": "accordion", "title": "データと方法・注意点", "blocks": [
            {"type": "text", "text": (
                f"・対象: {min(YEARS)}〜{max(YEARS)}年の JRA 平地のダートの出走で、前走（JRA 平地）のあるもの。取消・除外は除き、中止・失格は 3 着以内でないとして数えています。\n\n"
                "・初ダート: 前走が芝で、今回ダート。2015 年以前や地方でダートを走ったことのある馬も含むので、厳密には「芝からのダート替わり」です。\n\n"
                "・オッズをそろえた差: 複勝率（または勝率）を、初ダートかどうか・単勝オッズの対数とその 2 乗・コース（場・距離）で説明する線形確率モデル。"
                "同じレースの出走は独立でないので、標準誤差はレース単位のクラスタ頑健 SE です。差がゼロに近ければ、市場が初ダートを正しく評価していることを意味します。\n\n"
                "・枠番の効果は、コース（スタートから最初のコーナーまでの距離、芝スタートかどうか）で大きく変わります。ここではコース全体の平均で見ています。")},
        ]},
    ]
    return {
        "slug": "first-dirt",
        "title": "初ダートは読めない？——勝率はオッズ並み、外枠・先行の上積みはなし",
        "claim": "初ダートは成績が読めない",
        "verdict": "partial",
        "summary": (f"初ダートの複勝率は {pct(t3a)} と通常（{pct(t3b)}）より低いものの、オッズをそろえると差は {e3 * 100:+.1f}pt、勝率は {ew * 100:+.1f}pt と、市場はおおむね織り込み済みです。"
                    "外枠・前に行く脚質で初ダートが特に走るという傾向は見られませんでした。"),
        "categories": ["style/surface-switch"],
        "published": "2026-10-28",
        "updated": "2026-11-02",
        "dataset": {"period": f"{min(YEARS)}〜{max(YEARS)}年",
                    "conditions": "JRA全競馬場のダート（障害除く）。前走芝→今走ダートを「初ダート」、前走もダートを「通常」として集計",
                    "sample": f"{a['n'][0] + a['n'][1]:,}出走（うち初ダート {a['n'][0]:,}出走）"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    df = load()
    col = article(analyze(frame(df), df))
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(text)
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
