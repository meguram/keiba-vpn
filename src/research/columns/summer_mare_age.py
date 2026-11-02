"""めぐコラム『夏は牝馬』に、年齢別の節（「年齢で違うか」）を実データから集計して書き込む。

問い: 夏（6〜8月）に牝馬が牡馬と互角になる・オッズの見込みより勝つ、という傾向は、年齢（2歳・3歳・4歳・5歳以上）で違うか。

データ: compare_series.load()（2020〜2025 年の JRA 平地。取消・除外を除く）。牝馬と牡馬が一緒に走った混合戦の、牡・牝の出走（セン馬は除く）。
比べ方:
  年齢ごとに、夏と夏以外で 牝・牡 の勝率と、牝馬の実際の勝率とオッズから見込まれる勝率を並べる。
  オッズ（log とその 2 乗）をそろえた線形確率モデルで、年齢ごとの「牝 × 夏」の上積み（Holm 補正）と、
  年齢で上積みが変わるか（牝 × 夏 × 年齢 の交互作用がすべて 0 か、Wald 検定）を見る。
記事の他の部分は手で整えた書式なので、節（見出し「年齢で違うか」から次の見出しの手前まで）だけを文字列で差し替える。

使い方:
    python -m src.research.columns.summer_mare_age            # 集計して記事に書き込む
    python -m src.research.columns.summer_mare_age --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.columns.compare_series import _chart_text, _inline, _object_spans, _p, _r, _rci, _stat, load
from src.research.columns.race_context import joint_p, ols_fit
from src.research.columns.wind_venue import holm, pct, pfmt

REPO = Path(__file__).resolve().parents[3]
PATH = REPO / "frontend/content/columns/summer-mare.json"
HEADING = "年齢で違うか"
BEFORE = "馬券への活用"            # 節を置く場所（この見出しの手前）
AGES = ["2歳", "3歳", "4歳", "5歳以上"]
SEASONS = ["夏（6〜8月）", "夏以外"]


def frame(df: pd.DataFrame | None = None) -> pd.DataFrame:
    d = load() if df is None else df
    x = d[d["sex"].isin(["牝", "牡"]) & d["mixed_sex"]].copy()
    x["mare"] = (x["sex"] == "牝").astype(float)
    x["summer"] = x["month"].isin([6, 7, 8]).astype(float)
    x["season"] = np.where(x["summer"] == 1, SEASONS[0], SEASONS[1])
    x["lq"] = np.log(x["odds"])
    x["ag"] = np.select([x["age"] == 2, x["age"] == 3, x["age"] == 4], AGES[:3], AGES[3])
    return x


def _pair(parts: list[tuple[pd.Series, pd.Series]], names: tuple[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ser = [{"name": n, "values": [], "n": [], "ci": []} for n in names]
    ps = []
    for a, b in parts:
        st = []
        for s, v in zip(ser, (a, b)):
            m, n, ci, var = _stat(v)
            s["values"].append(_r(m)); s["n"].append(n); s["ci"].append(_rci(ci)); st.append((m, var))
        p = _p(st[0][1], st[1][1], st[0][0], st[1][0])
        ps.append(None if p is None else float(f"{p:.3g}"))
    return ser, {"a": names[0], "b": names[1], "p": ps}


def season_chart(x: pd.DataFrame, part, names: tuple[str, str], title: str, caption: str) -> dict[str, Any]:
    """横軸は年齢。季節（先頭 = 夏）をプルダウンで切り替える。"""
    def build(season: str):
        w = x[x["season"] == season]
        return _pair([part(w[w["ag"] == ag]) for ag in AGES], names)
    ser, cmp = build(SEASONS[0])
    s2, c2 = build(SEASONS[1])
    return {"type": "chart", "title": title, "xLabel": "年齢", "yLabel": "勝率", "xs": AGES, "series": ser, "compare": cmp,
            "filters": [{"label": "季節", "options": SEASONS}], "variants": {SEASONS[1]: {"series": s2, "compare": c2}}, "caption": caption}


def analyze(x: pd.DataFrame) -> dict[str, Any]:
    a: dict[str, Any] = {}
    a["tab"] = x.groupby(["ag", "season", "sex"]).agg(n=("win", "size"), win=("win", "mean"), implied=("implied", "mean"), ret=("ret", "mean"))
    rows = []
    for ag in AGES:
        f = ols_fit("win ~ mare*summer + lq + I(lq**2)", x[x["ag"] == ag])
        rows.append({"ag": ag, "e": float(f.params["mare:summer"]), "se": float(f.bse["mare:summer"]), "p": float(f.pvalues["mare:summer"])})
    for r, ph in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = ph
    a["reg"] = rows
    a["p_int"] = joint_p(ols_fit("win ~ mare*summer*C(ag) + lq + I(lq**2)", x), "mare:summer:")
    a["c_sex"] = season_chart(x, lambda w: (w.loc[w["sex"] == "牝", "win"], w.loc[w["sex"] == "牡", "win"]), ("牝", "牡"),
                              "年齢別 牡・牝馬の勝率（混合戦・季節で切り替え）",
                              "混合戦の牡・牝。季節を「夏以外」に切り替えると、夏以外の月の勝率を比べられる。2 歳戦は 6 月に始まるので、2 歳の「夏以外」は 9〜12 月。＊は差が有意（p<0.05）。")
    a["c_odds"] = season_chart(x[x["sex"] == "牝"], lambda w: (w["win"], w["implied"]), ("実際の勝率", "オッズから見込まれる勝率"),
                               "年齢別 牝馬の勝率（実際とオッズの見込み・混合戦）",
                               "混合戦の牝馬。オッズから見込まれる勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。季節で切り替えられる。＊は差が有意（p<0.05）。")
    return a


def blocks(a: dict[str, Any]) -> list[dict[str, Any]]:
    t = a["tab"]

    def g(ag: str, season: str, sex: str, k: str) -> float:
        return float(t.loc[(ag, season, sex), k])

    rows = []
    for ag in AGES:
        for s in SEASONS:
            rows.append([ag, s, f"{int(g(ag, s, '牝', 'n')):,}", pct(g(ag, s, "牝", "win")), pct(g(ag, s, "牡", "win")),
                         f"{(g(ag, s, '牡', 'win') - g(ag, s, '牝', 'win')) * 100:+.1f}pt", pct(g(ag, s, "牝", "implied")), pct(g(ag, s, "牝", "ret"))])
    reg_rows = [[r["ag"], f"{r['e'] * 100:+.2f}pt", f"±{1.96 * r['se'] * 100:.2f}pt", pfmt(r["p"])[2:] if r["p"] >= 0.001 else "<0.001",
                 pfmt(r["p_holm"])[2:] if r["p_holm"] >= 0.001 else "<0.001"] for r in a["reg"]]
    gaps = {ag: [(g(ag, s, "牡", "win") - g(ag, s, "牝", "win")) * 100 for s in SEASONS] for ag in AGES}
    dm = {ag: (g(ag, SEASONS[0], "牝", "win") - g(ag, SEASONS[1], "牝", "win")) * 100 for ag in AGES}
    dc = {ag: (g(ag, SEASONS[0], "牡", "win") - g(ag, SEASONS[1], "牡", "win")) * 100 for ag in AGES}
    lo, hi = min(r["e"] for r in a["reg"]) * 100, max(r["e"] for r in a["reg"]) * 100
    all_sig = all(r["p_holm"] < 0.05 for r in a["reg"])
    text1 = (
        "夏の牝馬の強さは、若い馬の成長（2〜3 歳の牝馬が夏に追いつく）だけで説明できるのかを確かめるため、年齢ごとに分けて見ました。\n\n"
        "勝率の牡−牝の差は、夏以外では "
        + "、".join(f"{ag} {gaps[ag][1]:+.1f}pt" for ag in AGES)
        + " と、どの年齢でも牡馬が上です。夏になると "
        + "、".join(f"{ag} {gaps[ag][0]:+.1f}pt" for ag in AGES)
        + " と、どの年齢でもほぼ互角になります。\n\n"
        "ただし、差が縮まる形は年齢で違います。夏以外から夏にかけての勝率の変化は、"
        + "、".join(f"{ag} 牝 {dm[ag]:+.1f}pt・牡 {dc[ag]:+.1f}pt" for ag in AGES)
        + "で、"
        + ("2〜3 歳は牝馬の勝率が上がり、4 歳は牡馬の勝率が下がることで、5 歳以上はその両方で差が縮まっています。"
           if dm["2歳"] > 1 and dm["3歳"] > 1 and dc["4歳"] < -1 and abs(dm["4歳"]) < 0.5 and dm["5歳以上"] > 0.5 and dc["5歳以上"] < -0.5
           else "年齢によって、牝馬が上がるのか牡馬が下がるのかが違います。")
        + "夏の古馬戦は相手関係（クラスや出走する馬の顔ぶれ）も変わるので、牡馬の勝率が下がる理由はこのデータだけでは分かりません。")
    text2 = (
        f"オッズをそろえた「夏の牝馬の上積み」（牝 × 夏）は、年齢ごとに {lo:+.1f}〜{hi:+.1f}pt で、"
        + ("4 区分とも多重比較を補正しても有意です。" if all_sig else "区分によっては補正後に有意でないものもあります。")
        + f"年齢で上積みが変わるかの検定は {pfmt(a['p_int'])} で、年齢による違いは見られません。"
        + ("夏の牝馬が見込みより勝つ傾向は、2 歳から古馬まで共通と考えられます。" if a["p_int"] >= 0.05 else "年齢によって上積みの大きさが違う可能性があります。")
        + "\n\nなお 3 歳は、春は 3 歳限定戦、夏からは古馬との混合戦が増えるので、季節の比較には相手の違いも混ざります。")
    return [
        {"type": "heading", "text": HEADING},
        {"type": "text", "text": text1},
        a["c_sex"],
        a["c_odds"],
        {"type": "table", "columns": ["年齢", "季節", "牝馬の出走数", "牝 勝率", "牡 勝率", "牡−牝", "牝 見込み勝率", "牝 単勝回収率"], "rows": rows,
         "caption": "混合戦の牡・牝（セン馬を除く）。見込み勝率は 1/単勝オッズ をレース内で合計 1 に直したもの。"},
        {"type": "text", "text": text2},
        {"type": "table", "columns": ["年齢", "牝×夏の上積み（勝率・オッズ調整）", "95%区間の幅", "p 値", "補正後の p 値"], "rows": reg_rows,
         "caption": "年齢ごとに、勝ちを 牝・夏・牝×夏・オッズ（log とその 2 乗）で説明する線形確率モデル（レース単位のクラスタ頑健 SE）。4 区分を同時に見ているので Holm 法で補正した p 値も示す。"},
    ]


def _block_text(b: dict[str, Any], ind: str) -> str:
    return _chart_text(b, ind) if b["type"] == "chart" else ("{\n" + ",\n".join(f"{ind}  {json.dumps(k, ensure_ascii=False)}: {_inline(v)}" for k, v in b.items()) + "\n" + ind + "}"
                                                              if b["type"] == "table" else _inline(b))


def splice(text: str, new: list[dict[str, Any]]) -> str:
    """見出し HEADING の節を new に置き換える（無ければ見出し BEFORE の手前に足す）。"""
    heads = {}
    for s, e in _object_spans(text):
        try:
            o = json.loads(text[s:e])
        except ValueError:
            continue
        if isinstance(o, dict) and o.get("type") == "heading":
            heads[o["text"]] = s
    if BEFORE not in heads:
        raise SystemExit(f"summer-mare: 見出し『{BEFORE}』が見つかりません")
    end = heads[BEFORE]
    start = heads.get(HEADING, end)
    ind = text[text.rfind("\n", 0, end) + 1:end]
    body = (",\n" + ind).join(_block_text(b, ind) for b in new) + ",\n" + ind
    out = text[:start] + body + text[end:]
    json.loads(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    a = analyze(frame())
    new = blocks(a)
    out = splice(PATH.read_text(encoding="utf-8"), new)
    if args.dry_run:
        print(json.dumps(new, ensure_ascii=False, indent=1)[:6000])
    else:
        PATH.write_text(out, encoding="utf-8")
        print(f"wrote {PATH}")


if __name__ == "__main__":
    main()
