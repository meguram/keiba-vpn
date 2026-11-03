"""めぐコラム『乗り替わりの効果』を実データから集計し、frontend/content/columns/jockey-change.json を書く。

騎手のランク S〜D:
  騎手ごとの複勝率を経験ベイズで縮め（jockey_common.ranks）、期間に RANK_MIN_RIDES 騎乗以上の騎手を
  上位 5% = S、〜25% = A、〜75% = B（真ん中の半分。全騎手の平均くらい）、〜95% = C、残り = D に分ける。
  5% / 20% / 50% / 20% / 5% の左右対称の切り方にして、B が「平均的な騎手」、S と D がそれぞれ両端の 1 割の半分になるようにした。

乗り替わりのヒートマップ:
  同じ馬の前走（JRA 平地）の騎手 → 今走の騎手 が違う出走を、ランク × ランクで集計する。
  ランクを決めるのに使った騎乗と、結果を見る騎乗が同じだと、ランクと結果が作りで相関するので、
  結果を見る年 Y のランクは Y 以外の年の騎乗から決める（1 年抜きの交差。騎乗の下限は LOYO_MIN_RIDES）。
  値は 複勝率 と「実際 − 人気（単勝オッズ）から見込まれる複勝率」。後者が 0 なら、乗り替わりは人気どおり。

使い方:
    python -m src.research.columns.jockey_change            # 集計して記事を書く
    python -m src.research.columns.jockey_change --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.columns import compare_series as cs
from src.research.columns import jockey_common as jc

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/jockey-change.json"

RANK_MIN_RIDES = 300
LOYO_MIN_RIDES = 250
HEAT_MIN_N = 100
RANK_DESC = {"S": "上位 5%", "A": "上位 5〜25%", "B": "真ん中の半分（25〜75%）", "C": "下位 5〜25%", "D": "下位 5%"}


def loyo_ranks(d: pd.DataFrame) -> pd.DataFrame:
    """出走ごとに、その年を除いた年の騎乗から決めた 今走の騎手のランク（rank_now）と前走の騎手のランク（rank_prev）。"""
    out = []
    for y in sorted(d["year"].unique()):
        r = jc.ranks(d[d["year"] != y], LOYO_MIN_RIDES)["rank"]
        x = d.loc[d["year"] == y, ["jockey_id", "prev_jockey"]]
        out.append(pd.DataFrame({"rank_now": x["jockey_id"].map(r), "rank_prev": x["prev_jockey"].map(r)}, index=x.index))
    return pd.concat(out).reindex(d.index)


def heat(c: pd.DataFrame) -> dict[str, Any]:
    vals_d, vals_r, ns, cis_d = [], [], [], []
    for a in jc.RANKS:
        rd, rr, rn, rc = [], [], [], []
        for b in jc.RANKS:
            x = jc.cell(c[(c["rank_prev"] == a) & (c["rank_now"] == b)], "top3")
            rd.append(x["diff"]); rr.append(x["rate"]); rn.append(x["n"]); rc.append(x["diff_ci"])
        vals_d.append(rd); vals_r.append(rr); ns.append(rn); cis_d.append(rc)
    return {"diff": vals_d, "rate": vals_r, "n": ns, "ci": cis_d}


def direction(c: pd.DataFrame) -> list[dict[str, Any]]:
    order = {k: i for i, k in enumerate(jc.RANKS)}
    step = c["rank_prev"].map(order) - c["rank_now"].map(order)       # 正 = 上のランクへ
    groups = [("2 ランク以上 上へ", step >= 2), ("1 ランク上へ", step == 1), ("同じランク", step == 0),
              ("1 ランク下へ", step == -1), ("2 ランク以上 下へ", step <= -2)]
    return [{"name": nm, **jc.cell(c[m], "top3"), "pop": float(c.loc[m, "popularity"].mean())} for nm, m in groups]


def analyze(d: pd.DataFrame) -> dict[str, Any]:
    names = jc.jockey_names(d)
    R = jc.ranks(d, RANK_MIN_RIDES)
    R["res"] = d.groupby("jockey_id")["res_top3"].mean()
    R["name"] = R.index.map(names)
    d = d.join(loyo_ranks(d))
    hp = d["prev_jockey"].notna()
    ch = hp & (d["jockey_id"] != d["prev_jockey"])
    c = d[ch & d["rank_now"].notna() & d["rank_prev"].notna()]
    cont = d[hp & ~ch & d["rank_now"].notna()]
    a: dict[str, Any] = {
        "names": names, "R": R, "n_runs": len(d), "n_prev": int(hp.sum()),
        "change": jc.cell(d[ch], "top3"), "cont": jc.cell(d[hp & ~ch], "top3"),
        "change_pop": float(d.loc[ch, "popularity"].mean()), "cont_pop": float(d.loc[hp & ~ch, "popularity"].mean()),
        "heat": heat(c), "n_heat": len(c), "dir": direction(c),
        "cont_by_rank": {k: jc.cell(cont[cont["rank_now"] == k], "top3") for k in jc.RANKS},
        "to_by_rank": {k: jc.cell(c[c["rank_now"] == k], "top3") for k in jc.RANKS},
        "cuts": R.groupby("rank")["eb"].agg(["min", "max", "size"]).reindex(jc.RANKS),
        "avg_top3": float(d["top3"].mean()),
    }
    a["robust"] = robustness(c)
    return a


def robustness(c: pd.DataFrame, boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """2 ランク以上の格下げの「実際 − 見込み」が、期間を分けても・同じ馬の騎乗をまとめても残るか。"""
    import numpy as np

    order = {k: i for i, k in enumerate(jc.RANKS)}
    step = c["rank_prev"].map(order) - c["rank_now"].map(order)
    down2 = c[step <= -2]
    halves = {nm: float(down2.loc[m, "res_top3"].mean()) for nm, m in
              (("2020〜2022年", down2["year"] <= 2022), ("2023〜2025年", down2["year"] >= 2023))}
    h = down2.groupby("horse_id")["res_top3"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        s = h.iloc[rng.integers(0, len(h), len(h))]
        bs.append(s["sum"].sum() / s["size"].sum())
    return {"halves": halves, "horse_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "horses": len(h),
            "res_win": float(down2["res_win"].mean()), "roi": float(down2["ret"].mean()), "roi_other": float(c.loc[step > -2, "ret"].mean())}


def compare_chart() -> dict[str, Any]:
    """乗り替わりの有無 × 人気帯（compare_series の定義をそのまま使う。compare_series を流し直しても同じものになる）。"""
    df = cs.load()
    pair = next(p for p in cs.pairs(df) if p.slug == "jockey-change")
    return pair.compute(df) | {"title": pair.title}


def rank_lists(a: dict[str, Any]) -> list[dict[str, Any]]:
    R, names = a["R"], a["names"]
    blocks = []
    for k in jc.RANKS:
        ids = jc.kana_order(R.index[R["rank"] == k].tolist(), names)
        rows = [[names[j], int(R.at[j, "n"]), jc.pct(R.at[j, "raw"]), jc.pct(R.at[j, "eb"]), jc.pt(R.at[j, "res"])] for j in ids]
        blocks.append({"type": "accordion", "title": f"{k} ランク（{RANK_DESC[k]}・{len(ids)}人）", "blocks": [
            {"type": "table", "columns": ["騎手", "騎乗数", "複勝率", "複勝率（縮めた値）", "実際 − 人気の見込み"], "rows": rows,
             "caption": "50 音順。2020〜2025年 JRA 平地。ランクは「複勝率（縮めた値）」で決めた。"}]})
    return blocks


def article(a: dict[str, Any], cmp_chart: dict[str, Any]) -> dict[str, Any]:
    h, ch, co = a["heat"], a["change"], a["cont"]
    cuts = a["cuts"]
    dirs = {x["name"]: x for x in a["dir"]}
    up2, down2, same = dirs["2 ランク以上 上へ"], dirs["2 ランク以上 下へ"], dirs["同じランク"]
    i = {k: n for n, k in enumerate(jc.RANKS)}
    toS = a["to_by_rank"]["S"]
    toD = a["to_by_rank"]["D"]
    contS = a["cont_by_rank"]["S"]
    sig_cells = [(r, c) for r in jc.RANKS for c in jc.RANKS
                 if h["ci"][i[r]][i[c]] and h["n"][i[r]][i[c]] >= HEAT_MIN_N and (h["ci"][i[r]][i[c]][0] > 0 or h["ci"][i[r]][i[c]][1] < 0)]
    sig_txt = "、".join(f"{r}→{c}（{jc.pt(h['diff'][i[r]][i[c]])}、{h['n'][i[r]][i[c]]:,}走）" for r, c in sig_cells) or "なし"

    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "継続騎乗 複勝率", "value": jc.pct(co["rate"]), "sub": f"人気の見込みとの差 {jc.pt(co['diff'])}"},
            {"label": "乗り替わり 複勝率", "value": jc.pct(ch["rate"]), "sub": f"人気の見込みとの差 {jc.pt(ch['diff'])}"},
            {"label": "S ランクへの乗り替わり", "value": jc.pct(toS["rate"]), "sub": f"見込みとの差 {jc.pt(toS['diff'])}（{toS['n']:,}走）"},
        ]},
        cmp_chart,
        {"type": "text", "text":
            f"前走と騎手が替わった出走の複勝率は {jc.pct(ch['rate'])}、同じ騎手が続けて乗った出走は {jc.pct(co['rate'])} で、"
            f"乗り替わりのほうが {abs((ch['rate'] - co['rate']) * 100):.1f}pt 低くなります。どの人気帯でも乗り替わりのほうが低めです。"
            f"ただし平均人気は乗り替わり {a['change_pop']:.1f} 番人気・継続 {a['cont_pop']:.1f} 番人気で、乗り替わる馬はもともと人気（＝評価）が低めです。"
            f"単勝オッズから見込まれる複勝率との差で比べると、乗り替わり {jc.pt(ch['diff'])}（95%区間 {jc.pt_ci(ch['diff_ci'])}）・"
            f"継続 {jc.pt(co['diff'])}（{jc.pt_ci(co['diff_ci'])}）で、差は {abs((ch['diff'] - co['diff']) * 100):.1f}pt に縮みます。"
            "乗り替わりの「マイナス」の大半は、人気にすでに織り込まれています。"},
        {"type": "heading", "text": "騎手を S〜D の 5 ランクに分ける"},
        {"type": "text", "text":
            f"2020〜2025年に {RANK_MIN_RIDES} 騎乗以上の {int(cuts['size'].sum())} 人を、複勝率で 5 つのランクに分けました。"
            "騎乗の少ない騎手の複勝率は偶然でぶれるので、経験ベイズで全騎手の平均に向けて縮めた値で並べています（騎乗が多いほど縮みは小さい）。\n\n"
            "切り方は 上位 5% = S、5〜25% = A、真ん中の半分 = B、下位 5〜25% = C、下位 5% = D。"
            "左右対称にして、B が「平均的な騎手」、S と D が両端のごく一部になるようにしました。"
            f"境目の複勝率（縮めた値）は、S {jc.pct(cuts.at['S', 'min'])} 以上、A {jc.pct(cuts.at['A', 'min'])}〜、B {jc.pct(cuts.at['B', 'min'])}〜、"
            f"C {jc.pct(cuts.at['C', 'min'])}〜、D はそれ未満です（全出走の複勝率は {jc.pct(a['avg_top3'])}）。\n\n"
            "なお、ランクを「実際 − 人気からの見込み」で決める方法も試しましたが、騎手ごとの差は小さく（市場が騎手の腕をよく織り込んでいる）、"
            "人気を集める上位騎手がむしろ下位に入るため、ランクは成績（複勝率）で決め、人気との差は乗り替わりの結果の側で見ることにしました。"},
        *rank_lists(a),
        {"type": "heading", "text": "どのランクから、どのランクへ"},
        {"type": "heatmap", "title": "前走の騎手ランク → 今走の騎手ランク：複勝率", "rowLabel": "前走", "colLabel": "今走",
         "rows": jc.RANKS, "cols": jc.RANKS, "values": h["rate"], "n": h["n"], "format": "pct", "center": round(a["avg_top3"], 4),
         "minN": HEAT_MIN_N, "legend": ["全体の平均より低い", "高い"],
         "caption": f"乗り替わりのあった {a['n_heat']:,} 走。色の中心は全出走の複勝率 {jc.pct(a['avg_top3'])}。"},
        {"type": "heatmap", "title": "前走の騎手ランク → 今走の騎手ランク：実際 − 人気からの見込み（複勝率）", "rowLabel": "前走", "colLabel": "今走",
         "rows": jc.RANKS, "cols": jc.RANKS, "values": h["diff"], "n": h["n"], "ci": h["ci"], "format": "pt", "center": 0,
         "minN": HEAT_MIN_N, "legend": ["人気より走らない", "人気より走る"],
         "caption": "0 なら人気（単勝オッズ）どおり。マスに触れると 95% 信頼区間が出る。ランクは結果を見る年を除いた年の騎乗から決めた（1 年抜きの交差）。"},
        {"type": "text", "text":
            f"複勝率のヒートマップは、右（今走が上のランク）ほど、上（前走が上のランク）ほど高くなります。"
            f"S ランクへの乗り替わりの複勝率は {jc.pct(toS['rate'])}、D ランクへは {jc.pct(toD['rate'])}。"
            f"ただしこれは、上位騎手に強い馬が集まる・弱い馬が下位騎手に回ることの表れで、騎手の腕の差だけではありません。\n\n"
            f"人気からの見込みとの差で見ると、色はずっと薄くなります。S ランクへの乗り替わりは {jc.pt(toS['diff'])}（{jc.pt_ci(toS['diff_ci'])}）、"
            f"D ランクへは {jc.pt(toD['diff'])}（{jc.pt_ci(toD['diff_ci'])}）。"
            f"95% 区間が 0 をまたがない組（{HEAT_MIN_N} 走以上）は {sig_txt} です。"
            "25 マスを同時に見ているので、1〜2 マスは偶然でも 0 から外れます。目立つのは、上位（S・A）の騎手から 2 つ下のランクへ替わったマスがそろって負になっている点で、"
            "下の「乗り替わりの方向」でまとめると、はっきりした差として残ります。"},
        {"type": "table", "columns": ["乗り替わりの方向", "出走数", "平均人気", "複勝率", "人気からの見込み", "実際 − 見込み（95%区間）"],
         "rows": [[x["name"], x["n"], f"{x['pop']:.1f}", jc.pct(x["rate"]), jc.pct(x["exp"]), f"{jc.pt(x['diff'])}（{jc.pt_ci(x['diff_ci'])}）"] for x in a["dir"]],
         "caption": "ランクの上下で乗り替わりをまとめたもの。"},
        {"type": "heading", "text": "まとめ"},
        {"type": "text", "text":
            f"• 上のランクへの乗り替わりほど複勝率は高い（2 ランク以上 上へ {jc.pct(up2['rate'])}、下へ {jc.pct(down2['rate'])}）。ただし人気も同じだけ動いている。\n"
            f"• 人気からの見込みとの差は、2 ランク以上 上へ {jc.pt(up2['diff'])}（{jc.pt_ci(up2['diff_ci'])}）、同じランク {jc.pt(same['diff'])}、"
            f"2 ランク以上 下へ {jc.pt(down2['diff'])}（{jc.pt_ci(down2['diff_ci'])}）。"
            "上へ・同じ・1 つ下へ の乗り替わりは、ほぼオッズどおり。2 つ以上下のランクへの乗り替わりだけは、人気より走らない。\n"
            f"• 参考: S ランクの騎手が続けて乗るときは 複勝率 {jc.pct(contS['rate'])}・見込みとの差 {jc.pt(contS['diff'])}。\n\n"
            "「誰に替わるか」で成績は大きく変わりますが、その差の大半は人気（オッズ）に反映されています。"
            f"例外は大きな格下げで、上位騎手から 2 つ以上下のランクの騎手へ替わった馬（{down2['n']:,}走、平均 {down2['pop']:.1f} 番人気）は、"
            f"複勝率が見込みより約 {abs(down2['diff']) * 100:.0f}pt 低く出ています。市場は格下げの影響を少し甘く見ている可能性があります。\n\n"
            f"この差は期間を分けても残り（{'、'.join(f'{k} {v * 100:+.1f}pt' for k, v in a['robust']['halves'].items())}）、"
            f"同じ馬の騎乗をまとめて数えた 95% 区間（{a['robust']['horses']:,} 頭）でも {a['robust']['horse_ci'][0] * 100:+.1f}〜{a['robust']['horse_ci'][1] * 100:+.1f}pt です。"
            f"勝率でも見込みより {a['robust']['res_win'] * 100:+.1f}pt、単勝回収率は {a['robust']['roi'] * 100:.1f}%（ほかの乗り替わりは {a['robust']['roi_other'] * 100:.1f}%）でした。"
            "なぜ格下げの馬が人気ほど走らないのか（上位騎手がより良い馬を選んだ、など）は、このデータでは確かめていません。"},
        {"type": "note", "text":
            "人気からの見込みは、単勝オッズから求めた勝率を 50 段階に分け、頭数の帯ごとに実際の複勝率をあてはめたもの（jockey_common.calibrate）。"
            "前走は同じ馬の JRA 平地の前走（障害・地方は含まない）。表の信頼区間は出走を独立とみなしたもの。結論の「2 ランク以上 下へ」は、同じ馬の騎乗をまとめたブートストラップでも区間がほぼ変わらないことを確かめた。"},
    ]
    return {
        "slug": "jockey-change",
        "title": "乗り替わりの効果——騎手ランク S〜D で「どこからどこへ」を見る",
        "claim": "乗り替わりは成績に影響する",
        "verdict": "partial",
        "summary": (f"乗り替わりの複勝率は {jc.pct(ch['rate'])} で継続騎乗（{jc.pct(co['rate'])}）より低く、上のランクの騎手へ替わるほど高くなります。"
                    f"ただし人気からの見込みとの差で見ると、乗り替わり全体は {jc.pt(ch['diff'])} で、多くはオッズに織り込まれています。"
                    f"例外は 2 つ以上下のランクへの乗り替わりで、見込みより {jc.pt(down2['diff'])}。"),
        "categories": ["jockey/change"],
        "published": "2026-10-28",
        "updated": "2026-11-03",
        "dataset": {"period": "2020〜2025年", "conditions": "JRA 平地（障害を除く）・同じ馬の前走との比較。取消・除外を除く",
                    "sample": f"{a['n_runs']:,}出走（前走のある {a['n_prev']:,} 走、うちランク付きの乗り替わり {a['n_heat']:,} 走）"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    a = analyze(jc.frame())
    col = article(a, compare_chart())
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(json.dumps([b for b in col["body"] if b["type"] in ("stats", "text", "heatmap", "table")][:12], ensure_ascii=False, indent=0)[:9000])
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
