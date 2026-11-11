"""厩舎コメントの語のワードクラウド（HTML）を作る。

    python -m src.scripts.data.trainer_comment_wordcloud <term_stats.csv> [--out docs/html/modeling/chapters/trainer_comment_wordcloud.html]

term_stats.csv は trainer_comment_words.py の出力。語の大きさ = 登場回数（コメント数）、色 = 好走寄り（青）／凡走寄り（橙）。
同じ配置で、色を『生の lift』と『人気補正の lift』に切り替えた図を並べる（人気の影響で色がどう消えるかを見比べる）。
静的な SVG だけで作る（スクリプトなし。本 build_docs_book に取り込める）。
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
W, H = 1040, 560
GREY = (148, 163, 184)
GOOD = (79, 139, 255)
BAD = (251, 146, 60)


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def color(lift: float, lim: float = 0.6) -> str:
    """log2(lift) を ±lim で飽和させた発散色。lift>1 は青、<1 は橙。"""
    x = max(-1.0, min(1.0, math.log2(max(lift, 1e-6)) / lim))
    c = _mix(GREY, GOOD if x > 0 else BAD, abs(x))
    return "#%02x%02x%02x" % c


def text_width(s: str, size: float) -> float:
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s) + size * 0.2


def layout(df: pd.DataFrame, smin=15.0, smax=84.0) -> list[dict]:
    n = df["n"].to_numpy(dtype=float)
    lo, hi = math.sqrt(n.min()), math.sqrt(n.max())
    placed: list[tuple[float, float, float, float]] = []
    out = []
    for r in df.sort_values("n", ascending=False).itertuples():
        size = smin + (smax - smin) * (math.sqrt(r.n) - lo) / (hi - lo + 1e-9)
        for _ in range(8):
            w, h = text_width(r.label, size), size * 1.12
            pos = None
            t = 0.0
            while t < 140:
                x = W / 2 + 6.5 * t * math.cos(t) * 1.0
                y = H / 2 + 6.5 * t * math.sin(t) * 0.5
                rect = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                if rect[0] >= 4 and rect[2] <= W - 4 and rect[1] >= 4 and rect[3] <= H - 4 and all(
                    rect[2] + 2 < p[0] or rect[0] - 2 > p[2] or rect[3] + 1 < p[1] or rect[1] - 1 > p[3] for p in placed
                ):
                    pos = (x, y, rect)
                    break
                t += 0.07
            if pos:
                placed.append(pos[2])
                out.append({"row": r, "x": pos[0], "y": pos[1], "size": size})
                break
            size *= 0.9
            if size < 9:
                break
    return out


def svg(items: list[dict], col: str, base: float) -> str:
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" style="width:100%;height:auto;background:#0b1224;border:1px solid #243358;border-radius:8px">']
    for it in items:
        r = it["row"]
        lift = getattr(r, col)
        tip = (f"{r.term}\n登場 {int(r.n):,} コメント / 3着以内 {r.good_rate*100:.1f}%（全体 {base*100:.1f}%）\n"
               f"生の lift {r.lift:.2f}（{r.lift_lo:.2f}〜{r.lift_hi:.2f}）  人気補正 lift {r.adj_lift:.2f}")
        parts.append(
            f'<text x="{it["x"]:.1f}" y="{it["y"]:.1f}" font-size="{it["size"]:.1f}" fill="{color(lift)}" text-anchor="middle" '
            f'dominant-baseline="central" font-weight="{700 if it["size"] > 34 else 500}"><title>{html.escape(tip)}</title>{html.escape(r.label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def legend(left: str, right: str, ticks: list[tuple[float, str]]) -> str:
    stops = ",".join(color(2 ** (x * 0.6)) for x in [i / 10 - 1 for i in range(21)])
    tk = "".join(f'<span style="position:absolute;left:{p*100:.1f}%;transform:translateX(-50%)">{html.escape(t)}</span>' for p, t in ticks)
    return (f'<div style="margin:10px 0 24px;max-width:520px"><div style="display:flex;justify-content:space-between" class="muted"><span>{left}</span><span>{right}</span></div>'
            f'<div style="height:12px;border-radius:6px;background:linear-gradient(90deg,{stops})"></div>'
            f'<div class="muted" style="position:relative;height:18px;font-size:11px">{tk}</div></div>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default=str(ROOT / "docs/html/modeling/chapters/trainer_comment_wordcloud.html"))
    ap.add_argument("--n-uni", type=int, default=110)
    ap.add_argument("--n-bi", type=int, default=70)
    a = ap.parse_args()
    T = pd.read_csv(a.csv)
    summ = json.loads(Path(a.summary).read_text()) if a.summary else {}
    base = float(summ.get("good_rate", T["good"].sum() / T["n"].sum()))
    n_rows = summ.get("n_rows", 0)

    uni = T[(T.kind == "uni") & (T.n >= 150)].nlargest(a.n_uni, "n").copy()
    uni["label"] = uni["term"]
    bi = T[(T.kind == "bi") & (T.n >= 150)].nlargest(a.n_bi, "n").copy()
    bi["label"] = bi["term"].str.replace("+", "＋", regex=False)
    li, lb = layout(uni), layout(bi)

    ticks = [(0.0, "×0.66"), (0.25, "×0.81"), (0.5, "×1.0"), (0.75, "×1.23"), (1.0, "×1.52")]
    leg = legend("凡走寄り（橙）", "好走寄り（青）", ticks)

    def top(df, col, asc, k=8):
        d = df.sort_values(col, ascending=asc).head(k)
        return "".join(f"<tr><td>{html.escape(r.term)}</td><td class='num'>{int(r.n):,}</td><td class='num'>{r.good_rate*100:.1f}%</td><td class='num'>{r.lift:.2f}</td><td class='num'>{r.adj_lift:.2f}</td></tr>" for r in d.itertuples())

    big = T[(T.n >= 300)]
    tbl_head = "<tr><th>語</th><th>登場</th><th>3着以内率</th><th>lift</th><th>人気補正 lift</th></tr>"
    html_out = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>厩舎コメントのワードクラウド — keiba-vpn</title><style>:root {{ --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }}
* {{ box-sizing: border-box; }} body {{ font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }}
a {{ color: var(--accent-2); }} h1 {{ font-size: 26px; }} h2 {{ font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; }}
table {{ border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); }} th, td {{ border: 1px solid var(--border); padding: 5px 10px; text-align: left; }} th {{ background: #1b2a4d; }}
td.num {{ text-align: right; font-family: monospace; }} .muted {{ color: var(--muted); font-size: 13px; }} .note {{ background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }}
.cols {{ display: flex; gap: 28px; flex-wrap: wrap; }}
</style></head><body>
<h1>厩舎コメントのワードクラウド</h1>
<p class="muted">2020〜2025 年の厩舎コメント {n_rows:,} 頭分。語の大きさ = その語が出たコメントの数（平方根で大きさに換算）、色 = その語を含む馬の 3 着以内率が全体（{base*100:.1f}%）の何倍か（lift、対数）。青 = 好走寄り、橙 = 凡走寄り、灰 = 全体並み。語にカーソルを載せると数値が出る。集計の作り方は <code>src/scripts/data/trainer_comment_words.py</code>。</p>
{leg}
<h2>1. 生の lift で色を付けた図（1 語）</h2>
<p>「楽しみ」「期待」「能力」「昇級」「勝負」などは青（好走寄り）、「叩く」「試す」「慣れ」「非力」「展開」などは橙（凡走寄り）。大きい語ほど件数が多く、色の信頼度も高い。</p>
{svg(li, "lift", base)}
<h2>2. 同じ配置で『人気補正の lift』に色を替えた図（1 語）</h2>
<div class="note">人気補正 lift = 実際の 3 着以内数 ÷ （その馬の単勝人気順位から見込まれる 3 着以内数）。1 図と同じ位置のまま、人気の影響を除くと<b>ほとんどの語が灰色</b>になる。語そのものが人気以上の情報を持つとは言えない（有意な語は 0 個）。ただし人気は発走直前の値でコメントを読んだ後の市場の評価を含むため、この補正はやや厳しめ。</div>
{svg(li, "adj_lift", base)}
<h2>3. 隣り合う 2 語の組（生の lift）</h2>
<p>件数の多い 70 組。「水準＋以上」「自己＋条件」「初戦＋楽しみ」などは青、「もう＋少し」「展開＋向く」「展開＋次第」などは橙。なお「勝負＋期待」「賞金＋加算」のように 2 倍前後の組は件数が少なく、この図には入らない（term_stats.csv を参照）。</p>
{svg(lb, "lift", base)}
<h2>4. 件数 300 以上の語の上位・下位</h2>
<div class="cols"><div><b>好走寄り（生の lift 上位）</b><table>{tbl_head}{top(big, "lift", False)}</table></div>
<div><b>凡走寄り（生の lift 下位）</b><table>{tbl_head}{top(big, "lift", True)}</table></div></div>
<p class="muted">同じレースの馬は 3 着以内の数が 3 頭に固定されて互いに独立ではないため、p 値はやや甘め。形態素解析は fugashi + unidic-lite（名詞・動詞・形容詞・形状詞・副詞の見出し語、直後が「ない」「ず」なら『(否)』）。</p>
</body></html>
"""
    Path(a.out).write_text(html_out, encoding="utf-8")
    print(f"書き出し: {a.out}  uni {len(li)}/{len(uni)}  bi {len(lb)}/{len(bi)}")


if __name__ == "__main__":
    main()
