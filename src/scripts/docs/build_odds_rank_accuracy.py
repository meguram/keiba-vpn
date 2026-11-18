"""想定オッズの人気順の精度（odds_rank_accuracy の結果 JSON）から、docs の md と HTML（1 ファイル・表とグラフ）を作る。

    python -m src.scripts.docs.build_odds_rank_accuracy

入力: docs/operations/odds_rank_accuracy_20261118.json（src/scripts/data/odds_rank_accuracy.py の出力の写し）
出力: docs/operations/odds_rank_accuracy_20261118.md・docs/html/operations/odds_rank_accuracy.html
数字は JSON から取る（文の中の数字も、ここで差し込む）。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/operations/odds_rank_accuracy_20261118.json"
MD = ROOT / "docs/operations/odds_rank_accuracy_20261118.md"
HT = ROOT / "docs/html/operations/odds_rank_accuracy.html"
DAY = "2026-11-18"


def pct(x, d=1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def pt(x, d=1) -> str:
    return "—" if x is None else f"{x * 100:+.{d}f}pt"


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def f2(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def ci(v, fmt=f3) -> str:
    """[平均, 下, 上] → 『0.915（0.913〜0.917）』"""
    if v is None or v[0] is None:
        return "—"
    return f"{fmt(v[0])}（{fmt(v[1])}〜{fmt(v[2])}）"


def cip(v) -> str:
    return "—" if v is None or v[0] is None else f"{pct(v[0])}（{pct(v[1])}〜{pct(v[2])}）"


def cipt(v) -> str:
    return "—" if v is None or v[0] is None else f"{pt(v[0])}（{pt(v[1])}〜{pt(v[2])}）"


def m(v):
    return None if v is None else (v[0] if isinstance(v, list) else v)


class Doc:
    """md と HTML に同じ中身を書く。"""

    def __init__(self) -> None:
        self.md: list[str] = []
        self.ht: list[str] = []

    def h(self, lv: int, t: str) -> None:
        self.md.append(f"\n{'#' * lv} {t}\n")
        self.ht.append(f"<h{lv}>{html.escape(t)}</h{lv}>")

    def p(self, t: str) -> None:
        self.md.append(t + "\n")
        self.ht.append(f"<p>{html.escape(t)}</p>")

    def ul(self, items: list[str]) -> None:
        self.md.append("\n".join(f"- {x}" for x in items) + "\n")
        self.ht.append("<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in items) + "</ul>")

    def table(self, head: list[str], rows: list[list[str]], left: int = 1) -> None:
        self.md.append("| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")
        th = "".join(f'<th class="{"l" if i < left else ""}">{html.escape(c)}</th>' for i, c in enumerate(head))
        tb = "".join("<tr>" + "".join(f'<td class="{"l" if i < left else ""}">{html.escape(c)}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
        self.ht.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>')

    def raw_html(self, s: str) -> None:
        self.ht.append(s)

    def details(self, summary: str) -> None:
        self.md.append(f"\n<details><summary>{summary}</summary>\n")
        self.ht.append(f"<details><summary>{html.escape(summary)}</summary>")

    def end_details(self) -> None:
        self.md.append("\n</details>\n")
        self.ht.append("</details>")


# ───────────── グラフ（インラインの SVG。値は title で hover に出す）
def hbar(rows: list[tuple[str, float, float | None, float | None, str]], vmax: float, fmt, unit_note: str = "") -> str:
    """横棒＋区間。rows: (ラベル, 値, 下, 上, 系列の CSS 変数)。"""
    W, lab, rh = 640, 170, 26
    H = rh * len(rows) + 24
    sc = (W - lab - 70) / vmax
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(unit_note)}">']
    for i in range(0, 6):
        x = lab + (W - lab - 70) * i / 5
        out.append(f'<line x1="{x:.1f}" y1="4" x2="{x:.1f}" y2="{H - 18}" stroke="var(--grid)" stroke-width="1"/>'
                   f'<text x="{x:.1f}" y="{H - 4}" text-anchor="middle">{html.escape(fmt(vmax * i / 5))}</text>')
    for k, (name, v, lo, hi, col) in enumerate(rows):
        y = 6 + k * rh
        w = max(v * sc, 1)
        tip = f"{name}: {fmt(v)}" + (f"（{fmt(lo)}〜{fmt(hi)}）" if lo is not None else "")
        out.append(f'<g><title>{html.escape(tip)}</title><rect x="{lab - 160}" y="{y - 2}" width="{W - lab + 160}" height="{rh}" fill="transparent"/>'
                   f'<text x="{lab - 8}" y="{y + 14}" text-anchor="end" style="fill:var(--text)">{html.escape(name)}</text>'
                   f'<rect x="{lab}" y="{y + 3}" width="{w:.1f}" height="14" rx="4" fill="var({col})"/>')
        if lo is not None and hi is not None:
            out.append(f'<line x1="{lab + lo * sc:.1f}" y1="{y + 10}" x2="{lab + hi * sc:.1f}" y2="{y + 10}" stroke="var(--text)" stroke-width="2"/>')
        out.append(f'<text x="{lab + w + 6:.1f}" y="{y + 14}" style="fill:var(--text)">{html.escape(fmt(v))}</text></g>')
    out.append("</svg>")
    return "".join(out)


def heat(t: dict) -> str:
    rows, cols, p = t["rows"], t["cols"], t["p"]
    lab = lambda x: "10〜" if x == 10 else str(x)  # noqa: E731
    th = "".join(f"<th>{lab(c)}</th>" for c in cols)
    body = []
    for r, row in zip(rows, p):
        tds = "".join(f'<td data-sort="{v:.4f}" style="background:color-mix(in srgb,var(--s1) {min(v * 100, 100):.0f}%,transparent)" '
                      f'title="T-60 {lab(r)} 番人気 → 確定 {lab(c)} 番人気: {v * 100:.1f}%">{v * 100:.0f}</td>' for c, v in zip(cols, row))
        body.append(f'<tr><th class="l">{lab(r)}</th>{tds}</tr>')
    return f'<div class="tw"><table class="heat"><thead><tr><th class="l">T-60＼確定</th>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


CSS = """
:root{--bg:#f7f8fa;--panel:#fff;--text:#1d2330;--muted:#5b6578;--border:#d8dde6;--head:#eef1f6;--grid:#e3e7ee;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s0:#9aa1ad}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;
--border:#2a3446;--head:#1d2638;--grid:#263043;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s0:#6b7486}}
:root[data-theme="dark"]{--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--head:#1d2638;--grid:#263043;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s0:#6b7486}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:1.4rem;margin:.2em 0}h2{font-size:1.15rem;margin:1.6em 0 .5em;border-bottom:1px solid var(--border);padding-bottom:.2em}
h3{font-size:1.02rem;margin:1.1em 0 .4em}
.muted{color:var(--muted);font-size:.88rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.card .k{color:var(--muted);font-size:.82rem}.card .v{font-size:1.3rem;font-weight:600}.card .s{color:var(--muted);font-size:.8rem}
.fig{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin:10px 0}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.84rem;color:var(--muted)}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.4em 0}
th,td{border:1px solid var(--border);padding:4px 6px;text-align:right;white-space:nowrap}
th{background:var(--head);font-weight:600}td.l,th.l{text-align:left;white-space:normal}
table.heat td{text-align:center;min-width:2.4em}
details{margin:.6em 0}summary{cursor:pointer;color:var(--muted)}
svg{display:block;width:100%;height:auto}svg text{fill:var(--muted);font-size:11px}
ul{padding-left:1.2em}
"""


def legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><i style="background:var({c})"></i>{html.escape(n)}</span>' for n, c in items) + "</div>"


E = 2.718281828459045


def rng_str(vals, fmt) -> str:
    vals = [v for v in vals if v is not None]
    return f"{fmt(min(vals))}〜{fmt(max(vals))}"


def build(r: dict) -> Doc:
    F, P = r["final"], r["pre"]
    o, vs = F["overall"], F["vs"]
    s25, pv25, u25, v825 = o["sec77_2025"], o["prev_pop_2025"], o["uniform_2025"], o["v8_2025"]
    s24, pw24 = o["sec77_2224"], o["p_win_2224"]
    t60 = P["phases"]["t60"]
    gap, gpred, g60 = F["by"]["gap_band"], F["by"]["gap_pred"], P["by"]["gap_t60"]
    dr = P["drift"]
    fav = dr["fav_share"]
    fl = P["filters"]
    cal = F["calib"]
    prod = cal.get("prod_2026")
    oof25 = cal["by_year"]["2025"]
    cls = F["by"]["cls"]
    cvr = F["cls_vs_rest"]
    low_cls = [k for k in ("新馬", "未勝利", "1勝", "2勝", "3勝", "OP・L", "重賞") if k in cvr and cvr[k]["spearman"][2] < 0]
    gl = ["1.3 倍未満", "1.3〜2 倍", "2〜3 倍", "3 倍以上"]
    d = Doc()

    # ── 冒頭
    d.raw_html(f'<h1>想定オッズで作った人気順の精度</h1><p class="muted">最終更新: {DAY}。調査だけ（モデル・本番は変えない）。費用 0 円（手元のデータだけ）。</p>')
    d.md.append(f"# 想定オッズで作った人気順の精度\n\n最終更新: {DAY}。調査だけ（モデル・本番は変えない）。費用 0 円（手元のデータだけ）。\n")
    d.h(2, "結論")
    concl = [
        f"想定オッズ（§77）の順は、確定の人気順とよく合う。2025 年（本番と同じ作り方で 2025 年を学習に使わない版）は、順位の相関（Spearman）{f3(m(s25['spearman']))}、1 番人気の的中 {pct(m(s25['top1']))}、上位 3 頭の重なり {f2(m(s25['top3']))} 頭（3 頭中）。",
        f"旧来の想定オッズ（V8）より合う（2025 年。相関 {vs['2025_vs_v8']['spearman'][0]:+.3f}、1 番人気の的中 {pt(vs['2025_vs_v8']['top1'][0])}）。AI 勝率の順・前走の人気の順よりも合う。差はどれも区間が 0 をまたがない。",
        f"外れやすいのは、1・2 番人気が近いレース。想定オッズの 2 位が 1 位の 1.3 倍未満なら、1 番人気の的中は {pct(m(gpred['1.3 倍未満']['sec77']['top1']))}。3 倍以上なら {pct(m(gpred['3 倍以上']['sec77']['top1']))}。この差は発走前に分かる。",
        f"クラスでは {'、'.join(low_cls)} で、それ以外のクラスより相関が下がる（区間が 0 をまたがない）。新馬が最も低い（{f3(m(cls['新馬']['sec77']['spearman']))}）。",
    ]
    if prod:
        concl.append(f"2026 年の本番の版は、2025 年より外れが大きい。対数オッズの誤差（RMSE）は {prod['rmse_log']:.3f}（{prod['days']} 日、2025 年の値は {oof25['rmse_log']:.3f}）。"
                     "上の順位の数字は、2026 年の本番では下がる見込み。順位そのものはまだ測っていない。")
    concl += [
        f"T-60 の人気と確定の人気の相関は {f3(m(t60['spearman']))}。T-60 の 1 番人気が確定で 1 番人気でなくなるのは {pct(fl['top1']['in60_out_f'][0])}（区間 {pct(fl['top1']['in60_out_f'][1])}〜{pct(fl['top1']['in60_out_f'][2])}。{t60['days']} 日・{t60['races']} レースと少ない）。T-60 は発走 60 分前の人気。",
        f"確定に向けて、上位人気に票が寄る。T-60 の 1 番人気の市場の確率は {pct(fav['t60'])} → {pct(fav['final'])}（{pt(fav['diff'][0])}）。人気薄のオッズは伸びる。",
        "本番の判定は T-60 の時点で行う。想定オッズと T-60 の人気の直接の比較が最も要るが、まだ測っていない。2026 年の想定オッズは GCS にしか無いため（下の「残り」）。",
    ]
    d.ul(concl)
    cards = [("相関（2025・確定）", f3(m(s25["spearman"])), f"旧来の V8 {f3(m(v825['spearman']))}"),
             ("1 番人気の的中（2025）", pct(m(s25["top1"])), f"無作為 {pct(u25['top1'])}"),
             ("上位 3 頭の重なり（2025）", f"{f2(m(s25['top3']))} 頭", f"無作為 {f2(u25['top3'])} 頭"),
             ("T-60 → 確定の相関", f3(m(t60["spearman"])), f"{t60['days']} 日・{t60['races']} レース")]
    d.raw_html('<div class="cards">' + "".join(f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div><div class="s">{html.escape(s)}</div></div>' for k, v, s in cards) + "</div>")

    # ── 測り方
    fi = [cal["by_year"][str(y)]["n"] for y in (2022, 2025)]
    d.h(2, "測り方")
    d.ul([
        "想定オッズ: 本番の画面と推奨が使う §77（sec77。src/pipeline/inference/odds_sec77.py）。配備の設定で t6（KEIBA_ODDS_T6）は有効になっていない。",
        "期間: 本番の版は 2020〜2025 年で学習している（serving/meta.json）。2025 年の本番の版の出力は学習の期間の中になる。"
        "そのため、同じ作り方で各年より前の年だけで学習した版（walk-forward の OOF。data/local/modeling/odds_pred/wp_oof_pred.parquet）の 2025 年を主にした。"
        "各年の版は Y−2 年までで学習し、Y−1 年を早期停止に使う（2022 年の版は 2020 年の 1 年、2025 年の版は 2020〜2023 年の 4 年で学習）。2022〜2024 年は年ごとの安定と AI 勝率との比較に使った。",
        "特徴量の前段の列のうち、s_time_pred・s_rank_pred（前段 S3）は年ごとの expanding OOF（src/scripts/data/tp07_base.py の stage）。v6_eval.build_frame が足す他の前段の列（b_dist_pred など）の切り方は、今回は確かめていない。",
        "予測の順: 生の出力 share_raw の大きい順。想定単勝は 1.0 倍で頭打ちになり、同順位ができる。そのため順位は生の出力で付けた。レースの中の順は変換で変わらない。",
        "実際の順: 確定の人気（race_odds 由来。同順位は平均の順位。実データに同順位は無かった）。2026 年の確定は race_result_flat の popularity。発走前は data/local/research/persona1/pre_odds.parquet の t60〜t02。",
        "基準: (a) 旧来の想定オッズ V8（同じ表の win_V8。各年より前だけで学習）、(b) AI 勝率（v6 の OOF の p_win。2022〜2024 年だけ。2025 年の版は 2025 年を学習に含むので使わない）、"
        "(c) 前走の人気（取消を飛ばした 1 つ前の出走。無い馬は最下位で、同順位は無作為）、(d) 無作為（式で出す。相関 0・1 番人気の的中 = 1/頭数・上位 3 頭の重なり = 9/頭数）。",
        f"区間: レースで束ねたブートストラップ（{r['boot']} 回・95%）。基準との差は、同じレースの差を束ねた。T-60 は日で束ねた区間も出したが、日が 9 個しかなく狭く出るので、参考にとどめる。",
        "2026 年は、人気・オッズの列とレースの条件の列だけを読んだ（着順・払戻は読まない）。",
    ])
    d.details("指標の言葉")
    d.ul(["順位の相関（Spearman・Kendall）: レースの中で、予測の順と実際の順がどれだけそろうか。1 で完全に一致、0 で無関係。レースごとに出して平均した。",
          "1 番人気の的中: 予測の 1 位の馬が、実際の 1 番人気か。",
          "上位 3 頭の重なり: 予測の上位 3 頭と、実際の 1〜3 番人気の重なる頭数（0〜3）。",
          "順位の平均の誤差: |予測の順位 − 実際の順位| の馬の平均（レースごと）。",
          "無作為比: 人気帯の順位の誤差 ÷ 同じ馬を無作為の順に置いたときの誤差の期待値。端の人気は片側にしかずれないので、帯どうしを比べるときはこの比で見る。"])
    d.end_details()

    # ── 部 1
    d.h(2, "1. 想定オッズの順 → 確定の人気")
    d.h(3, "年ごとと、基準との比較")
    row = lambda name, a: [name, f"{a['races']:,}", f3(m(a["spearman"])), f3(m(a["kendall"])), pct(m(a["top1"])), f2(m(a["top3"])), f2(m(a["mae_rank"]))]  # noqa: E731
    rows = [row(f"想定オッズ {y} 年", o[f"sec77_{y}"]) for y in (2022, 2023, 2024, 2025)]
    rows += [row("旧来の V8（2025）", v825), row("前走の人気（2025）", pv25),
             ["無作為（2025）", f"{s25['races']:,}", "0.000", "0.000", pct(u25["top1"]), f2(u25["top3"]), f2(u25["mae_rank"])],
             row("想定オッズ（2022〜24）", s24), row("AI 勝率の順（2022〜24）", pw24)]
    d.table(["順の作り方", "レース", "Spearman", "Kendall", "1 番人気の的中", "上位 3 頭の重なり", "順位の誤差"], rows)
    pv_rng = rng_str([m(o[f"prev_pop_{y}"]["spearman"]) for y in (2022, 2023, 2024, 2025)], f3)
    d.p(f"想定オッズの相関は 2022 年 {f3(m(o['sec77_2022']['spearman']))} → 2025 年 {f3(m(s25['spearman']))} と、年ごとに上がる。"
        f"前走の人気の相関は 4 年とも {pv_rng} で横ばい。そのため、市場の側の変化ではなく、学習の年数が増えた分と見られる（推定）。")
    d.raw_html('<div class="fig"><h3>順位の相関（レースの平均・95% 区間）</h3>' + legend([("想定オッズ §77", "--s1"), ("旧来の V8・AI 勝率", "--s2"), ("前走の人気", "--s3"), ("無作為", "--s0")]) +
               hbar([("§77 2025", *s25["spearman"], "--s1"), ("V8 2025", *v825["spearman"], "--s2"), ("前走の人気 2025", *pv25["spearman"], "--s3"),
                     ("無作為 2025", 0.0, None, None, "--s0"), ("§77 2022〜24", *s24["spearman"], "--s1"), ("AI 勝率 2022〜24", *pw24["spearman"], "--s2")],
                    1.0, lambda x: f"{x:.2f}", "順位の相関") + "</div>")
    d.h(3, "上乗せ（同じレースの差・95% 区間）")
    rows = []
    for lab, key in (("§77 − 旧来の V8（2025）", "2025_vs_v8"), ("§77 − 前走の人気（2025）", "2025_vs_prev_pop"), ("§77 − 無作為（2025）", "2025_vs_uniform"),
                     ("§77 − AI 勝率（2022〜24）", "2224_vs_p_win"), ("§77 − 旧来の V8（2022〜24）", "2224_vs_v8")):
        v = vs[key]
        rows.append([lab, ci(v["spearman"]), cipt(v["top1"]), ci(v["top3"], f2), ci(v["mae_rank"], f2)])
    d.table(["比べ方", "相関の差", "1 番人気の的中の差", "上位 3 頭の重なりの差", "順位の誤差の差（負が良い）"], rows)

    d.h(3, "分け方ごと（2025 年）")
    lab_map = {"surf": "芝ダ", "cls": "クラス", "fs_band": "頭数", "shinba": "新馬", "jusho": "重賞"}
    order = {"cls": ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP・L", "重賞"], "fs_band": ["〜10 頭", "11〜14 頭", "15〜18 頭"], "surf": ["芝", "ダート", "障害"]}
    rows = []
    for col in ("surf", "cls", "fs_band"):
        g = F["by"][col]
        for v in order.get(col, sorted(g)):
            if v not in g:
                continue
            x = g[v]
            rows.append([f"{lab_map[col]}: {v}", f"{x['sec77']['races']:,}", ci(x["sec77"]["spearman"]), cip(x["sec77"]["top1"]), f2(m(x["sec77"]["top3"])),
                         f3(m(x["v8"]["spearman"])), f3(m(x["prev_pop"]["spearman"]))])
    d.table(["分け方", "レース", "Spearman（区間）", "1 番人気の的中（区間）", "上位 3 頭の重なり", "V8 の相関", "前走の人気の相関"], rows)
    d.table(["クラス", "相関の差（そのクラス − それ以外）", "1 番人気の的中の差"],
            [[k, ci(cvr[k]["spearman"]), cipt(cvr[k]["top1"])] for k in order["cls"] if k in cvr])
    d.p(f"新馬は前走が無いので、前走の人気の相関は {f3(m(cls['新馬']['prev_pop']['spearman']))}（前走の人気がある馬の割合 {pct(F['coverage'].get('新馬', 0))}）。"
        "頭数が多いほど順位の誤差は大きいが、順位の幅が広いためで、相関はほぼ同じ。")
    d.details("月ごと（2025 年）")
    rows = [[k, f"{x['sec77']['races']:,}", f3(m(x["sec77"]["spearman"])), pct(m(x["sec77"]["top1"])), f2(m(x["sec77"]["top3"]))] for k, x in sorted(F["by"]["month"].items())]
    d.table(["月", "レース", "Spearman", "1 番人気の的中", "上位 3 頭の重なり"], rows)
    d.end_details()
    d.details("AI 勝率との比較をクラス・頭数で分けたもの（2022〜2024 年）")
    rows = []
    for col in ("cls", "fs_band"):
        g = F["by_pwin"][col]
        for v in order.get(col, sorted(g)):
            if v in g:
                x = g[v]
                rows.append([f"{lab_map[col]}: {v}", f"{x['sec77']['races']:,}", f3(m(x["sec77"]["spearman"])), f3(m(x["p_win"]["spearman"])), ci(x["vs_p_win"]["spearman"]), cipt(x["vs_p_win"]["top1"])])
    d.table(["分け方", "レース", "想定オッズの相関", "AI 勝率の相関", "相関の差", "1 番人気の的中の差"], rows)
    d.end_details()

    d.h(3, "1 番人気の的中は、1・2 番人気の差で決まる")
    d.raw_html('<div class="fig">' + hbar([(f"想定の 2 位/1 位 {k}", *gpred[k]["sec77"]["top1"], "--s1") for k in gl], 1.0, lambda x: f"{x * 100:.0f}%", "1 番人気の的中") +
               '<p class="muted">想定単勝で、2 位の馬のオッズが 1 位の何倍か（2025 年）。発走前に分かる。</p></div>')
    d.table(["1・2 番の差", "想定オッズで分ける（前もって分かる）", "確定のオッズで分ける（診断）", "参考: T-60 の 1 番人気が確定でも同じ（T-60 で分ける）"],
            [[k, f"{cip(gpred[k]['sec77']['top1'])}・{gpred[k]['sec77']['races']:,} R", f"{cip(gap[k]['sec77']['top1'])}・{gap[k]['sec77']['races']:,} R",
              f"{cip(g60[k]['top1'])}・{g60[k]['races']} R" if k in g60 else "—"] for k in gl])
    gsp = rng_str([m(gap[k]["sec77"]["spearman"]) for k in gl], f3)
    d.p(f"相関はどの段も {gsp} でほぼ同じ。外れるのは主に 1 番人気の取り違え。T-60 の市場でも、1・2 番人気が近いレースでは 1 番人気がよく入れ替わる（参考の列）。僅差の 1 番人気は、確定まで決まらない。")

    d.h(3, "人気帯ごとの順位の誤差")
    eb = [("1〜3 番人気", "err_1_3"), ("4〜6 番人気", "err_4_6"), ("7 番人気〜", "err_7p")]
    ratio = lambda a, k: f"{m(a[k]) / m(a['u_' + k]):.2f}" if a.get("u_" + k) and m(a.get("u_" + k)) else "—"  # noqa: E731
    d.table(["実際の人気", "§77 → 確定（2025）の誤差", "無作為比", "V8（2025）の無作為比", "T-60 → 確定（2026）の誤差", "無作為比"],
            [[n, ci(s25[k], f2), ratio(s25, k), ratio(v825, k), ci(t60[k], f2), ratio(t60, k)] for n, k in eb])
    d.p(f"無作為比で見ても、中位（4〜6 番人気）が最も悪い（§77 で {ratio(s25, 'err_4_6')}。1〜3 番人気は {ratio(s25, 'err_1_3')}）。T-60 の市場でも同じ並び。"
        "中位は似たオッズの馬が多く、わずかな票の差で順位が入れ替わる、という見立てと合う（ただし T-60 は 2026 年の別のレースで、同じ表の値は直接比べられない）。")

    d.h(3, "対数オッズの誤差と較正（想定単勝）")
    d.table(["年", "対数の誤差（平均の絶対値）", "RMSE", "平均（想定 − 確定）", "頭数"],
            [[f"{y} 年（OOF）", f"{v['mae_log']:.3f}", f"{v['rmse_log']:.3f}", f"{v['bias_log']:+.3f}", f"{v['n']:,}"] for y, v in cal["by_year"].items()] +
            ([[f"2026 年（本番の版・{prod['days']} 日）", "—", f"{prod['rmse_log']:.3f}", f"{prod['bias_log']:+.3f}", f"{prod['n']:,}"]] if prod else []))
    d.p("OOF の想定単勝（win_B）の変換は、各年より前の年の OOF で当てはめた直線（2022 年は前の年が無いので変換なし）。2025 年の値に当てはめの内側は混じらない。"
        + (f"2026 年の行は、本番の版（2020〜2025 年で学習）を {prod['first']}〜{prod['last']} に後から当てたもの（data/local/meta/odds_sec77_compare.json）。2025 年の OOF より誤差が大きく、短く出る向きも強い。" if prod else ""))
    rows = [[f"{i + 1}", f"{b['pred']:.1f}", f"{b['act']:.1f}", f"{b['act'] / b['pred']:.2f}", f"{b['mae']:.3f}", f"{b['n']:,}"] for i, b in enumerate(cal["bins"])]
    d.table(["想定の 10 分位（2025）", "想定単勝（幾何平均）", "確定単勝（幾何平均）", "確定/想定", "対数の誤差", "頭数"], rows)
    rr = [b["act"] / b["pred"] for b in cal["bins"]]
    d.p(f"想定単勝は 1〜9 分位で確定より短い（確定/想定 {min(rr[:-1]):.2f}〜{max(rr[:-1]):.2f}）。最も人気の無い 10 分位だけ長い（{rr[-1]:.2f}）。"
        "見立て（未検証）: 対数シェアを二乗誤差で学ぶと、ばらつきの大きい人気薄のシェアを小さく見積もる。そのうえレースの中で足して 1 にするので、人気薄が長すぎる分、他の馬が一律に短くなる。"
        "順位には影響しないが、推奨の期待値の計算には影響する（この調査の範囲外）。")
    d.details("人気帯ごとの対数の誤差（実際の人気で分けたもの）")
    d.table(["実際の人気", "対数の誤差", "平均（想定 − 確定）", "頭数"], [[k, f"{v['mae_log']:.3f}", f"{v['bias_log']:+.3f}", f"{v['n']:,}"] for k, v in cal["by_band"].items()])
    d.p("実際の値で分けると、どんな予測でも平均へ寄る向き（人気馬は長め・人気薄は短め）に見える。較正は上の想定の分位で見る。")
    d.end_details()

    # ── 部 2
    mins = P["phase_minutes"]
    days = P["days"]
    d.h(2, "2. T-60 の人気 → 確定の人気")
    d.p(f"材料は pre_odds（{min(days)}〜{max(days)}・{t60['days']} 日・{t60['races']} レース）。人気だけを読んだ。日数とレース数が少ないので、区間を必ず添える。")
    d.h(3, "時点ごと（確定に近いほど合う）")
    nm = {"t60": "T-60", "t15": "T-15", "t10": "T-10", "t05": "T-5", "t02": "T-2"}
    rows = []
    for ph in ("t60", "t15", "t10", "t05", "t02"):
        x = P["phases"].get(ph)
        if x:
            mm = mins.get(ph, {})
            rows.append([nm[ph], f"{mm.get('median', 0):.0f} 分前", f"{x['days']}", f"{x['races']}", ci(x["spearman"]), cip(x["top1"]), ci(x["top3"], f2)])
    d.table(["時点", "取った時刻（中央値）", "日", "レース", "Spearman", "1 番人気が同じ", "上位 3 頭の重なり"], rows)
    pc = P["phases_common"]
    if pc:
        d.table(["時点（4 時点がそろうレースだけ）", "レース", "Spearman", "1 番人気が同じ"],
                [[nm[k], f"{v['races']}", ci(v["spearman"]), cip(v["top1"])] for k, v in pc.items()])
    dc = t60.get("day_ci", {})
    d.p(f"T-2 は実際には発走の時刻ちょうど（中央値 {mins.get('t02', {}).get('median', 0):.0f} 分前）に取っていて、締切の直前にあたる。"
        f"T-60 の日で束ねた区間: 相関 {ci(dc.get('spearman'))}、1 番人気が同じ {cip(dc.get('top1'))}。"
        f"日は {t60['days']} 個しかなく、群の少ないブートストラップは区間を狭く出しやすい。そのため、判断にはレースで束ねた区間（広いほう）を使う。")
    d.h(3, "分け方ごと（T-60 → 確定）")
    lab2 = {"fs_band": "頭数", "cls": "クラス", "surf": "芝ダ", "shinba": "新馬", "time_band": "発走の時間帯"}
    order2 = {**order, "time_band": ["〜11 時台", "12〜13 時台", "14 時台〜"], "shinba": ["新馬", "新馬以外"]}
    rows = []
    nseg = 0
    for col in ("fs_band", "cls", "surf", "shinba", "time_band"):
        g = P["by"][col]
        for v in order2.get(col, sorted(g)):
            if v in g:
                x = g[v]
                nseg += 1
                rows.append([f"{lab2[col]}: {v}", f"{x['races']}", ci(x["spearman"]), cip(x["top1"]), f2(m(x["top3"]))])
    d.table(["分け方", "レース", "Spearman（区間）", "1 番人気が同じ（区間）", "上位 3 頭の重なり"], rows)
    cb = P["by"]["cls"]
    segs = [(f"{lab2[c]}: {v}", x["spearman"]) for c in ("fs_band", "cls", "surf", "shinba", "time_band") for v, x in P["by"][c].items() if x.get("spearman") and x["spearman"][1] is not None]
    nov = [(a, b_) for i, (a, ca) in enumerate(segs) for (b_, cb_) in segs[i + 1:] if ca[2] < cb_[1] or cb_[2] < ca[1]]
    ex_ = "、".join(f"{a} と {b_}" for a, b_ in nov[:4])
    d.p(f"相関の区間が重ならない組は {len(nov)} 組（{nseg} 区分の {nseg * (nseg - 1) // 2} 組のうち。例: {ex_}）。"
        "組の数が多く、多重比較を考えると、どの差も判断保留とする。"
        + (f"新馬の 1 番人気が同じ割合は低めに出るが、{cb['新馬']['races']} レースしかない。" if "新馬" in cb else ""))
    d.details("日ごとのばらつき（T-60 → 確定）")
    d.table(["日", "レース", "Spearman", "1 番人気が同じ"], [[k, f"{v['races']}", f3(v["spearman"]), pct(v["top1"])] for k, v in sorted(days.items())])
    p1 = m(t60["top1"])
    nfull = max(v["races"] for v in days.values())
    d.p(f"1 日 {nfull} レースなら、1 番人気が同じ割合の偶然の幅は 95% で ±{1.96 * (p1 * (1 - p1) / nfull) ** 0.5 * 100:.0f}pt 前後（二項）。日ごとの差はその範囲に収まる。")
    d.end_details()

    d.h(3, "何番人気が何番人気に動くか（行 = T-60 の人気、列 = 確定の人気。行の %）")
    d.raw_html(heat(P["transition"]))
    tl = lambda x: "10〜" if x == 10 else str(x)  # noqa: E731
    d.md.append("| T-60＼確定 | " + " | ".join(tl(c) for c in P["transition"]["cols"]) + " |\n|" + "---|" * (len(P["transition"]["cols"]) + 1) + "\n" +
                "\n".join(f"| {tl(r_)} | " + " | ".join(f"{v * 100:.0f}" for v in row_) + " |" for r_, row_ in zip(P["transition"]["rows"], P["transition"]["p"])) + "\n")
    tr = P["transition"]["p"]
    stay = [tr[i][i] for i in range(1, 8)]
    d.p(f"T-60 の 1 番人気は {tr[0][0] * 100:.0f}% が確定でも 1 番人気、{tr[0][1] * 100:.0f}% が 2 番人気。2〜8 番人気が同じ順位に残るのは {min(stay) * 100:.0f}〜{max(stay) * 100:.0f}% で、動く馬の多くは隣の順位に移る。")

    d.h(3, "対数オッズの変化 log(確定 / T-60)")
    ob = ["<3", "3-6", "6-10", "10-20", "20-50", "50+"]
    d.raw_html('<div class="fig">' + hbar([(f"T-60 の単勝 {k} 倍", max(dr['oband'][k]['mean'] + 0.2, 0.001), dr['oband'][k]['lo'] + 0.2, dr['oband'][k]['hi'] + 0.2, "--s1") for k in ob], 1.2,
                                          lambda x: f"{x - 0.2:+.2f}", "対数オッズの変化") +
               '<p class="muted">目盛りは log(確定/T-60)。0 より右は確定でオッズが伸びた（売れなかった）。左は縮んだ（売れた）。</p></div>')
    d.table(["T-60 の単勝", "頭数", "平均（区間）", "中央値", "確定/T-60（幾何平均の比）", "20% 超縮んだ割合"],
            [[f"{k} 倍", f"{dr['oband'][k]['n']:,}", f"{dr['oband'][k]['mean']:+.3f}（{dr['oband'][k]['lo']:+.3f}〜{dr['oband'][k]['hi']:+.3f}）", f"{dr['oband'][k]['median']:+.3f}",
              f"{E ** dr['oband'][k]['mean']:.2f} 倍", pct(dr["oband"][k]["share_down20"])] for k in ob])
    d.table(["T-60 の人気", "頭数", "平均（区間）", "確定/T-60（幾何平均の比）"],
            [[k, f"{dr['band'][k]['n']:,}", f"{dr['band'][k]['mean']:+.3f}（{dr['band'][k]['lo']:+.3f}〜{dr['band'][k]['hi']:+.3f}）", f"{E ** dr['band'][k]['mean']:.2f} 倍"]
             for k in ("1 番人気", "2〜3", "4〜6", "7〜")])
    ex = dr.get("existing", {})
    if ex.get("20-50"):
        d.p(f"オッズ帯ごとの平均は、既存の data/local/meta/odds_drift/latest_t60.json（同じ材料・別の集計）とほぼ同じ（例: 20〜50 倍 {dr['oband']['20-50']['mean']:+.3f}、既存 {ex['20-50']['mean']:+.3f}）。")
    nz = P["noise"]
    d.p(f"オッズ帯で見ると、最も縮むのは 3〜6 倍の帯（{dr['oband']['3-6']['mean']:+.3f}）で、3 倍未満（{dr['oband']['<3']['mean']:+.3f}）より大きい。"
        f"人気帯で見ると、1 番人気（{dr['band']['1 番人気']['mean']:+.3f}）と 2〜3 番人気（{dr['band']['2〜3']['mean']:+.3f}）はほぼ同じ。そのため「本命だけに寄る」とは言えず、「上位人気に寄る」と書く。"
        f"T-60 の 1 番人気の市場の確率の変化 {pt(fav['diff'][0])} は、レースで束ねた区間 {pt(fav['diff'][1])}〜{pt(fav['diff'][2])} で 0 をまたがない。")
    d.p(f"解釈は 2 つあり、どちらかに決められない。(1) 締切に近い票ほど情報を持つ（公開の知見）。(2) T-60 は票が少なく、人気薄のオッズが詰まって見える（最大オッズの中央値は T-60 {nz['t60']['max_odds_median']:.0f} 倍・確定 {nz['final']['max_odds_median']:.0f} 倍。"
        f"控除の目安 Σ1/オッズは {nz['t60']['overround_median']:.3f} と {nz['final']['overround_median']:.3f} で同じ）。(2) だけでも、人気薄のオッズは確定に向けて伸びる。")

    d.h(3, "T-60 の人気を確定の代わりに使うと（人気の絞り込みへの影響）")
    rows = []
    for k in (1, 2, 3, 5):
        x = fl[f"top{k}"]
        rows.append([f"{k} 番人気以内", cip(x["in60_out_f"]), pct(x["out60_in_f"]), pct(x["flip_share_all"]), cip(x["race_any_flip"])])
    d.table(["絞り込み", "T-60 で入る → 確定で外れる（区間）", "T-60 で外れる → 確定で入る", "食い違う馬（全馬のうち）", "顔ぶれが変わるレース（区間）"], rows)
    d.p(f"例: 「2 番人気以下を買う」（1 番人気を外す）を T-60 で判定すると、{pct(fl['top1']['race_any_flip'][0])} のレースで、外した馬が確定の 1 番人気と違う。"
        f"「3 番人気以内」の絞り込みでは、T-60 で選んだ馬の {pct(fl['top3']['in60_out_f'][0])} が確定で 4 番人気以下になり、{pct(fl['top3']['race_any_flip'][0])} のレースで顔ぶれが変わる。")
    d.p("過去のデータで調べた条件は確定の人気で作っている。それを T-60 の人気にそのまま当てると、この割合だけ別の馬を買うことになる。入れ替わるのは、条件の境目にいる馬。")

    d.h(3, "基準の順は、T-60 と確定のどちらに近いか（2026 年・同じレース）")
    b = P["base"]
    rows = [[nm_, ci(b[k]["vs_t60"]["spearman"]), ci(b[k]["vs_final"]["spearman"]), ci(b[k]["final_minus_t60"]["spearman"]), cipt(b[k]["final_minus_t60"]["top1"])]
            for nm_, k in (("AI 勝率の順", "p_win"), ("前走の人気", "prev_pop"))]
    d.table(["順の作り方", "T-60 との相関", "確定との相関", "差（確定 − T-60）", "1 番人気の的中の差"], rows)
    srcs = b.get("pwin_source", {})
    d.p("どちらの順も、T-60 より確定の人気に近い。T-60 の順位に混じる雑音が確定までに減るだけでも、この向きになる。そのため「締切に近い票ほど情報を持つ」の証拠とまでは言えない。"
        + (f"AI 勝率の出どころは、本番の当日の予測 {srcs.get('predictions_v6', 0):.0f} レース・後付けの予測 {srcs.get('predictions_backfill/bundle_v6_2025', 0):.0f} レース。" if srcs else ""))

    # ── 事象と根拠
    d.h(2, "事象と根拠")
    sp_rng = rng_str([m(o[f"sec77_{y}"]["spearman"]) for y in (2022, 2023, 2024, 2025)], f3)
    d.table(["事象", "根拠"], [
        [f"想定オッズの順は、確定の人気と相関 {f3(m(s25['spearman']))}・1 番人気の的中 {pct(m(s25['top1']))}（2025 年）",
         f"目的変数が確定の単勝の対数シェアそのもの（odds_pred_eval.py）。4 年すべてで {sp_rng}（学習の年数が増えるにつれて上がる）。"],
        [f"旧来の V8 より相関が {vs['2025_vs_v8']['spearman'][0]:+.3f}、AI 勝率の順より {vs['2224_vs_p_win']['spearman'][0]:+.3f} 高い",
         "同じレースの差の区間が 0 をまたがない。AI 勝率より合う点は、クラス・頭数のすべての分け方で同じ向き（折りたたみの表）。"
         "理由の見立て（未検証）: AI 勝率は着順を当てる学習で、市場が付ける人気の偏り（騎手・厩舎の人気など）を写さない。"],
        [f"1・2 番人気が近いと、1 番人気の的中が下がる（想定の差 1.3 倍未満で {pct(m(gpred['1.3 倍未満']['sec77']['top1']))}）",
         "想定オッズで分けても、確定のオッズで分けても、4 段で単調に上がる。T-60 の市場の 1 番人気も、僅差なら確定でよく入れ替わる。相関は段によらない。"],
        [f"中位（4〜6 番人気）の順位の誤差が最も大きい（無作為比 {ratio(s25, 'err_4_6')}）",
         "端の人気が片側にしかずれない分を無作為比で除いても、中位が最も悪い。T-60 → 確定でも同じ並び。"],
        [f"新馬で下がる（相関 {f3(m(cls['新馬']['sec77']['spearman']))}）",
         f"前走の情報が無い（前走の人気がある馬 {pct(F['coverage'].get('新馬', 0))}）。AI 勝率も新馬で下がる（2022〜24）。"
         "見立て（未検証）: 市場は調教・評判で人気を作るが、特徴量に無い（調教の列の有無は確かめていない）。"],
        [f"{'、'.join(k for k in low_cls if k != '新馬')} でも下がる",
         "それ以外のクラスとの差の区間が 0 をまたがない（上の表）。実力の近い馬がそろい、オッズの差が小さいため、と見る（見立て）。"],
        [f"T-60 → 確定で上位人気に票が寄る（1 番人気の市場の確率 {pt(fav['diff'][0])}・人気薄のオッズが伸びる）",
         "公開の知見: 締切に近い票ほど情報を持つ（Asch・Malkiel・Quandt 1982、Ottaviani・Sørensen）。ただし手元では、T-60 の票が少ないことによる雑音でも説明できる（上の節）。どちらかは決めていない。"],
    ])
    d.p("出典: Asch, Malkiel, Quandt (1982) Racetrack Betting and Informed Behavior, Journal of Financial Economics 10(2) https://econpapers.repec.org/RePEc:eee:jfinec:v:10:y:1982:i:2:p:187-194 ／ Ottaviani, Sørensen, The Timing of Bets and the Favorite-Longshot Bias (CEPR DP4092) https://cepr.org/publications/dp4092")

    # ── 限界と残り
    d.h(2, "注意と残り")
    d.ul([
        "主の期間は walk-forward の OOF（2025 年の版は 2020〜2023 年で学習）。本番の版（2020〜2025 年で学習）とは学習の年数が違う。本番の版の 2026 年の順位の精度は、まだ測っていない。対数の誤差では、2026 年の本番は 2025 年の OOF より外れが大きい（上の表）。",
        "2024 年の版（単勝・複勝）と 2025 年の版の複勝は、木の本数が上限の 1500 に届いている（data/local/modeling/odds_pred/wp_fit_info.json）。もっと木を増やせば、少し良くなる余地がある。",
        f"T-60 の材料は秋（{min(days)}〜{max(days)}）の {t60['days']} 日・{t60['races']} レースだけ。分け方ごとの差は判断保留。週が増えたら同じスクリプトで作り直す。",
        "上を重く見る指標（NDCG など）は出していない。買い方に関わる上位は、1 番人気の的中と上位 3 頭の重なりで見た。",
        "残り（最優先）: 想定オッズと T-60 の人気の直接の比較。本番の想定オッズは発走の約 70 分前（馬体重の発表の後）に作られ、判定は T-60 で行うため。"
        "2026 年の想定オッズ（final_odds_prediction）は GCS にしか無い。10/18 以降の約 300 文書を 1 回読むだけなら、転送は数 MB で 1 円未満の見込み。今回の決まり（GCS を読まない）に従い、読んでいない。読むかは統括役の判断。",
        "作り直しの手順: ODDS_RANK_DATA_ROOT に本体のリポジトリ、ODDS_RANK_OUT に出力先を入れて、nice -n 10 python -m src.scripts.data.odds_rank_accuracy を流す。結果を docs/operations/odds_rank_accuracy_20261118.json に写し、python -m src.scripts.docs.build_odds_rank_accuracy を流す。",
    ])
    return d


def main() -> None:
    r = json.loads(SRC.read_text(encoding="utf-8"))
    d = build(r)
    review = ROOT / "docs/operations/odds_rank_accuracy_20261118_review.md"
    md = "\n".join(d.md)
    ht = "\n".join(d.ht)
    if review.exists():                     # 別のエージェントのレビューと対応（手で書く）
        rv = review.read_text(encoding="utf-8")
        md += "\n" + rv
        ht += "<h2>レビュー（別のエージェント）と対応</h2>" + "".join(
            f"<p>{html.escape(line[2:] if line.startswith('- ') else line)}</p>" for line in rv.splitlines() if line.strip() and not line.startswith("#"))
    MD.write_text(md, encoding="utf-8")
    HT.write_text(f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                  f"<title>想定オッズの人気順の精度</title><style>{CSS}</style></head><body><main>{ht}</main></body></html>\n", encoding="utf-8")
    print(f"保存: {MD.relative_to(ROOT)}・{HT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
