"""トラックバイアスの HTML（docs/html/operations/final/track_bias.html と track_bias_labels.html）。build-all の出力から作る。"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.track_bias import data as D
from src.research.track_bias import day as Y

HTML_DIR = D.REPO / "docs/html/operations/final"
CAL_PATH = D.REPO / "config/track_bias/calibration.json"
SAMPLE_FIXED = [("2026-11-22", "東京", "依頼の日（11/22 の東京）"), ("2026-11-22", "京都", "依頼の日（11/22 の京都）"),
                ("2026-11-23", "東京", "依頼の日（11/23 の東京）"),
                ("2026-11-23", "福島", "11/23 は京都の開催が無い（東京と福島）ので、代わりに福島")]
NORMAL_FREQ = {"-2": 0.0228, "-1": 0.1359, "0": 0.6827, "1": 0.1359, "2": 0.0228}   # 偶然だけ（標準正規）のときの割合
METRIC_NAMES = {"front_early": "前後（序盤の位置）", "front_4c": "前後（4 角の位置）", "front_l3f": "前後（上がり 3F の遅さ）",
                "front_combo": "前後（序盤の位置 + 上がり 3F の遅さ）", "inner_a": "内外 案 A（馬番）",
                "inner_b": "内外 案 B（4 角の群の内から何頭目。単独は最も内）", "inner_b_r": "内外 案 B − 序盤の位置の分",
                "inner_b_g": "内外 案 B'（4 角の群の中の相対位置。単独は中ほど）", "inner_c": "内外 案 C（馬番 − 序盤の位置の分）"}


def mname(m: str) -> str:
    return METRIC_NAMES[m] + ("・主" if m in (Y.FRONT_MAIN, Y.INNER_MAIN) else "")
SURF = {"芝": "芝", "ダ": "ダート"}

CSS = """
:root { --bg:#f7f8fb; --panel:#ffffff; --panel2:#eef1f7; --border:#d5dbe7; --text:#1b2333; --muted:#5b6578;
  --accent:#2f5fc4; --ok:#1d7f45; --warn:#9a6200; --bad:#b4232c; --code:#eef1f7; --front:#b4232c; --back:#2f5fc4; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60;
  --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; --front:#f87171; --back:#7aa9ff; } }
:root[data-theme="dark"] { --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60; --text:#e6ecff; --muted:#9aa7c7;
  --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; --front:#f87171; --back:#7aa9ff; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 16px 80px; }
a { color:var(--accent); }
h1 { font-size:24px; margin:0 0 6px; line-height:1.4; }
h2 { font-size:19px; margin:40px 0 6px; padding-top:16px; border-top:1px solid var(--border); }
h3 { font-size:16px; margin:22px 0 4px; }
h4 { font-size:15px; margin:16px 0 4px; }
p, li { overflow-wrap:anywhere; }
.sub { color:var(--muted); font-size:13.5px; margin:0 0 8px; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; margin:8px 0 14px; border:1px solid var(--border); border-radius:8px; }
table { border-collapse:collapse; width:100%; min-width:640px; font-size:13px; background:var(--panel); }
th, td { border-bottom:1px solid var(--border); border-right:1px solid var(--border); padding:5px 8px; text-align:left; vertical-align:top; }
td.n { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
th { background:var(--panel2); color:var(--muted); font-weight:600; white-space:nowrap; }
code { font-family:ui-monospace,Menlo,monospace; background:var(--code); padding:1px 5px; border-radius:5px; font-size:12px; overflow-wrap:anywhere; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 16px; margin:12px 0; }
.lab { display:inline-block; font-weight:700; padding:1px 10px; border-radius:10px; border:1px solid var(--border); background:var(--panel2); }
.pos { color:var(--front); } .neg { color:var(--back); } .mu { color:var(--muted); }
details { margin:4px 0; } summary { cursor:pointer; color:var(--accent); }
"""


def esc(x) -> str:
    return html.escape(str(x))


def f(x, d=2, sign=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def zcell(z) -> str:
    cls = "pos" if (z is not None and np.isfinite(z) and z >= 1) else ("neg" if (z is not None and np.isfinite(z) and z <= -1) else "")
    return f'<td class="n {cls}">{f(z)}</td>'


def table(head: list[str], rows: list[list[str]], raw: bool = False) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(c if raw and c.startswith("<td") else f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def bullets(xs) -> str:
    return "<ul>" + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"


_SQ_ATTR = re.compile(r"\b(href|id|class)='([^'\"]*)'")


def page(title: str, body: str, desc: str) -> str:
    body = _SQ_ATTR.sub(r'\1="\2"', body)       # 本（build_docs_book.namespace）は二重引用符の id・href だけを張り替える
    return (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<meta name="description" content="{esc(desc)}"><title>{esc(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n')


# ---------------------------------------------------------------- サンプルの日の選び方（記憶で選ばない）

def pick_samples(days: pd.DataFrame, rm: pd.DataFrame) -> list[tuple[str, str, str]]:
    """固定の 4 つ + 2026 年の道悪の日 + 2017〜24 の前後の z の両端の日（レース 6 以上）。(日付, 場, 選んだ理由)。"""
    out = [(d, v, why) for d, v, why in SAMPLE_FIXED if ((days["date"] == d) & (days["venue"] == v)).any()]
    wet = rm[(rm["date"].str[:4] == "2026") & (rm["surf"] == "芝")].copy()
    wet["bad"] = wet["track_condition"].isin(["重", "不良"])
    w = wet.groupby(["date", "venue"])["bad"].agg(["sum", "size"]).reset_index().sort_values(["sum", "date"], ascending=[False, True])
    if len(w) and w.iloc[0]["sum"] > 0:
        out.append((w.iloc[0]["date"], w.iloc[0]["venue"], f"2026 年の道悪の日（芝の重・不良のレースが最も多い日。{int(w.iloc[0]['sum'])}/{int(w.iloc[0]['size'])} レース）"))
    c = days[(days["date"] >= Y.CAL_FROM) & (days["date"] <= Y.CAL_TO) & (days["n_races"] >= 6)]
    z = f"z_{Y.FRONT_MAIN}"
    hi, lo = c.loc[c[z].idxmax()], c.loc[c[z].idxmin()]
    out.append((hi["date"], hi["venue"], f"2017〜24 で前後の z が最も高い日（{SURF[hi['surf']]}・z {f(hi[z])}）"))
    out.append((lo["date"], lo["venue"], f"2017〜24 で前後の z が最も低い日（{SURF[lo['surf']]}・z {f(lo[z])}）"))
    zi = f"z_{Y.INNER_MAIN}"
    c = c[c[f"n_{Y.INNER_MAIN}"] >= 6]
    hi, lo = c.loc[c[zi].idxmax()], c.loc[c[zi].idxmin()]
    out.append((hi["date"], hi["venue"], f"2017〜24 で内外の z が最も高い日（{SURF[hi['surf']]}・z {f(hi[zi])}）"))
    out.append((lo["date"], lo["venue"], f"2017〜24 で内外の z が最も低い日（{SURF[lo['surf']]}・z {f(lo[zi])}）"))
    return out


# ---------------------------------------------------------------- 日の節

def passing(r) -> str:
    """通過順の文字列（無い・NaN は空）。"""
    v = getattr(r, "passing_order", "")
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in ("", "nan") else str(v).strip()


def pos_text(r) -> str:
    """通過順と、序盤・4 角の 番手/頭数（名札の判定に使う位置は序盤）。通過順が無ければ「通過なし」。"""
    po = passing(r)
    if not po or pd.isna(r.pos_first):
        return "通過なし"
    n = int(r.N)
    return f"通過 {po}・序盤 {int(r.pos_first)}/{n}・4 角 {int(r.pos_last)}/{n}"


def _top3_text(hh: pd.DataFrame) -> str:
    """1〜3 着の馬（同着はそのまま並べる）。"""
    t = hh[hh["fin"] <= 3].sort_values(["fin", "horse_number"])
    return " / ".join(f"{int(r.fin)}着 {int(r.horse_number)}番 {int(r.popularity) if pd.notna(r.popularity) else '?'}人気（{pos_text(r)}）"
                      for r in t.itertuples())


def horse_table(hh: pd.DataFrame) -> str:
    rows = []
    for r in hh.sort_values("fin").itertuples():
        e = r.y - r.q
        rows.append([f'<td class="n">{int(r.fin)}</td>', f'<td class="n">{int(r.horse_number)}</td>', f'<td class="n">{int(r.bracket_number)}</td>',
                     f"<td>{esc(r.horse_name)}</td>", f'<td class="n">{int(r.popularity) if pd.notna(r.popularity) else "—"}</td>',
                     f'<td class="n">{r.odds:.1f}</td>', f"<td>{esc(passing(r) or '通過なし')}</td>",
                     *([f'<td class="n">{int(r.pos_first)}/{int(r.N)}</td>', f'<td class="n">{int(r.pos_last)}/{int(r.N)}</td>',
                        f'<td class="n">{r.f_early:.2f}</td>', f'<td class="n">{r.f_4c:.2f}</td>'] if passing(r) else
                       ['<td class="n">—</td>'] * 2 + [f'<td class="n">{r.f_early:.2f}（最後方とみなす）</td>', f'<td class="n">{r.f_4c:.2f}</td>']), f'<td class="n">{"—" if pd.isna(r.lat4) or not passing(r) else f"{int(r.lat4) + 1}/{int(r.grp4)}"}</td>',
                     f'<td class="n">{r.q:.2f}</td>', f'<td class="n {"pos" if e > 0 else ""}">{f(e)}</td>',
                     f'<td class="n">{f(r.c_early)}</td>', f'<td class="n">{f(r.g_b_g)}</td>', f'<td class="n">{f(r.g_a)}</td>'])
    return table(["着順", "馬番", "枠", "馬名", "人気", "単勝", "通過順", "序盤 番手/頭数", "4 角 番手/頭数", "序盤の相対位置 f", "4 角の相対位置", "4 角の群（内から何頭目/頭数）",
                  "3 着内の見込み q", "y − q", "前らしさ c", "内らしさ（案 B'）", "内らしさ（馬番）"], rows, raw=True)


def pct(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.1f}%"


def l3f_text(ys: dict) -> str:
    """上がり 3F の版と前後の主の比べの文（数字から作る）。"""
    from src.research.track_bias.compare import TIE
    fe = ys.get(Y.FRONT_MAIN)
    if not fe:
        return ""
    xs = []
    for m in ("front_l3f", "front_combo"):
        y = ys.get(m)
        if not y or m == Y.FRONT_MAIN:
            continue
        hi = [n for k, n in (("split_half", "日の前半と後半"), ("next_day", "翌日"), ("same_meet", "同じ開催")) if y[k] > fe[k] + TIE]
        xs.append(f"「{METRIC_NAMES[m]}」は、持続で前後の主を {TIE:g} を超えて上回ったのが {('・'.join(hi) if hi else 'なし')}、"
                  f"本当の偏りの割合は {y['true_share']:.1%}（前後の主 {fe['true_share']:.1%}）")
    head = "上がり 3F を使った版は主に採らなかった。" if Y.FRONT_MAIN not in ("front_l3f", "front_combo") else "上がり 3F を使った版を主に採った。"
    return head + "。".join(xs) + "。"


def gain_text(ex: dict) -> str:
    """採った基準線の項の誤差の下がりの範囲（数字から）。"""
    g = [v["gain"] for m in ex.values() for k, v in m.items() if k != "both" and v.get("adopt")]
    if not g:
        return "基準線の追加の項は、どの値でも採らなかった。"
    return f"採った基準線の項の誤差の下がりは {min(g):.3%}〜{max(g):.3%} と小さい。決まり（0.05% 以上）で採ったが、効きは小さい。"


def best4c_text(ys: dict) -> str:
    """4 角の版の持続の位置（数字から）。"""
    if "front_4c" not in ys:
        return ""
    fr = {m: ys[m] for m in ys if m.startswith("front")}
    top = max(fr, key=lambda m: fr[m]["same_meet"])
    word = "最も高い" if top == "front_4c" else f"は {fr['front_4c']['same_meet']:.3f}（最も高いのは「{METRIC_NAMES[top]}」）"
    return f"前後の 4 角の版は、同じ開催の持続が{word if top == 'front_4c' else word}。着順の結果を含むので主の候補にしていない。"


def extras_text(extras: dict) -> str:
    """基準線に足した項の文（採った項から）。"""
    names = {"meet": "開催の何日目 × 仮柵", "firm": "クッション値・含水率"}
    fr = "・".join(names[x] for x in extras.get(Y.FRONT_MAIN, [])) or "なし"
    inn = "・".join(names[x] for x in extras.get(Y.INNER_MAIN, [])) or "なし"
    return (f"基準線に足した項は、前後の主が {fr}、内外の主が {inn}。採らなかった項（クッション値・含水率など）は、日の表に並べるだけにした。"
            "ほかの値ごとの採否は下の「作りの比較と採否」。")


def meet_text(d) -> str:
    """開催・仮柵・クッション値・含水率の 1 行（記述。名札の計算には入れていない値もある）。"""
    parts = []
    if pd.notna(d.get("kai")):
        parts.append(f"{int(d['kai'])} 回 {int(d['meet_day'])} 日目" if pd.notna(d.get("meet_day")) else f"{int(d['kai'])} 回")
    if d["surf"] == "芝" and str(d.get("rail") or "?") not in ("?", "-", "nan"):
        parts.append(f"仮柵 {d['rail']} コース")
    if d["surf"] == "芝":
        if pd.notna(d.get("cushion_value")):
            parts.append(f"クッション値 {d['cushion_value']:.1f}")
        if pd.notna(d.get("turf_moisture_goal")):
            parts.append(f"含水率（ゴール前 / 4 角）{pct(d['turf_moisture_goal'])} / {pct(d['turf_moisture_4corner'])}")
    elif pd.notna(d.get("dirt_moisture_goal")):
        parts.append(f"含水率（ゴール前 / 4 角）{pct(d['dirt_moisture_goal'])} / {pct(d['dirt_moisture_4corner'])}")
    return "開催: " + ("・".join(parts) if parts else "情報なし")


def day_section(date: str, venue: str, why: str, days, rm, hs, sg) -> str:
    out = [f"<h3>{esc(date)} {esc(venue)}</h3><p class='sub'>選んだ理由: {esc(why)}</p>"]
    for _, d in days[(days["date"] == date) & (days["venue"] == venue)].sort_values("surf", ascending=False).iterrows():
        s = d["surf"]
        out.append(f"<h4>{SURF[s]}: <span class='lab'>{esc(d['label'])}</span> <span class='mu'>レース {int(d['n_races'])}</span></h4>")
        out.append(f"<p class='sub'>{esc(meet_text(d))}</p>")
        out.append(table(["値", "z", "その日のずれ D（80% 区間）", "区分"],
                         [[esc(mname(m)), zcell(d[f"z_{m}"]), f'<td class="n">{f(d[f"D_{m}"])}（{f(d[f"lo_{m}"])}〜{f(d[f"hi_{m}"])}）</td>',
                           esc((Y.FRONT_WORDS if m.startswith("front") else Y.INNER_WORDS)[int(Y.level(d[f"z_{m}"]))]) if d[f"n_{m}"] >= Y.MIN_RACES else Y.NO_JUDGE]
                          for m in D.METRICS], raw=True))
        seg = sg[(sg["date"] == date) & (sg["venue"] == venue) & (sg["surf"] == s)]
        if len(seg):
            out.append("<p class='sub'>1 日の中の区切り（前後・序盤の位置。記述だけで区分は付けない）: "
                       + "、".join(f"{esc(r.segment)} {r.n_races} レース z {f(r.z)}" for r in seg.itertuples()) + "</p>")
        races = rm[(rm["date"] == date) & (rm["venue"] == venue) & (rm["surf"] == s)].sort_values("round")
        hd = hs[hs["race_id"].isin(races["race_id"])]
        fav = hd[(hd["popularity"] <= 3) & (hd["f_early"] <= 1 / 3)]
        lng = hd[(hd["popularity"] >= 6) & (hd["f_early"] > 2 / 3)]
        out.append(f"<p>人気と位置取り: 1〜3 番人気で序盤が前 1/3 の馬 {len(fav)} 頭の 3 着内 {fav['y'].sum():.1f} 頭（見込み {fav['q'].sum():.1f} 頭）。"
                   f"6 番人気以下で序盤が後ろ 1/3 の馬 {len(lng)} 頭の 3 着内 {lng['y'].sum():.1f} 頭（見込み {lng['q'].sum():.1f} 頭）。</p>")
        rows = []
        for r in races.itertuples():
            hh = hs[hs["race_id"] == r.race_id]
            rows.append([f'<td class="n">{int(r.round)}R</td>', f'<td class="n">{int(r.distance)}</td>', f"<td>{esc(r.grade)}</td>", f"<td>{esc(r.track_condition)}</td>",
                         f'<td class="n">{int(r.N)}</td>', zcell(getattr(r, f"SE_{Y.FRONT_MAIN}")), zcell(getattr(r, f"zc_{Y.FRONT_MAIN}")), zcell(getattr(r, "SE_front_4c")),
                         zcell(getattr(r, f"SE_{Y.INNER_MAIN}")), zcell(getattr(r, f"zc_{Y.INNER_MAIN}")),
                         zcell(getattr(r, "SE_inner_a")), zcell(getattr(r, "SE_inner_c")), f"<td>{esc(_top3_text(hh))}</td>"])
        out.append("<p class='sub'>レースごとの効き（S − E。正は前・内の馬が見込みより来た。「日の z への寄与」の和から μ/s を引くと日の z になる。色は ±1 以上）</p>")
        out.append(table(["R", "距離", "クラス", "馬場", "頭数", "前後 S−E", "前後 日の z への寄与", "前後 4 角 S−E", "内外 S−E", "内外 日の z への寄与",
                          "内外 A", "内外 C", "1〜3 着（人気・通過順・序盤と 4 角の 番手/頭数）"], rows, raw=True))
        det = []
        for r in races.itertuples():
            hh = hs[hs["race_id"] == r.race_id]
            det.append(f"<details><summary>{int(r.round)}R {int(r.distance)}m {esc(r.grade)} の馬ごとの表</summary>{horse_table(hh)}</details>")
        out.append("".join(det))
    return "".join(out)


# ---------------------------------------------------------------- 本体

def compare_section(cal: dict) -> str:
    """作りの候補の比べと採否（物差しと決まりは src/research/track_bias/compare.py の冒頭。数字を見る前に固定）。"""
    cp = cal.get("compare") or {}
    ys, ex = cp.get("yardstick") or {}, cp.get("extras_table") or {}
    out = ["<h2>作りの比較と採否</h2><p class='sub'>2016〜24 だけで比べた（2025・26 年は使っていない）。物差しと決まりは、数字を見る前に "
           "<code>src/research/track_bias/compare.py</code> の冒頭に書いた。</p>" + bullets([
               "基準線に足す項（開催の何日目 × 仮柵・クッション値と含水率）は、1 レースの値の as-of の誤差で決める。窓の誤差が 0.05% 以上下がった項を採る。"
               "偏りを基準線に吸わせると「本当の偏りの割合」は下がって見えるので、この項は割合では決めない。",
               "1 レースの値の候補は、追加の項なしの同じ基準線の上で、本当の偏りの割合（1 − 1/s²）と、3 つの持続（日の前半と後半・翌日・同じ開催のほかの日。芝・ダートで別に取って平均）で比べる。"
               "主を替えるのは、持続の 3 つのうち 2 つ以上で今の主より 0.01 を超えて高く、割合が下がらないときだけ。",
               "比べ方は、1 回目の比べの後にレビューの指摘で直した（候補ごとに基準線の項が違っていた・差の小さい勝ちも数えていた・面を混ぜていた）。直した後に比べ直した。",
           ])]
    rows = []
    for m in D.METRICS:
        y = ys.get(m)
        if not y:
            continue
        rows.append([esc(mname(m)), f'<td class="n">{y["true_share"]:.1%}</td>', f'<td class="n">{f(y["split_half"], 3)}</td>',
                     f'<td class="n">{f(y["next_day"], 3)}</td>', f'<td class="n">{f(y["same_meet"], 3)}</td>', esc("・".join(cal.get("extras", {}).get(m, [])) or "—")])
    out.append(table(["値", "本当の偏りの割合（芝・ダの平均）", "日の前半と後半の相関", "翌日との相関", "同じ開催のほかの日との相関", "基準線に足した項（主を決めた後）"], rows, raw=True))
    rows = []
    names = {"meet": "開催の何日目 × 仮柵（2018〜24 の誤差）", "firm": "クッション値・含水率（2022〜24 の誤差）"}
    for m in D.METRICS:
        for k, v in (ex.get(m) or {}).items():
            if k == "both":       # 2 つとも採った値の、合わせた誤差の確かめ
                good = v["mse"] <= min(v["singles"].values())
                rows.append([esc(mname(m)), "2 つを合わせる（2022〜24 の誤差）", f'<td class="n">{"1 つだけより良い" if good else "1 つだけより悪い"}</td>',
                             "合わせて採る" if good else "良い方の 1 つだけ採る"])
                continue
            rows.append([esc(mname(m)), esc(names.get(k, k)), f'<td class="n">{v["gain"]:+.3%}</td>'.replace("-", "−"), "採る" if v["adopt"] else "採らない"])
    out.append("<h3>基準線に足す項（誤差の下がり方。正が改善）</h3>" + table(["値", "項", "誤差の下がり", "採否"], rows, raw=True))
    def named(s: str) -> str:
        for k in sorted(METRIC_NAMES, key=len, reverse=True):
            s = s.replace(k, f"「{METRIC_NAMES[k]}」")
        return s
    notes = {k: [named(n) for n in v] for k, v in (cp.get("notes") or {}).items()}
    out.append("<h3>主の決め方の結果</h3>" + bullets(
        [f"前後: 主は「{esc(METRIC_NAMES[Y.FRONT_MAIN])}」。" + "。".join(esc(n) for n in notes.get("front", [])) + "。",
         f"内外: 主は「{esc(METRIC_NAMES[Y.INNER_MAIN])}」。" + "。".join(esc(n) for n in notes.get("inner", [])) + "。",
         gain_text(ex),
         best4c_text(ys),
         l3f_text(ys),
         "同じ開催の前の日を事前の値にして縮める案は採らなかった。前の日と同じ向きに寄せる作りなので、翌日・同じ開催の持続の物差しが作りで上がり、比べられない。"
         "名札も日ごとに独立でなくなる。",
         "芝の内回り・外回りは、表の列が 2024〜25 年にほぼ空なので比べられなかった。",
         ]))
    return "".join(out)


def write(out_dir: Path) -> str:
    cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    Y.set_main(meta)                     # 名札を作ったとき（build-all）の主。校正だけ流し直したときに食い違わない
    days = pd.read_parquet(out_dir / "daily_labels.parquet")
    rm = pd.read_parquet(out_dir / "race_metrics.parquet")
    hs = pd.read_parquet(out_dir / "horses.parquet")
    sg = pd.read_parquet(out_dir / "segments.parquet")
    made = pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M")
    judged = days[days["label"] != Y.NO_JUDGE]
    calp = judged[(judged["date"] >= Y.CAL_FROM) & (judged["date"] <= Y.CAL_TO)]
    fm, im = cal["metrics"][Y.FRONT_MAIN], cal["metrics"][Y.INNER_MAIN]
    ts = [cal["metrics"][m][s]["true_share"] for m in (Y.FRONT_MAIN, Y.INNER_MAIN) for s in ("芝", "ダ")]
    per = cal["persist"]
    corr_4c = float(calp[f"z_{Y.FRONT_MAIN}"].corr(calp["z_front_4c"]))
    agree_4c = float((calp[f"lv_{Y.FRONT_MAIN}"].clip(-1, 1) == calp["lv_front_4c"].clip(-1, 1)).mean())
    # 「強い」（|z| ≥ 2）の日のうち、偏りが無くても偶然で出る数の割合: 偶然だけなら z_raw ~ N(0, 1)、強いの線は z_raw で ≈ μ ± 2s
    exp_noise = obs = 0.0
    for s_ in ("芝", "ダ"):
        c_ = fm[s_]
        exp_noise += c_["days"] * 0.5 * (math.erfc((Y.TH_STRONG * c_["sd"] + c_["mu"]) / math.sqrt(2)) + math.erfc((Y.TH_STRONG * c_["sd"] - c_["mu"]) / math.sqrt(2)))
        obs += c_["days"] * (c_["freq"]["2"] + c_["freq"]["-2"])
    chance_strong = min(1.0, exp_noise / obs) if obs else float("nan")
    share_front = float((calp[f"lv_{Y.FRONT_MAIN}"] > 0).mean())
    share_back = float((calp[f"lv_{Y.FRONT_MAIN}"] < 0).mean())
    b = [f"<h1>トラックバイアスの判定（前後・内外）</h1><p class='sub'>作成 {made}（JST）。担当 track-bias。全部の開催日 × 場 × 芝・ダートに、1 行の名札を付けた。"
         f"対象 {judged['date'].min()}〜{judged['date'].max()} の {len(judged):,} 日（判定なしを除く）。全部の名札は <a href='track_bias_labels.html'>名札の一覧</a>。</p>"
         "<p class='box'>位置の式: 序盤の相対位置 f = (序盤の番手の順位 − 1) ÷ (頭数 − 1)（同じ番手は平均の順位）、前らしさ c = 0.5 − f。"
         "表の「序盤 x/n」の x は通過順の番手の数字そのもので、同じ番手の馬がいるときや番手が飛ぶときは、f は x から計算した値と少しずれる。前後の名札は、この数値と市場の見込みとの差だけで決まる（逃げ・差しなどの言葉の名札は使わない）。</p>"]
    b.append("<div class='box'><h2 style='margin-top:0;border:none;padding:0'>結論</h2>" + bullets([
        "その日・その場・芝ダのレースで、市場の見込み（確定単勝オッズからの 3 着内の確率）に比べて前の馬・内の馬がどれだけ来たかを測り、同じ条件のふだんの日と比べて 5 段に分けた。",
        f"名札は「内外 × 前後」の 1 文（例「内有利・前残り」「中立」）。2017〜24 の日の {share_front:.0%} が前残りの側（前残り・強い前残り）、{share_back:.0%} が差し有利の側だった。",
        f"1 日の値のばらつきのうち、本当の馬場の偏りの分は目安で {min(ts):.0%}〜{max(ts):.0%} にとどまる。残りはその日のレースの偶然のばらつきで、1 日 6 レースほどでは偏りと運を分けきれない。",
        f"そのため「強い」の名札の日のうち、目安で {chance_strong:.0%} ほどは、偏りが全く無くても偶然で同じ大きさのずれが出る数に当たる。名札は「その日の結果がそう出た」という記述で、馬場がそうだったことの証明ではない。",
        "2 つの目安は、1 レースの分散 V に日の偏りの分も少し入るので、本当の偏りの割合は小さめ・偶然の割合は大きめに出る向きがある。",
        f"翌日（土日の続き）の値との相関は 前後 {per[Y.FRONT_MAIN]['corr_next_day']:.2f}・内外 {per[Y.INNER_MAIN]['corr_next_day']:.2f} と弱い。これは次のレース・翌日を当てる道具ではない（当日の偏りで次を当てる話は RC-3 が閉じた）。",
        f"主の値は 前後「{METRIC_NAMES[Y.FRONT_MAIN]}」・内外「{METRIC_NAMES[Y.INNER_MAIN]}」。統括役の決定で内外は案 B を元にし、その変種を同じ物差しで比べて決めた（下の「作りの比較と採否」）。",
        extras_text(meta.get("extras") or {}),
    ]) + "</div>")
    b.append("<h2>判定の作り</h2>" + bullets([
        "対象: 平地の芝・ダート、発走した馬（取消・除外を除く。競走中止の馬は残して 3 着外とする）。3 着の同着は、残りの枠を同着の頭数で割る。頭数 5 未満・オッズか通過順の欠けがあるレース（直線の芝 1000m など）を除く。同じ馬が同じ日に 2 回出ている写しのレースは落とす。",
        "市場の見込み q: 確定単勝オッズの 1/オッズ を和 1 に直した勝率から、Harville の式で 3 着内の確率（レースで和 3）。y は 3 着内なら 1。",
        "前らしさ c = 0.5 − f。f は序盤の相対位置 = (序盤の番手の順位 − 1) ÷ (頭数 − 1)（序盤は通過順の最初の数字。角が 2 つのレースは 3 角。同じ番手は平均の順位）。序盤の先頭は +0.5、最後方は −0.5。",
        "逃げ・先行・差し・追込のような言葉の名札は使わない。表の位置は「番手/頭数」の数字だけで出し、前後の名札はこの数値（c）だけで決まる。",
        f"4 角の相対位置で作った版も表に並べる（日の z の相関 {corr_4c:.2f}）。主にしないのは、勝つ馬が 4 角で前へ上がって来る分だけ、4 角の位置に着順の結果が入るため。",
        "1 レースの値 S = Σ (y − q) × c。逃げた人気馬が負け（y − q が負で c が正）、後方の穴馬が来る（y − q が正で c が負）と大きく負になる。人気に逆らって位置取りが結果を動かしたレースを、そのまま数える。",
        "基準線 E（一般的な傾向）: 場 × 芝ダ × 距離帯（〜1400・〜1800・〜2200・それ以上）× クラス群（新馬・未勝利／1 勝／2・3 勝／OP 以上）× 馬場（良／稍重／重・不良）× 頭数群（〜8・9〜13・14〜）の、その日より前のレースだけの平均。"
        "空の区分が多いので、芝ダ → 芝ダ × 距離帯 → （場のコースの癖）＋（場を問わない条件の癖）へ順に縮めた（経験ベイズ）。縮める強さ k は 2018〜24 の as-of の誤差で選んだ（前後 " + str(cal["k"]["front_early"]) + "）。",
        "その日の値 D = Σ (S − E)、z = (D ÷ √ΣV − μ) ÷ s。V は前の日のレースの残り (S − E)² の平均（区分は芝ダ × 距離帯 × 頭数群）。μ と s は、2017〜24 の日（レース 3 以上）の実際の平均と SD（芝ダごと）。2016 年は基準線の溜め、2025・26 年は校正に使っていない。",
        f"区分（閾値は数字を見る前に決めた）: |z| &lt; {Y.TH_WEAK:g} は中立、{Y.TH_WEAK:g} 以上は有利、{Y.TH_STRONG:g} 以上は強い。レース {Y.MIN_RACES} 未満の日は判定なし。区間は D の 80% 区間（± 1.28 × s × √ΣV）。",
        "1 日の中の変化: 前半（〜6R）・後半（7R〜）と、その日の中で馬場の発表が替わったときの区切りの z を並べる（記述だけで区分は付けない）。"
        "区切りの z は区切りのレースだけの分散で割るので、レースごとの「日の z への寄与」を区切りで足した値とは一致しない。",
    ]))
    rows = []
    for m in D.METRICS:
        for s in ("芝", "ダ"):
            c = cal["metrics"][m][s]
            fr = c["freq"]
            rows.append([esc(mname(m)), SURF[s], f'<td class="n">{c["days"]:,}</td>'] +
                        [f'<td class="n">{fr[k]:.1%}</td>' for k in ("2", "1", "0", "-1", "-2")] +
                        [f'<td class="n">{c["sd"]:.3f}</td>', f'<td class="n">{c["true_share"]:.0%}</td>'])
    rows.append(["偶然だけのとき（標準正規）", "", '<td class="n">—</td>'] + [f'<td class="n">{NORMAL_FREQ[k]:.1%}</td>' for k in ("2", "1", "0", "-1", "-2")]
                + ['<td class="n">1.000</td>', '<td class="n">0%</td>'])
    b.append("<h2>5 段の出現率（2017〜24 の日）</h2><p class='sub'>前後は 強い前残り／前残り／中立／差し有利／強い差し有利、内外は 強い内有利／内有利／中立／外有利／強い外有利 の順。"
             "s は校正前の z の SD で、1 を超えた分（1 − 1/s²）が本当の偏りの割合の目安。</p>")
    b.append(table(["値", "芝ダ", "日の数", "強い +", "+", "中立", "−", "強い −", "s", "本当の偏りの割合"], rows, raw=True))
    b.append(f"<p class='sub'>序盤の位置と 4 角の位置の版の日の z の相関は {corr_4c:.2f}、3 段（+・中立・−）での一致は {agree_4c:.0%}。</p>")
    # 名札の分布
    rows = []
    for period, sel in (("2017〜24", calp), ("2025〜26", judged[judged["date"] >= "2025-01-01"])):
        for s in ("芝", "ダ"):
            g = sel[sel["surf"] == s]
            vc = g["label"].value_counts(normalize=True).head(8)
            rows.append([period, SURF[s], f'<td class="n">{len(g):,}</td>', "、".join(f"{esc(k)} {v:.0%}" for k, v in vc.items())])
    b.append("<h2>名札の分布</h2>" + table(["期間", "芝ダ", "日の数", "名札の割合（多い順の 8 つ）"], rows, raw=True))
    # 作りの比較と採否
    b.append(compare_section(cal))
    # 内外の案
    ca = calp[["z_inner_a", "z_inner_b", "z_inner_b_g", "z_inner_c", f"z_{Y.FRONT_MAIN}"]].corr()
    b.append("<h2>内外の案</h2>" + bullets([
        "案 A 馬番: 内らしさ g = 0.5 − 馬番の相対位置。全部のレースで取れて単純。弱点は、枠の効果が最初の角までの距離・脚質と混ざり、道中の進路ではないこと。",
        "案 B 4 角の併走の内外: 4 角の通過順の括弧（1 馬身未満の併走の群）の中は、内側の馬から書く（JRA の成績表の見方）。群の中で外にいた馬ほど g が低い。実際の進路に近い。"
        "元の B は単独の馬を最も内と数えるので、単独で走ることの多い逃げ・先行の馬を内側に数えてしまう。",
        "案 B'（主）: 群の中の相対位置（内 +0.5〜外 −0.5）を使い、単独の馬は中ほど（0）とする。並びに無い馬がいるレースは欠けにする。",
        "案 C 馬番 − 序盤の位置の分: レースの中で g（案 A）を序盤の前らしさ c に回帰した残り。内枠の先行馬が残った日を「内有利」と取り違えにくい。",
        f"日の z の相関（2017〜24）: A と B {ca.loc['z_inner_a', 'z_inner_b']:.2f}、B と B' {ca.loc['z_inner_b', 'z_inner_b_g']:.2f}、A と C {ca.loc['z_inner_a', 'z_inner_c']:.2f}、B' と C {ca.loc['z_inner_b_g', 'z_inner_c']:.2f}。"
        f"前後（主）との相関は A {ca.loc['z_inner_a', f'z_{Y.FRONT_MAIN}']:.2f}・B {ca.loc['z_inner_b', f'z_{Y.FRONT_MAIN}']:.2f}・B' {ca.loc['z_inner_b_g', f'z_{Y.FRONT_MAIN}']:.2f}・C {ca.loc['z_inner_c', f'z_{Y.FRONT_MAIN}']:.2f}。",
    ]))
    # サンプル
    b.append("<h2>サンプルの日</h2><p class='sub'>z は正が前残り・内有利、負が差し有利・外有利。±1 以上に色を付けた。</p>")
    for d, v, why in pick_samples(days, rm):
        b.append(day_section(d, v, why, days, rm, hs, sg))
    b.append("<h2>限界</h2>" + bullets([
        "1 日の値の大半は偶然のばらつき。名札は結果の記述で、馬場の状態の測定ではない。",
        "市場の見込み q は確定オッズからの Harville で、人気馬の 3 着内を少し大きく見込む。基準線も同じ q で作るので、ふだんの分は打ち消し合う。",
        "芝の内回り・外回りは基準線に入れていない。表の列（course_type）が 2024〜25 年はほぼ空で、区別できない。",
        "直線の芝 1000m（通過順が無い）は対象外。",
        f"開催・仮柵の表（course_position）は {meta.get('course_info_to')} まで（名札を作った時点）。それより後の日は、開催の何日目・仮柵が「?」の群に入り、その群の過去の平均に寄る。",
        f"内外の主も、前後の主と日の z の相関が {calp[f'z_{Y.INNER_MAIN}'].corr(calp[f'z_{Y.FRONT_MAIN}']):.2f} ある（2017〜24）。逃げ・先行の馬が内に入りやすい分か、本当に内と前が同じ日に有利になるのかは、この作りでは分けられない。",
    ]))
    b.append("<h2>画面での出し方</h2>" + bullets([
        "確定したレースの結果の欄（週の予測・レース詳細・めぐ分析）に、その日・その場・その面の名札・このレースの S − E と日の z への寄与・1〜3 着の位置（通過順・序盤と 4 角の 番手/頭数）を出す。",
        "出すのは会員だけ（レースレベルと同じ。結果の API は会員だけで、匿名には出さない）。速報（確定前）には付けない。",
        "値は表示のときに計算しない。ホストが開催日の結果の後に <code>python -m src.pipeline.track_bias.serving build</code>（約 20 秒）で作り、"
        "<code>data/page_reference/track_bias/</code>（年ごとの days・races と meta.json）に置く。Cloud Run は読むだけ。その日の作りの前は「算出前」と出す。",
        "画面の注記: 「その日の結果の振り返り。1 日の値の大半は偶然のばらつきで、次のレースの予測ではない。」",
    ]))
    b.append("<h2>作り方・再現</h2>" + bullets([
        "<code>nice -n 19 python -m src.research.track_bias calibrate</code>（2016〜24 だけで k と校正を決め、<code>config/track_bias/calibration.json</code> に書く）。",
        "<code>nice -n 19 python -m src.research.track_bias build-all</code>（全期間の名札。<code>data/local/research/track_bias/daily_labels.parquet</code>・年ごとの CSV）。開催日の夜に流し直すと新しい日が足される（30 秒ほど）。",
        "<code>python -m src.research.track_bias day --date 2026-11-22 --venue 東京</code>（その日の判定とレースごとの根拠の JSON）・<code>html</code>（この文書）。",
        "費用 0 円（手元の表だけ。GCS を読まない）。2025・26 年の結果は記述として読んだ（台帳の系統 track_bias・eval_only）。",
    ]))
    desc = "全部の開催日 × 場 × 芝・ダートに「内有利・前残り」などの名札を付ける、トラックバイアスの記述の道具。市場の見込みとの差 × 位置取り・枠を、同じ条件のふだんの日と比べて 5 段に分ける。"
    p1 = HTML_DIR / "track_bias.html"
    p1.write_text(page("トラックバイアスの判定", "".join(b), desc), encoding="utf-8")
    p2 = HTML_DIR / "track_bias_labels.html"
    p2.write_text(labels_page(days, made), encoding="utf-8")
    ps = [str(p1), str(p2)]
    for y in sorted(days["date"].str[:4].unique()):
        py = HTML_DIR / LABEL_YEARS_NAME.format(y=y)
        py.write_text(year_page(y, days, rm, hs, made), encoding="utf-8")
        ps.append(str(py))
    return "\n".join(ps)


LABEL_YEARS_NAME = "track_bias_labels_{y}.html"


def labels_page(days: pd.DataFrame, made: str) -> str:
    """日付 × 場 で 1 行（芝・ダートを横に並べる）。レースごとの 1〜3 着は年ごとのページ（year_page）。
    区間などの全部の列は data/local/research/track_bias/daily_labels_<年>.csv。"""
    years = sorted(days["date"].str[:4].unique(), reverse=True)
    links = "・".join(f"<a href='{LABEL_YEARS_NAME.format(y=y)}'>{y}</a>" for y in years)
    b = [f"<h1>トラックバイアスの名札の一覧</h1><p class='sub'>作成 {made}（JST）。作りと読み方は <a href='track_bias.html'>トラックバイアスの判定</a>。"
         "z は 前後 / 内外 の順で、正が前残り・内有利。2016 年は基準線の溜め（それより前の履歴が短い）なので粗い。"
         "区間などの全部の列は <code>data/local/research/track_bias/daily_labels_&lt;年&gt;.csv</code>。</p>"
         f"<p>レースごとの 1〜3 着（人気・通過順・序盤と 4 角の 番手/頭数）は年ごとのページ: {links}</p>"]

    def cell(r) -> str:
        if r is None:
            return "<td class='mu'>—</td>"
        return (f"<td>{esc(r['label'])} <span class='mu'>{f(r[f'z_{Y.FRONT_MAIN}'], 1)} / {f(r[f'z_{Y.INNER_MAIN}'], 1)}・{int(r['n_races'])}R</span></td>")

    last = years[0]
    for y, g in sorted(days.groupby(days["date"].str[:4]), reverse=True):
        rows = []
        for (d, v), gg in sorted(g.groupby(["date", "venue"]), key=lambda kv: (kv[0][0], kv[0][1]), reverse=True):
            by = {r["surf"]: r for _, r in gg.iterrows()}
            rows.append([f"<a href='{LABEL_YEARS_NAME.format(y=y)}#d{d}-{esc(v)}'>{esc(d)}</a>", esc(v), cell(by.get("芝")), cell(by.get("ダ"))])
        b.append(f"<details{' open' if y == last else ''}><summary>{y} 年（{len(rows):,} 開催日 × 場）</summary>"
                 + table(["日付", "場", "芝（名札・z・レース数）", "ダート（名札・z・レース数）"], rows, raw=True) + "</details>")
    return page("トラックバイアスの名札の一覧", "".join(b), "全部の開催日 × 場 × 芝・ダートのトラックバイアスの名札（前後・内外）の一覧。")


def year_page(y: str, days: pd.DataFrame, rm: pd.DataFrame, hs: pd.DataFrame, made: str) -> str:
    """その年の 日 × 場 × 芝ダ の名札と、レースごとの 1〜3 着（人気・通過順・序盤と 4 角の 番手/頭数）。"""
    dy = days[days["date"].str[:4] == y]
    ry = rm[rm["date"].str[:4] == y]
    hy = hs[hs["race_id"].isin(ry["race_id"]) & (hs["fin"] <= 3)]
    top = {rid: _top3_text(g) for rid, g in hy.groupby("race_id")}
    b = [f"<h1>トラックバイアスの名札 {y} 年</h1><p class='sub'>作成 {made}（JST）。<a href='track_bias_labels.html'>名札の一覧</a>・"
         "<a href='track_bias.html'>トラックバイアスの判定</a>。位置は 番手/頭数（序盤 = 通過順の最初の数字。前後の名札はこの序盤の位置で決まる）。"
         "S − E は 前後 / 内外 で、正が前・内の馬が見込みより来た。</p>"]
    for (d, v), gg in sorted(dy.groupby(["date", "venue"]), key=lambda kv: (kv[0][0], kv[0][1]), reverse=True):
        b.append(f"<h3 id='d{d}-{esc(v)}'>{esc(d)} {esc(v)}</h3>")
        for _, dd in gg.sort_values("surf", ascending=False).iterrows():
            s = dd["surf"]
            races = ry[(ry["date"] == d) & (ry["venue"] == v) & (ry["surf"] == s)].sort_values("round")
            rows = [[f"{int(r.round)}R", f"{int(r.distance)}", esc(r.grade), esc(r.track_condition),
                     f"{f(getattr(r, f'SE_{Y.FRONT_MAIN}'))} / {f(getattr(r, f'SE_{Y.INNER_MAIN}'))}", esc(top.get(r.race_id, ""))]
                    for r in races.itertuples()]
            b.append(f"<details><summary>{SURF[s]}: {esc(dd['label'])}（前後 z {f(dd[f'z_{Y.FRONT_MAIN}'], 1)}・内外 z {f(dd[f'z_{Y.INNER_MAIN}'], 1)}・"
                     f"{int(dd['n_races'])} レース）</summary>"
                     + table(["R", "距離", "クラス", "馬場", "S − E", "1〜3 着（人気・通過順・序盤と 4 角の 番手/頭数）"], rows) + "</details>")
    return page(f"トラックバイアスの名札 {y} 年", "".join(b), f"{y} 年の開催日 × 場 × 芝・ダートのトラックバイアスの名札と、レースごとの 1〜3 着の位置。")
