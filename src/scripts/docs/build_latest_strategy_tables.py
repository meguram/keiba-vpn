"""最新の買い方の成績表（docs/html/operations/final/latest_strategy_tables.html）を、記録の JSON から作る（数字を手で写さない）。

入力（src/scripts/data/strategy_tables.py・src/research/course/course_rule_bets.py が書く）:
  docs/operations/retrain_trials/strategy_tables_a1.json（本番の推奨）
  docs/operations/retrain_trials/strategy_tables_a2.json（第 2 回の買い方）
  docs/operations/retrain_trials/strategy_tables_b.json（めぐ分析の過去データの条件）
使い方: python -m src.scripts.docs.build_latest_strategy_tables
費用 0 円（手元のファイルだけ）。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/operations/retrain_trials"
DST = ROOT / "docs/html/operations/final/latest_strategy_tables.html"
TITLE = "最新の買い方の成績表"

CSS = """
:root { --bg:#f7f8fb; --panel:#ffffff; --panel2:#eef1f7; --border:#d5dbe7; --text:#1b2333; --muted:#5b6578;
  --accent:#2f5fc4; --ok:#1d7f45; --warn:#9a6200; --bad:#b4232c; --code:#eef1f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60;
  --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; } }
:root[data-theme="dark"] { --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60; --text:#e6ecff; --muted:#9aa7c7;
  --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:28px 16px 80px; }
a { color:var(--accent); }
h1 { font-size:24px; margin:0 0 6px; line-height:1.4; }
h2 { font-size:19px; margin:40px 0 6px; padding-top:16px; border-top:1px solid var(--border); }
h3 { font-size:16px; margin:22px 0 4px; }
p, li { overflow-wrap:anywhere; }
.sub { color:var(--muted); font-size:13.5px; margin:0 0 8px; }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; margin:8px 0 14px; border:1px solid var(--border); border-radius:8px; }
table { border-collapse:collapse; width:100%; min-width:760px; font-size:13.5px; background:var(--panel); }
th, td { border-bottom:1px solid var(--border); border-right:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; }
td.n { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
th { background:var(--panel2); color:var(--muted); font-weight:600; white-space:nowrap; }
tr:last-child td { border-bottom:none; }
code { font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; background:var(--code); padding:1px 5px; border-radius:5px; font-size:12px; overflow-wrap:anywhere; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 16px; margin:12px 0; }
.b { display:inline-block; font-size:11.5px; padding:0 8px; border-radius:10px; border:1px solid currentColor; white-space:nowrap; }
.ok { color:var(--ok); } .warn { color:var(--warn); } .bad { color:var(--bad); } .mu { color:var(--muted); }
details { background:var(--panel); border:1px solid var(--border); border-radius:10px; margin:8px 0; }
summary { cursor:pointer; padding:8px 12px; }
details[open] > summary { border-bottom:1px solid var(--border); }
.dbody { padding:8px 14px 12px; font-size:13.5px; }
"""

COLS = ["買い方・期間", "参戦／対象レース", "的中率（95% 区間）", "回収率（95% 区間）", "◎（該当馬）1-2-3-着外", "勝率", "複勝率"]


def esc(x) -> str:
    return html.escape(str(x))


def pct(x, d: int = 1) -> str:
    return "－" if x is None else f"{x * 100:.{d}f}%"


def rng(v, lo, hi) -> str:
    if v is None:
        return "－"
    if lo is None or hi is None:
        return pct(v)
    return f"{pct(v)}（{pct(lo, 0)}〜{pct(hi, 0)}）"


def num(x, d: int = 0) -> str:
    if x is None:
        return "－"
    return f"{x:,.{d}f}"


def rec(r: list | None, d: int = 0) -> str:
    return "－" if not r else "-".join(num(x, d) for x in r)


def roi_cls(v) -> str:
    if v is None:
        return ""
    return "ok" if v >= 1.0 else ("warn" if v >= 0.9 else "bad")


def row(label: str, cells: list[str]) -> str:
    tds = "".join(f'<td class="n">{c}</td>' for c in cells)
    return f"<tr><td>{label}</td>{tds}</tr>"


def table(rows: list[str], cols: list[str] = COLS) -> str:
    th = "".join(f"<th>{esc(c)}</th>" for c in cols)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def roi_cell(v, lo, hi) -> str:
    return f'<span class="{roi_cls(v)}">{rng(v, lo, hi)}</span>'


# ---------------------------------------------------------------- 行を作る
def a2_rows(a2: dict) -> tuple[list[str], dict]:
    names = {"train": "訓練 2019〜22", "eval": "評価 2023〜24", "test2025": "test 2025（2 回目の確かめ）", "main2026": "2026 年 主（{}）".format("〜".join((a2["periods"].get("main2026") or {}).get("dates") or ["?", "?"]))}
    out, by = [], {}
    for k, label in names.items():
        p = a2["periods"].get(k)
        if not p:
            continue
        m, h = p["metrics"], p["honmei"]
        r = row(esc(label) + ("" if p.get("reproduced") else ' <span class="b bad">記録と不一致</span>'),
                [f'{num(m["bet_races"])}／{num(m["races"])}', rng(m.get("hit_rate"), m.get("hit_lo"), m.get("hit_hi")),
                 roi_cell(m.get("roi"), m.get("roi_lo"), m.get("roi_hi")), rec(h["rec"], 1), pct(h["win"]), pct(h["place"])])
        out.append(r)
        by[k] = r
    return out, by


def a1_rows(a1: dict) -> tuple[list[str], dict]:
    out, by = [], {}
    v = a1.get("validation")
    if v:
        for k, label in (("pooled_2022_2024", "検証 2022〜24（旧モデル・設定を選んだ期間）"), ("2025", "検証 2025（旧モデル）")):
            m = v[k]
            out.append(row(esc("target ・" + label), ["{}／－".format(num(m["bet_races"])), pct(m.get("hit_rate")),
                                                     roi_cell(m.get("roi"), m.get("roi_lo"), m.get("roi_hi")), "記録なし", "－", "－"]))
    for key in ("sealed2026", "forward"):
        p = a1["periods"].get(key)
        if not p:
            continue
        t, lg, hm, t1 = p["target"], p["legacy"], p["honmei"], p["top1"]
        lab = p["label"]
        ta = t["all"]
        r = row(esc(f"target（本番の既定）・{lab}"),
                [f'{num(t["bought_races"])}／{num(t["races_ok"])}', "－" if not ta["bet_races"] else rng(ta["hit_rate"], ta["hit_lo"], ta["hit_hi"]),
                 "買い目なし（全レース推奨度 0）" if not ta["bet_races"] else roi_cell(ta["roi"], ta["roi_lo"], ta["roi_hi"]),
                 f'{rec(hm["rec"])}（◎ {num(hm["n"])} 頭）', pct(hm["win"]), pct(hm["place"])])
        out.append(r)
        by[f"t_{key}"] = r
        la = lg["all"]
        r = row(esc(f"従来方式（見送りも買った・参考）・{lab}"),
                [f'{num(la["bet_races"])}／{num(lg["races_ok"])}', rng(la["hit_rate"], la["hit_lo"], la["hit_hi"]),
                 roi_cell(la["roi"], la["roi_lo"], la["roi_hi"]), f'予測 1 位 {rec(t1["rec"])}', pct(t1["win"]), pct(t1["place"])])
        out.append(r)
        by[f"l_{key}"] = r
    return out, by


def a1_level_rows(a1: dict) -> list[str]:
    p = a1["periods"].get("sealed2026")
    if not p:
        return []
    rows = []
    for lv, s in sorted(p["legacy"]["by_level"].items(), key=lambda x: -int(x[0])):
        rows.append(row(esc(f"推奨度 {lv}" + ("（★なし）" if lv == "0" else "")),
                        [num(s["bet_races"]), rng(s["hit_rate"], s["hit_lo"], s["hit_hi"]), roi_cell(s["roi"], s["roi_lo"], s["roi_hi"])]))
    return rows


def a1_type_rows(a1: dict) -> list[str]:
    p = a1["periods"].get("sealed2026")
    if not p:
        return []
    jp = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連", "wide": "ワイド", "umatan": "馬単", "sanrenpuku": "3連複", "sanrentan": "3連単"}
    return [row(esc(jp.get(k, k)), [num(s["bet_races"]), rng(s["hit_rate"], s["hit_lo"], s["hit_hi"]), roi_cell(s["roi"], s["roi_lo"], s["roi_hi"])])
            for k, s in p["legacy"]["by_type"].items()]


def b_rows(b: dict, kind: str) -> tuple[list[str], dict]:
    names = {"train": "訓練 2016〜24（条件を見つけた期間・当てはめ）", "2025": "2025 年（第 3 版を選んだ期間・何度か見た）",
             "2026": f'2026 年（{b["values"].get("2026", {}).get("date_min", "")}〜{b["values"].get("2026", {}).get("date_max", "")}）'}
    out, by = [], {}
    for k, label in names.items():
        v = b["values"].get(k)
        if not v:
            continue
        s = v[kind]
        hl, hh = _wilson(s["hit"], s["bets"])
        r = row(esc(label), [f'{num(s["bet_races"])}／{num(v["races"])}（{num(s["bets"])} 点）', rng(s["hit_rate"], hl, hh),
                             roi_cell(s["roi"], s["roi_lo"], s["roi_hi"]), rec(s["rec"]), pct(s["win"]), pct(s["place"])])
        out.append(r)
        by[k] = r
    return out, by


def _wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


# ---------------------------------------------------------------- ページ
def build() -> str:
    a1 = json.loads((SRC / "strategy_tables_a1.json").read_text(encoding="utf-8"))
    a2 = json.loads((SRC / "strategy_tables_a2.json").read_text(encoding="utf-8"))
    b = json.loads((SRC / "strategy_tables_b.json").read_text(encoding="utf-8"))
    RS = json.loads((ROOT / "data/meta/course/rule_stats_v3.json").read_text(encoding="utf-8"))
    r1, b1 = a1_rows(a1)
    r2, b2 = a2_rows(a2)
    rbt, bbt = b_rows(b, "tansho")
    rbf, bbf = b_rows(b, "fukusho")

    def relabel(r: str, label: str) -> str:
        return "<tr><td>" + label + "</td>" + r.split("</td>", 1)[1]

    d26 = f'{(b["values"]["2026"].get("date_min") or "?")[5:].replace("-", "/")}〜{(b["values"]["2026"].get("date_max") or "?")[5:].replace("-", "/")}'
    w26 = "〜".join(x[5:].replace("-", "/") for x in (a2["periods"]["main2026"].get("dates") or ["", ""]))
    a1p = a1["periods"]["sealed2026"]
    if not a1p["target"]["bought_races"]:
        raise SystemExit("target の推奨が 0 件。係数が無いまま作った記録の疑い（2026-11-24 の誤り）。--recompute-target-bets で作り直してから流す")
    t_share = a1p["target"]["bought_races"] / max(a1p["target"]["races_ok"], 1)
    ta1 = a1p["target"]["all"]
    t_ci = ("回収率の区間は 100% を含む" if ta1["roi_lo"] <= 1.0 <= ta1["roi_hi"] else
            "回収率の区間の上限は 100% 未満" if ta1["roi_hi"] < 1.0 else "回収率の区間の下限は 100% を超える")
    p1 = a1p["label"]
    summary = [relabel(b1["t_sealed2026"], f"A1 本番の推奨 target（v6e・{p1}）"),
               relabel(b1["l_sealed2026"], f"A1 参考: 従来方式（見送りも買った・{p1}・楽観）"),
               relabel(b2["main2026"], f"A2 第 2 回の買い方（単勝・試行中・{w26}・楽観）"),
               relabel(bbt["2026"], f"B めぐ分析の条件・単勝（{d26}）"),
               relabel(bbf["2026"], f"B めぐ分析の条件・複勝（{d26}）")]
    srcs = a1["periods"]["sealed2026"].get("target_sources") or {}
    src_txt = "・".join(f"{k} {num(v)}" for k, v in sorted(srcs.items(), key=lambda x: -x[1]))
    m26 = a2["periods"]["main2026"]["metrics"]
    v_txt = f'版ごとの回収率は {pct(m26.get("sim_roi_min"))}〜{pct(m26.get("sim_roi_max"))}' if m26.get("sim_roi_min") is not None else ""
    if RS.get("sha256_cells") != b.get("sha256_cells") or RS.get("made_at") != b.get("stats_made_at"):
        raise SystemExit("めぐ分析の条件の表（rule_stats_v3.json）が B の数え直しのときと違う。B を流し直す")
    ns = sorted((x.get("n") or 0) for u, v in RS["values"]["train"].items() for j, x in enumerate(v["buy"])
                if max(RS["values"]["train"][u]["buy"][j].get("roi_t") or 0, RS["values"]["train"][u]["buy"][j].get("roi_f") or 0) >= 1.0)
    n_med = ns[len(ns) // 2] if ns else None
    n_excl = b["rules_total"] - len(ns)
    excl_txt = (f"「100% 以上」の絞り込みで落ちたのは {num(n_excl)} 条件（{pct(n_excl / b['rules_total'])}）だけで、偏りの主因ではない。"
                if b["rules_total"] and n_excl / b["rules_total"] < 0.1 else
                f"「100% 以上」の絞り込みで {num(n_excl)} 条件が落ち、その絞り込みも偏りを足している。")
    rep_all = all(v.get("reproduced") for v in a2["periods"].values())
    if not rep_all:
        raise SystemExit("A2 の記録の回収率を再現できない期間がある。原因を確かめてから作り直す")
    rep_txt = ("訓練・評価の回収率は、道具で流し直して記録（<code>final_candidate.json</code>）と一致した。test 2025・2026 年は凍結の道具が書いたレースごとの記録から数え、記録の回収率と一致した。"
               if rep_all else "記録の回収率と一致しなかった期間がある（表の印）。その期間の値は使わない。")
    b6 = b["values"]["2026"]
    lv_cols = ["推奨度", "参戦レース", "的中率（95% 区間）", "回収率（95% 区間）"]
    ty_cols = ["券種", "参戦レース", "的中率（95% 区間）", "回収率（95% 区間）"]
    made = max(a1["made_at"], a2["made_at"], b["made_at"])[:16].replace("T", " ")
    opt = a2.get("optimistic_final_odds") or {}

    def a2r(k):
        return (a2["periods"].get(k) or {}).get("metrics", {}).get("roi")

    def bt(k):
        return b["values"][k]["tansho"]["roi"]

    cands = [("A2（単勝）", a2["periods"]["main2026"]["metrics"]), ("B 単勝", b["values"]["2026"]["tansho"]), ("B 複勝", b["values"]["2026"]["fukusho"])]
    lines = []
    if all((m.get("roi") or 0) < 1.0 for _, m in cands):
        lines.append("<li>2026 年に回収率 100% を超えた買い方は無い。</li>")
    for name, m in cands:
        hi = m.get("roi_hi")
        if hi is None:
            continue
        extra = f"区間は 5 通りの版の平均に対するもので、{v_txt}。" if name.startswith("A2") and v_txt else ""
        lo = m.get("roi_lo")
        if hi < 1.0:
            verdict = f"区間の上限 {pct(hi)}。区間は 100% を含まない。"
        elif lo is not None and lo >= 1.0:
            verdict = f"区間の下限 {pct(lo)}。区間は 100% より上。"
        else:
            verdict = f"区間 {pct(lo)}〜{pct(hi)}。100% を含み、100% 超えは確かめられない。"
        lines.append(f"<li>{name}: {verdict}{extra}</li>")
    verdict_lines = "\n".join(lines)
    led = "config/data_access/ の ledger.jsonl"
    pf, pn = a2.get("pred_sha_frozen") or "", a2.get("pred_sha_now") or ""
    pred_txt = (f"予測ファイルは凍結の後に差し替わった（sha {esc(pf[:12])} → {esc(pn[:12])}。2026 年の部分の差し替え）。" if pf != pn
                else "予測ファイルは凍結のときと同じ。")
    body = f"""
<h1>{TITLE}</h1>
<p class="sub">作成 {esc(made)}（JST）。数字は記録と道具から生成（<code>src/scripts/docs/build_latest_strategy_tables.py</code>）。A1 の 2026 年は「{esc(p1)}」と「{esc((a1["periods"].get("forward") or {}).get("label", "前向き"))}」を分ける。</p>

<div class="box">
<b>どの表が何の成績か・注意</b>
<ul>
<li><b>A1</b> 今の本番で会員に出ている推奨（v6e＋target）。2026 年の日次の記録（後から作り直した backfill が大半で、会員が実際に見た推奨の記録ではない）の集計。target の行は 2026-11-24 に作り直した記録（それまでの記録は、ホストに想定オッズの係数が無いまま作られ、全レースが推奨度 0 だった）。従来方式の行は見送りのレースも買ったとして数え、計画に発走後のオッズを使うため楽観。</li>
<li><b>A2</b> 試行中の第 2 回の買い方（r04＋第 2 段＋frozen_20261117_r2、単勝 1 点だけ）。判定は T-60 のオッズの再現 5 通りの平均、精算は確定の払戻。予測の p は確定オッズの offset を含むので、T-60 の再現の値も楽観。2026 年は推論の経路の予測で、評価だけ（事前登録の判定は「否定」）。</li>
<li><b>B</b> めぐ分析の過去データの条件（凍結の第 3 版）のうち、訓練の回収率 100% 以上の条件に当てはまる馬を買う。条件はオッズ・人気を使わない（<code>src/research/graded/catalog.py:45</code>・<code>:54</code>）ので、オッズの時点による楽観は無い。楽観の元は探し方（訓練の行は当てはめ）。訓練の行は条件を見つけた期間の当てはめの値で、成績の目安にならない。2025 年も第 3 版を選ぶのに使った期間。</li>
</ul>
</div>

<h2 id="summary">結論: 2026 年の成績（期間は行ごとに違う）</h2>
<p class="sub">◎（該当馬）は、A1 は ◎ の印（従来方式の行は予測 1 位）、A2 は買った ◎、B は買った馬。A2 の着順は 5 通りの版の平均なので小数になる。対象レースは表ごとに違う（各表の注）。</p>
{table(summary)}
<ul>
{verdict_lines}
<li>A1 の target は、2026 年に {num(a1p["target"]["bought_races"])} レース（{pct(t_share)}）で推奨した。{t_ci}（下の A1 の注）。</li>
</ul>

<h2 id="a1">A1 本番の推奨（v6e＋target）</h2>
<p class="sub">記録: <code>data/local/prediction_results/*.json</code> の写し（{num(a1["days"])} 日・モデル {esc(", ".join(a1["models"]))}）。検証の行は既定方策を決めたとき（2026-10-24）の値で、v6e より前のモデル。判定に使ったオッズは記録に無く、楽観かは確かめていない。対象レースは、target が sec77 の想定オッズのそろったレース、従来方式が計画のできたレース。</p>
{table(r1)}
<ul>
<li>target: sec77 の想定オッズがそろった {num(a1p["target"]["races_ok"])} レース（{esc(src_txt)}）のうち {num(a1p["target"]["bought_races"])} レース（{pct(t_share)}）で推奨した（検証の同じ区切りでは約 30%）。外したレース（想定オッズの基準が違う・計算できない）は {num(a1p["target"]["races_basis_missing"])}。</li>
<li>target の計画は想定オッズ（AI の予測）だけで決まり、発走後のオッズを使わない。精算は確定の払戻で、推奨した買い目を全部買ったとして数える。</li>
<li>2026-11-24 より前の記録は、ホストに想定オッズの係数（<code>odds_pred_config.json</code>）が無いまま作られ、全レースが「計画できない」の見送り（推奨度 0）になっていた。本番の画面（Cloud Run）は係数を配備のときに重ねる作りで、配る係数の sha256 はホストのものと一致した。会員の画面そのものは確かめていない。</li>
<li>◎ の印が付いたのは {num(a1p["honmei"]["n"])} レース（全 {num(a1p["races"])} レース）。記録の ◎ は予測の従来の印で、作り直しでも変えていない。会員の画面の ◎ は推奨が出たレースでは買い目から決め直すので、そのレースでは食い違う（この ◎ の成績は画面の ◎ を表さない）。予測 1 位は全レースに付く。</li>
<li>従来方式は推奨度 0（見送り）のレースも買ったとして数える。推奨度 1 以上は {num(sum(v["bet_races"] for k, v in a1p["legacy"]["by_level"].items() if k != "0"))} レースだけ。前向きの行は {num(a1["periods"].get("forward", {}).get("legacy", {}).get("estimated_races", 0))} レースが払戻の概算を含む。</li>
<li>従来方式の行は、計画に使ったオッズが発走後に保存された最新（latest {num(a1p["legacy"]["odds_phase"].get("latest", 0))}・final {num(a1p["legacy"]["odds_phase"].get("final", 0))} レース）なので楽観。</li>
</ul>
<h3>従来方式の推奨度ごと（{esc(p1)}）</h3>
{table(a1_level_rows(a1), lv_cols)}
<h3>従来方式の券種ごと（同じ期間。1 レースに複数の券種がある）</h3>
{table(a1_type_rows(a1), ty_cols)}

<h2 id="a2">A2 第 2 回の買い方（試行中・単勝だけ）</h2>
<p class="sub">凍結 <code>{esc(a2["frozen"])}</code>（hash {esc(a2["hash"])}）。未勝利・新馬で、r（市場とのずれ）の最大が上位 60% のレースだけ、2 番人気以下・勝率 5% 以上で r が最大の馬の単勝 1 点。複勝は買わない。</p>
{table(r2)}
<ul>
<li>{rep_txt}</li>
<li>楽観（確定オッズで判定）の回収率は訓練 {pct(opt.get("train"))}・評価 {pct(opt.get("eval"))}・test 2025 {pct(opt.get("test2025"))}（記録の値）。判定の時点のオッズで大きく変わる。</li>
<li>予測の p は、どの期間も確定オッズの offset を含む（凍結の <code>note_decision</code>）。T-60 の再現で判定した値も楽観に偏る。</li>
<li>訓練・評価は学習の X で作った予測（OOF）、test 2025 は test の予測。2026 年は推論の経路の予測で、入力の経路は本番に近い（p は確定の offset のまま）。</li>
<li>対象レースは、予測と払戻がそろって計算できたレース。</li>
</ul>

<h2 id="b">B めぐ分析の「過去データの条件」で買う</h2>
<p class="sub">凍結の第 3 版（sha {esc(b["sha256_cells"][:12])}）。対象は良馬場・芝ダ・重賞以外のレース。単位（コース × クラスの群）ごとに、めぐ分析に出る買いの条件（上位 5 つ、全 {num(b["rules_total"])} 条件）を見る。訓練の単勝の回収率 100% 以上の条件は {num(b["rules_buy_tansho"])}、複勝は {num(b["rules_buy_fukusho"])}。1 頭が複数の条件に当てはまっても 1 点。</p>
<h3>単勝</h3>
{table(rbt)}
<h3>複勝</h3>
{table(rbf)}
<ul>
<li>買う条件（単勝か複勝の訓練の回収率 100% 以上）は {num(len(ns))} 条件で、訓練の走数の中央値は {num(n_med)}。うち {num(b["rules_buy_small_n"])} 条件は 30 走未満。少ない走数の高い回収率は偶然の幅が大きい。</li>
<li>対象レースは良馬場・芝ダ・重賞以外。複勝の的中は払戻が出たかで数える（7 頭以下の 3 着は外れ）。</li>
<li>2026 年は {esc(b6.get("date_min"))}〜{esc(b6.get("date_max"))} の表まで。</li>
<li>条件はオッズ・人気を使わないので、オッズの時点で買う馬は変わらない（表示の条件は前日版の切り口だけ。対象を良馬場で絞るのは当日の馬場）。払戻はどちらでも確定。</li>
<li>100% を超えない理由の診断は <a href="megu_why_below_100.html">めぐ分析が 100% を超えない理由</a>。</li>
<li>重賞の「好走データ予想」（B 版）は今回は入れていない。</li>
</ul>

<h2 id="evidence">事象と根拠</h2>
<ul>
<li><b>A1</b> 事象: 作り直す前の記録では、2026 年の target が全レース見送りだった。根拠: ホストの作業ツリーに <code>config/odds_pred/odds_pred_config.json</code> が無かった。そのため <code>target_bets.all_candidates</code> が計算できず、「確率が得られない」の見送りが推奨度 0 として記録された。係数を指して本番と同じ <code>_optimize_target</code> で計算し直すと、{pct(t_share)} のレースで推奨が出た。作り直しでは区切り（<code>race_ev_threshold</code> 2.0）を変えていない。</li>
<li><b>A2</b> 事象: 訓練 {pct(a2r("train"))} → 評価 {pct(a2r("eval"))} → test 2025 {pct(a2r("test2025"))} → 2026 年 {pct(a2r("main2026"))} と下がる。根拠（仮説・分けていない）: ① 訓練で買い方を選んだことによる平均への回帰（訓練→評価→test で既に下がっている）。② 2026 年は推論の経路の予測で入力が違う（未勝利・新馬の 68 レースのうち 11 レースで買う／見送るが入れ替わった。weekend-launch の 2026-11-18 の測定を <code>docs/operations/retrain_trials/bet_optimization_20261116.md</code> から引用）。どちらがどれだけかは分けていない。</li>
<li><b>B</b> 事象: 単勝の回収率は訓練 {pct(bt("train"))} に対し 2025 年 {pct(bt("2025"))}・2026 年 {pct(bt("2026"))}。根拠: 条件は訓練の走で見つけたもので、訓練の走数の中央値は {num(n_med)} と少ない。見つけた期間の値は大きく上に偏る（選択の偏り）。{excl_txt}</li>
</ul>

<details><summary>作り方・見た回数の台帳・照合（開く）</summary><div class="dbody">
<ul>
<li>A1: <code>python -m src.scripts.data.strategy_tables a1 --dir &lt;日次の記録の写し&gt;</code> → <code>docs/operations/retrain_trials/strategy_tables_a1.json</code>。target は「推奨どおり全部買う（all）」で数える（確定オッズで選ぶ actual_ge_est は使わない）。</li>
<li>A2: <code>python -m src.scripts.data.strategy_tables a2</code> → <code>strategy_tables_a2.json</code>。{pred_txt}{"訓練・評価の回収率と参戦数は記録と一致した。" if rep_all else "記録と一致しなかった期間がある。"}</li>
<li>B: <code>python -m src.research.course.course_rule_bets</code> → <code>strategy_tables_b.json</code>。めぐ分析の表（<code>data/meta/course/rule_stats_v3.json</code>）の訓練の値で判定。</li>
<li>見た回数の台帳（<code>{led}</code>）: A2 は bet_cycles を reproduce（2025・2026 年）、A1 は model を eval_only、B は course を eval_only で申告した。どれも採否・選び直しに使わない。</li>
<li>区間: 回収率はレース単位のブートストラップ（1,000 回）、率は Wilson。A2 の区間は記録の値（同じブートストラップ）。</li>
</ul>
</div></details>
"""
    return (f'<!doctype html>\n<html lang="ja">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{TITLE}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>{body}</main>\n</body>\n</html>\n")


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(build(), encoding="utf-8")
    print("書いた", DST)


if __name__ == "__main__":
    main()
