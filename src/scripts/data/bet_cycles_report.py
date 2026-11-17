"""推奨馬券の最適化サイクルの記録（HTML・md・要約 JSON）を作る（読み取りのみ・費用 0 円）。

python -m src.scripts.data.bet_cycles_report [--tag v6e] [--from-log]

- 入力: BET_CYCLES_OUT/<tag>/cycle_XX.json（bet_cycles run の出力）、config/bet_cycles/cycle_XX.json の notes（回顧・次の仮説）、
  BET_CYCLES_OUT/w4/w4_train.json（今の推奨の印のずれ）、BET_CYCLES_OUT/<tag>/stage_*.json（方策・推奨度・test。あれば）。
- 出力:
  - docs/operations/retrain_trials/bet_cycles_log.json（git に入れる要約。--from-log ならこれだけから作り直す）
  - docs/html/operations/bet_cycles.html（1 ファイルで完結、スマホ幅、ライト／ダーク）
  - docs/operations/retrain_trials/bet_optimization_20261116.md の「自動で作る表」の節（目印の間を書き換える）
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from src.scripts.data import bet_cycles as bc

ROOT = bc.ROOT
LOG = ROOT / "docs/operations/retrain_trials/bet_cycles_log.json"
HTML = ROOT / "docs/html/operations/bet_cycles.html"
MD = ROOT / "docs/operations/retrain_trials/bet_optimization_20261116.md"
MD_BEGIN, MD_END = "<!-- bet_cycles_report:begin -->", "<!-- bet_cycles_report:end -->"
KEYS = ("h_fav1_rate", "top_fav1_rate", "sim_roi_min", "sim_roi_max", "races", "bet_races", "bet_rate", "hit_rate", "roi", "roi_lo", "roi_hi", "avg_return", "median_return", "median_hit_return", "avg_cost",
        "roi_ex_top1", "roi_ex_top5", "daily_roi_sd", "max_drawdown", "max_drawdown_days", "plus_day_rate", "days")


def _m(d: dict | None) -> dict:
    return {k: (d or {}).get(k) for k in KEYS if (d or {}).get(k) is not None}


NOTES_TAG = "v6e"         # config/bet_cycles/cycle_XX.json の notes（回顧の文）は、試しの回し（v6e）の結果を見て書いた


def _notes(tag: str, cfg: dict, rec: dict) -> dict:
    """回顧の文。設定の notes は v6e で書いたので、別の予測には config/bet_cycles/notes_<tag>.json があればそれを使う。"""
    if tag == NOTES_TAG:
        return cfg.get("notes") or rec.get("notes") or {}
    p = bc.CFG_DIR / f"notes_{tag}.json"
    if p.exists():
        return (json.loads(p.read_text(encoding="utf-8")).get(str(rec["n"])) or {})
    return {"review": f"（回顧の文は試しの回し（{NOTES_TAG}）で書いた。この予測（{tag}）の値は表を見る）"}


def build_log(tag: str) -> dict:
    """cycle_XX.json から、git に入れる大きさの要約を作る。"""
    base = bc.OUT_DIR / tag
    cycles = []
    for p in sorted(base.glob("cycle_[0-9][0-9].json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        cfg_p = bc.CFG_DIR / p.name
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        s = r.get("selected") or {}
        retro = sorted(r.get("retro") or [], key=lambda x: (x["verdict"] == "保留", -(x.get("bet_races") or 0)))
        cycles.append({
            "n": r["n"], "title": r.get("title"), "hypothesis": r.get("hypothesis"), "change": r.get("change"),
            "n_variants": r.get("n_variants"), "n_policies": r.get("n_policies"), "n_eligible": r.get("n_eligible"), "n_tried_cum": r.get("n_tried_cum"),
            "selected": {"name": s.get("name"), "family": s.get("family"), "spec": s.get("spec"), "train": _m(s.get("train")), "eval": _m(s.get("eval")),
                         "train_fin": _m(s.get("train_fin")) if s.get("train_fin") else None, "eval_fin": _m(s.get("eval_fin")) if s.get("eval_fin") else None,
                         "train_folds": s.get("train_folds"), "kind": s.get("kind"), "cells": s.get("cells")} if s else None,
            "eval_check": r.get("eval_check"), "improved": r.get("improved"), "vs_prev_selected": r.get("vs_prev_selected"), "odds_robustness": r.get("odds_robustness_train"),
            "families": [{"family": f["family"], "name": f["name"], "train": _m(f["train"]), "train_folds": f.get("train_folds"),
                          "improved": f.get("improved"), "vs_prev": f.get("vs_prev"), "prev": f.get("prev")} for f in r.get("families") or []],
            "groups": {g: rows[:8] for g, rows in (r.get("groups") or {}).items()}, "miss_types": r.get("miss_types"), "h_reasons": r.get("h_reasons"),
            "retro": [{k: x.get(k) for k in ("axis", "axis_label", "level", "bet_races", "hit_rate", "roi", "roi_lo", "roi_hi", "roi_ex_top5",
                                             "log_hit", "log_payout", "log_cost", "verdict")} for x in retro[:30]],
            "baselines": ([{"name": b["name"], "train": _m(b["train"]), "eval": _m(b.get("eval"))} for b in r.get("baselines") or []]
                          if not cycles else []),          # 基準はどのサイクルでも同じなので、最初のサイクルにだけ
            "drop": r.get("drop"), "notes": _notes(tag, cfg, r), "selection_rule": r.get("selection_rule"),
        })
    log = {"tag": tag, "cycles": cycles}
    w4 = bc.OUT_DIR / "w4" / "w4_train.json"
    if w4.exists():
        log["w4"] = json.loads(w4.read_text(encoding="utf-8"))
    for st in sorted(base.glob("stage_*.json")):
        log.setdefault("stages", {})[st.stem] = json.loads(st.read_text(encoding="utf-8"))
    return log


# ── 書式 ─────────────────────────────────────────────────────────────────


def pct(x, d=1) -> str:
    return "－" if x is None else f"{x * 100:.{d}f}%"


def yen(x) -> str:
    return "－" if x is None else f"{x:,.0f}"


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def ci(m: dict) -> str:
    return "－" if m.get("roi_lo") is None else f"{m['roi_lo'] * 100:.0f}〜{m['roi_hi'] * 100:.0f}%"


MUST_HEAD = ["全レース", "参戦（率）", "予測1位が1番人気（参戦の中）", "◎ が1番人気（参戦の中）", "的中率", "回収率", "平均回収額", "中央値回収額", "的中時の中央値", "1R の購入額", "95% 区間", "上位5件を除く",
             "1 日の回収率の標準偏差", "最大の落ち込み（円・日数分）"]


def must_cells(m: dict) -> list[str]:
    if not m or not m.get("bet_races"):
        return [yen(m.get("races")) if m else "－", "0"] + ["－"] * (len(MUST_HEAD) - 2)
    return [yen(m["races"]), f"{m['bet_races']:,}（{pct(m['bet_rate'], 0)}）", pct(m.get("top_fav1_rate"), 0), pct(m.get("h_fav1_rate"), 0), pct(m["hit_rate"]), pct(m["roi"]), yen(m["avg_return"]),
            yen(m["median_return"]), yen(m["median_hit_return"]), yen(m["avg_cost"]), ci(m), pct(m.get("roi_ex_top5")),
            f"{m.get('daily_roi_sd', 0):.2f}", f"{yen(m.get('max_drawdown'))}・{m.get('max_drawdown_days', 0):.1f}"]


def table(head: list[str], rows: list[list[str]], left: int = 1) -> str:
    th = "".join(f'<th class="l">{esc(h)}</th>' if i < left else f"<th>{esc(h)}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="l">{c}</td>' if i < left else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def md_table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c).replace("|", "／") for c in r) + " |" for r in rows]
    return "\n".join(out)


def spec_text(s: dict | None) -> str:
    """買い方の中身を短い文にする。"""
    if not s:
        return "－"
    if "policy" in s:
        ty = s["policy"].get("typing", {})
        return f"方策（型の作り方: {ty.get('name', ty.get('kind'))}、型 {len(s['policy'].get('map', {}))} 個）"
    h = s.get("honmei") or {}
    if h.get("rule") == "fav1_exc":                 # 2 番人気以下から（中の決め方を書く）。1 番人気は連対率の予測が p2_min 以上のときだけ
        inner = spec_text({**s, "honmei": h.get("base") or {"rule": "fav1_penalty", "lam": 1.0}}).split(" → ")[0].replace("◎: ", "")
        rest = spec_text({**s, "honmei": {"rule": "top"}}).split(" → ", 1)[1]
        exc = (f"1番人気の r が {h.get('r_min')} 以上" if h.get("exc_by") == "r" else f"1番人気は連対率の予測 {h.get('p2_min', 0.85)} 以上")
        return f"◎: 2番人気以下で {inner}（{exc}のときだけ例外） → {rest}"
    rule = {"top": "予測1位", "fav_maxodds": f"{h.get('fav_k', 3)}番人気以内で最高オッズ", "ana_ev": f"{h.get('fav_k', 3)}番人気より下で期待値最大",
            "ana_p3": f"{h.get('fav_k', 3)}番人気より下で複勝率最大", "ev_max": "期待値最大",
            "fav1_penalty": f"1番人気の勝率に×(1−{h.get('lam', 0.5)})のうえで勝率最大",
            "r_max": f"{h.get('fav_k', 1)}番人気より下・勝率{h.get('p_min', 0.05)}以上" + (f"・r {h['r_min']} 以上" if h.get('r_min', -9.0) > -9.0 else "") + "で r（ずれ）最大", "switch": "型で切り替え", "market_fav": "市場の1番人気"}.get(h.get("rule", "top"), h.get("rule"))
    extra = []
    if h.get("max_rank"):
        extra.append(f"予測{h['max_rank']}位以内")
    if h.get("ev_min"):
        extra.append(f"期待値{h['ev_min']}以上")
    if h.get("p3_min"):
        extra.append(f"複勝率{h['p3_min']}以上")
    if h.get("fallback") == "skip":
        extra.append("候補なしは見送り")
    p = s.get("partners") or {}
    pm = {"p3": f"複勝率上位{p.get('n', 2)}頭", "pw": f"勝率上位{p.get('n', 2)}頭", "fav": f"{p.get('fav_k', 3)}番人気以内の複勝率上位{p.get('n', 2)}頭",
          "mix": f"人気{p.get('n', 2)}頭＋穴{p.get('n_ana', 2)}頭"}.get(p.get("mode", "p3"), "")
    tk = "＋".join({"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連流し", "wide": "ワイド流し", "umatan": "馬単1着固定", "umatan_multi": "馬単マルチ",
                   "trio1": "三連複◎1頭軸", "trio2": "三連複◎○2頭軸", "trio_form": "三連複フォーメーション", "tri_first": "三連単◎1着流し",
                   "tri_form": "三連単フォーメーション"}.get(t["type"], t["type"]) for t in s.get("tickets") or [])
    sk = s.get("skip") or {}
    skt = []
    for c, (lo, hi) in (sk.get("when") or {}).items():
        skt.append(f"{c}{'≥' + str(lo) if lo is not None else ''}{'<' + str(hi) if hi is not None else ''}だけ買う")
    for k, lab in (("h_pop_min", "◎が{}番人気以下"), ("h_pop_max", "◎が{}番人気以内"), ("h_ev_min", "◎の期待値{}以上")):
        if sk.get(k) is not None:
            skt.append(lab.format(sk[k]))
    mw = f"・本線{s['main_w']}倍" if int(s.get("main_w", 1)) > 1 else ""
    return f"◎: {rule}{'（' + '・'.join(extra) + '）' if extra else ''} → 相手: {pm} → {tk}{mw}" + (f"｜見送り: {'・'.join(skt)}" if skt else "")


def verdict_badge(v: str | None) -> str:
    if not v:
        return ""
    cls = "good" if v.startswith("上向き（基準との差") else ("hold" if v.startswith("上向き") else "bad")
    return f'<span class="badge {cls}">{esc(v)}</span>'


def chart(cycles: list[dict]) -> str:
    """選んだ 1 本の訓練の回収率（区間つき）と、評価の回収率（点）。100% と B0 の線。"""
    pts = [(c["n"], c["selected"]["train"], c["selected"]["eval"] or {}) for c in cycles
           if c.get("selected") and all(c["selected"]["train"].get(k) is not None for k in ("roi", "roi_lo", "roi_hi"))]
    if not pts:
        return ""
    W, H, L, R, T, B = 640, 260, 56, 12, 16, 40
    vals = [v for _, t, e in pts for v in (t.get("roi_lo"), t.get("roi_hi"), e.get("roi")) if v is not None] + [1.0, 0.85]
    lo, hi = min(vals) - 0.05, max(vals) + 0.05
    nmax = max(20, max(n for n, _, _ in pts))
    x = lambda n: L + (W - L - R) * (n - 0.5) / nmax          # noqa: E731
    y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)       # noqa: E731
    g = []
    step = 0.1 if hi - lo < 1.2 else 0.25
    v = round(lo / step) * step
    while v <= hi:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
                 f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v * 100:.0f}%</text>')
        v += step
    b0 = cycles[0]["baselines"][0]["train"]["roi"] if cycles[0].get("baselines") else None
    g.append(f'<line x1="{L}" x2="{W - R}" y1="{y(1.0):.1f}" y2="{y(1.0):.1f}" stroke="var(--text)" stroke-dasharray="4 3"/>')
    if b0:
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{y(b0):.1f}" y2="{y(b0):.1f}" stroke="var(--base)" stroke-dasharray="2 3"/>'
                 f'<text x="{W - R}" y="{y(b0) - 4:.1f}" text-anchor="end">B0 訓練 {b0 * 100:.1f}%</text>')
    for n, t, e in pts:
        g.append(f'<line x1="{x(n):.1f}" x2="{x(n):.1f}" y1="{y(t["roi_lo"]):.1f}" y2="{y(t["roi_hi"]):.1f}" stroke="var(--accent)" stroke-width="2"/>'
                 f'<circle cx="{x(n):.1f}" cy="{y(t["roi"]):.1f}" r="4" fill="var(--accent)"><title>cycle {n} 訓練 {pct(t["roi"])}（{ci(t)}）</title></circle>')
        if e.get("roi") is not None:
            g.append(f'<rect x="{x(n) + 5:.1f}" y="{y(e["roi"]) - 4:.1f}" width="8" height="8" fill="var(--hold)"><title>cycle {n} 評価 {pct(e["roi"])}</title></rect>')
        g.append(f'<text x="{x(n):.1f}" y="{H - B + 16}" text-anchor="middle">{n}</text>')
    g.append(f'<text x="{L}" y="{H - 6}">● 訓練の回収率（縦線は 95% 区間）　■ 評価の回収率（確かめだけ）　破線 = 100%</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="サイクルごとの選んだ買い方の回収率">{"".join(g)}</svg>'


CSS = """:root{--bg:#f7f8fa;--panel:#fff;--text:#1d2330;--muted:#5b6578;--border:#d8dde6;--accent:#2f6fde;--good:#1d8a4e;--bad:#b3261e;
--hold:#a86400;--base:#5b6578;--head:#eef1f6;--grid:#e3e7ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;
--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--hold:#f0b65a;--base:#9aa6ba;--head:#1d2638;--grid:#263043}}
:root[data-theme="dark"]{--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;
--bad:#ff8a80;--hold:#f0b65a;--base:#9aa6ba;--head:#1d2638;--grid:#263043}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:1.4rem;margin:.2em 0}h2{font-size:1.15rem;margin:1.6em 0 .5em;border-bottom:1px solid var(--border);padding-bottom:.2em}
h3{font-size:1.02rem;margin:0 0 .4em}h4{font-size:.95rem;margin:.9em 0 .3em}
.muted{color:var(--muted);font-size:.88rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.card .k{color:var(--muted);font-size:.82rem}.card .v{font-size:1rem;font-weight:600}
.cycle{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;margin:12px 0}
.badge{display:inline-block;border-radius:4px;padding:0 .5em;font-size:.82rem;font-weight:600;border:1px solid currentColor}
.good{color:var(--good)}.bad{color:var(--bad)}.hold{color:var(--hold)}.base{color:var(--base)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.4em 0}
th,td{border:1px solid var(--border);padding:4px 6px;text-align:right;white-space:nowrap}
th{background:var(--head);font-weight:600}td.l,th.l{text-align:left;white-space:normal;min-width:7em}
code{font-size:.82rem;background:var(--head);padding:0 .3em;border-radius:3px;word-break:break-all}
svg{display:block;width:100%;height:auto}svg text{fill:var(--muted);font-size:11px}
ul{padding-left:1.2em}details{margin:.4em 0}summary{cursor:pointer;color:var(--accent)}"""


INTRO = """<p>ユーザー（統括役経由、2026-11-16）: 「勝率・複勝率の分布という観点から、馬券最適化のサイクルを 20 サイクル回して、その時点で一番評価データでの回収率が良いものを採用したい。
各サイクルの履歴を追えるように。全体レースのうち何レースに参戦して、的中率、回収率、平均回収額、中央値回収額がどのくらいかを明確に。」
指示書: <code>/home/jovyan/work/progress/bet_optimizer_spec.md</code>。道具: <code>src/scripts/data/bet_cycles.py</code>（記録は <code>bet_cycles_report.py</code> で作り直す）。</p>
<ul>
<li><b>作法</b>: 訓練 2022〜23 で探す・選ぶ・回顧する。評価 2024 は「変わらず上向きか」を確かめるだけ（評価の値で選び直さない。次の仮説を考えるときに評価の値を見たことは各サイクルに書く）。test 2025 は凍結の後に 1 回だけ。</li>
<li><b>サイクルの数え方</b>: cycle 1〜13 は買い方の格子（試した買い方を増やす回）、cycle 14〜20 は選び方を決めた段（選び方そのものの交差検証・推奨度の段の作り方・1 番人気の方針の適用・最終モデルでの回し直し・p_top での確かめ・2020 年の扱い・4 年への組み替え）で、選ぶ対象は増やしていない（統括役の判断で 20 サイクルとして数えた）。</li>
<li><b>予測</b>: 本番 v6e の OOF で道具と 20 サイクルの枠組みを作り（試しの回し）、最終モデル（model-improve-cycles の cycles16_fix3）の OOF で回し直す（採否はこちらを正とする）。どちらも 2022・23 の予測は LOYO のアンサンブルで、重みの当てはめに 2024 の着順も入っている（漏れは弱いが、評価 2024 は完全に手つかずではない）。最終モデルの 2022〜24 の OOF は勝率だけなので、連対率・複勝率は割引 Harville で補った（p_top 入りの OOF と 2019〜21 年の OOF が届いたら、訓練 2019〜22・評価 2023〜24 で回し直し、そこで凍結する）。</li>
<li><b>1 番人気の方針</b>（ユーザー 2026-11-16）: ◎ は 2 番人気以下から。1 番人気は、予測の連対率がしきい値以上（訓練で連対率 85% 以上になる最も緩い値）の例外のときだけ。cycle 7 以降の主な系統。最後の候補でも、この方針に合わない買い方は参考として外す（2026-11-17、test を見る前に決めた）。</li>
<li><b>オッズ</b>: 確定オッズ（締切後）で ◎ の期待値・人気を決めている。実際に買う時点のオッズとはずれる。三連複・三連単はオッズが無いので確率だけで組む。精算は確定の払戻。オッズに揺れ（σ=0.1・0.2）を足した確かめを各サイクルに付けた。</li>
<li><b>印と買い目の整合</b>: 買い目は必ず ◎ から組み、◎ は軸・1 着・本線。R1〜R8（<code>docs/ideas/betting_patterns.md</code> §4）をどの買い方にも機械的に当て、違反した買い方は採らない。</li>
<li><b>選び方</b>（cycle 1 の結果を見た後に決めた）: 候補 = 参戦 300R 以上・違反なし・訓練の 2 年とも B0 より上・払戻の上位 5 件を除いた回収率も B0 以上。順位 = 訓練の回収率の 95% 区間の下限。前の最良は系統（型 1・型 A・型 B・方策など）ごとに持ち、改善は同じ系統の前の最良との対の差の区間が 0 より上のときだけ。「参考」の系統（制約なしの穴・ボックス）は並べるだけで選ばない。</li>
<li><b>最後の候補の決め方</b>（cycle 13 の後に固定）: 「訓練で一番良く、かつ評価でも上向き」の「訓練」は、<b>選び方そのものの交差検証</b>（各サイクルの格子の中から 2022 年だけで最良を選んで 2023 年に当て、その逆も行った回収率。方策はもとから交差検証の値）で測る。物差しは min（上位 5 件を除いた回収率, 年ごとの回収率）、条件は的中率 8% 以上・参戦 300R 以上。参考の系統は除く。</li>
<li><b>評価 2024 の扱いの注意</b>: 評価 2024 は、各サイクルで選んだ 1 本（10 本ほど）について見てきた。順位付け・選び直しには使っていないが、次の仮説を考えるときに見ているので、もう手つかずの確かめではない。手つかずなのは test 2025 だけ。</li>
<li><b>金額</b>: 1 点 100 円の均等が基本、1 レースの上限 2,000 円。中央値回収額は、的中率が 50% 未満だと 0 円になる（隣に的中したレースの払戻の中央値を並べた）。</li>
</ul>"""


def render_html(log: dict) -> str:
    cyc = log["cycles"]
    best = [c for c in cyc if c.get("selected")]
    last = best[-1] if best else None
    cards = []
    if last and not any(k.startswith("stage_00") for k in (log.get("stages") or {})):   # 結論があるときは、結論の段が凍結した買い方を示す
        s = last["selected"]
        cards.append(("今いちばん新しい選んだ買い方", f"cycle {last['n']}［{esc(s['family'])}］",
                      f"{esc(spec_text(s['spec']))}<br>訓練 {pct(s['train']['roi'])}（{ci(s['train'])}）・的中 {pct(s['train']['hit_rate'])}・評価 {pct(s['eval'].get('roi'))}"))
    n_done = len(cyc) + (7 if any(k.startswith("stage_cycles14_20") for k in (log.get("stages") or {})) else 0)   # 14〜20 は選び方を決めた段
    cards.append(("進み", f"{n_done} / 20 サイクル", f"試した買い方の累計 {cyc[-1].get('n_tried_cum', 0):,} 本" if cyc else ""))
    if cyc and cyc[0].get("baselines"):
        b = cyc[0]["baselines"][0]
        cards.append(("基準 B0（予測1位の単勝・全部買う）", f"訓練 {pct(b['train']['roi'])}・評価 {pct(b['eval'].get('roi'))}",
                      f"訓練 {b['train']['races']:,}R・評価 {b['eval'].get('races', 0):,}R"))
    out = [f"<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           f"<title>推奨馬券の最適化サイクル</title><style>{CSS}</style></head><body><main>",
           "<h1>推奨馬券の最適化サイクル</h1>",
           f"<p class=\"muted\">元の記録: <code>docs/operations/retrain_trials/bet_cycles_log.json</code>（tag={esc(log['tag'])}）。計画と結果の文: "
           f"<code>docs/operations/retrain_trials/bet_optimization_20261116.md</code>。何を考えてどうなったかの過程: <a href=\"bet_optimization_story.html\">推奨馬券の最適化の過程</a></p>",
           *[render_stage(k, log["stages"][k]) for k in sorted(log.get("stages") or {}) if k.startswith("stage_00")],   # 結論（stage_00_*）は前置きより上
           INTRO,
           '<div class="cards">' + "".join(f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{v}</div><div class="muted">{m}</div></div>'
                                           for k, v, m in cards) + "</div>"]
    out.append("<h2>推移</h2>" + chart(cyc))
    rows = []
    for c in cyc:
        s = c.get("selected")
        if not s:
            rows.append([f"{c['n']}", "－", f"（選べる候補なし: {esc(c.get('title'))}）"] + ["－"] * (len(MUST_HEAD) + 2))
            continue
        ev = c.get("eval_check") or {}
        rows.append([f"{c['n']}", esc(s["family"]), esc(spec_text(s["spec"]))] + must_cells(s["train"])
                    + [pct(s["eval"].get("roi")), verdict_badge(ev.get("verdict"))])
    out.append("<h3>必須の指標（各サイクルで選んだ 1 本。訓練 2022〜23）</h3>"
               + table(["回", "系統", "買い方"] + MUST_HEAD + ["評価の回収率", "評価の確かめ"], rows, left=3))
    rows = []
    for c in cyc:
        s = c.get("selected")
        if s:
            rows.append([f"{c['n']}", esc(s["family"])] + must_cells(s["eval"]))
    out.append("<h3>同じ 1 本の評価 2024（確かめだけ。順位付けには使わない）</h3>" + table(["回", "系統"] + MUST_HEAD, rows, left=2))
    if cyc and cyc[0].get("baselines"):
        rows = [[esc(b["name"]), "訓練"] + must_cells(b["train"]) for b in cyc[0]["baselines"]]
        rows += [[esc(b["name"]), "評価"] + must_cells(b["eval"]) for b in cyc[0]["baselines"]]
        out.append("<h2>基準</h2>" + table(["基準", "期間"] + MUST_HEAD, rows, left=2))
        d = cyc[0].get("drop") or {}
        out.append(f"<p class=\"muted\">母数: 訓練の期間の全レース {yen((d.get('train') or {}).get('all_races_in_period'))}R のうち、全頭の予測と 7 券種の払戻がそろった"
                   f" {yen((d.get('train') or {}).get('computed_races'))}R（予測の欠け {yen((d.get('train') or {}).get('no_pred'))}R）。"
                   f"評価は {yen((d.get('eval') or {}).get('all_races_in_period'))}R のうち {yen((d.get('eval') or {}).get('computed_races'))}R。参戦率は計算できたレースに対する割合。</p>")
    w4 = log.get("w4")
    if w4:
        m = w4["metrics"]
        out.append("<h2>今の推奨の印のずれ（W4）</h2>"
                   f"<p>今の推奨（<code>target_bets</code> の既定プラン、上限 2,000 円）を訓練 2022〜23 の {w4['races']:,}R で手元で動かした"
                   "（勝率・連対率・複勝率は OOF、単複は確定オッズ、組の券種は本番と同じ想定オッズ）。</p><ul>"
                   f"<li>買い目のあるレース: {w4['bet_races']:,}R（{pct(w4['bet_races'] / w4['races'], 0)}）</li>"
                   f"<li>買い目から決め直した ◎ がモデルの ◎（勝率最大）と違う: <b>{pct(w4['w4_rate'])}</b></li>"
                   f"<li>モデルの ◎ がどの買い目にも入っていない: {pct(w4['model_honmei_absent_rate'])}</li>"
                   f"<li>モデルの印（勝率の順）で見た R2〜R8 のどれかの違反: {pct(w4['any_violation_rate'])}"
                   f"（{'・'.join(f'{k} {pct(v, 0)}' for k, v in w4['violation_rates'].items())}）</li></ul>"
                   + table(["今の推奨（参考）"] + MUST_HEAD, [["訓練 2022〜23"] + must_cells(m)])
                   + "<p class=\"muted\">今の推奨の設定は 2022〜2025 年を見て作られており、ここでの回収率は標本外の値ではない。比べる基準ではなく参考。</p>")
    out.append("<h2>サイクルごとの記録</h2>")
    for c in cyc:
        out.append(render_cycle(c))
    st = log.get("stages") or {}
    for k in sorted(st):
        if not k.startswith("stage_00"):
            out.append(render_stage(k, st[k]))
    for tag, other in (log.get("others") or {}).items():          # 別の予測での回し（例: 試しの回し v6e）は要約だけ
        out.append(render_other(tag, other))
    out.append("</main></body></html>")
    return "\n".join(out)


def render_cycle(c: dict) -> str:
    s = c.get("selected")
    nt = c.get("notes") or {}
    dec = '<span class="badge good">改善（系統の最良を更新）</span>' if c.get("improved") else '<span class="badge base">改善なし（系統の最良は前のまま）</span>'
    h = [f'<section class="cycle" id="c{c["n"]:02d}"><h3>cycle {c["n"]}: {esc(c.get("title"))} {dec if s else ""}</h3>',
         f"<p><b>仮説</b>: {esc(c.get('hypothesis'))}</p><p><b>変えたこと</b>: {esc(c.get('change'))}"
         f"（試した {c.get('n_variants', 0)} 本{('・方策 ' + str(c['n_policies']) + ' 本') if c.get('n_policies') else ''}、条件を満たす {c.get('n_eligible', 0)} 本、累計 {c.get('n_tried_cum', 0)} 本）</p>"]
    if s:
        h.append(f"<p><b>選んだ</b>［{esc(s['family'])}］: {esc(spec_text(s['spec']))}<br><span class=\"muted\">{esc(s['name'])}</span></p>")
        rows_ = [["訓練" + ("（発走前で判定・本番の再現）" if s.get("train_fin") else "")] + must_cells(s["train"])]
        if s.get("train_fin"):
            rows_.append(["訓練（確定オッズで判定・楽観）"] + must_cells(s["train_fin"]))
        rows_.append(["評価（確かめ" + ("・本番の再現）" if s.get("eval_fin") else "）")] + must_cells(s["eval"]))
        if s.get("eval_fin"):
            rows_.append(["評価（確定オッズで判定・楽観）"] + must_cells(s["eval_fin"]))
        h.append(table(["期間"] + MUST_HEAD, rows_))
        ev = c.get("eval_check") or {}
        fo = s.get("train_folds") or {}
        rb = c.get("odds_robustness") or {}
        if rb:
            tail = "オッズの揺れ（訓練）: " + "・".join(f"σ={k} 平均 {pct(v['mean'])}（最小 {pct(v['min'])}）" for k, v in rb.items())
        elif s["train"].get("sim_roi_min") is not None:
            tail = f"発走前の再現の版ごとの回収率（訓練）: {pct(s['train'].get('sim_roi_min'))}〜{pct(s['train'].get('sim_roi_max'))}"
        else:
            tail = ""
        h.append(f"<p>訓練の年ごと: {'・'.join(f'{y} {pct(v)}' for y, v in fo.items())}。評価の確かめ: {verdict_badge(ev.get('verdict'))} "
                 f"（B0 との差 {ev.get('diff_vs_base', 0) * 100:+.1f} 点、区間 {ev.get('diff_lo', 0) * 100:+.1f}〜{ev.get('diff_hi', 0) * 100:+.1f}）。"
                 + tail + "</p>")
        if s.get("cells"):
            rows = [[esc(x["type"]), esc(x["chosen"]), yen(x["races_train"]), yen((x.get("train_cv") or {}).get("bet_races")),
                     pct((x.get("train_cv") or {}).get("roi")), pct((x.get("train_insample") or {}).get("roi"))] for x in s["cells"]]
            h.append("<h4>型 → 買い方の対応表（方策）</h4>" + table(["型", "選んだ買い方", "訓練のレース", "交差検証の参戦", "交差検証の回収率", "標本内の回収率"], rows, left=2))
    vp = c.get("vs_prev_selected")
    if vp:
        d = vp["diff"]
        h.append(f"<p class=\"muted\">前のサイクル（cycle {vp['cycle']}・{esc(vp.get('family'))}）で選んだ 1 本（訓練 {pct(vp['train_roi'])}）との差: "
                 f"{d[0] * 100:+.1f} 点（区間 {d[1] * 100:+.1f}〜{d[2] * 100:+.1f}）。系統の「改善」の印は、同じ系統の前の最良との比べ。</p>")
    fam = c.get("families") or []
    if fam:
        rows = []
        for f in fam:
            vp = f.get("vs_prev")
            rows.append([esc(f["family"]), esc(f["name"]), f"{f['train'].get('bet_races', 0):,}", pct(f["train"].get("hit_rate")), pct(f["train"].get("roi")),
                         ci(f["train"]), pct(f["train"].get("roi_ex_top5")),
                         ("改善" if f.get("improved") else "改善なし") + (f"（前 cycle {f['prev']['cycle']} との差 {vp[0] * 100:+.1f}、区間 {vp[1] * 100:+.1f}〜{vp[2] * 100:+.1f}）"
                                                                        if vp and f.get("prev") else "（最初）")])
        h.append("<h4>系統ごとの最良（訓練）</h4>" + table(["系統", "買い方", "参戦", "的中率", "回収率", "95% 区間", "上位5件を除く", "前の最良との比べ"], rows, left=2))
    for g, rows_g in (c.get("groups") or {}).items():
        rows = [[esc(x["group"]), f"{x['train'].get('bet_races', 0):,}", pct(x["train"].get("hit_rate")), pct(x["train"].get("roi")),
                 f"{pct(x['train'].get('roi_lo'), 0)}〜{pct(x['train'].get('roi_hi'), 0)}", pct(x["train"].get("roi_ex_top5")),
                 "○" if x.get("passes_filter") else "×"] for x in rows_g]
        h.append(f"<details><summary>{esc(g)} ごとの最良（訓練。回顧用）</summary>"
                 + table([g, "参戦", "的中率", "回収率", "95% 区間", "上位5件を除く", "候補の条件"], rows) + "</details>")
    retro = c.get("retro") or []
    if retro:
        rows = [[esc(x["axis_label"]), esc(x["level"]), f"{x['bet_races']:,}", pct(x["hit_rate"]), pct(x["roi"]), f"{pct(x['roi_lo'], 0)}〜{pct(x['roi_hi'], 0)}",
                 pct(x["roi_ex_top5"]), f"{(x.get('log_hit') or 0):+.2f}", f"{(x.get('log_payout') or 0):+.2f}",
                 f'<span class="{ {"良い": "good", "悪い": "bad"}.get(x["verdict"], "hold") }">{esc(x["verdict"])}</span>'] for x in retro]
        h.append("<h4>回顧（訓練の期間だけ）</h4><p class=\"muted\">選んだ買い方を条件で分けた。分解は全体に対する対数の比（的中率の分＋的中時の払戻の分＋購入額の分＝回収率の比）。"
                 "区間が全体の回収率を含むものは「保留」（偶然の幅）。件数の多い順に、良い・悪いを先に並べた。</p>"
                 + table(["条件", "水準", "参戦", "的中率", "回収率", "95% 区間", "上位5件を除く", "的中率の分", "払戻の分", "判定"], rows, left=2))
    if c.get("miss_types"):
        tot = sum(c["miss_types"].values()) or 1
        h.append("<p><b>不的中の典型</b>: " + "・".join(f"{esc(k)} {v:,}（{v / tot:.0%}）" for k, v in c["miss_types"].items()) + "</p>")
    if c.get("h_reasons"):
        h.append("<p class=\"muted\">◎ の決まり方: " + "・".join(f"{esc(k)} {v:,}" for k, v in c["h_reasons"].items()) + "</p>")
    if nt.get("review"):
        h.append(f"<p><b>回顧のまとめ</b>: {esc(nt['review'])}</p>")
    if nt.get("decision_note"):
        h.append(f"<p class=\"muted\">{esc(nt['decision_note'])}</p>")
    if nt.get("next"):
        h.append("<p><b>次の仮説</b></p><ul>" + "".join(f"<li>{esc(x)}</li>" for x in nt["next"]) + "</ul>")
    h.append("<p class=\"muted\">評価 2024 の値は、選んだ 1 本と基準についてだけ見た。次の仮説を考えるときにこの値を見ている。</p></section>")
    return "\n".join(h)


def render_other(tag: str, log: dict) -> str:
    """別の予測での回しの要約: 各サイクルで選んだ 1 本の必須の指標と、段（selcv・推奨度など）。"""
    h = [f'<h2 id="other-{esc(tag)}">別の予測での回し（{esc(tag)}）</h2>',
         f"<p class=\"muted\">{esc(log.get('note', ''))}</p>"]
    rows = []
    for c in log["cycles"]:
        s_ = c.get("selected")
        if s_:
            rows.append([f"{c['n']}", esc(s_["family"]), esc(spec_text(s_["spec"]))] + must_cells(s_["train"]) + [pct((s_.get("eval") or {}).get("roi"))])
    h.append(table(["回", "系統", "買い方"] + MUST_HEAD + ["評価の回収率"], rows, left=3))
    for k in sorted(log.get("stages") or {}):
        h.append(render_stage(k, log["stages"][k], prefix=f"［{tag}］", level=3))
    return "\n".join(h)


def render_stage(key: str, st: dict, prefix: str = "", level: int = 2) -> str:
    """方策・推奨度・test の段（bet_cycles の stage 出力。中身は各段の html_rows を表にする）。"""
    h = [f"<h{level}>{esc(prefix)}{esc(st.get('title', key))}</h{level}>"]
    if st.get("text"):
        h.append("".join(f"<p>{esc(p)}</p>" for p in st["text"]))
    for t in st.get("tables") or []:
        h.append(f"<h4>{esc(t['title'])}</h4>" + table(t["head"], [[esc(c) for c in r] for r in t["rows"]], left=t.get("left", 1)))
    return "\n".join(h)


def render_md_tables(log: dict) -> str:
    cyc = log["cycles"]
    rows = []
    for c in cyc:
        s = c.get("selected")
        if not s:
            continue
        t, e = s["train"], s["eval"]
        rows.append([c["n"], s["family"], spec_text(s["spec"]), f"{t['races']:,}", f"{t['bet_races']:,}（{pct(t['bet_rate'], 0)}）", pct(t["hit_rate"]),
                     pct(t["roi"]), yen(t["avg_return"]), yen(t["median_return"]), yen(t["median_hit_return"]), ci(t), pct(t.get("roi_ex_top5")),
                     f"{e.get('bet_races', 0):,}（{pct(e.get('bet_rate'), 0)}）", pct(e.get("hit_rate")), pct(e.get("roi")), (c.get("eval_check") or {}).get("verdict", "")])
    out = [MD_BEGIN, "", "### 各サイクルで選んだ 1 本（自動で作る表）", "",
           md_table(["回", "系統", "買い方", "全レース（訓練）", "参戦（率）", "的中率", "回収率", "平均回収額", "中央値回収額", "的中時の中央値", "95% 区間",
                     "上位5件を除く", "評価の参戦", "評価の的中率", "評価の回収率", "評価の確かめ"], rows), ""]
    for c in cyc:
        nt = c.get("notes") or {}
        one = lambda x: " ".join(str(x or "").split())   # noqa: E731  md の箇条書きを崩さない
        out += [f"#### cycle {c['n']}: {one(c.get('title'))}", "", f"- 仮説: {one(c.get('hypothesis'))}", f"- 変えたこと: {one(c.get('change'))}"]
        if nt.get("review"):
            out.append(f"- 回顧: {one(nt['review'])}")
        for x in nt.get("next") or []:
            out.append(f"- 次: {one(x)}")
        out.append("")
    out.append(MD_END)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6e", help="主に載せる予測（サイクルの記録をすべて載せる）")
    ap.add_argument("--others", default="", help="要約だけ並べる別の予測（カンマ区切り。最後は主 = 凍結した予測、従 = v6e など）")
    ap.add_argument("--from-log", action="store_true", help="bet_cycles_log.json だけから作り直す（cycle_XX.json が無い PC 用）")
    a = ap.parse_args()
    if a.from_log:
        log = json.loads(LOG.read_text(encoding="utf-8"))
    else:
        log = build_log(a.tag)
        notes = {"v6e": "試しの回し: 本番 v6e の OOF（p_win・p_top2・p_top3）。道具と 20 サイクルの枠組みを作るために回した。採否は最終モデルの回しを正とする。",
                 "fix3": "最終モデル（model-improve-cycles の cycles16_fix3）の 2022〜24 年の OOF（勝率だけ。連対率・複勝率は Harville で補った）。",
                 "fix3p": "最終モデルの p_top2・p_top3 入りの 2022〜24 年の OOF（handoff/oof_2022_2024。2025 年 test と同じ作り）。",
                 "final": "最後の材料: 最終モデルの p_top 入りの OOF、訓練 2019〜22・評価 2023〜24。この回しで凍結し、test 2025 を 1 回だけ見る。"}
        log["note"] = notes.get(a.tag, "")
        for t in [x for x in a.others.split(",") if x]:
            o = build_log(t)
            o["note"] = notes.get(t, "")
            o["cycles"] = [{k: c.get(k) for k in ("n", "title", "selected", "eval_check", "baselines")} for c in o["cycles"]]   # 要約だけ（容量。基準の中身は最初のサイクルにだけ入っている。ほかは空）
            log.setdefault("others", {})[t] = o
        LOG.write_text(json.dumps(log, ensure_ascii=False, separators=(",", ":"), default=bc._np) + "\n", encoding="utf-8")
    HTML.write_text(render_html(log), encoding="utf-8")
    if MD.exists():
        t = MD.read_text(encoding="utf-8")
        b = t.find(MD_BEGIN)
        e = t.find(MD_END, b + len(MD_BEGIN)) if b >= 0 else -1
        if b >= 0 and e >= 0:
            MD.write_text(t[:b] + render_md_tables(log) + t[e + len(MD_END):], encoding="utf-8")
            print(f"書いた: {MD.relative_to(ROOT)}（目印の間）")
        else:
            print(f"目印が無いので書かなかった: {MD.relative_to(ROOT)}")
    print(f"書いた: {LOG.relative_to(ROOT)}・{HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
