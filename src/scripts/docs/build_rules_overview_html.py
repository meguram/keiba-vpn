"""重賞と通常レースのルールの一覧（単位ごとに、適用している規則と、訓練・2025 年・2026 年の成績）を 1 枚の HTML にする。

使い方: python -m src.scripts.docs.build_rules_overview_html [出力 HTML]
入力（計算はしない。記録・凍結・評価の結果のファイルを読むだけ）:
- 重賞: <GRADED_PICKS_DIR>/history.json（好走データ予想のタブの元データ。凍結した B の規則を 2018 年以降の各回に当てはめた記録。
  src.research.graded.run picks_history の出力）と、条件の版の台帳 data/meta/graded/rules_versions.json（あれば）。
- 通常レース: <COURSE_OUT>/v3/per_unit_v3.json（凍結した第 3 版を当て直して単位ごとに数えた値。src.research.course.v3 per_unit の出力。
  全体の値はサイクルの記録・凍結・2026 年の評価と一致を確かめてある）・<COURSE_OUT>/v3/kept_<n>.json（条件）・
  src/research/course/prereg_course_v3.json（凍結）・<COURSE_OUT>/v3/course_v3_cycles.json（事象と根拠）、
  第 1 版・第 2 版の評価（<COURSE_OUT>/eval_course.json・v2/eval_course_v2.json の per_unit。2025〜2026 年の合計）。
出力: data/meta/graded/rules_overview.html（外部の JS なし。ライト・ダーク両対応、スマホ幅）。
印の決め方（根拠つき）:
- 「テストでも 100% 超」: 2025 年・2026 年のどちらの年も、◎ の単勝の回収率（通常レースは本番の再現）が 100% を超える。
- 「訓練とテストの差が大きい」: ◎ の勝率の 95% 区間（Wilson）が、訓練と 2025〜2026 年で重ならない（偶然の幅を超えた差）。
- 「件数が少ない」: 2025〜2026 年の ◎ が 10 レース未満（1 回の高配当で回収率が大きく動く）。
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
COURSE_OUT = Path(os.environ.get("COURSE_OUT") or "/home/jovyan/work/course_out")
PICKS_DIR = Path(os.environ.get("GRADED_PICKS_DIR") or (REPO / "data/local/graded_picks"))
LEDGER = Path(os.environ.get("GRADED_RULES_VERSIONS") or (REPO / "data/meta/graded/rules_versions.json"))
PREREG3 = REPO / "src/research/course/prereg_course_v3.json"
OUT = REPO / "data/meta/graded/rules_overview.html"
FEW = 10          # 2025〜2026 年の ◎ がこれ未満なら「件数が少ない」（単位の多くは 2 年で数レース）
Z = 1.96
COURSE_RE = re.compile(r"^(.{2})(芝|ダ)(\d+)")


def wilson(k: float, n: float) -> tuple[float, float] | None:
    """勝率の 95% 区間（Wilson）。n が 0 なら None。k・n は本番の再現の平均（小数）でもよい。"""
    if not n:
        return None
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def split_course(course: str) -> tuple[str, str, str]:
    m = COURSE_RE.match(course or "")
    return (m.group(1), m.group(2), m.group(3)) if m else ("", "", "")


def pooled(a: dict | None, b: dict | None) -> dict | None:
    """2 つの年の値を ◎ のレース数で重み付けして合わせる（通常レース用。勝率・単回収。回収率は払戻の解けたレースで割る値の近似。
    単位の値は 20 通りを足した比なので、◎ の数の重みで足し戻せる）。"""
    xs = [x for x in (a, b) if x and x.get("bet")]
    if not xs:
        return None
    n = sum(x["bet"] for x in xs)
    out = {"bet": n}
    for k in ("win", "roi_t"):
        v = [(x[k], x["bet"]) for x in xs if x.get(k) is not None]
        out[k] = sum(p * w for p, w in v) / sum(w for _, w in v) if v else None
    return out


def flags(tr: dict | None, y25: dict | None, y26: dict | None, te: dict | None = None) -> dict:
    """印: over = 2025・2026 年とも単回収 100% 超、gap = 勝率の 95% 区間が訓練とテストで重ならない、few = テストの ◎ が FEW 未満。
    te = 2025〜2026 年を合わせた値（重賞は回の記録から直接数える。無ければ pooled の近似）。"""
    te = te or pooled(y25, y26)
    over = bool(y25 and y26 and (y25.get("roi_t") or 0) > 1 and (y26.get("roi_t") or 0) > 1)
    gap = False
    if tr and te and tr.get("win") is not None and te.get("win") is not None:
        a, b = wilson(tr["win"] * tr["bet"], tr["bet"]), wilson(te["win"] * te["bet"], te["bet"])
        gap = bool(a and b and (a[1] < b[0] or b[1] < a[0]))
    d_roi = (te["roi_t"] - tr["roi_t"]) if te and tr and te.get("roi_t") is not None and tr.get("roi_t") is not None else None
    return {"over": over, "gap": gap, "few": (te["bet"] if te else 0) < FEW, "test": te, "d_roi": d_roi}


# ---------------------------------------------------------------- 重賞（B・history.json）
def graded_stats(races: list[dict]) -> dict | None:
    """◎ と消しの成績（picks.stats と同じ定義: 勝率・連対率・複勝率は全回、回収率は払戻の解けた回だけで割る）。"""
    hs = [(x["B"]["honmei"], x["pay_ok"]) for x in races if x.get("B") and x["B"].get("honmei")]
    if not hs:
        return None
    fins = [h["fin"] for h, _ in hs]
    paid = [h for h, p in hs if p]
    ks = [x["B"]["kill"] for x in races if x.get("B") and x["B"].get("kill") is not None]   # picks.stats と同じ（空の消しの回も数える）
    kf = [h["fin"] for k in ks for h in k]
    n = len(hs)
    return {"bet": n, "races": len(races), "win": sum(f == 1 for f in fins) / n, "rentai": sum(f <= 2 for f in fins) / n,
            "place": sum(f <= 3 for f in fins) / n,
            "roi_t": sum(h["tan"] for h in paid) / 100 / len(paid) if paid else None,
            "roi_f": sum(h["fuku"] for h in paid) / 100 / len(paid) if paid else None,
            "kill_precision": sum(f >= 4 for f in kf) / len(kf) if kf else None, "kill_coverage": len(ks) / len(races) if races else None,
            "killed": len(kf)}


def _rule_text(r: dict | None, kind: str) -> dict:
    if not r:
        return {"filter": None, "terms": [], "tie": ""}
    rr = r.get("rule") or {}
    return {"filter": rr.get("filter"), "terms": [{"text": t["text"], "eff": t.get("eff")} for t in rr.get("terms", [])], "tie": rr.get("tie", ""),
            "k": r.get("k"), "default": bool(r.get("default")), "dayof": bool(r.get("dayof")), "kind": kind}


def graded_rows(H: dict, ledger: dict | None = None) -> list[dict]:
    rows = []
    led = (ledger or {}).get("rkeys") or {}
    for rk, v in sorted(H["rkeys"].items()):
        by = {"train": [], "2025": [], "2026": []}
        for x in v["races"]:
            y = int(str(x["date"])[:4])
            key = "train" if y <= H["periods"]["dev"][1] else str(y) if str(y) in by else None
            if key:
                by[key].append(x)
        st = {k: graded_stats(xs) for k, xs in by.items()}
        te = graded_stats(by["2025"] + by["2026"])
        venue, surf, dist = split_course(v.get("course", ""))
        newer = len(led.get(rk) or [])
        rows.append({"unit": rk, "grade": v.get("grade", ""), "venue": v.get("venue") or venue, "course": v.get("course", ""), "surf": surf, "dist": dist,
                     "honmei": _rule_text(v.get("honmei"), "honmei"), "kill": _rule_text(v.get("kill"), "kill"),
                     # 規則の文と成績は history.json の最初の版（20 サイクルの B）。台帳に新しい版があれば、名前を出さずに「あり」と添える
                     "version": "最初の版（20 サイクルの B）" + (f"。台帳に新しい版が {newer} 個あり（この表の規則・成績は最初の版）" if newer else ""),
                     "p": st, "flags": flags(st["train"], st["2025"], st["2026"], te)})
    return rows


def graded_total(H: dict) -> dict:
    allr = [x for v in H["rkeys"].values() for x in v["races"]]
    dev_end = H["periods"]["dev"][1]
    return {"train": graded_stats([x for x in allr if int(x["date"][:4]) <= dev_end]),
            "2025": graded_stats([x for x in allr if x["date"][:4] == "2025"]), "2026": graded_stats([x for x in allr if x["date"][:4] == "2026"])}


# ---------------------------------------------------------------- 通常レース（第 3 版・per_unit_v3.json）
def _cm(a: dict | None) -> dict | None:
    """per_unit の値（no_filter・optimistic・reproduction）を表の 1 期間に直す。主は本番の再現、楽観の単複の回収率も並べる。"""
    if not a:
        return None
    r, o = a["reproduction"], a["optimistic"]
    return {"bet": r.get("bet_races") or 0, "races": r.get("races"), "coverage": r.get("coverage"), "win": r.get("win"), "rentai": r.get("rentai"),
            "place": r.get("place"), "roi_t": r.get("roi_t"), "roi_f": r.get("roi_f"), "roi_t_p05": r.get("roi_t_p05"), "roi_t_p95": r.get("roi_t_p95"),
            "roi_t_trim5": r.get("roi_t_trim5"), "opt_bet": o.get("bet_races"), "opt_roi_t": o.get("roi_t"), "opt_roi_f": o.get("roi_f"),
            "kill_precision": r.get("kill_precision"), "kill_coverage": r.get("kill_coverage"), "killed": r.get("killed")}


def _cond(b: dict, names: dict, kill: bool = False) -> dict:
    txt = " × ".join(f"「{names.get(t, t)}」が {lv}" for t, lv in zip(b["types"], b["levels"]))
    e = b.get("ev") or {}
    if kill:
        return {"text": txt, "n": b.get("n"), "hyp": e.get("hyp", "")}
    return {"text": txt, "hyp": e.get("hyp", ""), "dir": e.get("dir", ""), "stab": e.get("stab", ""), "breadth": e.get("breadth", ""),
            "p": e.get("p_null"), "n": b.get("n_t"), "k": b.get("k_t"), "w": b.get("w_t"), "edge": b.get("rate"), "rt": b.get("rt_t")}


def course_rows(U: dict, kept: dict, v1: dict | None, v2: dict | None, top: int = 5) -> list[dict]:
    names = U.get("type_names") or {}
    V = U["values"]
    p1 = {x["unit"]: x for x in (v1 or {}).get("per_unit", [])}
    p2 = {x["unit"]: x for x in (v2 or {}).get("per_unit", [])}
    units = sorted(set(V["train"]["units"]) | set(V["2025"]["units"]) | set(V["2026"]["units"]) | set(kept))
    rows = []
    for u in units:
        course, grp = (u.split("|", 1) + [""])[:2]
        venue, surf, dist = split_course(course)
        cu = kept.get(u) or {"buy": [], "kill": []}
        buy = sorted(cu["buy"], key=lambda b: -(b.get("rate") or 0))
        st = {k: _cm(V[k]["units"].get(u)) for k in ("train", "2025", "2026")}
        x1, x2 = p1.get(u), p2.get(u)
        rows.append({"unit": u, "course": course, "group": grp, "venue": venue, "surf": surf, "dist": dist, "layer": cu.get("layer", ""),
                     "n_buy": len(buy), "n_kill": len(cu["kill"]), "buy": [_cond(b, names) for b in buy[:top]],
                     "kill": [_cond(k, names, kill=True) for k in sorted(cu["kill"], key=lambda k: -(k.get("n") or 0))[:top]],
                     "p": st, "flags": flags(st["train"], st["2025"], st["2026"]),
                     "v1": {"bet": x1.get("bet_races"), "roi_t": x1.get("roi_t")} if x1 else None,
                     "v2": {"bet": x2["reproduction"].get("bet_races"), "roi_t": x2["reproduction"].get("roi_t")} if x2 else None})
    return rows


COURSE_RULE = [
    ("買いの条件の候補: 訓練 2016〜2024 年の、その単位（コース × クラスの群 × 良馬場。件数の少ない単位はコース全体＝「副」の層）の走で、切り口 2 つまでの組のうち、"
    "複勝率 70% 以上・単勝か複勝の回収率 100% 超（2 走以上）のもの。"),
    ("根拠の検査（5 つすべて）: 仮説の型（能力・適性・状態・展開・人の意味の通る組）／方向（隣の水準と向きがそろう）／年ごとの安定（3 年ごとの区切りで単位の平均より上）／"
    "広さ（同じ芝ダ・距離帯・群のほかのコースでも平均より上）／偶然の見込み（二項の p ≤ 0.05）。"),
    "市場より強い条件だけ: 訓練の勝ち数 ÷ 市場の勝率（確定オッズから）の和が 1.2 以上、かつ市場の勝率どおりならその勝ち数以上が出る確率（ポアソン）が 0.05 以下。",
    "◎: 当たる買いの条件の「市場より強い度合い（勝ち数 ÷ 市場の勝率の和）」の合計が最大の馬。消しに当たる馬は ◎ にしない。合計が 0 のレースは見送り。同点は馬番の小さい馬。",
    "消し: 訓練で 20 走以上・3 着以内が 0 の、切り口 2 つまでの組（仮説の型が通るものだけ）に当たる馬。当たる馬が 3 頭以上のレースだけ消しを付け、◎ は消さない。",
    ("買う／見送り: ◎ が発走前で 3 番人気以内のときだけ単勝・複勝を買う。本番の再現 = 確定オッズから作った発走前のオッズ 20 通りで判定した平均（主に見る値）。"
    "確定オッズで判定した値（楽観）も並べる。"),
]


def threshold_rows(T: dict | None, side: str, grades: dict[str, str], layer: str = "全馬") -> list[dict]:
    """閾値のルールの版（src.research.course.threshold search の出力。T[side][人気の層]）の単位ごとの行。無ければ空。"""
    if not T or side not in T or layer not in (T[side] or {}):
        return []
    rows = []
    for u, v in sorted((T[side][layer].get("units") or {}).items()):
        name, course = (u.split("|", 1) + [""])[:2]
        if side == "course":
            course, grp = name, course
        else:
            grp = grades.get(name, "")
        venue, surf, dist = split_course(course)
        rs = v["rules"]
        best = {str(L): any(r["reached"][str(L)] and not r.get("pop_alone", {}).get(str(L)) for r in rs) for L in (0.75, 0.6, 0.5)}   # 人気だけで届く線は数えない
        if side == "graded":
            grp = v.get("grade") or grp                                                 # 格は search の単位から（history の名前と合わない重賞がある）
        rows.append({"unit": u, "course": course, "group": grp, "grade": grp, "venue": venue, "surf": surf, "dist": dist,
                     "n_disc": v["n_disc"], "n_conf": v["n_conf"], "rules": rs, "best": best, "ref": v.get("ref")})
    return rows


def build(out: Path = OUT, history: Path | None = None) -> Path:
    hp = history or (PICKS_DIR / "history.json")
    H = json.loads(hp.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else None
    U = json.loads((COURSE_OUT / "v3/per_unit_v3.json").read_text(encoding="utf-8"))
    P3 = json.loads(PREREG3.read_text(encoding="utf-8"))
    if U.get("sha256_cells") != P3["sha256_cells"] or U.get("cycle") != P3["cycle"]:
        raise SystemExit("per_unit_v3.json が凍結の版と違う（src.research.course.v3 per_unit を流し直す）")
    kept = json.loads((COURSE_OUT / "v3" / P3["cells_file"]).read_text(encoding="utf-8"))
    log = json.loads((COURSE_OUT / "v3/course_v3_cycles.json").read_text(encoding="utf-8"))
    cyc = next((c for c in log["cycles"] if c["cycle"] == P3["cycle"]), {})
    rd = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    v1, v2 = rd(COURSE_OUT / "eval_course.json"), rd(COURSE_OUT / "v2/eval_course_v2.json")
    E3 = rd(COURSE_OUT / "v3/eval_course_v3.json") or {}
    g_rows, c_rows = graded_rows(H, ledger), course_rows(U, kept, v1, v2)
    T = rd(COURSE_OUT / "threshold/search.json")
    grades = {v["rkey"]: v.get("grade", "") for v in H["rkeys"].values()}
    layers = (T or {}).get("layers") or []
    th = {s: {ly: threshold_rows(T, s, grades, ly) for ly in layers} for s in ("graded", "course")}
    th_meta = ({k: T.get(k) for k in ("made_at", "disc", "conf", "lines", "min_disc", "min_conf", "q_max", "layers", "half_life", "note")}
               | {s: {ly: {k: (T[s].get(ly) or {}).get(k) for k in ("rules_n", "tests", "pairs", "pooled_selected", "pooled_pre5", "layer_base", "levels", "why", "reached",
                                                                "outer_name")}
                      for ly in layers} for s in ("graded", "course") if s in T}) if T else None
    n_led = len((ledger or {}).get("rkeys") or {})
    data = {
        "graded": {"rows": g_rows, "total": graded_total(H), "notes": H.get("notes", []), "data_until": H.get("data_until"), "frozen_at": H.get("frozen_at"),
                   "periods": H["periods"], "ledger_n": n_led},
        "course": {"rows": c_rows, "total": {k: _cm(U["values"][k]["total"]) for k in ("train", "2025", "2026")}, "rule": COURSE_RULE,
                   "cycle": P3["cycle"], "name": P3["params"].get("name", ""), "frozen_at": P3.get("frozen_at"), "event": cyc.get("event", ""),
                   "basis": cyc.get("basis", ""), "review": cyc.get("review", ""), "made_at": U.get("made_at"),
                   "real_t60": (E3.get("all") or {}).get("real_t60"),
                   "v1_total": (v1 or {}).get("all"), "v2_total": ((v2 or {}).get("all") or {}).get("reproduction"),
                   "v1_match": sum(1 for r in c_rows if r["v1"]), "v2_match": sum(1 for r in c_rows if r["v2"]),
                   "test_bought": sum(1 for r in c_rows if (r["flags"]["test"] or {}).get("bet")),
                   "test_many": sum(1 for r in c_rows if not r["flags"]["few"])},
        "few": FEW, "th": th, "th_meta": th_meta,
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=float).replace("<", "\\u003c")
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ルールの一覧</title><style>{CSS}
.bd{{display:inline-block;font-size:11px;padding:0 6px;border-radius:8px;margin:1px 2px;border:1px solid var(--border)}}
.bd.over{{color:var(--good);border-color:var(--good)}}.bd.gap{{color:var(--bad);border-color:var(--bad)}}.bd.few{{color:var(--muted)}}
tr.u0 td{{border-top:2px solid var(--border)}}td.r{{white-space:normal;min-width:260px;max-width:420px}}.sum td,.sum th{{cursor:default}}
ul.t{{margin:4px 0;padding-left:18px}}#tb td.wrap{{min-width:120px}}
</style></head><body><main>
<h1>ルールの一覧（重賞・通常レース）</h1>
<p class="muted">最小単位ごとに、今使っている規則と、訓練の期間・2025 年・2026 年の成績を並べた。数字は記録・凍結・評価の結果のファイルから作った（新しい評価は流していない）。
作り: src/scripts/docs/build_rules_overview_html.py。関連: <a href="graded_handicapper.html">重賞のルールベース予想家</a>・<a href="graded_cycles.html">重賞の分析サイクル</a>・
<a href="../course/course_v3_cycles.html">全レース版 第 3 版</a>・<a href="../course/course_v2.html">第 1 版・第 2 版</a></p>
<div class="panel wrap"><b>注意</b><ul class="t">
<li>2025・2026 年は、まっさらな確かめではない。重賞は規則の候補の作り方を決める過程で何度か見ている。通常レースは第 1 版・第 2 版の評価で 2 回見ており、
第 3 版では 2025 年を見ながら試行錯誤した（2025 年は選んだ期間）。凍結の後に 1 回だけ見た 2026 年が、いちばん偏りの少ない値（それでも 3 回目）。</li>
<li>訓練の期間の値は、規則を選んだ期間そのもの（当てはめ）なので良く見える。</li>
<li><b>事象</b>（下見の固定の値。計画書 §12.1）: 閾値のルールの下見で、2016〜21 年に複勝率 75% を超えた 2,434 条件（通常レース）は、2016〜21 年の 80.5% から 2022〜24 年には 57.7% に下がった。
重賞は 1 レース名でも、芝ダ × 距離帯にまとめても、2022〜24 年に 75% を保つ条件が無かった。<b>根拠</b>: 小さい単位で数千のルールを探すと偶然の上振れを拾い、選んだ後は平均へ戻る。
75% の条件を「偶然でない」と言うには、補正込みで数十〜百走が要る（計画書 course_class_handicapper.md §12）。</li>
<li>件数の少ない単位は偶然の幅が大きい。各行に ◎ のレース数と勝率の 95% 区間（Wilson）を付けた。重賞は 1 重賞あたり 2025・2026 年が 1〜2 回しかない。</li>
<li>印: <span class="bd over">テストでも 100% 超</span> = 2025 年・2026 年のどちらも ◎ の単回収が 100% 超（通常レースは本番の再現）。
<span class="bd gap">差が大きい</span> = ◎ の勝率の 95% 区間が、訓練と 2025〜2026 年で重ならない（偶然の幅を超えた落ち込み・上振れ）。
<span class="bd few">件数が少ない</span> = 2025〜2026 年の ◎ が {FEW} レース未満。</li>
</ul></div>
<div class="tabs"><button data-t="graded" class="on">重賞</button><button data-t="course">通常レース</button>
<button data-t="th_graded">閾値のルール（重賞）</button><button data-t="th_course">閾値のルール（通常）</button></div>
<div id="view"></div>
<script id="data" type="application/json">{blob}</script>
<script>{JS}</script>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", "重賞", len(g_rows), "通常", len(c_rows))
    return out


JS = r"""
const D=JSON.parse(document.getElementById('data').textContent);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';const n0=v=>v==null?'—':Math.round(v).toLocaleString();
const Z=1.96;function wil(p,n){if(!n||p==null)return null;const d=1+Z*Z/n,c=(p+Z*Z/(2*n))/d,h=Z*Math.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d;return [Math.max(0,c-h),Math.min(1,c+h)]}
const PER=[['train','訓練'],['2025','2025 年'],['2026','2026 年']];
const COLS={graded:[['bet','◎ のレース','n'],['win','勝率','w'],['rentai','連対率','p'],['place','複勝率','p'],['roi_t','単回収','p'],['roi_f','複回収','p'],
  ['kill_precision','消しの的中','p'],['kill_coverage','消しを付けたレース','p']],
 course:[['bet','◎ を買ったレース','n'],['coverage','買った割合','p'],['win','勝率','w'],['rentai','連対率','p'],['place','複勝率','p'],['roi_t','単回収（本番の再現）','r'],
  ['roi_f','複回収（本番の再現）','p'],['opt_roi_t','単回収（確定オッズ）','p'],['opt_roi_f','複回収（確定オッズ）','p'],['roi_t_trim5','単回収（上位 5 件を除く）','p'],
  ['kill_precision','消しの的中','p'],['kill_coverage','消しを付けたレース','p']]};
function cell(a,k,t){if(!a)return '<td class="n">—</td>';const v=a[k];
 if(t==='n')return `<td class="n" data-v="${v??''}">${n0(v)}</td>`;
 if(t==='w'){const w=wil(v,a.bet);return `<td class="n" data-v="${v??''}">${pct(v)}${w?` <span class="muted">[${pct(w[0])}, ${pct(w[1])}]</span>`:''}</td>`}
 if(t==='r'&&a.roi_t_p05!=null)return `<td class="n" data-v="${v??''}">${pct(v)} <span class="muted">(${pct(a.roi_t_p05)}〜${pct(a.roi_t_p95)})</span></td>`;
 const c=(k.indexOf('roi')>=0&&v!=null)?(v>1?' good':''):'';return `<td class="n${c}" data-v="${v??''}">${pct(v)}</td>`}
function badges(f){return (f.over?'<span class="bd over">テストでも 100% 超</span>':'')+(f.gap?'<span class="bd gap">差が大きい</span>':'')+(f.few?'<span class="bd few">件数が少ない</span>':'')}
let TAB='graded';const ST={graded:{sk:'roi_t',sp:'2026',asc:false},course:{sk:'roi_t',sp:'2026',asc:false}};
function rules(r){if(TAB==='graded'){const h=r.honmei,k=r.kill;
  const tl=(x,w)=>x.terms.map(t=>`<li>${esc(t.text)}${t.eff!=null?` <span class="muted">（効き ${(100*t.eff).toFixed(0)}%）</span>`:''}</li>`).join('');
  return `<details><summary>◎: ${esc(h.terms.map(t=>t.text.replace(/「(.+?)」.*?(高い|低い)ほど上/,'$1が$2')).slice(0,2).join('・'))}${h.terms.length>2?' ほか':''}</summary>
  <div class="muted">${h.filter?esc(h.filter)+'。':''}${h.default?'発見の回が無く、既定の規則。':''}${h.dayof?'当日の値（馬体重・馬場など）を使う規則（前日の予想では計算しない）。':''}</div><ul class="t">${tl(h)}</ul><div class="muted">${esc(h.tie)}</div>
  <b>消し</b>（点の低い順に ${k.k??'—'} 頭）: <div class="muted">${k.filter?esc(k.filter)+'。':''}${k.default?'既定の規則。':''}${k.dayof?'当日の値を使う規則。':''}</div><ul class="t">${tl(k)}</ul>
  <div class="muted">オッズの絞り込み・見送り: なし（毎回 ◎ を買う）。版: ${esc(r.version)}</div></details>`}
 const b=r.buy.map(c=>`<li>${esc(c.text)} <span class="muted">［${esc(c.hyp)}・方向 ${esc(c.dir)}・安定 ${esc(c.stab)}・広さ ${esc(c.breadth)}・p=${c.p!=null?c.p.toFixed(3):'—'}・訓練 ${c.k}/${c.n} が 3 着内・${c.w} 勝・市場より強い度合い ${c.edge!=null?c.edge.toFixed(2):'—'}・単回収 ${pct(c.rt)}］</span></li>`).join('');
 const k=r.kill.map(c=>`<li>${esc(c.text)} <span class="muted">［${esc(c.hyp)}・訓練 ${c.n} 走で 3 着内 0］</span></li>`).join('');
 const v=[r.v1?`第 1 版 ${pct(r.v1.roi_t)}（${n0(r.v1.bet)}R）`:'',r.v2?`第 2 版 ${pct(r.v2.roi_t)}（${n0(r.v2.bet)}R・本番の再現）`:''].filter(x=>x).join('／');
 return `<details><summary>買いの条件 ${r.n_buy} 個・消しの条件 ${r.n_kill} 個${r.layer==='副'?'（副の層）':''}</summary>
 ${r.n_buy?`<div>◎ の点に使う条件（市場より強い度合いの高い順に ${r.buy.length} 個まで。［仮説の型・根拠の検査の結果・訓練の値］）</div><ul class="t">${b}</ul>`:'<div class="muted">検査を通った買いの条件が無い（◎ を付けない＝見送り）。</div>'}
 ${r.n_kill?`<div>消しの条件（走数の多い順に ${r.kill.length} 個まで）</div><ul class="t">${k}</ul>`:''}
 ${v?`<div class="muted">単位ごとの 2025〜2026 年の単回収（合計）: ${v}</div>`:''}</details>`}
function opts(sel,vals,all){const cur=sel.value;sel.innerHTML=`<option value="">${all}</option>`+vals.map(v=>`<option${v===cur?' selected':''}>${esc(v)}</option>`).join('')}
function uniq(a){return [...new Set(a.filter(x=>x))].sort()}
function summary(){const S=D[TAB];const C=COLS[TAB];const rows=PER.map(([p,l])=>`<tr><td>${l}</td>${C.map(([k,_,t])=>cell(S.total[p],k,t)).join('')}</tr>`);
 if(TAB==='course'){if(S.real_t60)rows.push(`<tr><td>2026 年の発走前オッズのあるレース（本物の T-60）</td>${C.map(([k,_,t])=>cell(Object.assign({bet:S.real_t60.bet_races},S.real_t60),k,t)).join('')}</tr>`);
  if(S.v1_total)rows.push(`<tr><td class="muted">第 1 版 2025〜2026 年</td>${C.map(([k,_,t])=>cell(Object.assign({bet:S.v1_total.bet_races},S.v1_total),k,t)).join('')}</tr>`);
  if(S.v2_total)rows.push(`<tr><td class="muted">第 2 版 2025〜2026 年（本番の再現）</td>${C.map(([k,_,t])=>cell(Object.assign({bet:S.v2_total.bet_races},S.v2_total),k,t)).join('')}</tr>`)}
 return `<div class="scroll"><table class="sum"><thead><tr><th>期間（全体の合計）</th>${C.map(c=>`<th>${c[1]}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`}
function head(){const S=D[TAB];if(TAB==='graded')return `<div class="panel wrap"><b>重賞</b>: 最小単位は重賞（レース名 × コース。${S.rows.length} 重賞）。
 この表の規則は最初の版（20 サイクルの B）${S.ledger_n?`。条件の版の台帳に新しい版のある重賞が ${S.ledger_n} あり、その重賞は好走データ予想のタブでは新しい版を使う（行に添えた）`:'。好走データ予想のタブも、今はすべての重賞でこの版を使う'}。B は 2018〜2024 年の回だけで選び、${esc(S.frozen_at)} に凍結。
 ◎ は重賞ごとの点の式が最大の馬、消しは消しの点の式の低い馬から決まった頭数。オッズの絞り込み・見送りは無く、確定オッズは結果の払戻を数えるだけに使う（本番の再現と楽観の差が無いので、1 つの値だけ）。
 訓練 = ${S.periods.dev[0]}〜${S.periods.dev[1]} 年（規則を選んだ期間）。結果は ${esc(S.data_until)} まで。<br><span class="muted">事象と根拠: 規則の候補は、データを見ない乱数の候補（型の一覧と乱数だけ）から、各重賞の 2018〜2024 年の回で ◎ の勝率と単回収の目的が最良のものを選んだ。
 型ごとの根拠の検査（方向・年ごとの安定・広さ）は通していない。各型の「効き」は、点の差のうちその型が占める割合。通常レースの第 3 版のような根拠の検査は、重賞ではまだ入れていない。</span></div>`;
 return `<div class="panel wrap"><b>通常レース</b>: 最小単位はコース × クラスの群 × 良馬場（${S.rows.length} 単位）。規則は最新の凍結の版（全レース版 第 3 版のサイクル ${S.cycle}、${esc(S.frozen_at)} に凍結）。訓練 = 2016〜2024 年。
 <ul class="t">${S.rule.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>
 <div class="muted"><b>事象</b>: ${esc(S.event)}<br><b>根拠</b>: ${esc(S.basis)}<br>2025〜2026 年に ◎ を買った単位は ${S.test_bought}（うち ${D.few} レース以上は ${S.test_many}）。ほかの単位は、検査を通った買いの条件が無いか、◎ が 3 番人気以内にならず見送り。<br>単位ごとの値は、凍結した規則を当て直して数えた（${esc(S.made_at)}。全体の合計はサイクルの記録・凍結・2026 年の評価と一致を確かめた）。
 第 1 版・第 2 版の単位ごとの値は 2025〜2026 年の合計だけ（名前の一致: 第 1 版 ${S.v1_match}・第 2 版 ${S.v2_match} 単位）。</div></div>`}
function render(){const S=D[TAB],st=ST[TAB],C=COLS[TAB];
 let h=head()+'<h2>全体の合計</h2>'+summary()+`<h2>単位ごと</h2><div class="ctl">
 <select id="fv"><option value="">会場（すべて）</option></select><select id="fc"><option value="">コース（すべて）</option></select>
 ${TAB==='graded'?'<select id="fg"><option value="">格（すべて）</option></select>':'<select id="fg"><option value="">クラスの群（すべて）</option></select>'}
 <select id="fs"><option value="">芝ダ（すべて）</option><option>芝</option><option>ダ</option></select>
 <select id="ff"><option value="">印（すべて）</option><option value="over">テストでも 100% 超</option><option value="gap">差が大きい</option><option value="many">件数が少なくない</option></select>
 <input id="fq" placeholder="名前で絞る" size="10">
 <select id="sp">${PER.map(([p,l])=>`<option value="${p}"${p===st.sp?' selected':''}>並べ替えの期間: ${l}</option>`).join('')}<option value="dd"${st.sp==='dd'?' selected':''}>並べ替え: 単回収の差（テスト − 訓練）</option></select></div>
 <div id="cnt" class="muted"></div><div class="scroll"><table id="tb"><thead><tr><th data-k="unit" data-l="単位">単位</th><th>規則</th><th>期間</th>${C.map(([k,l])=>`<th data-k="${k}" data-l="${l}">${l}</th>`).join('')}</tr></thead><tbody></tbody></table></div>`;
 document.getElementById('view').innerHTML=h;
 const fv=document.getElementById('fv'),fc=document.getElementById('fc'),fg=document.getElementById('fg');
 opts(fv,uniq(S.rows.map(r=>r.venue)),'会場（すべて）');opts(fg,uniq(S.rows.map(r=>TAB==='graded'?r.grade:r.group)),TAB==='graded'?'格（すべて）':'クラスの群（すべて）');
 const upd=()=>{opts(fc,uniq(S.rows.filter(r=>!fv.value||r.venue===fv.value).map(r=>r.course)),'コース（すべて）');draw()};
 fv.onchange=upd;['fc','fg','fs','ff'].forEach(i=>document.getElementById(i).onchange=draw);document.getElementById('fq').oninput=draw;
 const arrows=()=>document.querySelectorAll('#tb th[data-k]').forEach(th=>{th.textContent=th.dataset.l+(st.sk===th.dataset.k&&st.sp!=='dd'?(st.asc?' ▲':' ▼'):'')});
 document.getElementById('sp').onchange=e=>{st.sp=e.target.value;arrows();draw()};   // 並べ替えを変えても絞り込みは残す
 document.querySelectorAll('#tb th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(st.sk===k)st.asc=!st.asc;else{st.sk=k;st.asc=k==='unit'}
  if(st.sp==='dd'){st.sp='2026';document.getElementById('sp').value='2026'}arrows();draw()});
 arrows();upd()}
function draw(){const S=D[TAB],st=ST[TAB],C=COLS[TAB];const g=i=>document.getElementById(i).value;const q=g('fq');
 let rs=S.rows.filter(r=>(!g('fv')||r.venue===g('fv'))&&(!g('fc')||r.course===g('fc'))&&(!g('fg')||(TAB==='graded'?r.grade:r.group)===g('fg'))&&(!g('fs')||r.surf===g('fs'))
  &&(!g('ff')||(g('ff')==='many'?!r.flags.few:r.flags[g('ff')]))&&(!q||r.unit.indexOf(q)>=0));
 const key=r=>st.sp==='dd'?r.flags.d_roi:(st.sk==='unit'?r.unit:(r.p[st.sp]?r.p[st.sp][st.sk]:null));
 rs.sort((a,b)=>{const x=key(a),y=key(b);if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*(st.asc?1:-1)});
 document.getElementById('cnt').textContent=`${rs.length} / ${S.rows.length} 単位`;
 document.querySelector('#tb tbody').innerHTML=rs.map(r=>PER.map(([p,l],i)=>`<tr class="${i===0?'u0':''}">${i===0?`<td rowspan="3" class="wrap"><b>${esc(r.unit)}</b>${TAB==='graded'?`<div class="muted">${esc(r.grade)}・${esc(r.course)}</div>`:''}<div>${badges(r.flags)}</div>${r.flags.d_roi!=null?`<div class="muted">単回収の差 ${(r.flags.d_roi>=0?'+':'')+(100*r.flags.d_roi).toFixed(1)} ポイント</div>`:''}</td><td rowspan="3" class="r">${rules(r)}</td>`:''}<td>${l}</td>${C.map(([k,_,t])=>cell(r.p[p],k,t)).join('')}</tr>`).join('')).join('')}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));TAB=b.dataset.t;TAB.startsWith('th_')?renderTh():render()});
// ---- 閾値のルールの版
const THS={th_graded:{sk:'up',asc:false,ly:null},th_course:{sk:'up',asc:false,ly:null}};
const LN=['0.75','0.6','0.5'];
const pp=v=>v==null?'—':(v>=0?'+':'')+(100*v).toFixed(1);
const stc=x=>x&&x.n?`<td class="n" data-v="${x.place}">[${x.w.map(v=>Math.round(v)).join('-')}-${Math.round(x.out)}]</td><td class="n">${pct(x.place)} <span class="muted">[${pct(x.lo)}, ${pct(x.hi)}]</span></td><td class="n">${pct(x.e)}</td><td class="n${x.roi_t>1?' good':''}">${pct(x.roi_t)}</td><td class="n${x.roi_f>1?' good':''}">${pct(x.roi_f)}</td>`:'<td class="n">0 走</td><td class="n">—</td><td class="n">—</td><td class="n">—</td><td class="n">—</td>';
const upc=u=>u?`<td class="n${u.lo>0?' good':''}">${pp(u.pt)} <span class="muted">[${pp(u.lo)}, ${pp(u.hi)}]</span></td>`:'<td class="n">—</td>';
function lineBadges(x){
 const b=LN.map(l=>!x.reached[l]?'':x.pop_alone[l]?`<span class="bd few">人気だけで ${Math.round(l*100)}% に届く（ルールの手柄ではない）</span>`:`<span class="bd over">${Math.round(l*100)}% に届いた${x.reached_unit[l]?'':'・寄せ先の値'}</span>`).join('');
 return b||'<span class="bd few">2022〜24 で 50% に届かない</span>'}
function curLayer(side){const M=D.th_meta;const st=THS[TAB];if(!st.ly)st.ly=M.layers[0];return st.ly}
function renderTh(){const side=TAB.slice(3),M=D.th_meta;const V=document.getElementById('view');
 if(!M||!D.th[side]||!Object.keys(D.th[side]).length){V.innerHTML='<p class="muted">閾値のルールの版の結果がまだ無い（src.research.course.threshold search を流すと出る）。</p>';return}
 const ly=curLayer(side),rows=D.th[side][ly]||[],m=(M[side]||{})[ly]||{};const ps=m.pooled_selected||{};const pd=ps.disc,pc=ps.conf_level;const rc=m.reached||{};
 let h=`<div class="panel wrap"><b>閾値のルール（${side==='graded'?'重賞':'通常レース'}）</b>: 粗い閾値を 1〜3 個 AND でつなぎ、全部を満たす馬だけを残す（点の足し合わせはしない）。
 探す ${M.disc[0]}〜${M.disc[1]} 年・採否 ${M.conf[0]}〜${M.conf[1]} 年。2025 年以降はまだ見ていない（凍結の後に 1 回だけ見る）。
 ${side==='graded'?'単位はレース名 × コース。1 レース名では走数が足りないので、同じコースの 3 勝クラス以上（全馬場）の走で探す（足りなければ同じ芝ダ × 距離帯の 3 勝以上）。':'単位はコース × クラスの群 × 良。探す期間の走数が '+M.min_disc+' に満たない条件は、コースだけ → 芝ダ × 距離帯 × 群に寄せる。'}
 <ul class="t"><li><b>買う時点の層</b>: 人気・オッズは探す切り口には使わず、買う時点の最後の絞り込み（${M.layers.map(esc).join("・")}）の層ごとに探す。単位の「良」・人気・オッズは当日（発走前）の値なので、前日ではなく当日のルール。主の指標は、層のほかの馬（ルールに当たらない馬）との差＝上乗せ（pt と 95% 区間）。基準線として層の全馬の値も並べる。
 探す・採否は確定の人気・オッズで層を切る。層の 2022〜24 は、発走前のオッズ 20 通りで層を切り直した「本番の再現」も並べる。</li>
 <li>根拠の検査（すべて満たしたものだけ）: ① オッズから見込む複勝率（確定の単勝オッズの帯 × 頭数の帯 × ${side==='graded'?'芝ダ・距離帯':'芝ダ・距離帯・群'}）を超え、層のほかの馬も上回る（違う仮説 ${(m.tests||0).toLocaleString()} 個で BH 補正、q ≤ ${M.q_max}）
 ② 閾値の 1 段ずらし（厳しい側の隣の段、無ければ緩い側）で、オッズから見込む値を超えた分が元の半分以上 ③ 2016〜18・19〜21 の両方で、区間の下限がオッズから見込む値を超える
 ④ 似た単位（${esc(m.outer_name||'')}）でもオッズから見込む値を超える ⑤ 最近の持続: 2022〜24 でもオッズから見込む値を超えた分が探す期間の半分以上。
 落ちた理由: ${Object.entries(m.why||{}).map(([k,v])=>esc(k)+' '+v.toLocaleString()).join('・')}。</li>
 <li><b>2022〜24 は採否（⑤）に使った期間</b>なので、その値は楽観になる。独立した確かめは凍結の後の 2025〜26 だけ。「届いた」= 探した層の 2022〜24 で ${M.min_conf} 走以上・区間の下限が線を超えた（層の全馬だけで線を超える「人気だけで届く」は、届いた数に数えない）。
 寄せ先を共有する単位は同じ判定を共有する（違う仮説の数で、50% ${(rc['0.5']||{}).hyp??0}・60% ${(rc['0.6']||{}).hyp??0}・75% ${(rc['0.75']||{}).hyp??0}。単位そのものの 2022〜24 で届いた単位は ${(rc['0.5']||{}).units_own??0}・${(rc['0.6']||{}).units_own??0}・${(rc['0.75']||{}).units_own??0}）。
 層の全馬だけで線を超えるときは「人気だけで届く」と書く。</li>
 <li>1 レースに当てはまる馬が何頭もいることがある。別のエージェントのレビューで、人気の層の無い 1 つ前の版の選んだ条件（平均 1.4 頭）を確かめると、レースで束ねたブートストラップの区間は Wilson の区間とほぼ同じだった（55.3〜59.6% と 55.3〜59.7%）。</li>
 ${m.layer_base?`<li><b>基準線（この層の全馬）</b>: 探す ${pct(m.layer_base.disc.place)}（${Math.round(m.layer_base.disc.n).toLocaleString()} 走）・採否に使った期間 ${pct(m.layer_base.conf_level.place)}（${Math.round(m.layer_base.conf_level.n).toLocaleString()} 走）、複回収 ${pct(m.layer_base.conf_level.roi_f)}。${side==='graded'?'重賞は 1 番人気でも 6 割前後なので、75% は目標になりにくい。':''}</li>`:''}
 ${m.pooled_pre5&&m.pooled_pre5.conf_level?`<li>⑤（最近の持続）と単位ごとの上位 3 個の選択の前の見込み: ①〜④ を通った ${m.pooled_pre5.pairs.toLocaleString()} 組の走の全体は、2022〜24 で ${pct(m.pooled_pre5.conf_level.place)}（オッズから見込む ${pct(m.pooled_pre5.conf_level.e)}）。選んだ後の値はこれより楽観。</li>`:'<li>⑤ の前の見込み: ①〜④ を通った組が無い（該当なし）。</li>'}
 ${pd&&pc?`<li><b>事象</b>: 選んだ条件に当てはまった走の全体（同じ走は 1 回だけ）は、探す期間 ${pct(pd.place)}（オッズから見込む ${pct(pd.e)}、${Math.round(pd.n).toLocaleString()} 走）→ 採否に使った期間 ${pct(pc.place)}（オッズから見込む ${pct(pc.e)}、${Math.round(pc.n).toLocaleString()} 走）、単回収 ${pct(pc.roi_t)}・複回収 ${pct(pc.roi_f)}。<b>根拠</b>: 多くのルールから選ぶと偶然の上振れを拾い、選んだ後は平均へ戻る。オッズから見込む値を超えた分が、閾値の中身。</li>`:''}</ul></div>
 <div class="ctl"><select id="fl">${M.layers.map(x=>`<option${x===ly?' selected':''}>${esc(x)}</option>`).join('')}</select><select id="fv"><option value="">会場（すべて）</option></select><select id="fc"><option value="">コース（すべて）</option></select>
 <select id="fg"><option value="">${side==='graded'?'格':'クラスの群'}（すべて）</option></select><select id="fs"><option value="">芝ダ（すべて）</option><option>芝</option><option>ダ</option></select>
 <select id="ff"><option value="">線（すべて）</option><option value="0.75">75% に届いた</option><option value="0.6">60% 以上に届いた</option><option value="0.5">50% 以上に届いた</option><option value="none">ルールが無い</option></select><input id="fq" placeholder="名前で絞る" size="10"></div>
 <div id="cnt" class="muted"></div><div class="scroll"><table id="tb"><thead><tr><th data-k="unit">単位</th><th>ルール（なぜ効くか・検査・3 期間）</th><th>期間</th><th>[1-2-3-外]</th><th>複勝率 [95% 区間]</th><th>オッズから見込む</th><th>単回収</th><th>複回収</th><th data-k="up">上乗せ（pt）[95% 区間]</th></tr></thead><tbody></tbody></table></div>`;
 V.innerHTML=h;const fv=document.getElementById('fv'),fc=document.getElementById('fc'),fg=document.getElementById('fg');
 document.getElementById('fl').onchange=e=>{THS[TAB].ly=e.target.value;renderTh()};
 opts(fv,uniq(rows.map(r=>r.venue)),'会場（すべて）');opts(fg,uniq(rows.map(r=>r.group)),(side==='graded'?'格':'クラスの群')+'（すべて）');
 const upd=()=>{opts(fc,uniq(rows.filter(r=>!fv.value||r.venue===fv.value).map(r=>r.course)),'コース（すべて）');drawTh()};
 fv.onchange=upd;['fc','fg','fs','ff'].forEach(i=>document.getElementById(i).onchange=drawTh);document.getElementById('fq').oninput=drawTh;
 document.querySelectorAll('#tb th[data-k]').forEach(th=>th.onclick=()=>{const st=THS[TAB],k=th.dataset.k;if(st.sk===k)st.asc=!st.asc;else{st.sk=k;st.asc=k==='unit'}drawTh()});upd()}
function drawTh(){const side=TAB.slice(3),st=THS[TAB],rows=D.th[side][st.ly]||[];const g=i=>document.getElementById(i).value;const q=g('fq'),f=g('ff');
 let rs=rows.filter(r=>(!g('fv')||r.venue===g('fv'))&&(!g('fc')||r.course===g('fc'))&&(!g('fg')||r.group===g('fg'))&&(!g('fs')||r.surf===g('fs'))
  &&(!f||(f==='none'?!r.rules.length:r.best[f]))&&(!q||r.unit.indexOf(q)>=0));
 const key=r=>st.sk==='unit'?r.unit:(r.rules.length?Math.max(...r.rules.map(x=>x.uplift?x.uplift.lo:-1)):null);   // 上乗せの区間の下限（探した層の 2022〜24）
 rs.sort((a,b)=>{const x=key(a),y=key(b);if(x==null&&y==null)return 0;if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*(st.asc?1:-1)});
 document.getElementById('cnt').textContent=`${rs.length} / ${rows.length} 単位（ルールのある単位 ${rows.filter(r=>r.rules.length).length}・75% に届いた単位 ${rows.filter(r=>r.best['0.75']).length}・60% 以上 ${rows.filter(r=>r.best['0.6']).length}・50% 以上 ${rows.filter(r=>r.best['0.5']).length}）`;
 const out=[];for(const r of rs){const head=`<b>${esc(r.unit)}</b><div class="muted">探す ${r.n_disc.toLocaleString()} 走・採否 ${r.n_conf.toLocaleString()} 走（この層）</div>`;
  if(!r.rules.length){const x=r.ref;if(!x){out.push(`<tr class="u0"><td class="wrap">${head}</td><td class="r muted">探す期間に ${D.th_meta.min_disc} 走以上のルールが無い（届かない）</td><td colspan="7"></td></tr>`);continue}
   const per=[['探す（'+esc(x.level)+'）',x.disc,null],['層の全馬（探す）',x.baseA,null],['採否の期間（'+esc(x.level)+'）',x.conf_level,null],['層の全馬（採否の期間）',x.baseB,null]];
   per.forEach(([l,v],j)=>out.push(`<tr class="${j===0?'u0':''}">${j===0?`<td rowspan="${per.length}" class="wrap">${head}</td><td rowspan="${per.length}" class="r"><span class="bd few">参考・不採用（オッズを超える中身が確かめられない）</span><div><b>${esc(x.text)}</b></div><div class="muted">オッズから見込む値を超える p が最小の条件。p = ${x.p_pop.toExponential(1)}、探した数で補正すると q = ${x.q_pop.toFixed(2)}。${x.passed1?'① は通ったが、②〜⑤ のどれかで落ちた。':'① で落ちた（偶然の幅に収まる）。'}</div></td>`:''}<td>${l}</td>${stc(v)}<td class="n">—</td></tr>`));continue}
  r.rules.forEach((x,i)=>{const per=[['探す（'+esc(x.level)+'）',x.disc,null],['採否の期間（'+esc(x.level)+'）',x.conf_level,x.uplift],['層の全馬（採否の期間）',x.baseB,null]];
   if(x.level!=='単位')per.push(['採否の期間（この単位）',x.conf,null]);
   if(x.conf_repro)per.push(['採否の期間・本番の再現（発走前のオッズ 20 通り）',x.conf_repro.rule,x.conf_repro.uplift],['層の全馬・本番の再現',x.conf_repro.base,null]);
   const P3=Object.entries(x.periods).map(([k,v])=>`${esc(k)} ${pct(v.rule.place)}（層の全馬 ${pct(v.base.place)}）`).join('／');
   per.forEach(([l,v,u],j)=>out.push(`<tr class="${i===0&&j===0?'u0':''}">${i===0&&j===0?`<td rowspan="${r.rules.reduce((a,y)=>a+3+(y.level!=='単位'?1:0)+(y.conf_repro?2:0),0)}" class="wrap">${head}</td>`:''}${j===0?`<td rowspan="${per.length}" class="r"><b>${esc(x.text)}</b><div>${lineBadges(x)}${x.trend==='弱まり続け'?'<span class="bd gap">効きが弱まり続け</span>':''}${x.y2024_reversed?'<span class="bd gap">2024 年は逆向き</span>':''}</div><div class="muted">${esc(x.hyp)}: ${esc(x.why)}</div><div class="muted">3 期間の複勝率: ${P3}・重み付き（半減期 ${x.weighted.half_life} 年）${pct(x.weighted.place)}（オッズから見込む ${pct(x.weighted.e)}）・流れ: ${esc(x.trend)}</div><div class="muted">オッズから見込む値を超える q=${x.q_pop.toFixed(3)}・1 段ずらしのオッズから見込む値を超えた分 ${x.neighbors.map(p=>pp(p)).join('／')||'—'}${x.shift==='対象なし'?'（順序の無い種類だけ）':''}・似た単位 ${pct(x.outer.place)}（オッズから見込む ${pct(x.outer.e)}）・線を言える: ${LN.filter(l=>x.pass[l]).map(l=>Math.round(l*100)+'%').join('・')||'—'}</div></td>`:''}<td>${l}</td>${stc(v)}${upc(u)}</tr>`))})}
 document.querySelector('#tb tbody').innerHTML=out.join('')}
render();
"""


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
