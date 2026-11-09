"""めぐコラム『通説シリーズ』: 通説を 1 本ずつ実データで確かめ、短い記事（JSON）と読み方ガイドを書く。

1 記事 = 1 つの通説（似た通説はまとめる）。記事は短く: 数字 3 つ・結論の文 2〜3 文・グラフ 1〜2 枚・データの注記 1 行。

判定の物差し（全記事共通）:
  『オッズから見込まれる複勝率』= 単勝オッズと頭数だけから見た 3 着内の確率（folklore_frame.expected_top3）。
  通説に当てはまる馬の実際の複勝率が、見込みよりどれだけ高いか（= 人気以上に走るか）を『オッズをそろえた差』として出し、
  当てはまらない馬の同じ差と比べる。p 値は 2 群の差の Welch の z 検定。
  傾向あり = 主の条件で p < 0.01 かつ通説の向きに 0.5pt 以上。逆向きに有意なら傾向なし（逆）。
  条件付き = 主の条件では言えないが、グラフの区分のどれかで Holm 補正後 p < 0.05（通説の向き・200 走以上）。
  当てはまる出走が 300 未満なら判断保留。

通説の定義（条件・区分・文）は folklore_specs.py。集計の土台は folklore_frame.py（先に作る）。

使い方:
    python -m src.research.columns.folklore              # 全記事を書く（frontend/content/columns/<slug>.json と guides/<slug>.json）
    python -m src.research.columns.folklore --only a,b   # 一部だけ
    python -m src.research.columns.folklore --dry-run    # 結果の一覧だけ出す
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.research.columns.folklore_frame import load_frame

REPO = Path(__file__).resolve().parents[3]
COLS = REPO / "frontend/content/columns"
GUIDES = COLS / "guides"
INDEX = COLS / "index.ts"
GUIDE_INDEX = GUIDES / "index.ts"
RESULTS = REPO / "data/local/research/folklore/results.json"
TODAY = dt.date.today().isoformat()

Mask = Callable[[pd.DataFrame], pd.Series]


@dataclass
class Axis:
    """2 枚目のグラフ: 主の条件に当てはまる馬を、この区分で分けて見る。"""
    label: str
    fn: Callable[[pd.DataFrame], pd.Series]      # 区分の文字列を返す（対象外は None）
    order: list[str]
    title: str = ""


@dataclass
class Spec:
    slug: str
    title: str                       # 「〜は走る？」の問いの部分。結論は自動で後ろに付ける
    claim: str
    categories: list[str]
    universe: Mask                   # 比べる母集団（前走のある馬など）
    groups: list[tuple[str, Mask]]   # 1 枚目のグラフの各パターン。先頭が主の条件
    direction: int = 1               # +1 = 『走る』という通説、-1 = 『割引・危ない』という通説
    axis: Axis | None = None
    question: str = ""               # 調べたこと（1 文）
    conditions: str = ""             # 対象条件（データの注記）
    definition: str = ""             # 条件の定義（1〜2 文。本文の注記）
    ref_label: str = "それ以外"
    variants: list[int] | None = None   # 『条件付き』の根拠に数えるパターン（groups の番号）。省略時は主の条件の一部になっているもの
    tips: dict[str, str] = field(default_factory=dict)   # verdict → 使い方の文（省略時は既定の文）
    extra_terms: list[dict[str, str]] = field(default_factory=list)


# ── 統計 ──────────────────────────────────────────────


def _cell(d: pd.DataFrame) -> dict[str, Any]:
    n = len(d)
    if n == 0:
        return {"n": 0, "a": None, "e": None, "r": None, "v": 0.0}
    res = d["top3"] - d["exp_top3"]
    return {"n": n, "a": float(d["top3"].mean()), "e": float(d["exp_top3"].mean()), "r": float(res.mean()),
            "v": float(res.var(ddof=1) / n) if n > 1 else 0.0, "roi": float(d["ret"].mean())}


def _p(c1: dict, c2: dict) -> float | None:
    if not c1["n"] or not c2["n"] or c1["v"] + c2["v"] <= 0:
        return None
    z = abs(c1["r"] - c2["r"]) / math.sqrt(c1["v"] + c2["v"])
    return math.erfc(z / math.sqrt(2))


def _p_vs_exp(c: dict) -> float | None:
    if not c["n"] or c["v"] <= 0:
        return None
    return math.erfc(abs(c["r"]) / math.sqrt(c["v"]) / math.sqrt(2))


def _wilson(p: float, n: int) -> list[float] | None:
    if not n or p is None:
        return None
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def holm(ps: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(ps) if p is not None]
    m = len(idx)
    out: list[float | None] = [None] * len(ps)
    run = 0.0
    for k, i in enumerate(sorted(idx, key=lambda i: ps[i])):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def analyze(spec: Spec, f: pd.DataFrame) -> dict[str, Any]:
    u = f[spec.universe(f).fillna(False).astype(bool)]
    out: dict[str, Any] = {"slug": spec.slug, "universe_n": len(u), "groups": []}
    main_mask = spec.groups[0][1](u).fillna(False).astype(bool)
    ref = _cell(u[~main_mask])
    out["ref"] = ref
    for label, fn in spec.groups:
        m = fn(u).fillna(False).astype(bool)
        c = _cell(u[m])
        c["label"] = label
        c["p_ref"] = _p(c, _cell(u[~m]))
        c["p_exp"] = _p_vs_exp(c)
        c["d"] = (c["r"] - _cell(u[~m])["r"]) if c["n"] else None
        c["subset"] = bool((m & ~main_mask).sum() == 0)
        out["groups"].append(c)
    for i, c in enumerate(out["groups"]):
        c["variant"] = i > 0 and (i in spec.variants if spec.variants is not None else c["subset"])
    out["main"] = out["groups"][0]
    if spec.axis:
        w, rest = u[main_mask], u[~main_mask]
        cats, rcats = spec.axis.fn(w), spec.axis.fn(rest)
        cells = []
        for k in spec.axis.order:
            c = _cell(w[cats == k])
            rc = _cell(rest[rcats == k])
            c["label"] = k
            c["p_exp"] = _p_vs_exp(c)
            c["p_ref"] = _p(c, rc)
            c["d"] = (c["r"] - rc["r"]) if c["n"] and rc["n"] else None
            cells.append(c)
        for c, ph in zip(cells, holm([c["p_ref"] if c["n"] >= 200 else None for c in cells])):
            c["p_holm"] = ph
        out["axis"] = cells
    out["verdict"] = verdict(spec, out)
    return out


def verdict(spec: Spec, a: dict) -> str:
    m = a["main"]
    if m["n"] < 300:
        return "inconclusive"
    d, p = m["d"], m["p_ref"]
    s = spec.direction
    if p is not None and p < 0.01 and d * s >= 0.005:
        return "supported"
    if p is not None and p < 0.01 and d * s < 0:
        return "not_supported"
    cells = [c for c in a.get("axis", []) if c.get("p_holm") is not None and c["p_holm"] < 0.05 and (c["d"] or 0) * s >= 0.005]
    gcells = [c for c in a["groups"][1:] if c.get("variant") and c["n"] >= 300 and c["p_ref"] is not None and c["p_ref"] < 0.01 / max(1, len(a["groups"]) - 1) and c["d"] * s >= 0.005]
    if cells or gcells:
        return "partial"
    return "not_supported"


# ── 文 ──────────────────────────────────────────────


def pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def pt(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:+.1f}pt"


def pfmt(p: float | None) -> str:
    if p is None:
        return "–"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


HEAD = {"supported": "人気以上に走る", "partial": "条件しだい", "not_supported": "人気どおり", "inconclusive": "数が足りない"}


def headline(spec: Spec, a: dict) -> str:
    m, v, s = a["main"], a["verdict"], spec.direction
    if v == "supported":
        return "人気以上に走る" if s > 0 else "人気ほど走らない"
    if v == "not_supported" and m["p_ref"] is not None and m["p_ref"] < 0.01:
        return "むしろ人気ほど走らない" if s > 0 else "むしろ人気以上に走る"
    if v == "partial":
        best = best_cell(spec, a)
        return f"{best['label'].replace(chr(10), '')}なら当てはまる" if best else "条件しだい"
    if v == "inconclusive":
        return "まだ数が足りない"
    return "人気どおりで上積みなし"


def best_cell(spec: Spec, a: dict) -> dict | None:
    s = spec.direction
    cand = [c for c in a.get("axis", []) if c.get("p_holm") is not None and c["p_holm"] < 0.05 and (c["d"] or 0) * s >= 0.005]
    cand += [c for c in a["groups"][1:] if c.get("variant") and c["n"] >= 300 and c["p_ref"] is not None and c["p_ref"] < 0.01 and c["d"] * s >= 0.005]
    if not cand:
        return None
    return max(cand, key=lambda c: c["d"] * s)


DEFAULT_TIPS = {
    "supported": "人気（オッズ）にまだ織り込まれていない上積みです。印を少し上げる材料になります。",
    "partial": "全体では人気どおりです。使うなら、差が出た条件に絞って見るのが無難です。",
    "not_supported": "人気（オッズ）がすでに織り込んでいます。この理由だけで印を動かす必要はありません。",
    "inconclusive": "当てはまるレースがまだ少なく、結論は出せません。",
}
DEFAULT_TIPS_NEG = {
    "supported": "人気ほど走らない傾向です。人気馬が当てはまるなら、評価を少し下げる材料になります。",
    "partial": "全体では人気どおりです。割り引くなら、差が出た条件に絞って見るのが無難です。",
    "not_supported": "人気（オッズ）がすでに織り込んでいます。この理由だけで割り引く必要はありません。",
    "inconclusive": "当てはまるレースがまだ少なく、結論は出せません。",
}


def reversed_(spec: Spec, a: dict) -> bool:
    m = a["main"]
    return a["verdict"] == "not_supported" and m["p_ref"] is not None and m["p_ref"] < 0.01 and m["d"] * spec.direction < 0


def tip(spec: Spec, a: dict) -> str:
    v = a["verdict"]
    if reversed_(spec, a):
        return ("通説とは逆に、人気になりすぎています。人気で当てはまるなら、少し割り引くくらいがちょうどよさそうです。" if spec.direction > 0
                else "通説とは逆に、人気より走っています。この理由で割り引くのは損になりやすいです。")
    return spec.tips.get(v) or (DEFAULT_TIPS if spec.direction > 0 else DEFAULT_TIPS_NEG)[v]


def summary(spec: Spec, a: dict) -> str:
    m, ref = a["main"], a["ref"]
    main = spec.groups[0][0].replace("\n", "")
    s = (f"{main}の複勝率は {pct(m['a'])}（オッズから見込まれる {pct(m['e'])}）。"
         f"{spec.ref_label}と比べたオッズをそろえた差は {pt(m['d'])}（{pfmt(m['p_ref'])}）でした。")
    b = best_cell(spec, a)
    if a["verdict"] == "partial" and b:
        s += f"{b['label'].replace(chr(10), '')}に限ると、差は {pt(b['d'])} です。"
    return s


# ── 記事 ──────────────────────────────────────────────


def chart(title: str, xlabel: str, cells: list[dict], caption: str) -> dict[str, Any]:
    act = {"name": "実際の複勝率", "values": [], "n": [], "ci": []}
    exp = {"name": "オッズから見込まれる複勝率", "values": [], "n": [], "ci": [], "baseline": True}
    ps = []
    for c in cells:
        act["values"].append(None if c["a"] is None else round(c["a"], 4)); act["n"].append(c["n"]); act["ci"].append(_wilson(c["a"], c["n"]))
        exp["values"].append(None if c["e"] is None else round(c["e"], 4)); exp["n"].append(c["n"]); exp["ci"].append(None)
        p = c.get("p_exp")
        ps.append(None if p is None else float(f"{p:.3g}"))
    return {"type": "chart", "title": title, "xLabel": xlabel, "yLabel": "複勝率", "xs": [c["label"] for c in cells],
            "series": [act, exp], "compare": {"a": act["name"], "b": exp["name"], "p": ps}, "minN": 100, "caption": caption}


def article(spec: Spec, a: dict, published: str) -> dict[str, Any]:
    m, ref = a["main"], a["ref"]
    main = spec.groups[0][0].replace("\n", "")
    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": f"{main}の複勝率", "value": pct(m["a"]), "sub": f"{m['n']:,} 走"},
            {"label": "オッズから見込まれる複勝率", "value": pct(m["e"]), "sub": "同じ人気の馬の平均"},
            {"label": "オッズをそろえた差", "value": pt(m["d"]), "sub": f"{spec.ref_label}と比べて・{pfmt(m['p_ref'])}"},
        ]},
        {"type": "text", "text": f"{summary(spec, a)}{tip(spec, a)}"},
    ]
    if len(spec.groups) > 1:
        body.append(chart(f"パターン別の複勝率（実際とオッズからの見込み）", "パターン", a["groups"],
                          "実際の線が見込みの点線より上なら、人気以上に走っています。印は見込みとの差が偶然とは考えにくい点（p<0.05）。"))
    if spec.axis and a.get("axis"):
        body.append(chart(spec.axis.title or f"{main}の馬を{spec.axis.label}で分けると", spec.axis.label, a["axis"],
                          "当てはまる馬だけを分けて、実際の複勝率と見込みを比べています。"))
    if spec.definition:
        body.append({"type": "note", "text": spec.definition})
    return {
        "slug": spec.slug,
        "title": f"{spec.title}——{headline(spec, a)}",
        "claim": spec.claim,
        "verdict": a["verdict"],
        "summary": summary(spec, a),
        "categories": spec.categories,
        "published": published,
        "updated": TODAY if published != TODAY else None,
        "dataset": {"period": "2020〜2025年", "conditions": spec.conditions or "JRA全競馬場の平地（障害除く）",
                    "sample": f"{a['universe_n']:,} 走（うち{main} {m['n']:,} 走）"},
        "body": body,
    }


def guide(spec: Spec, a: dict) -> dict[str, Any]:
    q = spec.question or f"『{spec.claim}』は本当か。"
    g: dict[str, Any] = {
        "lead": [f"調べたこと: {q}", f"答え: {headline(spec, a)}（同じ人気の馬と比べて {pt(a['main']['d'])}）。", f"使い方: {tip(spec, a)}"],
        "terms": ["複勝率", "オッズをそろえた差", "pt", "p値・q値"],
        "guides": [{"match": {"stats": True}, "title": "数字の読み方",
                    "text": "『オッズから見込まれる複勝率』は、同じくらいの人気の馬が普段どれだけ 3 着以内に来るか。実際がこれより高ければ、人気以上に走っています。"}],
    }
    if spec.extra_terms:
        g["glossary"] = spec.extra_terms
    return g


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if v is not None or k in ("values", "ci", "p")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write(spec: Spec, a: dict) -> None:
    path = COLS / f"{spec.slug}.json"
    published = TODAY
    if path.exists():
        published = json.loads(path.read_text(encoding="utf-8")).get("published", TODAY)
    path.write_text(json.dumps(_clean(article(spec, a, published)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (GUIDES / f"{spec.slug}.json").write_text(json.dumps(guide(spec, a), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    register(spec.slug)


def _ident(slug: str) -> str:
    return slug.replace("-", "_")


def register(slug: str) -> None:
    """content/columns/index.ts と guides/index.ts に記事を登録する（既にあれば何もしない）。"""
    ident = _ident(slug)
    t = INDEX.read_text(encoding="utf-8")
    if f'from "./{slug}.json"' not in t:
        lines = t.split("\n")
        last_imp = max(i for i, l in enumerate(lines) if l.startswith("import ") and l.endswith('.json";'))
        lines.insert(last_imp + 1, f'import {ident} from "./{slug}.json";')
        t = "\n".join(lines).replace("] as Column[];", f"  {ident},\n] as Column[];", 1)
        INDEX.write_text(t, encoding="utf-8")
    t = GUIDE_INDEX.read_text(encoding="utf-8")
    if f'from "./{slug}.json"' not in t:
        lines = t.split("\n")
        last_imp = max(i for i, l in enumerate(lines) if l.startswith("import ") and l.endswith('.json";'))
        lines.insert(last_imp + 1, f'import {ident} from "./{slug}.json";')
        t = "\n".join(lines).replace("\n};", f'\n  "{slug}": {ident} as ColumnGuide,\n}};', 1)
        GUIDE_INDEX.write_text(t, encoding="utf-8")


def main() -> None:
    from src.research.columns.folklore_specs import SPECS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    f = load_frame()
    results = {}
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
    for spec in SPECS:
        if only and spec.slug not in only:
            continue
        a = analyze(spec, f)
        m = a["main"]
        print(f"{spec.slug:32s} {a['verdict']:14s} n={m['n']:>7,} 実際={pct(m['a'])} 見込={pct(m['e'])} 差={pt(m['d'])} {pfmt(m['p_ref'])}  | "
              + " ".join(f"{g['label'].replace(chr(10), '')}:{pt(g['d'])}({g['n']})" for g in a["groups"][1:])
              + ("  | " + " ".join(f"{c['label'].replace(chr(10), '')}:{pt(c['d'])}({c['n']},{pfmt(c.get('p_holm'))})" for c in a.get("axis", [])) if a.get("axis") else ""))
        results[spec.slug] = _clean(a)
        if not args.dry_run:
            write(spec, a)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
