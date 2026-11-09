"""めぐコラム『通説シリーズ』: 通説を 1 本ずつ実データで確かめ、短い記事（JSON）と読み方ガイドを書く。

1 記事 = 1 つの通説（似た通説はまとめる）。記事は短く: 数字 3 つ・結論の文 2〜3 文・グラフ 1〜2 枚・データの注記 1 行。

判定の物差し（全記事共通）:
  『オッズから見込まれる複勝率』= 単勝オッズと頭数だけから見た 3 着内の確率（folklore_frame.expected_top3）。
  通説に当てはまる馬の実際の複勝率が、見込みよりどれだけ高いか（= 人気以上に走るか）を『オッズをそろえた差』として出し、
  当てはまらない馬の同じ差と比べる。p 値は 2 群の差の Welch の z 検定。
  傾向あり = 主の条件で p < 0.01 かつ通説の向きに 0.5pt 以上。逆向きに有意なら傾向なし（逆）。
  条件付き = 主の条件では言えないが、グラフの区分のどれかで Holm 補正後 p < 0.05（通説の向き・200 走以上）。
  当てはまる出走が 300 未満なら判断保留。
  検証の章に無い通説は、さらに期間を 2020〜2022 年と 2023〜2025 年に分け、レース単位のクラスタ頑健な検定で両方に同じ向き（p<0.05）が
  出るかを確かめる（apply_halves。2026-11-09 の監査: 出走を独立とみなす Welch の検定だけだと、片方の期間だけの差でも『傾向あり』になっていた）。
  再現しない『傾向あり』は『条件付き』へ、根拠の区分が再現しない『条件付き』は『傾向なし』へ下げる。

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
import os
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
VERIFY_TESTS = Path(os.environ.get("FOLKLORE_OUT", "/home/jovyan/work/folklore_out")) / "tests.parquet"
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
    hid: str | None = None              # 通説の検証の章（folklore_verify）の同じ定義の型。あれば判定はその格に従う
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
    out["reversed"] = out["verdict"] == "not_supported" and main_mask.any() and _engine_reversed(spec, out)
    return out


def _engine_reversed(spec: Spec, a: dict) -> bool:
    m = a["main"]
    return m["p_ref"] is not None and m["p_ref"] < 0.01 and m["d"] * spec.direction < 0


def load_verify() -> dict[str, dict] | None:
    """通説の検証の章の結果（全体の行）。2016〜2025 年・クラスタ頑健・発見→確認の 2 段の BH で、コラムの集計より厳しい。"""
    if not VERIFY_TESTS.exists():
        return None
    t = pd.read_parquet(VERIFY_TESTS)
    t = t[t["dim"] == "全体"]
    return {r.hid: {"adj": float(r.adj), "tier": str(r.adj_tier), "agree": bool(r.adj_agree)} for r in t.itertuples()}


def tier_verdict(tier: str, agree: bool) -> tuple[str, bool]:
    """検証の章の格と向きから (判定, 逆向きか) を決める。S・A・B: 通説の向きなら傾向あり、逆なら傾向なし（逆）。
    C: 向きが合えば条件付き、逆なら傾向なし（逆）。格なし: 傾向なし。"""
    if tier in ("S", "A", "B"):
        return ("supported", False) if agree else ("not_supported", True)
    if tier == "C":
        return ("partial", False) if agree else ("not_supported", True)
    return "not_supported", False


def apply_verify(spec: Spec, a: dict, v: dict | None) -> None:
    """検証の章の格で判定を決め直す（より厳しい検定を正とする）。
    S・A・B: 通説の向きなら傾向あり、逆なら傾向なし（逆）。C: 向きが合えば条件付き、逆なら傾向なし（逆）。
    格なし: コラムの集計で傾向ありでも『条件付き』に下げる（2020〜2025 年だけの傾向）。それ以外はコラムの判定のまま。"""
    if not spec.hid or not v or spec.hid not in v:
        return
    e = v[spec.hid]
    a["verify"] = {"hid": spec.hid, **e}
    tier, agree = e["tier"], e["agree"]
    if tier in ("S", "A", "B", "C"):
        a["verdict"], a["reversed"] = tier_verdict(tier, agree)
    else:
        a["reversed"] = False
        if a["verdict"] == "supported":
            a["verdict"] = "partial"
            a["recent_only"] = True


HALVES = (("2020〜2022年", lambda y: y <= 2022), ("2023〜2025年", lambda y: y >= 2023))


def _half_fit(u: pd.DataFrame, g: pd.Series) -> list[dict[str, float]]:
    """期間の前半・後半それぞれで、(3着内 − 見込み) を当てはまるか（0/1）へ回帰した差と p（レース単位のクラスタ頑健）。"""
    from src.research.columns.race_context import ols_fit

    d = u.assign(_res=u["top3"] - u["exp_top3"], _g=g.astype(float).to_numpy())
    years = pd.to_datetime(d["date"]).dt.year
    out = []
    for name, fn in HALVES:
        z = d[fn(years)]
        if z["_g"].sum() < 100 or (1 - z["_g"]).sum() < 100:
            out.append({"name": name, "d": None, "p": None, "n": int(z["_g"].sum())})
            continue
        r = ols_fit("_res ~ _g", z)
        out.append({"name": name, "d": float(r.params["_g"]), "p": float(r.pvalues["_g"]), "n": int(z["_g"].sum())})
    return out


def _replicated(h: list[dict], s: int) -> bool:
    return all(x["d"] is not None and x["p"] is not None and x["p"] < 0.05 and x["d"] * s > 0 for x in h)


def _cands(spec: Spec, a: dict) -> list[dict]:
    """条件付きの根拠になりうる区分（グラフの区分で Holm 補正後 p<0.05、または本命を絞った群で p<0.01。通説の向きに 0.5pt 以上）。"""
    s = spec.direction
    cand = [c for c in a.get("axis", []) if c.get("p_holm") is not None and c["p_holm"] < 0.05 and (c["d"] or 0) * s >= 0.005]
    cand += [c for c in a["groups"][1:] if c.get("variant") and c["n"] >= 300 and c["p_ref"] is not None and c["p_ref"] < 0.01 and c["d"] * s >= 0.005]
    return cand


def apply_halves(spec: Spec, a: dict, f: pd.DataFrame) -> None:
    """検証の章に無い通説の判定を、前半・後半の再現で確かめ直す。
    傾向あり: 本命の条件が両方の期間で再現すればそのまま。しなければ『条件付き』（再現した区分があればそれを、無ければ『片方の期間だけ』と書く）。
    条件付き: 両方の期間で再現した区分があればそのまま（その区分を根拠にする）、無ければ『傾向なし』。"""
    if a.get("verify") or a["verdict"] not in ("supported", "partial"):
        return
    s = spec.direction
    u = f[spec.universe(f).fillna(False).astype(bool)]
    main = spec.groups[0][1](u).fillna(False).astype(bool)
    a["halves"] = _half_fit(u, main)
    rep_labels = []
    for c in _cands(spec, a):
        if any(c is x for x in a.get("axis", [])):
            cat = spec.axis.fn(u)
            k = cat == c["label"]
            h = _half_fit(u[k], main[k])
        else:
            fn = next(fn for lab, fn in spec.groups if lab == c["label"])
            h = _half_fit(u, fn(u).fillna(False).astype(bool))
        c["halves"] = h
        if _replicated(h, s):
            rep_labels.append(c["label"])
    a["rep_labels"] = rep_labels
    if a["verdict"] == "supported":
        if not _replicated(a["halves"], s):
            a["verdict"], a["not_replicated"] = "partial", True
        return
    if not rep_labels:
        a["verdict"] = "not_supported"


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
    if reversed_(spec, a):
        return "むしろ人気ほど走らない" if s > 0 else "むしろ人気以上に走る"
    if v == "partial" and a.get("not_replicated") and not best_cell(spec, a):
        return "片方の期間だけの差で、言い切れない"
    if v == "partial" and a.get("recent_only"):
        return "近年は人気以上だが、長い期間では言い切れない" if s > 0 else "近年は人気ほど走らないが、長い期間では言い切れない"
    if v == "partial":
        best = best_cell(spec, a)
        return f"{best['label'].replace(chr(10), '')}なら当てはまる" if best else "条件しだい"
    if v == "inconclusive":
        return "まだ数が足りない"
    return "人気どおりで上積みなし"


def best_cell(spec: Spec, a: dict) -> dict | None:
    s = spec.direction
    cand = _cands(spec, a)
    if "rep_labels" in a:                       # 前半・後半で確かめた通説は、両方で再現した区分だけを根拠にする
        cand = [c for c in cand if c["label"] in a["rep_labels"]]
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
    return bool(a.get("reversed"))


def tip(spec: Spec, a: dict) -> str:
    v = a["verdict"]
    if reversed_(spec, a):
        return ("通説とは逆に、人気になりすぎています。人気で当てはまるなら、少し割り引くくらいがちょうどよさそうです。" if spec.direction > 0
                else "通説とは逆に、人気より走っています。この理由で割り引くのは損になりやすいです。")
    if a.get("recent_only"):
        return "2020〜2025 年では差がありますが、2016 年からの検証では偶然と区別できません。印を動かすなら小さめに。"
    if a.get("not_replicated"):
        return ("使うなら、どちらの期間でも差が出た条件に絞ってください。"
                if best_cell(spec, a) else "偶然の可能性があるので、印を動かすなら小さめに。")
    return spec.tips.get(v) or (DEFAULT_TIPS if spec.direction > 0 else DEFAULT_TIPS_NEG)[v]


def summary(spec: Spec, a: dict) -> str:
    m, ref = a["main"], a["ref"]
    main = spec.groups[0][0].replace("\n", "")
    s = (f"{main}の複勝率は {pct(m['a'])}（オッズから見込まれる {pct(m['e'])}）。"
         f"{spec.ref_label}と比べたオッズをそろえた差は {pt(m['d'])}（{pfmt(m['p_ref'])}）でした。")
    b = best_cell(spec, a)
    if a.get("halves") and a.get("not_replicated"):
        s += "期間を分けると " + "・".join(f"{h['name']} {pt(h['d'])}（{pfmt(h['p'])}）" for h in a["halves"] if h["d"] is not None) + "で、片方の期間でしか差が出ません。"
    if a["verdict"] == "partial" and b:
        s += f"{b['label'].replace(chr(10), '')}に限ると、差は {pt(b['d'])} です"
        s += "（2020〜2022 年・2023〜2025 年のどちらでも同じ向き）。" if "rep_labels" in a else "。"
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
    notes = [spec.definition] if spec.definition else []
    if a.get("verify"):
        e = a["verify"]
        grade = f"格 {e['tier']}" if e["tier"] != "無" else "格なし（偶然と区別できない）"
        notes.append(f"2016〜2025 年の厳しい検定（2016〜2022 年で見つけ、2023〜2025 年で確かめる）では {grade}、人気をそろえた差 {pt(e['adj'])}。判定はこちらに合わせています。")
    if notes:
        body.append({"type": "note", "text": " ".join(notes)})
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


def published_date(slug: str) -> str:
    """記事の公開日: 既にあればその日付、無ければ今日。"""
    path = COLS / f"{slug}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("published", TODAY)
    return TODAY


def write_doc(slug: str, doc: dict[str, Any], guide_doc: dict[str, Any]) -> None:
    """記事とガイドの JSON を書き、index.ts に登録する。"""
    (COLS / f"{slug}.json").write_text(json.dumps(_clean(doc), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (GUIDES / f"{slug}.json").write_text(json.dumps(guide_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    register(slug)


def write(spec: Spec, a: dict) -> None:
    write_doc(spec.slug, article(spec, a, published_date(spec.slug)), guide(spec, a))


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
    v = load_verify()
    print("検証の章の結果:", "あり" if v else "なし（コラムの集計だけで判定）")
    results = {}
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text(encoding="utf-8"))
    for spec in SPECS:
        if only and spec.slug not in only:
            continue
        a = analyze(spec, f)
        apply_verify(spec, a, v)
        apply_halves(spec, a, f)
        m = a["main"]
        print(f"{spec.slug:32s} {a['verdict']:14s} n={m['n']:>7,} 実際={pct(m['a'])} 見込={pct(m['e'])} 差={pt(m['d'])} {pfmt(m['p_ref'])}  | "
              + " ".join(f"{g['label'].replace(chr(10), '')}:{pt(g['d'])}({g['n']})" for g in a["groups"][1:])
              + ("  | " + " ".join(f"{c['label'].replace(chr(10), '')}:{pt(c['d'])}({c['n']},{pfmt(c.get('p_holm'))})" for c in a.get("axis", [])) if a.get("axis") else ""))
        results[spec.slug] = _clean(a)
        if not args.dry_run:
            write(spec, a)
    from src.research.columns.folklore_verified import build_all
    res, errors = build_all(only, args.dry_run)
    results.update(res)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if errors:
        raise SystemExit("検証の結果に無い型があるグループ:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
