"""めぐコラムの記事を、読む人の目線で短くする（数字 3 つ・結論の文・グラフ 1〜2 枚）。

各記事の集計コード（src/research/columns/<記事>.py）は詳しい本文を書くので、集計し直したあとにこれを通して短くする。
残すもの:
  ・数字の囲み（stats）
  ・最初の文（結論）。長ければ文の切れ目で約 260 字まで
  ・最初のグラフ（chart / 騎手の距離グラフ / 系統樹 / ヒートマップ）を 2 枚まで。グラフが無い記事は最初の表 1 つ
  ・『馬券への活用』『まとめ』などの見出しの直後の文があれば、その 1 文目（使い方）
読み方ガイド（guides/<slug>.json）は、要点を 3 行（調べたこと・答え・使い方）にし、残したブロックに当たるものだけ残す。
通説シリーズ（folklore.py が書く記事）は最初から短いので対象外。

使い方:
    python -m src.research.columns.concise              # 全記事を短くする（folklore の記事は除く）
    python -m src.research.columns.concise --only first-dirt
    python -m src.research.columns.concise --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
COLS = REPO / "frontend/content/columns"
GUIDES = COLS / "guides"
VISUAL = {"chart", "jockey_distance_chart", "treemap", "heatmap"}
USE_HEADINGS = ("馬券への活用", "まとめ", "使い方", "予想への活用", "馬券での使い方")
MAX_TEXT = 260
MAX_VISUAL = 2


def _short(text: str, limit: int = MAX_TEXT) -> str:
    """文（『。』）の切れ目で limit 字までにする。最初の 1 文が長ければそれだけ残す。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    parts = re.split(r"(?<=。)", text.replace("\n\n", "\n"))
    out = ""
    for p in parts:
        if not p.strip():
            continue
        if out and len(out) + len(p) > limit:
            break
        out += p
    return out.strip()


def _use_text(body: list[dict[str, Any]]) -> str | None:
    for i, b in enumerate(body):
        if b["type"] == "heading" and any(h in b["text"] for h in USE_HEADINGS):
            for nb in body[i + 1:]:
                if nb["type"] == "text":
                    t = nb["text"].strip().lstrip("・")
                    return _short(t.split("\n")[0], 160)
                if nb["type"] == "heading":
                    break
    return None


def concise_body(body: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stats = next((b for b in body if b["type"] == "stats"), None)
    if stats:
        out.append(stats)
    first = next((b for b in body if b["type"] == "text"), None)
    if first:
        out.append({"type": "text", "text": _short(first["text"])})
    visuals = [b for b in _blocks(body) if b["type"] in VISUAL][:MAX_VISUAL]      # 折りたたみの中のグラフも拾う
    if visuals:
        out += visuals
    else:
        tbl = next((b for b in _blocks(body) if b["type"] == "table"), None)
        if tbl:
            out.append(tbl)
    use = _use_text(body)
    if use and (not first or use not in first["text"]):
        out.append({"type": "text", "text": use})
    return out


def _blocks(body):
    for b in body:
        yield b
        if b["type"] == "accordion":
            yield from _blocks(b["blocks"])


def _hit(m: dict[str, Any], b: dict[str, Any]) -> bool:
    t = b["type"]
    if "stats" in m:
        return t == "stats"
    if "treemap" in m:
        return t == "treemap"
    if "heatmap" in m:
        return t == "heatmap"
    if "heading" in m:
        return t == "heading" and b["text"] == m["heading"]
    if "text" in m:
        return t in ("text", "note") and b["text"].startswith(m["text"])
    if "chart" in m:
        return t in ("chart", "jockey_distance_chart") and str(b.get("title", "")).startswith(m["chart"])
    if "table" in m:
        return t == "table" and b["columns"] == m["table"]
    return False


def concise_guide(g: dict[str, Any], body: list[dict[str, Any]]) -> dict[str, Any]:
    g = dict(g)
    lead = [x for x in g.get("lead", []) if not x.startswith("数字の見方")][:3]
    g["lead"] = [_short(x, 150) for x in lead]
    blocks = list(_blocks(body))
    g["guides"] = [e for e in g.get("guides", []) if any(_hit(e["match"], b) for b in blocks)][:3]
    g["terms"] = g.get("terms", [])[:5]
    return g


# 見る数字: 小数（単位は任意）と、単位付きの 10 以上の整数（桁区切りのカンマ可）
_UNITS = r"%|pt|頭|人|組|走|騎乗|レース|倍|秒|kg|m"
_NUM = re.compile(rf"(?<![\d.,])[+\-]?(?:\d+\.\d+(?:{_UNITS})?|(?:\d{{1,3}}(?:,\d{{3}})+|\d{{2,}})(?:{_UNITS})(?![a-zA-Z]))")


def _norm(s: str) -> str:
    """数字と単位の間の空白と、桁区切りのカンマを除き、符号をそろえる（「3.5 倍」と「3.5倍」、「1,568」と「1568」、「−」と「-」を同じに見る）"""
    s = s.replace("−", "-").replace("＋", "+")
    s = re.sub(r"(\d)\s+(?=%|pt|頭|人|組|走|騎乗|レース|倍|秒|kg|m(?![a-zA-Z]))", r"\1", s)
    return re.sub(r"(?<=\d),(?=\d{3})", "", s)


def stale_numbers(guide: dict[str, Any], doc: dict[str, Any]) -> list[str]:
    """ガイド（答え・使い方・読み方）の数字のうち、記事（全体）に無いもの。
    記事を集計し直してガイドが古いまま残ると、同じ画面に違う数字が並ぶ（2026-11 に 16 本あった）。
    閾値・定義・例えの数字（「100% 未満」「上がり 3F = 600m」）は、ガイドの numbers_ok に並べて外す。"""
    hay = _norm(json.dumps(doc, ensure_ascii=False))
    texts = list(guide.get("lead", [])) + [e.get("text", "") + " " + e.get("example", "") for e in guide.get("guides", [])]
    ok = {_norm(x) for x in guide.get("numbers_ok", [])}
    found = {n for t in texts for n in _NUM.findall(_norm(t))} - ok
    # 符号付き（+1.5pt）は同じ符号で探す。符号なし（1.5pt 低い）は、記事の符号を問わない
    return sorted(n for n in found if not re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", hay))


def is_folklore(slug: str) -> bool:
    from src.research.columns.folklore_verified import folklore_slugs
    return slug in folklore_slugs()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}
    for path in sorted(COLS.glob("*.json")):
        slug = path.stem
        if (only and slug not in only) or is_folklore(slug):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        gp = GUIDES / f"{slug}.json"
        guide = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
        before = len(json.dumps(doc["body"], ensure_ascii=False))
        doc["body"] = concise_body(doc["body"])
        after = len(json.dumps(doc["body"], ensure_ascii=False))
        print(f"{slug:32s} {before:7,} → {after:7,} 字（JSON） ブロック {len(doc['body'])}")
        if args.dry_run:
            continue
        out = json.dumps(doc, ensure_ascii=False, indent=2)
        if len(out.encode("utf-8")) > 300_000:
            # 大きな記事（グラフの切り替え用の系列が多いもの）は空白なしで書く。WSL の共有フォルダ（9p）では
            # 1MB を超える JSON を Next.js（webpack）が途中までしか読めず、ページが 500 になることがあった（2026-11-09）
            out = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
        path.write_text(out + "\n", encoding="utf-8")
        if guide is not None:
            gp.write_text(json.dumps(concise_guide(guide, doc["body"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # ガイドの数字が記事に無ければ止める（集計し直した後、ガイドの文を直し忘れないように）
    stale = {}
    for path in sorted(COLS.glob("*.json")):
        gp = GUIDES / f"{path.stem}.json"
        if (only and path.stem not in only) or not gp.exists():
            continue
        miss = stale_numbers(json.loads(gp.read_text(encoding="utf-8")), json.loads(path.read_text(encoding="utf-8")))
        if miss:
            stale[path.stem] = miss
    if stale:
        raise SystemExit("ガイドに、記事に無い数字があります。guides/<slug>.json を記事の値に直してください:\n"
                         + "\n".join(f"  {k}: {' '.join(v)}" for k, v in stale.items()))


if __name__ == "__main__":
    main()
