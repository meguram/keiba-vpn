#!/usr/bin/env python3
"""
docs/ と notebooks/feature_engineering/ の内容を単一 HTML に統合するビルダー。
出力: docs/html/keiba_ai_pipeline_proposal.html
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    print("pip install markdown", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
HTML_DIR = DOCS / "html"
OUT = HTML_DIR / "keiba_ai_pipeline_proposal.html"


def md_to_html(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
            "nl2br",
            "sane_lists",
        ],
    )


def extract_main(html: str) -> str:
    m = re.search(r"<main[^>]*>([\s\S]*?)</main>", html, re.I)
    if m:
        return m.group(1)
    m2 = re.search(r"<body[^>]*>([\s\S]*?)</body>", html, re.I)
    return m2.group(1).strip() if m2 else ""


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    parts: list[str] = []

    # --- Markdown ソース（docs 直下 + modeling）---
    md_files = sorted(DOCS.glob("*.md")) + sorted((DOCS / "modeling").glob("*.md"))

    for p in md_files:
        rel = p.relative_to(ROOT)
        parts.append(
            f'<section class="section md-source" id="src-{p.stem}">'
            f'<h2><span class="num">MD</span> {rel}</h2>'
            f'<p class="muted">Source file: <code class="inline">{rel}</code></p>'
            f'<div class="md-wrap">{md_to_html(read_text(p))}</div>'
            f"</section>"
        )

    # --- 既存 HTML の main 抽出 ---
    html_pages = [
        ("ml_pipeline", HTML_DIR / "ml_pipeline" / "index.html"),
        ("modeling_actions", HTML_DIR / "modeling_actions" / "index.html"),
        ("race_performance", HTML_DIR / "race_performance" / "index.html"),
        ("betting_rl_policy", HTML_DIR / "betting_rl_policy" / "index.html"),
        ("feature_engineering", HTML_DIR / "feature_engineering" / "index.html"),
        ("data_features", HTML_DIR / "data_features" / "index.html"),
    ]
    for sid, path in html_pages:
        if not path.exists():
            continue
        inner = extract_main(read_text(path))
        if not inner.strip():
            continue
        rel = path.relative_to(ROOT)
        parts.append(
            f'<section class="section html-embed" id="html-{sid}">'
            f'<h2><span class="num">HTML</span> {rel}</h2>'
            f'<p class="muted">Extracted from <code class="inline">{rel}</code> (main content only).</p>'
            f'<div class="embed-wrap">{inner}</div>'
            f"</section>"
        )

    # --- notebooks / feature_engineering ---
    nb_dir = ROOT / "notebooks" / "feature_engineering"
    nb_parts = [
        '<section class="section" id="notebooks-overview">',
        '<h2><span class="num">NB</span> notebooks / 特徴量エンジニアリング</h2>',
        '<div class="card">',
        "<h4>ディレクトリ構成</h4>",
        "<ul class=\"clean\">",
        f"<li><code class=\"inline\">notebooks/feature_engineering/llm_research.ipynb</code> — 可視化付き探索ノート</li>",
        f"<li><code class=\"inline\">run_10_fe_cycles.py</code> — 10 サイクル自動探索</li>",
        f"<li><code class=\"inline\">run_extended_fe_analysis.py</code> — 拡張特徴検証</li>",
        f"<li><code class=\"inline\">requirements.txt</code> — Jupyter 等</li>",
        f"<li><code class=\"inline\">_run_output/</code> — レポート・CSV・図</li>",
        "</ul></div>",
    ]
    for name in ("cycles_report.md", "extended_analysis.md", "requirements.txt"):
        p = nb_dir / "_run_output" / name if name.endswith(".md") else nb_dir / name
        if p.exists():
            if name.endswith(".md"):
                nb_parts.append(
                    f'<h3>{name}</h3><div class="md-wrap">{md_to_html(read_text(p))}</div>'
                )
            else:
                nb_parts.append(
                    f'<h3>{name}</h3><pre class="code">{_esc(read_text(p))}</pre>'
                )
    for csv_name in ("cycle10_race_level_spearman.csv", "extended_metrics.json"):
        cp = nb_dir / "_run_output" / csv_name
        if cp.exists():
            raw = read_text(cp)
            if len(raw) > 8000:
                raw = raw[:8000] + "\n... [truncated]\n"
            nb_parts.append(
                f'<h3>{csv_name}</h3><pre class="code">{_esc(raw)}</pre>'
            )
    # スクリプト全文（再現性）
    for script_name in ("run_10_fe_cycles.py", "run_extended_fe_analysis.py"):
        sp = nb_dir / script_name
        if sp.exists():
            body = read_text(sp)
            if len(body) > 120000:
                body = body[:120000] + "\n\n... [truncated]\n"
            nb_parts.append(
                f'<h3 id="nb-{script_name.replace(".", "-")}">{script_name}</h3>'
                f'<pre class="code">{_esc(body)}</pre>'
            )

    # ノートブック: セルソースをテキスト化
    ipynb = nb_dir / "llm_research.ipynb"
    if ipynb.exists():
        nb_parts.append('<h3 id="nb-llm-research-ipynb">llm_research.ipynb（セルソース抽出）</h3>')
        nb_parts.append(_ipynb_to_html(read_text(ipynb)))

    nb_parts.append("</section>")
    parts.insert(0, "".join(nb_parts))

    # --- TOC ---
    toc_lines = ['<a href="#hero">表紙・目的</a>', '<h4>Notebook</h4>', '<a href="#notebooks-overview">特徴量エンジニアリング</a>']
    toc_lines.append('<h4>Markdown</h4>')
    for p in md_files:
        toc_lines.append(f'<a href="#src-{p.stem}">{p.name}</a>')
    toc_lines.append('<h4>HTML 抽出</h4>')
    for sid, _ in html_pages:
        toc_lines.append(f'<a href="#html-{sid}">{sid}</a>')
    toc_lines.append('<h4>付録</h4>')
    toc_lines.append('<a href="#appendix-build">ビルドスクリプト</a>')
    toc_html = "\n".join(toc_lines)

    # 付録: 本 HTML の再生成方法
    build_self = Path(__file__).resolve()
    parts.append(
        '<section class="section" id="appendix-build">'
        '<h2><span class="num">付録</span> 統合 HTML の再生成</h2>'
        '<p class="muted">以下を実行すると <code class="inline">keiba_ai_pipeline_proposal.html</code> が上書きされます。</p>'
        f'<pre class="code">python3 {_esc(str(build_self.relative_to(ROOT)))}</pre>'
        f'<h3>build_master_proposal.py（全文）</h3><pre class="code">{_esc(read_text(build_self))}</pre>'
        "</section>"
    )

    # --- CSS: ml_pipeline から <style> 抽出 ---
    ml = read_text(HTML_DIR / "ml_pipeline" / "index.html")
    style_m = re.search(r"<style>([\s\S]*?)</style>", ml)
    extra_css = """
  .md-wrap { font-size: 14px; }
  .md-wrap h1 { font-size: 1.35rem; margin: 1.2em 0 0.5em; color: var(--accent-2); }
  .md-wrap h2 { font-size: 1.15rem; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
  .md-wrap h3 { font-size: 1.05rem; color: #c9d6ff; }
  .md-wrap table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
  .md-wrap th, .md-wrap td { border: 1px solid var(--border); padding: 6px 10px; }
  .md-wrap th { background: #1a2748; color: var(--muted); }
  .md-wrap pre { background: var(--code-bg); padding: 12px; border-radius: 8px; overflow-x: auto; }
  .md-wrap code { background: #1a2748; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
  .embed-wrap .layout { display: block; }
  .embed-wrap aside.sidebar { display: none; }
  .embed-wrap main.content { padding: 0; max-width: none; }
  .muted { color: var(--muted); font-size: 13px; }
  p.muted { margin: 4px 0 12px; }
"""
    style = (style_m.group(1) if style_m else "") + extra_css

    hero = """
  <header class="hero" id="hero">
    <div class="eyebrow">KEIBA-VPN / 企画書・完全版</div>
    <h1>競馬 AI パイプライン — 統合ドキュメント</h1>
    <p>
      本 HTML は <code class="inline">docs/</code> 配下の Markdown・既存 HTML ダッシュボード、
      および <code class="inline">notebooks/feature_engineering/</code> の探索成果を<strong>単一ファイル</strong>に統合したものです。
      オフライン閲覧・印刷・社内配布を想定しています。ソースパスは各章に記載しています。
    </p>
    <div class="hero-tags">
      <span class="tag">LayerA / LayerB</span>
      <span class="tag">FeatureStore</span>
      <span class="tag">MLflow</span>
      <span class="tag">Purged CV</span>
      <span class="tag">Race Performance</span>
      <span class="tag">Betting / RL</span>
    </div>
  </header>
"""

    doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>競馬 AI パイプライン — 統合企画書（完全版）</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
{style}
</style>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  if (typeof mermaid !== "undefined") {{
    mermaid.initialize({{ startOnLoad: true, theme: "dark" }});
  }}
}});
</script>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">
      <div class="logo">AI</div>
      <div>
        <div class="title">統合企画書</div>
        <div class="sub">Keiba AI Pipeline</div>
      </div>
    </div>
    <nav class="toc">
{toc_html}
    </nav>
  </aside>
  <main class="content">
{hero}
{chr(10).join(parts)}
    <footer>
      <span>Generated from docs/ + notebooks/feature_engineering/ — keiba-vpn</span>
      <span class="mono">keiba_ai_pipeline_proposal.html</span>
    </footer>
  </main>
</div>
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print("Wrote", OUT, "bytes", OUT.stat().st_size)


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _ipynb_to_html(json_str: str) -> str:
    try:
        j = json.loads(json_str)
    except json.JSONDecodeError:
        return f'<pre class="code">{_esc(json_str[:5000])}</pre>'
    chunks: list[str] = []
    for i, cell in enumerate(j.get("cells", [])):
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        ct = cell.get("cell_type", "?")
        if len(src) > 50000:
            src = src[:50000] + "\n\n... [cell truncated]\n"
        chunks.append(
            f'<div class="card soft"><h4>Cell {i} ({ct})</h4><pre class="code">{_esc(src)}</pre></div>'
        )
    return "\n".join(chunks)


if __name__ == "__main__":
    main()
