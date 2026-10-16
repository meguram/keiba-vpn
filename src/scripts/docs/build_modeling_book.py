#!/usr/bin/env python3
"""docs/html/modeling/chapters/ の各 HTML（章）を、部・章・節で目次をつけた 1 冊の本（docs/html/modeling/book.html）に統合する。

元の HTML は変えない（個別ページの URL・他からのリンク・他の人の編集はそのまま）。この本は元の HTML から**毎回作り直す**ので、
元の HTML を直したら再生成する。新しい HTML を modeling/ に足したら、下の PARTS にも載せる（載せ忘れはテストが検出する）。

  python -m src.scripts.docs.build_modeling_book
  python -m src.scripts.docs.build_modeling_book --out /tmp/book.html

やっていること:
  - 各 HTML から本文だけを取り出す（旧ページの左サイドバー・前後リンク・フッターは外す）
  - 壊れた属性（”_1” のような全角の引用符）を直す
  - id と # リンクを章ごとに名前空間化する（章の間で id が衝突しない）。他の章へのリンク（xxx.html#yyy）は本の中の位置へ張り替える
  - 各 HTML の CSS を、章の中だけに効くように範囲を限定して取り込む
  - 左に目次（部 → 章 → 節）。章ごとに「何について書かれているか」「更新日」「状態」「関連する章」を付ける
"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELING_DIR = REPO_ROOT / "docs" / "html" / "modeling"          # 本(book.html)を置く場所
CHAPTERS_DIR = MODELING_DIR / "chapters"                      # 各章の HTML（子ファイル）はここで管理する
DEFAULT_OUT = MODELING_DIR / "book.html"
JST = timezone(timedelta(hours=9))


@dataclass
class Chapter:
    slug: str                      # 元のファイル名（拡張子なし）
    blurb: str                     # 何について書かれているか（1〜2 文）
    kind: str = "設計"             # 設計 / 仕様 / 結果 / ログ / メモ
    see: list[tuple[str, str]] = field(default_factory=list)   # (関連する章の slug, 関係の説明)


@dataclass
class Part:
    title: str
    lead: str
    chapters: list[Chapter]


PARTS: list[Part] = [
    Part("第I部　全体像と方針", "モデルをどう作るかの考え方と、全体の設計。", [
        Chapter("modeling_coding_strategy", "モデリングを実装するときのコーディング方針。推奨ライブラリ、ディレクトリ構成、ファイル命名、メモリ節約の原則など。", "設計"),
        Chapter("staged_probability_pipeline", "勝率・連対率・複勝率を、位置取り → ペース → 当日馬場 → 血統適性 … と多段階で推定して最終確率につなぐパイプライン（Stage0〜7）の設計。", "設計",
                [("modeling_strategy_tp04_tp07", "実装の方針はこちら"), ("feature_strategy_tp03_tp04", "前段予測量（P1〜P7）の具体化はこちら")]),
        Chapter("modeling_strategy_tp04_tp07", "tp-04〜tp-07 のモデリング方針。既存ドキュメントの整理（踏襲・更新・矛盾の解消）、検証カタログ、採否基準、各 notebook の戦略。", "設計",
                [("feature_strategy_tp03_tp04", "使う特徴量の設計"), ("training_pipeline_decisions", "確定した結果")]),
        Chapter("advanced_modeling_techniques", "レース直前確率モデルに使う実戦的な ML テクニック。カテゴリ変数のエンコーディング（target encoding など）を中心に。", "メモ"),
        Chapter("reference_insights", "参考資料から取り込んだ設計メモ。ベースラインを先に固定する、確率の信頼性を ROI より先に評価する、など。", "メモ"),
    ]),
    Part("第II部　データと分割", "どのデータがいつ集まり、どう加工され、どう分割して使うか。", [
        Chapter("data_flow_schedule", "データの収集 → 保存 → 加工 → 統合の流れを、cron の発生時刻つきで示す。1 日のタイムライン、データ別の一覧、手動の工程、処理時間の記入欄。", "仕様",
                [("feature_strategy_tp03_tp04", "T-X で実行できるか（成立性）の調査はこちら")]),
        Chapter("horse_pre_race_dataset_spec", "horse_pre_race_dataset（学習の母表）の仕様。目的、設計思想、列の定義、カテゴリ変数の扱い。", "仕様"),
        Chapter("pre_race_feature_whitelist", "レース直前に使ってよい特徴量のホワイトリスト。テーブル別に、使う列と使わない列（オッズ・人気を入れない原則）を定める。", "仕様"),
        Chapter("train_valid_test_split_strategy", "Train / Valid / Test の分割戦略。なぜランダム分割が通用しないか、時系列 CV、補助評価セット、分布の把握。", "設計",
                [("training_pipeline_decisions", "実際に確定した分割と分布差は §2")]),
    ]),
    Part("第III部　特徴量", "特徴量の設計・算出方法と、当日に作るための仕組み。", [
        Chapter("feature_strategy_tp03_tp04", "tp-02〜tp-04 の特徴量戦略書。ファミリー A〜H（127 列）、前段予測量、ablation の計画、予測フローの図、T-X で実行できるか、特徴量と生データの依存関係。", "設計",
                [("training_pipeline_decisions", "確定した結果はこちら"), ("data_flow_schedule", "各データが集まる時刻"), ("feature_serving_low_memory", "当日の特徴量の作り方")]),
        Chapter("feature_engineering_exploration", "特徴量の探索メモ。出走メンバー依存の特徴、仕上げのリズム、コース・馬場、ペース・位置取りの合成、欠損自体を特徴にする、など。", "メモ"),
        Chapter("race_performance_rating_system", "レースパフォーマンス（TimeFigure、着差補正、斤量補正など）を算出するシステムの設計と実装。", "仕様"),
        Chapter("track_bias_pedigree_pipeline", "血統 × 馬場傾向（当日・前日の勝ち馬の上りと、父・母父の適性）の特徴量パイプライン。", "仕様"),
        Chapter("horse_pedigree_10gen_merge_design", "5 世代血統表を接ぎ木して、10 世代相当の血統を作る設計（アンカー選定、一意キー、アルゴリズム）。", "設計"),
        Chapter("pedigree_stage_research_notes", "血統・馬場傾向の研究方針メモ。ゴール、現状のパイプライン、GCS 負荷を抑える運用原則、未解決の論点。", "メモ"),
        Chapter("feature_serving_low_memory", "当日の特徴量を小さなメモリで作る設計。出走馬だけで計算できる理由、サービング用バンドル、検証、見つかった不具合。", "設計",
                [("feature_strategy_tp03_tp04", "特徴量の定義はこちら")]),
    ]),
    Part("第IV部　学習パイプラインと実験結果", "tp-00〜tp-08 の確定事項、チューニング工程、実験結果。", [
        Chapter("training_pipeline_decisions", "学習パイプライン（tp-00〜tp-04）で確定した事項のログ。分割、特徴量、データ完全化の経緯と結果、ablation、前段モデル。", "ログ",
                [("feature_strategy_tp03_tp04", "設計の根拠はこちら")]),
        Chapter("hyperparameter_tuning_workflow", "ハイパーパラメータ・モデル選定の工程。リーク防止の絶対ルール、二層 CV、凍結する順、目的関数、アンチパターン。", "設計"),
        Chapter("tp05a_experiment_report", "tp-05a の実験結果。勝ち馬確率の目的関数の比較（3 fold × 3 seed）。", "結果",
                [("modeling_experiment_report", "tp-05a〜tp-08 の総合はこちら")]),
        Chapter("modeling_experiment_report", "tp-05a〜tp-08 の総合実験結果。現在の予測パイプラインの精度、段階別の結果、馬券最適化の強化学習（tp-08）、MLflow。", "結果",
                [("training_pipeline_decisions", "tp-04 までの確定事項")]),
    ]),
    Part("第V部　オッズと馬券", "オッズの予測、複合馬券の確率、バックテストと馬券推奨の設計。", [
        Chapter("final_odds_model", "最終オッズ予測モデル。発走前のデータから単勝・複勝オッズを推定する LightGBM 3 ヘッドモデルの設計・学習・推論フロー。", "設計"),
        Chapter("exotic_bet_probabilities", "馬連・ワイド・馬単・3 連複の的中確率を、勝率・連対率・複勝率から一貫して計算するモデル（割引 Harville）。", "設計"),
        Chapter("betting_backtest_dataset_spec", "betting_backtest_dataset の仕様。確率テーブル・オッズ・結果を結んだバックテスト用データの列と粒度。", "仕様"),
        Chapter("betting_policy_reinforcement_learning", "馬券推奨レイヤと強化学習の設計議論。なぜ機械学習の延長と切り分けるか、行動サイクル、バックテスト環境。", "設計",
                [("modeling_experiment_report", "強化学習（tp-08）の結果")]),
    ]),
]

CHAPTERS: dict[str, Chapter] = {c.slug: c for p in PARTS for c in p.chapters}
KINDS = {"設計": "#7aa9ff", "仕様": "#34d399", "結果": "#fbbf24", "ログ": "#c084fc", "メモ": "#9aa7c7"}


# ── 読み込みと整形 ─────────────────────────────────────────────────────────────

_TAG = re.compile(r"<[^<>]*>")


def fix_curly_quotes(s: str) -> str:
    """タグの中の全角の引用符（class=”x” など）を半角に直す（旧ページの生成時の不具合）。"""
    return _TAG.sub(lambda m: m.group(0).replace("“", '"').replace("”", '"'), s)


def extract(raw: str) -> tuple[str, str, str]:
    """(タイトル, CSS, 本文) を返す。本文は h1・サイドバー・前後リンク・フッター・script を除いたもの。"""
    raw = fix_curly_quotes(raw)
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", raw, re.S))
    body = raw
    m = re.search(r"<article\b[^>]*>(.*?)</article>", raw, re.S) or re.search(r"<main\b[^>]*>(.*?)</main>", raw, re.S) \
        or re.search(r"<body\b[^>]*>(.*?)</body>", raw, re.S)
    if m:
        body = m.group(1)
    body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S)
    body = re.sub(r"<aside\b.*?</aside>", "", body, flags=re.S)
    body = re.sub(r"<nav class=\"doc-nav\">.*?</nav>", "", body, flags=re.S)
    body = re.sub(r"<footer\b.*?</footer>", "", body, flags=re.S)
    body = re.sub(r"<header class=\"hero\">.*?</header>", "", body, flags=re.S)
    h1 = re.search(r"<h1\b[^>]*>(.*?)</h1>", body, re.S)
    title = html.unescape(_TAG.sub("", h1.group(1))).strip() if h1 else ""
    if h1:
        body = body.replace(h1.group(0), "", 1)
    if not title:
        t = re.search(r"<title>(.*?)</title>", raw, re.S)
        title = html.unescape(re.sub(r"\s+—.*$", "", t.group(1))).strip() if t else ""
    return title, css, body.strip()


def namespace(body: str, slug: str, slugs: set[str]) -> str:
    """id・# リンク・url(#...) を章ごとに名前空間化し、他の章への相対リンクを本の中の位置に張り替える。"""
    body = re.sub(r'(\sid=")([^"]+)(")', lambda m: f'{m.group(1)}{slug}--{m.group(2)}{m.group(3)}', body)
    body = re.sub(r'(\s(?:xlink:)?href="#)([^"]+)(")', lambda m: f'{m.group(1)}{slug}--{m.group(2)}{m.group(3)}', body)
    body = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{slug}--{m.group(1)})", body)
    # 章のファイルは chapters/ にあり、本は 1 つ上の modeling/ にあるので、../ で始まる相対リンクを 1 階層浅くする
    body = re.sub(r'((?:href|src)=")\.\./(\.\./)', r"\1\2", body)

    def link(m: re.Match) -> str:
        pre, target, frag = m.group(1), m.group(2), m.group(3)
        stem = target[:-5]
        if stem not in slugs:
            return m.group(0)
        return f'{pre}"#{stem}--{frag[1:]}"' if frag else f'{pre}"#ch-{stem}"'

    return re.sub(r'(\shref=)"(?:\./)?([A-Za-z0-9_\-]+\.html)(#[^"]*)?"', link, body)


def sections(body: str, slug: str) -> list[tuple[str, str]]:
    """本文中の h2 から (id, 見出し) を返す。id が無い h2 には付ける（本文も書き換えて返せるよう別関数で付与）。"""
    out = []
    for m in re.finditer(r"<h2\b([^>]*)>(.*?)</h2>", body, re.S):
        idm = re.search(r'id="([^"]+)"', m.group(1))
        if idm:
            out.append((idm.group(1), html.unescape(_TAG.sub("", m.group(2))).strip()))
    return out


def ensure_h2_ids(body: str, slug: str) -> str:
    n = [0]

    def f(m: re.Match) -> str:
        n[0] += 1
        if re.search(r'\sid="', m.group(1)):
            return m.group(0)
        return f'<h2 id="{slug}--s{n[0]}"{m.group(1)}>{m.group(2)}</h2>'

    return re.sub(r"<h2\b([^>]*)>(.*?)</h2>", f, body, flags=re.S)


# ── CSS を章の中だけに効かせる ────────────────────────────────────────────────────

def _split_blocks(css: str) -> list[tuple[str, str]]:
    """トップレベルの (セレクタ or @規則の頭, 中身) を返す。"""
    out, i, n = [], 0, len(css)
    while i < n:
        j = css.find("{", i)
        if j < 0:
            break
        head = css[i:j].strip()
        depth, k = 1, j + 1
        while k < n and depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0)
            k += 1
        out.append((head, css[j + 1:k - 1]))
        i = k
    return out


def scope_css(css: str, scope: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = []
    for head, inner in _split_blocks(css):
        if head.startswith("@media") or head.startswith("@supports"):
            out.append(f"{head}{{{scope_css(inner, scope)}}}")
        elif head.startswith("@"):
            out.append(f"{head}{{{inner}}}")
        else:
            sels = []
            for s in (x.strip() for x in head.split(",")):
                if s in (":root", "html", "body", "html,body", "*"):
                    sels.append(scope if s != "*" else f"{scope}, {scope} *")
                elif s.startswith(("body ", "html ")):
                    sels.append(scope + " " + s.split(" ", 1)[1])
                else:
                    sels.append(f"{scope} {s}")
            out.append(f"{', '.join(sels)}{{{inner}}}")
    return "\n".join(out)


# ── 更新日 ───────────────────────────────────────────────────────────────────

def last_updated(path: Path) -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d", "--", str(path)], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
        if out:
            return out
    except OSError:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, JST).strftime("%Y-%m-%d")


# ── 組み立て ─────────────────────────────────────────────────────────────────

SHELL_CSS = """
:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --code-bg:#0b1428; }
* { box-sizing: border-box; } html { scroll-behavior: smooth; scroll-padding-top: 18px; }
body.bk { margin:0; background:#0b1226; color:var(--text); font:15px/1.8 "Noto Sans JP",system-ui,sans-serif; }
.bk-layout { display:grid; grid-template-columns: 330px 1fr; min-height:100vh; }
.bk-side { position:sticky; top:0; height:100vh; overflow-y:auto; background:#0a1124; border-right:1px solid var(--border); padding:18px 14px 40px; font-size:13px; }
.bk-side h1 { font-size:16px; margin:0 0 4px; } .bk-side .sub { color:var(--muted); font-size:12px; margin-bottom:10px; }
.bk-side input { width:100%; padding:7px 10px; border-radius:6px; border:1px solid var(--border); background:#0f1a36; color:var(--text); margin-bottom:10px; }
.bk-side .part { margin:14px 0 4px; color:var(--accent-2); font-weight:700; font-size:12.5px; letter-spacing:.02em; }
.bk-side a { color:var(--text); text-decoration:none; display:block; padding:2px 6px; border-radius:5px; line-height:1.45; }
.bk-side a:hover { background:#16254a; } .bk-side a.cur { background:#1c2f58; color:#fff; }
.bk-side .chap > a { font-weight:600; padding:4px 6px; } .bk-side .chap .num { color:var(--muted); margin-right:5px; font-weight:400; }
.bk-side .secs { display:none; margin:0 0 4px 14px; border-left:1px solid var(--border); } .bk-side .chap.open .secs { display:block; }
.bk-side .secs a { color:var(--muted); font-size:12px; padding:1px 6px; }
.bk-main { min-width:0; padding:0 0 120px; }
.bk-front, .bk-chapter { max-width:1040px; margin:0 auto; padding:30px 40px 10px; }
.bk-front h1 { font-size:30px; margin:20px 0 6px; } .bk-front h2 { font-size:19px; margin:30px 0 8px; border-top:1px solid var(--border); padding-top:14px; }
.bk-front table, .bk-chapter table.bk-map { border-collapse:collapse; width:100%; font-size:13.5px; margin:10px 0; }
.bk-front th, .bk-front td { border:1px solid var(--border); padding:6px 10px; text-align:left; vertical-align:top; } .bk-front th { background:var(--panel); color:var(--accent-2); }
.bk-front a, .bk-head a, .bk-pn a { color:var(--accent-2); }
.bk-front td:nth-child(1), .bk-front td:nth-child(3), .bk-front td:nth-child(5) { white-space:nowrap; }
.bk-head { border-top:3px solid var(--accent); margin-top:36px; padding-top:14px; } .bk-head .no { color:var(--muted); font-size:13px; letter-spacing:.06em; }
.bk-head h2.bk-title { font-size:26px; margin:2px 0 8px; border:0; padding:0; } .bk-head .meta { color:var(--muted); font-size:12.5px; margin:0 0 6px; }
.bk-head .blurb { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 14px; margin:8px 0 6px; font-size:14px; }
.bk-tag { display:inline-block; font-size:11.5px; border:1px solid currentColor; border-radius:4px; padding:0 6px; margin-right:6px; }
.bk-see { font-size:13px; color:var(--muted); margin:4px 0 0; } .bk-see a { color:var(--accent-2); }
.bk-pn { display:flex; justify-content:space-between; gap:12px; margin:28px 0 0; padding-top:12px; border-top:1px solid var(--border); font-size:13.5px; }
.bk-pn a { color:var(--accent-2); text-decoration:none; max-width:48%; }
.bk-body { padding-top:6px; } .bk-body img, .bk-body svg { max-width:100%; height:auto; }
.bk-chapter { background:transparent !important; max-width:1040px !important; margin:0 auto !important; padding:30px 40px 10px !important; }
.bk-body table { max-width:100%; } .bk-body pre { max-width:100%; overflow-x:auto; }
@media (max-width: 900px) { .bk-layout { grid-template-columns: 1fr; } .bk-side { position:relative; height:auto; } }
"""

SHELL_JS = """
(function(){
  const side = document.querySelector('.bk-side');
  const chaps = [...side.querySelectorAll('.chap')];
  const input = document.getElementById('bk-q');
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    chaps.forEach(c => {
      const hit = !q || c.textContent.toLowerCase().includes(q);
      c.style.display = hit ? '' : 'none';
      if (q && hit) c.classList.add('open');
    });
    side.querySelectorAll('.part').forEach(p => {
      let n = p.nextElementSibling, any = false;
      while (n && !n.classList.contains('part')) { if (n.style.display !== 'none') any = true; n = n.nextElementSibling; }
      p.style.display = any ? '' : 'none';
    });
  });
  const targets = [...document.querySelectorAll('.bk-chapter, .bk-body h2')];
  const link = id => side.querySelector('a[href="#' + CSS.escape(id) + '"]');
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const t = e.target; const ch = t.closest('.bk-chapter');
      side.querySelectorAll('a.cur').forEach(a => a.classList.remove('cur'));
      chaps.forEach(c => { if (!input.value) c.classList.toggle('open', ch && c.dataset.slug === ch.dataset.slug); });
      const a = link(t.id) || (ch && link('ch-' + ch.dataset.slug));
      if (a) { a.classList.add('cur'); a.scrollIntoView({block:'nearest'}); }
    });
  }, {rootMargin: '0px 0px -75% 0px'});
  targets.forEach(t => t.id && io.observe(t));
  if (window.mermaid) { mermaid.initialize({startOnLoad: false, theme: 'dark'}); mermaid.run({querySelector: '.mermaid'}).catch(()=>{}); }
})();
"""


def build(modeling_dir: Path = CHAPTERS_DIR) -> tuple[str, dict]:
    slugs = set(CHAPTERS)
    built: dict[str, dict] = {}
    css_groups: dict[str, str] = {}
    for slug in CHAPTERS:
        path = modeling_dir / f"{slug}.html"
        title, css, body = extract(path.read_text(encoding="utf-8"))
        body = ensure_h2_ids(body, slug)
        body = namespace(body, slug, slugs)
        gid = hashlib.md5(re.sub(r"\s+", "", css).encode()).hexdigest()[:6]
        css_groups.setdefault(gid, css)
        built[slug] = {"title": title, "body": body, "gid": gid, "secs": sections(body, slug), "updated": last_updated(path)}

    # 目次
    side, no = [], 0
    for part in PARTS:
        side.append(f'<div class="part">{html.escape(part.title)}</div>')
        for ch in part.chapters:
            no += 1
            b = built[ch.slug]
            secs = "".join(f'<a href="#{html.escape(i)}">{html.escape(t[:48])}</a>' for i, t in b["secs"])
            side.append(f'<div class="chap" data-slug="{ch.slug}"><a href="#ch-{ch.slug}"><span class="num">{no}</span>{html.escape(b["title"])}</a><div class="secs">{secs}</div></div>')

    # 前付き: 全体の地図
    rows, no = [], 0
    for part in PARTS:
        for ch in part.chapters:
            no += 1
            b = built[ch.slug]
            rows.append(f'<tr><td>{html.escape(part.title.split("　")[0].replace("第", "").replace("部", ""))}</td><td><a href="#ch-{ch.slug}">第{no}章　{html.escape(b["title"])}</a></td>'
                        f'<td><span class="bk-tag" style="color:{KINDS[ch.kind]}">{ch.kind}</span></td><td>{html.escape(ch.blurb)}</td><td>{b["updated"]}</td></tr>')
    front = (
        '<section class="bk-front" id="front"><h1>モデリング総覧</h1>'
        f'<p class="meta" style="color:#9aa7c7">docs/html/modeling/chapters/ の {len(CHAPTERS)} 本の HTML を、1 冊の本にまとめたもの。生成: {datetime.now(JST):%Y-%m-%d %H:%M} JST。'
        "元の各ページは chapters/ にそのまま残っている（各章の見出しの「単独ページ」から開ける）。この本は元のページから自動で作るので、<b>直すときは元のページを直して</b> "
        "<code>python -m src.scripts.docs.build_modeling_book</code> で作り直す。</p>"
        "<h2>どれを読めばよいか</h2><ul>"
        f'<li>今の精度・最新の結論 → <a href="#ch-modeling_experiment_report">{html.escape(built["modeling_experiment_report"]["title"])}</a></li>'
        f'<li>学習で何が確定したか → <a href="#ch-training_pipeline_decisions">{html.escape(built["training_pipeline_decisions"]["title"])}</a></li>'
        f'<li>データがいつ集まり、どこで特徴量になるか → <a href="#ch-data_flow_schedule">{html.escape(built["data_flow_schedule"]["title"])}</a></li>'
        f'<li>特徴量の定義と設計の根拠 → <a href="#ch-feature_strategy_tp03_tp04">{html.escape(built["feature_strategy_tp03_tp04"]["title"])}</a></li>'
        f'<li>馬券の確率・期待値 → <a href="#ch-exotic_bet_probabilities">{html.escape(built["exotic_bet_probabilities"]["title"])}</a></li></ul>'
        "<h2>章の種類</h2><p>" + "".join(f'<span class="bk-tag" style="color:{c}">{k}</span>' for k, c in KINDS.items()) +
        "　設計 = 方針や仕組みの設計 ／ 仕様 = 列・手順などの取り決め ／ 結果 = 実験の結果 ／ ログ = 確定事項の記録 ／ メモ = 調査・探索のメモ。"
        "「更新日」は元のページの最終コミット日。2026-05 の章は初期の設計で、後の tp 系の章で更新・確定された内容がある（各章の「関連する章」を参照）。</p>"
        '<h2>全体の地図</h2><table><tr><th>部</th><th>章</th><th>種類</th><th>何について書かれているか</th><th>更新日</th></tr>' + "".join(rows) + "</table></section>"
    )

    # 章
    chapters_html, no = [], 0
    order = [c for p in PARTS for c in p.chapters]
    for idx, ch in enumerate(order):
        no += 1
        b = built[ch.slug]
        see = "".join(f'<a href="#ch-{s}">第{[c.slug for c in order].index(s) + 1}章 {html.escape(built[s]["title"])}</a>（{html.escape(why)}）　' for s, why in ch.see if s in built)
        prev_a = f'<a href="#ch-{order[idx - 1].slug}">← 第{idx}章 {html.escape(built[order[idx - 1].slug]["title"])}</a>' if idx else "<span></span>"
        next_a = f'<a href="#ch-{order[idx + 1].slug}">第{idx + 2}章 {html.escape(built[order[idx + 1].slug]["title"])} →</a>' if idx + 1 < len(order) else "<span></span>"
        chapters_html.append(
            f'<section class="bk-chapter css-{b["gid"]}" id="ch-{ch.slug}" data-slug="{ch.slug}">'
            f'<div class="bk-head"><div class="no">第{no}章　<span class="bk-tag" style="color:{KINDS[ch.kind]}">{ch.kind}</span></div>'
            f'<h2 class="bk-title">{html.escape(b["title"])}</h2>'
            f'<p class="meta">更新日 {b["updated"]} ／ 元のファイル <a href="chapters/{ch.slug}.html">chapters/{ch.slug}.html</a>（単独ページ）</p>'
            f'<div class="blurb">{html.escape(ch.blurb)}</div>' + (f'<p class="bk-see">関連する章: {see}</p>' if see else "") + "</div>"
            f'<div class="bk-body html-body">{b["body"]}</div><div class="bk-pn">{prev_a}{next_a}</div></section>'
        )

    css = SHELL_CSS + "\n" + "\n".join(f"/* {gid} */\n" + scope_css(c, f".css-{gid}") for gid, c in css_groups.items()) + "\n.bk-chapter{background:transparent !important}"
    out = (
        '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>モデリング総覧 — keiba-vpn docs</title>"
        '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'
        f"<style>{css}</style></head><body class=\"bk\"><div class=\"bk-layout\"><nav class=\"bk-side\"><h1><a href=\"#front\">モデリング総覧</a></h1>"
        '<div class="sub"><a href="../index.html">← ドキュメント一覧</a></div><input id="bk-q" type="search" placeholder="章・節をしぼり込む">'
        + "".join(side) + '</nav><main class="bk-main">' + front + "".join(chapters_html) +
        '</main></div><script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script><script>' + SHELL_JS + "</script></body></html>"
    )
    return out, built


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out, built = build()
    args.out.write_text(out, encoding="utf-8")
    print(f"{args.out}: {len(built)} 章, {len(out) // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
