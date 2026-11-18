"""「いまのベスト分析」（docs/html/01_いまのベスト分析/）の各ページの中身を、元の記録から作る。

数字は手で写さない。元の記録（json・md・HTML）を読んで表にする。記録に無い値は「記録なし」と出し、推測で埋めない。
担当のセッションが結論を書く場所は、md の <!-- best:begin --> … <!-- best:end --> の印の中（あればそれをそのまま写す）。
印がまだ無い記録は、見出し（「結果」を含む節）から抜き出す。

生成は build_docs_shelves.py から呼ばれる（python -m src.scripts.docs.build_docs_shelves）。
"""
from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(os.environ.get("KEIBA_REPO_ROOT") or Path(__file__).resolve().parents[3])
DOCS = REPO / "docs"
HTML_DIR = DOCS / "html"
TRIALS = DOCS / "operations" / "retrain_trials"
NO_RECORD = "記録なし"
WIP = "作業中"

_TAG = re.compile(r"<[^<>]*>")
_BEST = re.compile(r"<!--\s*best:begin\s*-->(.*?)<!--\s*best:end\s*-->", re.S)


@dataclass
class Page:
    slug: str                      # 本の章の slug
    file: str                      # 01_いまのベスト分析/ の下のファイル名
    title: str
    blurb: str                     # 本の目次・入口の 1 行
    headline: str = ""             # 入口に出す結論の 1〜2 文（元の記録から）
    body: str = ""
    sources: list[Path] = field(default_factory=list)


# ── 小さな道具 ─────────────────────────────────────────────────────────────

def esc(s) -> str:
    return html.escape("" if s is None else str(s))


def text_of(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", fragment))).strip()


def first_sentence(s: str, limit: int = 160) -> str:
    """最初の文（括弧の中の「。」では切らない）。長ければ limit 字で切る。"""
    s = (s or "").strip()
    depth, end = 0, len(s)
    for i, ch in enumerate(s):
        if ch in "（(「『［[":
            depth += 1
        elif ch in "）)」』］]":
            depth = max(0, depth - 1)
        elif ch == "。" and depth == 0:
            end = i + 1
            break
    out = s[:end]
    return out if len(out) <= limit else out[:limit] + "…"


def headline_of(fragment: str, limit: int = 200) -> str:
    """HTML の最初の段落か箇条（見出しは飛ばす）の最初の文。"""
    for m in re.finditer(r"<(p|li)\b[^>]*>(.*?)</\1>", fragment, re.S):
        t = text_of(m.group(2))
        if len(t) >= 8:
            return first_sentence(t, limit)
    return first_sentence(text_of(fragment), limit)


def pct(x, digits: int = 1) -> str:
    return NO_RECORD if x is None else f"{x * 100:.{digits}f}%"


def num(x, digits: int = 4) -> str:
    return NO_RECORD if x is None else f"{x:.{digits}f}"


def rel(target: Path, out_dir: Path) -> str:
    return os.path.relpath(target, out_dir).replace(os.sep, "/")


def link(target: Path, out_dir: Path, label: str | None = None) -> str:
    return f'<a href="{esc(rel(target, out_dir))}">{esc(label or target.name)}</a>'


def table(head: list[str], rows: list[list], raw: bool = False) -> str:
    """表。raw=True のときセルの HTML をそのまま入れる（自分で作った HTML だけ）。"""
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c if raw else esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def md_to_html(text: str, src_dir: Path, out_dir: Path) -> str:
    """md を HTML にし、相対リンクを出力先から見たパスに直す（実在しない先はリンクを外して文字だけ残す）。"""
    import markdown  # 生成時だけ使う（requirements.txt）
    h = markdown.markdown(text, extensions=["tables", "fenced_code"])
    h = re.sub(r"<script\b.*?</script>", "", h, flags=re.S)

    def fix(m: re.Match) -> str:
        attr, url, inner = m.group(1), m.group(2), m.group(3)
        if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", url) or url.startswith("#") or url.startswith("/"):
            return m.group(0)
        path, _, frag = url.partition("#")
        target = (src_dir / path).resolve()
        if not path or not target.exists():
            return inner
        return f'<a {attr}="{esc(rel(target, out_dir))}{("#" + frag) if frag else ""}">{inner}</a>'
    return re.sub(r'<a (href)="([^"]*)"[^>]*>(.*?)</a>', fix, h, flags=re.S)


def best_block(md: Path) -> str | None:
    """md の <!-- best:begin --> … <!-- best:end --> の中身（無ければ None）。"""
    try:
        m = _BEST.search(md.read_text(encoding="utf-8"))
    except OSError:
        return None
    return m.group(1).strip() if m else None


def md_headings(md: Path) -> list[tuple[int, str, int]]:
    """(深さ, 見出し, 行番号)。コードブロックの中は除く。"""
    out, code = [], False
    for i, line in enumerate(md.read_text(encoding="utf-8").splitlines()):
        if line.startswith("```"):
            code = not code
            continue
        m = None if code else re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            out.append((len(m.group(1)), m.group(2).strip(), i))
    return out


def md_section(md: Path, line_no: int) -> str:
    """line_no の見出しから、同じ深さ以上の次の見出しの手前まで（見出しの行を含む）。"""
    lines = md.read_text(encoding="utf-8").splitlines()
    depth = len(re.match(r"^(#+)", lines[line_no]).group(1))
    end, code = len(lines), False
    for j in range(line_no + 1, len(lines)):
        if lines[j].startswith("```"):
            code = not code
        m = None if code else re.match(r"^(#+)\s", lines[j])
        if m and len(m.group(1)) <= depth:
            end = j
            break
    return "\n".join(lines[line_no:end])


_RESULT = re.compile(r"結果[（(と]|の結果$|結果$|廃止")
_NOT_RESULT = re.compile(r"先行の結果|調べた結果")


def result_headings(md: Path) -> list[tuple[int, str, int]]:
    """評価・結果の節の見出し（「先行の結果」「調べた結果」のような、評価でない見出しは除く）。"""
    return [h for h in md_headings(md) if h[0] <= 4 and _RESULT.search(h[1]) and not _NOT_RESULT.search(h[1])]


def html_tables(p: Path) -> list[tuple[str, list[list[str]]]]:
    """HTML の表を (直前の見出しの文字, 行の一覧（セルの文字）) で返す。1 行目は見出しの行。"""
    raw = p.read_text(encoding="utf-8", errors="ignore")
    out = []
    for m in re.finditer(r"<table\b.*?</table>", raw, re.S):
        hs = re.findall(r"<h[1-4]\b[^>]*>(.*?)</h[1-4]>", raw[:m.start()], re.S)
        rows = [[text_of(c) for c in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", r, re.S)]
                for r in re.findall(r"<tr\b.*?</tr>", m.group(0), re.S)]
        out.append((text_of(hs[-1]) if hs else "", [r for r in rows if r]))
    return out


def html_section_text(p: Path, heading_pat: str) -> str:
    """heading_pat に合う見出しから次の h2 の手前までの本文の文字。"""
    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"<(script|style)\b.*?</\1>", "", raw, flags=re.S)
    m = re.search(r"<h([234])\b[^>]*>((?:(?!</h\1>).)*?(?:" + heading_pat + r").*?)</h\1>", raw, re.S)
    if not m:
        return ""
    rest = raw[m.end():]
    n = re.search(r"<h[12]\b", rest)
    return text_of(rest[: n.start() if n else len(rest)])


def h(n: int, title: str, anchor: str) -> str:
    return f'<h{n} id="{anchor}">{esc(title)}</h{n}>'


def note(s: str) -> str:
    return f'<p class="src">{s}</p>'


def git_date(p: Path) -> str:
    """最後のコミット日（build_docs_shelves の git log 1 回分の表から）。"""
    from src.scripts.docs.build_docs_shelves import updated
    return updated(p)


# ── 馬の予測モデル ───────────────────────────────────────────────────────────

def _code_default_model() -> str | None:
    p = REPO / "src/api/local_predictions.py"
    try:
        m = re.search(r'environ\.get\("KEIBA_PRED_MODEL"\)\s*or\s*"([^"]+)"', p.read_text(encoding="utf-8"))
    except OSError:
        return None
    return m.group(1) if m else None


def _mv_tables() -> dict[str, list[list[str]]]:
    p = HTML_DIR / "operations" / "model_versions.html"
    out: dict[str, list[list[str]]] = {}
    if p.exists():
        for title, rows in html_tables(p):
            key = re.match(r"^(\d+(?:\.\d+)*)", title)
            out.setdefault(title, rows)
            if key:
                out.setdefault(key.group(1), rows)
    return out


def _mv_parent_lead() -> str:
    """model_versions §2.1 の表の前の段落（主指標・期間・市場の対数損失の定義）。"""
    p = HTML_DIR / "operations" / "model_versions.html"
    if not p.exists():
        return ""
    raw = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<h3\b[^>]*>\s*2\.1\b.*?</h3>(.*?)<table", raw, re.S)
    return text_of(m.group(1)) if m else ""


def build_model(out_dir: Path) -> Page:
    pg = Page("best_model", "馬の予測モデル.html", "馬の予測モデル（今の本番の版・採用時の精度・更新の年表・候補）",
              "今、本番で動いている親モデルの版と、いつ学習・採用したか、採用時の精度（test・市場との比較）、版ごとの更新の年表、"
              "改善サイクルの最良の候補と第 2 段（市場とのずれ）の結果。元は model_versions・model_cycles_log.json。")
    mv_html = HTML_DIR / "operations" / "model_versions.html"
    log_p = TRIALS / "model_cycles_log.json"
    md_p = TRIALS / "model_cycles_20261116.md"
    pg.sources = [mv_html, log_p, md_p, REPO / "src/api/local_predictions.py"]
    log = load_json(log_p) or {}
    tabs = _mv_tables()
    parts: list[str] = []

    # 1. 結論
    parts.append(h(2, "1. 結論（担当の記録）", "conclusion"))
    blk = best_block(md_p) if md_p.exists() else None
    concl = log.get("conclusion")
    if blk:
        parts.append(md_to_html(blk, md_p.parent, out_dir))
        parts.append(note(f"元: {link(md_p, out_dir)} の best の印の中（担当のセッションが書く）。"))
        pg.headline = headline_of(md_to_html(blk, md_p.parent, out_dir))
    if isinstance(concl, str) and concl and not concl.startswith("（"):
        parts.append(f"<p>{esc(concl)}</p>")
        parts.append(note(f"元: {link(log_p, out_dir)} の conclusion。"))
        pg.headline = pg.headline or first_sentence(concl, 200)
    elif isinstance(concl, list):
        parts.append("<ul>" + "".join(f"<li>{esc(c)}</li>" for c in concl) + "</ul>")
        parts.append(note(f"元: {link(log_p, out_dir)} の conclusion。"))
        pg.headline = pg.headline or first_sentence(str(concl[0]), 200)
    if not blk and not (concl and not str(concl).startswith("（")):
        parts.append(f"<p>{WIP}（担当の結論はまだ書かれていない。{esc(concl or '')}）</p>")

    # 2. 今、本番・stg で使っているモデル
    parts.append(h(2, "2. 今、本番・stg で使っているモデル", "now"))
    code_v = _code_default_model()
    cur = tabs.get("1") or next((v for k, v in tabs.items() if "採用一覧" in k), [])
    parent_row = next((r for r in cur[1:] if r and r[0].startswith("親モデル")), None)
    doc_v = parent_row[1] if parent_row and len(parent_row) > 1 else None
    rows = [["コードの既定（KEIBA_PRED_MODEL を設定しないとき）", code_v or NO_RECORD, "src/api/local_predictions.py"],
            ["model_versions の「本番」の版", doc_v or NO_RECORD, "docs/html/operations/model_versions.html §1"],
            ["突き合わせ", ("一致" if code_v and doc_v and code_v.lower() == doc_v.lower() else "食い違い・要確認"), "自動の比較"]]
    parts.append(table(["項目", "版", "元"], rows))
    hist = tabs.get("2.1", [])
    head = hist[0] if hist else []
    col_verdict = next((i for i, c in enumerate(head) if "採否" in c), None)
    col_main = next((i for i, c in enumerate(head) if "主指標" in c), None)
    prod_row = next((r for r in hist[1:] if col_verdict is not None and len(r) > col_verdict
                     and r[col_verdict].startswith("採用（本番）")), None)
    if prod_row and hist:
        parts.append("<p>本番の親モデルの版（model_versions §2.1 の「採用（本番）」の行）:</p>")
        parts.append(table(hist[0], [prod_row]))
    else:
        parts.append(f"<p>本番の親モデルの版の行: {NO_RECORD}</p>")
    if cur:
        parts.append("<details><summary>全モデルの採用一覧（model_versions §1）</summary>" + table(cur[0][:5], [r[:5] for r in cur[1:]]) + "</details>")
    lead = _mv_parent_lead()
    if lead:
        parts.append(f'<p class="src">学習の範囲（model_versions §2.1 の記載）: {esc(lead)}</p>')
    parts.append(f"<p><b>stg で使っている版</b>: {NO_RECORD}（stg の環境変数の記録は台帳に無い。上書きが無ければコードの既定 {esc(code_v or NO_RECORD)}）</p>")
    stg_notes = [n for n in log.get("notes", []) if "stg" in (n.get("title") or "")]
    parts.append("<p><b>stg の作り直しの比較</b>: " + (esc(stg_notes[-1]["title"]) + "（下の「事象と根拠」）" if stg_notes else f"{WIP}（model_cycles_log.json の notes にまだ無い）") + "</p>")
    parts.append(note(f"元: {link(mv_html, out_dir, 'model_versions.html')}（手で管理する採用の台帳）・コードの既定値。"
                      "本番（Cloud Run・VPS）の環境変数の実値は、この生成では読まない。"))

    # 3. 採用時の精度
    parts.append(h(2, "3. 採用時の精度", "accuracy"))
    mkt = re.search(r"市場\s*([0-9.]+)", lead)
    uni = re.search(r"一様\s*([0-9.]+)", lead)
    main = re.match(r"\s*([0-9.]+)", prod_row[col_main]) if prod_row and col_main is not None and len(prod_row) > col_main else None
    # 期間と指標の名前は、表の見出し（例「主指標（test 2025 勝率 LogLoss ／ 連対 ／ 複勝）」）から取る
    mh = re.search(r"（(.+?)）", head[col_main]) if col_main is not None else None
    main_label = mh.group(1).split("／")[0].strip() if mh else (head[col_main] if col_main is not None else "")
    per = re.match(r"((?:test|valid|LOYO)\s*[\d〜\-]+)", main_label)
    period_main = per.group(1) if per else NO_RECORD
    rows = []
    if main:
        v = main.group(1)
        rows.append([f"主指標: {main_label}", v, period_main, "model_versions §2.1 の表"])
        if mkt:
            rows.append(["市場（単勝オッズ）の同じ指標", mkt.group(1), period_main, "model_versions §2.1 の表の前の記載"])
            dg = min(len(v.split(".")[-1]), len(mkt.group(1).split(".")[-1]))   # 桁は少ない方にそろえる（記録に無い精度を作らない）
            rows.append(["市場との差（モデル − 市場。正ならモデルが市場より悪い）", f"{float(v) - float(mkt.group(1)):+.{dg}f}", period_main, "上の 2 行から計算"])
        if uni:
            rows.append(["一様（全馬同じ確率）の同じ指標", uni.group(1), period_main, "model_versions §2.1 の表の前の記載"])
    else:
        rows.append(["主指標", NO_RECORD, "", ""])
    auc = [(b.get("name"), b.get("auc")) for b in log.get("baselines", []) if b.get("auc") is not None]
    rows += [["AUC", num(a, 4), n, "model_cycles_log.json baselines"] for n, a in auc] or [["AUC", NO_RECORD, "", ""]]
    base = next((b for b in log.get("baselines", []) if "本番" in b.get("name", "")), None)
    period = first_sentence(log.get("baseline_note", ""), 40).split("。")[0] if log.get("baseline_note") else ""
    if base:
        period = f"{period}（改善サイクルの道具で後から測り直した値。名前: {base.get('name')}）"
        rows += [["勝ち馬の対数損失（レース単位）", num(base.get("ll"), 5), period, "model_cycles_log.json baselines"],
                 ["1 位の勝率", pct(base.get("top1_win")), period, "同上"],
                 ["1 位の 3 着内率", pct(base.get("top1_place")), period, "同上"],
                 ["◎（1 位）の単勝回収率", pct(base.get("roi_tansho")), period, "同上"],
                 ["◎（1 位）の複勝回収率", pct(base.get("roi_fukusho")), period, "同上"]]
    else:
        rows.append(["1 位の勝率・3 着内率・回収率", NO_RECORD, "", ""])
    parts.append(table(["指標", "値", "期間", "元"], rows))
    if prod_row and col_main is not None:
        parts.append(f'<p class="src">主指標の注記（model_versions §2.1 の本番の行のまま）: {esc(prod_row[col_main])}</p>')
    # 主モデルのサイクルの結果（比べた相手 ref と期間は記録の文のまま）
    mains = [c for c in log.get("cycles", []) if c.get("phase") == "main" and (c.get("result") or {}).get("metrics")]
    if mains:
        rows = []
        for c in mains:
            r = c["result"]
            for who, m in (("この回", r.get("metrics") or {}), ("比べた相手", r.get("ref_metrics") or {})):
                if m:
                    rows.append([c.get("id"), who, r.get("ref") if who == "比べた相手" else first_sentence(c.get("change") or "", 60),
                                 r.get("races", NO_RECORD), num(m.get("ll"), 5), pct(m.get("top1_win")), pct(m.get("top1_place")),
                                 pct(m.get("roi_tansho")), pct(m.get("roi_fukusho"))])
        parts.append("<p>主モデルのサイクルの結果（相手の欄の文が比べた相手と期間。r05 の相手は本番 v6e の同じ期間の値）:</p>"
                     + "<details><summary>表を開く</summary>" + table(["回", "どちら", "内容", "レース", "勝ち馬の対数損失", "1 位の勝率", "1 位の 3 着内率", "単勝回収率", "複勝回収率"], rows) + "</details>")
    for r in log.get("rules", [])[:1]:
        parts.append(f'<p class="src">改善サイクルの物差し: {esc(r)}</p>')

    # 4. 更新の年表
    parts.append(h(2, "4. 更新の年表", "timeline"))
    if hist:
        parts.append("<p>親モデルの版（model_versions §2.1。新しい版が上）:</p>" + table(hist[0], hist[1:]))
    cyc = log.get("cycles", [])
    if cyc:
        phases: list[str] = []
        for c in cyc:
            if c.get("phase") not in phases:
                phases.append(c.get("phase"))
        parts.append(f"<p>改善サイクル（記録にある {len(cyc)} 回。段（phase）ごと・古い順。段の意味は model_cycles.html）:</p>")
        for ph in phases:
            cs = [c for c in cyc if c.get("phase") == ph]
            rows = [[c.get("id"), c.get("date"), first_sentence(c.get("change") or c.get("hypothesis") or "", 90),
                     c.get("verdict") or NO_RECORD, first_sentence(c.get("reason") or "", 120)] for c in cs]
            parts.append(f"<details><summary>段 {esc(ph)}（{len(cs)} 回）</summary>" + table(["回", "日時", "変えたこと", "採否", "理由（1 文目）"], rows) + "</details>")
    mds = sorted(TRIALS.glob("*_2026*.md"))
    if mds:
        parts.append("<p>試しの記録（md）: " + "・".join(link(p, out_dir) for p in mds) + "</p>")

    # 5. 今進めている候補
    parts.append(h(2, "5. 今進めている候補", "candidate"))
    best = log.get("best") or {}
    rows = [["改善サイクルの最良", esc(best.get("name") or NO_RECORD)],
            ["最良の注記", esc(best.get("note") or NO_RECORD)],
            ["本番との差（vs 本番）", esc(log.get("vs_prod") or NO_RECORD)],
            ["状態（JSON の status の記述のまま。結論より古いことがある）", esc(log.get("status") or NO_RECORD)]]
    parts.append(table(["項目", "内容"], rows, raw=True))
    s2 = [c for c in cyc if c.get("phase") == "stage2"]
    if s2:
        rows = []
        for c in s2:
            m = (c.get("result") or {}).get("metrics") or {}
            rows.append([c.get("id"), c.get("date"), first_sentence(c.get("change") or "", 80), pct(m.get("top1_win")), pct(m.get("top1_place")),
                         pct(m.get("roi_tansho")), pct(m.get("roi_fukusho")), c.get("verdict") or NO_RECORD])
        parts.append("<p>第 2 段（市場とのずれで ◎ を選ぶ）:</p>" + table(["回", "日時", "変えたこと（◎ の選び方を含む）", "その回の ◎ の勝率", "◎ の 3 着内率", "単勝回収率", "複勝回収率", "採否"], rows))
        parts.append(note("第 2 段の ◎ は、各回の「変えたこと」にある選び方（市場とのずれで選ぶ）で、§3 の「1 位」（p の 1 位）とは別の馬。"
                          "値の期間は上の「改善サイクルの物差し」（model_cycles_log.json の rules）のとおり。"))
    else:
        parts.append(f"<p>第 2 段（市場とのずれ）: {NO_RECORD}</p>")

    # 6. 事象と根拠・限界
    parts.append(h(2, "6. 事象と根拠", "evidence"))
    for n in log.get("notes", []):
        body = n.get("body")
        items = body if isinstance(body, list) else [body]
        parts.append(f"<h3>{esc(n.get('title'))}</h3><ul>" + "".join(f"<li>{esc(b)}</li>" for b in items if b) + "</ul>")
    if not log.get("notes"):
        parts.append(f"<p>{NO_RECORD}</p>")

    parts.append(h(2, "7. 詳細へのリンク", "links"))
    parts.append("<ul>" + "".join(f"<li>{link(p, out_dir, t)}</li>" for p, t in [
        (HTML_DIR / "operations/model_cycles.html", "親モデルの改善サイクル（全サイクルの HTML）"),
        (mv_html, "モデルの版と採用状況（採用の台帳）"),
        (HTML_DIR / "operations/retrain_order.html", "モデル再学習の順番と依存関係"),
        (HTML_DIR / "modeling/chapters/training_pipeline_decisions.html", "学習パイプライン 確定事項ログ"),
        (log_p, "model_cycles_log.json（サイクルの記録）"),
    ] if p.exists()) + "</ul>")
    pg.headline = pg.headline or first_sentence(best.get("note") or "", 200)
    pg.headline = f"本番の版は {doc_v or code_v or NO_RECORD}（採用の台帳・コードの既定）。改善サイクルの{pg.headline}"
    pg.body = "\n".join(parts)
    return pg


# ── 推奨馬券 ────────────────────────────────────────────────────────────────

def _render_stage(st: dict) -> str:
    out = [f"<h3>{esc(st.get('title'))}</h3>"]
    txt = st.get("text") or []
    out.append("<ul>" + "".join(f"<li>{esc(t)}</li>" for t in (txt if isinstance(txt, list) else [txt])) + "</ul>")
    for t in st.get("tables") or []:
        out.append(f"<p><b>{esc(t.get('title'))}</b></p>" + table(t.get("head") or [], t.get("rows") or []))
    return "".join(out)


def build_bet(out_dir: Path) -> Page:
    pg = Page("best_bet", "推奨馬券の最適化.html", "推奨馬券の最適化（凍結した買い方と 2025 年 test の結果）",
              "推奨馬券の買い方の最適化の結論。凍結した買い方、訓練・評価・test 2025 の回収率と区間、基準（予測 1 位の単勝）との差、"
              "第 2 回の確かめ、サイクルの年表。元は bet_cycles_log.json と test の記録。")
    log_p = TRIALS / "bet_cycles_log.json"
    md_p = TRIALS / "bet_optimization_20261116.md"
    pg.sources = [log_p, md_p]
    log = load_json(log_p) or {}
    parts: list[str] = []
    parts.append(h(2, "1. 結論（担当の記録）", "conclusion"))
    blk = best_block(md_p) if md_p.exists() else None
    stages = log.get("stages") or {}
    if blk:
        parts.append(md_to_html(blk, md_p.parent, out_dir) + note(f"元: {link(md_p, out_dir)} の best の印の中（担当のセッションが書く）。"))
        pg.headline = headline_of(md_to_html(blk, md_p.parent, out_dir))
    c0 = stages.get("stage_00_conclusion")
    if c0:
        parts.append("<p><b>第 1 回の凍結の結論</b>（過去の記録。今の買い方の候補は、上の best の印のとおり第 2 回の凍結）</p>")
        parts.append(_render_stage(c0) + note(f"元: {link(log_p, out_dir)} の stages.stage_00_conclusion（第 1 回）。"))
        txt = c0.get("text") or []
        pg.headline = pg.headline or first_sentence(re.sub(r"^結論の見出し:\s*", "", txt[0] if txt else ""), 200)
    if not blk and not c0:
        parts.append(f"<p>{WIP}</p>")

    parts.append(h(2, "2. 2025 年 test の記録（凍結の後に 1 回だけ見たもの）", "test"))
    rows = []
    for p in sorted(TRIALS.glob("bet_test2025_*.json")):
        x = load_json(p) or {}
        m = ((x.get("periods") or {}).get("test") or {}).get("metrics") or {}
        b = ((x.get("periods") or {}).get("test") or {}).get("baselines") or {}
        b0 = next((v for k, v in b.items() if k.startswith("B0")), None) if isinstance(b, dict) else None
        b0roi = (b0.get("metrics") or b0).get("roi") if isinstance(b0, dict) else None
        rows.append([link(p, out_dir), esc(x.get("round") or 1), esc(x.get("hash")), esc(x.get("status")),
                     esc(f"{m.get('bet_races', NO_RECORD)} / {m.get('races', NO_RECORD)}"), pct(m.get("hit_rate")),
                     pct(m.get("roi")), esc(f"{pct(m.get('roi_lo'))}〜{pct(m.get('roi_hi'))}"), pct(m.get("roi_ex_top5")),
                     pct(b0roi), esc(m.get("avg_cost", NO_RECORD)), pct(m.get("bet_rate")), esc(x.get("round_note") or "")])
    parts.append(table(["記録", "回", "凍結の hash", "状態", "参戦 / 全レース", "的中率", "回収率", "95% 区間", "上位5件を除く", "B0 の回収率",
                        "1R の購入額", "参戦率", "注記"], rows, raw=True)
                 if rows else f"<p>{NO_RECORD}</p>")

    story = load_json(TRIALS / "bet_round2_story.json")
    if story and story.get("steps"):
        parts.append(h(2, "3. 第 2 回の経緯", "round2"))
        parts.append(table(["段", "結果", "次"], [[s.get("title"), s.get("result") or s.get("did") or "", s.get("next") or ""] for s in story["steps"]]))
        parts.append(note(f"元: {link(TRIALS / 'bet_round2_story.json', out_dir)}。"))

    parts.append(h(2, "4. 事象と根拠（各段の記録）", "stages"))
    for k, st in stages.items():
        if k != "stage_00_conclusion":
            parts.append("<details><summary>" + esc(st.get("title")) + "</summary>" + _render_stage(st) + "</details>")

    parts.append(h(2, "5. サイクルの年表", "timeline"))
    cyc = log.get("cycles") or []
    rows = []
    for c in cyc:
        sel = c.get("selected")
        rows.append([c.get("n"), c.get("title"), first_sentence(c.get("hypothesis") or "", 100),
                     (sel.get("family") if isinstance(sel, dict) else NO_RECORD) or NO_RECORD])
    parts.append(table(["回", "題", "仮説（1 文目）", "選んだ族"], rows) if rows else f"<p>{NO_RECORD}</p>")
    parts.append(h(2, "6. 詳細へのリンク", "links"))
    parts.append("<ul>" + "".join(f"<li>{link(p, out_dir, t)}</li>" for p, t in [
        (HTML_DIR / "operations/bet_cycles.html", "推奨馬券の最適化サイクル（全サイクルの HTML）"),
        (HTML_DIR / "operations/bet_optimization_story.html", "推奨馬券の最適化の過程"),
        (HTML_DIR / "operations/bet_r2_walkthrough.html", "第 2 回で凍結した買い方の実例"),
        (md_p, "推奨馬券の最適化の記録（md）"),
        (log_p, "bet_cycles_log.json"),
    ] if p.exists()) + "</ul>")
    pg.body = "\n".join(parts)
    return pg


# ── 重賞・通常レース（md の見出しから） ───────────────────────────────────────

def _md_theme(pg: Page, md_p: Path, out_dir: Path, meta_dir: Path, meta_files: list[Path], extra_links: list[tuple[Path, str]]) -> Page:
    parts: list[str] = []
    parts.append(h(2, "1. 結論（担当の記録）", "conclusion"))
    blk = best_block(md_p) if md_p.exists() else None
    res = result_headings(md_p) if md_p.exists() else []
    if blk:
        h_ = md_to_html(blk, md_p.parent, out_dir)
        parts.append(h_ + note(f"元: {link(md_p, out_dir)} の best の印の中（担当のセッションが書く）。"))
        pg.headline = headline_of(h_)
    elif res:
        last = res[-1]
        sec = md_section(md_p, last[2])
        parts.append(f"<p>担当の結論の印（best）はまだ無い。いちばん新しい「結果」の節（{esc(last[1])}）をそのまま写す。</p>")
        parts.append('<div class="mdsec">' + md_to_html(sec, md_p.parent, out_dir) + "</div>")
        parts.append(note(f"元: {link(md_p, out_dir)}。"))
        pg.headline = f"（担当の結論の印を待っている）いちばん新しい結果の節: {last[1]}"
    else:
        parts.append(f"<p>{NO_RECORD}</p>")

    parts.append(h(2, "2. 評価・結果の年表", "timeline"))
    if res:
        parts.append(table(["節", "見出し（日付・評価の回を含む）"], [[re.match(r"^([\d.x]+)", t).group(1) if re.match(r"^([\d.x]+)", t) else "", t] for _, t, _ in res]))
        parts.append(note(f"元: {link(md_p, out_dir)} の「結果」を含む見出し（文書の順）。"))
        # 上の節に含まれる見出し（例: 「2. 結果」の下の「2.1 …の結果」）は重ねて出さない
        with_detail, covered_until = [], -1
        for d, t, ln in res:
            if ln <= covered_until:
                continue
            with_detail.append((d, t, ln))
            covered_until = ln + len(md_section(md_p, ln).splitlines()) - 1
        parts.append("<details><summary>各回の結果の節（全文）</summary>" + "".join(
            '<div class="mdsec">' + md_to_html(md_section(md_p, ln), md_p.parent, out_dir) + "</div>" for _, _, ln in with_detail) + "</details>")
    else:
        parts.append(f"<p>{NO_RECORD}</p>")

    parts.append(h(2, "3. 詳細へのリンク（調査の HTML）", "links"))
    rows = []
    for p in meta_files:
        t = re.search(r"<title>(.*?)</title>", p.read_text(encoding="utf-8", errors="ignore")[:4000], re.S)
        rows.append([link(p, out_dir, text_of(t.group(1)) if t else p.name), esc(git_date(p)), esc(f"{p.stat().st_size // 1024:,} KB")])
    for p, t in extra_links:
        if p.exists():
            rows.append([link(p, out_dir, t), esc(git_date(p)), esc(f"{p.stat().st_size // 1024:,} KB")])
    parts.append(table(["文書", "更新日", "大きさ"], rows, raw=True))
    pg.body = "\n".join(parts)
    return pg


def build_graded(out_dir: Path, meta_files: list[Path]) -> Page:
    pg = Page("best_graded", "重賞の好走データ予想.html", "重賞の好走データ予想（ルールベース）",
              "重賞のデータ分析型の予想（同じレース名 × 同じコースの過去の好走・凡走の条件）の今の結論と、評価の回ごとの結果の年表。"
              "元は docs/ideas/graded_rule_based_handicapper.md と data/meta/graded の HTML。")
    md_p = DOCS / "ideas" / "graded_rule_based_handicapper.md"
    pg.sources = [md_p]
    _md_theme(pg, md_p, out_dir, REPO / "data/meta/graded", meta_files,
              [(HTML_DIR / "modeling/chapters/graded_handicapper.html", "重賞のルールベース予想家（本の要約の章）"),
               (md_p, "計画と経過の記録（md）"), (DOCS / "ideas/graded_handicapper_cuts.md", "切り口の目録（md）")])
    led = load_json(REPO / "data/meta/graded/cycles_forward_ledger.json")
    if isinstance(led, dict):
        races = led.get("races") or {}
        extra = table(["項目", "値"], [["前向きの記録の対象（レース数）", len(races)], ["事前登録の凍結", led.get("prereg_frozen_at") or NO_RECORD],
                                      ["最終更新", led.get("updated_at") or NO_RECORD]])
        pg.body += h(2, "4. 前向きの記録（結果が出る前に印を残したもの）", "forward") + extra + note("元: data/meta/graded/cycles_forward_ledger.json。")
    return pg


def build_course(out_dir: Path, meta_files: list[Path]) -> Page:
    pg = Page("best_course", "通常レースの好走データ抽出.html", "通常レースの好走データ抽出（全レース版）",
              "全レース（コース × クラスの群 × 良馬場）の好走データ抽出の今の結論と、版ごと（第 1〜3 版）の結果の年表。"
              "元は docs/ideas/course_class_handicapper.md と data/meta/course の HTML。")
    md_p = DOCS / "ideas" / "course_class_handicapper.md"
    pg.sources = [md_p]
    return _md_theme(pg, md_p, out_dir, REPO / "data/meta/course", meta_files,
                     [(md_p, "計画と経過の記録（md）")])


# ── 血統 ────────────────────────────────────────────────────────────────────

PEDIGREE_CHAPTERS = [
    ("modeling/chapters/pedigree_roi.html", "血統の回収率分析（tp-60）"),
    ("modeling/chapters/pedigree_side_cohit.html", "高頻度の種牡馬の特性と血統クラスタ対の同時好走（tp-73）"),
    ("modeling/chapters/pedigree_position_cross.html", "血統の位置・存在割合・クロス・類似種牡馬（tp-38）"),
    ("modeling/chapters/dam_age_firstfoal.html", "母の初仔・出産時年齢・産次と種牡馬の初年度産駒"),
    ("modeling/chapters/weight_growth_performance.html", "体重の増減と指数・成長時期の血統（tp-57）"),
]
_CONCL = r"結論|まとめ|要約|分かったこと|わかったこと|要点|採否"


def build_pedigree(out_dir: Path, meta_files: list[Path]) -> Page:
    pg = Page("best_pedigree", "血統分析のまとめ.html", "血統分析のまとめ",
              "血統の調査（回収率・同時好走・位置とクロス・母の産次・成長と血統・種牡馬の系統）の結論と、本番のモデルに入っている血統の列を 1 枚に並べ、各調査へリンクする。"
              "各節の結論は元の HTML の「結論・まとめ」の節から抜き出す。")
    mv = _mv_tables().get("1") or []
    ped_rows = [r[:5] for r in mv[1:] if r and "血統" in r[0]]
    prod = [h(2, "1. 本番のモデルに入っている血統", "production"),
            (table(mv[0][:5], ped_rows) if ped_rows else f"<p>{NO_RECORD}</p>")
            + note("元: model_versions.html §1 の採用一覧のうち、モデルの名前に「血統」を含む行。")]
    parts = prod + [h(2, "2. 調査ごとの結論", "conclusion"),
             note("各調査の HTML の「結論・まとめ・要約」の見出しの節を写す（無ければ冒頭の段落）。数字・期間は元の記載のまま。")]
    heads = []
    for i, (relp, title) in enumerate(PEDIGREE_CHAPTERS, 1):
        p = HTML_DIR / relp
        if not p.exists():
            continue
        pg.sources.append(p)
        txt = html_section_text(p, _CONCL)
        if not txt:
            from src.scripts.docs.build_docs_book import derive_blurb, extract
            txt = derive_blurb(extract(p.read_text(encoding="utf-8", errors="ignore"))[2])
        short = txt if len(txt) <= 700 else txt[:700] + "…"
        heads.append(first_sentence(txt, 120))
        parts.append(f'<h3 id="p{i}">{esc(title)}</h3><p>{esc(short)}</p>' + note(f"元: {link(p, out_dir)}（更新 {esc(git_date(p))}）。"))
    parts.append(h(2, "3. 種牡馬の系統（研究の HTML）", "sire-lines"))
    rows = []
    for p in meta_files:
        t = re.search(r"<title>(.*?)</title>", p.read_text(encoding="utf-8", errors="ignore")[:4000], re.S)
        rows.append([link(p, out_dir, text_of(t.group(1)) if t else p.name), esc(git_date(p)), esc(f"{p.stat().st_size // 1024:,} KB")])
    sl = HTML_DIR / "pedigree/sire_lines_all.html"
    if sl.exists():
        rows.append([link(sl, out_dir, "全種牡馬の系統（smartrc の区分）"), esc(git_date(sl)), esc(f"{sl.stat().st_size // 1024:,} KB")])
    parts.append(table(["文書", "更新日", "大きさ"], rows, raw=True))
    parts.append(h(2, "4. 関連の記録", "notes"))
    parts.append("<ul>" + "".join(f"<li>{link(p, out_dir, t)}</li>" for p, t in [
        (DOCS / "ideas/sire_dam_influence.md", "父と母の影響"), (DOCS / "ideas/euro_kick_sire_lines.md", "欧州型の瞬発力の系統"),
        (HTML_DIR / "pedigree/PERFORMANCE_GENES.html", "競走馬パフォーマンス遺伝子ナレッジベース（知識の資料。検証の記録ではない）")] if p.exists()) + "</ul>")
    parts.append(note("数字の更新は、各調査の HTML を作り直したときに反映される。"))
    pg.headline = heads[0] if heads else ""
    pg.body = "\n".join(parts)
    return pg
