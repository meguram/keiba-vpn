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
    lines, prev, code = [], "", False
    for ln in text.splitlines():
        if ln.startswith("```"):
            code = not code
        if not code:
            if re.match(r"^\s*(?:[-*+]|\d+\.)\s", ln) and prev.strip() and not re.match(r"^\s*(?:[-*+]|\d+\.)\s|^\s*\|", prev):
                lines.append("")   # 太字の行などの直後の箇条を、段落につなげずに箇条にする
            ln = re.sub(r"(?<=[\w.])\*(?=[・、。,）)\s]|$)", r"\\*", ln)   # pc_*・sr_* の * を強調の記号にしない
        lines.append(ln)
        prev = ln
    h = markdown.markdown("\n".join(lines), extensions=["tables", "fenced_code"])
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
    pg.sources = [log_p, md_p, *sorted(TRIALS.glob("bet_tiers2_*.json"))]
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
    for t2p in [TRIALS / "bet_tiers2_f96a5aeff33e.json"]:      # 推奨度の段を推奨あり／なしの 2 分類にまとめた数え直し（第 1 回の凍結）
        t2 = load_json(t2p) or {}
        if t2.get("tables"):
            parts.append(h(3, "推奨あり／なしの 2 分類（第 1 回の凍結の推奨度の段をまとめた）", "tiers2"))
            parts.append(_render_stage({**t2, "title": ""}).replace("<h3></h3>", "") + note(f"元: {link(t2p, out_dir)}（bet_cycles tiers2）。"))

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


# ── 走法軸 ────────────────────────────────────────────────────────────────

RUN_AXIS = HTML_DIR / "operations" / "running_map_course_axis.html"
RUN_RATIONALE = HTML_DIR / "operations" / "running_map_course_axis_rationale.html"
RUN_PAIRS_HTML = HTML_DIR / "operations" / "running_style_pairs.html"   # 走法が近い馬の組（bet-optimizer が出す予定）
RUN_PAIRS_MD = DOCS / "ideas" / "running_style_pairs.md"
RUN_AXES_MD = DOCS / "ideas" / "running_map_axes_fixed.md"
RUN_SCALE_MD = DOCS / "ideas" / "running_axes_scale_1to10.md"
TRIALS_README = TRIALS / "README.md"
PRECOMPUTE = REPO / "src" / "research" / "race_analysis" / "precompute.py"


def html_section_paras(p: Path, heading_pat: str, limit: int = 4) -> list[str]:
    """heading_pat に合う見出しの節の、段落と箇条の文（表は除く）。"""
    raw = re.sub(r"<(script|style)\b.*?</\1>", "", p.read_text(encoding="utf-8", errors="ignore"), flags=re.S)
    m = re.search(r"<h([234])\b[^>]*>((?:(?!</h\1>).)*?(?:" + heading_pat + r").*?)</h\1>", raw, re.S)
    if not m:
        return []
    rest = raw[m.end():]
    n = re.search(r"<h[12]\b", rest)
    sec = re.sub(r"<table\b.*?</table>", "", rest[: n.start() if n else len(rest)], flags=re.S)
    out = [text_of(x.group(2)) for x in re.finditer(r"<(p|li)\b[^>]*>(.*?)</\1>", sec, re.S)]
    return [x for x in out if len(x) >= 10 and not re.match(r"^(再現|再計算)\s*[:：]", x)][:limit]   # 手順の段落は除く


def _clip(s: str, n: int = 700) -> str:
    return s if len(s) <= n else s[:n] + "…"


def html_box_items(p: Path, label: str = "結論") -> list[str]:
    """<div class="box"><b>結論</b><ol><li>…</li></ol></div> の箇条（ページ冒頭の結論の枠）。"""
    raw = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'<div class="box[^"]*">\s*<(b|strong)>' + re.escape(label) + r"</\1>(.*?)</(?:ol|ul)>", raw, re.S)
    return [text_of(x) for x in re.findall(r"<li\b[^>]*>(.*?)</li>", m.group(2), re.S)] if m else []


def _running_axes_from_code() -> tuple[list[str], list[str], list[str]]:
    """running_map.py の AXES・MAP_KEYS・WET_AXIS から (地図の軸, 道悪の地図で替える軸, 表示だけの軸) の表示名。"""
    try:
        src = RUN_MAP_PY.read_text(encoding="utf-8")
    except OSError:
        return [], [], []
    names = dict(re.findall(r'\("(\w+)",\s*"ax_\w+",\s*-?1,\s*"([^"]+)"', src))
    mk = re.search(r"^MAP_KEYS\s*=\s*\[(.*?)\]", src, re.M)
    keys = re.findall(r'"(\w+)"', mk.group(1)) if mk else []
    wa = re.search(r"^WET_AXIS\s*=\s*\{(.*?)\}", src, re.M)
    wet = re.findall(r':\s*"(\w+)"', wa.group(1)) if wa else []
    if not names or not keys or not wet or any(k not in names for k in keys + wet):
        return [], [], []   # 書き方が変わって読めない。推測で振り分けない
    rest = [k for k in names if k not in keys and k not in wet]
    return [names[k] for k in keys], sorted({names[k] for k in wet}), [names[k] for k in rest]


def _version_comment(path: Path, var: str) -> tuple[str | None, list[tuple[str, str, str]]]:
    """VAR = "N"  # N: …（日付）。M: … の形のコメントを (版, 日付, 内容) に分ける（末尾の日付と決まり文句は落とす）。"""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return None, []
    m = re.search(r"^" + var + r'\s*=\s*"(\d+)"\s*#\s*(.*)$', src, re.M)
    if not m:
        return None, []
    rows = []
    bits = re.split(r"(?:^|(?<=。))\s*(\d+)[:：]\s*", m.group(2))   # 「…（日付）。38: …」の形で版ごとに区切る
    for num_, text in zip(bits[1::2], bits[2::2]):
        text = re.sub(r"分析ロジック・列定義を変えたら上げる.*$", "", text.strip()).strip()
        d = re.findall(r"20\d\d-\d\d-\d\d", text)
        text = re.sub(r"[（(]20\d\d-\d\d-\d\d[）)]。?$", "", text).strip()
        rows.append((num_, d[-1] if d else NO_RECORD, text))
    return m.group(1), rows


def _style_pair_boost() -> str | None:
    try:
        m = re.search(r"^STYLE_PAIR_BOOST\s*=\s*(\{.*?\})", BETTING_PY.read_text(encoding="utf-8"), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def _ul(items: list[str], n: int = 400) -> str:
    return "<ul>" + "".join(f"<li>{esc(_clip(x, n))}</li>" for x in items) + "</ul>" if items else f"<p>{NO_RECORD}</p>"


RUN_MAP_PY = REPO / "src" / "research" / "race" / "running_map.py"
BEST5_PY = REPO / "src" / "research" / "race" / "running_map_best5.py"
BETTING_PY = REPO / "src" / "pipeline" / "inference" / "betting.py"


def build_running(out_dir: Path) -> Page:
    pg = Page("best_running", "走法軸のまとめ.html", "走法軸のまとめ（走法マップ・コース適性・予測への効き）",
              "走法軸（ペース・位置取り・脚の使い方の型）の今の使い方（走法マップ・Best5・買い目の近い組）、コース適性の根拠と限界、"
              "親モデルへの効き（簡易版の履歴の検定）、版の年表。元は走法マップの調査の HTML・md・コード。")
    pg.sources = [p for p in (RUN_AXIS, RUN_RATIONALE, RUN_AXES_MD, RUN_SCALE_MD, RUN_PAIRS_MD, RUN_PAIRS_HTML, TRIALS_README,
                              RUN_MAP_PY, BEST5_PY, BETTING_PY, PRECOMPUTE) if p.exists()]
    parts: list[str] = []

    # 1. 結論（何か・今の使い方）
    parts.append(h(2, "1. 結論（走法軸とは・今の使い方）", "conclusion"))
    map_axes, wet_axes, show_axes = _running_axes_from_code()
    parts.append(table(["項目", "内容"], [
        ["地図・グループ・近い馬に使う軸", "・".join(map_axes) or NO_RECORD],
        ["道悪の地図の軸（WET_AXIS）", "・".join(wet_axes) or NO_RECORD],
        ["値の表示・色分けだけの軸", "・".join(show_axes) or NO_RECORD],
    ]) + note("元: src/research/race/running_map.py の AXES・MAP_KEYS・WET_AXIS（コードを正とする）。軸の定義は "
              f"{link(RUN_AXES_MD, out_dir)}、10 点満点の目盛りは {link(RUN_SCALE_MD, out_dir)}。"))
    mv = _mv_tables()
    cur = mv.get("1") or []
    own = [r[:4] for r in cur[1:] if r and r[0].startswith("走法")]
    rel_ = [r[:4] for r in cur[1:] if r and any(k in r[0] for k in ("追走", "ラップ", "展開"))]
    parts.append("<p><b>走法マップ（採用の台帳の行）</b></p>" + (table(cur[0][:4], own) if own else f"<p>{NO_RECORD}</p>"))
    t29 = mv.get("2.9") or []
    sec29 = [r for r in t29[1:] if r and r[0].startswith("走法")]
    if sec29:
        parts.append(table(t29[0], sec29) + note("元: model_versions.html §2.9 の走法マップの行。"))
    try:
        doc5 = re.search(r'^"""(.*?)"""', BEST5_PY.read_text(encoding="utf-8"), re.S)
    except OSError:
        doc5 = None
    if doc5:
        lines = [x.strip(" -") for x in doc5.group(1).strip().splitlines() if x.strip()]
        parts.append("<p><b>Best5（地図の位置から適性が合う順に 5 頭）</b></p>" + _ul(lines, 300)
                     + note("元: src/research/race/running_map_best5.py の説明。"))
    refl = html_section_paras(RUN_AXIS, r"反映", 6) if RUN_AXIS.exists() else []
    if refl:
        parts.append("<p><b>コース適性の今の作り（反映の節）</b></p>" + _ul(refl) + note(f"元: {link(RUN_AXIS, out_dir)} の「反映」の節。"))

    boost = _style_pair_boost()
    parts.append("<p><b>走法が近い馬の組（買い目推奨）</b>: "
                 + (f"買い目推奨の確率の倍率 STYLE_PAIR_BOOST = <code>{esc(boost)}</code>（src/pipeline/inference/betting.py）。" if boost else f"{NO_RECORD}。")
                 + ("調査の HTML: " + link(RUN_PAIRS_HTML, out_dir) if RUN_PAIRS_HTML.exists() else f"新しい調査の HTML は{WIP}。") + "</p>")
    if RUN_PAIRS_HTML.exists():
        items = html_box_items(RUN_PAIRS_HTML) or html_section_paras(RUN_PAIRS_HTML, _CONCL, 3)
        parts.append((_ul(items[:3], 300) if items else f"<p>{NO_RECORD}（結論の枠・節が見つからない）</p>")
                     + note(f"元: {link(RUN_PAIRS_HTML, out_dir)} の冒頭の結論の枠（無ければ結論の節）の初めの 3 項目。"))
    if RUN_PAIRS_MD.exists():
        hs = [x for x in md_headings(RUN_PAIRS_MD) if x[0] == 2 and re.search(r"結果|反映|残り", x[1])]
        parts.append("<details><summary>前の調査（md）の「結果」「反映」「残り」の節</summary>" + "".join(
            '<div class="mdsec">' + md_to_html(md_section(RUN_PAIRS_MD, ln), RUN_PAIRS_MD.parent, out_dir) + "</div>" for _, _, ln in hs)
            + "</details>" + note(f"元: {link(RUN_PAIRS_MD, out_dir)}（更新 {esc(git_date(RUN_PAIRS_MD))}）。"))

    # 親モデルへの効き
    res = top = None
    if TRIALS_README.exists():
        hs = md_headings(TRIALS_README)
        tops = [x for x in hs if x[1].startswith("走法軸の履歴")]
        top = tops[-1] if tops else None   # やり直した節が足されたら新しいほう
        if top:
            for d, t, ln in hs:
                if ln <= top[2]:
                    continue
                if d <= top[0]:
                    break   # 節の外
                if "結果" in t:
                    res = (d, t, ln)
                    break
    lim = ""
    if res:
        m = re.search(r"^- 限界:\s*(.+)$", md_section(TRIALS_README, res[2]), re.M)
        lim = m.group(1).strip() if m else ""
        parts.append(f"<p><b>親モデルへの効き（{esc(top[1])}）</b>: {esc(res[1])}。（このページの注: 節の本文のとおり、検定したのは簡易版の履歴）</p>"
                     + (f"<p class=\"src\">記録の限界: {esc(lim)}</p>" if lim else "") + note("数字は §2。"))
    bmap = {"umaren": "馬連", "umatan": "馬単", "wide": "ワイド", "sanrenpuku": "3 連複"}
    bvals = re.findall(r'"(\w+)":\s*([\d.]+)', boost or "")
    btxt = "・".join(f"{bmap.get(k, k)} ×{v}" for k, v in bvals)
    verdict = re.sub(r"^結果\s*[—-]\s*", "", res[1]) if res else ""
    axes_txt = "・".join(map_axes)
    pg.headline = ((f"走法マップの地図の軸は {axes_txt}（コード）。" if map_axes else "")
                   + (f"買い目推奨では走法が近い組の確率を {btxt}。" if btxt else "")
                   + (f"親モデルへの簡易版の履歴は {verdict}。" if res else ""))

    # 2. 数字
    parts.append(h(2, "2. 数字（親モデルへの効き。期間は節の中）", "numbers"))
    if res:
        rule = md_section(TRIALS_README, top[2]).split("\n### ")[0]
        parts.append("<details><summary>測る前に固定した合格線（節の前半）</summary><div class=\"mdsec\">"
                     + md_to_html(rule, TRIALS_README.parent, out_dir) + "</div></details>")
        parts.append('<div class="mdsec">' + md_to_html(md_section(TRIALS_README, res[2]), TRIALS_README.parent, out_dir) + "</div>"
                     + note(f"元: {link(TRIALS_README, out_dir)} の「{esc(top[1])}」の節。（このページの注: 表の「関門 3」は、固定した合格線の関門 1 の条件 3）"))
    else:
        parts.append(f"<p>{NO_RECORD}</p>")

    # 3. 事象と根拠
    parts.append(h(2, "3. 事象と根拠（コースごとの適性）", "evidence"))
    for p, title in [(RUN_RATIONALE, "rationale（得点分布の差から基準を引く測り方）の結論"),
                     (RUN_AXIS, "course_axis（寄与率の測り方）の結論")]:
        if p.exists():
            parts.append(f"<h3>{esc(title)}</h3>" + _ul(html_box_items(p), 600)
                         + note(f"元: {link(p, out_dir)}（更新 {esc(git_date(p))}）の冒頭の「結論」の枠。"))

    # 4. 限界
    parts.append(h(2, "4. 限界", "limits"))
    for p, label in [(RUN_RATIONALE, "rationale（得点分布の差から基準を引く測り方）"), (RUN_AXIS, "course_axis（寄与率の測り方。「反映」の節の前の画面の値を含む）")]:
        if p.exists():
            parts.append(f"<h3>{esc(label)}</h3>" + _ul(html_section_paras(p, r"限界", 8)) + note(f"元: {link(p, out_dir)} の「限界」の節。"))
    if lim:
        parts.append(f"<h3>親モデルへの効き</h3><p>{esc(lim)}</p>")

    # 5. 今の版
    parts.append(h(2, "5. 今の版", "version"))
    rver, rhist = _version_comment(RUN_MAP_PY, "REFERENCE_VERSION")
    aver, ahist = _version_comment(PRECOMPUTE, "ANALYSIS_VERSION")
    mv_run = next((r for r in cur[1:] if r and r[0].startswith("走法マップ")), None)
    mv_ver = re.search(r"ANALYSIS_VERSION\s*(\d+)", mv_run[1]) if mv_run and len(mv_run) > 1 else None
    parts.append(table(["何の版", "版", "その版の日付", "元"], [
        ["走法マップの基準（REFERENCE_VERSION）", rver or NO_RECORD, rhist[0][1] if rhist and rhist[0][0] == rver else NO_RECORD, "src/research/race/running_map.py"],
        ["分析全体（ANALYSIS_VERSION）", aver or NO_RECORD, ahist[0][1] if ahist and ahist[0][0] == aver else NO_RECORD, "src/research/race_analysis/precompute.py"],
        ["採用の台帳の記載", mv_run[1] if mv_run and len(mv_run) > 1 else NO_RECORD, "", "model_versions.html §1"]]))
    if mv_ver and aver and mv_ver.group(1) != aver:
        parts.append(f"<p><b>食い違い</b>: 採用の台帳は ANALYSIS_VERSION {esc(mv_ver.group(1))}、コードは {esc(aver)}。</p>")

    # 6. 年表
    parts.append(h(2, "6. 経緯の年表", "timeline"))
    for title, hist_ in [("走法マップの基準（REFERENCE_VERSION）", rhist), ("分析全体（ANALYSIS_VERSION）のうち「走法」を含む版", [r for r in ahist if "走法" in r[2]])]:
        rows = sorted(([v, d, t] for v, d, t in hist_), key=lambda r: int(r[0]), reverse=True)   # 版の番号の新しい順
        parts.append(f"<h3>{esc(title)}</h3>" + (table(["版", "日付", "変えたこと"], rows) if rows else f"<p>{NO_RECORD}</p>"))
    parts.append(note("元: running_map.py の REFERENCE_VERSION のコメント（全件）と、precompute.py の ANALYSIS_VERSION のコメントのうち「走法」を含む版。"
                      "日付はコメントの記載で、無い版は「記録なし」。"))

    # 7. 詳細
    parts.append(h(2, "7. 詳細へのリンク", "links"))
    parts.append("<ul>" + "".join(f"<li>{link(p, out_dir, t)}（更新 {esc(git_date(p))}）</li>" for p, t in [
        (RUN_RATIONALE, "走法軸 × コースの適性：優位の根拠（今の測り方）"),
        (RUN_AXIS, "走法マップ 良馬場のコース別 軸の寄与と根拠（前の測り方・反映）"),
        (RUN_AXES_MD, "走法マップの軸の確定定義（md）"),
        (RUN_SCALE_MD, "走法軸の 1〜10 の目盛り（md）"),
        (RUN_PAIRS_MD, "走法が近い馬は一緒に来るか（md）"),
        (RUN_PAIRS_HTML, "走法が近い馬は一緒に来るか（HTML）"),
        (TRIALS_README, "再学習の試しの記録（README。走法軸の履歴の節）"),
    ] if p.exists()) + "</ul>")
    if rel_:
        parts.append("<p>関連する別のモデル（採用の台帳の行）:</p>" + table(cur[0][:4], rel_) + "<ul>" + "".join(
            f"<li>{link(p, out_dir, t)}</li>" for p, t in [(HTML_DIR / "modeling/chapters/tracking_difficulty_v2.html", "追走難度 v2（tp-54）"),
                                                           (HTML_DIR / "modeling/chapters/lap_forecast.html", "ハロンごとのラップ予想（tp-55）")] if p.exists()) + "</ul>")
    pg.headline = pg.headline or NO_RECORD
    pg.body = "\n".join(parts)
    return pg
