#!/usr/bin/env python3
"""公開側サイトのビルド（標準ライブラリのみ）。限定側のアプリ（frontend/）とは独立。

使い方:
  python site_public/build.py                       # site_public/dist/ に出力して検査（リンク・HTML）
  python site_public/build.py --out /tmp/pub        # 出力先を変える
  python site_public/build.py --site-url https://pub.megu-keiba.com
  python site_public/build.py --preview-ads         # 広告枠の位置を灰色の枠で表示（レイアウト確認用。本番では使わない）
  python site_public/build.py --strict              # 【要入力】が残っていたら失敗（公開前の最終確認）
"""
from __future__ import annotations

import argparse
import html as htmllib
import json
import re
import shutil
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
PLACEHOLDER_RE = re.compile(r"(【要(?:入力|確認)[^】]*】)")
META_RE = re.compile(r"^\s*<!--meta\n(.*?)\n-->\s*", re.S)

AI_BOTS = ["GPTBot", "ChatGPT-User", "OAI-SearchBot", "ClaudeBot", "anthropic-ai", "CCBot", "Google-Extended",
           "PerplexityBot", "Bytespider", "Amazonbot", "Applebot-Extended", "cohere-ai", "Meta-ExternalAgent"]

NAV = [("/", "ホーム"), ("/articles/", "解説記事"), ("/glossary/", "用語集"), ("/faq/", "よくある質問"), ("/about/", "このサイトについて")]
FOOTER_LINKS = [("/operator/", "運営者情報"), ("/contact/", "お問い合わせ"), ("/privacy/", "プライバシーポリシー"),
                ("/terms/", "利用条件・免責"), ("/responsible/", "公営競技についての注意")]

ARTICLE_NOTE = ("<strong>ご注意</strong>この記事は分析手法の解説です。モデルの予測や過去の検証結果は、将来の的中や利益を"
                "保証するものではありません。馬券の購入は自己の責任で、無理のない範囲で行ってください（<a href=\"/responsible/\">公営競技についての注意</a>）。")


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "config.json").read_text(encoding="utf-8"))


def parse_fragment(text: str, name: str = "") -> tuple[dict, str]:
    m = META_RE.match(text)
    if not m:
        raise ValueError(f"{name}: 先頭に <!--meta ... --> がありません")
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    for req in ("title", "description"):
        if not meta.get(req):
            raise ValueError(f"{name}: meta.{req} がありません")
    return meta, text[m.end():]


def esc(s: str) -> str:
    return htmllib.escape(s, quote=True)


def mark_placeholders(body: str) -> str:
    return PLACEHOLDER_RE.sub(r'<span class="placeholder">\1</span>', body)


def add_heading_ids(body: str) -> tuple[str, list[tuple[str, str]]]:
    toc: list[tuple[str, str]] = []

    def repl(m: re.Match) -> str:
        sid = f"s{len(toc) + 1}"
        toc.append((sid, re.sub(r"<[^>]+>", "", m.group(1))))
        return f'<h2 id="{sid}">{m.group(1)}</h2>'

    return re.sub(r"<h2>(.*?)</h2>", repl, body, flags=re.S), toc


def ad_slot(position: str, cfg: dict, preview: bool) -> str:
    """広告枠。無効（既定）のときは何も出さない。--preview-ads のときだけレイアウト確認用の枠を出す。"""
    if cfg.get("ads", {}).get("enabled"):
        return f'<aside class="ad-slot" data-slot="{esc(position)}" aria-label="広告"></aside>'
    if preview:
        return f'<aside class="ad-slot" data-slot="{esc(position)}" aria-label="広告枠（無効・プレビュー）">広告枠（{esc(position)}・無効）</aside>'
    return ""


def layout(*, cfg: dict, url_path: str, title: str, description: str, body: str, kind: str = "page", date: str | None = None,
           preview_ads: bool = False, noindex: bool = False) -> str:
    base = cfg["site_url"].rstrip("/")
    canonical = base + url_path
    site = cfg["site_name"]
    full_title = title if title == site else f"{title} | {site}"
    nav = "".join(f'<li><a href="{h}">{esc(t)}</a></li>' for h, t in NAV)
    foot = "".join(f'<li><a href="{h}">{esc(t)}</a></li>' for h, t in FOOTER_LINKS)
    robots = '<meta name="robots" content="noindex">' if noindex else ""
    ld = ""
    if kind == "article":
        ld_obj = {"@context": "https://schema.org", "@type": "Article", "headline": title, "description": description,
                  "inLanguage": cfg.get("language", "ja"), "datePublished": date or cfg["updated"], "dateModified": cfg["updated"],
                  "mainEntityOfPage": canonical, "author": {"@type": "Person", "name": "運営者"}}
        ld = f'<script type="application/ld+json">{json.dumps(ld_obj, ensure_ascii=False)}</script>'
    og_type = "article" if kind == "article" else "website"
    return f"""<!doctype html>
<html lang="{esc(cfg.get('language', 'ja'))}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(full_title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="canonical" href="{esc(canonical)}">
{robots}
<meta property="og:type" content="{og_type}">
<meta property="og:title" content="{esc(full_title)}">
<meta property="og:description" content="{esc(description)}">
<meta property="og:url" content="{esc(canonical)}">
<meta property="og:site_name" content="{esc(site)}">
<meta property="og:locale" content="ja_JP">
<meta name="twitter:card" content="summary">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/assets/style.css">
{ld}
</head>
<body>
<a class="skip" href="#main">本文へ移動</a>
<header class="site"><div class="wrap"><a class="brand" href="/">{esc(site)}</a>
<nav class="main" aria-label="メインメニュー"><ul>{nav}</ul></nav></div></header>
<main id="main"><div class="wrap">
{body}
</div></main>
<footer class="site"><div class="wrap">
<p>{esc(site)} は個人が運営する分析の記録です。掲載内容は情報提供を目的としたもので、馬券の購入を勧誘するものではありません。20歳未満の方は馬券を購入できません。</p>
<ul>{foot}</ul>
<p>更新日: {esc(cfg['updated'])}</p>
</div></footer>
</body>
</html>
"""


def substitute(body: str, cfg: dict, article_cards: str = "") -> str:
    op = cfg["operator"]
    rep = {"{{ARTICLE_CARDS}}": article_cards, "{{SITE_NAME}}": esc(cfg["site_name"]), "{{UPDATED}}": esc(cfg["updated"]),
           "{{OPERATOR_NAME}}": esc(op["name"]), "{{OPERATOR_CONTACT}}": esc(op["contact"]), "{{OPERATOR_REGION}}": esc(op["region"])}
    for k, v in rep.items():
        body = body.replace(k, v)
    return body


def load_content(root: Path = ROOT) -> tuple[dict[str, tuple[dict, str]], dict[str, tuple[dict, str]]]:
    pages = {p.stem: parse_fragment(p.read_text(encoding="utf-8"), str(p)) for p in sorted((root / "content" / "pages").glob("*.html"))}
    articles = {p.stem: parse_fragment(p.read_text(encoding="utf-8"), str(p)) for p in sorted((root / "content" / "articles").glob("*.html"))}
    return pages, articles


def build(out: Path, cfg: dict, *, preview_ads: bool = False, root: Path = ROOT) -> list[str]:
    """dist を作り、書いたページの URL パス一覧（404 を除く）を返す。"""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(root / "static", out / "assets")
    pages, articles = load_content(root)
    order = [s for s in cfg["articles_order"] if s in articles] + [s for s in articles if s not in cfg["articles_order"]]
    urls: list[str] = []

    def write(url_path: str, content: str) -> None:
        target = out / url_path.lstrip("/")
        if url_path.endswith("/"):
            target = target / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    cards = "".join(
        f'<li><a href="/articles/{s}/">{esc(articles[s][0]["title"])}</a><p>{esc(articles[s][0]["description"])}</p></li>' for s in order)
    cards_html = f'<ul class="card-list">{cards}</ul>'

    for slug, (meta, body) in pages.items():
        url = "/" if slug == "index" else f"/{slug}/"
        b = mark_placeholders(substitute(body, cfg, cards_html))
        top = ad_slot("top", cfg, preview_ads)
        title = cfg["site_name"] if slug == "index" else meta["title"]
        write(url, layout(cfg=cfg, url_path=url, title=title, description=meta["description"], body=top + b, preview_ads=preview_ads))
        urls.append(url)

    write("/articles/", layout(
        cfg=cfg, url_path="/articles/", title="解説記事の一覧", description="確率の較正、期待値と回収率、展開の読み方など、分析の考え方を解説した記事の一覧です。",
        body=f"<h1>解説記事の一覧</h1><p class=\"lead\">予測の作り方・検証の仕方・限界について、個別のレースや馬の情報を含まない形で解説しています。</p>{cards_html}",
        preview_ads=preview_ads))
    urls.append("/articles/")

    for slug in order:
        meta, body = articles[slug]
        b, toc = add_heading_ids(mark_placeholders(substitute(body, cfg)))
        toc_html = ""
        if len(toc) >= 3:
            toc_html = '<nav aria-label="目次"><p class="meta">目次</p><ol>' + "".join(f'<li><a href="#{i}">{esc(t)}</a></li>' for i, t in toc) + "</ol></nav>"
        date = meta.get("date", cfg["updated"])
        page = (f'<article><h1>{esc(meta["title"])}</h1><p class="meta">公開: {esc(date)} ／ 更新: {esc(cfg["updated"])}</p>'
                f'<p class="lead">{esc(meta["description"])}</p>{toc_html}{ad_slot("article-top", cfg, preview_ads)}{b}'
                f'{ad_slot("article-bottom", cfg, preview_ads)}<div class="note" role="note">{ARTICLE_NOTE}</div></article>')
        url = f"/articles/{slug}/"
        write(url, layout(cfg=cfg, url_path=url, title=meta["title"], description=meta["description"], body=page, kind="article",
                          date=date, preview_ads=preview_ads))
        urls.append(url)

    write("/404.html", layout(cfg=cfg, url_path="/404.html", title="ページが見つかりません", description="お探しのページは見つかりませんでした。",
                              body='<h1>ページが見つかりません</h1><p>URL をご確認のうえ、<a href="/">ホーム</a>または<a href="/articles/">解説記事の一覧</a>からお探しください。</p>',
                              noindex=True))

    base = cfg["site_url"].rstrip("/")
    sm = "".join(f"<url><loc>{esc(base + u)}</loc><lastmod>{esc(cfg['updated'])}</lastmod></url>" for u in urls)
    (out / "sitemap.xml").write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{sm}</urlset>\n', encoding="utf-8")
    bots = "\n".join(f"User-agent: {b}\nDisallow: /\n" for b in AI_BOTS)
    (out / "robots.txt").write_text(
        f"# 検索エンジンは許可。AI の学習・収集系は拒否（Cloudflare の AI クローラー設定と合わせる）\nUser-agent: *\nAllow: /\n\n{bots}\nSitemap: {base}/sitemap.xml\n", encoding="utf-8")
    (out / "ads.txt").write_text(
        "# ads.txt（雛形）。広告は未導入のため、レコードは書いていません。\n# 広告ネットワークの審査に通ったら、指定された 1 行（例: <広告システムのドメイン>, <パブリッシャーID>, DIRECT, <認証ID>）をここに追加する。\n", encoding="utf-8")
    (out / "_headers").write_text(
        "/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n  X-Frame-Options: DENY\n"
        "  Permissions-Policy: camera=(), microphone=(), geolocation=(), interest-cohort=()\n"
        "  Content-Security-Policy: default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'\n"
        "/assets/*\n  Cache-Control: public, max-age=31536000, immutable\n", encoding="utf-8")
    return urls


# ─────────────────────────── 検査（HTML の簡易検証・リンク切れ）───────────────────────────

class _Checker(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.h1 = 0
        self.has_title = False
        self.has_desc = False
        self.has_canonical = False
        self.lang = ""
        self.heading_levels: list[int] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang", "")
        if tag == "title":
            self.has_title = True
        if tag == "meta" and a.get("name") == "description" and a.get("content"):
            self.has_desc = True
        if tag == "link" and a.get("rel") == "canonical":
            self.has_canonical = True
        if tag == "h1":
            self.h1 += 1
        if re.fullmatch(r"h[1-6]", tag):
            self.heading_levels.append(int(tag[1]))
        if tag == "img" and "alt" not in a:
            self.errors.append("img に alt がありません")
        if "id" in a:
            self.ids.append(a["id"])
        if tag == "a" and "href" in a:
            self.hrefs.append(a["href"])
        if tag == "link" and "href" in a and a.get("rel") != "canonical":
            self.hrefs.append(a["href"])
        if tag in ("img", "script", "source") and "src" in a:
            self.srcs.append(a["src"])
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img" and "alt" not in a:
            self.errors.append("img に alt がありません")
        if "id" in a:
            self.ids.append(a["id"])

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"閉じタグの不一致: </{tag}>（開いているのは {self.stack[-1] if self.stack else 'なし'}）")
            if tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                self.stack.pop()
        else:
            self.stack.pop()


def validate_html(text: str) -> tuple[list[str], _Checker]:
    c = _Checker()
    c.feed(text)
    c.close()
    errs = list(c.errors)
    if c.stack:
        errs.append(f"閉じていないタグ: {c.stack}")
    if not text.lstrip().lower().startswith("<!doctype html>"):
        errs.append("doctype がありません")
    if not c.lang:
        errs.append("html の lang がありません")
    if not c.has_title:
        errs.append("title がありません")
    if not c.has_desc:
        errs.append("meta description がありません")
    if not c.has_canonical:
        errs.append("canonical がありません")
    if c.h1 != 1:
        errs.append(f"h1 は 1 つであるべきです（{c.h1} 個）")
    if len(set(c.ids)) != len(c.ids):
        errs.append("id が重複しています")
    prev = 0
    for lv in c.heading_levels:
        if prev and lv > prev + 1:
            errs.append(f"見出しの階層が飛んでいます（h{prev} → h{lv}）")
        prev = lv
    return errs, c


def _target_file(out: Path, href: str) -> Path | None:
    path = href.split("#", 1)[0].split("?", 1)[0]
    if not path:
        return None
    p = out / path.lstrip("/")
    if path.endswith("/"):
        p = p / "index.html"
    return p


def check_site(out: Path) -> list[str]:
    """出力した全 HTML の検証とリンク切れ検査。外部リンクは禁止（公開側は外部へのリンクを持たない方針）。"""
    errors: list[str] = []
    htmls = sorted(out.rglob("*.html"))
    ids_by_file: dict[Path, set[str]] = {}
    parsed: dict[Path, _Checker] = {}
    for f in htmls:
        errs, c = validate_html(f.read_text(encoding="utf-8"))
        errors += [f"{f.relative_to(out)}: {e}" for e in errs]
        ids_by_file[f] = set(c.ids)
        parsed[f] = c
    for f, c in parsed.items():
        for ref in c.hrefs + c.srcs:
            if re.match(r"^[a-z][a-z0-9+.-]*:", ref, re.I) or ref.startswith("//"):
                errors.append(f"{f.relative_to(out)}: 外部・特殊リンクは使えません: {ref}")
                continue
            if ref.startswith("#"):
                if ref[1:] and ref[1:] not in ids_by_file[f]:
                    errors.append(f"{f.relative_to(out)}: ページ内リンク切れ {ref}")
                continue
            if not ref.startswith("/"):
                errors.append(f"{f.relative_to(out)}: 相対リンクは使えません（ルートからの絶対パスにする）: {ref}")
                continue
            t = _target_file(out, ref)
            if t is None or not t.exists():
                errors.append(f"{f.relative_to(out)}: リンク切れ {ref}")
                continue
            frag = ref.split("#", 1)[1] if "#" in ref else ""
            if frag and t.suffix == ".html" and frag not in ids_by_file.get(t, set()):
                errors.append(f"{f.relative_to(out)}: リンク先に id がありません {ref}")
    for need in ("robots.txt", "sitemap.xml", "ads.txt", "_headers", "404.html", "assets/style.css"):
        if not (out / need).exists():
            errors.append(f"出力に {need} がありません")
    return errors


def find_placeholders(root: Path = ROOT) -> list[str]:
    hits: list[str] = []
    for p in [root / "config.json", *sorted((root / "content").rglob("*.html"))]:
        for m in PLACEHOLDER_RE.finditer(p.read_text(encoding="utf-8")):
            hits.append(f"{p.relative_to(root)}: {m.group(1)}")
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--site-url")
    ap.add_argument("--preview-ads", action="store_true")
    ap.add_argument("--strict", action="store_true", help="【要入力】【要確認】が残っていたら失敗")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.site_url:
        cfg["site_url"] = args.site_url
    out = Path(args.out)
    urls = build(out, cfg, preview_ads=args.preview_ads)
    print(f"ページ {len(urls)} 件を {out} に出力しました")
    rc = 0
    if not args.no_check:
        errs = check_site(out)
        for e in errs:
            print("ERROR:", e)
        rc = 1 if errs else 0
        print("検査:", "OK（HTML・リンク）" if not errs else f"{len(errs)} 件の問題")
    ph = find_placeholders()
    if ph:
        print(f"未入力・要確認の箇所が {len(ph)} 件あります（公開前に解消）:")
        for h in ph[:12]:
            print("  -", h)
        if args.strict:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
