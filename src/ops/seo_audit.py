"""監視ポータルの『SEO 点検』: 公開側サイト（site_public）を一時フォルダにビルドし、出力した HTML と robots/sitemap を点検する。

検索流入を取れる状態か（技術・ページごとの書き方・公開前の残り作業）を、ネットワークなしで決まる項目だけで採点する。
実サイトへの到達（公開済みか）だけは短い HTTP 確認。限定側（app.megu-keiba.com）は Access で守る前提で、全拒否が正しい。
監視ポータル（Cloud Run）のイメージには ``site_public/`` が無いので、**ここは CI で動かして結果を GCS（``ops/seo/audit.json``）に置く**
（``python -m src.scripts.ops.seo_audit_publish``）。ポータルは ``read`` で読むだけ。
"""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE = REPO_ROOT / "site_public"
FRONT = REPO_ROOT / "frontend"

TITLE_RANGE = (10, 40)
DESC_RANGE = (40, 140)
THIN_CHARS = 800


def _load_build(site: Path):
    spec = importlib.util.spec_from_file_location("site_public_build", site / "build.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def item(level: str, label: str, detail: str = "", fix: str = "") -> dict:
    return {"level": level, "label": label, "detail": detail, "fix": fix}


def _text(html: str) -> str:
    body = re.search(r"<body[^>]*>(.*)</body>", html, re.S | re.I)
    s = re.sub(r"<(script|style|nav|header|footer)\b.*?</\1>", "", body.group(1) if body else html, flags=re.S | re.I)
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", s))


def page_metrics(url: str, html: str) -> dict:
    g = lambda rx: (re.search(rx, html, re.S | re.I) or [None, ""])[1].strip()  # noqa: E731
    title, desc = g(r"<title>(.*?)</title>"), g(r'<meta name="description" content="(.*?)"')
    m = {
        "url": url, "title": title, "title_len": len(title), "desc_len": len(desc), "h1": len(re.findall(r"<h1\b", html, re.I)),
        "chars": len(_text(html)), "canonical": bool(re.search(r'<link rel="canonical"', html)), "og": "og:title" in html,
        "jsonld": "application/ld+json" in html, "noindex": bool(re.search(r'name="robots" content="[^"]*noindex', html)),
        "lang_ja": bool(re.search(r'<html[^>]*lang="ja"', html)), "viewport": 'name="viewport"' in html,
        "imgs_no_alt": len(re.findall(r"<img\b(?![^>]*\balt=)", html, re.I)),
        "links_internal": len(re.findall(r'<a [^>]*href="/[^"]*"', html)),
    }
    issues = []
    if not m["noindex"]:
        if not TITLE_RANGE[0] <= m["title_len"] <= TITLE_RANGE[1]:
            issues.append(f"タイトルが {m['title_len']} 文字（{TITLE_RANGE[0]}〜{TITLE_RANGE[1]} 文字が目安）")
        if not DESC_RANGE[0] <= m["desc_len"] <= DESC_RANGE[1]:
            issues.append(f"説明文が {m['desc_len']} 文字（{DESC_RANGE[0]}〜{DESC_RANGE[1]} 文字が目安）")
        if m["h1"] != 1:
            issues.append(f"h1 が {m['h1']} 個（1 個が目安）")
        if m["chars"] < THIN_CHARS and url.startswith("/articles/") and url != "/articles/":
            issues.append(f"本文が約 {m['chars']} 文字（記事は {THIN_CHARS} 文字以上が目安）")
        if m["imgs_no_alt"]:
            issues.append(f"alt のない画像が {m['imgs_no_alt']} 件")
        if m["links_internal"] < 3:
            issues.append(f"内部リンクが {m['links_internal']} 本（少ない）")
    m["issues"] = issues
    return m


def _live(get: Callable[[str], tuple[int | None, str]], url: str) -> tuple[int | None, str]:
    try:
        return get(url)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}"


def default_get(url: str) -> tuple[int | None, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "keiba-monitor-seo/1"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:  # noqa: S310
            return r.status, r.read(20000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"[:120]


def audit(get: Callable[[str], tuple[int | None, str]] | None = None, site: Path | None = None, front: Path | None = None) -> dict:
    site, front = site or SITE, front or FRONT
    b = _load_build(site)
    cfg = b.load_config(site / "config.json")
    base = (cfg.get("site_url") or "").rstrip("/")
    groups: list[dict] = []

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dist"
        urls = b.build(out, cfg, root=site)
        errors = b.check_site(out)
        pages = [page_metrics(u, (out / (u.lstrip("/") + "index.html" if u.endswith("/") else u.lstrip("/"))).read_text(encoding="utf-8")) for u in urls]
        robots = (out / "robots.txt").read_text(encoding="utf-8")
        sitemap = (out / "sitemap.xml").read_text(encoding="utf-8")
        has_404 = (out / "404.html").exists()
        notfound_noindex = has_404 and "noindex" in (out / "404.html").read_text(encoding="utf-8")
    placeholders = b.find_placeholders(site)

    indexable = [p for p in pages if not p["noindex"]]
    in_sitemap = len(re.findall(r"<loc>", sitemap))
    tech = [
        item("ok" if re.search(r"User-agent: \*\s*\nAllow: /", robots) else "ng", "robots.txt が検索エンジンを許可",
             "User-agent: * / Allow: /" if "Allow: /" in robots else "検索を許可していない", "build.py の robots 生成を確認"),
        item("ok" if "Sitemap:" in robots else "ng", "robots.txt に Sitemap の行がある"),
        item("ok" if in_sitemap == len(indexable) else "warn", f"sitemap.xml の URL 数 {in_sitemap} / 索引対象 {len(indexable)}"),
        item("ok" if all(p["canonical"] for p in indexable) else "ng", "全ページに canonical がある"),
        item("ok" if all(p["lang_ja"] and p["viewport"] for p in pages) else "warn", "全ページに lang=ja と viewport がある"),
        item("ok" if all(p["og"] for p in indexable) else "warn", "全ページに OGP（SNS・検索の見え方）がある"),
        item("ok" if all(p["jsonld"] for p in pages if p["url"].startswith("/articles/") and p["url"] != "/articles/") else "warn", "記事に構造化データ（JSON-LD）がある"),
        item("ok" if notfound_noindex else "warn", "404 ページが noindex"),
        item("ok" if not errors else "ng", "HTML の検証・リンク切れ検査", f"{len(errors)} 件: " + "; ".join(errors[:3]) if errors else "問題なし"),
    ]
    groups.append({"title": "技術", "items": tech})

    titles = Counter(p["title"] for p in indexable)
    page_issues = sum(len(p["issues"]) for p in indexable)
    dup = [t for t, c in titles.items() if c > 1]
    content = [
        item("ok" if not page_issues else "warn", f"ページごとの書き方（タイトル・説明・見出し・本文量）: 指摘 {page_issues} 件", "ページ一覧の『指摘』欄を参照"),
        item("ok" if not dup else "warn", "タイトルの重複がない", ", ".join(dup[:3])),
        item("info", f"索引対象 {len(indexable)} ページ（記事 {sum(1 for p in indexable if p['url'].startswith('/articles/') and p['url'] != '/articles/')} 本）"),
    ]
    groups.append({"title": "コンテンツ", "items": content})

    blockers = []
    if placeholders:
        blockers.append(item("ng", f"未入力の項目が {len(placeholders)} 件残っています（【要入力】【要確認】）", "; ".join(placeholders[:4]) + (" ほか" if len(placeholders) > 4 else ""),
                             "python3 site_public/build.py --strict が通るまで"))
    if cfg.get("ads", {}).get("enabled"):
        blockers.append(item("warn", "広告が有効です", "審査・規約・法令の確認が済んでいるか"))
    blockers.append(item("info", "Search Console・解析は未連携", "公開後に所有確認をすると、検索順位・表示回数・クリックをここに出せます"))
    groups.append({"title": "公開前の残り作業", "items": blockers})

    live = []
    if base:
        get = get or default_get
        code, body = _live(get, base + "/robots.txt")
        if code == 200 and "Sitemap:" in body:
            live.append(item("ok", f"{base} が公開されている", "robots.txt を取得できた"))
        elif code == 200:
            live.append(item("warn", f"{base} は応答するが robots.txt が想定と違う", "旧サイトの可能性"))
        else:
            live.append(item("info", f"{base} はまだ公開されていません", f"応答 {code}（公開前なら正常）"))
        scode, _ = _live(get, base + "/sitemap.xml")
        if code == 200:
            live.append(item("ok" if scode == 200 else "ng", "sitemap.xml を取得できる", f"応答 {scode}"))
    groups.append({"title": "公開状況", "items": live})

    fr = front / "app" / "robots.ts"
    app_items = []
    try:
        txt = fr.read_text(encoding="utf-8")
        app_items.append(item("ok" if re.search(r'userAgent:\s*"\*",\s*disallow:\s*"/"', txt) else "ng", "限定側（app）は全クローラーを拒否", "会員向けの画面は検索に出さない方針（Access でも保護）"))
    except OSError:
        app_items.append(item("info", "限定側の robots.ts を読めない"))
    groups.append({"title": "限定側（app.megu-keiba.com）", "items": app_items})

    scored = [i for g in groups for i in g["items"] if i["level"] != "info"]
    ok_n = sum(1 for i in scored if i["level"] == "ok")
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "site_url": base, "score": round(100 * ok_n / len(scored)) if scored else None,
            "counts": {"ok": ok_n, "warn": sum(1 for i in scored if i["level"] == "warn"), "ng": sum(1 for i in scored if i["level"] == "ng")},
            "groups": groups, "pages": pages, "placeholders": len(placeholders)}


OBJECT = "ops/seo/audit.json"


def publish(io: Any, **kw: Any) -> dict:
    """点検して GCS に置く。ビルドが落ちても『error』を置いて、ポータルが古い良好な結果を出し続けないようにする。"""
    try:
        val = audit(**kw)
        val["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        val = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "groups": [], "pages": [], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    io.put_bytes(OBJECT, json.dumps(val, ensure_ascii=False).encode("utf-8"), content_type="application/json")
    return val


def read(io: Any) -> dict:
    try:
        raw = io.get_bytes(OBJECT)
        return json.loads(raw.decode("utf-8")) if raw else {"status": "not_collected", "groups": [], "pages": []}
    except Exception:  # noqa: BLE001
        return {"status": "error", "error": "読み込めません", "groups": [], "pages": []}
