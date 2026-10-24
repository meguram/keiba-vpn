"""公開側サイト（site_public/）のビルド・検査・安全確認のテスト。標準ライブラリのみで動く。"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SITE = Path(__file__).resolve().parents[2] / "site_public"
sys.path.insert(0, str(SITE))

import build as B  # noqa: E402
import check_public_safety as S  # noqa: E402


class SiteBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="site_public_"))
        cls.cfg = B.load_config()
        cls.out = cls.tmp / "dist"
        cls.urls = B.build(cls.out, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_build_has_no_html_or_link_errors(self):
        self.assertEqual(B.check_site(self.out), [])

    def test_expected_pages_exist(self):
        for u in ["/", "/about/", "/articles/", "/glossary/", "/faq/", "/contact/", "/operator/", "/privacy/", "/terms/", "/responsible/"]:
            self.assertIn(u, self.urls)
            self.assertTrue((self.out / u.lstrip("/") / "index.html").exists() or u == "/" and (self.out / "index.html").exists(), u)
        for name in ["404.html", "robots.txt", "sitemap.xml", "ads.txt", "_headers", "assets/style.css", "assets/favicon.svg"]:
            self.assertTrue((self.out / name).exists(), name)

    def test_articles_count_and_listed_in_order(self):
        arts = [u for u in self.urls if u.startswith("/articles/") and u != "/articles/"]
        self.assertGreaterEqual(len(arts), 5)
        self.assertLessEqual(len(arts), 8)
        self.assertEqual(arts, [f"/articles/{s}/" for s in self.cfg["articles_order"]])

    def test_every_article_has_disclaimer_toc_jsonld_and_canonical(self):
        for s in self.cfg["articles_order"]:
            html = (self.out / "articles" / s / "index.html").read_text(encoding="utf-8")
            self.assertIn("保証するものではありません", html, s)
            self.assertGreaterEqual(html.count("<h2 id="), 3, s)
            self.assertIn('<nav aria-label="目次">', html, s)
            m = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)
            ld = json.loads(m.group(1))
            self.assertEqual(ld["@type"], "Article")
            for k in ("headline", "description", "datePublished", "dateModified", "inLanguage", "mainEntityOfPage"):
                self.assertIn(k, ld)
            self.assertIn(f'<link rel="canonical" href="{self.cfg["site_url"]}/articles/{s}/">', html)

    def test_sitemap_and_robots(self):
        sm = (self.out / "sitemap.xml").read_text(encoding="utf-8")
        for u in self.urls:
            self.assertIn(f"<loc>{self.cfg['site_url']}{u}</loc>", sm)
        self.assertNotIn("404", sm)
        rb = (self.out / "robots.txt").read_text(encoding="utf-8")
        self.assertIn("User-agent: *\nAllow: /", rb)
        self.assertIn("User-agent: GPTBot\nDisallow: /", rb)
        self.assertIn(f"Sitemap: {self.cfg['site_url']}/sitemap.xml", rb)

    def test_ads_disabled_by_default(self):
        self.assertFalse(self.cfg["ads"]["enabled"])
        for f in self.out.rglob("*.html"):
            self.assertNotIn("ad-slot", f.read_text(encoding="utf-8"), f.name)
        lines = [ln for ln in (self.out / "ads.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertTrue(all(ln.startswith("#") for ln in lines))

    def test_preview_ads_shows_placeholders_only_when_asked(self):
        out = self.tmp / "dist_ads"
        B.build(out, self.cfg, preview_ads=True)
        html = (out / "articles" / "reading-pace" / "index.html").read_text(encoding="utf-8")
        self.assertIn('class="ad-slot"', html)
        self.assertIn("無効", html)

    def test_no_scripts_or_trackers_and_csp(self):
        for f in self.out.rglob("*.html"):
            t = f.read_text(encoding="utf-8")
            for m in re.finditer(r"<script([^>]*)>", t):
                self.assertIn("application/ld+json", m.group(1), f.name)
            self.assertNotIn("analytics", t.lower())
        h = (self.out / "_headers").read_text(encoding="utf-8")
        self.assertIn("script-src 'none'", h)
        self.assertIn("frame-ancestors 'none'", h)

    def test_placeholders_are_styled_in_output(self):
        html = (self.out / "operator" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<span class="placeholder">【要入力', html)

    def test_no_link_to_private_app(self):
        for f in self.out.rglob("*"):
            if f.is_file() and f.suffix in (".html", ".txt", ".xml"):
                t = f.read_text(encoding="utf-8")
                self.assertNotIn("app.megu-keiba.com", t, f.name)
                self.assertNotIn("monitor.megu-keiba.com", t, f.name)


class ValidatorTest(unittest.TestCase):
    OK = ('<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>t</title><meta name="description" content="d">'
          '<link rel="canonical" href="https://x/"></head><body><h1>a</h1><h2 id="a">b</h2><p>x<br>y</p></body></html>')

    def test_ok_page_passes(self):
        self.assertEqual(B.validate_html(self.OK)[0], [])

    def test_detects_problems(self):
        bad = {
            "unclosed": self.OK.replace("<p>x<br>y</p>", "<p>x<br>y"),
            "mismatch": self.OK.replace("</p>", "</div>"),
            "no_alt": self.OK.replace("<p>", '<img src="/a.png"><p>'),
            "dup_id": self.OK.replace("<p>", '<p id="a">'),
            "two_h1": self.OK.replace("<h2", "<h1").replace("</h2>", "</h1>"),
            "skip_level": self.OK.replace("<h2", "<h4").replace("</h2>", "</h4>"),
            "no_lang": self.OK.replace(' lang="ja"', ""),
            "no_desc": self.OK.replace('<meta name="description" content="d">', ""),
        }
        for name, html in bad.items():
            self.assertTrue(B.validate_html(html)[0], name)

    def test_check_site_finds_broken_and_external_links(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            B.build(tmp / "d", B.load_config())
            idx = tmp / "d" / "index.html"
            t = idx.read_text(encoding="utf-8")
            idx.write_text(t.replace("</main>", '<a href="/nothing/">x</a><a href="https://example.org/">y</a><a href="#nope">z</a></main>'), encoding="utf-8")
            errs = " ".join(B.check_site(tmp / "d"))
            self.assertIn("リンク切れ /nothing/", errs)
            self.assertIn("外部・特殊リンク", errs)
            self.assertIn("ページ内リンク切れ #nope", errs)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_find_placeholders_and_marking(self):
        self.assertIn('<span class="placeholder">【要入力：x】</span>', B.mark_placeholders("a【要入力：x】b"))
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "content" / "pages").mkdir(parents=True)
            (tmp / "config.json").write_text('{"a":"【要確認：y】"}', encoding="utf-8")
            (tmp / "content" / "pages" / "p.html").write_text("ok", encoding="utf-8")
            self.assertEqual(len(B.find_placeholders(tmp)), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fragment_requires_meta(self):
        with self.assertRaises(ValueError):
            B.parse_fragment("<h2>x</h2>", "t")
        with self.assertRaises(ValueError):
            B.parse_fragment("<!--meta\ntitle: a\n-->\nx", "t")


class SafetyScanTest(unittest.TestCase):
    ALLOWED = {"www.megu-keiba.com", "schema.org"}

    def scan(self, text):
        return S.scan_text(text, self.ALLOWED, "t")

    def test_repo_content_is_clean(self):
        hits = S.scan_paths([SITE / "config.json", SITE / "content", SITE / "static"], S.load_allowed_domains(), SITE)
        self.assertEqual(hits, [])

    def test_built_output_is_clean(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            B.build(tmp / "d", B.load_config())
            self.assertEqual(S.scan_paths([tmp / "d"], S.load_allowed_domains(), tmp), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_detects_forbidden_patterns(self):
        cases = {
            "データ元": "出典は NetKeiba です",
            "レースID": "race 202604040201 の結果",
            "馬ID": "horse 2019104415",
            "JWT": "token eyJhIjoiMTIzNDU2Nzg5MDEyMzQ1Njc4OTA",
            "Slack": "https://hooks.slack.com/services/x",
            "内部ホスト": "https://app.megu-keiba.com/",
            "IP": "server 192.168.0.12",
            "中核データ(語)": "めぐ指数は 62.5",
            "印": "◎ 3番",
            "メール": "連絡は sample.contact.0000@gmail.com まで",
            "外部URL": "see https://evil.example.net/x",
            "電話": "03-1234-5678",
            "郵便": "〒100-0001",
            "代入": "api_key = abcdef123456",
        }
        for name, text in cases.items():
            self.assertTrue(self.scan(text), name)

    def test_allows_safe_text(self):
        for text in ["期待値は 1.2 です", "2026年10月21日", "回収率 100%", "https://www.megu-keiba.com/articles/", "mail me@example.com",
                     "https://schema.org/Article"]:
            self.assertEqual(self.scan(text), [], text)


if __name__ == "__main__":
    unittest.main()
