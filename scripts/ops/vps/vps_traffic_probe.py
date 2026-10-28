"""VPS の nginx アクセスログを日別に集計する（監視ポータルのアクセス解析用）。``traffic_client.py`` が使う（単体でも ``python3 vps_traffic_probe.py`` で JSON を 1 行出す）。

読み取りだけ・標準ライブラリのみ（VPS にリポジトリは無い）。
生の IP・UA は出さない: 訪問者数は VPS の中で ``sha256(日付 + IP + UA)`` の個数に畳み、集計値だけを返す。

出力: ``{"probe_version", "at", "readable", "reason", "files", "days": {YYYYMMDD: {...}}, "windows": {...}}``
日付は JST。ログの時刻に付いたオフセットを JST に直して日を決める。
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

PROBE_VERSION = 1
JST = timezone(timedelta(hours=9))
LOG_GLOB = os.environ.get("KEIBA_NGINX_ACCESS_GLOB", "/var/log/nginx/access.log*")
KEEP_TOP = 60  # 日ごとに残す上位件数（長期の合算に使う）
OWN_HOSTS = ("megu-keiba.com",)

LINE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>[A-Z]+) (?P<target>\S+)(?: [^"]*)?" (?P<status>\d{3}) (?P<bytes>\S+) '
    r'"(?P<ref>[^"]*)" "(?P<ua>[^"]*)"'
)
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
ASSET_EXT = (".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".avif", ".woff", ".woff2", ".ttf", ".map", ".json", ".txt", ".xml", ".webmanifest")
SEO_FILES = ("/robots.txt", "/sitemap.xml", "/ads.txt")

# 名前 → (区分, 正規表現)。上から最初に一致したものを採る
BOTS: list[tuple[str, str, re.Pattern[str]]] = [(n, k, re.compile(p, re.I)) for n, k, p in [
    ("Googlebot", "search", r"googlebot|google-inspectiontool|storebot-google|googleother|apis-google|adsbot-google|mediapartners-google"),
    ("Bingbot", "search", r"bingbot|bingpreview|msnbot"),
    ("Yahoo Slurp", "search", r"slurp|yahoo.*(?:bot|crawler)|yahoocachesystem"),
    ("DuckDuckBot", "search", r"duckduckbot"),
    ("Baiduspider", "search", r"baiduspider"),
    ("Yandex", "search", r"yandex"),
    ("Applebot", "search", r"applebot"),
    ("GPTBot", "ai", r"gptbot|chatgpt-user|oai-searchbot"),
    ("ClaudeBot", "ai", r"claudebot|claude-web|anthropic-ai"),
    ("Perplexity", "ai", r"perplexity"),
    ("Google-Extended", "ai", r"google-extended"),
    ("CCBot", "ai", r"ccbot|bytespider|amazonbot|cohere|diffbot|meta-externalagent|facebookbot"),
    ("Ahrefs", "seo_tool", r"ahrefs"),
    ("Semrush", "seo_tool", r"semrush"),
    ("MJ12", "seo_tool", r"mj12bot|dotbot|petalbot|screaming frog"),
    ("SNS プレビュー", "social", r"facebookexternalhit|twitterbot|slackbot|linebot|line-poker|discordbot|whatsapp|telegrambot"),
    ("監視ツール", "tool", r"uptimerobot|pingdom|statuscake|cloudflare-healthchecks|cloudflare-traffic-manager|better ?uptime|healthcheck|keiba-"),
    ("スクリプト", "tool", r"curl/|wget/|python-requests|python-urllib|go-http-client|okhttp|java/|libwww|httpx|aiohttp|node-fetch|axios"),
    ("その他の bot", "other_bot", r"bot\b|crawler|spider|scrapy|headless"),
]]
MOBILE = re.compile(r"iphone|ipod|android.*mobile|windows phone|mobile", re.I)
TABLET = re.compile(r"ipad|android(?!.*mobile)|tablet", re.I)

SEARCH_REFERRERS = {"google": "Google", "yahoo": "Yahoo!", "bing": "Bing", "duckduckgo": "DuckDuckGo", "baidu": "Baidu", "ecosia": "Ecosia"}
SOCIAL_REFERRERS = {"t.co": "X", "twitter.com": "X", "x.com": "X", "facebook.com": "Facebook", "l.facebook.com": "Facebook", "line.me": "LINE",
                    "instagram.com": "Instagram", "youtube.com": "YouTube", "note.com": "note", "b.hatena.ne.jp": "はてブ", "reddit.com": "Reddit"}


def parse_ts(s: str) -> datetime | None:
    m = re.match(r"(\d+)/(\w+)/(\d+):(\d+):(\d+):(\d+) ([+-])(\d{2})(\d{2})", s)
    if not m or m.group(2) not in MONTHS:
        return None
    d, mon, y, hh, mm, ss, sign, oh, om = m.groups()
    off = timedelta(hours=int(oh), minutes=int(om)) * (1 if sign == "+" else -1)
    try:
        return datetime(int(y), MONTHS[mon], int(d), int(hh), int(mm), int(ss), tzinfo=timezone(off)).astimezone(JST)
    except ValueError:
        return None


def bot_of(ua: str) -> tuple[str, str] | None:
    for name, kind, rx in BOTS:
        if rx.search(ua):
            return name, kind
    return None


def device_of(ua: str) -> str:
    if TABLET.search(ua):
        return "tablet"
    if MOBILE.search(ua):
        return "mobile"
    return "desktop"


_ID = re.compile(r"/\d{6,}(?=/|$)")
_HASH = re.compile(r"/[0-9a-f]{16,}(?=/|$)")
_NUM = re.compile(r"/\d+(?=/|$)")


def pattern_of(path: str) -> str:
    """個別のレース・馬などを 1 つの型にまとめる（/race/202604040401 → /race/:id）。"""
    p = _HASH.sub("/:hash", _ID.sub("/:id", path))
    return _NUM.sub("/:n", p) or "/"


def referrer_of(ref: str) -> tuple[str, str] | None:
    """('search'|'social'|'other', 表示名)。直接アクセス・自サイト内は None。"""
    if not ref or ref == "-":
        return None
    m = re.match(r"https?://([^/:?#]+)", ref, re.I)
    if not m:
        return None
    host = m.group(1).lower()
    if any(host == h or host.endswith("." + h) for h in OWN_HOSTS):
        return None
    for key, label in SEARCH_REFERRERS.items():
        if key in host.split("."):
            return "search", label
    for key, label in SOCIAL_REFERRERS.items():
        if host == key or host.endswith("." + key):
            return "social", label
    return "other", host


class Day:
    def __init__(self) -> None:
        self.requests = 0
        self.page_views = 0
        self.rsc = 0
        self.api = 0
        self.assets = 0
        self.humans: set[str] = set()
        self.status = Counter()
        self.bot_kind = Counter()
        self.bots = Counter()
        self.bot_status = Counter()
        self.pages = Counter()
        self.patterns = Counter()
        self.referrers = Counter()
        self.referrer_kind = Counter()
        self.err_pages = Counter()
        self.err5xx = Counter()
        self.hours = [0] * 24
        self.devices = Counter()
        self.seo_files = Counter()
        self.internal = 0

    def out(self) -> dict:
        top = lambda c, n=KEEP_TOP: [[k, v] for k, v in c.most_common(n)]  # noqa: E731
        return {
            "requests": self.requests, "page_views": self.page_views, "rsc": self.rsc, "api": self.api, "assets": self.assets,
            "visitors": len(self.humans), "internal": self.internal,
            "status": dict(self.status), "bot_kind": dict(self.bot_kind), "bots": top(self.bots, 20), "bot_status": dict(self.bot_status),
            "pages": top(self.pages), "patterns": top(self.patterns, 30), "referrers": top(self.referrers, 30), "referrer_kind": dict(self.referrer_kind),
            "err_pages": top(self.err_pages, 30), "err5xx": top(self.err5xx, 20), "hours": self.hours, "devices": dict(self.devices),
            "seo_files": dict(self.seo_files),
        }


def classify_path(path: str) -> str:
    if path.startswith("/api/"):
        return "api"
    if path.startswith(("/_next/", "/static/", "/data/image/", "/assets/")) or path.lower().endswith(ASSET_EXT) and path not in SEO_FILES:
        return "asset"
    return "page"


def analyze(lines, days: dict[str, Day], windows: dict[str, set[str]], today: str, now: datetime) -> int:
    """ログ行を日別に積む。解析できた行数を返す。"""
    n = 0
    win_days = {"today": {today}, "yesterday": {(now - timedelta(days=1)).strftime("%Y%m%d")},
                "7d": {(now - timedelta(days=i)).strftime("%Y%m%d") for i in range(7)},
                "14d": {(now - timedelta(days=i)).strftime("%Y%m%d") for i in range(14)}}
    for raw in lines:
        m = LINE.match(raw)
        if not m:
            continue
        ts = parse_ts(m.group("ts"))
        if ts is None:
            continue
        n += 1
        day = ts.strftime("%Y%m%d")
        d = days.setdefault(day, Day())
        d.requests += 1
        ip, ua, method, status = m.group("ip"), m.group("ua"), m.group("method"), int(m.group("status"))
        target = m.group("target")
        path, _, query = target.partition("?")
        d.status[f"{status // 100}xx"] += 1
        if ip in ("127.0.0.1", "::1") or path in ("/api/health", "/healthz"):
            d.internal += 1
            continue
        bot = bot_of(ua)
        kind = classify_path(path)
        if path in SEO_FILES:
            d.seo_files[f"{path} {status // 100}xx"] += 1
        if bot:
            d.bot_kind[bot[1]] += 1
            d.bots[bot[0]] += 1
            d.bot_status[f"{bot[0]} {status // 100}xx"] += 1
            continue
        if kind == "asset" or path in SEO_FILES:
            d.assets += 1
            continue
        if kind == "api":
            d.api += 1
            if status >= 500:
                d.err5xx[path] += 1
            continue
        if method != "GET":
            continue
        if "_rsc=" in query:
            d.rsc += 1
            continue
        if status >= 500:
            d.err5xx[path] += 1
        if status == 404:
            d.err_pages[path] += 1
            continue
        if status not in (200, 304):
            continue
        d.page_views += 1
        d.pages[path] += 1
        d.patterns[pattern_of(path)] += 1
        d.hours[ts.hour] += 1
        d.devices[device_of(ua)] += 1
        ref = referrer_of(m.group("ref"))
        if ref:
            d.referrer_kind[ref[0]] += 1
            d.referrers[f"{ref[0]}:{ref[1]}"] += 1
        else:
            d.referrer_kind["direct"] += 1
        vid = hashlib.sha256(f"{day}|{ip}|{ua}".encode()).hexdigest()[:20]
        d.humans.add(vid)
        # 期間をまたぐ訪問者は日付を外した ID で数える
        wid = hashlib.sha256(f"{ip}|{ua}".encode()).hexdigest()[:20]
        for w, ds in win_days.items():
            if day in ds:
                windows.setdefault(w, set()).add(wid)
    return n


def open_log(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def run(glob_pat: str = LOG_GLOB, now: datetime | None = None) -> dict:
    now = now or datetime.now(JST)
    res: dict = {"probe_version": PROBE_VERSION, "at": now.isoformat(timespec="seconds"), "readable": False, "reason": None, "files": [], "days": {}, "windows": {}}
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        res["reason"] = f"ログが見つからない: {glob_pat}"
        return res
    days: dict[str, Day] = {}
    windows: dict[str, set[str]] = {}
    unreadable: list[str] = []
    lines_total = 0
    for p in paths:
        try:
            with open_log(p) as fh:
                lines_total += analyze(fh, days, windows, now.strftime("%Y%m%d"), now)
            res["files"].append(os.path.basename(p))
        except PermissionError:
            unreadable.append(os.path.basename(p))
        except (OSError, EOFError) as exc:  # 壊れた gz など 1 つで全体を落とさない
            unreadable.append(f"{os.path.basename(p)}({type(exc).__name__})")
    if not res["files"]:
        res["reason"] = "ログを読む権限がありません（" + ", ".join(unreadable[:3]) + "）"
        return res
    res["readable"] = True
    if unreadable:
        res["unreadable"] = unreadable
    res["lines"] = lines_total
    res["days"] = {k: v.out() for k, v in sorted(days.items())}
    res["windows"] = {k: len(v) for k, v in windows.items()}
    return res


if __name__ == "__main__":
    t0 = time.time()
    out = run()
    out["took_s"] = round(time.time() - t0, 2)
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
