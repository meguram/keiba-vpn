"""サイトへのスクレイピング対策の判定エンジン（レート制限・bot 判定・ハニーポット・ログイン試行制限・未ログイン検知）。

フレームワークに依存しない純粋なロジック。FastAPI / Flask へは src/api/middleware/anti_scraping.py が接続する。
設定は config/anti_scraping.json（30 秒ごとに読み直し）。mode は off / detect（ログのみ）/ enforce（ブロック）。
状態はプロセス内メモリ（再起動で消える。ブロックは短時間のものが中心なので許容）。ログだけ追記ファイルに残す。
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "anti_scraping.json"
RELOAD_SECONDS = 30.0
FORWARD_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded", "cf-connecting-ip", "x-forwarded-host")
RACE_ID_RE = re.compile(r"(?<!\d)(\d{12})(?!\d)")
HORSE_ID_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
MODES = ("off", "detect", "enforce")
BEHAVIORAL_SIGNALS = frozenset({"enumeration", "inhuman_speed"})

DEFAULTS: dict[str, Any] = {
    "mode": {"auth": "detect", "rate_limit": "detect", "bot": "detect", "honeypot": "enforce", "login": "enforce", "session": "detect"},
    "trusted_proxies": ["127.0.0.1", "::1"],
    "client_ip_headers": ["cf-connecting-ip", "x-real-ip", "x-forwarded-for"],
    "allowlist_ips": [],
    # 別の場所（例: ConoHa VPS の UI）が Access のサービストークンで API を呼ぶ構成用。cf-connecting-ip がここに載っている接続元
    # （=その VPS の IP）からのときだけ、edge_ip_header の値を本当のクライアント IP として採る。空なら従来どおり（何もしない）
    "edge_sources": [],
    "edge_ip_header": "x-keiba-client-ip",
    "internal_path_prefixes": ["/api/health", "/api/v1/health", "/api/scrape-jobs", "/api/v1/scrape-jobs"],
    "static_prefixes": ["/static/", "/_next/", "/data/image/", "/favicon", "/robots.txt"],
    "auth": {"protected_prefixes": ["/api/"], "public_exempt_prefixes": ["/api/health", "/api/auth/", "/api/v1/auth/", "/api/_hp/"]},
    "rate_limit": {"data_per_min": 120, "data_burst_per_10s": 40, "page_per_min": 180, "static_per_min": 600,
                   "user_data_per_min": 240, "distinct_races_per_5min": 60, "distinct_horses_per_5min": 120},
    "block": {"tiers": [{"after_violations": 3, "seconds": 900}, {"after_violations": 8, "seconds": 86400}, {"after_violations": 20, "seconds": 604800}],
              "violation_decay_seconds": 3600},
    "bot": {"threshold": 6, "decay_seconds": 600,
            "weights": {"ua_missing": 4, "ua_scraper_lib": 5, "ua_headless": 4, "claims_browser_without_sec_fetch": 3,
                        "no_accept_language_on_data": 1, "no_referer_on_data": 1, "no_cookie_on_data": 1,
                        "api_without_page_visit": 2, "enumeration": 4, "inhuman_speed": 3},
            "scraper_ua_patterns": ["python-requests", "curl/", "wget/", "scrapy", "go-http-client", "httpx", "aiohttp"],
            "headless_ua_patterns": ["headlesschrome", "headless"],
            "page_visit_memory_seconds": 1800, "enumeration_sequential_run": 8},
    "honeypot": {"path_prefixes": ["/api/_hp/", "/api/archive/"], "block_seconds": 604800},
    "login": {"paths": ["/login", "/api/v1/auth/login"], "max_failures": 5, "window_seconds": 900,
              "lockout_seconds": [900, 3600, 86400], "global_failures_per_hour_alert": 30},
    "session": {"distinct_ips_window_seconds": 600, "distinct_ips_limit": 4, "block_seconds": 900},
    "log": {"dir": "data/local/meta/anti_scraping", "same_event_throttle_seconds": 60},
    "notify": {"blocks_per_hour_alert": 50},
}


def _merge(base: dict, over: Mapping) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Decision:
    """action: allow / block。detected は検知した理由（detect モードでは allow のまま理由だけ残る）。"""

    action: str = "allow"
    status: int = 200
    reason: str = ""
    retry_after: int = 0
    detected: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:8]


def _parse_ip(s: str | None):
    if not s:
        return None
    s = s.strip().strip("[]")
    if s.lower().startswith("::ffff:"):
        s = s[7:]
    s = re.sub(r"%.*$", "", s)
    try:
        return ipaddress.ip_address(s)
    except ValueError:
        # "1.2.3.4:5678" 形式
        m = re.match(r"^(\d+\.\d+\.\d+\.\d+):\d+$", s)
        if m:
            try:
                return ipaddress.ip_address(m.group(1))
            except ValueError:
                return None
        return None


def _in_nets(ip, entries: list[str]) -> bool:
    if ip is None:
        return False
    for e in entries:
        try:
            if "/" in e:
                if ip in ipaddress.ip_network(e, strict=False):
                    return True
            elif ip == ipaddress.ip_address(e.strip().strip("[]")):
                return True
        except ValueError:
            continue
    return False


class _Window:
    """タイムスタンプの滑動窓カウンタ。"""

    def __init__(self) -> None:
        self.q: deque[float] = deque()

    def add(self, now: float, span: float) -> int:
        self.q.append(now)
        self.trim(now, span)
        return len(self.q)

    def trim(self, now: float, span: float) -> None:
        while self.q and self.q[0] <= now - span:
            self.q.popleft()

    def count(self, now: float, span: float) -> int:
        self.trim(now, span)
        return len(self.q)


class AntiScraping:
    def __init__(self, config_path: Path | None = None, *, clock: Callable[[], float] = time.time,
                 log_dir: Path | None = None, env: Mapping[str, str] | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.clock = clock
        self._log_dir_override = Path(log_dir) if log_dir else None
        self.env = os.environ if env is None else env
        self._cfg: dict[str, Any] = DEFAULTS
        self._cfg_loaded = -1e18
        self._cfg_mtime = -1.0
        self._lock = threading.RLock()
        self._windows: dict[str, _Window] = {}
        self._viol: dict[str, list[float]] = {}
        self._blocked: dict[str, tuple[float, str]] = {}
        self._scores: dict[str, tuple[float, float, frozenset]] = {}
        self._page_seen: dict[str, float] = {}
        self._ids: dict[str, dict[str, float]] = {}
        self._seq: dict[str, deque] = {}
        self._login_fail: dict[str, deque] = {}
        self._login_lock: dict[str, tuple[float, int]] = {}
        self._login_global: deque = deque()
        self._sess_ips: dict[str, dict[str, float]] = {}
        self._log_throttle: dict[tuple[str, str], float] = {}
        self._counters: deque = deque()

    # ───────── 設定 ─────────
    def config(self) -> dict[str, Any]:
        now = self.clock()
        if now - self._cfg_loaded < RELOAD_SECONDS:
            return self._cfg
        with self._lock:
            self._cfg_loaded = now
            try:
                # config/anti_scraping.local.json があれば上書きする（ホスト固有の値＝VPS の IP など、リポジトリに載せない設定用）
                local = self.config_path.with_name(self.config_path.stem + ".local.json")
                st = (self.config_path.stat().st_mtime, local.stat().st_mtime if local.exists() else 0.0)
                if st != self._cfg_mtime:
                    cfg = _merge(DEFAULTS, json.loads(self.config_path.read_text(encoding="utf-8")))
                    if local.exists():
                        cfg = _merge(cfg, json.loads(local.read_text(encoding="utf-8")))
                    self._cfg = cfg
                    self._cfg_mtime = st
            except (OSError, ValueError):
                self._cfg = DEFAULTS
        return self._cfg

    def mode(self, layer: str) -> str:
        override = str(self.env.get("KEIBA_ANTI_SCRAPING", "")).strip().lower()
        if override == "off":
            return "off"
        m = str(self.config().get("mode", {}).get(layer, "detect")).lower()
        return m if m in MODES else "detect"

    # ───────── クライアント IP ─────────
    def client_ip(self, peer: str | None, headers: Mapping[str, str]) -> str:
        """直接の接続元が信頼できるプロキシ（既定はループバック）のときだけ転送ヘッダを採用する。"""
        cfg = self.config()
        peer_ip = _parse_ip(peer)
        trusted = list(cfg.get("trusted_proxies", []))
        if peer_ip is None or not _in_nets(peer_ip, trusted):
            return str(peer_ip) if peer_ip else (peer or "unknown")
        h = {k.lower(): v for k, v in headers.items()}
        edge = self._ip_from_edge(h, cfg)
        if edge:
            return edge
        found = self._ip_from_headers(h, cfg, trusted)
        if found:
            return found
        if self.has_forward_headers(headers):
            # 転送ヘッダはあるが本当のクライアント IP が無い（トンネルがループバックから中継している等）。全員が同じ IP に見えて
            # 巻き添えで制限しないよう、UA と言語から作った指紋をキーにする（IP が取れる前段に変えると精度が上がる）
            return "fp:" + sha8(h.get("user-agent", "") + "|" + h.get("accept-language", ""))
        return str(peer_ip)

    def _ip_from_edge(self, h: Mapping[str, str], cfg: Mapping[str, Any]) -> str:
        """edge_sources（信頼する中継元の IP）から来た要求だけ、edge_ip_header の値をクライアント IP として採る。"""
        sources = list(cfg.get("edge_sources") or [])
        if not sources:
            return ""
        src = _parse_ip((h.get("cf-connecting-ip") or "").split(",")[0])
        if src is None or not _in_nets(src, sources):
            return ""
        name = str(cfg.get("edge_ip_header") or "x-keiba-client-ip").lower()
        ip = _parse_ip((h.get(name) or "").split(",")[0])
        return str(ip) if ip is not None else ""

    def _ip_from_headers(self, h: Mapping[str, str], cfg: Mapping[str, Any], trusted: list[str]) -> str:
        for name in cfg.get("client_ip_headers", []):
            v = h.get(name.lower())
            if not v:
                continue
            if name.lower() == "x-forwarded-for":
                # 右から見て、信頼できるプロキシでない最初の値（偽装された先頭値を採らない）
                parts = [p.strip() for p in v.split(",") if p.strip()]
                for p in reversed(parts):
                    ip = _parse_ip(p)
                    if ip is not None and not _in_nets(ip, trusted):
                        return str(ip)
                if parts:
                    ip = _parse_ip(parts[0])
                    if ip is not None and not _in_nets(ip, trusted):
                        return str(ip)
                continue
            ip = _parse_ip(v.split(",")[0])
            if ip is not None and not _in_nets(ip, trusted):
                return str(ip)
        return ""

    def has_forward_headers(self, headers: Mapping[str, str]) -> bool:
        h = {k.lower() for k in headers.keys()}
        return any(x in h for x in FORWARD_HEADERS)

    # ───────── 分類 ─────────
    def path_kind(self, path: str) -> str:
        cfg = self.config()
        for p in cfg.get("honeypot", {}).get("path_prefixes", []):
            if path.startswith(p):
                return "honeypot"
        for p in cfg.get("internal_path_prefixes", []):
            if path.startswith(p):
                return "internal"
        for p in cfg.get("static_prefixes", []):
            if path.startswith(p):
                return "static"
        if path.startswith("/api/"):
            return "data"
        return "page"

    def is_internal_caller(self, peer: str | None, headers: Mapping[str, str], ip: str) -> bool:
        """cron・ランナー・監視ポータルなど内部ジョブ。信頼プロキシからの転送ヘッダの無い直接アクセス、または許可リスト。"""
        cfg = self.config()
        if _in_nets(_parse_ip(ip), list(cfg.get("allowlist_ips", []))):
            return True
        peer_ip = _parse_ip(peer)
        if peer_ip is not None and _in_nets(peer_ip, list(cfg.get("trusted_proxies", []))) and not self.has_forward_headers(headers):
            return True
        return False

    # ───────── 内部の窓・ブロック ─────────
    def _win(self, key: str) -> _Window:
        w = self._windows.get(key)
        if w is None:
            if len(self._windows) > 50000:
                self._gc()
            w = self._windows[key] = _Window()
        return w

    def _gc(self) -> None:
        now = self.clock()
        self._windows = {k: w for k, w in self._windows.items() if w.q and w.q[-1] > now - 600}
        self._blocked = {k: v for k, v in self._blocked.items() if v[0] > now}
        self._ids = {k: d for k, d in self._ids.items() if d and max(d.values()) > now - 600}

    def is_blocked(self, key: str) -> tuple[bool, int, str]:
        now = self.clock()
        b = self._blocked.get(key)
        if not b:
            return False, 0, ""
        until, why = b
        if until <= now:
            self._blocked.pop(key, None)
            return False, 0, ""
        return True, int(until - now) + 1, why

    def block(self, key: str, seconds: float, why: str) -> None:
        with self._lock:
            now = self.clock()
            cur = self._blocked.get(key)
            until = now + seconds
            if not cur or cur[0] < until:
                self._blocked[key] = (until, why)

    def unblock(self, key: str) -> None:
        self._blocked.pop(key, None)

    def _violation(self, key: str, why: str) -> int:
        """違反を数え、段階的ブロックの対象になったら秒数を返す（戻り値: ブロック秒、0=まだ）。"""
        cfg = self.config()["block"]
        now = self.clock()
        decay = float(cfg.get("violation_decay_seconds", 3600))
        lst = [t for t in self._viol.get(key, []) if t > now - decay]
        lst.append(now)
        self._viol[key] = lst
        secs = 0
        for tier in sorted(cfg.get("tiers", []), key=lambda t: t["after_violations"]):
            if len(lst) >= int(tier["after_violations"]):
                secs = int(tier["seconds"])
        return secs

    # ───────── ログ ─────────
    def log_dir(self) -> Path:
        if self._log_dir_override:
            return self._log_dir_override
        d = Path(self.config().get("log", {}).get("dir", "data/local/meta/anti_scraping"))
        return d if d.is_absolute() else REPO_ROOT / d

    def event(self, kind: str, action: str, *, ip: str = "", user: str = "", ua: str = "", path: str = "",
              reason: str = "", score: float | None = None, **extra: Any) -> None:
        now = self.clock()
        thr = float(self.config().get("log", {}).get("same_event_throttle_seconds", 60))
        k = (ip or user, f"{kind}:{reason}")
        last = self._log_throttle.get(k, -1e18)
        if now - last < thr:
            return
        self._log_throttle[k] = now
        if len(self._log_throttle) > 20000:
            self._log_throttle = {a: b for a, b in self._log_throttle.items() if b > now - thr}
        self._counters.append((now, action))
        while self._counters and self._counters[0][0] < now - 3600:
            self._counters.popleft()
        rec = {"ts": round(now, 3), "kind": kind, "action": action, "ip": ip, "user": user,
               "ua": (ua or "")[:120], "path": path[:160], "reason": reason}
        if score is not None:
            rec["score"] = round(score, 2)
        rec.update(extra)
        try:
            d = self.log_dir()
            d.mkdir(parents=True, exist_ok=True)
            day = time.strftime("%Y%m%d", time.gmtime(now + 9 * 3600))
            with open(d / f"events-{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def blocks_last_hour(self) -> int:
        now = self.clock()
        return sum(1 for t, a in self._counters if a == "block" and t > now - 3600)

    # ───────── bot スコア ─────────
    def _bot_signals(self, kind: str, headers: Mapping[str, str]) -> list[str]:
        cfg = self.config()["bot"]
        h = {k.lower(): v for k, v in headers.items()}
        ua = h.get("user-agent", "")
        low = ua.lower()
        out: list[str] = []
        if not ua.strip():
            out.append("ua_missing")
        else:
            if any(p in low for p in cfg.get("scraper_ua_patterns", [])):
                out.append("ua_scraper_lib")
            if any(p in low for p in cfg.get("headless_ua_patterns", [])):
                out.append("ua_headless")
            claims_browser = ("mozilla/" in low) and not any(x in low for x in ("bot", "spider", "crawl"))
            if claims_browser and kind in ("data", "page") and not any(k.startswith("sec-fetch-") for k in h):
                out.append("claims_browser_without_sec_fetch")
        if kind == "data":
            if not h.get("accept-language"):
                out.append("no_accept_language_on_data")
            if not h.get("referer"):
                out.append("no_referer_on_data")
            if not h.get("cookie"):
                out.append("no_cookie_on_data")
        return out

    def _enumeration(self, key: str, path: str, now: float) -> bool:
        """連番の race_id を順番に取りに来ているか。"""
        m = RACE_ID_RE.search(path)
        if not m:
            return False
        n = int(m.group(1))
        seq = self._seq.setdefault(key, deque(maxlen=64))
        seq.append((now, n))
        need = int(self.config()["bot"].get("enumeration_sequential_run", 8))
        if len(seq) < need:
            return False
        last = list(seq)[-need:]
        if last[-1][0] - last[0][0] > 120:
            return False
        diffs = [b[1] - a[1] for a, b in zip(last, last[1:])]
        return all(0 < d <= 3 for d in diffs)

    # ───────── 本体 ─────────
    def check(self, *, peer: str | None, headers: Mapping[str, str], method: str, path: str,
              authed: bool = False, session_token: str = "") -> Decision:
        cfg = self.config()
        d = Decision()
        ip = self.client_ip(peer, headers)
        kind = self.path_kind(path)
        h = {k.lower(): v for k, v in headers.items()}
        ua = h.get("user-agent", "")
        user = sha8(session_token) if session_token else ""

        if self.mode("rate_limit") == "off" and self.mode("bot") == "off" and self.mode("auth") == "off" and self.mode("honeypot") == "off":
            return d
        if self.is_internal_caller(peer, headers, ip):
            return d
        now = self.clock()

        # 既存のブロック
        for key in (f"ip:{ip}", f"tok:{user}" if user else ""):
            if not key:
                continue
            b, ra, why = self.is_blocked(key)
            if b and kind != "internal":
                d.detected.append(f"blocked:{why}")
                self.event("blocked", "block", ip=ip, user=user, ua=ua, path=path, reason=why)
                return Decision("block", 403, f"blocked:{why}", ra, d.detected)

        # ハニーポット
        if kind == "honeypot":
            m = self.mode("honeypot")
            d.detected.append("honeypot")
            self.event("honeypot", "block" if m == "enforce" else "detect", ip=ip, user=user, ua=ua, path=path, reason="honeypot")
            if m == "enforce":
                self.block(f"ip:{ip}", float(cfg["honeypot"].get("block_seconds", 604800)), "honeypot")
                return Decision("block", 403, "honeypot", int(cfg["honeypot"].get("block_seconds", 604800)), d.detected)
            return d

        if kind == "internal":
            return d

        # 未ログイン検知（401 を返す判断は最後。未ログインの連打もレート制限・bot 判定の対象にするため）
        auth_dec: Decision | None = None
        am = self.mode("auth")
        if am != "off" and kind == "data" and not authed and self._needs_auth(path):
            d.detected.append("anon_protected")
            self.event("auth", "block" if am == "enforce" else "detect", ip=ip, user=user, ua=ua, path=path, reason="anon_protected")
            if am == "enforce":
                auth_dec = Decision("block", 401, "login_required", 0, d.detected)

        # 閲覧の記録（人間はまずページ/静的資産を読む）
        if kind in ("page", "static"):
            self._page_seen[ip] = now
            if len(self._page_seen) > 50000:
                lim = now - float(cfg["bot"].get("page_visit_memory_seconds", 1800))
                self._page_seen = {k: v for k, v in self._page_seen.items() if v > lim}

        # レート制限
        rl = self.mode("rate_limit")
        if rl != "off":
            r = cfg["rate_limit"]
            limits: list[tuple[str, int, float]] = []
            if kind == "data":
                limits += [(f"d60:{ip}", int(r["data_per_min"]), 60.0), (f"d10:{ip}", int(r["data_burst_per_10s"]), 10.0)]
                if user:
                    limits.append((f"u60:{user}", int(r["user_data_per_min"]), 60.0))
            elif kind == "page":
                limits.append((f"p60:{ip}", int(r["page_per_min"]), 60.0))
            else:
                limits.append((f"s60:{ip}", int(r["static_per_min"]), 60.0))
            over = ""
            for key, lim, span in limits:
                if self._win(key).add(now, span) > lim:
                    over = key.split(":")[0]
            if kind == "data":
                for name, rx, lim_key in (("races", RACE_ID_RE, "distinct_races_per_5min"), ("horses", HORSE_ID_RE, "distinct_horses_per_5min")):
                    m = rx.search(path)
                    if m:
                        who = user or ip
                        dd = self._ids.setdefault(f"{name}:{who}", {})
                        dd[m.group(1)] = now
                        for k0 in [k for k, t in dd.items() if t < now - 300]:
                            dd.pop(k0, None)
                        if len(dd) > int(r[lim_key]):
                            over = over or f"distinct_{name}"
            if over:
                d.detected.append(f"rate:{over}")
                secs = self._violation(f"ip:{ip}", over)
                self.event("rate", "block" if rl == "enforce" else "detect", ip=ip, user=user, ua=ua, path=path, reason=over)
                if rl == "enforce":
                    if secs:
                        self.block(f"ip:{ip}", secs, f"rate:{over}")
                        return Decision("block", 429, f"rate:{over}", secs, d.detected)
                    return Decision("block", 429, f"rate:{over}", 30, d.detected)

        # bot スコア
        bm = self.mode("bot")
        if bm != "off" and kind in ("data", "page"):
            bc = cfg["bot"]
            w = bc["weights"]
            sigs = self._bot_signals(kind, headers)
            seen = self._page_seen.get(ip, -1e18)
            if kind == "data" and now - seen > float(bc.get("page_visit_memory_seconds", 1800)):
                sigs.append("api_without_page_visit")
            if kind == "data" and self._enumeration(user or ip, path, now):
                sigs.append("enumeration")
            if kind == "data" and self._win(f"d10:{ip}").count(now, 10.0) > 3 * int(cfg["rate_limit"]["data_burst_per_10s"]) // 4 and (not ua or "ua_scraper_lib" in sigs):
                sigs.append("inhuman_speed")
            cur, ts, seen = self._scores.get(ip, (0.0, now, frozenset()))
            cur *= max(0.0, 1.0 - (now - ts) / float(bc.get("decay_seconds", 600)))
            if cur <= 0.01:
                cur, seen = 0.0, frozenset()
            # 静的な特徴（UA・ヘッダの欠落など）は同じ窓の中で 1 回だけ加点、行動の特徴（連番・速度）は毎回加点
            add = 0.0
            for sg in sigs:
                if sg in BEHAVIORAL_SIGNALS:
                    add += float(w.get(sg, 0))
                elif sg not in seen:
                    add += float(w.get(sg, 0))
                    seen = seen | {sg}
            cur += add
            self._scores[ip] = (cur, now, seen)
            if len(self._scores) > 50000:
                self._scores = {k: v for k, v in self._scores.items() if v[0] > 0.5 and v[1] > now - 600}
            if cur >= float(bc["threshold"]):
                d.detected.append("bot_score")
                self.event("bot", "block" if bm == "enforce" else "detect", ip=ip, user=user, ua=ua, path=path,
                           reason="bot_score:" + ",".join(sigs), score=cur)
                if bm == "enforce":
                    secs = self._violation(f"ip:{ip}", "bot") or 900
                    self.block(f"ip:{ip}", secs, "bot")
                    return Decision("block", 403, "bot", secs, d.detected)

        # セッションの多重 IP
        if user and self.mode("session") != "off":
            sc = cfg["session"]
            dd = self._sess_ips.setdefault(user, {})
            dd[ip] = now
            for k0 in [k for k, t in dd.items() if t < now - float(sc["distinct_ips_window_seconds"])]:
                dd.pop(k0, None)
            if len(dd) >= int(sc["distinct_ips_limit"]):
                d.detected.append("session_many_ips")
                sm = self.mode("session")
                self.event("session", "block" if sm == "enforce" else "detect", ip=ip, user=user, ua=ua, path=path,
                           reason=f"session_many_ips:{len(dd)}")
                if sm == "enforce":
                    self.block(f"tok:{user}", float(sc["block_seconds"]), "session_many_ips")
                    return Decision("block", 403, "session_many_ips", int(sc["block_seconds"]), d.detected)
        return auth_dec or d

    def _needs_auth(self, path: str) -> bool:
        a = self.config().get("auth", {})
        if not any(path.startswith(p) for p in a.get("protected_prefixes", [])):
            return False
        return not any(path.startswith(p) for p in a.get("public_exempt_prefixes", []))

    # ───────── ログイン試行 ─────────
    def is_login_path(self, path: str) -> bool:
        return path in self.config().get("login", {}).get("paths", [])

    def login_check(self, ip: str) -> tuple[bool, int]:
        """ログイン試行を受け付けてよいか。(許可, 待つ秒数)。"""
        if self.mode("login") == "off":
            return True, 0
        now = self.clock()
        lk = self._login_lock.get(ip)
        if lk and lk[0] > now:
            return False, int(lk[0] - now) + 1
        return True, 0

    def login_record(self, ip: str, ok: bool) -> None:
        if self.mode("login") == "off":
            return
        cfg = self.config()["login"]
        now = self.clock()
        if ok:
            self._login_fail.pop(ip, None)
            return
        q = self._login_fail.setdefault(ip, deque())
        q.append(now)
        while q and q[0] < now - float(cfg["window_seconds"]):
            q.popleft()
        self._login_global.append(now)
        while self._login_global and self._login_global[0] < now - 3600:
            self._login_global.popleft()
        self.event("login", "detect", ip=ip, reason=f"login_failed:{len(q)}")
        if len(q) >= int(cfg["max_failures"]):
            prev = self._login_lock.get(ip, (0.0, 0))
            level = prev[1] if prev[0] > now - 86400 else 0
            secs = cfg["lockout_seconds"][min(level, len(cfg["lockout_seconds"]) - 1)]
            self._login_lock[ip] = (now + secs, level + 1)
            q.clear()
            self.event("login", "block", ip=ip, reason=f"login_locked:{secs}s")

    def login_global_alert(self) -> bool:
        return len(self._login_global) >= int(self.config()["login"].get("global_failures_per_hour_alert", 30))

    # ───────── 集計 ─────────
    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        return {
            "blocked": {k: int(v[0] - now) for k, v in self._blocked.items() if v[0] > now},
            "blocks_last_hour": self.blocks_last_hour(),
            "login_global_last_hour": len(self._login_global),
            "modes": {k: self.mode(k) for k in DEFAULTS["mode"]},
        }


_engine: AntiScraping | None = None
_engine_lock = threading.Lock()


def get_engine() -> AntiScraping:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = AntiScraping()
    return _engine
