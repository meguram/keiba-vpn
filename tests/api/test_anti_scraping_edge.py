"""別の場所（ConoHa VPS の UI など）が API を中継する構成での、クライアント IP の受け渡し（edge_sources / edge_ip_header）。"""

from __future__ import annotations

import json

from src.utils.anti_scraping import AntiScraping

VPS = "198.51.100.10"


def make(tmp_path, **over):
    cfg = {"mode": {"auth": "off", "rate_limit": "enforce", "bot": "off", "honeypot": "enforce", "login": "enforce", "session": "off"}}
    cfg.update(over)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return AntiScraping(p, clock=lambda: 1_800_000_000.0, log_dir=tmp_path / "log", env={})


def test_default_ignores_edge_header(tmp_path):
    eng = make(tmp_path)
    h = {"cf-connecting-ip": VPS, "x-keiba-client-ip": "203.0.113.9"}
    # edge_sources が空（既定）なら従来どおり cf-connecting-ip を採る
    assert eng.client_ip("127.0.0.1", h) == VPS


def test_edge_header_used_only_from_listed_source(tmp_path):
    eng = make(tmp_path, edge_sources=[VPS])
    assert eng.client_ip("127.0.0.1", {"cf-connecting-ip": VPS, "x-keiba-client-ip": "203.0.113.9"}) == "203.0.113.9"
    # VPS 以外の接続元が同じヘッダーを付けても無視（偽装できない）
    assert eng.client_ip("127.0.0.1", {"cf-connecting-ip": "198.51.100.7", "x-keiba-client-ip": "203.0.113.9"}) == "198.51.100.7"


def test_edge_header_ignored_from_untrusted_peer(tmp_path):
    eng = make(tmp_path, edge_sources=[VPS])
    h = {"cf-connecting-ip": VPS, "x-keiba-client-ip": "203.0.113.9"}
    assert eng.client_ip("198.51.100.9", h) == "198.51.100.9"  # 直接の接続元が信頼プロキシでなければ転送ヘッダは全部無視


def test_edge_header_missing_or_garbage_falls_back(tmp_path):
    eng = make(tmp_path, edge_sources=[VPS])
    assert eng.client_ip("127.0.0.1", {"cf-connecting-ip": VPS}) == VPS
    assert eng.client_ip("127.0.0.1", {"cf-connecting-ip": VPS, "x-keiba-client-ip": "not-an-ip"}) == VPS


def test_custom_header_name_and_cidr(tmp_path):
    eng = make(tmp_path, edge_sources=["198.51.100.0/24"], edge_ip_header="X-Real-Client")
    assert eng.client_ip("127.0.0.1", {"cf-connecting-ip": VPS, "x-real-client": "203.0.113.20"}) == "203.0.113.20"


def test_rate_limit_is_per_real_client_not_per_vps(tmp_path):
    eng = make(tmp_path, edge_sources=[VPS], rate_limit={"data_per_min": 3, "data_burst_per_10s": 100})
    base = {"user-agent": "Mozilla/5.0 Chrome/120", "accept-language": "ja", "referer": "https://x/", "cookie": "a=b", "sec-fetch-mode": "cors", "sec-fetch-site": "same-origin"}

    def call(client):
        return eng.check(peer="127.0.0.1", headers={**base, "cf-connecting-ip": VPS, "x-keiba-client-ip": client}, method="GET", path="/api/x/1")

    for _ in range(3):
        assert call("203.0.113.1").action == "allow"
    assert call("203.0.113.1").action != "allow"          # 同じクライアントは制限される
    assert call("203.0.113.2").action == "allow"          # 別のクライアントは巻き添えにならない
