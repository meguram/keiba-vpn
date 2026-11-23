"""VPS の nginx の API の関所（templates/nginx-api-gate.conf・nginx-api-cache.conf）が、API の決まりとずれないこと。

- Cookie も Authorization も無い要求に nginx が 401 を返すパスは、src/api/tier.py の会員だけの一覧と同じ。本文も API の 401 と同じ。
- /api/ の location の制限は、外の実 IP ごとの枠と 127.0.0.1 の別枠の 2 つ（3 つのサイトのテンプレートとも）。
- 期限切れの古いものは stale-while-revalidate の秒数の内だけ（use_stale に updating を入れない）。
"""
import json
import re
from pathlib import Path

from src.api import tier

TPL = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "vps" / "templates"


def _gate_patterns() -> list[str]:
    text = (TPL / "nginx-api-gate.conf").read_text(encoding="utf-8")
    block = text[text.index("map \"$keiba_api_nocache$request_method$uri\" $keiba_member_401 {"):]
    block = block[:block.index("\n}")]
    return re.findall(r'"~\^0\(\?!OPTIONS\)\[A-Z\]\+(/[^"]+)" 1;', block)


def test_gate_patterns_match_tier_member_only_list():
    want = [p.pattern for p in tier._MEMBER_ONLY_PATTERNS]
    assert all(w.startswith("^/") for w in want)
    assert _gate_patterns() == [w[1:] for w in want]      # 先頭の ^ を除いた形で同じ順・同じ数


def test_gate_401_body_matches_api_body():
    text = (TPL / "nginx-api-cache.conf").read_text(encoding="utf-8")
    m = re.search(r"return 401 '([^']+)';", text)
    assert m and json.loads(m.group(1)) == tier.MEMBER_REQUIRED_BODY
    assert "if ($keiba_member_401)" in text


def test_gate_only_for_requests_without_cookie_and_authorization():
    zones = (TPL / "nginx-zones.conf").read_text(encoding="utf-8")
    assert 'map "$http_cookie$http_authorization" $keiba_api_nocache' in zones   # 0 = どちらも無い
    pats = _gate_patterns()
    assert pats
    for p in pats:
        assert re.match(p, "/x") is None
    full = [re.compile("^0(?!OPTIONS)[A-Z]+" + p) for p in pats]
    hit = lambda key: any(f.search(key) for f in full)  # noqa: E731
    assert hit("0GET/api/v1/betting/race-values")
    assert hit("0POST/api/v1/betting/optimize")
    assert not hit("0OPTIONS/api/v1/betting/race-values")          # API と同じく OPTIONS には掛けない
    assert not hit("1GET/api/v1/betting/race-values")              # Cookie か Authorization があれば API に任せる
    assert not hit("0GET/api/growth-curve/status")
    assert not hit("0GET/api/race/202605050811/result-status")


def test_api_location_uses_split_limits_in_all_site_templates():
    for name in ("nginx-keiba-ui.conf", "nginx-keiba-ui-cloudrun.conf", "nginx-keiba-ui-stg.conf"):
        s = (TPL / name).read_text(encoding="utf-8")
        api = s[s.index("location ~ ^/api/ {"):]
        api = api[:api.index("\n    }")]
        assert "limit_req zone=keiba_api_ext burst=60 nodelay;" in api, name
        assert "limit_req zone=keiba_api_lo burst=300 nodelay;" in api, name
        assert "zone=keiba_api " not in api, name
    zones = (TPL / "nginx-zones.conf").read_text(encoding="utf-8")   # 新規の導入（vps_02）でもサイトの参照する枠がある
    sec = zones[zones.index("# API の制限の枠（外の実 IP ごと・127.0.0.1 の別枠）"):zones.index("# API の制限の枠ここまで")]
    assert "zone=keiba_api_ext:10m rate=120r/m" in sec and "zone=keiba_api_lo:1m rate=300r/m" in sec
    assert '"127.0.0.1" "";' in sec and '"warm:$lo_port"' in sec and '"lo:$lo_port2"' in sec


def test_stale_only_within_swr_window():
    text = (TPL / "nginx-api-cache.conf").read_text(encoding="utf-8")
    use_stale = [l for l in text.splitlines() if l.startswith("proxy_cache_use_stale")]
    assert len(use_stale) == 1 and "updating" not in use_stale[0]
    assert "proxy_cache_background_update on;" in text and "keepalive_timeout 0;" in text
    assert "proxy_ignore_headers Vary;" in text and "proxy_set_header Accept-Encoding gzip;" in text and "gunzip on;" in text


def test_stg_maintenance_status_uses_own_cache_key():
    gate = (TPL / "nginx-api-gate.conf").read_text(encoding="utf-8")
    assert 'map "$keiba_cache_readonly$uri" $keiba_cache_key_prefix' in gate
    snip = (TPL / "nginx-api-cache.conf").read_text(encoding="utf-8")
    assert "proxy_cache_key $keiba_cache_key_prefix$request_uri;" in snip
