"""共有キャッシュ（VPS の nginx・Cloudflare の Cache Rule）に置ける応答に、会員だけに出す値が混じらないこと。

docs/operations/latency_plan_20261118.md の L6（ユーザーの条件: 会員だけに出す値が匿名の応答に混じらないことをテストで確かめる）。
Cloudflare の規則は「/api/ の許可リストで、Cookie と Authorization の無い要求だけ」をエッジに置く。Cloudflare は Vary: Cookie を見ないので、
守りは (1) public を付けるのは Cookie・Authorization の無い匿名の要求だけ、(2) その中身は匿名の区分で落とした後のもの、の 2 つになる。外部接続なし。
"""

from __future__ import annotations

import re

import pytest
from fastapi.responses import JSONResponse

from src.api.tier import LOCKED_GROUPS
from tests.api.test_tiered_access import (  # noqa: F401  tier_env・_no_anti_scraping はフィクスチャとして使う
    _DETAIL_FULL,
    _ODDS_KEYS,
    TIERS,
    TOP,
    _fastapi,
    _get,
    _no_anti_scraping,
    _walk_keys,
    tier_env,
)

RACE = "202608020311"
PATHS = (
    f"/api/race/{RACE}/predictions",
    f"/api/race/{RACE}",
    "/api/prediction-results?scope=day&date=20261018",
    "/api/prediction-results/days",
    "/api/prediction-results/news",
)
_GUEST_LOCKED = {k for ks in LOCKED_GROUPS.values() for k in ks}


@pytest.fixture()
def client(tier_env, monkeypatch):
    appmod, c = _fastapi()

    async def fake_pred(_rid):
        return JSONResponse(TOP)

    async def fake_detail(_rid):
        return JSONResponse(_DETAIL_FULL)

    monkeypatch.setattr(appmod, "_race_predictions_impl", fake_pred)
    monkeypatch.setattr(appmod, "_race_detail_impl", fake_detail)
    return c


@pytest.mark.parametrize("path", PATHS)
def test_public_only_for_anonymous_and_body_has_no_member_values(client, path):
    g = _get(client, path, "guest")
    assert g.status_code == 200, path
    cc = g.headers.get("cache-control", "")
    assert "public" in cc, (path, cc)                               # 匿名の応答は共有キャッシュに置ける（L6 の効果がある）
    if "public" in cc:
        assert "set-cookie" not in g.headers, path                 # 共有キャッシュに Set-Cookie を載せない
        keys = set(_walk_keys(g.json()))
        assert not (_GUEST_LOCKED & keys), (path, _GUEST_LOCKED & keys)
        assert not ({"race_odds", "horse_id", "bet_suggestion"} & keys), path
        if path.startswith("/api/race/"):
            assert not ({"odds", "popularity", "win_odds", "ev_win"} & keys), path
    for t in TIERS:
        if t == "guest":
            continue
        r = _get(client, path, t)
        assert "public" not in r.headers.get("cache-control", ""), (path, t, r.headers.get("cache-control"))
        assert "cloudflare-cdn-cache-control" not in r.headers, (path, t)      # Cloudflare にも置かせない


@pytest.mark.parametrize("path", PATHS)
def test_any_cookie_or_authorization_is_not_shared(client, path):
    """セッション以外の Cookie（解析・Cloudflare の Cookie など）や Authorization が付いた要求も、共有しない（Cloudflare の規則の条件と同じ線）。"""
    for headers in ({"Cookie": "_ga=GA1.1.1"}, {"Authorization": "Bearer x"}):
        r = client.get(path, headers=headers)
        assert "public" not in r.headers.get("cache-control", ""), (path, headers)
        assert "cloudflare-cdn-cache-control" not in r.headers, (path, headers)


def test_guest_predictions_are_shared_and_locked(client):
    """匿名の予測は共有キャッシュに置ける（L6 の効果がある）うえで、会員限定・実オッズの項目が無い。"""
    g = _get(client, f"/api/race/{RACE}/predictions", "guest")
    assert "public" in g.headers.get("cache-control", "")
    # Cloudflare には、nginx に 5 分以上置く応答だけ 5 分。今年の確定前のレース（s-maxage 30 秒）は置かせない（古さの上限を延ばさない）
    m = re.search(r"s-maxage=(\d+)", g.headers.get("cache-control", ""))
    want = "public, max-age=300" if m and int(m.group(1)) >= 300 else "no-store"
    assert g.headers.get("cloudflare-cdn-cache-control") == want
    keys = set(g.json()) | {k for e in g.json()["predictions"] for k in e}
    assert not (_GUEST_LOCKED & keys) and not ({"win_odds", "popularity", "ev_win"} & keys)
