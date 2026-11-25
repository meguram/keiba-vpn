"""API の GET 応答への短いキャッシュ指示（許可リスト方式・private・Vary: Cookie）。"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api import http_cache_headers as H

CFG = {"enabled": True, "rules": [{"prefix": "/api/race/", "max_age": 30}, {"prefix": "/api/race-calendar", "max_age": 300}],
       "never": ["/api/health", "/api/v1/auth"]}


def test_max_age_rules():
    f = lambda p, m="GET", st=200, cc=False: H.max_age_for(p, m, st, cc, CFG)  # noqa: E731
    assert f("/api/race/202604040102/predictions") == 30
    assert f("/api/race-calendar") == 300
    assert f("/api/v1/auth/status") is None and f("/api/health") is None
    assert f("/api/other") is None
    assert f("/api/race/x", m="POST") is None and f("/api/race/x", st=404) is None and f("/api/race/x", cc=True) is None
    assert H.max_age_for("/api/race/x", "GET", 200, False, {**CFG, "enabled": False}) is None


def test_fastapi_headers(monkeypatch):
    monkeypatch.setattr(H, "config", lambda: CFG)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    @app.get("/api/v1/auth/status")
    def auth():
        return {"ok": True}

    @app.get("/api/race-calendar")
    def cal():
        return JSONResponse({"a": 1}, headers={"Cache-Control": "no-store"})

    H.install_fastapi(app)
    c = TestClient(app)
    r = c.get("/api/race/1")
    assert r.headers["cache-control"] == "private, max-age=30" and "cookie" in r.headers["vary"].lower()
    assert "cache-control" not in c.get("/api/v1/auth/status").headers
    assert c.get("/api/race-calendar").headers["cache-control"] == "no-store"      # すでにある指示は上書きしない


def test_shared_cache_only_for_anonymous(monkeypatch):
    cfg = {**CFG, "rules": [{"prefix": "/api/race/", "max_age": 30, "s_maxage": 45}, {"prefix": "/api/race-calendar", "max_age": 300}]}
    monkeypatch.setattr(H, "config", lambda: cfg)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    @app.get("/api/race-calendar")
    def cal():
        return {"a": 1}

    H.install_fastapi(app)
    c = TestClient(app)
    anon = c.get("/api/race/1")
    assert anon.headers["cache-control"] == "public, max-age=30, s-maxage=45" and "cookie" in anon.headers["vary"].lower()
    assert c.get("/api/race/1", headers={"Cookie": "keiba_dev_session=x"}).headers["cache-control"] == "private, max-age=30"
    assert c.get("/api/race/1", headers={"Authorization": "Bearer x"}).headers["cache-control"] == "private, max-age=30"
    assert c.get("/api/race-calendar").headers["cache-control"] == "private, max-age=300"     # s_maxage の無いルールは常に private


def test_dev_auto_login_is_not_anonymous(monkeypatch):
    monkeypatch.setenv("KEIBA_ENV", "dev")
    assert H.is_anonymous_request("", "", "127.0.0.1", {"host": "localhost:23110"}) is False
    assert H.is_anonymous_request("", "", "203.0.113.5", {"host": "stg.example.com"}) is True
    assert H.is_anonymous_request("a=b", "", "203.0.113.5", {}) is False


# ---- 確定済みのレースは共有キャッシュに長く置く ----
from datetime import datetime, timedelta, timezone  # noqa: E402

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=JST)
SCFG = {
    "enabled": True,
    "rules": [{"prefix": "/api/race/", "max_age": 30, "s_maxage": 30}, {"prefix": "/api/race-list/", "max_age": 60, "s_maxage": 60}],
    "never": ["/api/health", "/api/admin"],
    "settled": {"s_maxage": 604800, "grace_days": 3, "prefixes": ["/api/race-list/", "/api/race/"]},
}


def test_settled_by_date_token():
    f = lambda p: H.settled_s_maxage_for(p, SCFG, NOW)  # noqa: E731
    assert f("/api/race-list/20261026") == 604800            # ちょうど3日前 = 確定
    assert f("/api/race-list/20261027") is None              # 2日前はまだ動く
    assert f("/api/race-list/20261029") is None
    assert f("/api/race-list/20261301") is None              # 不正な日付は短いまま
    assert f("/api/other/20200101") is None                  # 許可リスト外
    assert f("/api/admin/20200101") is None


def test_settled_by_race_id_year_rule_and_peek(monkeypatch):
    f = lambda p: H.settled_s_maxage_for(p, SCFG, NOW)  # noqa: E731
    monkeypatch.setattr(H, "_race_date_peek", None)
    assert f("/api/race/202506050811/predictions") == 604800     # 前の年 → 確定扱い
    assert f("/api/race/202606050811/predictions") is None       # 今年で日付不明 → 短いまま
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: {"202606050811": "20260906", "202606050812": "20261028"}.get(rid, ""))
    assert f("/api/race/202606050811") == 604800                 # 索引で9月と分かる
    assert f("/api/race/202606050812") is None                   # 昨日のレースは動く
    assert f("/api/race/202606050899") is None                   # 索引に無い
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: (_ for _ in ()).throw(RuntimeError("boom")))
    assert f("/api/race/202606050811") is None                   # 例外は短いまま（落ちない）


def test_settled_disabled_or_zero():
    assert H.settled_s_maxage_for("/api/race-list/20200101", {**SCFG, "enabled": False}, NOW) is None
    assert H.settled_s_maxage_for("/api/race-list/20200101", {**SCFG, "settled": {}}, NOW) is None


def test_cache_control_value_settled_only_for_anonymous():
    assert H.cache_control_value(30, 30, True, 604800) == "public, max-age=30, s-maxage=604800"
    assert H.cache_control_value(30, 30, True) == "public, max-age=30, s-maxage=30"
    assert H.cache_control_value(30, 30, False, 604800) == "private, max-age=30"
    assert H.cache_control_value(30, None, True, 604800) == "private, max-age=30"     # s_maxage の無いルールは長くしない


def test_fastapi_settled_header(monkeypatch):
    monkeypatch.setattr(H, "config", lambda: SCFG)
    monkeypatch.setattr(H, "_race_date_peek", None)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    H.install_fastapi(app)
    c = TestClient(app)
    y = datetime.now(JST).year
    assert c.get(f"/api/race/{y - 1}06050811").headers["cache-control"] == "public, max-age=30, s-maxage=604800"
    assert c.get(f"/api/race/{y - 1}06050811", headers={"Cookie": "k=v"}).headers["cache-control"] == "private, max-age=30"
    assert c.get(f"/api/race/{y}06050811").headers["cache-control"] == "public, max-age=30, s-maxage=30"


def test_fastapi_date_list_anonymous_cacheable(monkeypatch):
    """日付一覧(scrape-dates 等)は匿名なら自分では Cache-Control を付けず、許可リストの public, s-maxage に任せる。会員は no-store。"""
    cfg = {"enabled": True, "rules": [{"prefix": "/api/scrape-dates", "max_age": 30, "s_maxage": 120}], "never": []}
    monkeypatch.setattr(H, "config", lambda: cfg)
    from fastapi import Request

    from src.api.app import _NO_STORE_HEADERS, _no_store_unless_anonymous

    app = FastAPI()

    @app.get("/api/scrape-dates")
    def dates(request: Request):
        return JSONResponse({"dates": []}, headers=_no_store_unless_anonymous(request))

    H.install_fastapi(app)
    c = TestClient(app)
    assert c.get("/api/scrape-dates").headers["cache-control"] == "public, max-age=30, s-maxage=120"
    assert c.get("/api/scrape-dates", headers={"Cookie": "k=v"}).headers["cache-control"] == _NO_STORE_HEADERS["Cache-Control"]
    assert c.get("/api/scrape-dates", headers={"Authorization": "Bearer x"}).headers["cache-control"] == _NO_STORE_HEADERS["Cache-Control"]


def test_upcoming_dated_races_get_six_hours_and_long_prefixes_thirty_days():
    cfg = {**SCFG, "settled": {**SCFG["settled"], "upcoming_s_maxage": 21600},
           "long": {"s_maxage": 2592000, "prefixes": ["/api/horse/"]}}
    f = lambda p: H.settled_s_maxage_for(p, cfg, NOW)  # noqa: E731
    assert f("/api/race-list/20261026") == 604800            # 確定済みは長く
    assert f("/api/race-list/20261029") == 21600             # 当日は 6 時間（上書きが漏れたときの上限）
    assert f("/api/race-list/20261101") == 21600             # 未来も 6 時間
    assert f("/api/race-list/20261301") is None              # 日付が読めないものは短いまま
    assert f("/api/race/202699999999") is None               # 日付不明の race_id（索引なし）も短いまま
    assert f("/api/horse/2021101429/detail") == 2592000      # 馬のページは 30 日（馬の目印で上書き）
    assert f("/api/admin/x") is None
    assert H.cache_control_value(30, 30, True, f("/api/race-list/20261029")) == "public, max-age=30, s-maxage=21600"
    assert H.cache_control_value(30, 30, False, f("/api/race-list/20261029")) == "private, max-age=30"   # 会員は共有しない


def test_upcoming_race_id_stays_short_even_when_indexed(monkeypatch):
    """race_id は索引で日付が分かっても、確定前は 6 時間にしない（cache-warm が取り直さない final-odds・result-status なども前方一致で当たるため）。"""
    cfg = {**SCFG, "settled": {**SCFG["settled"], "upcoming_s_maxage": 21600}}
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: {"202606050811": "20261029", "202606050812": "20261025"}.get(rid, ""))
    f = lambda p: H.settled_s_maxage_for(p, cfg, NOW)  # noqa: E731
    assert f("/api/race/202606050811") is None                     # 当日のレース → 短い s_maxage のまま
    assert f("/api/race/202606050811/final-odds") is None
    assert f("/api/v1/races/202606050811/running-map") is None
    assert f("/api/race/202606050812/predictions") == 604800       # 4 日前（確定済み）→ 長く
    assert f("/api/race-list/20261029") == 21600                   # 開催日のパスは今どおり 6 時間


def test_app_race_date_peek_builds_index_in_background_without_gcs(monkeypatch, tmp_path):
    """app の peek: 初回は待たずに返し、裏のスレッドが race_lists（local_only・ローカルのファイル）から索引を作る。GCS は触らない。"""
    import json
    import time as _t

    import src.api.app as A

    class _Store:
        """race_lists だけを持つ storage の代わり。GCS への経路（list_blobs・_get_bucket）を呼んだら落とす。"""
        def __init__(self, d):
            self.d = d

        def list_keys(self, category, year=None):
            assert category == "race_lists"
            return sorted(p.stem for p in self.d.glob("*.json"))

        def load(self, category, key, bypass_cache=False):
            assert category == "race_lists"
            return json.loads((self.d / f"{key}.json").read_text(encoding="utf-8"))

        def _get_bucket(self):
            raise AssertionError("GCS を読んではいけない")

    (tmp_path / "20261025.json").write_text(json.dumps({"races": [{"race_id": "202606050812"}]}), encoding="utf-8")
    st = _Store(tmp_path)
    monkeypatch.setattr(A, "_get_storage", lambda: st)
    monkeypatch.setattr(A, "_race_date_index", {})
    monkeypatch.setattr(A, "_race_date_index_built", False)
    monkeypatch.setattr(A, "_race_date_index_tried_at", None)
    assert H._race_date_peek is A._peek_race_date
    first = A._peek_race_date("202606050812")
    for _ in range(300):
        if A._race_date_index_built:
            break
        _t.sleep(0.01)
    assert first in ("", "20261025")
    assert A._race_date_index_built
    assert A._peek_race_date("202606050812") == "20261025"
    assert A._resolve_date_for_race(st, "202606050812") == "20261025"


def test_race_lists_is_local_only_and_never_lists_gcs(monkeypatch, tmp_path):
    """索引の元の race_lists は local_only: 一覧も読み出しもローカルだけで、GCS（_get_bucket）に落ちない。"""
    from src.scraper.storage import HybridStorage

    assert HybridStorage.CATEGORY_MAP["race_lists"] == "local_only"
    st = HybridStorage.__new__(HybridStorage)
    monkeypatch.setattr(st, "_page_ref_dir", tmp_path / "pr", raising=False)
    monkeypatch.setattr(st, "_local_dir", tmp_path / "local", raising=False)

    def _no_gcs(*a, **k):
        raise AssertionError("GCS を読んではいけない")

    monkeypatch.setattr(st, "_get_bucket", _no_gcs, raising=False)
    assert st.list_keys("race_lists") == []
    assert st.load("race_lists", "20261025") is None


def test_stale_while_revalidate_bounded_by_s_maxage():
    """匿名の public 応答だけ stale-while-revalidate=min(上限, s-maxage) を付ける。古さの上限は s-maxage＋その秒数。会員には付けない。"""
    assert H.cache_control_value(30, 30, True, None, 60) == "public, max-age=30, s-maxage=30, stale-while-revalidate=30"   # 当日のレース: 最大 60 秒
    assert H.cache_control_value(60, 60, True, None, 60) == "public, max-age=60, s-maxage=60, stale-while-revalidate=60"
    assert H.cache_control_value(30, 600, True, None, 60) == "public, max-age=30, s-maxage=600, stale-while-revalidate=60"
    assert H.cache_control_value(30, 30, True, 2592000, 60) == "public, max-age=30, s-maxage=2592000, stale-while-revalidate=60"
    assert H.cache_control_value(30, 30, False, None, 60) == "private, max-age=30"
    assert H.cache_control_value(30, 30, True, None, 0) == "public, max-age=30, s-maxage=30"
    assert H.stale_while_revalidate_cap({"stale_while_revalidate": 60}) == 60
    assert H.stale_while_revalidate_cap({}) == 0
    assert H.stale_while_revalidate_cap({"stale_while_revalidate": "x"}) == 0


def test_fastapi_adds_stale_while_revalidate_only_for_anonymous(monkeypatch):
    cfg = {**SCFG, "stale_while_revalidate": 60}
    monkeypatch.setattr(H, "config", lambda: cfg)
    monkeypatch.setattr(H, "_race_date_peek", None)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    H.install_fastapi(app)
    c = TestClient(app)
    y = datetime.now(JST).year
    assert c.get(f"/api/race/{y}06050811").headers["cache-control"] == "public, max-age=30, s-maxage=30, stale-while-revalidate=30"
    assert c.get(f"/api/race/{y}06050811", headers={"Cookie": "k=v"}).headers["cache-control"] == "private, max-age=30"


def test_confirmed_results_are_settled_without_waiting_grace_days(monkeypatch):
    """結果が出た（このプロセスが見た）レース・開催日は、開催からの日数を待たずに長く置く。cache-warm が取り直すパスだけ。
    確定の結果があり前日以前なら 30 日、それ以外（当日・日付不明・速報だけ）は 6 時間。知らないものは今までどおり日数の規則。"""
    cfg = {**SCFG, "settled": {**SCFG["settled"], "upcoming_s_maxage": 21600}}
    dates = {"202606050811": "20261029", "202606050812": "20261029", "202606050801": "20261027", "202606050802": "20261027"}
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: dates.get(rid, ""))
    levels = {"202606050811": "confirmed", "202606050801": "confirmed", "202606050802": "flash",
              "20261028": "confirmed", "20261027": "flash"}
    monkeypatch.setattr(H, "_settled_peek", lambda t: levels.get(t, ""))
    f = lambda p: H.settled_s_maxage_for(p, cfg, NOW)  # noqa: E731
    assert f("/api/race/202606050811") == 21600                   # 当日は確定でも 6 時間（同じ日の取り直しに備える）
    assert f("/api/race/202606050811/predictions") == 21600
    assert f("/api/race/202606050811/entry/2021101429/card") is None      # cache-warm が取り直さないパスは日数の規則のまま
    assert f("/api/v1/races/202606050811/running-map") is None
    assert f("/api/race/202606050812") is None                    # 同じ日でも、結果を見ていないレースは短いまま
    assert f("/api/race/202606050801") == 604800                  # 確定の結果があり前日以前 → 長く（日数の規則なら 2 日前はまだ動く）
    assert f("/api/race/202606050802") == 21600                   # 速報だけは何日たっても 6 時間まで（一部の馬・審議中の仮の着順がありうる）
    assert f("/api/race-list/20261028") == 604800                 # 前日以前で全レースが確定した日の一覧
    assert f("/api/race-list/20261027") == 21600                  # 速報だけの日（2 日前。日数の規則でもまだ）は 6 時間
    assert f("/api/race-list/20261029") == 21600                  # 結果がまだの日は今どおり 6 時間
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: "")
    assert f("/api/race/202606050801") == 21600                   # 日付が分からなければ 6 時間
    monkeypatch.setattr(H, "_settled_peek", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    assert f("/api/race/202606050801") is None                    # 覗き見の失敗は日数の規則に落ちる


def test_app_marks_result_levels_from_results_it_already_read(monkeypatch):
    """app: 結果を読んだ処理（result-status・レース詳細）が、確定・速報を見たら覚え、覗き見がその段を返す。確定は速報で上書きしない。"""
    import src.api.app as A

    monkeypatch.setenv("KEIBA_ANTI_SCRAPING", "off")
    assert H._settled_peek is A._result_level
    A._mark_result("202606050811", "flash")
    A._mark_result("202606050811", "confirmed")
    A._mark_result("202606050811", "flash")
    assert A._result_level("202606050811") == "confirmed" and A._result_level("202606050812") == ""
    entries = {"entries": [{"horse_id": "x", "rank": 1}]}

    class _St:
        def __init__(self, cats):
            self.cats = cats

        def exists(self, cat, key):
            return cat in self.cats

        def load(self, cat, key):
            return entries if cat in self.cats else None

    c = TestClient(A.app)
    for rid, cats, want in (("202606050899", {"race_result", "race_result_on_time"}, "confirmed"),
                            ("202606050898", {"race_result_on_time"}, "flash"),
                            ("202606050897", set(), "")):
        monkeypatch.setattr(A, "_get_storage", lambda cats=cats: _St(cats))
        r = c.get(f"/api/race/{rid}/result-status")
        assert r.status_code == 200 and A._result_level(rid) == want, (rid, r.json())
    assert A._has_finish_entries(entries) and not A._has_finish_entries({"results": [1]}) and not A._has_finish_entries(None)


def test_repo_config_stale_while_revalidate_is_small():
    """本物の設定: 上限は 60 秒以下（古さの上限が s-maxage＋60 秒に収まる）。"""
    import json
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[2] / "config" / "http_cache.json").read_text(encoding="utf-8"))
    assert 0 < H.stale_while_revalidate_cap(cfg) <= 60


def test_cloudflare_cdn_header_only_on_public():
    """Cloudflare のエッジの期限（Cloudflare-CDN-Cache-Control）は、長く置く public の応答にだけ短く付ける（latency_plan_20261118.md の L6）。
    当日のレース（s-maxage が cdn_max_age 未満）・保守と認証の状態（max-age=0）は no-store（Cloudflare に置かせない）。"""
    c = {"cdn_max_age": 300}
    assert H.cdn_cache_control_value("public, max-age=30, s-maxage=2592000, stale-while-revalidate=60", c) == "public, max-age=300"
    assert H.cdn_cache_control_value("public, max-age=30, s-maxage=300", c) == "public, max-age=300"
    assert H.cdn_cache_control_value("public, max-age=30, s-maxage=30, stale-while-revalidate=30", c) == "no-store"
    assert H.cdn_cache_control_value("public, max-age=30", c) == "no-store"
    assert H.cdn_cache_control_value(H.shared_short_value(300, {"stale_while_revalidate": 60}), c) == "no-store"
    assert H.cdn_cache_control_value(H.shared_short_value(120, {}), c) == "no-store"
    assert H.cdn_cache_control_value("public, max-age=30, s-maxage=2592000", {"cdn_max_age": 0}) == "no-store"
    assert H.cdn_cache_control_value("private, max-age=30", c) is None
    assert H.cdn_cache_control_value("no-store", c) is None
    assert H.cdn_cache_control_value("public, max-age=30, s-maxage=2592000", {"cdn_max_age": "x"}) == "no-store"   # 壊れた設定でも 500 にしない


def test_cloudflare_cdn_header_today_or_later_is_not_stored(monkeypatch):
    """当日・未来と分かるレース・開催日は、s-maxage が長くても Cloudflare に置かせない（当日は cache-warm が nginx を取り直し続ける）。"""
    monkeypatch.setattr(H, "_race_date_peek", lambda rid: {"202606050811": "20261124", "202606050812": "20261101"}.get(rid, ""))
    c = {**SCFG, "cdn_max_age": 300}
    today = datetime(2026, 11, 24, 10, 0, tzinfo=JST)
    long_cc = "public, max-age=30, s-maxage=21600, stale-while-revalidate=60"
    assert H.cdn_cache_control_value(long_cc, c, "/api/race-list/20261124", today) == "no-store"          # 当日の一覧（6 時間）
    assert H.cdn_cache_control_value(long_cc, c, "/api/race-list/20261129", today) == "no-store"          # 未来
    assert H.cdn_cache_control_value(long_cc, c, "/api/race/202606050811/predictions", today) == "no-store"  # 速報の印が付いた当日のレース
    assert H.cdn_cache_control_value(long_cc, c, "/api/race-list/20261123", today) == "public, max-age=300"  # 前日
    assert H.cdn_cache_control_value(long_cc, c, "/api/race/202606050812", today) == "public, max-age=300"
    assert H.cdn_cache_control_value(long_cc, c, "/api/horse/2021101429/detail", today) == "public, max-age=300"
    # 日付が分からない（索引がまだ無い）: 確定済みの 30 日（前の年）でなければ当日かもしれないので置かせない
    assert H.cdn_cache_control_value(long_cc, c, "/api/race/202606059999", today) == "no-store"
    assert H.cdn_cache_control_value("public, max-age=30, s-maxage=2592000", c, "/api/race/202506059999", today) == "public, max-age=300"


def test_cloudflare_cdn_header_keeps_route_value(monkeypatch):
    """ルートがすでに Cloudflare-CDN-Cache-Control を付けていれば、ミドルウェアは変えない。"""
    cfg = {**CFG, "cdn_max_age": 300}
    monkeypatch.setattr(H, "config", lambda: cfg)
    app = FastAPI()

    @app.get("/api/own")
    def own():
        return JSONResponse({"ok": True}, headers={"Cache-Control": "public, max-age=30, s-maxage=600",
                                                   "Cloudflare-CDN-Cache-Control": "public, max-age=60"})

    H.install_fastapi(app)
    assert TestClient(app).get("/api/own").headers["cloudflare-cdn-cache-control"] == "public, max-age=60"


def test_repo_cdn_max_age_is_short():
    import json
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[2] / "config" / "http_cache.json").read_text(encoding="utf-8"))
    assert 0 < int(cfg["cdn_max_age"]) <= 300     # cache-warm の上書きが届かない Cloudflare には短くだけ


def test_cloudflare_cdn_header_fastapi(monkeypatch):
    cfg = {**CFG, "cdn_max_age": 300, "stale_while_revalidate": 60,
           "rules": [{"prefix": "/api/race/", "max_age": 30, "s_maxage": 600}, {"prefix": "/api/live/", "max_age": 30, "s_maxage": 30}]}
    monkeypatch.setattr(H, "config", lambda: cfg)
    monkeypatch.setattr(H, "_race_date_peek", None)
    app = FastAPI()

    @app.get("/api/race/{rid}")
    def race(rid: str):
        return {"id": rid}

    @app.get("/api/live/{rid}")
    def live(rid: str):
        return {"id": rid}

    @app.get("/api/status")
    def status():
        return JSONResponse({"ok": True}, headers={"Cache-Control": H.shared_short_value(300, cfg)})

    H.install_fastapi(app)
    c = TestClient(app)
    assert c.get("/api/race/1").headers["cloudflare-cdn-cache-control"] == "public, max-age=300"
    assert c.get("/api/live/1").headers["cloudflare-cdn-cache-control"] == "no-store"
    assert c.get("/api/status").headers["cloudflare-cdn-cache-control"] == "no-store"   # ルートが自分で付けた shared_short にも付く
    assert "cloudflare-cdn-cache-control" not in c.get("/api/race/1", headers={"Cookie": "a=b"}).headers
    assert "cloudflare-cdn-cache-control" not in c.get("/api/race/1", headers={"Authorization": "Bearer x"}).headers


def test_cloudflare_cdn_header_flask(monkeypatch):
    from flask import Flask, jsonify

    cfg = {**CFG, "cdn_max_age": 300, "rules": [{"prefix": "/api/v1/races/", "max_age": 30, "s_maxage": 600}]}
    monkeypatch.setattr(H, "config", lambda: cfg)
    app = Flask(__name__)

    @app.get("/api/v1/races/<rid>")
    def race(rid):
        return {"id": rid}

    @app.get("/api/v1/auth/status")
    def auth():
        r = jsonify({"logged_in": False})
        r.headers["Cache-Control"] = H.shared_short_value(300, cfg)
        return r

    H.install_flask(app)
    c = app.test_client()
    assert c.get("/api/v1/races/1").headers["Cloudflare-CDN-Cache-Control"] == "public, max-age=300"
    assert c.get("/api/v1/auth/status").headers["Cloudflare-CDN-Cache-Control"] == "no-store"
    c.set_cookie("a", "b")
    assert "Cloudflare-CDN-Cache-Control" not in c.get("/api/v1/races/1").headers
