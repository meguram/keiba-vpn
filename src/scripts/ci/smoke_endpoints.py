"""ユーザーが実際に触る画面・API が「サーバに繋がり、dev フィクスチャのデータで応答する」ことの最小チェック。

起動済みの Next.js（フロント。/api は Flask / FastAPI へプロキシ）に対して、最小 UI（週間予測 → レース詳細 →
推奨馬券）が使うページと API を叩く。確認するのは **アクセス可否と応答の中身が空でないこと** までで、
stg/prod の実データが正しいかは保証しない（CI は dev フィクスチャ = dev_fixture/ の 1 開催日 + モックで動かす）。

  python -m src.scripts.ci.smoke_endpoints --base http://127.0.0.1:23100 [--wait 180] [--require-db]

失敗は GitHub Actions の annotation（::error）にも出す。終了コード: 0=全部 OK / 1=失敗あり / 2=サーバ未起動。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]


def _req(url: str, *, method: str = "GET", body: dict | None = None, timeout: float = 60.0) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", "Accept": "*/*"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return 0, str(e).encode()


def wait_until_up(base: str, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if _req(base + "/api/v1/health", timeout=5)[0] == 200 and _req(base + "/api/health", timeout=5)[0] == 200:
            return True
        time.sleep(2)
    return False


def fixture_info() -> tuple[str, int]:
    man = json.loads((REPO_ROOT / "dev_fixture" / "MANIFEST.json").read_text(encoding="utf-8"))
    return str(man["date"]), int(man["races"])


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def check(name: str, fn: Callable[[], str]) -> Result:
    try:
        return Result(name, True, fn())
    except AssertionError as e:
        return Result(name, False, str(e) or "assertion failed")
    except Exception as e:  # noqa: BLE001 — 例外も失敗として記録して続行する
        return Result(name, False, f"{type(e).__name__}: {e}")


def get_json(base: str, path: str, *, method: str = "GET", body: dict | None = None, timeout: float = 90.0) -> Any:
    code, raw = _req(base + path, method=method, body=body, timeout=timeout)
    assert code == 200, f"HTTP {code} {raw[:200].decode('utf-8', 'replace')}"
    try:
        return json.loads(raw)
    except ValueError as e:
        raise AssertionError(f"JSON ではない応答: {raw[:120]!r}") from e


def run_checks(base: str, *, require_db: bool = False) -> list[Result]:
    date, n_races = fixture_info()
    out: list[Result] = []
    state: dict[str, Any] = {}

    # ── ページ（ユーザーが開く URL。HTML が返り 5xx でないこと） ──
    def page(path: str) -> Callable[[], str]:
        def f() -> str:
            code, raw = _req(base + path, timeout=90)
            assert code == 200, f"HTTP {code}"
            assert b"<html" in raw[:2000].lower() or b"<!doctype" in raw[:200].lower(), "HTML ではない"
            return "200"
        return f

    race_page_path = "/race/{race_id}"  # race_id は後で確定
    for p in ("/", "/login", "/weekly-predictions", "/races", "/betting"):
        out.append(check(f"page {p}", page(p)))

    # ── ヘルスと認証 ──
    out.append(check("api /api/v1/health (Flask)", lambda: str(get_json(base, "/api/v1/health").get("status", "ok"))))
    out.append(check("api /api/health (FastAPI)", lambda: str(get_json(base, "/api/health").get("status", "ok"))))

    def auth() -> str:
        d = get_json(base, "/api/v1/auth/status")
        assert d.get("logged_in") is True, f"dev 自動ログインになっていない: {d}"
        return "logged_in"
    out.append(check("api /api/v1/auth/status（dev 自動ログイン）", auth))

    # ── 週間予測 → レース詳細 ──
    def scrape_dates() -> str:
        d = get_json(base, "/api/scrape-dates")
        dates = [x if isinstance(x, str) else x.get("date") for x in d.get("dates", [])]
        assert date in dates, f"フィクスチャ日 {date} が開催日一覧に無い: {dates}"
        return f"{len(dates)} 日"
    out.append(check("api /api/scrape-dates", scrape_dates))

    def race_list() -> str:
        d = get_json(base, f"/api/race-list/{date}")
        races = d.get("races", [])
        assert len(races) == n_races, f"レース数 {len(races)} != {n_races}"
        state["race_id"] = races[0]["race_id"] if races else None
        assert state["race_id"], "race_id が取れない"
        return f"{len(races)}R"
    out.append(check(f"api /api/race-list/{date}", race_list))

    def race_values() -> str:
        d = get_json(base, f"/api/v1/betting/race-values?date={date}")
        assert len(d.get("races", [])) == n_races, f"価値表示のレース数 {len(d.get('races', []))} != {n_races}"
        return f"{len(d['races'])}R"
    out.append(check("api /api/v1/betting/race-values", race_values))

    rid = state.get("race_id") or "202601020611"
    out.append(check(f"page {race_page_path.format(race_id=rid)}", page(race_page_path.format(race_id=rid))))

    def race_detail() -> str:
        # フィクスチャ(dev_fixture/)が持つカテゴリだけを見る: 出馬表・結果・単複オッズ・2連系/3連複オッズ
        d = get_json(base, f"/api/race/{rid}")
        for cat in ("race_shutuba", "race_result", "race_odds"):
            assert len((d.get(cat) or {}).get("entries", [])) > 0, f"{cat} が空"
        assert (d.get("race_pair_odds") or {}).get("umaren"), "race_pair_odds（馬連）が空"
        assert d["race_shutuba"].get("race_name"), "出馬表のレース名が空"
        return d["race_shutuba"]["race_name"]
    out.append(check("api /api/race/{id}", race_detail))

    def predictions() -> str:
        d = get_json(base, f"/api/race/{rid}/predictions")
        assert d.get("has_prediction") is True, "予測が無い"
        assert len(d.get("predictions", [])) > 0, "予測の馬リストが空"
        return f"{len(d['predictions'])}頭"
    out.append(check("api /api/race/{id}/predictions", predictions))

    # ── 推奨馬券（期待値 → 買い目） ──
    def optimize() -> str:
        d = get_json(base, "/api/v1/betting/optimize", method="POST", body={
            "race_id": rid, "bankroll": 10000, "max_total_bet": 2000, "odds_mode": "auto",
            "bet_types": ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"],
        })
        assert len(d.get("candidates", [])) > 0, f"買い目が空（odds_note={d.get('odds_note')}）"
        assert 0 < d.get("total_bet", 0) <= 2000, f"合計金額が上限の範囲外: {d.get('total_bet')}"
        return f"{len(d['candidates'])}点 / {d['total_bet']}円"
    out.append(check("api POST /api/v1/betting/optimize", optimize))

    # ── DB 依存（レース一覧のフォールバック元。DB 無しの dev では 503 を許容） ──
    def races_db() -> str:
        code, raw = _req(base + "/api/v1/races", timeout=60)
        if require_db:
            assert code == 200, f"HTTP {code} {raw[:160].decode('utf-8', 'replace')}"
        else:
            assert code in (200, 503), f"HTTP {code}"
        return str(code)
    out.append(check("api /api/v1/races（DB）", races_db))

    # ── 無効化した機能は 5xx にならず、案内（リダイレクト）で返ること ──
    out.append(check("page /megu-index（無効機能は 5xx にしない）", lambda: str(_ok_lt500(base + "/megu-index"))))
    return out


def _ok_lt500(url: str) -> int:
    code, _ = _req(url, timeout=60)
    assert 0 < code < 500, f"HTTP {code}"
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:23100", help="Next.js の URL（/api はここからプロキシされる）")
    ap.add_argument("--wait", type=float, default=0, help="サーバが応答するまで最大この秒数待つ")
    ap.add_argument("--require-db", action="store_true", help="/api/v1/races（Postgres）も 200 を必須にする")
    args = ap.parse_args(argv)

    if not wait_until_up(args.base, args.wait):
        print(f"::error title=smoke::サーバに接続できません: {args.base}（Flask /api/v1/health・FastAPI /api/health が {args.wait:.0f} 秒以内に 200 を返さない）")
        return 2

    results = run_checks(args.base, require_db=args.require_db)
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{'OK  ' if r.ok else 'FAIL'} {r.name:<{width}}  {r.detail}")
    failed = [r for r in results if not r.ok]
    for r in failed:
        print(f"::error title=smoke failed::{r.name}: {r.detail[:300]}")
    print(f"\n{len(results) - len(failed)}/{len(results)} OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
