"""AdSense の確かめ（外から見える所と、審査の状態）。1 画面の結果を出す。

    python -m src.ops.adsense_check            # 外から見える所 ＋ 審査の状態（トークンが無ければ「未設定」）
    python -m src.ops.adsense_check --no-api   # 外から見える所だけ

- 外から見える所（費用 0）: ルートと app・stg の ads.txt（転送の先・中身）、robots.txt が広告の巡回を許すか、
  匿名で開いたトップ・レースの画面の HTML に所有の確認の meta があるか。
  スクリプト（adsbygoogle.js）は画面の側で区分が分かった後に読み込むので、HTML には出ない。
  有料会員・開発者に読み込まないことは、会員の資格を使わずに lib/adsense.test.mjs で確かめる。
- 審査の状態: AdSense Management API v2 の accounts.sites.list（読むだけの権限 adsense.readonly）。
  資格は環境変数 ADSENSE_OAUTH_CLIENT_ID・ADSENSE_OAUTH_CLIENT_SECRET・ADSENSE_OAUTH_REFRESH_TOKEN、
  または ADSENSE_OAUTH_FILE（その 3 つを client_id・client_secret・refresh_token に持つ JSON）。
  値・トークンは画面にもログにも出さない。手順は docs/operations/adsense_plan_20261125.md。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser

PUB_ID = "pub-5321507754924902"
ADS_LINE = f"google.com, {PUB_ID}, DIRECT, f08c47fec0942fa0"
APP = "https://app.megu-keiba.com"
STG = "https://stg.megu-keiba.com"
ROOT = "https://megu-keiba.com"
PAGES = ["/", "/races", "/learn", "/learn/guide", "/learn/glossary", "/learn/column/even-gate"]
LEARN_MIN_CHARS = 300
ADS_BOTS = ["Mediapartners-Google", "Google-Display-Ads-Bot"]
UA = "keiba-adsense-check/1"
TIMEOUT = 15

TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://adsense.googleapis.com/v2"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _get(url: str, follow: bool = True) -> tuple[int, str, str, list[str]]:
    """(状態, 最後の URL, 本文, 転送の列)。通信の失敗は状態 0。"""
    chain: list[str] = []
    cur = url
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(6):
        req = urllib.request.Request(cur, headers={"User-Agent": UA})
        try:
            with opener.open(req, timeout=TIMEOUT) as r:
                return r.status, cur, r.read(400_000).decode("utf-8", "replace"), chain
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location") if e.code in (301, 302, 303, 307, 308) else None
            nxt = urllib.parse.urljoin(cur, loc) if loc else ""
            short = urllib.parse.urlsplit(nxt)._replace(query="", fragment="").geturl() if nxt else ""
            if loc and follow:
                chain.append(f"{e.code} → {short}")  # 転送先のクエリ（Access の署名など）は出さない
                cur = nxt
                continue
            body = e.read(4000).decode("utf-8", "replace") if e.fp else ""
            if loc:
                chain.append(f"{e.code} → {short}")
            return e.code, cur, body, chain
        except Exception as e:  # 名前が引けない・接続できない
            return 0, cur, f"{type(e).__name__}: {e}", chain
    return 0, cur, "転送が多すぎる", chain


def check_ads_txt(base: str) -> tuple[bool, str]:
    code, last, body, chain = _get(base + "/ads.txt")
    via = f"（{' / '.join(chain)}）" if chain else ""
    if code == 0:
        return False, f"届かない {body[:80]}"
    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
    ok = code == 200 and ADS_LINE in lines
    return ok, f"HTTP {code}{via} {'行あり' if ADS_LINE in lines else '行なし'}"


def check_robots(base: str) -> tuple[bool, str]:
    code, _, body, _ = _get(base + "/robots.txt")
    if code != 200:
        return False, f"HTTP {code}"
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(body.splitlines())
    res = {b: rp.can_fetch(b, base + "/") for b in ADS_BOTS}
    return all(res.values()), " ".join(f"{b}={'可' if v else '不可'}" for b, v in res.items())


def check_page(base: str, path: str) -> tuple[bool, str]:
    code, _, body, chain = _get(base + path)
    if code != 200:
        return False, f"HTTP {code}{' ' + chain[-1] if chain else ''}"
    meta = re.search(r'<meta[^>]+name="google-adsense-account"[^>]+content="ca-' + PUB_ID + '"', body) is not None
    n = text_chars(body)
    # 文章のページ（/learn）は本文が HTML に入っていること（サーバで描く）。クライアントで描く形に戻ったら落とす
    thin = path.startswith("/learn") and n < LEARN_MIN_CHARS
    return meta and not thin, f"meta {'あり' if meta else 'なし'}・本文 {n:,} 字{'（少なすぎ）' if thin else ''}"


def text_chars(html: str) -> int:
    """HTML の本文の文字数（script・style・タグ・空白を除く）。審査の巡回が読める中身の目安"""
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", html, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return len(re.sub(r"\s+", "", t))


class CredsError(Exception):
    """資格の置き方の誤り（未設定とは分けて出す）。"""


def _oauth_creds() -> dict[str, str] | None:
    path = os.environ.get("ADSENSE_OAUTH_FILE", "").strip()
    if path:
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                d = json.load(f)
        except OSError as e:
            raise CredsError(f"ADSENSE_OAUTH_FILE が読めない（{type(e).__name__}）") from None
        except ValueError:
            raise CredsError("ADSENSE_OAUTH_FILE が JSON でない") from None
        c = {k: str(d.get(k, "")) for k in ("client_id", "client_secret", "refresh_token")}
        if not all(c.values()):
            raise CredsError("ADSENSE_OAUTH_FILE に client_id・client_secret・refresh_token のどれかが無い")
        return c
    c = {
        "client_id": os.environ.get("ADSENSE_OAUTH_CLIENT_ID", ""),
        "client_secret": os.environ.get("ADSENSE_OAUTH_CLIENT_SECRET", ""),
        "refresh_token": os.environ.get("ADSENSE_OAUTH_REFRESH_TOKEN", ""),
    }
    return c if all(c.values()) else None


def _api_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def review_state() -> list[str]:
    """審査の状態の行。資格・トークンの値は返さない。"""
    try:
        creds = _oauth_creds()
    except CredsError as e:
        return [f"審査の状態: 設定の誤り（{e}）"]
    if creds is None:
        return ["審査の状態: 未設定（ADSENSE_OAUTH_* か ADSENSE_OAUTH_FILE が無い。手順は docs/operations/adsense_plan_20261125.md §7）"]
    data = urllib.parse.urlencode({**creds, "grant_type": "refresh_token"}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=TIMEOUT) as r:
            token = json.loads(r.read().decode("utf-8")).get("access_token", "")
    except urllib.error.HTTPError as e:
        return [f"審査の状態: トークンの更新に失敗（HTTP {e.code}。同意の取り消し・期限切れなら同意をやり直す）"]
    except Exception as e:
        return [f"審査の状態: トークンの更新に失敗（{type(e).__name__}）"]
    if not token:
        return ["審査の状態: トークンの更新に失敗（応答に access_token が無い）"]
    out: list[str] = []
    try:
        accounts = _api_json(f"{API}/accounts", token).get("accounts", [])
        if not accounts:
            return ["審査の状態: アカウントが見えない（同意したアカウントを確かめる）"]
        for a in accounts:
            out.append(f"アカウント {a.get('name', '?')}: {a.get('state', '?')}")
            sites = _api_json(f"{API}/{a['name']}/sites?pageSize=100", token).get("sites", [])
            if not sites:
                out.append("  サイト: 登録なし")
            for s in sites:
                out.append(f"  サイト {s.get('domain', '?')}: {s.get('state', '?')}"
                           f"{'（自動広告 on）' if s.get('autoAdsEnabled') else ''}")
    except urllib.error.HTTPError as e:
        out.append(f"審査の状態: API の読み出しに失敗（HTTP {e.code}）")
    except Exception as e:
        out.append(f"審査の状態: API の読み出しに失敗（{type(e).__name__}）")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-api", action="store_true", help="審査の状態（API）を読まない")
    args = ap.parse_args(argv)

    rows: list[tuple[str, bool | None, str]] = []
    for name, base in (("ルート", ROOT), ("本番", APP), ("stg", STG)):
        ok, msg = check_ads_txt(base)
        rows.append((f"ads.txt {name}", ok, msg))
    ok, msg = check_robots(APP)
    rows.append(("robots.txt 本番", ok, msg))
    for p in PAGES:
        ok, msg = check_page(APP, p)
        rows.append((f"HTML 本番 {p}（匿名）", ok, msg))

    print("AdSense の確かめ")
    for label, ok, msg in rows:
        mark = "OK " if ok else "NG "
        if label.endswith("stg") and not ok:
            mark = "-- "  # stg は Access の内側。巡回は届かず、審査にも使わない
        print(f"  {mark}{label}: {msg}")
    print("  注: スクリプトは画面の側で匿名・無料会員と分かった後に読む（HTML には出ない）。"
          "会員に読まないことは frontend/lib/adsense.test.mjs で確かめる。")
    if not args.no_api:
        for ln in review_state():
            print("  " + ln)
    # ルートと本番の ads.txt・robots・meta が揃えば 0
    must = [r for r in rows if not r[0].endswith("stg")]
    return 0 if all(r[1] for r in must) else 1


if __name__ == "__main__":
    sys.exit(main())
