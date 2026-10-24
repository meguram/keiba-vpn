"""SporthorseData 向けの「疑似セッション」（人手で Cloudflare を通過したブラウザのクッキー）の保存と検証。

サイトは curl だけでは 403 になるため、**ご自身のブラウザで一度ページを開いたあと**のクッキーを
``data/meta/dormello/sporthorse_session.json``（git 管理外）に保存し、研究用 CLI が再利用する。

    # ユーザースクリプトがローカル入力ツールへ送った JSON を保存（手動でも可）
    python -m src.research.pedigree.sporthorse_session import --in cookies.json

    # セッションがまだ有効か確認（実アクセスあり）
    python -m src.research.pedigree.sporthorse_session probe

環境変数 ``SPORTHORSE_SESSION`` で保存先パスを上書きできる。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SESSION_PATH = Path("data/meta/dormello/sporthorse_session.json")
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
PROBE_URL = "https://sporthorse-data.com/pedigree/kris"
_CHALLENGE = re.compile(r"(?i)(just a moment|cf-browser-verification|challenge-platform)")


def session_path(explicit: Path | None = None) -> Path:
    if explicit:
        return explicit
    env = os.environ.get("SPORTHORSE_SESSION", "").strip()
    return Path(env) if env else DEFAULT_SESSION_PATH


@dataclass
class SessionState:
    user_agent: str
    cookies: dict[str, str]
    saved_at: str | None = None

    def cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items() if k and v)

    def to_dict(self) -> dict:
        return {"user_agent": self.user_agent, "cookies": self.cookies, "saved_at": self.saved_at}

    @classmethod
    def from_dict(cls, data: dict) -> SessionState:
        cookies = data.get("cookies") or {}
        if not isinstance(cookies, dict):
            raise ValueError("cookies はオブジェクトである必要があります")
        ua = (data.get("user_agent") or DEFAULT_UA).strip()
        return cls(user_agent=ua, cookies={str(k): str(v) for k, v in cookies.items()}, saved_at=data.get("saved_at"))


def load_session(path: Path | None = None) -> SessionState:
    p = session_path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"セッションファイルがありません: {p}\n"
            "ブラウザで https://sporthorse-data.com/pedigree/ を開き、"
            "ユーザースクリプトまたは manual_entry_server の「セッションを保存」で登録してください。"
        )
    return SessionState.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_session(state: SessionState, path: Path | None = None) -> Path:
    p = session_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not state.saved_at:
        state.saved_at = datetime.now(timezone.utc).isoformat()
    p.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


def normalize_cookie_import(data: object) -> dict[str, str]:
    """Chrome / Tampermonkey / 手書き JSON から name->value を抽出する。"""
    if isinstance(data, dict) and "cookies" in data:
        raw = data["cookies"]
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        if isinstance(raw, list):
            return _cookies_from_list(raw)
    if isinstance(data, list):
        return _cookies_from_list(data)
    raise ValueError("cookies の形式が読めません（dict.cookies または Cookie オブジェクトの配列）")


def _cookies_from_list(items: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in items:
        if not isinstance(c, dict):
            continue
        dom = str(c.get("domain") or "")
        if dom and "sporthorse" not in dom:
            continue
        name, val = c.get("name"), c.get("value")
        if name and val is not None:
            out[str(name)] = str(val)
    return out


def import_session(payload: object, user_agent: str | None = None, path: Path | None = None) -> Path:
    cookies = normalize_cookie_import(payload)
    if not cookies:
        raise ValueError("sporthorse-data.com 向けのクッキーが 1 件もありません")
    state = SessionState(user_agent=(user_agent or DEFAULT_UA).strip(), cookies=cookies)
    return save_session(state, path)


def is_challenge(html: str, headers: dict[str, str]) -> bool:
    if headers.get("cf-mitigated", "").lower() == "challenge":
        return True
    return bool(_CHALLENGE.search(html[:8000]))


def fetch_html(url: str, state: SessionState, timeout: float = 30.0) -> tuple[int, str, dict[str, str]]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": state.user_agent,
            "Cookie": state.cookie_header(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://sporthorse-data.com/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, body, hdrs
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        hdrs = {k.lower(): v for k, v in e.headers.items()}
        return e.code, body, hdrs


def probe(state: SessionState | None = None, path: Path | None = None) -> int:
    st = state or load_session(path)
    code, html, hdrs = fetch_html(PROBE_URL, st)
    if code != 200 or is_challenge(html, hdrs):
        print(f"NG: HTTP {code}（Cloudflare チャレンジの可能性）。ブラウザで再度開いてセッションを更新してください。", flush=True)
        return 2
    if not re.search(r"(?i)date of birth|born", html):
        print("NG: 200 だが誕生日らしき文言がありません（別ページかセッション不一致）", flush=True)
        return 2
    print(f"OK: セッション有効（{PROBE_URL}、クッキー {len(st.cookies)} 件）", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="JSON からセッションを保存")
    imp.add_argument("--in", dest="inp", required=True, type=Path)
    imp.add_argument("--session", type=Path)
    imp.add_argument("--user-agent", default="")
    sub.add_parser("probe", help="保存済みセッションを実ページで検証")
    ap.add_argument("--session", type=Path, help="セッションファイル（既定: data/meta/dormello/sporthorse_session.json）")
    a = ap.parse_args(argv)
    sess_arg = getattr(a, "session", None)
    if a.cmd == "import":
        data = json.loads(a.inp.read_text(encoding="utf-8"))
        p = import_session(data, user_agent=a.user_agent or None, path=sess_arg)
        print(f"保存: {p}（クッキー {len(load_session(p).cookies)} 件）", flush=True)
        return probe(load_session(p), p)
    return probe(path=sess_arg)


if __name__ == "__main__":
    sys.exit(main())
