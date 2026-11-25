"""AdSense の審査の状態を読むための、一度だけの同意（OAuth・読むだけの権限）。

    python -m src.ops.adsense_oauth_setup --client-file <OAuth クライアントの JSON> --out <保存先 .json>

- Google Cloud で作った「デスクトップ アプリ」の OAuth クライアント（client_secret_*.json）を読む。
- 画面に出す URL をブラウザで開き、AdSense のアカウントで同意する（権限は adsense.readonly だけ）。
- 同意の後、手元の 127.0.0.1 で受けた認可コードを更新トークンに替え、
  client_id・client_secret・refresh_token を --out に書く（権限 0600）。値は画面に出さない。
- ブラウザが別の PC にあるときは、同意の後に開けなかった URL（http://127.0.0.1:...）を丸ごと貼る。
"""
from __future__ import annotations

import argparse
import hashlib
import base64
import http.server
import json
import os
import secrets
import sys
import threading
import urllib.parse
import urllib.request

SCOPE = "https://www.googleapis.com/auth/adsense.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def _client(path: str) -> tuple[str, str]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    c = d.get("installed")
    if not c:
        # web 型は 127.0.0.1 の任意の番号へ戻せず、同意の後に redirect_uri_mismatch になる
        raise SystemExit("デスクトップ アプリ型の OAuth クライアント（JSON の installed）を使ってください")
    return c["client_id"], c["client_secret"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--client-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=0, help="受け口の番号（0 は空いている番号）")
    args = ap.parse_args(argv)

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_abs = os.path.abspath(os.path.expanduser(args.out))
    if os.path.commonpath([repo, out_abs]) == repo:
        print("--out はリポジトリの外に置いてください（例: ~/.config/keiba/adsense_oauth.json）。commit の事故を防ぐため。", file=sys.stderr)
        return 1
    args.out = out_abs
    cid, csecret = _client(args.client_file)
    state = secrets.token_urlsafe(16)
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    got: dict[str, str] = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
            got.update(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("受け取りました。この画面は閉じてかまいません。".encode())

        def log_message(self, *a):  # 認可コードを記録に出さない
            pass

    srv = http.server.HTTPServer(("127.0.0.1", args.port), H)
    redirect = f"http://127.0.0.1:{srv.server_port}/"
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": redirect, "response_type": "code", "scope": SCOPE,
        "access_type": "offline", "prompt": "consent", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })
    print("次の URL をブラウザで開き、AdSense のアカウントで同意してください（読むだけの権限）:\n")
    print(url + "\n")
    print("同じ PC のブラウザなら自動で受け取ります。別の PC なら、同意の後に開けなかった URL を貼ってください。")
    t = threading.Thread(target=srv.handle_request, daemon=True)
    t.start()
    t.join(timeout=1)
    while t.is_alive() and not got:
        try:
            pasted = input("URL（自動で受け取ったら Enter）: ").strip()
        except EOFError:
            pasted = ""
        if pasted:
            got.update(dict(urllib.parse.parse_qsl(urllib.parse.urlparse(pasted).query)))
            break
        t.join(timeout=1)
    srv.server_close()
    if got.get("state") != state or not got.get("code"):
        print("受け取れなかった（state の不一致か、コードが無い）。やり直してください。", file=sys.stderr)
        return 1
    data = urllib.parse.urlencode({
        "code": got["code"], "client_id": cid, "client_secret": csecret,
        "redirect_uri": redirect, "grant_type": "authorization_code", "code_verifier": verifier,
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=20) as r:
        tok = json.loads(r.read().decode("utf-8"))
    if not tok.get("refresh_token"):
        print("更新トークンが返らなかった。Google アカウントの「サードパーティのアクセス」で取り消してからやり直す。", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(args.out), mode=0o700, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)  # 既にあるファイルは O_CREAT の権限が効かないので、書く前に絞る
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"client_id": cid, "client_secret": csecret, "refresh_token": tok["refresh_token"]}, f)
    print(f"保存しました: {args.out}（権限 0600。値は表示しません）")
    print(f"確かめ: ADSENSE_OAUTH_FILE={args.out} python -m src.ops.adsense_check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
