"""共通ログイン（特定の人に配る 1 つの ID）のパスワードを、毎月 1 日に自動で更新し、新しい情報を Slack に通知する。

  python -m src.scripts.ops.shared_login ensure [--no-notify]   # 毎日 00:05 JST に cron から。月が替わっていれば更新（冪等）
  python -m src.scripts.ops.shared_login show                    # 今月のログイン情報を表示（管理者が配布するとき）

仕組み: 月ごとのパスワードは サーバの秘密（ペッパー）+ 年月 から決まる値（保存しない）。ID は .env の KEIBA_SHARED_LOGIN_ID（未設定なら何もしない）。
更新するとパスワード変更扱いになり、それ以前のログイン（cookie）はすべて使えなくなる。Slack（SLACK_WEBHOOK_URL）が送れなかった場合は、次の実行で再送する。
``KEIBA_SHARED_LOGIN_DRY=1``（GCP Jobs の並走期間）: 名簿と状態を読んで『何をするか』（created/rotated/notified/ok）を出すだけで、名簿・状態の書き込みと通知はしない。
名簿の正本がホストにある間、Jobs のローカルの名簿を書き換えても公開側（Cloud Run）に届かないため。
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST = timezone(timedelta(hours=9))
ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"      # 紛らわしい文字（0 O 1 l i）を除く


def month_of(now: datetime | None = None) -> str:
    return (now or datetime.now(JST)).astimezone(JST).strftime("%Y-%m")


def password_for(month: str) -> str:
    from src.api import auth

    d = hmac.new(auth._pepper(), b"shared-login:" + month.encode(), hashlib.sha256).digest()
    chars = "".join(ALPHABET[b % len(ALPHABET)] for b in d[:16])
    return "-".join(chars[i:i + 4] for i in range(0, 16, 4))


def _state_path() -> Path:
    from src.api import auth

    return Path(auth.MEMBERS_LOCAL_FILE).with_name("shared_login_state.json")


def _state() -> dict:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_state(d: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def message(user_id: str, month: str, password: str) -> str:
    y, m = month.split("-")
    return (f"【めぐ競馬】共通ログイン情報を更新しました（{int(y)}年{int(m)}月分）\nログインID: {user_id}\nパスワード: {password}\n"
            "有効: 月末まで（翌月1日に自動で更新され、これまでのログインは解除されます）\n※ このメッセージは配布後に削除してください")


def ensure(user_id: str, now: datetime | None = None, notify: bool = True, dry: bool = False) -> str:
    """戻り値: created / rotated / notified（通知だけ再送）/ ok（変更なし）/ disabled（無効にされているので触らない）。
    dry=True は判定だけ（名簿・状態を書かず、通知しない）。戻り値は同じ語に ``would-`` を付ける（ok と disabled はそのまま）。"""
    from src.api import auth
    from src.scripts.ops import manage_users as mu
    from src.utils.notify import notify_slack

    month = month_of(now)
    pw = password_for(month)
    rec = auth.member_record(user_id)
    result = "ok"
    if rec is None:
        if not dry:
            mu.create_user(user_id, pw)
        result = "created"
    elif rec.get("enabled", True) is False:
        return "disabled"
    elif not auth.verify_member_login(user_id, pw):
        if not dry:
            mu.set_password(user_id, pw)
        result = "rotated"
    st = _state()
    if result in ("created", "rotated"):
        st = {"month": month, "notified": False, "changed_at": int(time.time())}
    elif st.get("month") != month:
        st = {"month": month, "notified": False, "changed_at": int(time.time())}      # 状態ファイルが無い・古い（初回）→ 今月分を通知
    if dry:
        if result == "ok" and not st.get("notified"):
            result = "notified"
        return result if result == "ok" else f"would-{result}"
    if notify and not st.get("notified"):
        if notify_slack(message(user_id, month, pw)):
            st["notified"] = True
            result = result if result != "ok" else "notified"
        else:
            print("Slack に送れませんでした（SLACK_WEBHOOK_URL 未設定か送信失敗）。次の実行で再送します。確認: python -m src.scripts.ops.shared_login show", file=sys.stderr)
    _write_state(st)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["ensure", "show"])
    ap.add_argument("--no-notify", action="store_true")
    a = ap.parse_args(argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()      # cron からでも .env の秘密（DEV_SECRET_KEY・SLACK_WEBHOOK_URL・KEIBA_SHARED_LOGIN_ID）を読む
    uid = (os.environ.get("KEIBA_SHARED_LOGIN_ID") or "").strip()
    if not uid:
        print("KEIBA_SHARED_LOGIN_ID が未設定のため何もしません")
        return 0
    if a.cmd == "show":
        m = month_of()
        print(f"{m}\nログインID: {uid}\nパスワード: {password_for(m)}")
        return 0
    from src.scripts.ops import manage_users as mu

    try:
        print(ensure(uid, notify=not a.no_notify, dry=os.environ.get("KEIBA_SHARED_LOGIN_DRY", "").strip() == "1"))
    except mu.UserError as e:
        print(f"失敗: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
