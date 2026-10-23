"""会員ユーザー（ログイン ID とパスワード）の管理。ID もパスワードも平文では保存せず、管理者本人でも元の値は読み出せない。

保存するもの: ID の HMAC（ペッパーつき。元の ID には戻せない）、パスワードの PBKDF2-SHA256＋ペッパーの HMAC、有効/無効、登録日時。表示名は持たない（ログイン中の表示は、本人が入力した ID）。
一覧には「指紋」（ID の HMAC の先頭 8 文字）だけが出る。操作の対象は ID（入力すると照合する）か指紋で指定する。

  python -m src.scripts.ops.manage_users list
  python -m src.scripts.ops.manage_users add <id> [--password 初期パスワード | --generate | --password-stdin]    # 既定は対話入力（2 回）
  python -m src.scripts.ops.manage_users passwd <id|指紋> [--password 新パスワード | --generate | --password-stdin]
  python -m src.scripts.ops.manage_users disable <id|指紋> | enable <id|指紋> | remove <id|指紋>
  （--password はシェルの履歴に残る。残したくないときは対話入力か --password-stdin を使う）

保存先: data/local/auth/members.json（git に入れない。環境変数 KEIBA_MEMBERS_FILE で変更）。config/members.json（動作確認用の test）は読むだけ。
API は会員ファイルの更新を自動で読み直す（再起動不要）。無効化・削除したユーザーは、ログイン中でも次のリクエストから使えなくなる。
本番ではホストで実行する: ssh keiba-remote 'cd /home/jovyan/work/keiba-vpn && sudo -n /opt/conda/bin/python3 -m src.scripts.ops.manage_users list'
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

ITERATIONS = 600_000
ID_RE = re.compile(r"^[A-Za-z0-9._@-]{3,64}$")
MIN_PW = 10
ALGO = "pbkdf2_sha256_pepper"


class UserError(ValueError):
    """入力の誤り（画面・CLI にそのまま出してよい文言）。"""


def _path() -> Path:
    from src.api import auth

    return Path(auth.MEMBERS_LOCAL_FILE).resolve()


def _load(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("users"), list):
            return d
    except (OSError, ValueError):
        pass
    return {"_comment": "ユーザー管理（manage_users）が書く会員アカウント。ID・パスワードは一方向のハッシュだけ（元に戻せない）。git に入れない。", "users": []}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


@contextlib.contextmanager
def locked(path: Path | None = None):
    """会員ファイルの読み書きを、CLI・管理 API・決済の Webhook が同時に行っても壊れないように、排他ロックで囲む。"""
    p = (path or _path()).with_suffix(".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.gmtime(time.time() + 9 * 3600))


def check_password_rules(user_id: str, password: str) -> None:
    if len(password or "") < MIN_PW:
        raise UserError(f"パスワードは {MIN_PW} 文字以上にしてください")
    if (user_id and password.lower() == user_id.strip().lower()):
        raise UserError("パスワードに ID と同じ文字列は使えません")


def _hash(password: str) -> dict:
    from src.api import auth

    salt = secrets.token_bytes(16)
    return {"algo": ALGO, "iterations": ITERATIONS, "salt": salt.hex(), "hash": auth.password_digest(password, salt, ITERATIONS)}


def _resolve(data: dict, ident: str) -> dict:
    """ID（平文）または指紋（8 文字の 16 進）で対象を決める。見つからなければ UserError。"""
    from src.api import auth

    ident = (ident or "").strip()
    k = auth.id_key(ident) if ident else ""
    hit = [u for u in data["users"] if u.get("idk") == k]
    if not hit and re.fullmatch(r"[0-9a-f]{8}", ident):
        hit = [u for u in data["users"] if str(u.get("idk", "")).startswith(ident)]
    if len(hit) != 1:
        raise UserError("対象のユーザーが見つかりません（config/members.json の test は、ここからは変更できません）")
    return hit[0]


def list_users() -> list[dict]:
    """一覧（ID は出ない）。fp=指紋、source=config（動作確認用）/ managed（この管理で作成）。"""
    from src.api import auth

    local = _load(_path())["users"]
    out = []
    for u in auth._read_users(auth._MEMBERS_FILE):
        out.append({"fp": "-", "source": "config", "label": u.get("id"), "enabled": bool(not u.get("requires_env") or auth._env_on(str(u["requires_env"]))), "created_at": None, "note": f"{u.get('requires_env')} が有効なときだけ"})
    for u in local:
        out.append({"fp": str(u["idk"])[:8], "source": "managed", "label": None, "enabled": u.get("enabled", True) is not False, "created_at": u.get("created_at"), "note": ""})
    return out


def create_user(user_id: str, password: str, billing: dict | None = None) -> str:
    """ユーザーを作る。戻り値は指紋。billing を渡すと課金の状態（pending など）つきで作る（決済が済むまでログインできない）。"""
    from src.api import auth

    uid = (user_id or "").strip()
    if not ID_RE.match(uid):
        raise UserError("ログインID は英数字と . _ @ - の 3〜64 文字にしてください")
    check_password_rules(uid, password)
    p = _path()
    with locked(p):
        data = _load(p)
        k = auth.id_key(uid)
        if any(u.get("idk") == k for u in data["users"]) or any((u.get("id") or "").lower() == uid.lower() for u in auth._read_users(auth._MEMBERS_FILE)):
            raise UserError("そのログインID はすでに使われています")
        rec = {"idk": k, **_hash(password), "role": "member", "enabled": True, "created_at": _now(), "updated_at": _now()}
        if billing is not None:
            rec["billing"] = dict(billing)
        data["users"].append(rec)
        _save(p, data)
    return k[:8]


# ── 決済（Stripe）連携用 ──
def update_billing(*, idk: str | None = None, customer: str | None = None, **fields) -> str:
    """idk か Stripe の顧客 ID で対象を決め、billing の項目（status・customer・subscription・period_end など）を更新する。見つからなければ UserError。"""
    p = _path()
    with locked(p):
        data = _load(p)
        hit = [u for u in data["users"] if (idk and u.get("idk") == idk) or (customer and (u.get("billing") or {}).get("customer") == customer)]
        if len(hit) != 1:
            raise UserError("対象のユーザーが見つかりません")
        u = hit[0]
        b = u.setdefault("billing", {})
        b.update({k: v for k, v in fields.items() if v is not None})
        if customer:
            b["customer"] = customer
        u["updated_at"] = _now()
        _save(p, data)
        return str(u["idk"])[:8]


def remove_pending(max_age_hours: float = 24.0, now: float | None = None) -> int:
    """決済が済まないまま残っている仮登録（billing.status=pending）を、一定時間後に消す。消した件数を返す。"""
    limit = (now if now is not None else time.time()) - max_age_hours * 3600
    p = _path()
    with locked(p):
        data = _load(p)
        keep = [u for u in data["users"] if not ((u.get("billing") or {}).get("status") == "pending" and float((u.get("billing") or {}).get("created_ts") or 0) < limit)]
        removed = len(data["users"]) - len(keep)
        if removed:
            data["users"] = keep
            _save(p, data)
    return removed


def set_password(ident: str, password: str) -> str:
    p = _path()
    data = _load(p)
    u = _resolve(data, ident)
    if re.fullmatch(r"[A-Za-z0-9._@-]{3,64}", (ident or "").strip()) and not re.fullmatch(r"[0-9a-f]{8}", (ident or "").strip()):
        check_password_rules(ident, password)
    else:
        check_password_rules("", password)
    u.update(_hash(password))
    u["updated_at"] = _now()
    _save(p, data)
    return str(u["idk"])[:8]


def set_enabled(ident: str, on: bool) -> str:
    p = _path()
    data = _load(p)
    u = _resolve(data, ident)
    u["enabled"] = bool(on)
    u["updated_at"] = _now()
    _save(p, data)
    return str(u["idk"])[:8]


def remove_user(ident: str) -> str:
    p = _path()
    data = _load(p)
    u = _resolve(data, ident)
    data["users"].remove(u)
    _save(p, data)
    return str(u["idk"])[:8]


# ── CLI ──
def _get_password(args, user_id: str) -> tuple[str, bool]:
    if args.generate:
        return secrets.token_urlsafe(12), True
    if args.password is not None:
        return args.password, False
    if args.password_stdin:
        return sys.stdin.readline().rstrip("\n"), False
    pw = getpass.getpass("パスワード: ")
    if getpass.getpass("もう一度: ") != pw:
        raise UserError("パスワードが一致しません")
    return pw, False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    for name in ("add", "passwd"):
        s = sp.add_parser(name)
        s.add_argument("id")
        g = s.add_mutually_exclusive_group()
        g.add_argument("--password", help="パスワードを指定する（シェルの履歴に残る）。10 文字以上")
        g.add_argument("--generate", action="store_true", help="ランダムなパスワードを作って 1 回だけ表示")
        g.add_argument("--password-stdin", action="store_true", help="標準入力 1 行目をパスワードにする")
    for name in ("disable", "enable", "remove"):
        sp.add_parser(name).add_argument("id")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "list":
            rows = list_users()
            if not rows:
                print("ユーザーはいません")
            for r in rows:
                print(f"{r['fp']:<9}{r['source']:<9}{'有効' if r['enabled'] else '無効':<5}{r['created_at'] or '-':<27}{r['label'] or ''} {r['note']}".rstrip())
            return 0
        if a.cmd in ("add", "passwd"):
            pw, gen = _get_password(a, a.id)
            fp = create_user(a.id, pw) if a.cmd == "add" else set_password(a.id, pw)
            print(("追加しました" if a.cmd == "add" else "パスワードを変更しました") + f"（指紋 {fp}）")
            if gen:
                print(f"パスワード（この 1 回だけ表示）: {pw}")
            return 0
        fn = {"disable": lambda i: set_enabled(i, False), "enable": lambda i: set_enabled(i, True), "remove": remove_user}[a.cmd]
        fp = fn(a.id)
        print({"disable": "無効にしました（ログイン中のセッションも使えなくなります）", "enable": "有効にしました", "remove": "削除しました"}[a.cmd] + f"（指紋 {fp}）")
        return 0
    except UserError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
