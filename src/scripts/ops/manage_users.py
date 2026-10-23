"""会員ユーザー（ログイン ID とパスワード）の管理。パスワードはソルト付き PBKDF2-SHA256 のハッシュだけを保存し、画面にも記録にも出さない。

  python -m src.scripts.ops.manage_users list
  python -m src.scripts.ops.manage_users add <id> [--name 表示名] [--password 初期パスワード | --generate | --password-stdin]    # 既定は対話入力（2 回）
  python -m src.scripts.ops.manage_users passwd <id> [--password 新パスワード | --generate | --password-stdin]
  （--password はシェルの履歴に残る。残したくないときは対話入力か --password-stdin を使う）
  python -m src.scripts.ops.manage_users rename <id> --name 表示名
  python -m src.scripts.ops.manage_users disable <id> | enable <id> | remove <id>

保存先: data/local/auth/members.json（git に入れない。環境変数 KEIBA_MEMBERS_FILE で変更）。config/members.json（動作確認用の test）は読むだけで、ここからは変更しない。
API は会員ファイルの更新を自動で読み直す（再起動不要）。無効化・削除したユーザーは、すでにログイン中でも次のリクエストから使えなくなる。
本番ではホストで実行する: ssh keiba-remote 'cd /home/jovyan/work/keiba-vpn && sudo -n /opt/conda/bin/python3 -m src.scripts.ops.manage_users list'
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
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
    return {"_comment": "ユーザー管理 CLI（src/scripts/ops/manage_users.py）が書く会員アカウント。パスワードはハッシュだけ。git に入れない。", "users": []}


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


def hash_password(password: str) -> dict:
    salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS).hex()
    return {"algo": "pbkdf2_sha256", "iterations": ITERATIONS, "salt": salt.hex(), "hash": h}


def check_password_rules(user_id: str, password: str) -> None:
    if len(password) < MIN_PW:
        raise SystemExit(f"パスワードは {MIN_PW} 文字以上にしてください")
    if password.lower() == user_id.lower():
        raise SystemExit("パスワードに ID と同じ文字列は使えません")


def _get_password(args, user_id: str) -> tuple[str, bool]:
    """(パスワード, 生成したか)。"""
    if args.generate:
        return secrets.token_urlsafe(12), True
    if args.password is not None:
        check_password_rules(user_id, args.password)
        return args.password, False
    if args.password_stdin:
        pw = sys.stdin.readline().rstrip("\n")
    else:
        pw = getpass.getpass("パスワード: ")
        if getpass.getpass("もう一度: ") != pw:
            raise SystemExit("パスワードが一致しません")
    check_password_rules(user_id, pw)
    return pw, False


def _find(data: dict, user_id: str) -> dict | None:
    return next((u for u in data["users"] if u.get("id") == user_id), None)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+09:00", time.gmtime(time.time() + 9 * 3600))


def cmd_list(args) -> int:
    from src.api import auth

    local = {u["id"]: u for u in _load(_path())["users"]}
    rows = []
    for u in auth._read_users(auth._MEMBERS_FILE):
        if u["id"] not in local:
            rows.append((u["id"], "config", u.get("name") or "", "有効" if auth.member_active(u["id"]) else f"無効（{u.get('requires_env') or '-'} が未設定）"))
    for uid, u in local.items():
        rows.append((uid, "管理ファイル", u.get("name") or "", "有効" if auth.member_active(uid) else "無効"))
    if not rows:
        print("ユーザーはいません")
    for r in rows:
        print("  ".join(str(x) for x in r))
    return 0


def cmd_add(args) -> int:
    if not ID_RE.match(args.id):
        raise SystemExit("ID は英数字と . _ @ - の 3〜64 文字にしてください")
    p = _path()
    data = _load(p)
    if _find(data, args.id):
        raise SystemExit(f"{args.id} はすでにあります（パスワード変更は passwd）")
    pw, gen = _get_password(args, args.id)
    data["users"].append({"id": args.id, "name": args.name or args.id, "role": "member", **hash_password(pw), "enabled": True, "created_at": _now(), "updated_at": _now()})
    _save(p, data)
    print(f"追加しました: {args.id}")
    if gen:
        print(f"パスワード（この 1 回だけ表示）: {pw}")
    return 0


def _mutate(args, fn, msg: str) -> int:
    p = _path()
    data = _load(p)
    u = _find(data, args.id)
    if not u:
        raise SystemExit(f"{args.id} は管理ファイルにありません（config/members.json のユーザーはここから変更できません）")
    fn(data, u)
    _save(p, data)
    print(msg.format(id=args.id))
    return 0


def cmd_passwd(args) -> int:
    pw, gen = _get_password(args, args.id)

    def f(d, u):
        u.update(hash_password(pw))
        u["updated_at"] = _now()

    rc = _mutate(args, f, "パスワードを変更しました: {id}")
    if gen:
        print(f"パスワード（この 1 回だけ表示）: {pw}")
    return rc


def cmd_rename(args) -> int:
    def f(d, u):
        u["name"] = args.name
        u["updated_at"] = _now()

    return _mutate(args, f, "表示名を変更しました: {id}")


def _set_enabled(on: bool):
    def run(args) -> int:
        def f(d, u):
            u["enabled"] = on
            u["updated_at"] = _now()

        return _mutate(args, f, ("有効にしました: {id}" if on else "無効にしました（ログイン中のセッションも使えなくなります）: {id}"))

    return run


def cmd_remove(args) -> int:
    return _mutate(args, lambda d, u: d["users"].remove(u), "削除しました: {id}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list").set_defaults(fn=cmd_list)
    for name, fn in (("add", cmd_add), ("passwd", cmd_passwd)):
        s = sp.add_parser(name)
        s.add_argument("id")
        if name == "add":
            s.add_argument("--name")
        g = s.add_mutually_exclusive_group()
        g.add_argument("--password", help="パスワードを指定する（シェルの履歴に残る）。10 文字以上")
        g.add_argument("--generate", action="store_true", help="ランダムなパスワードを作って 1 回だけ表示")
        g.add_argument("--password-stdin", action="store_true", help="標準入力 1 行目をパスワードにする")
        s.set_defaults(fn=fn)
    s = sp.add_parser("rename"); s.add_argument("id"); s.add_argument("--name", required=True); s.set_defaults(fn=cmd_rename)
    for name, fn in (("disable", _set_enabled(False)), ("enable", _set_enabled(True)), ("remove", cmd_remove)):
        s = sp.add_parser(name); s.add_argument("id"); s.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
