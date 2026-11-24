"""Secret Manager の最小の窓口（REST・google-auth だけ。追加のライブラリは使わない）。

共有ログインの毎月の切り替え（src/scripts/ops/login_rotate.py）が、パスワードの秘密 1 つの版を読み・足し・消すためだけに使う。
値はこのモジュールの外に返すだけで、ログ・例外の文には出さない（例外は HTTP の状態と操作の名前だけ）。
費用: 版の読み出し・追加・一覧は 1 回の Job で数回（月 1 万回まで無料）。有効な版は 1 版 約 9 円/月（destroy で外れる）。
"""

from __future__ import annotations

import base64
import os
from typing import Protocol

API = "https://secretmanager.googleapis.com/v1"


class SecretError(RuntimeError):
    """Secret Manager の操作の失敗（文言に秘密の値は入れない）。"""


class SecretStore(Protocol):
    def latest(self) -> tuple[str, bytes] | None: ...          # (版の名前, 中身)。版が無ければ None
    def read(self, version: str) -> bytes: ...
    def add(self, data: bytes) -> str: ...                      # 足した版の名前
    def enabled_versions(self) -> list[str]: ...                # 有効な版の名前（新しい順）
    def destroy(self, version: str) -> None: ...


class GcpSecretStore:
    """1 つの秘密（projects/<P>/secrets/<name>）を扱う。認証は実行環境の既定（Cloud Run Jobs ならその SA）。"""

    def __init__(self, name: str, project: str | None = None) -> None:
        project = project or os.environ.get("GCS_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
        if not project:
            raise SecretError("プロジェクトが未設定です（GCS_PROJECT_ID）")
        self.secret = f"projects/{project}/secrets/{name}"
        self._session = None

    def _s(self):
        if self._session is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession

            creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            self._session = AuthorizedSession(creds)
        return self._session

    def _call(self, op: str, method: str, url: str, **kw):
        r = self._s().request(method, url, timeout=30, **kw)
        if r.status_code >= 300:
            raise SecretError(f"Secret Manager の {op} に失敗しました（HTTP {r.status_code}）")
        return r.json() if r.content else {}

    def latest(self) -> tuple[str, bytes] | None:
        r = self._s().request("GET", f"{API}/{self.secret}/versions/latest:access", timeout=30)
        if r.status_code in (404, 400, 412):        # 版が無い（作ったばかりの秘密）・最新が無効
            if not self.enabled_versions():
                return None
        if r.status_code >= 300:
            raise SecretError(f"Secret Manager の読み出しに失敗しました（HTTP {r.status_code}）")
        d = r.json()
        return d["name"], base64.b64decode(d["payload"]["data"])

    def read(self, version: str) -> bytes:
        d = self._call("読み出し", "GET", f"{API}/{version}:access")
        return base64.b64decode(d["payload"]["data"])

    def add(self, data: bytes) -> str:
        d = self._call("版の追加", "POST", f"{API}/{self.secret}:addVersion", json={"payload": {"data": base64.b64encode(data).decode()}})
        return d["name"]

    def enabled_versions(self) -> list[str]:
        out: list[str] = []
        token = ""
        while True:
            q = {"filter": "state:ENABLED", "pageSize": 100}
            if token:
                q["pageToken"] = token
            d = self._call("版の一覧", "GET", f"{API}/{self.secret}/versions", params=q)
            out += [v["name"] for v in d.get("versions", [])]
            token = d.get("nextPageToken") or ""
            if not token:
                break
        return sorted(out, key=lambda n: int(n.rsplit("/", 1)[-1]), reverse=True)

    def destroy(self, version: str) -> None:
        self._call("版の destroy", "POST", f"{API}/{version}:destroy", json={})


class MemorySecretStore:
    """テスト用（版は 1, 2, … の連番）。"""

    def __init__(self) -> None:
        self.versions: dict[str, bytes] = {}
        self.destroyed: list[str] = []
        self.n = 0

    def latest(self) -> tuple[str, bytes] | None:
        """本物と同じく「最後に作った版」。それが消されていれば、有効な版が無いときだけ None、あれば失敗。"""
        name = f"v/{self.n}"
        if name in self.versions:
            return name, self.versions[name]
        if not self.versions:
            return None
        raise SecretError("最新の版が無効です")

    def read(self, version: str) -> bytes:
        if version not in self.versions:
            raise SecretError("版がありません")
        return self.versions[version]

    def add(self, data: bytes) -> str:
        self.n += 1
        name = f"v/{self.n}"
        self.versions[name] = bytes(data)
        return name

    def enabled_versions(self) -> list[str]:
        return sorted(self.versions, key=lambda n: int(n.rsplit("/", 1)[-1]), reverse=True)

    def destroy(self, version: str) -> None:
        self.versions.pop(version, None)
        self.destroyed.append(version)
