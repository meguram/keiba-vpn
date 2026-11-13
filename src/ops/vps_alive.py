"""VPS 全体の停止の検知（ホストの cron、15 分おき）。GCP を起こさない・GCS を読まない。

    python -m src.ops.vps_alive            # 確かめて、状態が変わったときだけ Slack
    python -m src.ops.vps_alive --dry-run  # 確かめるだけ

公開の画面の ``https://app.megu-keiba.com/version.json`` を取りに行く（Cloudflare → VPS の Tunnel → nginx → Next の静的ファイル。
``/api/`` ではないので Cloud Run を起こさない。Cloudflare のキャッシュは DYNAMIC で、毎回 VPS まで届く）。
中身が JSON で env=prod なら生きているとみなす。2 回続けて失敗したら「VPS 停止の疑い」、戻ったら「復旧」を 1 回ずつ送る。
VPS の中の自己診断（keiba-ui-health）では、VPS 全体の停止・回線断は知らせられないため（定期実行の一覧の点検 #1）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

URL = os.environ.get("KEIBA_VPS_ALIVE_URL", "https://app.megu-keiba.com/version.json")
STATE = Path(os.environ.get("KEIBA_VPS_ALIVE_STATE", str(Path(__file__).resolve().parents[2] / "data/local/meta/vps_alive_state.json")))
FAILS_TO_ALERT = 2


def probe(url: str = URL, timeout: float = 15.0) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(f"{url}?t={int(time.time())}", headers={"User-Agent": "keiba-vps-alive/1", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310  固定の自前の URL
            body = r.read(4096).decode("utf-8", "replace")
            if r.status != 200:
                return False, f"HTTP {r.status}"
        d = json.loads(body)
        if d.get("env") != "prod" or not d.get("release"):
            return False, "version.json の中身が想定と違う（Access の画面などの疑い）"
        return True, str(d.get("release"))
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:120]}"


def step(state: dict[str, Any], ok: bool, detail: str, now: float) -> tuple[dict[str, Any], str | None]:
    """(新しい状態, 送る種類 'down'|'up'|None)。"""
    st = dict(state)
    if ok:
        was_down = bool(st.get("down"))
        st.update({"fails": 0, "down": False, "last_ok": now, "detail": detail})
        return st, ("up" if was_down else None)
    st["fails"] = int(st.get("fails") or 0) + 1
    st["detail"] = detail
    st.setdefault("first_fail", now)
    if st["fails"] >= FAILS_TO_ALERT and not st.get("down"):
        st["down"] = True
        return st, "down"
    return st, None


def main(argv: list[str] | None = None, *, probe_fn: Any = probe, notify: Any = None, now: float | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    now = time.time() if now is None else now
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()   # cron は .env を読まないので、Slack の webhook をここで読む
    ok, detail = probe_fn()
    print(json.dumps({"ok": ok, "detail": detail}, ensure_ascii=False))
    if a.dry_run:
        return 0
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    if ok:
        state.pop("first_fail", None)
    new, send = step(state, ok, detail, now)
    if send:
        if notify is None:
            from src.utils.slack_notify import notify_event

            notify = notify_event
        if send == "down":
            r = notify("VPS の停止の疑い（公開の画面に届かない）", [f"{URL} に {new['fails']} 回続けて届かない: {detail}",
                                                           "確かめ方: ssh keiba-vps 'systemctl is-active keiba-ui nginx cloudflared-prod'"], severity="critical")
        else:
            r = notify("VPS が復旧しました（公開の画面に届く）", [f"版 {detail}"], severity="normal")
        if isinstance(r, dict) and not r.get("ok", True):
            # 送れなかった: 状態を変えずに次の回に送り直す
            print(json.dumps({"notify_failed": r.get("error") or r.get("reason") or "unknown"}, ensure_ascii=False))
            new = dict(state, fails=new["fails"], detail=detail)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
