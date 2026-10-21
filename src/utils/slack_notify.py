"""Slack 通知（Block Kit の組み立て・送信・プレビュー）。

既存の ``src.utils.notify.notify_slack``（テキスト 1 件の簡易通知）は変更せず、構造化した通知はこのモジュールを使う。

- 宛先は Incoming Webhook の環境変数で決める（値はコード・ログ・文書に出さない。変数名だけ書く）:
  ``SLACK_WEBHOOK_URL``（既定）/ ``SLACK_WEBHOOK_URL_RESULTS`` / ``SLACK_WEBHOOK_URL_REPORT`` / ``SLACK_WEBHOOK_URL_ALERT``。
  個別の変数が空なら ``SLACK_WEBHOOK_URL`` にまとめて送る（チャンネルを 1 つにしても動く）。
- 送信は失敗しても本処理を止めない（例外を投げず、結果の辞書を返す）。429 は Retry-After に従い、5xx・接続エラーは指数バックオフで再試行。
- 本文は ``attachments[0].blocks`` に入れて、重要度の色（左の帯）を付ける。header ブロックは attachment 内で使えない環境があるので、
  見出しは太字の section にする。
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from typing import Any, Callable

logger = logging.getLogger("utils.slack_notify")

SEVERITIES = ("normal", "warning", "critical")
SEVERITY_LABEL = {"normal": "通常", "warning": "注意", "critical": "緊急"}
SEVERITY_COLOR = {"normal": "#3b82f6", "good": "#2eb67d", "warning": "#ecb22e", "critical": "#e01e5a"}

CHANNEL_ENV = {
    "default": "SLACK_WEBHOOK_URL",
    "results": "SLACK_WEBHOOK_URL_RESULTS",
    "report": "SLACK_WEBHOOK_URL_REPORT",
    "alert": "SLACK_WEBHOOK_URL_ALERT",
}

MAX_BLOCKS = 50          # Slack の上限（メッセージ全体）
MAX_SECTION_TEXT = 3000  # section.text の上限
MAX_FIELD_TEXT = 2000    # section.fields[] の上限
MAX_FIELDS = 10
MAX_CONTEXT = 10

_URL_RE = re.compile(r"https?://hooks\.slack\.com/[^\s\"'<>]+", re.I)


# ───────────────────────── 整形 ─────────────────────────

def esc(text: Any) -> str:
    """mrkdwn の特殊文字（& < >）をエスケープする。"""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(limit - 1, 0)] + "…"


def fmt_yen(n: float | int | None, *, signed: bool = False) -> str:
    if n is None:
        return "—"
    v = int(round(n))
    s = f"{abs(v):,}"
    if signed:
        return ("+" if v > 0 else "-" if v < 0 else "±") + "¥" + s
    return ("-" if v < 0 else "") + "¥" + s


def fmt_pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def fmt_ratio(num: int | float, den: int | float, digits: int = 1) -> str:
    """件数と率: ``3/8 (37.5%)``。分母 0 は ``0/0``。"""
    if not den:
        return f"{int(num)}/{int(den)}"
    return f"{int(num)}/{int(den)} ({num / den * 100:.{digits}f}%)"


def arrow(delta: float | None, *, eps: float = 1e-9) -> str:
    """差の向き。None は空文字。"""
    if delta is None:
        return ""
    return "▲" if delta > eps else "▼" if delta < -eps else "→"


def fmt_delta_pt(delta: float | None) -> str:
    """率の差（ポイント）。0.052 → ``▲5.2pt``。"""
    if delta is None:
        return ""
    return f"{arrow(delta)}{abs(delta) * 100:.1f}pt"


def jp_weekday(date_yyyymmdd: str) -> str:
    from datetime import datetime

    d = datetime.strptime(date_yyyymmdd, "%Y%m%d")
    return "月火水木金土日"[d.weekday()]


# ───────────────────────── Block Kit の部品 ─────────────────────────

def section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": clip(text, MAX_SECTION_TEXT)}}


def fields(pairs: list[tuple[str, str]]) -> dict:
    """2 列の項目。各項目は ``*見出し*\\n値``。10 個まで。"""
    fs = [{"type": "mrkdwn", "text": clip(f"*{label}*\n{value}", MAX_FIELD_TEXT)} for label, value in pairs[:MAX_FIELDS]]
    return {"type": "section", "fields": fs}


def context(items: list[str]) -> dict:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": clip(t, MAX_FIELD_TEXT)} for t in items[:MAX_CONTEXT]]}


def divider() -> dict:
    return {"type": "divider"}


def code_table(headers: list[str], rows: list[list[str]], *, aligns: str | None = None) -> str:
    """等幅のテキスト表（``` で囲む）。全角を 2 幅として数える。aligns は 'l'/'r' の文字列。"""
    def width(s: str) -> int:
        return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)

    cols = len(headers)
    aligns = aligns or "l" + "r" * (cols - 1)
    data = [headers] + rows
    ws = [max(width(str(r[i])) for r in data) for i in range(cols)]

    def cell(s: str, i: int) -> str:
        pad = ws[i] - width(s)
        return s + " " * pad if aligns[i] == "l" else " " * pad + s

    lines = ["  ".join(cell(str(r[i]), i) for i in range(cols)).rstrip() for r in data]
    return "```" + "\n".join(lines) + "```"


def build_message(title: str, blocks: list[dict], *, severity: str = "normal", fallback: str | None = None,
                  color: str | None = None) -> dict:
    """Webhook に送るペイロード。本文は attachments[0].blocks（色の帯つき）。

    ブロック数が上限を超えたら、末尾を削って注記を足す（メッセージ全体が拒否されるのを防ぐ）。
    """
    sev = severity if severity in SEVERITIES else "normal"
    bl = list(blocks)
    if len(bl) > MAX_BLOCKS - 1:
        bl = bl[: MAX_BLOCKS - 2] + [context(["（ブロック数の上限のため一部を省略しました）"])]
    return {
        "text": clip(fallback or title, 3000),
        "attachments": [{"color": color or SEVERITY_COLOR[sev], "blocks": bl}],
    }


# ───────────────────────── 宛先・送信 ─────────────────────────

def webhook_url(channel: str = "default", env: dict[str, str] | None = None) -> str:
    """チャンネル名 → Webhook URL。個別の変数が空なら SLACK_WEBHOOK_URL を使う。未設定は空文字。"""
    e = os.environ if env is None else env
    name = CHANNEL_ENV.get(channel, CHANNEL_ENV["default"])
    url = (e.get(name) or "").strip()
    if not url and name != CHANNEL_ENV["default"]:
        url = (e.get(CHANNEL_ENV["default"]) or "").strip()
    return url


def redact(text: str, env: dict[str, str] | None = None) -> str:
    """ログ・エラー文に Webhook URL が混ざらないよう伏せる。"""
    e = os.environ if env is None else env
    out = _URL_RE.sub("<webhook>", str(text))
    for name in CHANNEL_ENV.values():
        v = (e.get(name) or "").strip()
        if v:
            out = out.replace(v, "<webhook>")
    return out


_ENV_LABELS = {"stg": "STG", "prod": "PROD", "dev": "DEV"}
_LOCATION_LABELS = {"host": "ホストPC", "vps": "VPS", "gcp": "GCP"}
_ENV_MARK = "_keiba_env_label"


def env_label(env: dict[str, str] | None = None) -> str:
    """通知に付ける環境名（STG / PROD / DEV）。KEIBA_NOTIFY_ENV_LABEL があればそれを優先。KEIBA_ENV 未設定は空（付けない）。"""
    e = os.environ if env is None else env
    explicit = (e.get("KEIBA_NOTIFY_ENV_LABEL") or "").strip()
    if explicit:
        return explicit[:12]
    return _ENV_LABELS.get((e.get("KEIBA_ENV") or "").strip().lower(), "")


def with_env_label(payload: dict, env: dict[str, str] | None = None) -> dict:
    """どの環境（stg/prod）・どこで動くジョブからの通知かが分かるよう、通知文の先頭と末尾に環境名を付ける（二重には付けない）。"""
    label = env_label(env)
    if not label or payload.get(_ENV_MARK):
        return payload
    e = os.environ if env is None else env
    loc = _LOCATION_LABELS.get((e.get("KEIBA_RUN_LOCATION") or "").strip().lower(), "")
    out = dict(payload)
    out["text"] = clip(f"【{label}】{payload.get('text') or ''}", 3000)
    atts = [dict(a) for a in payload.get("attachments") or []]
    if atts:
        blocks = list(atts[0].get("blocks") or [])
        blocks.append(context([f"環境: *{label}*" + (f" ・ 実行場所: {loc}" if loc else "")]))
        atts[0]["blocks"] = blocks[:MAX_BLOCKS]
        out["attachments"] = atts
    out[_ENV_MARK] = True
    return out


def send_payload(payload: dict, *, channel: str = "default", timeout: float = 8.0, retries: int = 3,
                 backoff: float = 1.5, sleep: Callable[[float], None] = time.sleep,
                 post: Callable[..., Any] | None = None, env: dict[str, str] | None = None) -> dict:
    """ペイロードを送る。例外は投げない。戻り値: ``{"ok": bool, "status": int|None, "attempts": int, "error": str|None}``。

    - Webhook 未設定は ``ok=False, error="webhook_not_configured"``（何も送らない）。
    - 429 は Retry-After（秒）に従う。5xx・接続エラーは backoff^n 秒待って再試行。4xx（429 以外）は再試行しない。
    """
    url = webhook_url(channel, env)
    if not url:
        return {"ok": False, "status": None, "attempts": 0, "error": "webhook_not_configured"}
    payload = with_env_label(payload, env)
    payload.pop(_ENV_MARK, None)
    if post is None:
        import requests

        post = requests.post
    last_err: str | None = None
    status: int | None = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            resp = post(url, json=payload, timeout=timeout)
            status = int(getattr(resp, "status_code", 0) or 0)
            if 200 <= status < 300:
                return {"ok": True, "status": status, "attempts": attempt, "error": None}
            last_err = f"http_{status}"
            if status == 429:
                try:
                    wait = float((getattr(resp, "headers", {}) or {}).get("Retry-After") or backoff ** attempt)
                except (TypeError, ValueError):
                    wait = backoff ** attempt
                if attempt < retries:
                    sleep(min(wait, 30.0))
                continue
            if 400 <= status < 500:
                break
        except Exception as exc:  # noqa: BLE001
            last_err = redact(f"{type(exc).__name__}: {exc}", env)
            status = None
        if attempt < retries:
            sleep(backoff ** attempt)
    logger.warning("Slack 通知の送信に失敗 (channel=%s, status=%s): %s", channel, status, last_err)
    return {"ok": False, "status": status, "attempts": attempt, "error": last_err}


def send_test(*, channel: str = "default", **kw: Any) -> dict:
    """接続テスト用の最小メッセージ（実行はユーザーの確認後に行う）。"""
    payload = build_message("接続テスト", [section("*接続テスト*\nkeiba-vpn からの Slack 通知の疎通確認です。このメッセージは無視してください。")],
                            severity="normal", fallback="keiba-vpn 接続テスト")
    return send_payload(payload, channel=channel, **kw)


def notify_event(title: str, lines: list[str], *, severity: str = "warning", channel: str = "alert",
                 link: str | None = None, **kw: Any) -> dict:
    """運用の異常などの 1 件通知（見出し + 箇条書き）。重要度で色と見出しの語を変える。"""
    sev = severity if severity in SEVERITIES else "warning"
    blocks = [section(f"*[{SEVERITY_LABEL[sev]}] {esc(title)}*")]
    if lines:
        blocks.append(section("\n".join("• " + esc(x) for x in lines)))
    if link:
        blocks.append(context([f"<{link}|詳細を開く>"]))
    return send_payload(build_message(title, blocks, severity=sev, fallback=f"[{SEVERITY_LABEL[sev]}] {title}"), channel=channel, **kw)


# ───────────────────────── プレビュー（送らずに見た目を確認する） ─────────────────────────

def _mrkdwn_to_html(text: str) -> str:
    t = html.escape(text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
    t = re.sub(r"```(.*?)```", lambda m: "<pre>" + m.group(1).strip("\n") + "</pre>", t, flags=re.S)
    t = re.sub(r"\*(.+?)\*", r"<b>\1</b>", t)
    t = re.sub(r"&lt;(https?://[^|&]+)\|([^&]+?)&gt;", r'<a href="\1">\2</a>', t)
    return t.replace("\n", "<br>")


def render_preview_html(payload: dict, *, title: str = "Slack 通知プレビュー") -> str:
    """ペイロードを簡易に HTML 化する（Slack の見た目に近い目安。実際の表示は Slack による）。"""
    parts: list[str] = []
    for att in payload.get("attachments", []):
        color = att.get("color") or "#999"
        inner: list[str] = []
        for b in att.get("blocks", []):
            t = b.get("type")
            if t == "section" and "text" in b:
                inner.append(f"<div class='sec'>{_mrkdwn_to_html(b['text']['text'])}</div>")
            elif t == "section" and "fields" in b:
                cells = "".join(f"<div class='fld'>{_mrkdwn_to_html(f['text'])}</div>" for f in b["fields"])
                inner.append(f"<div class='flds'>{cells}</div>")
            elif t == "context":
                inner.append("<div class='ctx'>" + " ・ ".join(_mrkdwn_to_html(e["text"]) for e in b["elements"]) + "</div>")
            elif t == "divider":
                inner.append("<hr>")
        parts.append(f"<div class='att' style='border-left:4px solid {color}'>" + "".join(inner) + "</div>")
    style = (
        "body{font-family:-apple-system,'Segoe UI','Noto Sans JP',sans-serif;background:#f4f4f4;margin:0;padding:16px}"
        ".msg{max-width:560px;margin:0 auto;background:#fff;border-radius:8px;padding:12px 14px;box-shadow:0 1px 3px #0002}"
        ".fb{color:#555;font-size:12px;margin-bottom:6px}.att{padding:4px 0 4px 10px;margin:6px 0}"
        ".sec{margin:6px 0;font-size:14px;line-height:1.45}.flds{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin:6px 0}"
        ".fld{font-size:13px;line-height:1.4}.ctx{color:#666;font-size:12px;margin:6px 0}pre{background:#f6f8fa;border:1px solid #e1e4e8;"
        "border-radius:6px;padding:6px 8px;font-size:12px;overflow-x:auto;margin:4px 0}hr{border:0;border-top:1px solid #ddd}"
        "@media(max-width:480px){.flds{grid-template-columns:1fr}}"
    )
    return (
        "<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{style}</style></head><body><div class='msg'>"
        f"<div class='fb'>通知テキスト（プッシュ通知に出る文）: {html.escape(str(payload.get('text', '')))}</div>"
        + "".join(parts) + "</div></body></html>"
    )


def render_preview_text(payload: dict) -> str:
    """ペイロードを等幅のテキストにする（文書に見本として載せる用。Slack の見た目の目安）。"""
    def plain(t: str) -> str:
        t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        t = re.sub(r"```\n?", "", t)
        return re.sub(r"\*(.+?)\*", r"\1", t)

    out: list[str] = []
    for att in payload.get("attachments", []):
        out.append("┃ " + "─" * 40)
        for b in att.get("blocks", []):
            t = b.get("type")
            if t == "section" and "text" in b:
                out += ["┃ " + ln for ln in plain(b["text"]["text"]).splitlines()]
            elif t == "section" and "fields" in b:
                for f in b["fields"]:
                    head, _, val = plain(f["text"]).partition("\n")
                    out.append(f"┃ {head}: {val}")
            elif t == "context":
                out.append("┃ （" + " ・ ".join(plain(e["text"]) for e in b["elements"]) + "）")
            elif t == "divider":
                out.append("┃ " + "─" * 40)
    return "\n".join(out)


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
