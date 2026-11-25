"""共有ログイン（会員が共有する 1 つの ID）のパスワードを、毎月 1 日 06:00 JST に乱数の新しい値へ切り替える。

計画と手順: docs/operations/shared_login_rotation.md（計画の経緯は shared_login_rotation_plan_20261119.md）。

  python -m src.scripts.ops.login_rotate run [--step prepare|draft|switch] [--now 2027-01-01T06:00]   # Cloud Run Job keiba-login-rotate
  python -m src.scripts.ops.login_rotate rollback                                                      # 切り替えを 1 つ戻す
  （ホスト）sudo gcloud secrets versions access latest --secret=KEIBA_SHARED_LOGIN_PW … | sudo python3 -m src.scripts.ops.login_rotate draft
  （ホスト）sudo python3 -m src.scripts.ops.login_rotate export-pepper | bash scripts/gcp/put_secret.sh KEIBA_MEMBERS_PEPPER

役（run は日付で振り分ける。Scheduler は 1・25〜31 日の 06:00 JST に起こす）:
- 準備（25 日〜末日の 4 日前）: 来月の「次の版」を乱数で作り、Secret Manager に置き、名簿に HMAC（next_mac）を書く。
- 下書き（末日の 3 日前〜末日）: 準備をやり直し、Secret Manager と名簿がそろっていれば draft_month を書いて「下書きができた」と通知。
  draft_month の後は次の版を作り直さない（note に出した値と切り替わる値をずらさない）。
- 切り替え（1 日）: 次の版が名簿の next_mac と合うときだけ、名簿のハッシュを世代の条件つきで書き換える（ここが確定の点）。
  その後で Secret Manager を追いつかせる。次の版が無い・合わないときは切り替えない（誰も知らないパスワードにしない）。

守り:
- 名簿は GCS の正本だけを読み書きする（KEIBA_MEMBERS_STORE=gcs でなければ止める。手元のファイルに書いて成功を返さない）。
- 共有の記録は ID の idk（ペッパーつきの HMAC）で探す。見つからなければ作らずに止める（＝ペッパーが API と違う合図）。
- 平文・HMAC・Secret Manager の中身は、標準出力・ログ・Slack・例外の文に出さない（draft の画面表示だけ）。
- 結果は秘密の無い ops/login_rotate/status.json に書く（毎朝の schedule-check が 1 日の切り替えの失敗に気づく）。
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

JST = timezone(timedelta(hours=9))
SECRET_NAME = "KEIBA_SHARED_LOGIN_PW"
STATUS_KEY = "ops/login_rotate/status.json"
ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"      # 紛らわしい文字（0 O 1 l i）を除く 31 文字。16 字で約 79 ビット
SWITCH_HOUR = 6
LOGIN_URL = "https://app.megu-keiba.com/login"
KEEP_VERSIONS = 2                                    # 新しい 2 版は消さない（戻しと、同時の実行の相手の版を守る）
FIELDS_NEXT = ("next_month", "next_mac", "draft_month")


class RotateError(RuntimeError):
    """切り替えを止める理由（文言に秘密を入れない。Slack・status.json にそのまま出す）。"""


# ── 日付 ──
def month_of(now: datetime) -> str:
    return now.astimezone(JST).strftime("%Y-%m")


def add_months(month: str, n: int) -> str:
    y, m = (int(x) for x in month.split("-"))
    k = y * 12 + (m - 1) + n
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


def draft_day(now: datetime) -> int:
    """下書きの日＝末日の 3 日前（11 月なら 27 日・2 月なら 25 日）。"""
    t = now.astimezone(JST)
    return calendar.monthrange(t.year, t.month)[1] - 3


def step_for(now: datetime) -> str | None:
    d = now.astimezone(JST).day
    if d == 1:
        return "switch"
    if d >= draft_day(now):
        return "draft"
    if d >= 25:
        return "prepare"
    return None


# ── 値 ──
def generate_password() -> str:
    chars = "".join(secrets.choice(ALPHABET) for _ in range(16))
    return "-".join(chars[i:i + 4] for i in range(0, 16, 4))


def pw_mac(password: str) -> str:
    """名簿に置く HMAC（ペッパーつき）。平文は約 79 ビットの乱数なので、速い HMAC でも総当たりは成り立たない。"""
    from src.api import auth

    return hmac.new(auth._pepper(), b"shared-login-pw:" + (password or "").encode(), hashlib.sha256).hexdigest()


def _mac_eq(password: str, mac: str | None) -> bool:
    return bool(mac) and hmac.compare_digest(pw_mac(password), str(mac))


def verify_hash(rec: dict, password: str) -> bool:
    """記録の今のハッシュ（PBKDF2＋ペッパー）と合うか。"""
    from src.api import auth

    try:
        if rec.get("algo") != "pbkdf2_sha256_pepper":
            return False
        got = auth.password_digest(password, bytes.fromhex(rec["salt"]), int(rec["iterations"]))
        return hmac.compare_digest(got, str(rec["hash"]))
    except (KeyError, TypeError, ValueError):
        return False


def _parse_payload(raw: bytes | None) -> dict:
    if not raw:
        return {"current": None, "next": None}
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise RotateError("Secret Manager の中身の形が違います") from e
    if not isinstance(d, dict):
        raise RotateError("Secret Manager の中身の形が違います")
    out = {}
    for k in ("current", "next"):
        v = d.get(k)
        out[k] = v if isinstance(v, dict) and v.get("month") and v.get("password") else None
    return out


def _dump_payload(current: dict | None, nxt: dict | None) -> bytes:
    return json.dumps({"current": current, "next": nxt}, ensure_ascii=False).encode("utf-8")


# ── 名簿 ──
def _shared_id() -> str:
    uid = (os.environ.get("KEIBA_SHARED_LOGIN_ID") or "").strip()
    if not uid:
        raise RotateError("KEIBA_SHARED_LOGIN_ID が未設定です")
    return uid


def _require_gcs() -> None:
    from src.scripts.ops import manage_users as mu

    if not mu.gcs_store_enabled():
        raise RotateError("名簿の正本が GCS ではありません（KEIBA_MEMBERS_STORE=gcs と GCS_BUCKET が要る）。手元のファイルには書きません")


def _find(data: dict, uid: str) -> dict:
    from src.api import auth

    k = auth.id_key(uid)
    hit = [u for u in data.get("users", []) if u.get("idk") == k]
    if len(hit) != 1:
        raise RotateError("名簿に共有の記録が見つかりません（ID かペッパーが API と違う）。作らずに止めます")
    return hit[0]


def _read_record(uid: str) -> dict:
    from src.scripts.ops import manage_users as mu

    try:
        return dict(_find(mu._current(), uid))
    except mu.UserError as e:
        raise RotateError(str(e)) from e


def _transact(uid: str, fn):
    """名簿の共有の記録を、世代の条件つきで変える。判定は fn の中でやり直す（世代の衝突で読み直したときにも効く）。"""
    from src.scripts.ops import manage_users as mu

    try:
        return mu._transact(lambda data: fn(_find(data, uid)))
    except mu.UserError as e:
        raise RotateError(str(e)) from e


# ── Secret Manager ──
def _store():
    from src.ops.secret_store import GcpSecretStore

    return GcpSecretStore(SECRET_NAME)


def _latest(sm) -> tuple[str | None, dict]:
    got = sm.latest()
    if got is None:
        return None, _parse_payload(None)
    return got[0], _parse_payload(got[1])


def _find_version(sm, pred) -> dict | None:
    """有効な版の中から、pred(payload) が真のものを新しい順に探す（最新が壊れた・同時の実行で上書きされたときの立て直し）。"""
    for v in sm.enabled_versions():
        try:
            p = _parse_payload(sm.read(v))
        except Exception:  # noqa: BLE001
            continue
        if pred(p):
            return p
    return None


def _cleanup(sm, rec: dict) -> int:
    """新しい KEEP_VERSIONS 版より古く、名簿の今の版・次の版のどちらとも合わない版を destroy する（順番だけでは消さない）。"""
    n = 0
    for v in sm.enabled_versions()[KEEP_VERSIONS:]:
        try:
            p = _parse_payload(sm.read(v))
        except Exception:  # noqa: BLE001
            continue                                # 読めない版は消さない（一時的な失敗で、今の値・次の値を失わない）
        macs = [rec.get(k) for k in ("cur_mac", "next_mac", "prev_mac")]
        keep = any(x and any(_mac_eq(x["password"], m) for m in macs) for x in (p["current"], p["next"]))
        if not keep:
            sm.destroy(v)
            n += 1
    return n


# ── 役 ──
def prepare(now: datetime, sm, uid: str) -> str:
    """来月の次の版を用意する。戻り値: ok（そろっている）/ repaired（版を立て直した）/ created。"""
    _require_gcs()
    target = add_months(month_of(now), 1)
    rec = _read_record(uid)
    if rec.get("enabled", True) is False:
        raise RotateError("共有の記録が無効にされています。切り替えの準備をしません")
    _, p = _latest(sm)
    nx = p["next"]
    if rec.get("next_month") == target and rec.get("next_mac"):
        if nx and nx["month"] == target and _mac_eq(nx["password"], rec["next_mac"]):
            return "ok"
        found = _find_version(sm, lambda q: bool(q["next"] and q["next"]["month"] == target and _mac_eq(q["next"]["password"], rec["next_mac"])))
        if found:
            sm.add(_dump_payload(p["current"] or found["current"], found["next"]))
            return "repaired"
        if rec.get("draft_month") == target:
            raise RotateError("下書きの後で、次の版が Secret Manager に見つかりません。作り直しません（note の値とずれるため）")
    elif rec.get("draft_month") == target:
        raise RotateError("下書きの印があるのに、名簿に次の版がありません。作り直しません")
    observed = rec.get("next_mac") if rec.get("next_month") == target else None
    pw = generate_password()
    sm.add(_dump_payload(p["current"], {"month": target, "password": pw}))
    mac = pw_mac(pw)

    def fn(r: dict) -> str:
        cur = r.get("next_mac") if r.get("next_month") == target else None
        if r.get("draft_month") == target or cur != observed:
            return "conflict"                       # 同時に別の実行が書いた。名簿の値を正とする
        r.update({"next_month": target, "next_mac": mac, "updated_at": _now_iso()})
        r.pop("draft_month", None)
        return "created"

    res = _transact(uid, fn)
    if res == "conflict":
        rec = _read_record(uid)
        found = _find_version(sm, lambda q: bool(q["next"] and q["next"]["month"] == target and _mac_eq(q["next"]["password"], rec.get("next_mac"))))
        if not found:
            raise RotateError("同時の実行と重なり、名簿の次の版に合う版が見つかりません")
        sm.add(_dump_payload(found["current"], found["next"]))
        return "repaired"
    return res


def draft(now: datetime, sm, uid: str, notify) -> str:
    """下書き: 準備をやり直し、そろっていれば印を書いて通知する（1 か月に 1 回）。戻り値: drafted / ok（済み）。"""
    target = add_months(month_of(now), 1)
    prepare(now, sm, uid)
    rec = _read_record(uid)
    _, p = _latest(sm)
    nx = p["next"]
    if not (nx and nx["month"] == target and _mac_eq(nx["password"], rec.get("next_mac"))):
        raise RotateError("次の版が名簿と合いません（下書きを出しません）")
    cur = p["current"]
    if rec.get("rotation_month") and not (cur and verify_hash(rec, cur["password"])):
        raise RotateError("今の版が名簿と合いません（下書きを出しません）")
    if rec.get("draft_month") == target:
        return "ok"
    y, m = target.split("-")
    if not notify(f"【めぐ競馬】共有ログインの {int(y)}年{int(m)}月分の下書きができました。"
                  f"{int(m)}月1日 {SWITCH_HOUR}:00 に切り替わります。取り出しの 1 行は docs/operations/shared_login_rotation.md（ホストで実行）。"):
        raise RotateError("下書きの通知を Slack に送れませんでした（印を付けず、翌日にやり直します）")

    def fn(r: dict) -> str:
        if r.get("draft_month") == target:
            return "ok"
        if r.get("next_month") != target or r.get("next_mac") != rec.get("next_mac"):
            raise RotateError("下書きの直前に次の版が変わりました")
        r["draft_month"] = target
        return "drafted"

    try:
        return _transact(uid, fn)
    except Exception:
        notify("【めぐ競馬】先ほどの「下書きができました」は取り消します（印を書けませんでした）。note にはまだ出さないでください。翌日にやり直します。")
        raise


def switch(now: datetime, sm, uid: str, notify) -> str:
    """切り替え。戻り値: switched / ok（今月は済み。Secret Manager を追いつかせた）。"""
    _require_gcs()
    month = month_of(now)
    prev = add_months(month, -1)
    rec = _read_record(uid)
    _, p = _latest(sm)
    if rec.get("rotation_month") == month:
        _catch_up(sm, rec, p, month)
        return "ok"
    if rec.get("enabled", True) is False:
        raise RotateError("共有の記録が無効にされています。切り替えません")
    if rec.get("draft_month") != month:
        raise RotateError("今月の下書き（note に出す値の確認）が済んでいません。切り替えません（会員は先月のパスワードのまま使えます）")
    nx = p["next"]
    if not (nx and nx["month"] == month and _mac_eq(nx["password"], rec.get("next_mac"))):
        found = _find_version(sm, lambda q: bool(q["next"] and q["next"]["month"] == month and _mac_eq(q["next"]["password"], rec.get("next_mac"))))
        if not found:
            raise RotateError("今月の次の版が無いか、名簿と合いません。切り替えません（会員は先月のパスワードのまま使えます）")
        nx = found["next"]
    new_pw = nx["password"]
    old_pw = None
    cur = p["current"]
    if cur and verify_hash(rec, cur["password"]):
        old_pw = cur["password"]
    else:                                             # 初回（今の版が派生の値）
        from src.scripts.ops import shared_login as legacy

        cand = legacy.password_for(prev)
        if verify_hash(rec, cand):
            old_pw = cand
        else:
            cand = legacy.password_for(month)        # 旧経路が先に今月へ進めていた場合
            if verify_hash(rec, cand):
                old_pw = cand
    from src.scripts.ops import manage_users as mu

    h = mu._hash(new_pw)
    new_mac = pw_mac(new_pw)
    prev_mac = pw_mac(old_pw) if old_pw else None
    ts = int(time.time())

    def fn(r: dict) -> str:
        if r.get("rotation_month") == month:
            return "ok"
        if r.get("next_month") != month or r.get("next_mac") != new_mac or r.get("draft_month") != month:
            raise RotateError("切り替えの直前に次の版・下書きの印が変わりました。切り替えません")
        old = {k: r.get(k) for k in ("algo", "iterations", "salt", "hash")}
        r["prev"] = {**old, "month": r.get("rotation_month") or prev, "cur_mac": prev_mac}
        r.update(h)
        r["pw_changed_ts"] = ts                      # これ以前のログインは使えなくなる（今と同じ）
        r["rotation_month"] = month
        r["cur_mac"] = new_mac
        if prev_mac:
            r["prev_mac"] = prev_mac
        else:
            r.pop("prev_mac", None)
        for k in FIELDS_NEXT:
            r.pop(k, None)
        r["updated_at"] = _now_iso()
        return "switched"

    res = _transact(uid, fn)
    rec2 = _read_record(uid)
    if res == "switched":
        if not verify_hash(rec2, new_pw):
            raise RotateError("切り替え済みですが、書いた後の照合に失敗しました。rollback を検討してください")
        y, m = month.split("-")
        notify(f"【めぐ競馬】共有ログインのパスワードを {int(y)}年{int(m)}月分に切り替えました（これまでのログインは数分以内に切れます）。"
               + ("" if prev_mac else " 先月の値を確かめられなかったため、先月のパスワードの案内は出ません。"))
    try:
        _catch_up(sm, rec2, _latest(sm)[1], month)
    except Exception as e:  # noqa: BLE001
        why = str(e) if isinstance(e, (RotateError,)) or type(e).__name__ == "SecretError" else type(e).__name__
        raise RotateError(f"切り替え済み（会員のパスワードは新しい値）。Secret Manager の同期だけ失敗しました。Job をもう一度流してください: {why}") from e
    return res


def _catch_up(sm, rec: dict, p: dict, month: str) -> None:
    """名簿が確定した後で、Secret Manager の最新を {"current": 今月, "next": null} にそろえる。"""
    cur = p["current"]
    if cur and cur["month"] == month and _mac_eq(cur["password"], rec.get("cur_mac")) and not p["next"]:
        _cleanup(sm, rec)
        return
    pw = None
    for x in (p["next"], cur):
        if x and _mac_eq(x["password"], rec.get("cur_mac")):
            pw = x["password"]
            break
    if pw is None:
        found = _find_version(sm, lambda q: any(x and _mac_eq(x["password"], rec.get("cur_mac")) for x in (q["current"], q["next"])))
        if not found:
            raise RotateError("名簿は切り替わっていますが、Secret Manager に今月の値が見つかりません")
        pw = next(x["password"] for x in (found["current"], found["next"]) if x and _mac_eq(x["password"], rec.get("cur_mac")))
    sm.add(_dump_payload({"month": month, "password": pw}, None))
    _cleanup(sm, rec)


def rollback(now: datetime, sm, uid: str, notify) -> str:
    """直前の切り替えを戻す（名簿の prev のハッシュに戻す。平文を取り出さない）。"""
    _require_gcs()
    rec = _read_record(uid)
    pv = rec.get("prev")
    if not (isinstance(pv, dict) and pv.get("hash") and pv.get("salt")):
        raise RotateError("戻す先（前のハッシュ）がありません")
    ts = int(time.time())

    def fn(r: dict) -> str:
        q = r.get("prev")
        if not (isinstance(q, dict) and q.get("hash")):
            return "ok"
        for k in ("algo", "iterations", "salt", "hash"):
            r[k] = q.get(k)
        r["rotation_month"] = q.get("month")
        r["cur_mac"] = q.get("cur_mac")
        if not r["cur_mac"]:
            r.pop("cur_mac", None)
        r["pw_changed_ts"] = ts
        for k in ("prev", "prev_mac", *FIELDS_NEXT):
            r.pop(k, None)
        r["updated_at"] = _now_iso()
        return "rolled-back"

    res = _transact(uid, fn)
    if res == "rolled-back":
        rec2 = _read_record(uid)
        _, p = _latest(sm)
        old = None
        if rec2.get("cur_mac"):
            found = _find_version(sm, lambda q: any(x and _mac_eq(x["password"], rec2["cur_mac"]) for x in (q["current"], q["next"])))
            if found:
                old = next(x for x in (found["current"], found["next"]) if x and _mac_eq(x["password"], rec2["cur_mac"]))
            else:                                    # 旧方式（派生の値）から移った初回の切り替えを戻したとき
                from src.scripts.ops import shared_login as legacy

                cand = legacy.password_for(str(rec2.get("rotation_month") or ""))
                if _mac_eq(cand, rec2["cur_mac"]):
                    old = {"month": rec2.get("rotation_month"), "password": cand}
        sm.add(_dump_payload({"month": rec2.get("rotation_month"), "password": old["password"]} if old else None, None))
        notify("【めぐ競馬】共有ログインのパスワードを、先月の値に戻しました（rollback）。note で会員に知らせてください。"
               + ("" if old else " 戻した値を Secret Manager に入れられなかったため、次の自動の切り替えは止まります（手順書の「戻し方」）。"))
    return res


# ── 実行 ──
def _now_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _notify(msg: str) -> bool:
    from src.utils import notify as n

    return n.notify_slack(msg)


def _write_status(step: str, month: str, result: str, error: str = "") -> None:
    """秘密の無い結果のファイル（毎朝の schedule-check が読む）。書けなくても本体は止めない。"""
    try:
        from src.scripts.ops import manage_users as mu

        body = {"step": step, "month": month, "result": result, "error": error, "at": _now_iso(), "ts": int(time.time())}
        mu._backend().upload(STATUS_KEY, json.dumps(body, ensure_ascii=False).encode("utf-8"))
    except Exception:  # noqa: BLE001
        print("status.json を書けませんでした", file=sys.stderr)


def run(now: datetime, step: str | None = None, sm=None, notify=None) -> str:
    step = step or step_for(now)
    if step is None:
        return "skip"
    notify = notify or _notify
    month = month_of(now) if step == "switch" else add_months(month_of(now), 1)
    try:
        sm = sm or _store()
        uid = _shared_id()
        if step == "prepare":
            res = prepare(now, sm, uid)
        elif step == "draft":
            res = draft(now, sm, uid, notify)
        elif step == "switch":
            res = switch(now, sm, uid, notify)
        else:
            raise RotateError(f"不明な役です: {step}")
    except Exception as e:  # noqa: BLE001
        msg = str(e) if isinstance(e, RotateError) else f"{type(e).__name__}（詳細はジョブのログ）"
        _write_status(step, month, "failed", msg)
        head = "は済みましたが、後始末に失敗しました" if msg.startswith("切り替え済み") else "に失敗しました"
        notify(f"【めぐ競馬】共有ログインの{_STEP_JA.get(step, step)}{head}（{month}）: {msg}")
        raise
    _write_status(step, month, res)
    return res


_STEP_JA = {"prepare": "準備", "draft": "下書き", "switch": "切り替え"}


def render_draft(payload: dict, rec: dict, month_now: str) -> str:
    """note に貼る文面（画面に出すだけ）。名簿と照合できたものだけを出す。"""
    target = add_months(month_now, 1)
    nx = payload.get("next")
    if not (nx and nx.get("month") == target and _mac_eq(nx["password"], rec.get("next_mac"))):
        raise RotateError("次の版が名簿と合いません（表示しません）")
    cur = payload.get("current")
    cur_pw = None
    if cur and verify_hash(rec, cur["password"]):                  # 月は比べない（rollback の後は current の月が前の月のまま）
        cur_pw = cur["password"]
    else:
        from src.scripts.ops import shared_login as legacy

        cand = legacy.password_for(month_now)
        if verify_hash(rec, cand):
            cur_pw = cand
    if cur_pw is None:
        raise RotateError("今月の値が名簿と合いません（表示しません）")
    y0, m0 = (int(x) for x in month_now.split("-"))
    y1, m1 = (int(x) for x in target.split("-"))
    return "\n".join([
        f"【めぐ競馬】{y1}年{m1}月の共有ログインのお知らせ",
        "",
        f"ログイン: {LOGIN_URL}",
        f"ID: {_shared_id()}",
        "",
        f"■ {m0}月のパスワード（{m1}月1日 {SWITCH_HOUR}:00 まで）: {cur_pw}",
        f"■ {m1}月のパスワード（{y1}年{m1}月1日 {SWITCH_HOUR}:00 から）: {nx['password']}",
        "",
        "ご注意",
        f"- {m1}月1日 {SWITCH_HOUR}:00 に切り替わり、それまでのログインは数分以内に切れます。新しいパスワードでログインし直してください。",
        f"- 切り替えの後に{m0}月のパスワードを入れると、「パスワードが変わりました」と案内します（失敗の回数には数えません）。",
        "- パスワードを他の人に教えたり、SNS などに書いたりしないでください。",
    ])


def export_pepper() -> str:
    """今のペッパー（DEV_SECRET_KEY から導いた値）を Secret Manager に入れる形（hex:…）で返す。"""
    from src.api import auth

    return "hex:" + auth._pepper().hex()


def _parse_now(s: str | None) -> datetime:
    if not s:
        return datetime.now(JST)
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=JST)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["run", "rollback", "draft", "export-pepper"])
    ap.add_argument("--step", choices=["prepare", "draft", "switch"])
    ap.add_argument("--now", help="試験用の時刻（JST。例 2027-01-01T06:00）")
    a = ap.parse_args(argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    try:
        if a.cmd == "export-pepper":
            if sys.stdout.isatty():
                print("画面には出しません。パイプで put_secret.sh に渡してください", file=sys.stderr)
                return 2
            sys.stdout.write(export_pepper())
            return 0
        if a.cmd == "draft":
            if sys.stdin.isatty():
                print("標準入力に Secret Manager の中身を渡してください（docs の 1 行）", file=sys.stderr)
                return 2
            payload = _parse_payload(sys.stdin.buffer.read())
            _require_gcs()
            print(render_draft(payload, _read_record(_shared_id()), month_of(_parse_now(a.now))))
            return 0
        if a.cmd == "rollback":
            print(rollback(_parse_now(a.now), _store(), _shared_id(), _notify))
            return 0
        print(run(_parse_now(a.now), a.step))
        return 0
    except RotateError as e:
        print(f"止めました: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
