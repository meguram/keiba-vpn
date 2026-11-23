"""VPS のフォルダから返す API の応答を、ホストが手元のデータだけで作る（edge-render）。

docs/operations/edge_cache_plan_20261121.md §3・§10。

- API の app（Cloud Run と同じ FastAPI＋Flask の Gateway）を同じプロセスの中で呼び、応答の本文をファイルにする。
- **GCS を読まない**（CLAUDE.md の費用の上限）。GCS の入力は、ホストが保存のときに写した ``data/local/mirror/``
  （``HybridStorage._maybe_write_local_mirror``。設定は ``data/local/meta/local_mirror_config.json``）だけを読む。
  - ``HybridStorage`` の置き場を写しの ``MirrorStore`` に差し替える（dev のモックと同じ口）。
  - 写しの対象外のカテゴリを読もうとした・GCS のクライアントを作ろうとした・外へ HTTP を出そうとした・保存しようとした、
    のどれかが起きた応答は「印」が立ち、捨てる（ファイルを作らず、VPS は今の Cloud Run の経路に流す）。
  - 写しを有効にした日（``mirror_since``）より後に出馬表が出たレースだけを作る。写しに無い＝GCS にも無い、とみなせるため。
- 区分（会員）は、このプロセスの中だけの使い捨ての鍵で署名したトークンで渡す（auth のコードは変えない）。
  接続元はループバックにしない（開発者の自動ログインを効かせない）。
- 本文の MD5 が前回と同じなら書き換えない（rsync が送らない）。素の .json と .json.gz を並べて置く（nginx の gzip_static）。

使い方（ホスト。PYTHONPATH は作る版のツリー）::

    python -m src.ops.edge_render render --env stg --out /home/jovyan/work/keiba-edge/out [--race RID ...] [--date YYYYMMDD ...]
    python -m src.ops.edge_render prune --mirror-days 40     # 40 日より古い写しを消す
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

JST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "edge_render.json"
SYNTH_UID = "edge-render"
# 環境から引き継ぐのは OS の基本だけ（ホストの .env・cron の KEIBA_* を持ち込まない）
_KEEP_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "PYTHONPATH", "PYTHONPYCACHEPREFIX", "TMPDIR", "USER", "LOGNAME",
             "KEIBA_EDGE_MIRROR_DIR", "KEIBA_PAGE_REFERENCE_DIR")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_env(cfg: dict[str, Any], env: str) -> dict[str, str]:
    """作る処理の環境変数（Cloud Run の本番・stg の値）。空の値は「未設定」。"""
    e = cfg["env"]
    out = {**e.get("common", {}), **e.get(env, {})}
    return {k: str(v) for k, v in out.items() if not k.startswith("_")}


def apply_clean_environ(cfg: dict[str, Any], env: str, secret: str) -> None:
    """os.environ を作り直す。API を import する前に呼ぶ。"""
    keep = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    for k, v in render_env(cfg, env).items():
        if v != "":
            os.environ[k] = v
    os.environ.update({
        "KEIBA_SKIP_DOTENV": "1",          # ホストの .env を読まない
        # 無いバケット名。GCS_BUCKET の有無で分岐する口（予測成績・gcs_io など）も GCS のクライアントを作りに行き、守りに当たる。
        # HybridStorage は init の差し替えで空に戻し、置き場は MirrorStore にする
        "GCS_BUCKET": "edge-render-no-gcs",
        "KEIBA_DEV_AUTO_LOGIN": "0",
        "KEIBA_PROXY_KEY_REQUIRED": "0",   # 同じプロセスの中で呼ぶだけ（外の口は開かない）
        "KEIBA_PRERENDER": "0",
        "KEIBA_PREWARM": "0",
        "KEIBA_MIRROR_PULL": "0",
        "KEIBA_SERVING_STATE_PULL": "0",
        "KEIBA_WARMUP_ON_START": "0",
        "KEIBA_MAINTENANCE_WINDOW": "0",   # 保守の時間（03:00〜04:00）に作っても 503 にしない
        "KEIBA_CLOUDRUN_LAZY": "1",        # cloudrun_entry の import で Cloud Run の app（取り込み・起動処理）を組まない
        "DEV_SECRET_KEY": secret,          # このプロセスだけの使い捨ての鍵
    })


# ---------------------------------------------------------------- 印（応答を捨てる理由）

class Flags:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def hit(self, reason: str) -> None:
        if len(self.reasons) < 200:
            self.reasons.append(reason)

    def clear(self) -> None:
        self.reasons = []


FLAGS = Flags()


class EdgeGuardError(RuntimeError):
    pass


def _blocked(what: str):
    def f(*a, **k):
        FLAGS.hit(what)
        raise EdgeGuardError(f"edge-render では使えない: {what}")
    return f


def mirrored_categories(base: Path) -> set[str]:
    try:
        c = json.loads((base / "data" / "local" / "meta" / "local_mirror_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(x) for x in (c.get("categories") or [])} if c.get("enabled") else set()


def make_mirror_store(base: Path, cats: set[str], fallback: dict[str, str] | None = None):
    from src.scraper.dev_store import DevStore

    fb = {c: DevStore(base / rel) for c, rel in (fallback or {}).items()}

    class MirrorStore(DevStore):
        """ホストの写し（data/local/mirror）。対象外のカテゴリは「分からない」なので印を立てる。書き込みは禁止。

        レースのキーは、対象の日を絞ってあるので「写しに無い＝GCS にも無い」。馬・その他のキーは、写しの有効化より前に
        保存されたきり（写しに無い）かもしれないので、無ければ印を立てる。"""

        def _ok(self, category: str) -> bool:
            if category in cats or category in fb:
                return True
            FLAGS.hit(f"写しの対象外: {category}")
            return False

        def _path_any(self, category, key, id_type) -> Path | None:
            if category in cats:
                p = self.path(category, key, id_type)
                if p.is_file():
                    return p
            if category in fb:
                p = _flat_path(fb[category], key, id_type)
                if p.is_file():
                    return p
            return None

        def _miss(self, category, id_type) -> None:
            if id_type != "race":
                FLAGS.hit(f"写しに無い: {category}")

        def read(self, category, key, id_type):
            if not self._ok(category):
                return None
            p = self._path_any(category, key, id_type)
            if p is None:
                self._miss(category, id_type)
                return None
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                FLAGS.hit(f"写しが読めない: {category}")
                return None

        def exists(self, category, key, id_type):
            if not self._ok(category):
                return False
            if self._path_any(category, key, id_type) is None:
                self._miss(category, id_type)
                return False
            return True

        def list_keys(self, category, id_type, year=None):
            if not self._ok(category):
                return []
            if id_type != "race" or category not in cats:
                FLAGS.hit(f"一覧（馬・別の置き場）: {category}")      # 写しは保存された分だけなので、馬の一覧は欠ける
                return []
            return super().list_keys(category, id_type, year)

        def mtimes(self, category, id_type, keys):
            if not self._ok(category):
                return {}
            out: dict[str, float] = {}
            for k in keys:
                p = self._path_any(category, k, id_type)
                if p is None:
                    self._miss(category, id_type)
                    continue
                out[k] = p.stat().st_mtime
            return out

        def write(self, category, key, id_type, data):
            FLAGS.hit(f"保存: {category}")
            raise EdgeGuardError("edge-render は保存しない")

    return MirrorStore(mirror_root(base))


def _flat_path(store, key: str, id_type: str) -> Path:
    """別の置き場は <root>/<shard>/<key>.json（カテゴリの段が無い）。"""
    shard = key[:4] if len(key) >= 4 else ("_" if id_type == "horse" else "unknown")
    return store.root / shard / f"{key}.json"


class _WarnToFlag:
    """作る間の WARNING 以上のログを印にする（API の付与の失敗は握りつぶされ、欠けた 200 になるため）。"""

    def __new__(cls):
        import logging

        class H(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    FLAGS.hit(f"WARNING: {record.name}: {record.getMessage()[:120]}")

        return H(level=logging.WARNING)


def install_guards(base: Path) -> None:
    """GCS・外への通信・保存を塞ぎ、HybridStorage の置き場を写しにする。API を import する前に呼ぶ。"""
    import socket

    from google.cloud import storage as gcs

    gcs.Client.__init__ = _blocked("GCS のクライアント")
    orig_connect = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else ""
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in ("127.0.0.1", "::1", "localhost"):
            FLAGS.hit(f"外への通信: {host}")
            raise EdgeGuardError(f"edge-render は外へ通信しない: {host}")
        return orig_connect(self, address)

    socket.socket.connect = connect
    socket.create_connection = _blocked("外への通信（create_connection）")
    try:
        import requests

        requests.Session.request = _blocked("外への HTTP（requests）")
    except ImportError:
        pass
    import urllib.request

    urllib.request.urlopen = _blocked("外への HTTP（urllib）")

    from src.scraper import storage as st

    cats = mirrored_categories(base)
    fallback = dict(load_config().get("local_fallback") or {})
    fallback.pop("_comment", None)
    orig_init = st.HybridStorage.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._bucket_name = ""
        self._dev_store = make_mirror_store(base, cats, fallback)

    st.HybridStorage.__init__ = init
    # バケット名が空だと _dev_store を通らずに空を返す口（印にする）
    st.HybridStorage.exists_gcs = _blocked("exists_gcs")
    st.HybridStorage.get_scraped_at_from_blob = _blocked("get_scraped_at_from_blob")
    st.HybridStorage.list_years = _blocked("list_years")
    # 週ごとの参照回数（data/local/meta/disk_l2_weekly_access.json）を書かない（ホストの状態に触れない）
    st.HybridStorage._bump_disk_hot_l2_access = lambda self, category, key: 0
    st.HybridStorage._persist_disk_hot_l2_state_unlocked = lambda self: None
    st.HybridStorage._load_disk_hot_l2_state_initial = lambda self: None
    orig_note = st.note_degraded

    def note():
        FLAGS.hit("note_degraded")
        orig_note()

    st.note_degraded = note
    import logging

    logging.getLogger().addHandler(_WarnToFlag())


def assert_storages_guarded() -> None:
    from src.scraper import storage as st

    bad = [s for s in list(st.HybridStorage._instances) if type(s._dev_store).__name__ != "MirrorStore" or s._bucket_name]
    if bad:
        raise EdgeGuardError(f"写しに向いていない HybridStorage が {len(bad)} 個ある")


def install_member(auth_mod) -> dict[str, str]:
    """区分ごとの Cookie。会員は使い捨ての鍵で署名した、名簿に無い合成の ID（このプロセスの中だけで有効）。"""
    orig_rec, orig_active = auth_mod.member_record, auth_mod.member_active
    auth_mod.member_record = lambda u: {"plan": "paid", "pw_changed_ts": 0} if u == SYNTH_UID else orig_rec(u)
    auth_mod.member_active = lambda u: True if u == SYNTH_UID else orig_active(u)
    return {
        "anon": "",
        "member": auth_mod._make_token(role=auth_mod.ROLE_MEMBER, uid=SYNTH_UID),
    }


def build_gateway():
    """Cloud Run と同じ Gateway（FastAPI＋Flask）。取り込み・起動処理は付けない。"""
    from src.api.app import app as fastapi_app
    from src.api.cloudrun_entry import Gateway, _wsgi_bridge
    from src.api.flask_app import create_app

    return Gateway(fastapi_app, _wsgi_bridge(create_app()), None)


# ---------------------------------------------------------------- 対象のレース

def race_lists_dir(base: Path) -> Path:
    o = os.environ.get("KEIBA_PAGE_REFERENCE_DIR", "").strip()
    return (Path(o) if o else base / "data" / "page_reference") / "race_lists"


def races_of(base: Path, date: str) -> list[str]:
    try:
        d = json.loads((race_lists_dir(base) / f"{date}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(r.get("race_id")) for r in (d.get("races") or []) if str(r.get("race_id", "")).isdigit()]


def eligible_dates(cfg: dict[str, Any], today: datetime) -> list[str]:
    """写しが揃っている（出馬表が写しの有効化より後）かつ確定済み（開催から grace 日）で、直近 window 日の開催日。"""
    since = datetime.strptime(cfg["mirror_since"], "%Y%m%d").replace(tzinfo=JST) + timedelta(days=int(cfg.get("entries_lead_days", 5)))
    last = today - timedelta(days=int(cfg.get("settled_grace_days", 1)))
    first = max(since, today - timedelta(days=int(cfg.get("window_days", 35))))
    out = []
    d = first
    while d.date() <= last.date():
        out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def mirror_root(base: Path) -> Path:
    o = os.environ.get("KEIBA_EDGE_MIRROR_DIR", "").strip()
    return Path(o) if o else base / "data" / "local" / "mirror"


def in_mirror(base: Path, category: str, rid: str) -> bool:
    return (mirror_root(base) / category / rid[:4] / f"{rid}.json").is_file()


def race_settled_in_mirror(base: Path, rid: str) -> bool:
    """写しに出馬表と結果がある（確定済みで、写しの有効化の後に保存された）レース。"""
    return in_mirror(base, "race_shutuba", rid) and in_mirror(base, "race_result", rid)


# ---------------------------------------------------------------- 入力の指紋（変わっていなければ作り直さない）

def _stat_sig(p: Path) -> str:
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "-"


def race_fingerprint(base: Path, rid: str, cats: Iterable[str]) -> str:
    """そのレースの写し（レースのカテゴリ）の更新時刻と大きさ。推論の出力（data/local/inference）などは含まない → 深夜の --full で拾う。"""
    root = mirror_root(base)
    return hashlib.md5("|".join(_stat_sig(root / c / rid[:4] / f"{rid}.json") for c in sorted(cats)).encode()).hexdigest()


def date_fingerprint(base: Path, date: str, cats: Iterable[str]) -> str:
    parts = [_stat_sig(race_lists_dir(base) / f"{date}.json")] + [race_fingerprint(base, r, cats) for r in races_of(base, date)]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def global_fingerprint(cfg: dict[str, Any], env: str, sha: str) -> str:
    return hashlib.md5(json.dumps([sha, env, cfg], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def target_fingerprints(cfg: dict[str, Any], base: Path, env: str, sha: str, targets: list[tuple[str, str]]) -> dict[str, str]:
    cats = sorted(mirrored_categories(base))
    g = global_fingerprint(cfg, env, sha)
    out: dict[str, str] = {}
    for tier, path in targets:
        key = target_key(path)
        f = date_fingerprint(base, key, cats) if len(key) == 8 else race_fingerprint(base, key, cats)
        out[out_rel(tier, path)] = hashlib.md5(f"{g}|{f}".encode()).hexdigest()
    return out


# ---------------------------------------------------------------- 書き出し

def out_rel(tier: str, path: str) -> str:
    return f"{tier}{path}.json"


def write_if_changed(out_dir: Path, rel: str, body: bytes, manifest: dict[str, str]) -> bool:
    md5 = hashlib.md5(body).hexdigest()
    p = out_dir / rel
    if manifest.get(rel) == md5 and p.is_file() and p.with_name(p.name + ".gz").is_file():
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    for target, data in ((p.with_name(p.name + ".gz"), gzip.compress(body, compresslevel=9, mtime=0)), (p, body)):
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    manifest[rel] = md5
    return True


def tree_sha(base: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(base), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()[:12]
    except Exception:  # noqa: BLE001
        return "unknown"


async def render_all(app, targets: list[tuple[str, str]], cookies: dict[str, str], out_dir: Path, manifest: dict[str, str],
                     fps: dict[str, str] | None = None, prev_fps: dict[str, str] | None = None) -> dict[str, Any]:
    import httpx

    from src.api.auth import COOKIE_NAME

    stats: dict[str, Any] = {"written": 0, "same": 0, "skipped": 0, "unchanged_input": 0, "reasons": {}, "skipped_targets": []}
    fps = fps or {}
    prev_fps = prev_fps if prev_fps is not None else {}
    transport = httpx.ASGITransport(app=app, client=("192.0.2.10", 40000))   # ループバックにしない
    async with httpx.AsyncClient(transport=transport, base_url="http://edge-render", timeout=300) as c:
        for tier, path in targets:
            rel = out_rel(tier, path)
            fp = fps.get(rel)
            if fp and prev_fps.get(rel) == fp and rel in manifest and (out_dir / rel).is_file():
                stats["unchanged_input"] += 1
                continue
            FLAGS.clear()
            c.cookies.clear()
            if cookies.get(tier):
                c.cookies.set(COOKIE_NAME, cookies[tier])
            try:
                r = await c.get(path, headers={"X-Keiba-Fresh": "1", "Accept-Encoding": "identity"})
                status, body = r.status_code, r.content
            except Exception as e:  # noqa: BLE001
                FLAGS.hit(f"例外: {type(e).__name__}")
                status, body = 0, b""
            reason = None
            if FLAGS.reasons:
                reason = " / ".join(sorted(set(FLAGS.reasons)))
            elif status != 200:
                reason = f"status {status}"
            if reason:
                stats["skipped"] += 1
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                stats["skipped_targets"].append((tier, path))
                continue
            if fp:
                prev_fps[rel] = fp
            if write_if_changed(out_dir, rel, body, manifest):
                stats["written"] += 1
            else:
                stats["same"] += 1
    return stats


def targets_for(cfg: dict[str, Any], base: Path, dates: Iterable[str], races: Iterable[str], tiers: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    rids: list[str] = [r for r in races if race_settled_in_mirror(base, r)]
    for d in dates:
        all_day = races_of(base, d)
        day = [r for r in all_day if race_settled_in_mirror(base, r)]
        rids.extend(day)
        if not day or len(day) != len(all_day):       # その日の全レースが写しに揃っていないと、一覧（race-list）が欠ける
            continue
        for t in tiers:
            out.extend((t, p.format(date=d)) for p in cfg["date_paths"].get(t, []))
    for rid in dict.fromkeys(rids):
        for t in tiers:
            out.extend((t, p.format(rid=rid)) for p in cfg["race_paths"].get(t, []))
    return out


def cmd_render(args: argparse.Namespace) -> int:
    import asyncio

    cfg = load_config()
    if not cfg.get("enabled"):
        print("config/edge_render.json の enabled が false なので何もしない")
        return 0
    base = Path(args.base).resolve() if args.base else ROOT
    apply_clean_environ(cfg, args.env, secrets.token_hex(32))
    sys.path.insert(0, str(ROOT))
    install_guards(base)
    from src.api import auth

    cookies = install_member(auth)
    tiers = args.tier or list(cfg.get("tiers") or ["anon"])
    today = datetime.now(JST)
    dates = args.date or ([] if args.race else eligible_dates(cfg, today))
    targets = targets_for(cfg, base, dates, args.race or [], tiers)
    sha = tree_sha(ROOT)
    out_dir = Path(args.out) / args.env / sha
    man_path = out_dir / "_meta" / "manifest.json"
    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    fp_path = out_dir / "_meta" / "fingerprints.json"
    try:
        prev_fps = {} if args.full else json.loads(fp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prev_fps = {}
    fps = target_fingerprints(cfg, base, args.env, sha, targets)
    t0 = time.time()
    app = build_gateway()
    assert_storages_guarded()
    stats = asyncio.run(render_all(app, targets, cookies, out_dir, manifest, fps, prev_fps))
    # 捨てた対象があるレース・日は、同じ回のほかの対象（プロセスの中のキャッシュで欠けを持ち越しうる）と前の回のファイルも消す。
    # VPS からも push で消え、今の Cloud Run の経路に戻る（印が立ったら Cloud Run に流す）
    dirty = dirty_keys(base, targets, stats.pop("skipped_targets"))
    stats["dirty_keys"] = len(dirty)
    stats["dropped"] = prune_outputs(out_dir, manifest, prev_fps, {out_rel(t, p) for t, p in targets if target_key(p) not in dirty}, only={out_rel(t, p) for t, p in targets})
    if not args.race and not args.date:
        # 範囲（直近 window 日・写しに揃ったレース）を外れたものは消す（VPS からも push で消える → Cloud Run の経路に戻る）
        stats["removed"] = prune_outputs(out_dir, manifest, prev_fps, {out_rel(t, p) for t, p in targets})
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    fp_path.write_text(json.dumps(prev_fps, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    hb = {"generated_at": datetime.now(JST).isoformat(timespec="seconds"), "env": args.env, "sha": sha, "targets": len(targets), **stats,
          "seconds": round(time.time() - t0, 1)}
    (out_dir / "_meta" / "heartbeat.json").write_text(json.dumps(hb, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hb, ensure_ascii=False))
    return 0


def target_key(path: str) -> str:
    """対象の鍵（race_id 12 桁か開催日 8 桁）。"""
    return next((x for x in path.strip("/").split("/") if x.isdigit()), "")


def dirty_keys(base: Path, targets: list[tuple[str, str]], skipped: list[tuple[str, str]]) -> set[str]:
    bad = {target_key(p) for _, p in skipped}
    for _, p in targets:
        k = target_key(p)
        if len(k) == 8 and k not in bad and any(r in bad for r in races_of(base, k)):
            bad.add(k)
    return bad


def prune_outputs(out_dir: Path, manifest: dict[str, str], fps: dict[str, str], want: set[str], only: set[str] | None = None) -> int:
    """manifest のうち want に無いものを消す（only を渡したら、その中だけ）。指紋も消すので、次の回に作り直す。"""
    n = 0
    for rel in [r for r in list(manifest) + list(fps) if r not in want and (only is None or r in only)]:
        if rel not in manifest and rel not in fps:
            continue
        for q in (out_dir / rel, out_dir / (rel + ".gz")):
            q.unlink(missing_ok=True)
        if manifest.pop(rel, None) is not None:
            n += 1
        fps.pop(rel, None)
    return n


_REMOTE_APPLY = r"""
import json, os, shutil, sys
root, inc = sys.argv[1], sys.argv[2]
dele = json.load(open(os.path.join(inc, "_delete.json")))
keep_p = os.path.join(inc, "_keep.json")
keep = set(json.load(open(keep_p))) if os.path.exists(keep_p) else None
for dp, _, fs in os.walk(inc):
    for f in fs:
        if f in ("_delete.json", "_keep.json"):
            continue
        src = os.path.join(dp, f)
        dst = os.path.join(root, os.path.relpath(src, inc))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
for rel in dele:
    for q in (rel, rel + ".gz"):
        try:
            os.unlink(os.path.join(root, q))
        except FileNotFoundError:
            pass
if keep is not None:      # 送り直し: 手元の記録に無いファイル（VPS にだけ残ったもの）を消す。VPS の中だけを歩く
    for dp, _, fs in os.walk(root):
        rel_dir = os.path.relpath(dp, root)
        if rel_dir == "_meta" or rel_dir.startswith("_meta" + os.sep):
            continue
        for f in fs:
            rel = os.path.normpath(os.path.join(rel_dir, f))
            base = rel[:-3] if rel.endswith(".gz") else rel
            if base.endswith(".json") and base not in keep:
                os.unlink(os.path.join(dp, f))
shutil.rmtree(inc, ignore_errors=True)
"""


def plan_push(manifest: dict[str, str], sent: dict[str, str]) -> tuple[list[str], list[str]]:
    """送るもの（新しい・変わった）と、VPS から消すもの（手元の記録から消えた）。GCS もリモートの一覧も使わない。"""
    send = sorted(r for r, m in manifest.items() if sent.get(r) != m)
    delete = sorted(r for r in sent if r not in manifest)
    return send, delete


def remote_script(env: str, sha: str) -> str:
    import shlex

    rdir = f"/srv/keiba-edge/{env}"
    vdir = f"{rdir}/{sha}"
    inc = f"{rdir}/.incoming-{sha}"
    return (
        f"set -e; umask 027; rm -rf {inc}; mkdir -p {inc} {vdir}; tar -xzf - -C {inc}; "
        f"python3 -c {shlex.quote(_REMOTE_APPLY)} {vdir} {inc}; "
        f"cd {rdir}; ln -sfn {sha} .current.tmp; mv -T .current.tmp current; "
        f"ls -1dt -- */ | sed 's:/$::' | grep -E '^[0-9a-f]{{12}}$' | grep -v '^{sha}$' | tail -n +2 | xargs -r rm -rf --"
    )


def cmd_push(args: argparse.Namespace) -> int:
    """作った版の置き場を VPS へ送る（ssh・tar。ホストに rsync は無い）。VPS では 1 ファイルずつ os.replace で置き換え、最後に current を付け替える。"""
    import io
    import tarfile

    if not (len(args.sha) == 12 and all(c in "0123456789abcdef" for c in args.sha)):
        print("--sha は 12 桁の 16 進", file=sys.stderr)
        return 2
    out_dir = Path(args.out) / args.env / args.sha
    meta = out_dir / "_meta"
    manifest = json.loads((meta / "manifest.json").read_text(encoding="utf-8"))
    sent_path = meta / f"sent-{args.remote}.json"
    try:
        sent = {} if args.resync else json.loads(sent_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        sent = {}
    send, delete = plan_push(manifest, sent)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in send:
            for q in (rel, rel + ".gz"):
                tf.add(out_dir / q, arcname=q)
        tf.add(meta / "heartbeat.json", arcname="_meta/heartbeat.json")
        extra = {"_delete.json": delete}
        if args.resync:
            extra["_keep.json"] = sorted(manifest)
        for name, obj in extra.items():
            data = json.dumps(obj).encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    r = subprocess.run(["ssh", args.remote, remote_script(args.env, args.sha)], input=buf.getvalue(), capture_output=True)
    if r.returncode != 0:
        print(f"送れなかった: {r.stderr.decode(errors='replace')[-500:]}", file=sys.stderr)
        return 1
    for rel in send:
        sent[rel] = manifest[rel]
    for rel in delete:
        sent.pop(rel, None)
    sent_path.write_text(json.dumps(sent, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"sent": len(send), "deleted": len(delete), "bytes": len(buf.getvalue()), "sha": args.sha}, ensure_ascii=False))
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    """写し（data/local/mirror）のうち、更新から mirror_days 日を過ぎたものを消す。"""
    base = Path(args.base).resolve() if args.base else ROOT
    root = base / "data" / "local" / "mirror"
    cut = time.time() - args.mirror_days * 86400
    n = 0
    for p in root.rglob("*.json") if root.is_dir() else []:
        try:
            if p.stat().st_mtime < cut:
                p.unlink()
                n += 1
        except OSError:
            continue
    print(f"写しを {n} 件消した（{args.mirror_days} 日より古い）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--env", choices=("stg", "prod"), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--base", default="", help="データの置き場のルート（data/local・data/page_reference）。既定はこのツリー")
    r.add_argument("--date", action="append", help="開催日（YYYYMMDD）。省略時は設定の範囲")
    r.add_argument("--race", action="append", help="race_id（12 桁）")
    r.add_argument("--tier", action="append", choices=("anon", "member"))
    r.add_argument("--full", action="store_true", help="入力の指紋を見ず全部作り直す（深夜。推論の出力の変化を拾う）")
    u = sub.add_parser("push")
    u.add_argument("--env", choices=("stg", "prod"), required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--sha", required=True)
    u.add_argument("--remote", default="keiba-vps")
    u.add_argument("--resync", action="store_true", help="送った記録を捨てて全部送り、VPS にだけあるファイルを消す（深夜。VPS の中身との食い違いを直す）")
    p = sub.add_parser("prune")
    p.add_argument("--base", default="")
    p.add_argument("--mirror-days", type=int, default=40, help="作る窓（35 日）＋出馬表が先に出る日数より長く")
    a = ap.parse_args(argv)
    return {"render": cmd_render, "push": cmd_push, "prune": cmd_prune}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
