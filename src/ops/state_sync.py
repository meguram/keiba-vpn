"""ジョブの入出力ファイル（``data/…`` 以下の相対パス）を GCS の ``state/`` と同期する。

Cloud Run Jobs のコンテナは毎回空で、ディスクはメモリ上にある。そこで job_map の ``state_in`` を実行前に ``pull``、
``state_out`` を実行後に ``push`` する。**合計サイズに上限（``max_mb``）を付け、超えたら取らずに失敗させる**（メモリを使い切らないため）。

- ``state/`` は GCP ジョブ専用の置き場で、ホストが使う本番データ（GCS の構造化 JSON）には触れない。
- 並走期間は、ホストが同じ成果物を ``host_snapshot/<日付>/`` に上げ（``push --prefix host_snapshot/<日付>``）、``parity_check`` が突き合わせる。
- ホストで ``plan`` を実行すると、各パスの実サイズを job_map の上限と並べて出す（見積りの確認用）。

- 項目に ``"bundle": "<名前>"`` を付けると、その項目（path ＋ include に合うファイル）は 1 つの束
  ``<prefix>/_bundle/<path>/<名前>.tar.gz`` で読み書きする。数万ファイルの木（例: 馬ごとの成長曲線 約 5.8 万件）を
  ファイル単位で読み書きすると、1 回で数万回の GCS 操作になる（書き込みは Class A で 1 回約 46 円）。束なら読み 1 回・書き 1 回。

CLI: ``python -m src.ops.state_sync plan|pull|push --job <job_id> [--root .] [--prefix state]``
並走期間のホスト側: ``python -m src.ops.state_sync snapshot [--date YYYYMMDD] [--root .]``（job_map の各ジョブの parity パスのうち、ホストに実在するものを
``host_snapshot/<日付>/`` に上げる。cron に 1 行足す: ``scripts/cron/install_host_snapshot_cron.sh``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.ops.gcs_io import BlobIO, file_md5_b64, list_meta

DEFAULT_PREFIX = "state"
MB = 1024 * 1024


class StateBudgetError(RuntimeError):
    """state の合計が上限を超えた。"""


def expand(entries: list[dict[str, Any]], params: dict[str, str] | None = None) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for e in entries or []:
        path = str(e["path"])
        if params:
            for k, v in params.items():
                path = path.replace("{" + k + "}", v)
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"state のパスが不正: {path}")
        out.append((path.strip("/"), int(e.get("max_mb", 0))))
    return out


def includes(entries: list[dict[str, Any]], params: dict[str, str] | None = None) -> list[list[str] | None]:
    """expand() と同じ順の、エントリごとの include（``path`` からの相対パスの glob。省略 = すべて）。
    大きい表のうち、そのジョブが実際に読むファイルだけを取る・上げるために使う（例: ``"*/race_result_flat.parquet"``）。"""
    out: list[list[str] | None] = []
    for e in entries or []:
        inc = e.get("include")
        if inc and params:                                   # include の glob にも {year} などを使える（path と同じ置換）
            # {recent_day} は直近の日付（params["recent_days"]。カンマ区切り）ごとに 1 つずつの glob に広げる。
            # 日付フォルダの木を丸ごと取ると 1 回で数千件の読み出しになる（費用の上限: CLAUDE.md）ので、読む日だけを取る
            days = [d for d in str(params.get("recent_days") or "").split(",") if d]
            inc = [x for g in inc for x in ([str(g).replace("{recent_day}", d) for d in days] if "{recent_day}" in str(g) else [str(g)])]
            inc = [_fill(str(x), params) for x in inc]
        out.append([str(x) for x in inc] if inc else None)
    return out


def _fill(text: str, params: dict[str, str]) -> str:
    for k, v in params.items():
        text = text.replace("{" + k + "}", v)
    return text


def _match(rel_in_dir: str, inc: list[str] | None) -> bool:
    import fnmatch

    return inc is None or any(fnmatch.fnmatch(rel_in_dir, g) for g in inc)


def _files(root: Path, rel: str, inc: list[str] | None = None) -> list[Path]:
    p = root / rel
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    return sorted(f for f in p.rglob("*") if f.is_file() and _match(f.relative_to(p).as_posix(), inc))


def plan(entries: list[dict[str, Any]], root: Path, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
    rows = []
    for (rel, max_mb), inc in zip(expand(entries, params), includes(entries, params)):
        files = _files(root, rel, inc)
        mb = sum(f.stat().st_size for f in files) / MB
        rows.append({"path": rel, "exists": bool(files), "files": len(files), "local_mb": round(mb, 1), "max_mb": max_mb,
                     "over": bool(max_mb and mb > max_mb)})
    return rows


def bundle_key(prefix: str, rel: str, name: str) -> str:
    return f"{prefix}/_bundle/{rel}/{name}.tar.gz"


def _bundles(entries: list[dict[str, Any]]) -> list[str | None]:
    """expand() と同じ順の、項目ごとの束の名前（無ければ None）。"""
    out: list[str | None] = []
    for e in entries or []:
        b = e.get("bundle")
        if b is not None and (not isinstance(b, str) or not b or "/" in b or ".." in b):
            raise ValueError(f"state の bundle 名が不正: {b!r}")
        out.append(b or None)
    return out


def _pull_bundle(io: BlobIO, key: str, root: Path, rel: str, max_mb: int) -> tuple[int, int]:
    """束を取って root/rel の下に展開する。(展開したファイル数, 束の大きさ)。束が無ければ (0, 0)（初回）。"""
    import tarfile
    import tempfile

    meta = list_meta(io, key)
    if key not in meta:
        return 0, 0
    size = meta[key][0]
    root.mkdir(parents=True, exist_ok=True)
    dest_root = (root / rel).resolve()
    with tempfile.TemporaryDirectory(dir=root) as td:
        tmp = Path(td) / "b.tar.gz"
        if not io.download_file(key, tmp):
            return 0, 0
        n = 0
        total = 0
        # 1 回の走査（ストリーム）で検査と展開をする（getmembers の後に展開すると gzip を 2 回伸長する）。
        # 不正なパス・上限超過は、その時点で止めて失敗させる（途中まで展開した分は、どうせジョブごと失敗する）
        with tarfile.open(tmp, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                target = (dest_root / m.name).resolve()
                if m.name.startswith("/") or not str(target).startswith(str(dest_root) + "/"):
                    raise ValueError(f"束 {key} に不正なパス: {m.name}")
                total += m.size
                if max_mb and total > max_mb * MB:
                    raise StateBudgetError(f"state {rel}（束 {key}）: 展開後の大きさが上限 {max_mb}MB を超える")
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, target.open("wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                n += 1
    return n, size


def _push_bundle(io: BlobIO, key: str, root: Path, rel: str, files: list[Path]) -> tuple[bool, int]:
    """files を 1 つの束にして上げる。GCS に同じ中身（md5）があれば上げない。(上げたか, 束の大きさ)。"""
    import gzip
    import tarfile
    import tempfile

    base = root / rel
    with tempfile.TemporaryDirectory(dir=root) as td:
        tmp = Path(td) / "b.tar.gz"
        # 中身が同じなら同じバイト列になるように（md5 で書き込みを省くため）、順序・時刻・所有者を固定する
        with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
            for f in sorted(files):
                ti = tf.gettarinfo(str(f), arcname=f.relative_to(base).as_posix())
                ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
                with f.open("rb") as fh:
                    tf.addfile(ti, fh)
        size = tmp.stat().st_size
        have = list_meta(io, key).get(key)
        if have and have[0] == size and have[1] == file_md5_b64(tmp):
            return False, size
        io.upload_file(tmp, key)
        return True, size


def pull(io: BlobIO, entries: list[dict[str, Any]], root: Path, *, prefix: str = DEFAULT_PREFIX,
         params: dict[str, str] | None = None) -> dict[str, Any]:
    """GCS → ローカル。上限超過は何も取らずに ``StateBudgetError``。既にある同サイズのファイルは取り直さない。

    サイズは一覧 1 回で取る（キーごとの問い合わせは、ファイル数 × 2 回の GCS 操作になり、起動のたびに費用がかかる）。
    """
    plan_rows = []
    files = skipped = 0
    total_all = 0
    bundled = 0
    for (rel, max_mb), inc, bname in zip(expand(entries, params), includes(entries, params), _bundles(entries)):
        if bname:
            n, size = _pull_bundle(io, bundle_key(prefix, rel, bname), root, rel, max_mb)
            files += n
            bundled += n
            total_all += size
            continue
        meta = list_meta(io, f"{prefix}/{rel}")
        keys = [k for k in meta if k == f"{prefix}/{rel}" or (k.startswith(f"{prefix}/{rel}/") and _match(k[len(prefix) + len(rel) + 2:], inc))]
        total = sum(meta[k][0] for k in keys)
        if max_mb and total > max_mb * MB:
            raise StateBudgetError(f"state {rel}: {total / MB:.0f}MB が上限 {max_mb}MB を超える（メモリ上のディスクを使い切る）")
        plan_rows.append((rel, [(k, meta[k][0]) for k in keys], total))
    for rel, keys, total in plan_rows:
        for k, size in keys:
            dest = root / k[len(prefix) + 1:]
            if dest.is_file() and dest.stat().st_size == size:
                skipped += 1
                continue
            if io.download_file(k, dest):
                files += 1
                total_all += size
    out = {"downloaded": files, "skipped_same": skipped, "mb": round(total_all / MB, 1)}
    if bundled:
        out["from_bundles"] = bundled
    return out


def push(io: BlobIO, entries: list[dict[str, Any]], root: Path, *, prefix: str = DEFAULT_PREFIX,
         since: float | None = None, params: dict[str, str] | None = None) -> dict[str, Any]:
    """ローカル → GCS。``since``（epoch 秒）以降に更新されたファイルだけ上げる。上限超過は ``StateBudgetError``。

    GCS に同じ中身（サイズと md5 が一致）があるファイルは上げない（書き込みの回数と転送を減らす）。
    """
    todo: list[tuple[Path, str]] = []
    remote: dict[str, tuple[int, str]] = {}
    n = up = skipped = 0
    for (rel, max_mb), inc, bname in zip(expand(entries, params), includes(entries, params), _bundles(entries)):
        files = _files(root, rel, inc)
        mb = sum(f.stat().st_size for f in files)
        if max_mb and mb > max_mb * MB:
            raise StateBudgetError(f"state {rel}: {mb / MB:.0f}MB が上限 {max_mb}MB を超える（上限を見直すか出力を絞る）")
        if bname:
            # 束は木全体を 1 つにする（取り込んだ分も含める）。更新が 1 つも無ければ上げない
            if files and (since is None or any(f.stat().st_mtime >= since for f in files)):
                did, size = _push_bundle(io, bundle_key(prefix, rel, bname), root, rel, files)
                n, up, skipped = n + int(did), up + (size if did else 0), skipped + int(not did)
            continue
        files = [f for f in files if since is None or f.stat().st_mtime >= since]
        if files:
            remote.update(list_meta(io, f"{prefix}/{rel}"))
        todo += [(f, f"{prefix}/{f.relative_to(root).as_posix()}") for f in files]
    for f, key in todo:
        size = f.stat().st_size
        have = remote.get(key)
        if have and have[0] == size and have[1] == file_md5_b64(f):
            skipped += 1
            continue
        io.upload_file(f, key)
        n += 1
        up += size
    return {"uploaded": n, "skipped_same": skipped, "mb": round(up / MB, 1)}


def snapshot(io: BlobIO, doc: dict[str, Any], root: Path, date: str) -> list[dict[str, Any]]:
    """ホストの成果物（parity パス）を host_snapshot/<date>/ に上げる。ホストに無いパスは飛ばす（理由つきで返す）。"""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.ops import job_dispatch, job_map

    params = job_dispatch.build_params(datetime.strptime(date, "%Y%m%d").replace(hour=7, tzinfo=ZoneInfo("Asia/Tokyo")))
    out = []
    for j in job_map.by_destination("gcp", doc):
        par = j.get("parity") or {}
        if par.get("kind", "none") == "none" or not j.get("enabled", True):
            continue
        entries = [{"path": p, "max_mb": 0} for p in par.get("paths") or []]
        present = [e for e in entries if plan([e], root, params)[0]["exists"]]
        if not present:
            out.append({"job": j["id"], "skipped": "ホストに成果物が無い"})
            continue
        r = push(io, present, root, prefix=f"host_snapshot/{date}", params=params)
        out.append({"job": j["id"], **r})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "pull", "push", "snapshot"])
    ap.add_argument("--job", default=None)
    ap.add_argument("--date", default=None, help="snapshot の対象日 YYYYMMDD（既定: 今日 JST）")
    ap.add_argument("--root", default=".")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--which", choices=["in", "out"], default=None, help="plan/push/pull の対象（既定: pull=in, push=out, plan=両方）")
    ap.add_argument("--param", action="append", default=[], help="KEY=VALUE（パスの {KEY} を置換）")
    ns = ap.parse_args(argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()           # ホストでの手動実行・種まき（seed_state_from_host.sh）でも .env の GCS_*（バケット・資格情報）を読む。Jobs には .env が無いので何もしない
    from src.ops import job_map

    if ns.action == "snapshot":
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from src.ops.gcs_io import default_io

        date = ns.date or datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
        print(json.dumps({"date": date, "results": snapshot(default_io(), job_map.load(), Path(ns.root).resolve(), date)}, ensure_ascii=False))
        return 0
    if not ns.job:
        print("--job が必要です", file=sys.stderr)
        return 2
    job = job_map.job_by_id(ns.job)
    if job is None:
        print(f"未知のジョブ: {ns.job}", file=sys.stderr)
        return 2
    params = dict(p.split("=", 1) for p in ns.param)
    root = Path(ns.root).resolve()
    if ns.action == "plan":
        for which in (["in", "out"] if ns.which is None else [ns.which]):
            rows = plan(job.get(f"state_{which}") or [], root, params)
            print(json.dumps({"job": ns.job, "which": which, "rows": rows}, ensure_ascii=False))
        return 0
    from src.ops.gcs_io import default_io

    io = default_io()
    if ns.action == "pull":
        print(json.dumps(pull(io, job.get("state_in") or [], root, prefix=ns.prefix, params=params), ensure_ascii=False))
    else:
        which = ns.which or "out"
        print(json.dumps(push(io, job.get(f"state_{which}") or [], root, prefix=ns.prefix, params=params), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
