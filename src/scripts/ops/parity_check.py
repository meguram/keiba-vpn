"""並走検証: 同じ日をホストと GCP で処理した成果物を比べて、一致を判定する。

    # 2 つのディレクトリを直接比べる
    python -m src.scripts.ops.parity_check compare --kind parquet_frame --left A --right B --keys race_id,horse_id [--rtol 1e-6 --atol 1e-6]
    # job_map の parity 定義どおりに、ホストの成果物（host_snapshot/<日付>/…）と GCP の成果物（state/…）を GCS から取って比べる
    python -m src.scripts.ops.parity_check auto --date YYYYMMDD [--job export-tables-daily …] [--write] [--send]

比べ方（``config/gcp_job_map.json`` の各ジョブの ``parity``）:
- ``parquet_frame``: 行を ``keys``（空なら浮動小数以外の全列）で突き合わせ、片方にしか無い行・列の差・数値の差（rtol/atol）を数える。
- ``json_tree``: ファイル集合と、JSON の中身を再帰比較（浮動小数は許容誤差内なら一致。生成時刻などの揮発キーは無視）。
- ``file_hash``: ファイル集合と sha256。

結果は ``ops/parity/<日付>/<job>.json``（--write）に残り、切替判定（cron_cutover_status）が『成果物の一致』として読む。
ホスト側の成果物は並走期間だけ ``python -m src.ops.state_sync push --job <id> --which out --prefix host_snapshot/<日付>`` で GCS に上げる
（手順は docs/operations/gcp_jobs_migration.md）。Slack には要約だけを送る（``slack_notify.send_payload``）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
PARITY_PREFIX = "ops/parity"
HOST_SNAPSHOT_PREFIX = "host_snapshot"
VOLATILE_KEYS = {"generated_at", "created_at", "computed_at", "updated_at", "saved_at", "elapsed_sec", "build_sec", "warm_sec", "at", "detected_at",
                 "_meta", "processed_at", "run_at", "timestamp", "built_at"}
CAP = 20


# ───────────────────────── 比較の部品 ─────────────────────────

def list_files(root: Path) -> dict[str, Path]:
    root = Path(root)
    if root.is_file():
        return {root.name: root}
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith(".")}


def _close(a: Any, b: Any, rtol: float, atol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    return a == b


def json_diff(a: Any, b: Any, rtol: float, atol: float, path: str = "$", out: list[str] | None = None, ignore: set[str] = VOLATILE_KEYS) -> list[str]:
    out = [] if out is None else out
    if len(out) >= CAP:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in ignore:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: {'右のみ' if k in b else '左のみ'}")
            else:
                json_diff(a[k], b[k], rtol, atol, f"{path}.{k}", out, ignore)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: 長さ {len(a)} ≠ {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                json_diff(x, y, rtol, atol, f"{path}[{i}]", out, ignore)
    elif not _close(a, b, rtol, atol):
        out.append(f"{path}: {str(a)[:40]} ≠ {str(b)[:40]}")
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_frames(files: dict[str, Path]):
    import pandas as pd

    frames = []
    for rel, p in files.items():
        if p.suffix == ".parquet":
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_frames(left: Any, right: Any, keys: list[str], rtol: float, atol: float) -> dict[str, Any]:
    import numpy as np

    res: dict[str, Any] = {"rows_left": int(len(left)), "rows_right": int(len(right)), "problems": []}
    lc, rc = set(left.columns), set(right.columns)
    if lc != rc:
        res["problems"].append(f"列の差: 左のみ {sorted(lc - rc)[:8]} / 右のみ {sorted(rc - lc)[:8]}")
    cols = sorted(lc & rc)
    if not cols:
        return res
    if keys and not set(keys) <= set(cols):
        res["problems"].append(f"keys {keys} が両方に無い")
        return res
    kcols = keys or [c for c in cols if str(left[c].dtype) not in ("float64", "float32")] or cols
    left = left[cols].sort_values(kcols, kind="mergesort").reset_index(drop=True)
    right = right[cols].sort_values(kcols, kind="mergesort").reset_index(drop=True)
    if keys:
        m = left[keys].merge(right[keys], how="outer", on=keys, indicator=True)
        lo, ro = int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum())
        if lo or ro:
            res["problems"].append(f"キーの差: 左のみ {lo} 行 / 右のみ {ro} 行")
        left = left.merge(m[m["_merge"] == "both"][keys], on=keys).sort_values(keys, kind="mergesort").reset_index(drop=True)
        right = right.merge(m[m["_merge"] == "both"][keys], on=keys).sort_values(keys, kind="mergesort").reset_index(drop=True)
    elif len(left) != len(right):
        res["problems"].append(f"行数 {len(left)} ≠ {len(right)}")
        return res
    max_abs = 0.0
    bad_cols: list[str] = []
    for c in cols:
        a, b = left[c], right[c]
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            ok = np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
            if not ok.all():
                bad_cols.append(c)
                max_abs = max(max_abs, float(np.nanmax(np.abs(x[~ok] - y[~ok]))) if (~ok & ~np.isnan(x) & ~np.isnan(y)).any() else max_abs)
        else:
            if not (a.fillna("<NA>").astype(str).to_numpy() == b.fillna("<NA>").astype(str).to_numpy()).all():
                bad_cols.append(c)
    if bad_cols:
        res["problems"].append(f"値が許容誤差を超えて違う列: {bad_cols[:10]}（最大差 {max_abs:.3g}）")
    res["max_abs_diff"] = max_abs
    return res


def compare_dirs(kind: str, left: Path, right: Path, *, keys: list[str] | None = None, rtol: float = 1e-6, atol: float = 1e-6) -> dict[str, Any]:
    lf, rf = list_files(left), list_files(right)
    rep: dict[str, Any] = {"kind": kind, "files_left": len(lf), "files_right": len(rf), "only_left": sorted(set(lf) - set(rf))[:CAP],
                           "only_right": sorted(set(rf) - set(lf))[:CAP], "problems": []}
    if not lf and not rf:
        rep.update(ok=False, status="empty", problems=["両側とも空（比較できない）"])
        return rep
    if rep["only_left"] or rep["only_right"]:
        rep["problems"].append(f"ファイルの差: 左のみ {len(set(lf) - set(rf))} / 右のみ {len(set(rf) - set(lf))}")
    common = sorted(set(lf) & set(rf))
    if kind == "file_hash":
        bad = [r for r in common if sha256_file(lf[r]) != sha256_file(rf[r])]
        if bad:
            rep["problems"].append(f"内容が違うファイル {len(bad)} 件: {bad[:5]}")
    elif kind == "json_tree":
        diffs: list[str] = []
        for r in common:
            if not r.endswith((".json", ".jsonl")):
                continue
            try:
                if r.endswith(".jsonl"):
                    a = [json.loads(x) for x in lf[r].read_text(encoding="utf-8").splitlines() if x.strip()]
                    b = [json.loads(x) for x in rf[r].read_text(encoding="utf-8").splitlines() if x.strip()]
                else:
                    a, b = json.loads(lf[r].read_text(encoding="utf-8")), json.loads(rf[r].read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                diffs.append(f"{r}: 読めない {exc}")
                continue
            for d in json_diff(a, b, rtol, atol):
                diffs.append(f"{r} {d}")
        rep["json_diffs"] = diffs[:CAP]
        if diffs:
            rep["problems"].append(f"JSON の差 {len(diffs)} 件（例: {diffs[0]}）")
    elif kind == "parquet_frame":
        fr = compare_frames(_read_frames({r: lf[r] for r in common}), _read_frames({r: rf[r] for r in common}), keys or [], rtol, atol)
        rep.update({k: v for k, v in fr.items() if k != "problems"})
        rep["problems"] += fr["problems"]
    else:
        rep["problems"].append(f"不明な kind: {kind}")
    rep["ok"] = not rep["problems"]
    rep["status"] = "match" if rep["ok"] else "mismatch"
    return rep


# ───────────────────────── GCS の突き合わせ ─────────────────────────

def _materialize(io: Any, prefix: str, dest: Path) -> int:
    n = 0
    for k in io.list(prefix):
        if io.download_file(k, dest / k[len(prefix):].lstrip("/")):
            n += 1
    return n


def check_job(io: Any, job: dict[str, Any], date: str, now: datetime, workdir: Path) -> dict[str, Any]:
    from src.ops.job_dispatch import build_params

    par = job.get("parity") or {"kind": "none"}
    params = build_params(datetime.strptime(date, "%Y%m%d").replace(hour=7, tzinfo=JST))
    rep: dict[str, Any] = {"job": job["id"], "date": date, "kind": par["kind"], "checked_at": now.isoformat(timespec="seconds"), "paths": []}
    if par["kind"] == "none":
        rep.update(ok=None, status="not_applicable")
        return rep
    results = []
    for raw in par.get("paths") or []:
        path = raw
        for k, v in params.items():
            path = path.replace("{" + k + "}", v)
        l_dir, r_dir = workdir / job["id"] / "host" / path, workdir / job["id"] / "gcp" / path
        nl = _materialize(io, f"{HOST_SNAPSHOT_PREFIX}/{date}/{path}", l_dir)
        nr = _materialize(io, f"state/{path}", r_dir)
        if nl == 0:
            results.append({"path": path, "ok": False, "status": "no_host_snapshot", "problems": ["ホスト側の成果物が host_snapshot に無い"]})
            continue
        if nr == 0:
            results.append({"path": path, "ok": False, "status": "no_gcp_output", "problems": ["GCP 側の成果物が state に無い"]})
            continue
        r = compare_dirs(par["kind"], l_dir, r_dir, keys=par.get("keys") or [], rtol=float(par.get("rtol", 1e-6)), atol=float(par.get("atol", 1e-6)))
        results.append({"path": path, **r})
    rep["paths"] = results
    rep["ok"] = bool(results) and all(x.get("ok") for x in results)
    rep["status"] = "match" if rep["ok"] else ("no_data" if any(x.get("status", "").startswith("no_") for x in results) else "mismatch")
    return rep


def build_summary(reports: list[dict[str, Any]], date: str) -> dict[str, Any]:
    from src.utils import slack_notify as SN

    rows = [[r["job"], {"match": "一致", "mismatch": "不一致", "no_data": "データなし", "not_applicable": "対象外"}.get(r["status"], r["status"])] for r in reports]
    bad = [r for r in reports if r["status"] in ("mismatch", "no_data")]
    sev = "warning" if bad else "normal"
    blocks = [SN.section(f"*並走検証 {date}*：{sum(1 for r in reports if r['status'] == 'match')} 件一致 / {len(bad)} 件に問題"),
              SN.section(SN.code_table(["ジョブ", "結果"], rows, aligns="ll"))]
    for r in bad[:5]:
        probs = [p for x in r["paths"] for p in (x.get("problems") or [])][:2]
        blocks.append(SN.context([f"{r['job']}: " + " / ".join(probs)]))
    return SN.build_message(f"並走検証 {date}", blocks, severity=sev, fallback=f"並走検証 {date}: 問題 {len(bad)} 件")


def run_auto(io: Any, date: str, job_ids: list[str] | None, *, write: bool, send: bool, now: datetime | None = None, doc: dict[str, Any] | None = None,
             send_fn: Any = None) -> dict[str, Any]:
    from src.ops import job_map
    from src.ops.gcs_io import write_json

    doc = doc or job_map.load()
    now = now or datetime.now(JST)
    jobs = [j for j in job_map.by_destination("gcp", doc) if (j.get("parity") or {}).get("kind", "none") != "none" and (not job_ids or j["id"] in job_ids)]
    reports = []
    with tempfile.TemporaryDirectory(prefix="parity_") as td:
        for j in jobs:
            rep = check_job(io, j, date, now, Path(td))
            reports.append(rep)
            if write:
                write_json(io, f"{PARITY_PREFIX}/{date}/{j['id']}.json", rep)
    out = {"date": date, "reports": [{k: r[k] for k in ("job", "status", "ok")} for r in reports]}
    if send and os.environ.get("KEIBA_SLACK_NOTIFY", "") == "1" and reports:
        from src.utils import slack_notify as SN

        out["slack"] = (send_fn or SN.send_payload)(build_summary(reports, date), channel="report")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--kind", required=True, choices=["parquet_frame", "json_tree", "file_hash"])
    c.add_argument("--left", required=True)
    c.add_argument("--right", required=True)
    c.add_argument("--keys", default="")
    c.add_argument("--rtol", type=float, default=1e-6)
    c.add_argument("--atol", type=float, default=1e-6)
    a = sub.add_parser("auto")
    a.add_argument("--date", required=True)
    a.add_argument("--job", action="append", default=[])
    a.add_argument("--write", action="store_true")
    a.add_argument("--send", action="store_true")
    ns = ap.parse_args(argv)
    if ns.cmd == "compare":
        rep = compare_dirs(ns.kind, Path(ns.left), Path(ns.right), keys=[k for k in ns.keys.split(",") if k], rtol=ns.rtol, atol=ns.atol)
        print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
        return 0 if rep["ok"] else 1
    from src.ops.gcs_io import default_io
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    out = run_auto(default_io(), ns.date, ns.job or None, write=ns.write, send=ns.send)
    print(json.dumps(out, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
