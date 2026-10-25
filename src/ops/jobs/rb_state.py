"""v6 の RB（めぐ指数/netkeiba 指数の履歴 16 列）の推論用の履歴状態を、前日までに確定した走で作り直す（日次の Jobs ``rb-state-daily``）。

    python -m src.ops.jobs.rb_state [--out data/local/inference/state/rb_v6] [--years 2020 2021 ...]

ホストの ``scripts/cron/rb_state_daily.sh`` と同じ手順を 1 つのモジュールにまとめたもの:
tp65_prep（走ごとの指数の表）→ build_rb_state build → 検査（件数が前回より減っていない・最終日が戻っていない）→ ファイル単位で原子的に差し替え。
入力は ``data/page_reference/tables``（state_in）、出力は ``data/local/inference/state/rb_v6``（state_out）。検査に通らなければ古い状態を残して終了コード 1。
予測のジョブ（predict-race / eve-precompute）は ``data/local/inference/state`` を state_in で受け取るので、モデルとして別に配る必要はない。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(os.environ.get("KEIBA_V6_RB_STATE") or Path(os.environ.get("KEIBA_INFERENCE_DIR") or ROOT / "data/local/inference") / "state" / "rb_v6")
MIN_ROWS = 100_000
FILES = ("hist.parquet", "params.json")          # 差し替える順（履歴 → 定数）


def years_default(now: datetime | None = None) -> list[int]:
    now = now or datetime.now(timezone(timedelta(hours=9)))
    return list(range(2020, now.year + 1))


def check(old: Path, new: Path) -> tuple[bool, str]:
    """新しい状態の検査。履歴の件数が前回の 99% 未満・最終日が戻っていれば不合格。"""
    import pandas as pd

    n = pd.read_parquet(new / "hist.parquet", columns=["date"])
    pn = json.loads((new / "params.json").read_text(encoding="utf-8"))
    if len(n) < MIN_ROWS:
        return False, f"件数が少なすぎる {len(n)}"
    if (old / "hist.parquet").exists() and (old / "params.json").exists():
        o = pd.read_parquet(old / "hist.parquet", columns=["date"])
        po = json.loads((old / "params.json").read_text(encoding="utf-8"))
        if len(n) < len(o) * 0.99:
            return False, f"件数が減った {len(o)} -> {len(n)}"
        if str(pn.get("cut")) < str(po.get("cut")):
            return False, f"最終日が戻った {po.get('cut')} -> {pn.get('cut')}"
        return True, f"件数 {len(o):,} -> {len(n):,}、最終日 {po.get('cut')} -> {pn.get('cut')}"
    return True, f"新規 件数 {len(n):,}、最終日 {pn.get('cut')}"


def swap(new: Path, out: Path) -> None:
    """ファイル単位で原子的に差し替える（同じファイルシステム上の rename）。"""
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        tmp = out / f".{name}.new"
        shutil.copyfile(new / name, tmp)
        os.replace(tmp, out / name)


def _module(mod: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", mod, *args], check=True, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")})


def run(out: Path | None = None, years: list[int] | None = None, runner: Callable[..., None] | None = None) -> dict[str, Any]:
    out = Path(out or DEFAULT_OUT)
    years = years or years_default()
    runner = runner or _module
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".rb_v6_build.", dir=str(out.parent)))
    try:
        runner("src.scripts.data.tp65_prep", "--out", str(tmp / "prep"), "--years", *[str(y) for y in years])
        runner("src.scripts.data.build_rb_state", "build", "--runs", str(tmp / "prep" / "runs.parquet"), "--out", str(tmp / "state"))
        ok, msg = check(out, tmp / "state")
        if not ok:
            return {"status": "error", "reason": f"検査に通らない: {msg}。古い状態を残す"}
        swap(tmp / "state", out)
        return {"status": "ok", "detail": msg, "out": str(out)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--years", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    try:
        res = run(Path(a.out) if a.out else None, a.years)
    except subprocess.CalledProcessError as exc:
        res = {"status": "error", "reason": f"{exc.cmd[2] if len(exc.cmd) > 2 else exc.cmd} が失敗（終了コード {exc.returncode}）。古い状態を残す"}
    print(json.dumps(res, ensure_ascii=False))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
