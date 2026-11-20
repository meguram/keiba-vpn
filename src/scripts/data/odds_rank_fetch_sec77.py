"""2026 年の本番の想定オッズ（final_odds_prediction。§77）を、発走前オッズ（pre_odds）のあるレースだけ GCS から 1 回だけ読み、手元に保存する。

    python -m src.scripts.data.odds_rank_fetch_sec77 [--out DIR] [--cap-mb 20]

統括役の承認（2026-11-18）による一度きりの読み出し。繰り返す処理には入れない。
- 読み出しは src.ops.gcs_io.default_io()（.env のサービスアカウント）。一覧（list）は使わない。race_id から名前（chuou/data/preprocessed/netkeiba/pc/final_odds_prediction/<年>/<race_id>.json）を組み立てて 1 件ずつ読む。
- 読み出しは GcsMeter（candidate_runner）で数え、上限（既定 20MB）を超えたら止める。一覧を呼んだら止める。
- 手元に保存したものと、無かったもの（not_found.json）は読み直さない。
- 見込み: 約 300 文書 × 数 KB = 数 MB。GCS から GCP の外への転送 約 18 円/GiB・読み出し 約 0.06 円/千回なので 1 円未満。
読んだ量は <out>/fetch_log.json に追記する。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("ODDS_RANK_DATA_ROOT") or ROOT)
PRE = DATA / "data/local/research/persona1/pre_odds.parquet"
OUT = Path("/home/jovyan/work/odds_rank_out/sec77_2026")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cap-mb", type=float, default=20.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    nf_f = out / "not_found.json"
    not_found = set(json.loads(nf_f.read_text(encoding="utf-8"))) if nf_f.exists() else set()

    rids = sorted(r for r in pd.read_parquet(PRE, columns=["race_id", "phase"]).query("phase == 't60'")["race_id"].astype(str).unique() if r.startswith("2026"))
    todo = [r for r in rids if not (out / f"{r}.json").exists() and r not in not_found]
    print(f"対象 {len(rids)} レース、読むのは {len(todo)}", flush=True)
    if not todo:
        return

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv(dotenv_path=DATA / ".env")          # .env のサービスアカウント（google.auth.default には頼らない）
    from src.config.data_paths import GCS_BASE
    from src.config.gcp_credentials import gcp_credentials_available
    from src.ops.gcs_io import default_io

    if not gcp_credentials_available():
        raise SystemExit(".env にサービスアカウントの鍵が無い（ADC には落とさない）")
    from src.pipeline.inference.candidate_runner import GcsMeter

    meter = GcsMeter(a.cap_mb)
    io = default_io()
    got = miss = 0
    t0 = time.time()
    stopped = ""
    try:
        for rid in todo:
            if meter.over or meter.lists:
                stopped = f"上限で止めた: 読み出し {meter.bytes / 1e6:.2f}MB・一覧 {meter.lists} 回"
                print(stopped, flush=True)
                break
            b = io.get_bytes(f"{GCS_BASE}/final_odds_prediction/{rid[:4]}/{rid}.json")   # 属性の取得 + 本体（無ければ None）
            if b is None:
                not_found.add(rid)
                miss += 1
                continue
            tmp = out / f".{rid}.json.tmp"      # 書きかけを保存済みとみなさない
            tmp.write_bytes(b)
            os.replace(tmp, out / f"{rid}.json")
            got += 1
    except BaseException as e:                  # 例外・中断でも、読んだ量の記録は残す
        stopped = f"例外で止めた: {e.__class__.__name__}: {e}"
        raise
    finally:
        nf_f.write_text(json.dumps(sorted(not_found)), encoding="utf-8")
        log = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "saved": got, "not_found": miss, "docs": got + miss, "gcs_ops_approx": 2 * got + miss, "bytes": meter.bytes,
               "lists": meter.lists, "sec": round(time.time() - t0, 1), "cap_mb": a.cap_mb, "stopped": stopped}
        lf = out / "fetch_log.json"
        hist = json.loads(lf.read_text(encoding="utf-8")) if lf.exists() else []
        lf.write_text(json.dumps(hist + [log], ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(log, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
