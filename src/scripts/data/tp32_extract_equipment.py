"""tp-32: GCS の smartrc_race から、馬具などの『出走ごとの申告値』と馬ごとの全戦績（fullresults）を取り出して表にする。

python -m src.scripts.data.tp32_extract_equipment            # 出力: data/local/modeling/training_pipeline_v4/tp32/{runners_eq,career_eq}.parquet
  runners_eq: 出走表の行（race_id, horse_id, date, blinker, handicap, invite, ustat, tozai, jcode, jcode_before, interval, …）。取得時点の値。
  career_eq : fullresults の行（馬×戦績。当時の blinker・handicap・ustat・interval・jcode・trackkind・range・direction・place など全項目）。
              各戦績は確定した過去の事実で、取得が出走後でも値は変わらない。netkeiba の race_id は rcode から作る（年+場+回+日+R）。
読み取り専用（GCS には書かない）。環境変数: PROCS（既定 4）。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp32"
BASE_PREFIX = "chuou/data/preprocessed/netkeiba/pc/smartrc_race"
YEARS = ("2020", "2021", "2022", "2023", "2024", "2025", "2026")
RUNNER_KEYS = ["blinker", "handicap", "invite", "ustat", "tozai", "jcode", "jcode_before", "interval", "age", "sex", "futan", "weight", "vary",
               "tcode", "ocode", "pcode", "entry", "uno", "wno"]


def rcode_to_race_id(rcode: str) -> str:
    """'2025081007030601'（年 月日 場 回 日 R）→ netkeiba の race_id '202507030601'。"""
    r = str(rcode)
    return r[0:4] + r[8:16] if len(r) >= 16 else ""


def extract_race(d: dict, race_id: str) -> tuple[list[dict], list[dict]]:
    meta = d.get("_meta") or {}
    saved = str(meta.get("saved_at") or meta.get("fetched_at") or meta.get("created_at") or "")
    runners, careers = [], []
    for r in d.get("runners") or []:
        hid = str(r.get("hcode") or "").strip()
        if hid:
            runners.append({"race_id": race_id, "horse_id": hid, "date": str(d.get("rcode") or "")[:8], "meta_saved": saved,
                            **{k: r.get(k) for k in RUNNER_KEYS}})
    for hid, rows in (d.get("fullresults") or {}).items():
        for x in rows or []:
            rec = {k: (v if isinstance(v, (str, int, float)) or v is None else str(v)) for k, v in x.items()}
            rec["horse_id"] = str(hid).strip()
            rec["race_id"] = rcode_to_race_id(x.get("rcode"))
            careers.append(rec)
    return runners, careers


def _bucket():
    os.chdir(ROOT)
    from src.scraper.storage import HybridStorage

    return HybridStorage()._get_bucket()


def _work(names):
    b = _bucket()
    R, C = [], []
    for nm in names:
        rid = nm.rsplit("/", 1)[-1][:-5]
        r, c = extract_race(json.loads(b.blob(nm).download_as_bytes()), rid)
        R += r
        C += c
    return pd.DataFrame(R), pd.DataFrame(C)


def main() -> None:
    from multiprocessing import Pool

    OUT.mkdir(parents=True, exist_ok=True)
    b = _bucket()
    names = []
    for y in YEARS:
        names += [x.name for x in b.list_blobs(prefix=f"{BASE_PREFIX}/{y}/") if x.name.endswith(".json")]
    print("files", len(names), flush=True)
    chunks = [names[i:i + 100] for i in range(0, len(names), 100)]
    R, C = [], []
    t0 = time.time()
    with Pool(int(os.environ.get("PROCS", "4"))) as p:
        for i, (r, c) in enumerate(p.imap(_work, chunks), 1):
            R.append(r)
            C.append(c)
            if i % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(chunks)} {time.time() - t0:.0f}s", flush=True)
    runners = pd.concat(R, ignore_index=True)
    career = pd.concat(C, ignore_index=True).drop_duplicates(["horse_id", "rcode"]).reset_index(drop=True)
    runners.to_parquet(OUT / "runners_eq.parquet", index=False)
    career.to_parquet(OUT / "career_eq.parquet", index=False)
    print("runners", runners.shape, "career", career.shape, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
