"""tp-32: 『出走前の値』と『出走後に取得した値』の一致を測る（前向きの検証）。

python -m src.scripts.data.verify_equipment_snapshot [--out PATH]
  (1) smartrc_prerace（eve / t60 のスナップショット）の runners と、同じレースの smartrc_race（出走後の再取得で上書きされた最新）を、
      馬ごとに項目別で比べる（blinker・handicap・invite・ustat・tozai・jcode・interval など）。old_pr は参考に出す（特徴量に使わない列）。
  (2) 馬柱（race_shutuba_past）の blinker（馬名の右の印 B）と、出走後の smartrc_race の blinker を比べる。
  不一致が出たら、出走後取得の過去データ（学習に使っている SmartRC 由来の馬具）が出走前の値とずれている証拠。件数が溜まるほど判定が確かになる。
結果: data/local/meta/equipment_snapshot_verify.json（既定）。読み取りのみ。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIELDS = ["blinker", "handicap", "invite", "ustat", "tozai", "jcode", "jcode_before", "interval", "age", "sex", "futan", "weight", "vary", "old_pr"]


def compare_runners(pre: list[dict], post: list[dict], fields: list[str] = FIELDS) -> dict[str, list[int]]:
    """項目ごとに [比べた馬の数, 一致した数]。"""
    pm = {str(r.get("hcode")): r for r in pre}
    out: dict[str, list[int]] = {f: [0, 0] for f in fields}
    for r in post:
        p = pm.get(str(r.get("hcode")))
        if p is None:
            continue
        for f in fields:
            if f in p or f in r:
                out[f][0] += 1
                out[f][1] += int(str(p.get(f)).strip() == str(r.get(f)).strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/local/meta/equipment_snapshot_verify.json"))
    a = ap.parse_args()
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    tot: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    pastB = [0, 0]
    races = 0
    for rid in st.list_keys("smartrc_prerace"):
        doc = st.load("smartrc_prerace", rid, bypass_cache=True)
        post = st.load("smartrc_race", rid, bypass_cache=True)
        if not doc or not post or not post.get("runners"):
            continue
        races += 1
        for ph, snap in (doc.get("phases") or {}).items():
            for f, (n, k) in compare_runners(snap.get("runners") or [], post["runners"]).items():
                tot[ph][f][0] += n
                tot[ph][f][1] += k
        past = st.load("race_shutuba_past", rid, bypass_cache=True) or {}
        pb = {str(e.get("horse_id")): e.get("blinker") for e in past.get("entries") or [] if e.get("blinker") is not None}
        for r in post["runners"]:
            h = str(r.get("hcode"))
            if h in pb:
                pastB[0] += 1
                pastB[1] += int(str(pb[h]) == str(r.get("blinker")).strip())
    res = {"races": races, "phases": {ph: {f: {"n": v[0], "match": v[1], "rate": (v[1] / v[0] if v[0] else None)} for f, v in d.items()} for ph, d in tot.items()},
           "shutuba_past_blinker_vs_smartrc": {"n": pastB[0], "match": pastB[1], "rate": (pastB[1] / pastB[0] if pastB[0] else None)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
