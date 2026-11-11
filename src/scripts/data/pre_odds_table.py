"""発走前のオッズ（t60・t15・t10・t05・t02 のスナップショット）を 1 枚の表にまとめる。

ペルソナ 1 の検証で、確定オッズ（締切後）ではなく『購入時点に近いオッズ』で期待値と回収率を評価するための入力。
race_odds の文書（HybridStorage）の snapshots を読む。保存は 2026-10-17 から始まったばかりなので、量は少ない（週に約 70 レース）。

python -m src.scripts.data.pre_odds_table [--since 20261017] [--out PATH]

読み出しの量: 結果が確定した新しいレースだけ読む。すでに表にあるレース（manifest）は読み直さない（差分だけ）。
1 レース = 文書 1 件（約 10〜20 KB）。週 70 レースなら読み出し 70 回・約 1 MB で、月の費用は 1 円未満。GCS の一覧は取らない。
出力: data/local/research/persona1/pre_odds.parquet（1 行 = レース × 馬番 × 時点）と pre_odds_done.json（読んだレース ID と、スナップショットが無かったレース）。
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/research/persona1/pre_odds.parquet"
TABLES = ROOT / "data/page_reference/tables"
PHASES = ("t60", "t15", "t10", "t05", "t02")


def race_ids_since(since: str) -> list[str]:
    ids: set[str] = set()
    for y in range(int(since[:4]), date.today().year + 1):
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            r = pd.read_parquet(f, columns=["race_id", "date"]).drop_duplicates("race_id")
            ids |= set(r.loc[r["date"].astype(str).str.replace("-", "") >= since, "race_id"].astype(str))
    return sorted(ids)


def rows_of(rid: str, doc: dict) -> list[dict]:
    out = []
    for s in doc.get("snapshots") or []:
        ph = str(s.get("phase"))
        if ph not in PHASES:
            continue
        for e in s.get("entries") or []:
            out.append({"race_id": rid, "phase": ph, "snapshot_at": s.get("snapshot_at"), "horse_number": int(e.get("horse_number") or 0),
                        "win_odds": float(e.get("win_odds") or 0), "place_odds_min": float(e.get("place_odds_min") or 0),
                        "place_odds_max": float(e.get("place_odds_max") or 0), "popularity": int(e.get("popularity") or 0)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20261017")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    done_f = out.with_name("pre_odds_done.json")
    done: dict = json.loads(done_f.read_text(encoding="utf-8")) if done_f.exists() else {"read": [], "no_snapshot": {}}
    if isinstance(done["no_snapshot"], list):          # 古い形式（リスト）→ 読んだ日つきの辞書
        done["no_snapshot"] = {r: date.today().isoformat() for r in done["no_snapshot"]}
    today = date.today()
    # スナップショットが無かったレースは、読んでから 14 日間は読み直す（後から補われることがある）。それを過ぎたら読まない
    retry = {r for r, d0 in done["no_snapshot"].items() if today - date.fromisoformat(d0) <= timedelta(days=14)}
    seen = set(done["read"]) | (set(done["no_snapshot"]) - retry)
    todo = [r for r in race_ids_since(a.since) if r not in seen]
    print(f"新しく読むレース {len(todo)}（読み済み {len(seen)}、うち再試行 {len(retry)}）")
    if not todo:
        return
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    rows: list[dict] = []
    failed = 0
    for rid in todo:
        doc = st.load("race_odds", rid)
        if not doc:              # 読み出しの失敗（GCS のタイムアウト・権限など）か、文書が無い。記録せず次回に回す
            failed += 1
            continue
        rr = rows_of(rid, doc)
        if rr:
            rows += rr
            if rid not in done["read"]:
                done["read"].append(rid)
            done["no_snapshot"].pop(rid, None)
        else:
            done["no_snapshot"].setdefault(rid, today.isoformat())      # 文書はあるが snapshots が空
    if failed:
        print(f"読み出しに失敗（次回に回す）: {failed} 件" + ("。半分以上が失敗している。GCS の接続を確認すること" if failed * 2 > len(todo) else ""))
    new = pd.DataFrame(rows)
    if out.exists() and len(new):
        new = pd.concat([pd.read_parquet(out), new], ignore_index=True).drop_duplicates(["race_id", "horse_number", "phase"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(new):
        new.to_parquet(out, index=False)
    done_f.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    print(f"{out}: {len(new):,} 行、レース {new['race_id'].nunique() if len(new) else 0}、スナップショット無し {len(done['no_snapshot'])}")


if __name__ == "__main__":
    main()
