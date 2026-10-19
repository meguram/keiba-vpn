"""dev 用: dev ストアの馬の履歴に合わせた『めぐ指数』のモックを data/page_reference/tables/<年>/megu_index_flat.parquet に作る。

成長曲線の表示確認のためだけのもの（実データではない）。KEIBA_ENV=dev でしか動かず、実データのファイルがあれば上書きしない。
  python -m src.scripts.data.make_dev_megu_mock
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def _h(*parts: str) -> float:
    return int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> int:
    if (os.environ.get("KEIBA_ENV") or "").lower() != "dev":
        print("KEIBA_ENV=dev でのみ実行できます", file=sys.stderr)
        return 1
    rows: dict[int, list[dict]] = {}
    for f in sorted((ROOT / "data" / "dev_mock" / "horse_result").rglob("*.json")):
        d = json.loads(f.read_text())
        hid = str(d.get("horse_id") or f.stem)
        hist = sorted((r for r in d.get("race_history") or [] if r.get("race_id") and r.get("date")), key=lambda r: str(r["date"]))
        base = 35 + 30 * _h(hid, "base")
        for i, r in enumerate(hist):
            y = int(str(r["date"])[:4])
            v = base + 4.0 * i + 8 * (_h(hid, str(r["race_id"])) - 0.5)
            rows.setdefault(y, []).append({
                "race_id": str(r["race_id"]), "horse_id": hid, "megu_index": round(v, 1), "computation_status": "valid",
                "megu_index_v3a_abs_robust": round(v, 1), "megu_v3a_status": "ok", "megu_v3a_low_confidence": False,
                "date": str(r["date"]).replace("/", "-")[:10],
            })
    # 出馬表の過去走（馬柱）にいる馬にも、条件（競馬場・面・距離・クラス・斤量・馬齢）つきのモックを作る（想定めぐ指数の表示確認用）
    for f in sorted((ROOT / "data" / "dev_mock" / "race_shutuba_past").rglob("*.json")):
        d = json.loads(f.read_text())
        for e in d.get("entries") or []:
            hid = str(e.get("horse_id") or "")
            base = 35 + 30 * _h(hid, "base")
            for i, pr in enumerate(sorted((x for x in e.get("past_races") or [] if x.get("race_id") and x.get("date")), key=lambda r: str(r["date"]))):
                m = re.search(r"\d{4}\.\d{2}\.\d{2} (\S+) \d+ (\S+) ([芝ダ障])(\d+) .*?\d+人 \S+ ([\d.]+) ", str(pr.get("raw") or ""))
                venue, klass, weight = (m.group(1), m.group(2), float(m.group(5))) if m else (None, "", None)
                grade = "新馬" if "新馬" in klass else "未勝利" if "未勝利" in klass else "3勝" if "3勝" in klass else "2勝" if "2勝" in klass else "1勝" if "1勝" in klass else "OP"
                v = base + 3.0 * i + 8 * (_h(hid, str(pr["race_id"]), "p") - 0.5) - (0.5 * float(pr.get("finish_position") or 8))
                y = int(str(pr["date"])[:4])
                rows.setdefault(y, []).append({
                    "race_id": str(pr["race_id"]), "horse_id": hid, "megu_index": round(v, 1), "computation_status": "valid",
                    "megu_index_v3a_abs_robust": round(v, 1), "megu_v3a_status": "ok", "megu_v3a_low_confidence": False,
                    "date": str(pr["date"]).replace(".", "-")[:10], "venue": venue, "surface": pr.get("surface"), "distance": float(pr.get("distance") or 0) or None,
                    "grade": grade, "jockey_weight": weight, "sex_age": "牡4",
                })
    for y, rs in rows.items():
        out = ROOT / "data" / "page_reference" / "tables" / str(y) / "megu_index_flat.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and "dev_mock" not in pd.read_parquet(out).columns:
            print(f"{out} は実データのため上書きしません")
            continue
        df = pd.DataFrame(rs).sort_values("venue", na_position="last", kind="stable").drop_duplicates(["race_id", "horse_id"])      # 条件つきの行を優先
        df["dev_mock"] = True
        df.to_parquet(out, index=False)
        print(f"{out}: {len(df)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
