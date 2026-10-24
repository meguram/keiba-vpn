"""tp-77 の進捗・結果の要約（Markdown）。Claude が止まっていても、ホストで定期的に書き出して読めるようにする。

python -m src.scripts.data.ptrain_summary status > STATUS.md    # 実行中の処理・完了した評価・ファイルの有無
python -m src.scripts.data.ptrain_summary table                  # 特徴量の評価の比較表（基準 z:RB@e との差と、年別・上限）
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.pipeline.features import feature_sets as fs
from src.scripts.data import tp40_eval as e40
from src.scripts.data import v6_eval as ev

OUT = Path(os.environ.get("PTRAIN_OUT", "/home/jovyan/work/ptrain_out"))
BASE = "z:RB@e"


def _eval_dir() -> Path:
    return ev._paths("")[0]


def rows() -> list[dict]:
    cfg = fs.load()
    d = _eval_dir()
    lv = json.loads((d / "levels.json").read_text(encoding="utf-8")) if (d / "levels.json").exists() else {}
    cmpf = d / f"results_vs_{BASE}.json"
    pairs = json.loads(cmpf.read_text(encoding="utf-8")).get("pairs", {}) if cmpf.exists() else {}
    b = (lv.get(BASE) or {}).get("levels")
    out = []
    for name, s in cfg["sets"].items():
        if name == "baseline":
            continue
        n = fs.v6_eval_name(name, cfg)
        r = {"name": name, "doc": s.get("doc", ""), "done": False}
        p = pairs.get(n)
        if p:
            o = p["overall"]
            ys = [(y["Δwin"] + y["Δtop2"] + y["Δtop3"]) / 3 for y in p["by_year"].values()]
            r.update(done=True, dwin=o["Δwin"], dtop2=o["Δtop2"], dtop3=o["Δtop3"], hi=(o["win_hi"], o["top2_hi"], o["top3_hi"]), years=ys)
        elif b and (lv.get(n) or {}).get("levels"):
            l = lv[n]["levels"]
            r.update(done=True, dwin=l["logloss_win"] - b["logloss_win"], dtop2=l["logloss_top2"] - b["logloss_top2"], dtop3=l["logloss_top3"] - b["logloss_top3"])
        if r["done"]:
            r["d3"] = (r["dwin"] + r["dtop2"] + r["dtop3"]) / 3
        out.append(r)
    return out


def verdict(r: dict) -> str:
    if not r.get("done"):
        return "未完了"
    if "hi" in r and all(h < 0 for h in r["hi"]) and r["d3"] <= -0.0002:
        return "採用候補（3目的とも上限<0かつ平均差≤−0.0002）"
    if "years" in r and all(y < 0 for y in r["years"]) and r["d3"] <= -0.0001:
        return "束で評価する候補（3年とも負、平均差≤−0.0001）"
    return "上乗せなし" if r["d3"] > -0.0001 else "点推定のみ負（年別・上限で未達）"


def table() -> str:
    L = ["| 特徴量の足し方 | 勝率 | 連対 | 複勝 | 平均差 | 判定 |", "|---|---|---|---|---|---|"]
    for r in rows():
        if r["done"]:
            L.append(f"| {r['doc'][:60]} | {r['dwin']:+.5f} | {r['dtop2']:+.5f} | {r['dtop3']:+.5f} | {r['d3']:+.5f} | {verdict(r)} |")
        else:
            L.append(f"| {r['doc'][:60]} | - | - | - | - | 未完了 |")
    return "\n".join(L)


def status() -> str:
    jst = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + 9 * 3600 - time.timezone * 0)) if False else time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    L = [f"# tp-77 進捗（{jst}、JST は +9 時間）", ""]
    own = sorted(Path("/tmp/keiba_heavy_slots").glob("*.owner"))
    L += ["## 実行中（共有スロット）"] + [f"- {p.read_text().strip()}" for p in own] + [""]
    log = OUT / "phase1.log"
    if log.exists():
        L += ["## 直近の評価の開始・終了"] + [f"- {x}" for x in log.read_text(encoding="utf-8").splitlines()[-10:]] + [""]
    R = rows()
    done = [r for r in R if r["done"]]
    L += [f"## 特徴量の評価（完了 {len(done)} / {len(R)}）", "", table(), ""]
    L += ["## ファイル"] + [f"- {n}: {'あり' if (OUT / n).exists() else 'なし'}" for n in ("pace_rows.parquet", "sire_rows.parquet", "cross_rows.parquet", "weight_rows.parquet", "workout_rows.parquet", "cross_report.json", "workout_report.json", "wind_report.json")]
    ap = OUT / "autopilot.log"
    if ap.exists():
        L += ["", "## 自動実行（autopilot）の記録"] + [f"- {x}" for x in ap.read_text(encoding="utf-8").splitlines()[-12:]]
    return "\n".join(L)


if __name__ == "__main__":
    import sys

    print(status() if (sys.argv[1:] or ["status"])[0] == "status" else table())
