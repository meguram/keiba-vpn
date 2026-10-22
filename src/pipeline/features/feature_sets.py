"""特徴量セット（今までの基準 / 復活セット）の定義の読み出し。定義の正は config/feature_sets.json。

学習の評価（src/scripts/data/v6_eval.py）が、セット名から『基準 + 追加する群』を選べるようにする。
- external_groups(): 追加する群の parquet（行キー race_id, horse_id）を v6_eval の EXTERNAL に渡す形で返す。
- v6_eval_name(set): v6_eval の run に渡す構成名（例 z:RB+REV_tp32_FIRST+...@e）。
使い方: python -m src.pipeline.features.feature_sets show [セット名]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "feature_sets.json"
BASELINE_NAME = "z:RB@e"          # 今までの基準（v6e の構成）を v6_eval の名前で表したもの


def load(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def set_names(cfg: dict | None = None) -> list[str]:
    return list((cfg or load())["sets"])


def resolve(name: str, cfg: dict | None = None, ready_only: bool = False) -> list[str]:
    """セットに足す群の名前（v6_eval の外部群の名前）を返す。基準は空。ready_only なら、学習表に列がある（すぐ評価できる）群だけ。"""
    cfg = cfg or load()
    if name not in cfg["sets"]:
        raise KeyError(f"未知の特徴量セット: {name}（{', '.join(cfg['sets'])}）")
    v = cfg["sets"][name]
    return list(v.get("add_ready", v["add"]) if ready_only else v["add"])


def external_groups(work_root: Path | None = None, cfg: dict | None = None) -> dict[str, tuple[Path, list[str] | None]]:
    """v6_eval.EXTERNAL に足す {名前: (parquet のパス, 使う列 or None=キー以外すべて)}。パスが無い群（v6_eval が名前で作るもの）は除く。"""
    cfg = cfg or load()
    out: dict[str, tuple[Path, list[str] | None]] = {}
    for name, g in cfg["groups"].items():
        p = g.get("path") or ""
        if not p or not name.startswith(("REV_", "S76_")):
            continue
        path = Path(p)
        out[name] = (path if path.is_absolute() else ROOT / path, g.get("cols") or None)
    return out


def v6_eval_name(name: str, cfg: dict | None = None) -> str:
    """v6_eval の run に渡す構成名。基準は z:RB@e、復活セットは群を足す。"""
    add = resolve(name, cfg, ready_only=True)   # 列がまだ無い群（合成スコアなど）は、作った後に add へ戻る
    return BASELINE_NAME if not add else "z:RB+" + "+".join(add) + "@e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("name", nargs="?", default="")
    args = ap.parse_args()
    cfg = load()
    for n in ([args.name] if args.name else set_names(cfg)):
        add = resolve(n, cfg)
        cols = sum((cfg["groups"].get(a) or {}).get("n_cols") or 0 for a in add)
        unready = [a for a in add if a not in resolve(n, cfg, ready_only=True)]
        extra = f"（未構築: {', '.join(unready)}）" if unready else ""
        print(f"{n}: 追加 {len(add)} 群 / 約 {cols} 列{extra} -> {v6_eval_name(n, cfg)[:120]}")


if __name__ == "__main__":
    main()
