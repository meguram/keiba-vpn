"""基準フレーム（z:RB@e）の列名を書き出す（ペルソナの入力群を割り当てるための一覧）。出力: PTRAIN_OUT/frame_cols.json"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.scripts.data import tp27_eval as t
from src.scripts.data import v6_eval as ev

OUT = Path(os.environ.get("PTRAIN_OUT", "data/local/modeling/ptrain"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, base, meta, folds, lp = t.setup()
    X = ev.build_frame(ev.config_of(ev.__dict__.get("BASELINE", "z:RB@e")), df)
    rows = [{"col": c, "dtype": str(X[c].dtype), "miss": float(X[c].isna().mean())} for c in X.columns]
    (OUT / "frame_cols.json").write_text(json.dumps({"n": len(rows), "n_rows": len(X), "cols": rows}, ensure_ascii=False), encoding="utf-8")
    print("cols", len(rows), "rows", len(X))


if __name__ == "__main__":
    main()
