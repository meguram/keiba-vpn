"""血統の列（ダートの適性の 4 列: dirt_sire・dirt_bms・dirt_mat5・dirt_lv5）を、2016〜24 の全走に付ける（担当 model-relabel）。

- 作りは pedigree_lab の凍結の第 4 版のまま: src/research/pedigree_lab/forward_record.py の `dirt_aptitude`。
  推論の側（src/research/pedigree_persona/columns.py の build_day）も同じ関数を呼ぶので、学習と推論で同じ値になる。
- as-of（その日より前の走だけ・自分の過去走を引く）。未知の種牡馬・牝系は縮めで全体の水準に寄る。
- 入力は pedigree_lab の feat_v4（2016〜24、417,278 行）と 5 代血統。feat_v4 に無い走（障害・取消など）は NaN のまま。
- 前向きの登録 prereg_dirt_mat5 の比べの相手は凍結の v5bag のまま（この研究のモデルに入れても登録は変えない）。
費用 0 円（手元の表だけ。GCS を読まない）。
  KEIBA_SESSION=model-relabel nice -n 19 python -m src.research.model_relabel.mr_ped
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.model_relabel import mr_labels as LB

ROOT = Path(__file__).resolve().parents[3]
PED_COLS = ["dirt_sire", "dirt_bms", "dirt_mat5", "dirt_lv5"]
DEST = LB.OUT / "ped_dirt4.parquet"


def build() -> Path:
    from src.ops import data_ledger as DL
    from src.research.pedigree_lab import aptitude as ap_
    from src.research.pedigree_lab import data, forward_record
    import hashlib
    import json
    frozen = [ROOT / "src/research/pedigree_lab" / f for f in
              ("forward_record.py", "aptitude.py", "aptitude_checks.py", "data.py", "features4.py", "features2.py")]
    inputs = {n: hashlib.sha256((data.OUT / n).read_bytes()).hexdigest() for n in ("feat_v4.parquet", "ped.parquet")}
    tok = DL.guard(tool="src/research/model_relabel/mr_ped.py:build", family="model_relabel", version="ped-dirt4-v4",
                   period=("2016-01-01", "2024-12-31"), purpose="reproduce", reads=("着順",), frozen_paths=frozen,
                   reason="凍結の第 4 版のダートの 4 列（as-of の産駒の成績の集計）を特徴量として作るだけ。成績は数えない")
    F = ap_.base_frame()
    assert F["date"].max() <= pd.Timestamp("2024-12-31")
    cols = forward_record.dirt_aptitude(F, data.load_ped())
    out = F[["race_id", "horse_id"]].astype(str).copy()
    for c in PED_COLS:
        out[c] = cols[c]
    out = out.drop_duplicates(["race_id", "horse_id"])
    out.to_parquet(DEST, index=False)
    # 入力の表の sha256（学習の側の値の元を残す。推論の側は pedigree_persona の check_frozen が研究の道具の sha を照合する）
    DEST.with_suffix(".json").write_text(json.dumps({"inputs_sha256": inputs, "rows": len(out)}, ensure_ascii=False, indent=1), encoding="utf-8")
    tok.finish(result_path=DEST)
    print("ped_dirt4", len(out), out[PED_COLS].notna().mean().round(3).to_dict(), flush=True)
    return DEST


if __name__ == "__main__":
    build()
