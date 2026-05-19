"""5gen pedigree の sex フィールドを race_result_flat から **即時** 反映する。

背景:
    patch_sex.py が誤った判定関数 (parse_horse_sex_from_ped_html, mare_line_box 優先) を
    使っていたため、ほぼ全ての馬が '牝' と判定されていた (87% 牝、0.28% 牡)。

このスクリプトは:
    1. data/local/tables/*/race_result_flat.parquet から sex_age を取得
    2. horse_id → sex (牡 / 牝 / セ) を構築 (競走経験のある 35,891 頭分)
    3. 5gen JSON ファイルの sex フィールドを上書き (race_result_flat ベース)
    4. 「race_result_flat に存在しない 5gen 馬 (= 競走経験なし = 純血統登録馬)」は
       未反映として残し、後続の patch_sex (修正版) に任せる

Usage:
    python -m src.scripts.scraping.fix_sex_from_race_results
        [--clear-bad]   # 既存の誤判定 sex を全クリアしてから反映
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PED_DIR = ROOT / "data" / "local" / "horse_pedigree_5gen"


def _parse_sex(sex_age: str) -> str:
    if not isinstance(sex_age, str):
        return ""
    if sex_age.startswith("牡"): return "牡"
    if sex_age.startswith("牝"): return "牝"
    if sex_age.startswith("セ"): return "セ"
    return ""


def main(clear_bad: bool = False) -> int:
    print("[load] race_result_flat (全年) ...", flush=True)
    files = sorted(glob.glob(str(ROOT / "data/local/tables/*/race_result_flat.parquet")))
    dfs = [pd.read_parquet(f, columns=["horse_id", "sex_age"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["horse_id"] = df["horse_id"].astype(str)
    df["sex"] = df["sex_age"].apply(_parse_sex)
    df = df[df["sex"] != ""]
    horse_to_sex = df.drop_duplicates("horse_id").set_index("horse_id")["sex"].to_dict()
    print(f"  race_result_flat に存在: {len(horse_to_sex):,} 頭")

    # 1. 全 5gen ファイルを走査
    n_total = 0
    n_updated = 0
    n_cleared = 0
    n_no_match = 0
    n_already_correct = 0
    sex_stat_after = {"牡": 0, "牝": 0, "セ": 0, "": 0}

    for f in PED_DIR.rglob("*.json"):
        n_total += 1
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        hid = str(rec.get("horse_id") or "").strip()
        if not hid:
            continue
        target_sex = horse_to_sex.get(hid)
        old_sex = str(rec.get("sex") or "").strip()

        if target_sex:
            # race_result_flat にある → 確実な sex で上書き (現在値が違えば更新)
            if old_sex != target_sex:
                rec["sex"] = target_sex
                f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                n_updated += 1
            else:
                n_already_correct += 1
        else:
            # race_result_flat に無し (= 競走経験なし)
            if clear_bad and old_sex:
                # 既存の誤判定 (sex='牝' 等) は信頼できないのでクリア
                rec["sex"] = ""
                f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                n_cleared += 1
            n_no_match += 1

        sex_stat_after[str(rec.get("sex") or "")] = (
            sex_stat_after.get(str(rec.get("sex") or ""), 0) + 1
        )

    print(f"\n=== 結果 ===")
    print(f"  全 5gen ファイル数:                {n_total:,}")
    print(f"  race_result_flat ベースで更新:    {n_updated:,}")
    print(f"  すでに正しかった:                 {n_already_correct:,}")
    print(f"  race_result_flat に無し:          {n_no_match:,}")
    if clear_bad:
        print(f"  誤判定をクリアした (空に):        {n_cleared:,}")
    print()
    print(f"  反映後 sex 分布:")
    for k, v in sex_stat_after.items():
        print(f"    '{k or '(空)'}': {v:,} ({v/max(n_total,1)*100:.1f}%)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear-bad", action="store_true",
                    help="race_result_flat に存在しない馬の既存 sex (誤判定の可能性大) をクリア")
    args = ap.parse_args()
    raise SystemExit(main(clear_bad=args.clear_bad))
