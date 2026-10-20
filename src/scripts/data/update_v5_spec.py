"""保存済みの bundle_v5_<年>/ のアンサンブルの重み（bundle.json の spec）を、tp27_v5 の LOYO の重みに更新する（再学習はしない）。

python -m src.scripts.data.update_v5_spec --fit-end 2025 [--ens v5]
最終学習を LOYO の完了前に始めた場合に使う（その場合は V4 の v4_e2 の重みで保存されている）。更新前の bundle.json は bundle.json.before_loyo に退避し、fit_info.json の ens_source を書き換える。
"""

from __future__ import annotations

import argparse
import json
import shutil

from src.scripts.data import fit_final_v5 as f5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], required=True)
    ap.add_argument("--ens", default="v5")
    a = ap.parse_args()
    d = f5.OUT / f"bundle_v5_{a.fit_end}"
    meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    names = [m["name"] for m in meta["members"]]
    spec, src = f5.load_spec_v5(names, a.ens)
    if "LOYO" not in src or "v5" not in src:
        raise SystemExit(f"LOYO の重みが見つからない: {src}")
    if not (d / "bundle.json.before_loyo").exists():
        shutil.copy2(d / "bundle.json", d / "bundle.json.before_loyo")
    meta["spec"] = spec
    (d / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    fi = json.loads((d / "fit_info.json").read_text(encoding="utf-8"))
    fi["ens_source"], fi["ens"] = src, a.ens
    (d / "fit_info.json").write_text(json.dumps(fi, ensure_ascii=False, indent=1), encoding="utf-8")
    print("更新:", d, "|", src)


if __name__ == "__main__":
    main()
