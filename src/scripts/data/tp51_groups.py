"""tp-51: V4 の特徴量（v4_frame の 1,062 列）を意味のあるグループに分ける（列名・ブロック定義から機械的に）。

python -m src.scripts.data.tp51_groups table [--cols JSON] [--out CSV]    # 列→グループの表（CSV）と件数
グループ（16）: 列名の接頭辞と、feature_families.json / v4_columns.json（tp-27）のブロックから決める。相対化列（*_rz / *_rrank / *_rgap）は元の列と同じグループ。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RULES: list[tuple[str, str]] = [
    # (グループ, 正規表現)  上から順に最初に合うものを採用
    ("TL", r"^tl_"),
    ("STAGE_AUX", r"^(s_(time|rank)|mtl_)"),
    ("PED", r"^(pd_|sire|dam|sire_x|dam_sire)"),
    ("CONN_OWNER", r"^(o_|ob_)"),
    ("CONN_JKTR", r"^(jockey_prior|trainer_prior|pair_|n_jockeys)"),
    ("FIELD_MEM", r"^fm_"),
    ("FIELD_DEMO", r"^(fd_|fdr_)"),
    ("FIELD_CONN", r"^fc_"),
    ("FIELD_ENV", r"^(fe_|fr_|frn_)"),
    ("FIELD_PACE", r"^(f1_|f2_|f3_|f5_|frp_)"),
    ("EXP", r"^exp_"),
    ("STATE", r"^(d_|rest_|chg_|weight|age$|sex$|jockey_weight|weight_change)"),
    ("HIST_SPEED", r"^(a_|sr_|pc_)"),
    ("AFFINITY", r"^b_"),
    ("HIST_RESULT", r"^(horse_prior|c_|e_|lag\d|margin|attr|seq_|td_)"),
    ("COND", r"^(distance|field_size|round|horse_number|bracket_number|start_hour|surface|direction|grade|race_class|weather|track_condition|venue_code|weight_rule|course_type|track_|venue|race)"),
]
REL_SUFFIX = re.compile(r"(_rz|_rrank|_rgap)$")
GROUPS = [g for g, _ in RULES] + ["OTHER"]


def assign(cols: list[str]) -> dict[str, str]:
    out = {}
    for c in cols:
        base = REL_SUFFIX.sub("", c)
        for g, pat in RULES:
            if re.match(pat, base):
                out[c] = g
                break
        else:
            out[c] = "OTHER"
    return out


def group_columns(cols: list[str]) -> dict[str, list[str]]:
    a = assign(cols)
    d: dict[str, list[str]] = {g: [] for g in GROUPS}
    for c in cols:
        d[a[c]].append(c)
    return {g: v for g, v in d.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table"])
    ap.add_argument("--cols", default="/home/jovyan/work/tp51_out/v4_frame_cols.json")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cols = json.loads(Path(a.cols).read_text(encoding="utf-8"))
    asg = assign(cols)
    rows = ["column,group,is_relative"] + [f"{c},{asg[c]},{int(bool(REL_SUFFIX.search(c)))}" for c in cols]
    if a.out:
        Path(a.out).write_text("\n".join(rows) + "\n", encoding="utf-8")
    g = group_columns(cols)
    print(json.dumps({k: len(v) for k, v in g.items()}, ensure_ascii=False))
    print("OTHER:", g.get("OTHER", [])[:40])


if __name__ == "__main__":
    main()
