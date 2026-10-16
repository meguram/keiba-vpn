"""特徴量を役割ごとのグループに分ける（階層スタッキング tp-18 用）。

L1 では、グループごとに別のモデルを作る（各モデルには CONTEXT = レース条件・枠番も渡す）。L2 がグループの出力を組み合わせる。
"""

from __future__ import annotations

import re

CONTEXT = ["surface", "direction", "grade", "race_class", "weather", "track_condition", "venue_code", "weight_rule", "course_type", "distance", "field_size", "round",
           "start_hour", "horse_number", "bracket_number"]

# ファミリー（feature_families.json）→ グループ
FAMILY_TO_GROUP = {"F3": "RATING", "C": "RATING", "H": "RATING", "A": "STYLE", "F2": "STYLE", "G": "STYLE", "E": "STYLE", "B": "APTITUDE", "F1": "APTITUDE", "F5": "APTITUDE", "D": "CONDITION"}

# ファミリーに入っていない列の割り当て
NAME_TO_GROUP = {
    **{c: "RATING" for c in ["horse_prior_races", "horse_prior_wins", "horse_prior_top3", "horse_prior_win_rate", "horse_prior_top3_rate", "horse_prior_avg_finish"]},
    **{c: "CONNECTIONS" for c in ["jockey_prior_races", "jockey_prior_wins", "jockey_prior_top3", "jockey_prior_win_rate", "jockey_prior_top3_rate",
                                  "trainer_prior_races", "trainer_prior_wins", "trainer_prior_top3", "trainer_prior_win_rate", "trainer_prior_top3_rate"]},
    **{c: "APTITUDE" for c in ["sire_prior_win_rate_surface", "dam_sire_prior_win_rate_surface", "sire_x_track_bias_diff", "dam_sire_x_track_bias_diff",
                               "sire_x_track_bias_weighted", "dam_sire_x_track_bias_weighted"]},
    **{c: "CONDITION" for c in ["weight", "weight_change", "jockey_weight", "age", "sex", "track_bias_cond_bucket", "track_bias_dist_band", "track_bias_winner_3f_same_day_prior",
                                "track_bias_winner_3f_prev_day", "track_bias_winner_3f_diff", "track_bias_winner_3f_weighted"]},
}
GROUPS = ["RATING", "STYLE", "APTITUDE", "CONDITION", "CONNECTIONS"]
_REL = re.compile(r"^(?P<src>.+?)(_rz|_rrank|_rgap)$")


def assign_groups(columns: list[str], families: dict[str, list[str]], stage_cols: list[str], extras: list[str], ped: list[str]) -> dict[str, list[str]]:
    """columns（モデルに入れる全列）→ {グループ: 列}。CONTEXT は含めない。どの列もちょうど 1 つのグループに入る（相対化の列は元の列と同じグループ）。"""
    base_group: dict[str, str] = dict(NAME_TO_GROUP)
    for fam, g in FAMILY_TO_GROUP.items():
        for c in families.get(fam, []):
            base_group[c] = g
    for c in stage_cols:
        base_group[c] = "RATING"
    for c in extras:
        base_group[c] = "CONDITION"
    for c in ped:
        base_group[c] = "APTITUDE"
    out: dict[str, list[str]] = {g: [] for g in GROUPS}
    unknown = []
    for c in columns:
        if c in CONTEXT:
            continue
        g = base_group.get(c)
        if g is None:
            m = _REL.match(c)
            if m:
                g = base_group.get(m.group("src"))
        if g is None:
            unknown.append(c)
            continue
        out[g].append(c)
    if unknown:
        raise ValueError(f"グループが決まらない列: {unknown}")
    return out
