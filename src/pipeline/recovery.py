"""回収率ポスト計算（MASTER §6-3 / AREA-01 §4-3）。"""


def calculate_recovery_rate(
    win_prob: float,
    win_odds: float,
    show_prob: float,
    place_odds_mid: float,
) -> dict[str, float]:
    """T-6/T-7: win_roi / show_roi を小数点2桁で返す。"""
    win_roi = win_prob * win_odds * 100
    show_roi = show_prob * place_odds_mid * 100
    return {"win_roi": round(win_roi, 2), "show_roi": round(show_roi, 2)}


def is_value_bet(expected_win_roi: float | None, expected_show_roi: float | None) -> bool:
    """F-14: 単回収率または複回収率が 100 以上ならバリューベット候補。"""
    win_ok = expected_win_roi is not None and expected_win_roi >= 100
    show_ok = expected_show_roi is not None and expected_show_roi >= 100
    return win_ok or show_ok
