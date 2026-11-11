

def test_drop_duplicate_entries_keeps_pair_odds_rows():
    """組合せオッズ（odds_type・pair を持ち horse_id が無い）は race_id だけで潰さない。"""
    import pandas as pd
    from src.scraper.export_tables import _drop_duplicate_entries

    df = pd.DataFrame({
        "race_id": ["r1", "r1", "r1", "r1"],
        "odds_type": ["umaren", "umaren", "wide", "umaren"],
        "pair": [[1, 2], [1, 3], [1, 2], [1, 2]],
        "odds": [5.0, 9.0, 2.0, 6.0],
    })
    out = _drop_duplicate_entries(df)
    assert len(out) == 3
    assert out[(out.odds_type == "umaren") & (out.pair.astype(str) == "[1, 2]")]["odds"].iloc[0] == 6.0
