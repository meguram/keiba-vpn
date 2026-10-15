import polars as pl

from src.scraper.parse_drift import drift


def _write(tmp, year, rows):
    d = tmp / str(year)
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(d / "x_flat.parquet")


def _races(year, n, month, **over):
    out = []
    for i in range(n):
        r = {"race_id": f"{year}{month:02d}{i:04d}", "date": f"{year}-{month:02d}-10", "surface": "芝", "race_name": "ok", "horse_name": "a"}
        r.update({k: (v(i) if callable(v) else v) for k, v in over.items()})
        out.append(r)
    return out


def test_detects_column_that_suddenly_becomes_empty_and_mojibake(tmp_path):
    base = sum((_races(2024, 60, m) for m in range(1, 13)), [])
    _write(tmp_path, 2024, base)
    recent = _races(2026, 60, 1) + _races(2026, 60, 2, surface="") + _races(2026, 60, 3, race_name="��")
    _write(tmp_path, 2026, recent)
    f = drift(tmp_path, [2024], [2026], months=3)
    keys = {(x["column"], x["kind"], x["month"]) for x in f}
    assert ("surface", "空", "2026-02") in keys
    assert ("race_name", "文字化け", "2026-03") in keys
    assert not any(x["month"] == "2026-01" for x in f)


def test_no_findings_when_stable(tmp_path):
    _write(tmp_path, 2024, sum((_races(2024, 60, m) for m in range(1, 13)), []))
    _write(tmp_path, 2026, _races(2026, 60, 1) + _races(2026, 60, 2))
    assert drift(tmp_path, [2024], [2026]) == []
