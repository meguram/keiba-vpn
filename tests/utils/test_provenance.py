from src.utils.provenance import public_ok, source_of, stamp


def test_defaults_and_stamp():
    assert source_of({}, "race_shutuba") == "netkeiba" and source_of({}, "jra_cushion") == "jra"
    assert source_of({}, "unknown_category") == "unknown"                    # 不明は安全側
    d = {"race_id": "x"}
    s = stamp(d, "jra")
    assert s["_meta"]["source"] == "jra" and "_meta" not in d                # 元は変えない
    assert source_of(s, "race_shutuba") == "jra"                             # 印が既定より優先


def test_public_ok():
    assert public_ok("own") and not public_ok("netkeiba") and not public_ok("smartrc") and not public_ok("unknown")
    assert not public_ok("jra") and public_ok("jra", jra_ok=True)           # JRA は規約確認が済んでから
