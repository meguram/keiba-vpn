"""推論の産地・誕生日の欠けを埋める: serving_v4.fill_attrs_from_profile と update_serving_state.retry_birthplace。"""

import pandas as pd

from src.pipeline.features import serving_v4 as sv4
from src.scripts.data import update_serving_state as uss


def test_parse_birthday_handles_japanese_and_dates():
    b = sv4.parse_birthday(["2024年2月2日", pd.Timestamp("2023-04-05"), None, "不明"])
    assert b.iloc[0] == pd.Timestamp("2024-02-02") and b.iloc[1] == pd.Timestamp("2023-04-05")
    assert pd.isna(b.iloc[2]) and pd.isna(b.iloc[3])


def _attrs():
    return pd.DataFrame({"birthplace": ["浦河町", None, ""], "birthday": [pd.Timestamp("2020-03-01"), pd.Timestamp("2021-04-02"), pd.NaT],
                         "owner": ["A", "B", "C"]}, index=pd.Index(["h1", "h2", "h5"], name="horse_id"))


def test_fill_keeps_existing_values_and_other_columns():
    prof = pd.DataFrame({"horse_id": ["h1", "h2", "h3", "h4", "h5", "h6"],
                         "birthplace": ["別の町", "新ひだか町", "日高町", None, "様似町", ""],
                         "birthday": ["2020年3月1日", "2019年1月1日", "2024年5月6日", "2024年1月1日", "2024年7月7日", "2024年1月1日"]})
    out = sv4.fill_attrs_from_profile(_attrs(), prof)
    assert out.loc["h1", "birthplace"] == "浦河町"                                                   # 既にある馬は変えない
    assert out.loc["h2", "birthplace"] == "新ひだか町" and out.loc["h2", "owner"] == "B"              # 産地だけ欠けていた行は、ほかの列を残す
    assert out.loc["h2", "birthday"] == pd.Timestamp("2021-04-02")                                    # 既にある誕生日は変えない
    assert out.loc["h5", "birthplace"] == "様似町" and out.loc["h5", "birthday"] == pd.Timestamp("2024-07-07")   # 空文字の産地も欠けとして埋める
    assert out.loc["h3", "birthplace"] == "日高町" and out.loc["h3", "birthday"] == pd.Timestamp("2024-05-06")
    assert "h4" not in out.index and "h6" not in out.index                                           # プロフィールにも産地が無い（空文字も）馬は足さない
    assert out.index.is_unique and len(out) == 4


def test_fill_excludes_trained_horses():
    prof = pd.DataFrame({"horse_id": ["h2", "h3"], "birthplace": ["新ひだか町", "日高町"], "birthday": ["2019年1月1日", "2024年5月6日"]})
    out = sv4.fill_attrs_from_profile(_attrs(), prof, exclude={"h2"})
    assert pd.isna(out.loc["h2", "birthplace"]) and out.loc["h3", "birthplace"] == "日高町"           # 学習の期間に走った馬は埋めない


def test_fill_noop_without_profile():
    a = _attrs()
    assert sv4.fill_attrs_from_profile(a, None).equals(a)
    assert sv4.fill_attrs_from_profile(a, pd.DataFrame(columns=["horse_id", "owner"])).equals(a)


class _Store:
    def __init__(self, docs):
        self.docs, self.calls = docs, []

    def load(self, cat, key):
        self.calls.append((cat, key))
        return self.docs.get((cat, key))


def test_retry_birthplace_reads_profile_once_per_horse():
    dyn = pd.DataFrame({"horse_id": ["n1", "n2", "n3", "n4"], "owner": ["A", "B", "C", "D"], "birthplace": [None, "日高町", "", None],
                        "birthday": [None, "2024年4月1日", None, None]})
    st = _Store({("horse_profile", "n1"): {"birthplace": "浦河町", "birthday": "2024年3月1日"}, ("horse_profile", "n3"): {}})
    out = uss.retry_birthplace(dyn, ran={"n1", "n2", "n3"}, storage=st)
    o = out.set_index("horse_id")
    assert o.loc["n1", "birthplace"] == "浦河町" and o.loc["n1", "birthday"] == "2024年3月1日" and o.loc["n1", "bp_retry"]
    assert o.loc["n3", "bp_retry"] and o.loc["n3", "birthplace"] == ""                     # 読んでも無ければ印だけ付ける
    assert not o.loc["n2", "bp_retry"] and not o.loc["n4", "bp_retry"]                     # 産地がある馬・走っていない馬は読まない
    assert sorted(st.calls) == [("horse_profile", "n1"), ("horse_profile", "n3")]
    st2 = _Store({})
    uss.retry_birthplace(out, ran={"n1", "n2", "n3"}, storage=st2)
    assert st2.calls == []                                                                  # 一度読んだ馬は二度と読まない


def test_birthday_as_text_makes_parquet_writable(tmp_path):
    old = pd.DataFrame({"horse_id": ["a"], "birthday": ["2024年2月2日"]})
    new = pd.DataFrame({"horse_id": ["b", "c"], "birthday": [pd.Timestamp("2024-04-21"), None]})   # 推論の表で日付に変わった値が混ざる
    dyn = pd.concat([old, new], ignore_index=True)
    dyn["birthday"] = uss.birthday_as_text(dyn["birthday"])
    dyn.to_parquet(tmp_path / "p.parquet", index=False)
    assert pd.read_parquet(tmp_path / "p.parquet")["birthday"].tolist() == ["2024年2月2日", "2024年4月21日", None]


def test_fill_into_all_nan_float_column_without_warning():
    import warnings

    a = pd.DataFrame({"birthplace": [float("nan")], "birthday": [pd.NaT]}, index=pd.Index(["h1"], name="horse_id"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = sv4.fill_attrs_from_profile(a, pd.DataFrame({"horse_id": ["h1"], "birthplace": ["浦河町"], "birthday": ["2024年1月1日"]}))
    assert out.loc["h1", "birthplace"] == "浦河町"
