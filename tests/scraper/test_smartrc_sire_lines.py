"""smartrc の系統の台帳（smartrc_sire_lines）: コード→名前、手元の表からの台帳づくり、父系の祖先での補完。"""

import json

import pandas as pd
import pytest

from src.scraper import smartrc_sire_lines as ssl

MASTER = {
    "large": {"1000011": {"name": "サンデー系"}, "1000010": {"name": "ミスプロ系"}},
    "small": {"2000046": {"name": "ディープ系", "large": "1000011"}, "2000039": {"name": "キングマンボ系", "large": "1000010"}},
    "type": {"3000001": {"name": "日本型サンデー系"}},
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "master.json").write_text(json.dumps(MASTER, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(ssl, "MASTER", tmp_path / "master.json")
    sm = pd.DataFrame({"horse_id": ["h1", "h2", "h3", "h4"], "sire_id": ["S_DEEP", "S_DEEP", "S_KK", "S_NEW"],
                       "sire_name": ["ディープインパクト", "ディープインパクト", "キングカメハメハ", "新種牡馬"]})
    sm.to_parquet(tmp_path / "map.parquet")
    monkeypatch.setattr(ssl, "SIRE_MAP", tmp_path / "map.parquet")
    tp = tmp_path / "tp32"
    tp.mkdir()
    pd.DataFrame({"hcode": ["h1", "h2", "h3"], "f_code": ["c1", "c1", "c2"], "f_name": ["", "", ""],
                  "f_llcode": ["1000011", "1000011", "1000010"], "f_slcode": ["2000046", "2000046", "2000039"],
                  "f_clcode": ["3000001", "3000001", None]}).to_parquet(tp / "career_eq.parquet")
    monkeypatch.setattr(ssl, "TP32", tp)
    return tmp_path / "out.json"


def test_names_fill_large_from_small():
    n = ssl.names({"small": "2000046", "type": "3000001"}, MASTER)
    assert n["small_name"] == "ディープ系" and n["large"] == "1000011" and n["large_name"] == "サンデー系" and n["type_name"] == "日本型サンデー系"


def test_build_from_tables_and_blank_sires_are_left_to_netkeiba(env):
    res = ssl.build(env)
    assert res == {"sires": 3, "known": 2, "missing": 1, "conflicts": 0}
    doc = ssl.load(env)
    assert doc["sires"]["S_DEEP"]["small_name"] == "ディープ系" and doc["sires"]["S_DEEP"]["n"] == 2
    assert not ssl.is_complete(env)                         # S_NEW はまだ smartrc で確かめていない
    doc["tried"] = {"S_NEW": "2026-11-06"}                  # smartrc で確かめたら系統が空欄だった
    ssl._save(doc, env)
    ssl.build(env)
    assert ssl.is_complete(env)                             # 空欄の種牡馬は台帳に入れないが、確かめ終えたので完了
    assert ssl.labels(env) == {"S_DEEP": "ディープ系", "S_KK": "キングマンボ系"}     # S_NEW は netkeiba の系統に任せる


def test_parse_bloods_and_type_entry_keep_colors():
    js = ("x kLBloods:[{code:'1000011',name:'サンデー系',addr:'サンデー',color:'#66ff66'}] kSBloods:[{code:'2000046',llcode:'1000011',name:'ディープ系'}] "
          "kCBloods:[{code:'3000009',name:'欧州型ミスプロ系',addr:'欧ミスプ',color:'#00CCFF'},{code:'3000005',name:'米国型ミスプロ系',addr:'米ミスプ'}] y")
    p = ssl.parse_bloods(js)
    e = ssl._type_entry(p["kCBloods"][0])
    assert e == {"name": "欧州型ミスプロ系", "short": "欧ミスプ", "color": "#00CCFF", "country": "欧州型"}
    assert ssl._type_entry(p["kCBloods"][1])["color"] == "#BA945B"        # 画面に色が無くても、既知の国系統の色で補う（取り直しで消さない）
    with pytest.raises(RuntimeError):
        ssl.parse_bloods("kLBloods:[]")


def test_country_and_colors_table():
    assert ssl._country("欧州型ミスプロ系") == "欧州型" and ssl._country("米国型(その他)") == "米国型" and ssl._country("日本型サンデー系") == "日本型"
    assert ssl._country("その他") is None
    m = {"large": {"1000010": {"name": "ミスプロ系", "color": "Moccasin"}},
         "type": {"3000009": {"name": "欧州型ミスプロ系", "color": "#00CCFF", "country": "欧州型"}}}
    doc = {"sires": {"S1": {"name": "ドゥラメンテ", "large": "1000010", "large_name": "ミスプロ系", "small_name": "キングマンボ系", "type": "3000009", "type_name": "欧州型ミスプロ系"}},
           "missing": ["S_BLANK", "S_NEW"], "tried": {"S_BLANK": "2026-11-06"}}
    t = ssl.colors_table(doc, m, names={"S_BLANK": "空欄の種牡馬"}).set_index("sire_id")
    assert t.loc["S1", "country"] == "欧州型" and t.loc["S1", "country_color"] == "#00CCFF" and t.loc["S1", "large_color"] == "Moccasin"
    assert t.loc["S1", "status"] == "ok" and t.loc["S_BLANK", "status"] == "blank" and t.loc["S_NEW", "status"] == "unchecked"   # 試して空欄／まだ試していない
    assert t.loc["S_BLANK", "name"] == "空欄の種牡馬" and t.loc["S_BLANK", "country"] is None
