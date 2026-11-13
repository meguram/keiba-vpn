"""種牡馬系統の分析のまとめページ（build_sire_line_hub_html）: タブ・埋め込み・図・未計算のとき。"""

import html
import re

from src.scripts.docs import build_sire_line_hub_html as H

POS = ("父", "母父", "母母父", "母母母父")


def _inf(v):
    return {"n": 100, "full_r2": v * 2, "shapley": {p: v / (i + 1) for i, p in enumerate(POS)}, "alone": {p: v for p in POS},
            "p": {p: 0.05 for p in POS}, "null95": {p: 0.001 for p in POS}, "ci90": {p: [0, v] for p in POS}}


def _results():
    traits = {"broad": ["芝ダ（芝−ダ）"], "fine": ["TS"], "place": ["芝の出走割合"]}
    lines = ["Aサンデー系", "B系", "C系"]
    e = {"cols": ["TS", "ギア"], "lines": lines, "n": {a: 40 for a in lines},
         "profile": {a: {"TS": 0.1 * i, "ギア": -0.1 * i} for i, a in enumerate(lines)},
         "xy": {a: [i, -i] for i, a in enumerate(lines)}, "dendro_order": ["B系", "Aサンデー系", "C系"],
         "linkage": [[0, 1, 0.5, 2], [2, 3, 1.0, 3]], "near": {a: [[b, 0.3] for b in lines if b != a] for a in lines},
         "nn_stability": {a: {} for a in lines}}
    return {"built": "2026-11-13 12:00", "n_horses": 100, "n_runs": 900, "coverage": {p: 0.9 for p in POS}, "gene_share": {"父": .5, "母父": .25},
            "traits": traits, "reliability": {"TS": {"reliability": 0.7}},
            "A_line": {t: _inf(0.01) for g in traits.values() for t in g}, "A_stallion": {}, "A_line_complete": {}, "n_complete": 50,
            "D": {"TS": {"父": {"B系": [-0.2, 40], "C系": [0.3, 40]}, "母父": {"B系": [0.1, 40]}}}, "E": {"父": e, "母父": e}}


SRC = "<html><body><script>var a = 1 < 2 && '</textarea>' && \"q\";</script><p id=\"x\">中身 &amp;</p></body></html>"


def _embeds(tmp_path):
    p = tmp_path / "m.html"
    p.write_text(SRC, encoding="utf-8")
    return {"matrix": p, "lines": p, "marking": tmp_path / "none.html"}


def test_tabs_and_embeds(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REPO", tmp_path)
    out = H.build(_results(), {"summary": "# 見出し\n- **太字** と `code`\n  - 入れ子", "euro": "本文"}, _embeds(tmp_path), {"B系": "#00ffff"})
    assert len(re.findall(r'<section class=slh-tab data-tab="', out)) == 7
    assert len(re.findall(r'<button type=button data-tab="', out)) == 7
    # 埋め込みは textarea の中でエスケープされ、元の HTML の </textarea> でページが切れない
    srcs = re.findall(r"<textarea class=src[^>]*>(.*?)</textarea>", out, re.S)
    assert len(srcs) == 2
    assert html.unescape(srcs[0]) == SRC                 # 元の HTML がそのまま戻る（途中の </textarea> で切れていない）
    assert "<script>" not in srcs[0] and 'id="x"' not in srcs[0]
    assert "まだありません" in out                        # マーキングの元ページが無い
    assert "<b>太字</b>" in out and "<code>code</code>" in out


def test_linemap_svg_has_every_line(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REPO", tmp_path)
    out = H.build(_results(), {}, _embeds(tmp_path), {})
    sec = out.split('<section class=slh-tab data-tab="linemap"')[1].split("</section>")[0]
    svg = sec.split("<svg class=map")[1].split("</svg>")[0]
    for a in ("Aサンデー系", "B系", "C系"):
        assert f">{a}</text>" in svg
    assert sec.count("<svg class=dendro") == 2            # 父・母父の 2 通り
    assert sec.count('class=mapblk data-map="1" hidden') == 1


def test_missing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REPO", tmp_path)
    out = H.build(None, {}, _embeds(tmp_path), {})
    for tab in ("influence", "linemap"):
        assert "未計算" in out.split(f'<section class=slh-tab data-tab="{tab}"')[1].split("</section>")[0]


def test_md_to_html_escapes():
    h = H.md_to_html("# a<b>\n- x & y")
    assert "<h2>a&lt;b&gt;</h2>" in h and "x &amp; y" in h
