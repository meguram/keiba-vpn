"""種牡馬系統の分析のまとめページ（build_sire_line_hub_html）: タブ・埋め込み・図・未計算のとき。"""

import html
import json
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
    assert len(re.findall(r'<section class=slh-tab data-tab="', out)) == 8
    assert len(re.findall(r'<button type=button data-tab="', out)) == 8
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
    for tab in ("influence", "linemap", "stats"):
        assert "未計算" in out.split(f'<section class=slh-tab data-tab="{tab}"')[1].split("</section>")[0]


def test_md_to_html_escapes():
    h = H.md_to_html("# a<b>\n- x & y")
    assert "<h2>a&lt;b&gt;</h2>" in h and "x &amp; y" in h


def _stats():
    return {"built": "2026-11-13", "pops": {"all": "全馬", "top_h": "2勝馬", "top_r": "2勝レース"}, "grans": {"小": "小系統", "id": "種牡馬"},
            "positions": ["父", "母父"], "groups": ["基本"], "min_horses": 30, "min_runs_sire": 100,
            "metrics": [{"key": "top3", "label": "複勝率", "group": "基本", "fmt": "pct"}],
            "n": {"all": {"runs": 10, "horses": 2}},
            "overall": {k: ["全体", 2, 10, [0.2], [10]] for k in ("all", "top_h", "top_r")},
            "data": {"all": {"父": {"小": [["</script><b>x系", 40, 200, [0.3], [200]]]}}}}


def test_stats_tab_embeds_json_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REPO", tmp_path)
    out = H.build(_results(), {}, _embeds(tmp_path), {}, _stats())
    m = re.search(r'<script type="application/json" id="slh-stats">(.*?)</script>', out, re.S)
    assert m and json.loads(m.group(1))["data"]["all"]["父"]["小"][0][0] == "</script><b>x系"   # 名前の </script> でページが切れない
    sec = out.split('<section class=slh-tab data-tab="stats"')[1].split("</section>")[0]
    assert 'data-k=pop' in sec and 'data-k=gran' in sec and "全馬" in sec


def test_flow_placement_from_config():
    """本物の形: ジョブ ID と取得タスク名は別（recent-retry-morning → fetch_task.task = recent-retry）。いまは切り替えの記録か担当表で決める"""
    jobs = {"jobs": [
        {"id": "recent-retry-morning", "destination": "vps", "phase": 4, "fetch_task": {"task": "recent-retry"}, "origin": {"cron_job_id": "recent-retry"}},
        {"id": "race-odds-per-race", "destination": "vps", "fetch_task": {"task": "race-odds-t60"}, "extra_tasks": ["race-odds-t15"], "origin": {}},
        {"id": "export-tables-daily", "destination": "gcp", "phase": 3, "origin": {"cron_job_id": "export_tables_daily"}},
        {"id": "sire-line", "destination": "host", "origin": {"cron_job_id": "sire-line"}}]}
    j = {x["id"]: x for x in jobs["jobs"]}
    assert H.placement("recent-retry-morning", j, set(), {}) == ("vps", "host")
    assert H.placement("recent-retry-morning", j, {"recent-retry"}, {}) == ("vps", "vps")         # 担当表は取得タスク名で照合
    assert H.placement("recent-retry-morning", j, {"recent-retry-morning"}, {}) == ("vps", "host")  # ジョブ ID では照合しない
    assert H.placement("race-odds-per-race", j, {"race-odds-t15"}, {}) == ("vps", "vps")            # 追加のタスクも見る
    assert H.placement("export-tables-daily", j, set(), {}) == ("gcp", "host")                     # 段階ではなく切り替えの記録で決める
    assert H.placement("export-tables-daily", j, set(), {"export_tables_daily": "gcp"}) == ("gcp", "gcp")
    assert H.placement("sire-line", j, set(), {}, "vps") == ("vps", "host")                         # 計画の目標で上書き
    assert H.placement(None, j, set(), {}) == ("host", "host")                                     # 研究用は学習PC に残す
    html_ = H.flow_section(jobs, {"agent_tasks": []}, {"current_phase": 1, "phases": [{"id": 1, "name": "UI を VPS へ"}]}, {"owners": {}})
    assert "移行の段階 1: UI を VPS へ" in html_ and "目標 VPS" in html_ and "いま 学習PC" in html_ and "目標 GCP" in html_
