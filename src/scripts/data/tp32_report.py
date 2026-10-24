"""tp-32 の結果を docs の 1 章（equipment_and_change.html）と /tmp/tp32_report.{md,json} にする。

python -m src.scripts.data.tp32_report   # ホストで実行（data/local/modeling/training_pipeline_v4/tp32/ の結果を読む）
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp32"
CHAPTER = ROOT / "docs/html/modeling/chapters/equipment_and_change.html"

JP = {"EQ": "馬具（ブリンカー）: 当日の着用・初着用・外した・着用の継続・着用時と非着用時の成績の差", "DIST": "距離の延長・短縮の段階（±200/400/600m 以上）と、その変更での過去成績",
      "SURF": "芝↔ダートの替わり × 距離の延長・短縮（芝短縮替わり、ダート延長替わり など）", "FIRST": "初経験: 初芝・初ダート・初距離帯・初コース・初の左右回り・初の重/不良・初の長距離/短距離・初の多頭数・初の競馬場",
      "REST": "休み明け（日数帯）× 叩き何戦目 〔tp-34 の担当。参考〕", "CLASS": "昇級初戦・降級初戦・格上挑戦", "TRAN": "転厩初戦（調教師替わり）と転厩後の走数",
      "JOCKEY": "乗り替わり、前走騎手との成績率の差（格の差）、継続騎乗", "GELD": "去勢明け", "WEIGHT": "馬体重の大幅増減・斤量の変化", "TRANSPORT": "東西の遠征（美浦↔栗東と開催場）〔tp-33 の担当。参考〕",
      "AGE": "年齢の節目（3 歳の春/夏/秋、古馬初戦）", "COMBO": "状況の組み合わせ（休み明け初戦×ブリンカー初着用、短縮初戦×昇級、昇級×乗り替わり、芝ダ替わり×距離替わり、去勢明け×休み明け ほか）"}


def _rows(res: dict, key: str = "vs_V4") -> list[dict]:
    out = []
    for g, s in (res.get("groups") or {}).items():
        v = s.get(key) or s.get("vs_V4") or {}
        out.append({"group": g, "cols": s.get("cols"), "dwin": v.get("Δwin"), "hi": v.get("win_hi"), "dtop2": v.get("Δtop2"), "dtop3": v.get("Δtop3"), "adopted": bool(v.get("採用"))})
    return out


def _fmt(x) -> str:
    return "-" if x is None else f"{x:+.5f}"


def build() -> None:
    s1 = json.loads((OUT / "stage1/results.json").read_text(encoding="utf-8")) if (OUT / "stage1/results.json").exists() else {}
    s2 = json.loads((OUT / "stage2/results.json").read_text(encoding="utf-8")) if (OUT / "stage2/results.json").exists() else {}
    r1, r2 = _rows(s1), _rows(s2)
    base1 = (s1.get("base") or {}).get("levels", {})
    base2 = (s2.get("base") or {}).get("levels", {})
    adopted = [r["group"] for r in r2 if r["adopted"]]
    res = {"stage1": r1, "stage2": r2, "base_stage1": base1, "base_stage2": base2, "adopted": adopted, "tried_groups": len(r1)}
    Path("/tmp/tp32_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# tp-32 馬具と『初めて・替わり』系の状況特徴", "", f"試した群 {len(r1)}。基準 V4（1,062 列、新しい土台）: 第 1 段 勝率 {base1.get('logloss_win', float('nan')):.5f}、第 2 段 {base2.get('logloss_win', float('nan')):.5f}。", "",
          "## 第 1 段（valid 2022+2023、seed 1 本）", "| 群 | 列数 | Δ勝率 | CI 上限 | Δ連対 | Δ複勝 | 通過 |", "|---|---|---|---|---|---|---|"]
    passed = set((s2.get("groups") or {}).keys())
    for r in r1:
        md.append(f"| {r['group']} | {r['cols']} | {_fmt(r['dwin'])} | {_fmt(r['hi'])} | {_fmt(r['dtop2'])} | {_fmt(r['dtop3'])} | {'○' if r['group'] in passed else '×'} |")
    md += ["", "## 第 2 段（valid 2024、新 seed 2 本）", "| 群 | 列数 | Δ勝率 | CI 上限 | Δ連対 | Δ複勝 | 採否 |", "|---|---|---|---|---|---|---|"]
    for r in r2:
        md.append(f"| {r['group']} | {r['cols']} | {_fmt(r['dwin'])} | {_fmt(r['hi'])} | {_fmt(r['dtop2'])} | {_fmt(r['dtop3'])} | {'採用' if r['adopted'] else '不採用'} |")
    md += ["", f"採用: {adopted or 'なし'}"]
    Path("/tmp/tp32_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def table(rows: list[dict], last: str) -> str:
        t = "<table><tr><th>群</th><th>内容</th><th>列数</th><th>Δ勝率</th><th>CI 上限</th><th>Δ連対</th><th>Δ複勝</th><th>" + last + "</th></tr>"
        for r in rows:
            mark = ("採用" if r["adopted"] else "不採用") if last == "採否" else ("○" if r["group"] in passed else "×")
            t += (f"<tr><td>{r['group']}</td><td>{html.escape(JP.get(r['group'], ''))}</td><td class='num'>{r['cols']}</td><td class='num'>{_fmt(r['dwin'])}</td>"
                  f"<td class='num'>{_fmt(r['hi'])}</td><td class='num'>{_fmt(r['dtop2'])}</td><td class='num'>{_fmt(r['dtop3'])}</td><td>{mark}</td></tr>")
        return t + "</table>"

    audit = Path("/tmp/tp32_data_audit.md")
    audit_html = "<pre style='white-space:pre-wrap'>" + html.escape(audit.read_text(encoding="utf-8")) + "</pre>" if audit.exists() else ""
    note = (f"第 2 段で採用基準を満たした群: <b>{', '.join(adopted)}</b>。" if adopted else "第 2 段で採用基準を満たした群はありません。")
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>馬具と「初めて・替わり」系の状況特徴（tp-32）— keiba-vpn</title><style>:root {{ --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }}
* {{ box-sizing: border-box; }} body {{ font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }}
a {{ color: var(--accent-2); }} h1 {{ font-size: 26px; }} h2 {{ font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; }} h3 {{ font-size: 16px; margin: 22px 0 8px; }}
table {{ border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); }} th, td {{ border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; }} th {{ background: #1b2a4d; }}
td.num {{ text-align: right; font-family: monospace; }} .muted {{ color: var(--muted); font-size: 13px; }} .note {{ background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }}</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>馬具と「初めて・替わり」系の状況特徴（tp-32）</h1>
<p class="muted">ブリンカーなどの馬具、距離の延長・短縮、芝↔ダートの替わり、初経験、昇級・降級、転厩、乗り替わり、去勢明け、馬体重・斤量の変化、年齢の節目と、それらの組み合わせについて、「その状況かどうか」と「その状況での過去の成績（馬自身の as-of 親和度・全馬の as-of 成績率・初経験フラグ）」を特徴量にして、V4（1,062 列）に対する増分を測った。休み明け・間隔は tp-34、輸送・滞在は tp-33 の担当で、ここでは参考の結果だけを載せる。</p>
<div class="note"><b>要点.</b> {note} 評価は valid 2022〜24 の LightGBM（第 1 段: 2022+2023・seed 1 本、第 2 段: 2024・新しい seed 2 本）。採否は『CI 上限&lt;0・効果≥0.0002・連対/複勝の悪化 0.0002 以内』。test 2025・2026 年は使っていない。</div>
<h2>1. 馬具データの所在と使えるか</h2>
{audit_html}
<h2>2. 第 1 段（valid 2022+2023）</h2>
<p class="muted">基準 V4: 勝率 {base1.get('logloss_win', float('nan')):.5f}。</p>
{table(r1, '通過')}
<h2>3. 第 2 段（valid 2024、新 seed）</h2>
<p class="muted">基準 V4: 勝率 {base2.get('logloss_win', float('nan')):.5f}。</p>
{table(r2, '採否')}
<h2>4. 今後の取得（cron）と前向きの検証</h2>
<ul>
<li>馬柱（race_shutuba_past）の馬名の右の印 B（ブリンカー）を保存するようパーサーを拡張。前日夕方（eve）の馬柱取得で、出走前の blinker が毎開催日 約 400 頭溜まる。</li>
<li>KEIBA_EVE_SMARTRC_SNAPSHOT=1 で、eve の SmartRC 取得時に出走表（runners）を <code>smartrc_prerace</code> に phase ごとに不変で保存（出走後の再取得で上書きされない）。</li>
<li>備考（horse_race_remarks）の日次の増分取得: 毎日 03:00 JST、直近 14 日に走った馬のうち直近の出走の行が無い馬だけ取り直す（<code>scripts/cron/install_horse_remarks_cron.sh</code>）。</li>
<li><code>verify_equipment_snapshot</code> で、出走前の値（スナップショット・馬柱 B）と出走後の smartrc_race を比較し、一致率を測る。2〜3 開催週（約 1,000 頭以上）溜まれば信頼区間つきで判定できる。</li>
</ul>
</body></html>
"""
    CHAPTER.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    build()
