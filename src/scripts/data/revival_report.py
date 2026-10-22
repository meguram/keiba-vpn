"""復活セットの章（docs/html/modeling/chapters/revival_feature_set.html）を、台帳 CSV と config/feature_sets.json から作る。"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "docs" / "operations" / "revival_feature_ledger.csv"
CONFIG = ROOT / "config" / "feature_sets.json"
CHECK = ROOT / "docs" / "operations" / "revival" / "check_sets.json"
OUT = ROOT / "docs" / "html" / "modeling" / "chapters" / "revival_feature_set.html"

HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>復活セット: 過去に不採用だった特徴量を、緩めた基準で分ける — keiba-vpn</title>
<style>:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.num { text-align: right; font-family: monospace; } .muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
code { background: #0b1428; padding: 1px 5px; border-radius: 4px; }</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
"""


def e(x) -> str:
    return html.escape("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))


def f(x, nd: int = 5) -> str:
    return "—" if x is None or pd.isna(x) else f"{float(x):+.{nd}f}"


def tbl(df: pd.DataFrame, cols: list[tuple[str, str]], num: set[str] | None = None) -> str:
    num = num or set()
    s = "<table><tr>" + "".join(f"<th>{h}</th>" for h, _ in cols) + "</tr>\n"
    for _, r in df.iterrows():
        s += "<tr>" + "".join(f"<td class='num'>{e(r[c])}</td>" if c in num else f"<td>{e(r[c])}</td>" for _, c in cols) + "</tr>\n"
    return s + "</table>\n"


def build(df: pd.DataFrame, cfg: dict, chk: dict | None) -> str:
    st = df["status"].value_counts()
    cand = df[df["kind"].eq("group")]
    pool = cand[~cand["status"].isin(["採用済み"]) & ~cand["status"].str.startswith("不採用のまま(")]
    n_s = {k: int(pool[k].sum()) for k in ("S1", "S2", "S3", "S4", "S5")}
    r = df[df["status"].str.contains("R1|R2", regex=True) & df["kind"].eq("group")].copy()
    r["level"] = r["status"].str.extract(r"(R[12])")[0]
    r["rep"] = r["redundant_with"].fillna("").eq("")
    reps = r[r["rep"]].sort_values(["level", "d_win"])
    sets = cfg["sets"]
    s = HEAD
    s += "<h1>復活セット: 過去に不採用だった特徴量を、緩めた基準で分ける</h1>\n"
    s += ("<p class='muted'>ユーザー依頼（2026-10-22）: 『今までの採用検討ログをもとに、復活する特徴量を分けておいてほしい。最終的にそれあり（復活セット）となし（今までの基準）で精度比較したい』。"
          "学習はしていない。台帳 <code>docs/operations/revival_feature_ledger.csv</code>、定義 <code>config/feature_sets.json</code>、"
          "読み出し <code>src/pipeline/features/feature_sets.py</code>、集計 <code>src/scripts/data/revival_scan.py</code>・<code>revival_ledger.py</code>。</p>\n")
    s += "<div class='note'><b>要点.</b><ul>\n"
    s += (f"<li><b>台帳 {len(df)} 行</b>（群 {int((df['kind'] == 'group').sum())}・束 {int((df['kind'] == 'bundle').sum())}・構成 {int((df['kind'] == 'composition').sum())}）。"
          f"採用済み {int(st.get('採用済み', 0))}、<b>R1 候補 {int((r['level'] == 'R1').sum())}</b>、<b>R2 候補 {int((r['level'] == 'R2').sum())}</b>"
          f"（代表に絞ると R1 {int(((reps['level'] == 'R1')).sum())}・R2 {int(((reps['level'] == 'R2')).sum())}）、"
          f"リーク・市場依存・既存と重複で不採用のまま {int(sum(v for k, v in st.items() if k.startswith('不採用のまま(')))}、年別が取れず要再集計 {int(df['needs_recompute'].sum())}。</li>\n")
    s += (f"<li><b>復活セットは 3 つ</b>: R1（{len(sets['revival_r1']['add'])} 群）、R1+R2（{len(sets['revival_r1r2']['add'])} 群）、広い版（{len(sets['revival_wide']['add'])} 群・"
          "弱いシグナルの束ねを見る）。基準は v6e（<code>z:RB@e</code>）。</li>\n")
    s += ("<li><b>復活候補はすべて、推論側の生成経路が未整備</b>（採用済みの CY・ALL_BEST・RB・前段 e だけが本番の保存値から作れる）。そのため R に入る群は"
          "『オフライン評価では比べられるが、本番で使うには整備が要る』と読む。工数の目安を 5 節に付けた。</li>\n")
    s += ("<li><b>注意: R1 でも、v6e に足して効く保証はない。</b> 台帳の増分は V4・V5・d_CY_BEST に対するもので、v6e（RB・前段 e）に対する増分ではない。"
          "RB に足して増分が消えた群（クラス基準・展開利・非 JRA 過去走ほか）が実際にあった。また 138 群を検定しているので、R2 の基準（S3・S4）は偶然でも"
          f"一定数が通る（S4 は片側で約 16% の割合）。<b>比較の主役は R1 と、束ねた広い版で、R2 は参考</b>。</li>\n")
    s += "</ul></div>\n"
    s += "<h2>1. 基準の段階と、通る群の数</h2>\n"
    s += ("<table><tr><th>段階</th><th>条件（勝率の LogLoss の差。負が改善）</th><th>通る群（採用済み・リーク等を除く）</th></tr>\n"
          f"<tr><td>S1（従来）</td><td>片側 95% 上限 &lt; 0 かつ 効果 ≤ −0.0002</td><td class='num'>{n_s['S1']}</td></tr>\n"
          f"<tr><td>S2</td><td>2 期間以上で同符号、かつ点推定 ≤ −0.0002（区間は問わない）</td><td class='num'>{n_s['S2']}</td></tr>\n"
          f"<tr><td>S3</td><td>2 期間以上で同符号、かつ点推定 ≤ −0.0001</td><td class='num'>{n_s['S3']}</td></tr>\n"
          f"<tr><td>S4</td><td>懐疑的な事前分布 N(0, 0.0002²) の事後で P(効果&lt;0) ≥ 0.8</td><td class='num'>{n_s['S4']}</td></tr>\n"
          f"<tr><td>S5</td><td>片側 90% 上限 &lt; 0</td><td class='num'>{n_s['S5']}</td></tr></table>\n")
    s += ("<p class='muted'>どの段階も、連対・複勝の悪化が +0.0002 以内であること（place_ok）を条件にする。期間は、OOF が保存されていれば年（2022・2023・2024）、"
          "無ければ段階（段階 1 = 2022+2023、段階 2 = 2024）。標準誤差は、ログに残る片側 95% 上限から逆算した（OOF から再集計した群は、再集計の標準誤差を 1.4 倍して使う）。"
          "段階の定義と偽陽性率の見積りは feature-screening（別の調査）が扱い、結果が出たら取り込む。</p>\n")
    s += "<h2>2. 復活候補（代表）</h2>\n"
    s += tbl(reps.assign(n=reps["n_cols"].map(lambda v: "—" if pd.isna(v) else int(v))),
             [("区分", "level"), ("群", "id"), ("列数", "n"), ("勝率 Δ", "d_win"), ("期間", "granularity"), ("クラスタ", "cluster"), ("工数", "serving_effort")], {"n", "d_win"})
    s += "<p class='muted'>勝率 Δ は期間の加重平均（負が改善）。クラスタは同じ情報を持つ群の束で、同じクラスタに複数の候補がある場合は、区分（R1 優先）と効果の大きさで代表を 1 つ選んだ。残りは 3 節の広い版にだけ入る。</p>\n"
    red = r[~r["rep"]].sort_values(["level", "d_win"])
    s += "<h3>冗長として代表から外した群（広い版には入る）</h3>\n"
    s += tbl(red.assign(n=red["n_cols"].map(lambda v: "—" if pd.isna(v) else int(v))), [("区分", "level"), ("群", "id"), ("列数", "n"), ("勝率 Δ", "d_win"), ("代表", "redundant_with")], {"n", "d_win"})
    s += "<h2>3. 復活セットの定義（比べる 4 通り）</h2>\n<table><tr><th>名前</th><th>内容</th><th>足す群</th><th>追加列（約）</th><th>v6_eval の構成名</th></tr>\n"
    for k, v in sets.items():
        cols = sum((cfg["groups"].get(a) or {}).get("n_cols") or 0 for a in v["add"])
        name = "z:RB@e" if not v["add"] else "z:RB+" + "+".join(v["add"]) + "@e"
        short = e(name) if len(name) < 90 else e(name[:88]) + "…"
        s += f"<tr><td><code>{k}</code></td><td>{e(v['doc'])}</td><td class='num'>{len(v['add'])}</td><td class='num'>{int(cols)}</td><td><code>{short}</code></td></tr>\n"
    s += "</table>\n"
    s += ("<p class='muted'>基準 <code>baseline</code> は今までの基準（v6e = V4 + tl_ 再取得版 + CY + ALL_BEST + RB、前段 e。バンドルは 1,231 列）。"
          "足す群は、学習表と同じ行キー（race_id, horse_id）の群 parquet で、2022〜2024 年の評価行を欠けなく・重複なく覆うことを確認した"
          + ("（35 群、基準のフレームとの列名の重なりなし）" if chk else "") + "。RS59 は v6_eval が名前で作る。後日談の変種（AF）などは学習表に列が無く、構築が要る"
          f"（{', '.join(cfg.get('unbuilt', [])) or 'なし'}）。</p>\n")
    s += "<h2>4. 分けた理由別の内訳（不採用のまま）</h2>\n"
    why = df[df["status"].str.startswith("不採用のまま")]["status"].value_counts().rename_axis("区分").reset_index(name="件数")
    s += tbl(why, [("区分", "区分"), ("件数", "件数")], {"件数"})
    lk = df[df["leak"].fillna("").ne("") | df["market_dep"].fillna(False) | df["dup_known"].fillna(False)]
    s += tbl(lk.assign(r=lk["leak"].fillna("").where(lk["leak"].fillna("").ne(""), lk["reason"])), [("群", "id"), ("理由", "r")])
    s += ("<p class='muted'>リーク疑い: 前走の消耗の群（R6_prev）は見かけ −0.0038 が端数リーク（修正後 +0.00010）で、台帳の数字は修正前の評価。市場依存: tp-29 は人気を使う。"
          "既存と重複: 木モデルが同じ情報を持つ（tp-15・26）、現行前段と情報が重なる（tp-25a）、タイプの能力差が V5 にある（tp-56）。</p>\n")
    s += "<h2>5. 推論側で再現できるか（要整備）</h2>\n"
    nb = reps.assign(n=reps["n_cols"].map(lambda v: "—" if pd.isna(v) else int(v)))
    s += tbl(nb, [("群", "id"), ("列数", "n"), ("工数", "serving_effort"), ("整備の内容", "serving_note")], {"n"})
    s += ("<p class='muted'>工数: S = 既存の履歴から作れる、M = 状態表の追加（RB と同じ日次更新の仕組み）、L = 取得か再構築が要る。"
          "前段 b 版で、本番の保存値から 9 列を再現できず使えなかった例があるため、採用の判断の前に、推論側のパリティ（過去日で学習側と比べる）を必須にする。</p>\n")
    s += "<h2>6. 比較の実行計画（学習再開後）</h2>\n"
    s += ("<ol>\n<li><b>比べる構成</b>: A = 今までの基準（<code>z:RB@e</code>）、B = A + R1、C = A + R1 + R2、D = A + 広い版。"
          "<code>python -m src.pipeline.features.feature_sets show</code> が各構成名を出す。実行は <code>python -m src.scripts.data.v6_eval run &lt;構成名&gt; --tag rev</code>。</li>\n"
          "<li><b>評価</b>: LOYO（2022・2023・2024）、LightGBM、seed は構成間で共通（0・1、確認で 2・3）。勝率・連対・複勝の LogLoss、レース単位の paired bootstrap。"
          "層別は、初出走・履歴 3 走未満、クラス別、面別。test 2025 は、選んだ 1 構成について 1 回だけ。2026 年は選択に使わない。</li>\n"
          "<li><b>採否</b>: 構成単位で、二段の基準（従来の S1、または緩めた S2 を満たし、連対・複勝の悪化が +0.0002 以内）。D は束ねた効果を見るためで、合計の増分が"
          "シャドウ特徴量の束を上回るかを併せて見る（feature-screening の手順）。</li>\n"
          "<li><b>推論側</b>: 採用候補になった群だけ、5 節の整備を行い、v6_parity と同じ手順で学習側との一致を確認してから shadow に入れる。</li>\n"
          "<li><b>所要の見積り</b>（過去の実績からの目安で、実測ではない）: 1 構成の LOYO 評価（3 年 × 2 seed × 3 目的）は 0.5〜1 時間、4 構成で 2〜4 時間（直列）。"
          "seed 2・3 の確認は 1〜2 時間。通った場合の LOYO アンサンブルと最終学習（v6e と同じ手順）は 8〜9 時間。</li>\n</ol>\n")
    s += "<h2>7. 限界</h2>\n<ul>\n"
    s += ("<li>増分の基準が揃っていない。V4（tp-32〜42）、V5（tp-60）、d_CY_BEST（v6）に対する値で、v6e に対する増分ではない（重複で縮む見込み）。</li>\n"
          f"<li>年別の OOF が保存されていない群は、段階（2022+2023 と 2024）の 2 期間でしか符号を見られない（{int(df['needs_recompute'].sum())} 群が要再集計）。再集計は学習なしでできる範囲を超える（OOF が無い）ため、学習再開後に行う。</li>\n"
          "<li>各群は、過去の第 1 段・第 2 段の選別を通ってきた（または落ちた）ものなので、選択バイアスがある。基準を緩めた S2〜S4 は偽陽性を含む。偽陽性率の見積りは feature-screening のシャドウ特徴量で行う。</li>\n"
          "<li>2026 年は使っていない。市場オッズ・人気は特徴量にしていない（tp-29 の市場版は除外した）。</li>\n</ul>\n")
    s += "<p class='muted'>関連: <a href='feature_gap_audit.html'>特徴量のやり残しの棚卸し</a> / <a href='v6_features.html'>v6 への追加候補の評価</a> / <a href='training_pipeline_decisions.html'>決定ログ</a></p>\n</body></html>\n"
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    df = pd.read_csv(LEDGER)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    chk = json.loads(CHECK.read_text(encoding="utf-8")) if CHECK.exists() else None
    Path(a.out).write_text(build(df, cfg, chk), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
