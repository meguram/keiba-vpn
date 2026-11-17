"""推奨馬券の最適化の「過程の物語」の HTML を作る（読み取りのみ・費用 0 円）。

python -m src.scripts.data.bet_story_report

依頼（ユーザー、統括役経由 2026-11-17）: 「これ、特定のhtmlファイルでまとめておいてください。何を思考して、どうなったのか、過程がわかるように。」
サイクルの記録（docs/html/operations/bet_cycles.html）とは別に、何を考えて、何を試し、どうなったかを時系列で書く。
各節は「考えたこと」「やったこと」「結果（数字）」「次にどうしたか」に分ける。数字は記録のファイルから引く（手で写さない）:
- docs/operations/retrain_trials/bet_cycles_log.json（第 1 回の各サイクル・W4・段）
- docs/operations/retrain_trials/bet_test2025_<hash>.json（第 1 回の test 2025）
- docs/operations/retrain_trials/bet_repro_20261117.json（本番の再現: ずれの表・第 1 回の凍結の測り直し・cycle 3 の例）
- docs/operations/retrain_trials/bet_round2_story.json（第 2 回。あれば）
出力: docs/html/operations/bet_optimization_story.html（1 ファイルで完結、外部の JS なし、スマホ幅、ライト／ダーク）
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRIALS = ROOT / "docs/operations/retrain_trials"
LOG = TRIALS / "bet_cycles_log.json"
REPRO = TRIALS / "bet_repro_20261117.json"
ROUND2 = TRIALS / "bet_round2_story.json"
OUT = ROOT / "docs/html/operations/bet_optimization_story.html"


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def pct(x, d: int = 1) -> str:
    return "－" if x is None else f"{x * 100:.{d}f}%"


def table(head: list[str], rows: list[list[str]], left: int = 1) -> str:
    th = "".join(f'<th class="l">{esc(h)}</th>' if i < left else f"<th>{esc(h)}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td class="l">{esc(c)}</td>' if i < left else f"<td>{esc(c)}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def step(title: str, think: str, did: str, result: str, nxt: str, extra: str = "") -> str:
    """1 つの段: 考えたこと・やったこと・結果・次にどうしたか。result と extra は HTML（表など）を入れてよい（呼ぶ側でエスケープ済み）。"""
    return (f'<section class="step"><h3>{esc(title)}</h3>'
            f'<dl><dt>考えたこと</dt><dd>{esc(think)}</dd><dt>やったこと</dt><dd>{esc(did)}</dd>'
            f'<dt>結果</dt><dd>{result}</dd><dt>次にどうしたか</dt><dd>{esc(nxt)}</dd></dl>{extra}</section>')


CSS = """:root{color-scheme:light dark;--bg:#f7f8fa;--panel:#fff;--text:#1d2330;--muted:#5b6578;--border:#d8dde6;--accent:#2f6fde;--good:#1d8a4e;--bad:#b3261e;--hold:#a86400;--head:#eef1f6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--hold:#f0b65a;--head:#1d2638}}
:root[data-theme="dark"]{--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--hold:#f0b65a;--head:#1d2638}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:920px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2em 0}
h2{font-size:1.15rem;margin:1.8em 0 .5em;border-bottom:1px solid var(--border);padding-bottom:.2em}h3{font-size:1rem;margin:0 0 .4em}
.muted{color:var(--muted);font-size:.88rem}.lead{background:var(--panel);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:6px;padding:10px 14px}
.step{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;margin:12px 0}
dl{margin:0}dt{color:var(--muted);font-size:.84rem;margin-top:.5em}dd{margin:0 0 0 .2em}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5em 0}
th,td{border:1px solid var(--border);padding:4px 6px;text-align:right;white-space:nowrap}th{background:var(--head);font-weight:600}td.l,th.l{text-align:left;white-space:normal}
code{font-size:.82rem;background:var(--head);padding:0 .3em;border-radius:3px;word-break:break-all}.good{color:var(--good)}.bad{color:var(--bad)}
ul{padding-left:1.2em}"""


def sel(cyc: list[dict], n: int) -> dict:
    c = next((x for x in cyc if x["n"] == n), None)
    return (c or {}).get("selected") or {}


def render(log: dict, test: dict, repro: dict, r2: dict | None) -> str:
    v6 = (log.get("others") or {}).get("v6e") or log            # 第 1 回の試しの回し（v6e）の各サイクル
    cyc = v6["cycles"]
    fin = log.get("cycles") or []                                 # 第 1 回の最終モデル（final）の各サイクル
    from src.scripts.data.bet_cycles_report import spec_text
    out = [f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>推奨馬券の最適化の過程</title><style>{CSS}</style></head><body><main>",
           "<h1>推奨馬券の最適化の過程 — 何を考え、何を試し、どうなったか</h1>",
           '<p class="muted">数字は記録のファイル（<code>docs/operations/retrain_trials/bet_cycles_log.json</code>・<code>bet_test2025_*.json</code>・'
           '<code>bet_repro_20261117.json</code>）から <code>src/scripts/data/bet_story_report.py</code> が引いて作る。サイクルごとの細かい記録は '
           '<a href="bet_cycles.html">推奨馬券の最適化サイクル</a>。</p>']
    # 1. 依頼とゴール
    out.append("<h2>1. ユーザーの依頼とゴール</h2>")
    out.append('<div class="lead"><ul>'
               "<li>勝率・複勝率の分布の型ごとに、一番良い買い方を 20 サイクルで探す。各サイクルの履歴を追えるように。</li>"
               "<li>必ず出す指標: 全レースのうち何レースに参戦したか（参戦率）、的中率、回収率、平均回収額、中央値回収額。</li>"
               "<li>買い目は本命（◎）を起点に組み、印と矛盾させない（◎ は軸・1 着・本線）。◎ は基本 2 番人気以下から。1 番人気は「ほぼ確実に勝ち負け」のときだけ例外。</li>"
               "<li>訓練データで回収率の高いやり方を見つけ、評価データで変わらず上向きであることを確かめる。test は凍結の後に 1 回だけ。</li>"
               "<li>最終ゴール（第 2 回から）: 評価でも test でも、変わらず回収率 100% を超える抽出の決まり。判定に使うオッズは本番で使える時点のもの。</li></ul></div>")
    # 2. 第 1 回
    out.append("<h2>2. 第 1 回の 20 サイクル（2026-11-16〜17）</h2>")
    out.append('<p class="muted">2-1〜2-4 の数字は、道具を作るために回した試しの回し（本番 v6e の予測、訓練 2022〜23・評価 2024）の値。'
               '採否は最終モデル（訓練 2019〜22・評価 2023〜24）の回しを正とし、2-3・2-5 に最終モデルの値も並べた。</p>')
    b0 = (cyc[0].get("baselines") or [{}])[0]
    s2, s3, s5, s7, s8, s9 = (sel(cyc, n) for n in (2, 3, 5, 7, 8, 9))
    f3, f7 = sel(fin, 3), sel(fin, 7)
    tr = lambda s_: (s_.get("train") or {})                    # noqa: E731
    ev = lambda s_: (s_.get("eval") or {})                     # noqa: E731
    out.append(step(
        "2-1. まず予測 1 位を ◎ にした（cycle 1・5）",
        "予測 1 位を本命にして単勝・複勝・流しを買えば、モデルの力がそのまま回収率に出るはず。",
        "◎ の決め方を 11 通り × 券種 5 通りで総当たり。人気の ◎ は自信のあるレースだけ・少ない点数にも絞った（cycle 5）。",
        f"基準 B0（予測 1 位の単勝を全部買う）は訓練 {pct(tr(b0).get('roi'))}・評価 {pct(ev(b0).get('roi'))}。"
        f"予測 1 位の ◎ はどう絞っても届かず、最良でも {pct(tr(s5).get('roi'))}（cycle 5）。",
        "人気の ◎ では市場に勝てない。人気になっていないが来る馬（市場の注目の穴）を ◎ にする方向へ。"))
    out.append(step(
        "2-2. 穴の ◎ と見送りで上がった（cycle 2・3）",
        "市場より予測が高い穴の馬を ◎ にし、期待値の低いレースを見送れば、払戻の分で回収率が上がる。",
        f"cycle 2 の選んだ形: {spec_text(s2.get('spec'))}。cycle 3 で、期待値の最大が低いレースの見送りを足した。",
        f"cycle 2 で訓練 {pct(tr(s2).get('roi'))}（区間の下限 {pct(tr(s2).get('roi_lo'))}）。cycle 3 で訓練 {pct(tr(s3).get('roi'))} まで上がった。",
        f"ただし cycle 3 の形（{spec_text(s3.get('spec'))}）は的中 {pct(tr(s3).get('hit_rate'))}。評価でどうなるかを見ることにした。"))
    out.append(step(
        "2-3. 1 点買いは評価で崩れた（cycle 3・7）",
        "的中率の低い買い方は、少数の高配当に頼っていて、年が変わると崩れるのではないか。",
        "選んだ 1 本を評価 2024 で確かめた（評価で選び直しはしない）。",
        f"試しの回し: cycle 3 は訓練 {pct(tr(s3).get('roi'))} → 評価 {pct(ev(s3).get('roi'))}、cycle 7 は訓練 {pct(tr(s7).get('roi'))} → 評価 {pct(ev(s7).get('roi'))}。"
        f"（最終モデルの回しでは、cycle 3 は訓練 {pct(tr(f3).get('roi'))} → 評価 {pct(ev(f3).get('roi'))}、cycle 7 は訓練 {pct(tr(f7).get('roi'))} → 評価 {pct(ev(f7).get('roi'))} で、"
        f"選ばれた形も違う。崩れ方はモデルと期間で変わる。）",
        f"選び方を頑健にした（的中率 8% 以上、上位 5 件を除いた回収率と年ごとの最小で順位）。cycle 8 は的中 {pct(tr(s8).get('hit_rate'))}・訓練 {pct(tr(s8).get('roi'))}。"))
    sv = (v6.get("stages") or {}).get("stage_selcv") or {}
    srows = sorted([r for r in sv.get("rows", []) if r.get("score") is not None], key=lambda r: -r["score"])
    rows = [[str(r.get("cycle")), r.get("kind", ""), pct((r.get("cv") or {}).get("roi")), pct(r.get("score"))] for r in srows[:6]]
    r3 = next((r for r in sv.get("rows", []) if r.get("cycle") == 3 and str(r.get("kind", "")).startswith("固定")), {})
    out.append(step(
        "2-4. 選び方の交差検証で、標本内の楽観が分かった（cycle 14）",
        "数百本の買い方から一番を選ぶと、偶然に良かったものを選ぶ。選び方そのものを年で交差検証しないと、本当の力が分からない。",
        "各サイクルの格子の中から、ある年だけで選んで別の年に当てる。最後の候補は、この値で決める決まりにした（test の前に固定）。",
        f"標本内の値は大きく下がった（例: cycle 3 は標本内 {pct(tr(s3).get('roi'))} → 選び方の交差検証 {pct((r3.get('cv') or {}).get('roi'))}。"
        "1 番人気の方針に合う候補だけで数え直した値）。固定の格子は年ごとに違う形が選ばれ、安定しなかった。"
        + table(["回", "種類", "交差検証の回収率", "物差し（上位 5 件除き・年ごとの最小）"], rows, left=2),
        "型ごとに買い方を選ぶ方策（期待値の最大の 5 分位で型を分ける）が、交差検証で一番になった。"))
    tp = (test.get("periods") or {}).get("test") or {}
    tm = tp.get("metrics") or {}
    tb = next((v["roi"] for k, v in (tp.get("baselines") or {}).items() if k.startswith("B0")), None)
    d = tp.get("diff_vs_B0") or [None, None, None]
    out.append(step(
        "2-5. 最終モデルで回し直し、凍結して test 2025 を 1 回だけ見た（cycle 15〜20）",
        "最後の材料（最終モデルの p_top 入りの OOF、訓練 2019〜22・評価 2023〜24）で、同じ決まりで選び直す。",
        "1 番人気の方針に合わない形を候補から外し、凍結を commit してから test 2025 を 1 回だけ見た。",
        f"test 2025 は回収率 {pct(tm.get('roi'))}（区間 {pct(tm.get('roi_lo'))}〜{pct(tm.get('roi_hi'))}）、的中 {pct(tm.get('hit_rate'))}、"
        f"参戦 {(tm.get('bet_races') or 0):,} / {(tm.get('races') or 0):,}。基準 B0 は {pct(tb)}、差は {(d[0] or 0) * 100:+.1f} 点（区間 {(d[1] or 0) * 100:+.1f}〜{(d[2] or 0) * 100:+.1f}）。",
        "結論: 基準よりはっきり上だが、回収率は 100% 前後。ここでユーザーから「確定オッズはレース後まで分からない。本番をどう再現してテストするのか」と問われた。"))
    # 3. W4
    w4 = log.get("w4") or v6.get("w4") or {}
    out.append("<h2>3. 今の画面の推奨の、印のずれ（W4）</h2>")
    out.append(step(
        "今の推奨は、モデルの ◎ と同じ馬を本線にしているか",
        "今の推奨は期待値で買い目を作り、買い目から印を決め直している。モデルの ◎ と食い違っているのではないか。",
        "今の推奨（target_bets）を訓練 2022〜23 で手元で動かし、買い目から決めた ◎ とモデルの ◎ を比べた。",
        f"参戦 {pct((w4.get('bet_races') or 0) / max(w4.get('races') or 1, 1), 0)} のうち、表示の ◎ がモデルの ◎ と違うのが {pct(w4.get('w4_rate'))}、"
        f"モデルの ◎ がどの買い目にも入っていないのが {pct(w4.get('model_honmei_absent_rate'))}。",
        "最適化した買い方は ◎ を必ず軸に置き、印と買い目の矛盾は 0 になるよう作った。"))
    # 4. 本番の再現
    out.append("<h2>4. 本番の再現（発走前のオッズ）を入れた理由と作り方</h2>")
    bands = repro.get("drift_by_band") or []
    out.append(step(
        "4-1. 確定オッズで判定するのは、本番ではできない",
        "◎ の決め方・期待値・人気・見送りは、買う時点（発走の約 60 分前）のオッズでしか決められない。確定オッズは払戻を数えるときだけに使うべき。",
        f"2026 年の発走前オッズ（{repro.get('n_races_drift', 0)} レース）で、T-60 と確定の対数の比を人気帯ごとに集めた。",
        "人気薄ほど、T-60 から確定にかけてオッズが大きく上がる。"
        + table(["人気帯", "件数", "確定 ÷ T-60（対数の平均を倍に直した値）", "確定 ÷ T-60 の 10%〜90%（倍）"],
                [[b["band"], f"{b['n']:,}", f"約 {b['ratio']:.2f} 倍", f"{__import__('math').exp(b['p10']):.2f}〜{__import__('math').exp(b['p90']):.2f} 倍"] for b in bands], left=1),
        "確定オッズで測った「穴の期待値」は、本番の時点では大きく下がるはず。過去の年を、発走前のオッズで判定した形に直して測り直すことにした。"))
    mth = repro.get("method") or {}
    out.append(step(
        "4-2. 発走前のオッズの作り方（1 版目と、作り直した最終版）",
        "過去の年には確定オッズしかないので、上のずれを確定オッズに逆向きに当てて、発走前のオッズを何通りか作る。",
        f"1 版目: {mth.get('v1', '')} 最終版: {mth.get('final', '')}",
        "判定はその版、精算は確定の払戻。すべての表に「確定オッズで判定（楽観）」と「発走前で判定（本番の再現）」を並べ、採否は本番の再現で決める。",
        f"偏りの注意を記録に書いた: 予測の p は確定の時点のまま、ずれの分布の元の {repro.get('n_races_drift', 0)} レースは 2026 年の確かめにも使う、版を並べた平均なので 1 日の振れは小さく出る。"))
    # 5. 結果
    fz = repro.get("frozen_v1_repro") or {}
    ex = repro.get("cycle3_example") or {}
    rows = []
    for nm, lab in (("train", "訓練 2019〜22"), ("eval", "評価 2023〜24")):
        x = fz.get(nm) or {}
        r_ = x.get("repro") or {}
        rows.append([lab, f"{pct(r_.get('roi'))}（{pct(r_.get('roi_lo'))}〜{pct(r_.get('roi_hi'))}）", pct(x.get("final_odds_roi")), pct(x.get("B0_repro")),
                     pct(r_.get("hit_rate")), f"{(r_.get('bet_races') or 0):,} / {(r_.get('races') or 0):,}"])
    out.append("<h2>5. 本番の再現で測り直した結果</h2>")
    out.append(step(
        "第 1 回で凍結した買い方を、本番の時点のオッズで判定すると",
        "確定オッズで判定した値は楽観ではないか。本番の時点で判定しても 100% を超えるか。",
        "第 1 回の凍結した買い方（期待値の最大の 5 分位で型を分ける方策）を、本番の再現（5 通り）で訓練・評価に当てた。test は見ていない。",
        table(["期間", "本番の再現（区間）", "確定で判定（楽観）", "B0（再現）", "的中率", "参戦 / 全レース"], rows, left=1)
        + f"<p>試しの回し（v6e、訓練 2022〜23・評価 2024）の cycle 3 の最良（確定で判定 {pct(ex.get('final_odds_best_train_roi'))}）は、本番の再現（2 通りの版の試し）で選び直すと訓練 "
          f"{pct((ex.get('repro_best_train') or {}).get('roi'))}・評価 {pct((ex.get('repro_best_eval') or {}).get('roi'))}。</p>",
        "学び: 確定オッズで見えた穴の期待値は、本番の時点では見えない。第 1 回の結論（基準よりは上、100% 前後）は本番の再現でも保たれるが、"
        "100% を安定して超える形にはなっていない。第 2 回は、本番の再現の値で 100% 超えを目指す。"))
    # 6. 第 2 回
    out.append("<h2>6. 第 2 回（第 2 段のモデル）の方針と、これから</h2>")
    if r2:
        for st in r2.get("steps", []):
            out.append(step(st.get("title", ""), st.get("think", ""), st.get("did", ""), esc(st.get("result", "")), st.get("next", "")))
    else:
        out.append(step(
            "第 2 段のモデル（市場とのずれを学習）で、もう一度 20 サイクル",
            "第 2 段のモデルは市場とのずれ r = log(p/q) を学んでいる。r が大きい馬を ◎ にすれば、市場の注目の穴をより正しく選べるはず。",
            "◎ の決め方に r を使う候補を足した（2 番人気以下で r が最大、1 番人気の例外も r で）。最初から頑健な選び方・1 番人気の方針・本番の再現で回す。"
            "2025 年 test は 2 回目の確かめ（別のモデル・別の買い方）として 1 回だけ、2026 年（とくに本物の T-60 で判定する版）を、より手つかずの確かめとして並べる。",
            "（材料が届いたら回して、ここに追記する）",
            "第 2 回の材料（第 2 段のモデルの予測）を待っている。"))
    # 7. 限界と判断
    out.append("<h2>7. 限界と、ユーザーに判断してほしいこと</h2>")
    out.append('<div class="lead"><ul>'
               "<li>固定の買い方は、年ごとに違う形が選ばれ、選び方が安定しない。方策の値には、同じサイクルの方策どうしから選び直した楽観が残る。</li>"
               "<li>評価 2023〜24 は何度も見てきたので手つかずではない。手つかずの確かめは test 2025（第 1 回で 1 回使った）と、2026 年の本物の T-60 の版。</li>"
               "<li>本番の再現は、予測の p を確定の時点のまま使う近似。発走前の予測が過去の年に無いため。</li>"
               "<li>判断してほしいこと: 第 2 回で 100% を安定して超える形が見つからない場合、推奨は「基準よりは上の、100% 前後の買い方」として出すか、出さないか。"
               "前向きの記録（rule_tracking）に足して、本物の T-60 で確かめ続けるか。</li></ul></div>")
    out.append("</main></body></html>")
    return "\n".join(out)


def main() -> None:
    log = json.loads(LOG.read_text(encoding="utf-8"))
    tests = [json.loads(x.read_text(encoding="utf-8")) for x in sorted(TRIALS.glob("bet_test2025_[0-9a-f]*.json"))]
    tests = [t for t in tests if int(t.get("round", 1) or 1) == 1 and t.get("status") == "done"]     # 第 1 回の test（第 2 回は別の節）
    test = tests[0] if tests else {}
    repro = json.loads(REPRO.read_text(encoding="utf-8")) if REPRO.exists() else {}
    r2 = json.loads(ROUND2.read_text(encoding="utf-8")) if ROUND2.exists() else None
    OUT.write_text(render(log, test, repro, r2), encoding="utf-8")
    print(f"書いた: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
