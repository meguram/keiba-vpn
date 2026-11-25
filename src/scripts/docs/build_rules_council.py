"""ルールの会議（担当 rules-council）の議題ごとの頁を、結果の JSON と事前登録から作る（数字を手で写さない）。

頁:
  docs/html/operations/final/rules_council_t60_band.html   RC-1 本物の T-60 で再現の帯を確かめる
  docs/html/operations/final/rules_council_favorites.html  RC-2 一桁オッズ × 全体 × 深さ 1
  docs/html/operations/final/rules_council_daybias.html    RC-3 当日の馬場の偏り
  docs/html/operations/final/rules_council_debut.html      RC-4 初出走の馬の目録に無い列
  docs/html/operations/final/rules_council_summary.html    まとめ（議題の一覧・通算の数え・勝ち筋の台帳）
入力: docs/operations/retrain_trials/rules_council_*.json、config/rules_council/prereg_rc2.json、docs/operations/retrain_trials/rules_council_pages.json（結論の文）
使い方: python -m src.scripts.docs.build_rules_council
費用 0 円（手元のファイルだけ）。
"""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scripts.docs.build_latest_strategy_tables import CSS

ROOT = Path(__file__).resolve().parents[3]
TRI = ROOT / "docs/operations/retrain_trials"
DST = ROOT / "docs/html/operations/final"
PREREG2 = ROOT / "config/rules_council/prereg_rc2.json"
PAGES = TRI / "rules_council_pages.json"
JST = timezone(timedelta(hours=9))


def esc(x) -> str:
    return html.escape(str(x))


def pc(x, d: int = 1) -> str:
    return "－" if x is None else f"{x * 100:.{d}f}%"


def nm(x, d: int = 2) -> str:
    return "－" if x is None else f"{x:.{d}f}"


def table(head: list[str], rows: list[list], num_cols=frozenset()) -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f'<td class="n">{c}</td>' if i in num_cols else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fold(summary: str, body: str) -> str:
    return f'<details><summary>{esc(summary)}</summary><div class="dbody">{body}</div></details>'


def page(title: str, body: list[str]) -> str:
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    head = ['<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>", f"<h1>{esc(title)}</h1>",
            f'<p class="sub">作成 {now}（JST）。担当 rules-council。2025・2026 年の着順・払戻は開いていない。</p>']
    return "\n".join(head + body + ["</main></body></html>"])


def bullets(xs: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"


def build_rc1(T: dict, texts: dict) -> str:
    mv = T["versions"]
    b = ["<h2>結論</h2>", '<div class="box">' + bullets(texts["conclusion"]) + "</div>"]
    rows = []
    for v, xs in mv.items():
        for c, s in xs.items():
            rows.append([esc(v), esc(c), f"{s['本物の母集団の走']:.0f}", pc(s["入れ替わりの割合（q）"]), pc(s["再現どうしの入れ替わり（床）"]),
                         nm(s["向きの偏り（外→内 ÷ 内→外、q）"]), nm(s["q の和の比（再現 ÷ 本物）"], 3),
                         "○" if s["元の線を満たす"] else "×", "○" if s["替えた線を満たす"] else "×"])
    b.append("<h2>数字</h2>")
    b.append(f"<p>本物の T-60 は {T['period'][0]}〜{T['period'][1]} の {T['races']} レース（T-60 の記録がある {T['runners_with_t60']:,} 走）。"
             f"主の版は、開催日の前半（{T['races_first']} レース、〜{T['split_day']} の前日）でずれを作り、後半（{T['races_second']} レース）で比べた。再現 {T['seeds']} 本の中央値。</p>")
    b.append(table(["版", "一桁の上端", "本物の走", "入れ替わり（q）", "再現どうしの床", "向きの偏り", "q の和の比", "元の線", "替えた線"], rows, {2, 3, 4, 5, 6}))
    fb = T["参考: 確定オッズで母集団を決めた場合（後半、< 10）"]
    b.append(f"<p>参考: 確定オッズで母集団を決めると、入れ替わり {pc(fb['入れ替わりの割合（q）'])}・偏り {nm(fb['向きの偏り（外→内 ÷ 内→外、q）'])}"
             "（外→内に偏る＝締め切りに向けて人気になった馬が入る、先読みの向き）。</p>")
    b.append("<h2>線と判定</h2>" + bullets(texts["lines"]))
    b.append("<h2>作り方・再現</h2>" + bullets(["<code>KEIBA_SESSION=rules-council python -m src.research.rules_council.t60_check --seeds 5</code>"
                                                "（結果 <code>docs/operations/retrain_trials/rules_council_t60.json</code>）。",
                                                "読むのは 2026 年の確定オッズ・確定の人気・本物の T-60 のオッズだけ（台帳 eval_only）。費用 0 円。"]))
    return page("本物の T-60 で再現の帯を確かめる（RC-1）", b)


def build_rc2(R: dict, P: dict, texts: dict) -> str:
    s = R["search"]
    base = s["reference"]["baseline_all"]
    sha = hashlib.sha256(PREREG2.read_bytes()).hexdigest()[:12]
    b = ["<h2>結論</h2>", '<div class="box">' + bullets(texts["conclusion"]) + "</div>"]
    b.append("<h2>作り（事前登録の版 3）</h2>" + bullets(texts["design"]))
    sb = P["stage_B"]["result"]["arms"]["1000"]
    b.append("<h2>検出力の模擬（合成・着順なし）</h2>")
    b.append(table(["植えた回収率", "候補になる", "候補かつ確かめの側も通る"],
                   [[k, pc(v, 1), pc(sb["plant_cand_and_confirm"][k], 1)] for k, v in sb["plant_cand"].items()], {1, 2}))
    b.append(f"<p>効きなしで 1 つでも候補になる確率（FWER）{pc(sb['fwer_cand'], 1)}（200 回）。下限 E ≥ 1,000（{P['stage_B']['result']['n_elig']['1000']} 水準）。</p>")
    b.append("<h2>本物の段 S'（探す側 70%）</h2>")
    b.append(f"<p>基準線（探す側の一桁を全部買う）: {base['n']:,} 走・回収率 {pc(base['roi'])}（{pc(base['roi_ci95'][0])}〜{pc(base['roi_ci95'][1])}）・"
             f"市場比 ÷ 帯の比 {nm(base['mr_band'], 3)}。1 段目の線（本物の入れ替えの帰無 200 回の 95% 点）: 増やす向き {nm(s['lines']['0']['t_up'])}・消し {nm(s['lines']['0']['t_down'])}。</p>")
    b.append(f"<p>1 段目を通った増やす向き {s['n_up1']}・消し {s['n_down1']}。2 段目（損益分岐の Holm）を通った水準 {s['n_stage2']}。候補 {s['n_candidates']}。"
             "下の表は並べ替えの帰無を越えたずれの記録で、買い方の候補ではない（回収率はどれも 100% に届かない）。</p>")
    b.append(table(["水準（増やす向き）", "z（帯の比）", "損益分岐の z", "走", "回収率", "市場比 ÷ 帯の比"],
                   [[esc(r["level"]), nm(r["z_band"]), nm(r["z_be"]), f"{r['n']:,}", pc(r["roi"]), nm(r["mr_band"], 3)] for r in s["up1"]], {1, 2, 3, 4, 5}))
    b.append(table(["水準（消しの向き）", "z（帯の比）", "走", "回収率", "市場比 ÷ 帯の比"],
                   [[esc(r["level"]), nm(r["z_band"]), f"{r['n']:,}", pc(r["roi"]), nm(r["mr_band"], 3)] for r in s["down1"]], {1, 2, 3, 4}))
    b.append(table(["再現の種", "検定した水準", "1 段目（増やす向き）", "2 段目"],
                   [[str(1000 + int(k)), str(v["n_elig"]), str(v["up1"]), str(len(v["s2"]))] for k, v in s["replicates"].items()], {1, 2, 3}))
    b.append(fold("基準線の年ごとの回収率", table(["年", "走", "回収率"], [[y, f"{v['n']:,}", pc(v["roi"])] for y, v in base["by_year"].items()], {1, 2})))
    b.append("<h2>見た回数と楽観の元</h2>" + bullets(texts["looks"]))
    b.append("<h2>作り方・再現</h2>" + bullets([
        f"事前登録 <code>config/rules_council/prereg_rc2.json</code>（sha256 {sha}…）。決定は <code>docs/operations/roi_plus_council/decisions.md</code> の RC-0〜RC-2。",
        "<code>python -m src.research.rules_council.rc_frame</code>（全走の表）→ <code>rc2s power --mode k70</code>（検出力の模擬）→ "
        "<code>rc2_real null</code>（帰無の線）→ <code>rc2_real search</code>（段 S'）。<code>KEIBA_SESSION=rules-council</code>・<code>nice -n 19</code>。",
        "結果 <code>docs/operations/retrain_trials/rules_council_rc2.json</code>。費用 0 円（手元の表だけ。GCS は読まない）。"]))
    return page("一桁オッズ × 全体 × 深さ 1 のルール探し（RC-2）", b)


def build_rc3(Bd: dict, texts: dict) -> str:
    b = ["<h2>結論</h2>", '<div class="box">' + bullets(texts["conclusion"]) + "</div>"]
    rows = []
    for k, nm_ in (("inner", "内外（枠 1〜4）"), ("front", "前後（最後の角 3 番手以内）")):
        o = Bd[k]
        for k2, lab in (("ge_2", "有利な側（+2 以上）"), ("ge_3", "有利な側（+3 以上）"), ("le_-2", "逆の側（−2 以下）"), ("le_-3", "逆の側（−3 以下）")):
            v = o[k2]
            t3 = v.get("top3_mult", v.get("top3_mult_other_side"))
            w1 = v.get("win_mult", v.get("win_mult_other_side"))
            rows.append([nm_, lab, f"{v['n']:,}", nm(t3, 3), nm(w1, 3)])
    b.append("<h2>上限の計算（市場を使わない。市場が全く値段に入れていないときの上限）</h2>")
    b.append(f"<p>1 レースの内枠の 3 着内の数の信頼度 ρ = {nm(Bd['rho_race_inner'], 3)}（観測の分散 {nm(Bd['var_obs_inner'], 3)}、偏りが無いときの超幾何の分散 {nm(Bd['var_null_inner'], 3)}）。"
             f"日の数 {Bd['inner']['days']:,}、レース {Bd['inner']['races']:,}。倍率の最大 {nm(Bd['max_mult'], 3)}。</p>")
    b.append(f"<p>日の中の相関（奇数・偶数のレース）: 内外 {nm(Bd['inner']['split_half_corr'], 3)}・前後 {nm(Bd['front']['split_half_corr'], 3)}。"
             f"直前 2 レースの和と次のレースの相関: 内外 {nm(Bd['inner']['corr_prev2_next'], 3)}・前後 {nm(Bd['front']['corr_prev2_next'], 3)}。</p>")
    b.append(table(["偏り", "直前 2 レースの偏りの和", "次のレース", "3 着内の倍率", "1 着の倍率"], rows, {2, 3, 4}))
    b.append("<h2>判定と見た回数</h2>" + bullets(texts["judge"]))
    b.append("<h2>作り方・再現</h2>" + bullets(["<code>python -m src.research.rules_council.daybias bound</code>（結果 <code>docs/operations/retrain_trials/rules_council_rc3_bound.json</code>）。費用 0 円。",
                                                "決定は <code>docs/operations/roi_plus_council/decisions.md</code> の「RC-3 の判断」。"]))
    return page("当日の馬場の偏りのルール探し（RC-3）", b)


def build_rc4(Bd: dict, texts: dict) -> str:
    b = ["<h2>結論</h2>", '<div class="box">' + bullets(texts["conclusion"]) + "</div>"]
    rows = []
    for col, r in Bd["cols"].items():
        for band in ("一桁", "穴馬"):
            v = r.get(band)
            if not v:
                continue
            for t in ("top20", "top10"):
                rows.append([esc(col), band, {"top20": "上位 20%", "top10": "上位 10%"}[t], f"{v[t + '_runs']:,}", f"{v[t + '_wins']:.0f}", nm(v[t + "_lift"]), nm(v[t + "_Q"]),
                             nm(v[t + "_lift_over_Q"]), nm(v[t + "_se"])])
    b.append(f"<h2>上限と市場の値付け（探す側 70%・再現 ≤ 50 倍の初出走 {Bd['runs']:,} 走のうち 2018〜24 の列の分かる走）</h2>")
    b.append("<p>上限 ＝ 列の上位の勝率 ÷ 同じ帯の初出走の勝率（市場を使わない）。Q ＝ 上位の群の確定の q の平均 ÷ 帯の q の平均（オッズだけ）。"
             "上限 ÷ Q は探す側の市場比の粗い値。要る倍率は 一桁 約 1.27・穴馬 約 1.3。</p>")
    b.append(table(["列", "帯", "上位（前の年の分布の分位）", "走", "勝ち", "上限", "Q", "上限 ÷ Q", "SE"], rows, {3, 4, 5, 6, 7, 8}))
    cy = Bd["coverage_by_year"]
    b.append(fold("生産者・母が付く率（初出走の年・その後の出走の数）", table(["年", "生産者", "母"], [[y, pc(v["has_breeder"], 0), pc(v["has_dam"], 0)] for y, v in cy.items()], {1, 2})
                  + table(["その後の出走（2016〜19 年の初出走）", "生産者", "母"], [[k, pc(v["has_breeder"], 1), pc(v["has_dam"], 1)] for k, v in Bd["coverage_by_later_runs_2016_19"].items()], {1, 2})))
    b.append("<h2>判定と見た回数</h2>" + bullets(texts["judge"]))
    b.append("<h2>作り方・再現</h2>" + bullets(["<code>python -m src.research.rules_council.debut</code>（列）→ <code>debut bound</code>（結果 <code>docs/operations/retrain_trials/rules_council_rc4_bound.json</code>）。費用 0 円。",
                                                "決定は <code>docs/operations/roi_plus_council/decisions.md</code> の「RC-4 の判断」。"]))
    return page("初出走の馬の目録に無い列のルール探し（RC-4）", b)


def build_summary(texts: dict) -> str:
    b = ["<h2>結論</h2>", '<div class="box">' + bullets(texts["conclusion"]) + "</div>"]
    from src.scripts.docs.build_rules_council_table import results_html
    b.append(results_html())
    b.append("<h2>議題の一覧</h2>" + table(["議題", "日付", "中身", "結果", "頁"],
                                         [[esc(r["id"]), esc(r["date"]), esc(r["what"]), esc(r["result"]),
                                           f'<a href="{esc(r["page"])}">頁</a>' if r.get("page") else "－"] for r in texts["agendas"]]))
    b.append("<h2>勝ち筋の台帳（市場比が 1 を超えたか）</h2>" + table(["源", "状態", "大きさ", "注"], [[esc(x) for x in r] for r in texts["ledger"]]))
    b.append("<h2>通算の数え</h2>" + table(["項目", "値"], [[esc(k), esc(v)] for k, v in texts["counts"]]))
    if texts.get("retro"):
        b.append("<h2>中間の振り返り（RC-0〜RC-5）</h2>" + bullets(texts["retro"]))
    b.append("<h2>次に見ること</h2>" + bullets(texts["next"]))
    return page("ルールの会議のまとめ", b)


def main() -> None:
    texts = json.loads(PAGES.read_text(encoding="utf-8"))
    DST.mkdir(parents=True, exist_ok=True)
    out = {"rules_council_t60_band.html": build_rc1(json.loads((TRI / "rules_council_t60.json").read_text(encoding="utf-8")), texts["rc1"]),
           "rules_council_favorites.html": build_rc2(json.loads((TRI / "rules_council_rc2.json").read_text(encoding="utf-8")),
                                                     json.loads(PREREG2.read_text(encoding="utf-8")), texts["rc2"]),
           "rules_council_daybias.html": build_rc3(json.loads((TRI / "rules_council_rc3_bound.json").read_text(encoding="utf-8")), texts["rc3"]),
           "rules_council_debut.html": build_rc4(json.loads((TRI / "rules_council_rc4_bound.json").read_text(encoding="utf-8")), texts["rc4"]),
           "rules_council_summary.html": build_summary(texts["summary"])}
    for k, v in out.items():
        (DST / k).write_text(v, encoding="utf-8")
        print("wrote", DST / k)


if __name__ == "__main__":
    main()
