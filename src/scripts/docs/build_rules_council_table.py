"""ルールの会議の「やり方ごとの成績表」を、結果の JSON から作る（数字を手で写さない。R0-07 の決定）。

出力:
  docs/operations/roi_plus_council/easy_summary.md の「## 結果」の節（<!-- results:begin --> 〜 <!-- results:end --> の間を差し替える）
  docs/operations/roi_plus_council/easy/RC-*.md（1 議題 1 枚の子ページ）
  rules_council_summary.html の表（build_rules_council.build_summary が results_html() を呼ぶ）
入力（どれも手元のファイル。費用 0 円）:
  docs/operations/retrain_trials/rules_council_results_table.json（R0-07 の 1 回の集計。探す側 70%。RC-6 の行は 2-1 の写し）
  docs/operations/megu_cycles/confirm_log.json・final_once.json（めぐの 2017〜21・2022〜24・2025。写し）
使い方: python -m src.scripts.docs.build_rules_council_table
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRI = ROOT / "docs/operations/retrain_trials"
MEGU = ROOT / "docs/operations/megu_cycles"
COUNCIL = ROOT / "docs/operations/roi_plus_council"
EASY = COUNCIL / "easy"
SUMMARY_MD = COUNCIL / "easy_summary.md"
BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"

HEAD_NOTE = ("この表の数字は合否ではない。探した期間の値で、選んだ後・割引き前の値。100% 超は未来でも続く根拠にならない。"
             "年・帯・分解は記述で、選び直しと合否に使わない。")
WHY_NOTE = "説明の確かめを通っても、回収率 100% 超の根拠にはならない（基本理念。protocol.md の 0-0）。"
WHY_EASY = "説明が当たっても、もうかる根拠にはならない。"
CHECK_LABELS = ("通った", "合わない", "確かめられない", "確かめ中", "後付け", "新しい予測が足りない", "ルールなし")
MEGU_WHY = ("なぜ効くのか: 5 つとも RC-6 と同じ説明（AI が、市場が読み切れない情報を持つ）。説明の確かめは、c13・c21・c23 が「確かめ中」。"
            "c18（5 分前）は「確かめられない（作りなおしたオッズの作りに依存。確定しない）」。c24 は会議で札を決めていない（2025 年は不合格で、前向きの記録に入れていない）。")
HEAD_EASY = "かんたんに言うと: ここの数字は、ルールを探すのに使ったレースの成績なので、よく見えやすい。この数字だけで「もうかる」とは言えない。"
MEGU_NOTE = ("この表は、探すのに使っていないレース（2022〜24 の確かめと、2025 の最後の 1 回）の値。点数は作りなおしたオッズ 5 通りの平均で、"
             "上の表（1 通りの値）とは数え方が違う。「同じ人気の基準」は同じ人気の順位の馬を全部買った値で、上の表の基準とは別。")

# 行の定義（文は decisions.md の各節と results_table.py の docstring をかみくだいた写し。数字は JSON から入れる）
ROWS = [
    {"id": "RC-2", "why": "お客さんは前のレースの着順を重く見すぎる。負けた馬は、負けた理由を考えずに嫌う人が増え、少し人気が下がる。ただし効きは 1.05〜1.10 倍で、もうかる線（約 1.27 倍）に届かない（説明は結果を見た後に書いた）。", "check": "後付け（確定しない。ルールは不合格）", "page": "RC-02.md", "key": "RC-2 和集合", "period": "2016〜24", "name": "人気の馬 × 前のレースの負け方など 12 の型",
     "easy": "10 倍より下の馬で、前のレースで負けた馬などをまとめて買う",
     "horse": "発走 60 分前の単勝オッズ（作りなおした値）が 10 倍より下、かつ 12 の型（前のレースで 6〜9 着・4〜5 着、上がりが 4 位より下で 4 着より下 など）のどれかに当たる。1 頭が何個当たっても 1 点",
     "race": "芝・ダートの全レース（新馬〜G1。障害は入れない）。当てはまる馬が 0 頭なら見送り。何頭いても全部買う",
     "judge": "見つからず（不合格）。型の効きは 1.05〜1.10 倍で、もうかるのに要る約 1.27 倍に届かない", "live": "作れる（前のレースの成績だけ）"},
    {"id": "RC-3", "why": "－（ルールが決まる前にやめた）", "check": "ルールなし", "page": "RC-03.md", "key": "RC-3", "period": "－", "name": "その日の馬場のくせ",
     "easy": "その日、内の枠や前に行った馬が勝っているかを見て買う",
     "horse": "数えない（やめた時点で、買い方が決まっていなかった）", "race": "－",
     "judge": "やめた（いちばん良く見ても 1.18 倍で、約 1.27 倍に届かない）", "live": "前のレースの結果が発走 75 分前に分かるかが要る"},
    {"id": "RC-4", "why": "－（やめた議題。確定するルールが無い）", "check": "ルールなし", "page": "RC-04.md", "key": "RC-4 生産者", "period": "2020〜24", "name": "はじめて走る馬 × 育てた牧場の成績",
     "easy": "はじめて走る馬を、育てた牧場のこれまでの成績で選ぶ",
     "horse": "はじめて走る馬で、発走 60 分前の単勝オッズ（作りなおした値）が 10 倍より下、かつ育てた牧場のその日までの初出走が 10 頭以上で、その勝ちの割合（数が少ないと控えめに直す）が前の年までの上位 2 割に入る",
     "race": "新馬・未勝利で、はじめて走る馬がいるレース。0 頭なら見送り",
     "judge": "やめた（効きが約 1.27 倍に届かない・数が少ない）", "live": "作れる（牧場の表の更新が要る）"},
    {"id": "RC-5", "why": "－（やめた議題。確定するルールが無い）", "check": "ルールなし", "page": "RC-05.md", "key": "RC-5 H1", "period": "2016〜24", "name": "その日の雨で馬場が悪くなった × ぬかるみが得意",
     "easy": "その日に雨で馬場が悪くなったら、ぬかるみが得意な馬を買う",
     "horse": "発走 60 分前の単勝オッズ（作りなおした値）が 10 倍より下、かつ前の日までに、やや重い馬場より悪い日に 3 着以内が 1 回以上",
     "race": "発走 75 分前までに終わった直前のレースの馬場が、その日の最初のレースより悪い芝・ダートのレース。0 頭なら見送り",
     "judge": "やめた（数が年 112 頭ほどで少ない）", "live": "作れる（馬場の発表の時刻が要る）"},
    {"id": "(b)", "why": "－（「前のレースで好走した馬は買われ過ぎる」という RC-2 の逆の考え。やめた議題）", "check": "ルールなし", "page": "RC-b.md", "key": "(b) 残した側（均一・種 0）", "period": "2017〜24", "name": "AI の買い目から、人気なのに負けやすい型を消す",
     "easy": "AI の買い目から、人気なのに負けやすい型の馬をのぞく",
     "horse": "AI の買い目（下の RC-6）から、発走 60 分前の単勝オッズ（作りなおした値）が 10 倍より下で「人気なのに負けやすい 7 の型」に当たる馬をのぞく（残した側）",
     "race": "RC-6 と同じ（平場・重賞以外）",
     "judge": "やめた（消す馬が、10 倍より下の買い目の 6 割と重なり、消しの効きだけを分けられない）", "live": "作れる"},
    {"id": "RC-6", "why": "AI は血統・調教・ラップなど、お客さんが読み切れない情報を勝つ見込みに入れている。市場はそれを値段に入れ切れない（説明は結果の前に書いた）。", "check": "確かめ中（本物のオッズで記録している c13 の買い目が 300 点に届いた時に確かめる）", "page": "RC-06.md", "key": "RC-6 c13（2-1 の写し）", "period": "2017〜24", "name": "AI × オッズ（c13）",
     "easy": "AI の勝つ見込み × オッズが 1.3 以上の馬を買う",
     "horse": "AI の勝つ見込みと、オッズから見た見込みを、昔のデータで決めた重みで組み合わせる。その見込み × 発走 60 分前の単勝オッズ（作りなおした値）が 1.3 以上、かつオッズが 50 倍以下",
     "race": "平場（重賞以外の新馬〜オープン）。0 頭なら見送り。何頭いても全部買う",
     "judge": "前向きの記録中（2026-11-28 から。1 年で判定）", "live": "作れる（本番の AI の計算の通り道は別に確かめる）"},
]

MEGU_CANDS = [("c13_flat", "c13 平場・60 分前"), ("c18_flat_t05", "c18 平場・5 分前"), ("c21_maiden_t60", "c21 新馬未勝利・60 分前"),
              ("c23_flat_t15", "c23 平場・15 分前"), ("c24_flat_v7", "c24 平場・60 分前（AI の版 7）")]


def pc(x, d: int = 1) -> str:
    return "－" if x is None or isinstance(x, str) else f"{x * 100:.{d}f}%"


def ci(v) -> str:
    return "－" if not v else f"{v[0] * 100:.0f}〜{v[1] * 100:.0f}%"


def load() -> dict:
    T = json.loads((TRI / "rules_council_results_table.json").read_text(encoding="utf-8"))
    conf = {c["spec"]["name"]: c for c in json.loads((MEGU / "confirm_log.json").read_text(encoding="utf-8"))}
    fin = json.loads((MEGU / "final_once.json").read_text(encoding="utf-8"))["results"]
    return {"T": T, "conf": conf, "fin": fin}


def seeds_text(m: dict) -> str:
    v = [x for x in (m.get("roi_5seeds") or []) if x is not None]
    if not v:
        return "－"
    return f"{sum(v) / len(v) * 100:.1f}%（{min(v) * 100:.0f}〜{max(v) * 100:.0f}%）"


def search_row(r: dict, m: dict | None) -> dict:
    """探した期間（探す側 70%）の数字。m は results_table.json の行。"""
    m = m or {}
    if r["id"] == "RC-6":                                     # 2-1 の写し（数え直していない。stake の重み・90% の幅）
        return {"races": "記録なし", "bets": f"{m['bets']:,.0f}（買う割合の重み）", "hit": "未集計",
                "roi": f"{pc(m['roi'])}（90% の幅 {m['roi_ci90'][0] * 100:.1f}〜{m['roi_ci90'][1] * 100:.1f}%）", "seeds": "－（この値が 5 通りの平均）",
                "base": f"{pc(m['base_ii_final_band_only'])}（確定オッズの帯）", "nopay": pc(m["base_ii_final_band_only"] * m["excess_mr"]), "note": m.get("note", "")}
    races = "－"
    if "races_pop" in m:
        races = (f"{m['races_pop']:,} 中 {m['races_bought']:,} 買い・{m['races_skipped_zero']:,} 見送り"
                 + (f"・{m['races_unjudgeable']:,} 判定できない" if m.get("races_unjudgeable") else ""))
    if not m.get("bets"):
        return {"races": races, "bets": "0" if "bets" in m else "－", "hit": "－", "roi": "－", "base": "－", "seeds": "－", "nopay": "－", "note": m.get("note", "")}
    return {"races": races, "bets": f"{m['bets']:,.0f}", "hit": f"{pc(m['hit_rate_bets'])}（レース {pc(m['hit_rate_races'])}）",
            "roi": f"{pc(m['roi'])}（95% の幅 {ci(m.get('roi_ci95'))}）", "seeds": seeds_text(m), "base": pc(m.get("base_judge_band")),
            "nopay": pc(m["decomp_band_x_class"]["roi_without_payout_term"]), "note": m.get("note", "")}


def results_rows(D: dict) -> list[dict]:
    rows = D["T"]["rows"]
    return [r | search_row(r, rows.get(r["key"])) for r in ROWS]


def megu_rows(D: dict) -> list[list[str]]:
    out = []
    for k, nm in MEGU_CANDS:
        c, f = D["conf"][k], D["fin"][k]
        s, cf = c["select"], c["confirm"]
        out.append([nm, f"{s['bets_per_year']:.0f}", pc(s["hit_rate"]), f"{pc(s['roi'])}（{ci(s.get('roi_ci'))}）",
                    f"{cf['bets_per_year']:.0f}", pc(cf["hit_rate"]), f"{pc(cf['roi'])}（{ci(cf.get('roi_ci'))}）", pc(cf["base_same_pop"]),
                    f"{f['bets']:.0f}", pc(f["hit_rate"]), f"{pc(f['roi'])}（{ci(f.get('roi_ci'))}）", "合格" if f.get("pass") else "不合格"])
    return out


def b_notes(D: dict) -> list[str]:
    """(b) の残した側が高く見える理由を、分解の数字で書く（1 号の指示）。"""
    rows = D["T"]["rows"]
    a, k, x = rows["(b) c13 全部（均一・種 0）"], rows["(b) 残した側（均一・種 0）"], rows["(b) 消した側（均一・種 0）"]
    ks, xs = rows["(b) 残した側（stake）"], rows["(b) 消した側（stake）"]
    da, dk = a["decomp_band_x_class"], k["decomp_band_x_class"]
    return [f"AI の買い目を全部買うと、1 点 100 円・1 通り（種 1000）で {pc(a['roi'])}（{pc(a['hit_rate_bets'])} の的中）、5 通りの平均で {seeds_text(a)}。1 通り目は 5 通りの中で上の方。",
            f"残した側の勝つ割合の上乗せ（市場比 ÷ 帯の比）は {da['excess_mr']:.3f} → {dk['excess_mr']:.3f} で、消しても増えていない。",
            f"高く見える分は、勝った馬の払戻のぶれ（{da['payout_term']:.3f} → {dk['payout_term']:.3f}）だけから来ている。ぶれを除くと、残した側 {pc(dk['roi_without_payout_term'])}・全部 {pc(da['roi_without_payout_term'])}。",
            f"消した側は、1 点 100 円・1 通りで {pc(x['roi'])}（5 通りの平均 {seeds_text(x)}）、買う割合の重みで {pc(xs['roi'])}（残した側は {pc(ks['roi'])}）。数え方で 100% のどちら側かが入れかわる。"]


def md_table(head: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c).replace("|", "／") for c in r) + " |" for r in rows])


def easy_section(D: dict) -> str:
    R = results_rows(D)
    a = md_table(["やり方", "ひとことで", "買う馬の選び方", "買うレースの選び方", "期間", "対象・買った・見送ったレース", "点数", "的中率（1 点あたり・1 レースあたり）",
                  "100 円あたりの戻り（1 通り）", "5 通りの平均（いちばん低い〜高い）", "同じオッズの帯を全部買った戻り", "判定", "なぜ効くのか（言語化）", "説明の確かめ"],
                 [[f"[{r['id']} {r['name']}](easy/{r['page']})", r["easy"], r["horse"], r["race"], r["period"], r["races"], r["bets"], r["hit"], r["roi"], r["seeds"],
                   r["base"], r["judge"], r["why"], r["check"]] for r in R])
    b = md_table(["候補", "選ぶ期間 2017〜21: 年あたりの点数", "的中率", "戻り（95%）", "確かめ 2022〜24: 年あたりの点数", "的中率", "戻り（95%）", "同じ人気の基準",
                  "最後の 1 回 2025: 点数", "的中率", "戻り（95%）", "2025 の合格の印"], megu_rows(D))
    return "\n".join([BEGIN, "## 結果", "", f"**注意**: {HEAD_NOTE}", "", HEAD_EASY, "", WHY_EASY, "",
                      "### 探した期間の数字（開催の 7 割。単勝 1 点 100 円。判定は作りなおした発走 60 分前のオッズ）", "", a, "",
                      "- 期間は行ごとに違う（RC-4 は 2020〜24、(b) と RC-6 は 2017〜24）。行どうしをならべて比べない。",
                      "- 「1 通り」は、作りなおしたオッズの 1 通り目（種 1000）の値。「5 通りの平均」は 5 通りに替えた値の平均。幅は開催ごとにまとめてくり返し数えた 95% の幅。",
                      "- 「同じオッズの帯を全部買った戻り」は、同じオッズの帯（作りなおした値）とクラスの馬を全部買った値。",
                      "- RC-6 の行は、前に数えた値の写し（買う割合の重み・90% の幅）。1 点 100 円で数えた値は (b) の子ページにある。",
                      "- RC-4 は、線を引く昔の年（2016〜19）の成績が高く出ていて、2020〜24 の馬はどれも線に届かず 0 点だった。",
                      "- RC-1（作りなおしたオッズの確かめ）は買い方ではないので、表に入れていない。", "",
                      "### 確かめ・最後の 1 回の数字（めぐの AI × オッズ。単勝。出典 docs/operations/megu_cycles/confirm_log.json・final_once.json）", "",
                      MEGU_NOTE, "", b, "",
                      "- 2022〜24 で 5 つを同時に割り引くと、100% を超えると言えるのは c18 だけ（cycle_50.md）。",
                      f"- {MEGU_WHY}",
                      "- 2025 の合格の印は「2025 年の戻り ≥ 100% かつ 2022〜24 の 0.9 倍以上」（final_once.json の pass）。前向きの記録に進む条件で、100% 超の証明ではない。幅の下は 5 つとも 100% を割る。",
                      "- 前向き（2026-11-28 から、本物のオッズ）: c13・c21・c23 の系統と、RC-2 の和集合を **記録中**。", END])


LESSONS = {
    "RC-2": ["RC-2 の 12 の型で選んだ 10 倍より下の馬は、100 円が 80 円台にしかもどらない。", "型の効きは少しある（1.05 倍前後）が、競馬の取り分（約 2 割）をうめられない。",
             "これからのレースで、記録だけとる（11/28 から）。"],
    "RC-3": ["その日の馬場のくせは、次のレースまで少ししか続かない（いちばん良く見ても 1.18 倍）。", "買い方を決める前にやめたので、成績は数えていない。"],
    "RC-4": ["牧場の成績の表は、昔の年ほど「長く走った馬」だけが残っていて、成績が高く出る。",
             "その昔の年で線を引くと、2020 年からの馬はどれも線に届かず、買う馬が 0 頭になった。",
             "次に同じ種類の表を使うときは、線も 2020 年からの馬だけで引く。"],
    "RC-5": ["馬場が悪くなった日のレースは、9 年で探した期間に 231 しかない。", "数が少なすぎて、もうかるかどうかを決められない。"],
    "(b)": ["消しても、勝つ割合の上乗せは増えていない。残した側が高く見えるのは、勝った馬の払戻のぶれのせい（上の数字）。", "合否には使わない（損を減らす記録）。"],
    "RC-6": ["探した期間では基準を上回ったが、作りなおしたオッズの 5 通りの平均と、幅の下は 100% 前後。",
             "作りなおしたオッズの違いで、買う馬の 3〜6 割が入れかわる。", "これからのレース（本物のオッズ）だけで確かめる。1 年で「望みがあるか」を判定する。"],
}


def child_pages(D: dict) -> dict[str, str]:
    rows = D["T"]["rows"]
    pages = {}
    for r in results_rows(D):
        m = rows.get(r["key"]) or {}
        lines = [f"# {r['id']} {r['name']}", "", "[成績表にもどる](../easy_summary.md)", "", f"**注意**: {HEAD_NOTE}", "", HEAD_EASY, "",
                 "## 何を確かめたか", "", f"- {r['easy']}と、もうかるか。", "",
                 "## 買い方", "", f"- **買う馬**: {r['horse']}", f"- **買うレース**: {r['race']}", "- 単勝を 1 点 100 円。", "",
                 f"## 数字（探した期間: {r['period']} の開催の 7 割）", ""]
        if r["id"] == "RC-6":
            lines += [f"- 点数: {r['bets']}", f"- 的中率: {r['hit']}", f"- 100 円あたりの戻り: {r['roi']}", f"- 確定オッズの帯を全部買った戻り: {r['base']}",
                      f"- 勝った馬の払戻のぶれを除いた戻り: {r['nopay']}（運か、帯の中で高い方を選んだ分かは分けられない）",
                      "- 1 点 100 円・1 通りで数えた値は [(b) の子ページ](RC-b.md) にある。"]
        elif m.get("bets"):
            lines += [f"- レース: {r['races']}", f"- 点数: {r['bets']}（1 レースあたり 平均 {m['bets_per_race_mean']:.1f}・最大 {m['bets_per_race_max']:.0f}）",
                      f"- 的中率: {r['hit']}", f"- 100 円あたりの戻り（1 通り）: {r['roi']}。5 通りの平均 {r['seeds']}",
                      f"- 同じオッズの帯を全部買った戻り: {r['base']}",
                      f"- 勝った馬の払戻のぶれを除いた戻り: {r['nopay']}（同じ確定オッズの帯の中で、勝った馬の払戻が平均より高かった分を除いた値）",
                      f"- いちばん大きい負けの続き: {m['max_drawdown_units']:.0f} 点ぶん（1 点 100 円なら {m['max_drawdown_units'] * 100:,.0f} 円）。いちばん長い連敗 {m['longest_losing_races']} レース",
                      "- 年ごと: " + "・".join(f"{y} {v * 100:.0f}%" for y, v in m["roi_by_year"].items())]
        else:
            lines += ([f"- レース: {r['races']}", "- 買い目は 0 点。"] if r["bets"] == "0" else ["- 数えていない。"])
            if r["id"] == "RC-4":
                lines += ["- 上位 2 割という線は、16 通りためした中でいちばん良かったもの（選んだ後の値）。上位 1 割は 0.98 倍だった。",
                          "- 線を引く昔の年（2016〜19）は、長く走った馬だけが表に残っていて、成績が高く出る。"]
        if r["id"] == "(b)":
            lines += [f"- {x}" for x in b_notes(D)]
        lines += ["", "## 判定と理由", "", f"- {r['judge']}", f"- 本番で同じ買い方ができるか: {r['live']}", "",
                  "## なぜ効くのか（言語化）", "", f"- {r['why']}", f"- 説明の確かめ: {r['check']}", f"- {WHY_EASY}", "",
                  "## 分かったこと・次に生かすこと", ""]
        lines += [f"- {x}" for x in LESSONS.get(r["id"], [])]
        lines += ["", "詳しい根拠: [decisions.md](../decisions.md)・[まとめの頁](../../../html/operations/final/rules_council_summary.html)", ""]
        pages[r["page"]] = "\n".join(lines)
    return pages


def results_html() -> str:
    """rules_council_summary.html に入れる表（build_rules_council が呼ぶ）。"""
    D = load()
    e = html.escape
    R = results_rows(D)
    head = ["やり方", "買う馬の選び方", "買うレースの選び方", "期間", "対象・買った・見送ったレース", "点数", "的中率", "回収率（1 通り・幅）", "5 通りの平均（低〜高）",
            "払戻の項を除いた回収率", "基準線（判定の帯 × クラス群を全部買う）", "判定", "本番で作れるか", "なぜ効くのか（言語化）", "説明の確かめ"]
    body = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in
                                    [f"{r['id']} {r['name']}", r["horse"], r["race"], r["period"], r["races"], r["bets"], r["hit"], r["roi"], r["seeds"], r["nopay"],
                                     r["base"], r["judge"], r["live"], r["why"], r["check"]]) + "</tr>" for r in R)
    mh = ["候補", "2017〜21 点/年", "的中", "回収率", "2022〜24 点/年", "的中", "回収率", "同じ人気の基準", "2025 点", "的中", "回収率", "2025 の合格の印"]
    mb = "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in megu_rows(D))
    return ("<h2>やり方ごとの成績表</h2>" + f'<div class="box"><p>{e(HEAD_NOTE)}</p><p>{e(WHY_NOTE)}</p></div>'
            + "<h3>探した期間（開催の 70%・単勝 1 点 100 円・判定は再現 T-60）</h3>"
            + f'<div class="scroll"><table><thead><tr>{"".join(f"<th>{e(h)}</th>" for h in head)}</tr></thead><tbody>{body}</tbody></table></div>'
            + "<p>RC-6 は 2-1 の写し（stake の重み・90% の幅）。(b) の残した側が高く見える理由: " + e(" ".join(b_notes(D))) + "</p>"
            + "<h3>確かめ・最後の 1 回（めぐの候補。出典 docs/operations/megu_cycles/confirm_log.json・final_once.json）</h3>"
            + f"<p>{e(MEGU_NOTE)}</p><p>{e(MEGU_WHY)}</p>"
            + f'<div class="scroll"><table><thead><tr>{"".join(f"<th>{e(h)}</th>" for h in mh)}</tr></thead><tbody>{mb}</tbody></table></div>'
            + "<p>詳しい数字（分解・年ごと・5 本の幅）は docs/operations/retrain_trials/rules_council_results_table.json。子ページは docs/operations/roi_plus_council/easy/。</p>")


def main() -> None:
    D = load()
    EASY.mkdir(parents=True, exist_ok=True)
    for k, v in child_pages(D).items():
        (EASY / k).write_text(v, encoding="utf-8")
        print("wrote", EASY / k)
    s = SUMMARY_MD.read_text(encoding="utf-8")
    sec = easy_section(D)
    if BEGIN in s:
        a, rest = s.split(BEGIN, 1)
        s = a + sec + rest.split(END, 1)[1]
    else:
        i, j = s.index("## 結果"), s.index("## 結論")
        s = s[:i] + sec + "\n\n" + s[j:]
    SUMMARY_MD.write_text(s, encoding="utf-8")
    print("wrote", SUMMARY_MD)


if __name__ == "__main__":
    main()
