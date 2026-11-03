"""めぐコラム『騎手の得意クラス・競馬場・時期』を実データから集計し、frontend/content/columns/jockey-class-venue-season.json を書く。

問い: クラス（新馬・未勝利・1〜3 勝・OP・重賞）で騎手の成績の傾向は変わるか。騎手ごとに得意な競馬場・時期（季節）はあるか。

物差し: 複勝率の「実際 − 騎乗馬の人気（単勝オッズ）から見込まれる値」（jockey_common.calibrate）。
  生の複勝率は騎乗馬の強さで決まる部分が大きく、条件ごとに乗る馬の強さも違うため。
得意・苦手: jockey-long-distance と同じ 2 軸（jockey_common.two_axis）。
  (a) 騎手間: その条件で、騎手の複勝率が全騎手の平均より高い / 低い。
  (b) 騎手内: その条件の「実際 − 見込み」が、同じ騎手のほかの条件より高い / 低い。
  両方なら「得意」「苦手」。検定は Benjamini–Hochberg（q<0.10）で多重比較を補正する。
対象: 2020〜2025 年に MIN_RIDES 騎乗以上の騎手（主要騎手）。

使い方:
    python -m src.research.columns.jockey_class_venue_season            # 集計して記事を書く
    python -m src.research.columns.jockey_class_venue_season --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binom

from src.research.columns import jockey_common as jc
from src.research.columns.jockey_change import loyo_ranks

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/jockey-class-venue-season.json"

MIN_RIDES = 1000
RANK_MIN_RIDES = 300
CHART_MIN_N = 50       # これ未満の点は点線
DIMS = {
    "cls": ("クラス", jc.CLASSES),
    "venue": ("競馬場", jc.VENUES),
    "season": ("季節", jc.SEASONS),
}


def flat(s: str) -> str:
    return s.split("\n")[0]


def judge_text(t: pd.DataFrame, jid: str) -> str:
    x = t[(t["jockey_id"] == jid) & ~t["short"].isin(["–", "標本不足"])]
    if x.empty:
        return "判定: はっきりした得意・苦手なし"
    return "判定: " + "、".join(f"{flat(r['level'])}＝{r['short']}（{jc.pt(r['diff'])}）" for _, r in x.iterrows())


def jockey_chart(d: pd.DataFrame, t: pd.DataFrame, col: str, ids: list[str], names: dict[str, str]) -> dict[str, Any]:
    label, levels = DIMS[col]
    variants: dict[str, Any] = {}
    first: list[dict[str, Any]] | None = None
    first_cap = ""
    d = jc.level_adjusted(d, col, "top3")
    for j in ids:
        dj = d[d["jockey_id"] == j]
        cells = [jc.cell(dj[dj[col] == lv], "top3") for lv in levels]
        ser = [{"name": "実際 − 人気の見込み（複勝率）", **jc.chart_values(cells, "diff")}]
        cap = f"{names[j]}。{judge_text(t, j)}"
        if first is None:
            first, first_cap = ser, cap
        else:
            variants[names[j]] = {"series": ser, "caption": cap}
    return {
        "type": "chart", "title": f"騎手別 {label}ごとの 複勝率の 実際 − 人気の見込み", "xLabel": label, "yLabel": "実際 − 見込み（pt）",
        "xs": levels, "series": first, "format": "pt", "minN": CHART_MIN_N,
        "filters": [{"label": "騎手", "options": [names[j] for j in ids]}], "variants": variants,
        "caption": first_cap,
    }


def rank_class_chart(d: pd.DataFrame, rank: pd.Series) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """rank: 出走ごとの今走の騎手のランク（その年を除いた年から決めたもの）。"""
    d = jc.level_adjusted(d, "cls", "top3").assign(rank=rank)
    cells = {k: [jc.cell(d[(d["rank"] == k) & (d["cls"] == c)], "top3") for c in jc.CLASSES] for k in jc.RANKS}
    diff = {"type": "chart", "title": "騎手ランク別 クラスごとの 複勝率の 実際 − 人気の見込み", "xLabel": "クラス", "yLabel": "実際 − 見込み（pt）",
            "xs": jc.CLASSES, "format": "pt", "minN": CHART_MIN_N,
            "series": [{"name": f"{k} ランク", **jc.chart_values(cells[k], "diff")} for k in jc.RANKS],
            "caption": "ランクは jockey-change と同じ（300 騎乗以上の騎手を、縮めた複勝率で S 上位 5%・A 〜25%・B 〜75%・C 〜95%・D 残り。"
                       "結果を見る年はその年を除いた年の騎乗からランクを決める＝1 年抜きの交差）。縦線は 95% 信頼区間。ボタンでランクを切り替え・重ねられる。"}
    share = {"type": "chart", "title": "騎手ランク別 クラスごとの 複勝率", "xLabel": "クラス", "yLabel": "複勝率", "xs": jc.CLASSES,
             "series": [{"name": f"{k} ランク", **jc.chart_values(cells[k], "rate")} for k in jc.RANKS],
             "caption": "生の複勝率（騎乗馬の強さを含む）。"}
    return diff, share, cells


def summary_rows(tables: dict[str, pd.DataFrame], ids: list[str], names: dict[str, str]) -> list[list[str]]:
    rows = []
    for j in jc.kana_order(ids, names):
        row = [names[j]]
        for col in DIMS:
            t = tables[col]
            x = t[(t["jockey_id"] == j) & ~t["short"].isin(["–", "標本不足"])]
            row.append("、".join(f"{flat(r['level'])} {r['short']}" for _, r in x.iterrows()) or "–")
        rows.append(row)
    return rows


def analyze(d: pd.DataFrame) -> dict[str, Any]:
    names = jc.jockey_names(d)
    cnt = d.groupby("jockey_id").size()
    ids = jc.kana_order(cnt[cnt >= MIN_RIDES].index.tolist(), names)
    tables = {col: jc.two_axis(d, ids, col, levels, "top3") for col, (_, levels) in DIMS.items()}
    R = jc.ranks(d, RANK_MIN_RIDES)
    # 結果を見る出走のランクは、その年を除いた年から決める（ランクの元の成績と結果が同じ出走で重ならないように）
    rk = loyo_ranks(d)["rank_now"]
    diff_chart, rate_chart, rc = rank_class_chart(d, rk)
    cls_all = {c: jc.cell(d[d["cls"] == c], "top3") for c in jc.CLASSES}
    rank_all = {k: jc.cell(d[rk == k], "top3") for k in jc.RANKS}
    return {"rank_all": rank_all, "names": names, "ids": ids, "tables": tables, "R": R, "diff_chart": diff_chart, "rate_chart": rate_chart, "rc": rc,
            "cls_all": cls_all, "n_runs": len(d), "d": d}


def counts(t: pd.DataFrame) -> dict[str, int]:
    x = t[t["short"] != "標本不足"]
    return {"tests": len(x), "p05": int((x["pb"] < 0.05).sum()), **x["short"].value_counts().to_dict()}


def flagged(t: pd.DataFrame, names: dict[str, str]) -> list[list[Any]]:
    x = t[~t["short"].isin(["–", "標本不足"])].copy()
    x = x.sort_values(["short", "qb"])
    return [[names[r["jockey_id"]], flat(r["level"]), int(r["n"]), jc.pct(r["rate"]), jc.pct(r["avg"]), f"{jc.pt(r['diff'])}（{jc.pt_ci(r['diff_ci'])}）",
             jc.pt(r["other_diff"]), f"{r['qb']:.3f}", r["label"]] for _, r in x.iterrows()]


def article(a: dict[str, Any]) -> dict[str, Any]:
    names, ids, tables, rc = a["names"], a["ids"], a["tables"], a["rc"]
    cs = {col: counts(t) for col, t in tables.items()}
    i = {c: k for k, c in enumerate(jc.CLASSES)}
    sig = [(k, c, x) for k in jc.RANKS for c, x in zip(jc.CLASSES, rc[k])
           if x["diff_ci"] and x["n"] >= CHART_MIN_N and (x["diff_ci"][0] > 0 or x["diff_ci"][1] < 0)]
    p_tail = float(binom.sf(len(sig) - 1, 35, 0.05))        # 差が無くても、35 点のうち len(sig) 点以上が 0 から外れる確率
    n_low = sum(1 for k, c, x in sig if k in ("C", "D") and c in ("未勝利", "1勝") and x["diff"] < 0)
    sig_txt = "、".join(f"{k} ランクの{c}（{jc.pt(x['diff'])}、{jc.pt_ci(x['diff_ci'])}、{x['n']:,}走）" for k, c, x in sig) or "ありません"
    s_all = a["rank_all"]["S"]
    flag_rows = flagged(tables["cls"], names) + flagged(tables["venue"], names) + flagged(tables["season"], names)
    n_flag = len(flag_rows)

    def ctext(col: str) -> str:
        c = cs[col]
        return (f"{DIMS[col][0]}: {c['tests']} 組のうち 得意 {c.get('得意', 0)}・苦手 {c.get('苦手', 0)}・相対的に上 {c.get('相対的に上', 0)}・相対的に下 {c.get('相対的に下', 0)}"
                f"（補正前 p<0.05 は {c['p05']} 組。偶然でも約 {round(c['tests'] * 0.05)} 組は出る）")

    def rk_line(k: str) -> str:
        return "・".join(f"{c} {jc.pt(x['diff'])}" for c, x in zip(jc.CLASSES, rc[k]))

    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "対象の騎手", "value": f"{len(ids)}人", "sub": f"2020〜2025年に {MIN_RIDES:,} 騎乗以上"},
            {"label": "S ランクの騎手 複勝率", "value": jc.pct(s_all["rate"]), "sub": f"人気の見込みとの差 {jc.pt(s_all['diff'])}（{s_all['n']:,}走）"},
            {"label": "補正後も残った 得意・苦手", "value": f"{n_flag}組", "sub": f"クラス・競馬場・季節 の計 {sum(c['tests'] for c in cs.values())} 組中"},
        ]},
        {"type": "heading", "text": "物差し: 人気から見込まれる複勝率との差"},
        {"type": "text", "text":
            "騎手の成績を条件別に比べるとき、生の複勝率では「その条件でどんな馬に乗ったか」が混ざります。"
            "トップ騎手は重賞で人気馬に乗り、若手はローカルの下級条件で人気薄に乗る、という違いがそのまま成績の差に見えてしまいます。"
            "そこでこの記事では、騎乗馬の単勝オッズから見込まれる複勝率を出し、実際との差（実際 − 見込み）で比べます。"
            "0 なら人気どおり、プラスなら人気以上に馬券になっています。\n\n"
            "得意・苦手は jockey-long-distance と同じ 2 軸で判定します。"
            "(a) その条件で全騎手の平均より複勝率が高い（低い）か、(b) その条件の「実際 − 見込み」が同じ騎手のほかの条件より高い（低い）か。"
            "両方そろえば「得意」「苦手」、(b) だけなら「相対的に上 / 下」と書きます。騎手 × 条件の数だけ検定を繰り返すので、"
            f"Benjamini–Hochberg 補正の q<{jc.Q_SIG} を差ありとしました。"},
        {"type": "heading", "text": "クラス別の傾向"},
        a["rate_chart"],
        a["diff_chart"],
        {"type": "text", "text":
            f"生の複勝率では、どのクラスでも S ランクが圧倒的に高く、D ランクが低い並びは変わりません。"
            f"人気からの見込みとの差で見ると、ランクの差はほぼ消えます（S ランク: {rk_line('S')}。B ランク: {rk_line('B')}）。"
            f"ランク × クラスの 35 点のうち、95% 区間が 0 をまたがないのは {sig_txt}。"

            + ("35 点を同時に見れば 1〜2 点は偶然でも外れるので、クラスによって騎手の上積みが変わる、という傾向は見えません。"
               if len(sig) <= 2 else
               f"35 点を同時に見れば 1〜2 点は偶然でも外れ、{len(sig)} 点以上が偶然で出る確率も約 {p_tail * 100:.0f}% あるので、"
               "クラスによって騎手の上積みが変わる、とまでは言えません。"
               + ("C・D ランクの未勝利・1 勝クラスのマイナスは向きがそろっていますが、差は小さく、候補にとどまります。" if n_low >= 2 else "")
               if p_tail >= 0.05 else
               f"35 点を同時に見れば 1〜2 点は偶然でも外れますが、{len(sig)} 点はそれより多く、"
               + (f"そのうち {n_low} 点は下位ランク（C・D）の未勝利・1 勝クラスのマイナスです。下位ランクの騎手は下級条件で人気ほど走らない、という形がうかがえます。"
                  if n_low >= 2 else "ただ、ランクやクラスにそろった形は見えません。")
               + f"ただし差は 1 点あたり {min(abs(x['diff']) for _, _, x in sig) * 100:.1f}〜{max(abs(x['diff']) for _, _, x in sig) * 100:.1f}pt と小さく、"
               "区間の端は 0 に近いものが多い点に注意してください。")
            + "上位騎手の成績が良いのは、どのクラスでも人気馬に乗っているためで、その大半はオッズに織り込まれています。"},
        {"type": "heading", "text": "騎手ごとの 得意クラス・競馬場・季節"},
        {"type": "text", "text":
            "下の 3 つのグラフは、プルダウンで騎手を選ぶと、その騎手の クラス・競馬場・季節 ごとの「実際 − 見込み」を出します（騎手は 50 音順）。"
            f"縦線は 95% 信頼区間、点線・白抜きの点は {CHART_MIN_N} 走未満（参考）。グラフの下に、その騎手の 2 軸の判定を出しています。"},
        jockey_chart(a["d"], tables["cls"], "cls", ids, names),
        jockey_chart(a["d"], tables["venue"], "venue", ids, names),
        jockey_chart(a["d"], tables["season"], "season", ids, names),
        {"type": "heading", "text": "判定の結果"},
        {"type": "text", "text":
            "\n".join(f"• {ctext(c)}" for c in DIMS) + "\n\n"
            + (f"多重比較を補正しても残った「得意・苦手・相対的に上/下」は {n_flag} 組です（下の表）。" if n_flag else
               "多重比較を補正すると、「得意」「苦手」と言える組は残りませんでした。")
            + " 補正前の p<0.05 の組の数は、どの切り口でも偶然で出る数とほぼ同じかやや多い程度で、"
            "「この騎手はこの競馬場・この季節が得意」という話の多くは、人気を考えると偶然のぶれの範囲に収まります。"
            "グラフで大きく外れて見える点も、95% 区間の広さと合わせて見てください。"},
        *([{"type": "table", "columns": ["騎手", "条件", "騎乗数", "複勝率", "全騎手の平均", "実際 − 見込み（95%区間）", "ほかの条件の 実際 − 見込み", "q 値", "判定"],
            "rows": flag_rows, "caption": "多重比較を補正しても (b) 騎手内の差が残った組。"}] if n_flag else []),
        {"type": "accordion", "title": f"全 {len(ids)} 人の判定一覧（50 音順）", "blocks": [
            {"type": "table", "columns": ["騎手", "クラス", "競馬場", "季節"], "rows": summary_rows(tables, ids, names),
             "caption": "– は、はっきりした得意・苦手なし（騎乗 50 未満の条件は判定しない）。"}]},
        {"type": "note", "text":
            "人気からの見込みは、単勝オッズから求めた勝率を 50 段階に分け、頭数の帯ごとに実際の複勝率をあてはめたもの。"
            "さらに条件（クラス・競馬場・季節）ごとに、全騎手で平均した 実際 − 見込み が 0 になるようにずらしている"
            "（例: 未勝利戦は全体に人気より +0.8pt ほど走るので、その分を条件の性質として除き、騎手の差だけを見る）。"
            "オッズには騎手の評価も含まれるため、「実際 − 見込み」は市場が織り込んだ以上の上積みで、騎手の腕そのものより小さく出ます。"
            "クラスは レース結果の格付け（新馬・未勝利・1〜3 勝・OP と L・G1〜G3 を重賞）、季節は 春 3〜5 月・夏 6〜8 月・秋 9〜11 月・冬 12〜2 月。"},
    ]
    return {
        "slug": "jockey-class-venue-season",
        "title": "騎手の得意クラス・競馬場・季節——人気を考えると「得意」はどれだけ残るか",
        "claim": "騎手には得意なクラス・競馬場・時期がある",
        "verdict": "partial" if n_flag else "not_supported",
        "summary": (f"主要騎手 {len(ids)} 人の成績を、クラス・競馬場・季節ごとに、騎乗馬の人気から見込まれる複勝率との差で比べました。"
                    f"生の成績の差の多くは乗る馬の強さで説明でき、多重比較を補正して残った得意・苦手は {n_flag} 組でした。"),
        "categories": ["jockey/class-venue"],
        "published": "2026-11-02",
        "dataset": {"period": "2020〜2025年", "conditions": f"JRA 平地（障害を除く）。取消・除外を除く。{MIN_RIDES:,} 騎乗以上の騎手",
                    "sample": f"{a['n_runs']:,}出走"},
        "body": body,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    a = analyze(jc.frame())
    col = article(a)
    text = json.dumps(col, ensure_ascii=False, indent=2) + "\n"
    json.loads(text, parse_constant=lambda t: (_ for _ in ()).throw(ValueError(t)))
    if args.dry_run:
        print(json.dumps([b for b in col["body"] if b["type"] in ("stats", "text", "table")], ensure_ascii=False, indent=0)[:8000])
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT} ({len(text) // 1024} KiB)")


if __name__ == "__main__":
    main()
