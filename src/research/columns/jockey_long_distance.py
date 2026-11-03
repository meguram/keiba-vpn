"""めぐコラム『川田は長距離が苦手？』を実データから集計し、frontend/content/columns/jockey-long-distance.json を書く。

以前の記事は数値が手書きで、グラフは 20 騎乗未満の距離帯を出していなかった。ここでは:
  - 騎手ごと・距離帯ごとの勝率に、騎乗数 n と「騎乗馬の人気から見込まれる勝率」を付ける。
    画面は n<20 の点（とそれにつながる線）を点線で描く（minN）。5 騎乗未満の点は出さない。
  - 「苦手」を 2 軸で判定する（jockey_common.two_axis）。
    (a) 騎手間: その距離帯で、騎手の勝率が全騎手の平均より低いか。
    (b) 騎手内: その距離帯の「実際 − 人気からの見込み」が、同じ騎手のほかの距離帯より低いか。
    両方なら「苦手」。片方だけのときは「平均より上だが、ほかの距離ほどではない」などと書き分ける。

使い方:
    python -m src.research.columns.jockey_long_distance            # 集計して記事を書く
    python -m src.research.columns.jockey_long_distance --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.research.columns import jockey_common as jc
from src.research.columns.compare_series import _p, _stat

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "frontend/content/columns/jockey-long-distance.json"

CHART_MIN_RIDES = 200      # グラフに出す騎手（期間の騎乗数）
TABLE_MIN_RIDES = 500      # 2 軸の判定表に出す騎手
MIN_N = 20                 # これ未満の点は点線
SHOW_N = 5                 # これ未満の点は出さない
FOCUS = {"川田将雅": "川田", "C.ルメール": "ルメール"}


def chart_block(d: pd.DataFrame, names: dict[str, str]) -> dict[str, Any]:
    cnt = d.groupby("jockey_id").size().sort_values(ascending=False)
    ids = cnt[cnt >= CHART_MIN_RIDES].index.tolist()
    base = d.groupby("dist9")["win"].mean()
    g = d[d["jockey_id"].isin(ids)].groupby(["jockey_id", "dist9"]).agg(n=("win", "size"), w=("win", "mean"), e=("exp_win", "mean"))
    jockeys = []
    for rank, j in enumerate(ids, 1):
        vals, ns, ex = [], [], []
        for b in jc.DIST9:
            if (j, b) in g.index:
                r = g.loc[(j, b)]
                n = int(r["n"])
                ns.append(n)
                vals.append(round(float(r["w"]), 4) if n >= SHOW_N else None)
                ex.append(round(float(r["e"]), 4) if n >= SHOW_N else None)
            else:
                ns.append(0); vals.append(None); ex.append(None)
        jockeys.append({"name": names[j], "rank": rank, "reading": jc.jockey_reading_of(j, names[j]), "values": vals, "n": ns, "expected": ex})
    return {
        "type": "jockey_distance_chart",
        "title": "騎手別 距離帯別勝率（全騎手 プルダウンで選択）",
        "xs": jc.DIST9,
        "baseline": [round(float(base.get(b)), 4) for b in jc.DIST9],
        "jockeys": jockeys,
        "minN": MIN_N,
        "baselineLabel": "全騎手の平均",
        "expectedLabel": "騎乗馬の人気からの見込み",
        "caption": f"2020〜2025年 JRA 平地。期間に{CHART_MIN_RIDES}騎乗以上の{len(ids)}人。点線・白抜きの点は{MIN_N}騎乗未満（参考）、"
                   f"{SHOW_N}騎乗未満の距離帯は省略。紫の点線は、その騎手が乗った馬の単勝オッズから見込まれる勝率。",
    }


def judge(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cnt = d.groupby("jockey_id").size()
    ids = cnt[cnt >= TABLE_MIN_RIDES].index.tolist()
    return jc.two_axis(d, ids, "dist4", jc.DIST4, "win"), jc.two_axis(d, ids, "dist4", jc.DIST4, "top3"), ids


def raw_within(d: pd.DataFrame, jid: str, level: str) -> dict[str, Any]:
    """生の勝率で、その距離帯とほかの距離帯を比べる（騎乗馬の強さを考えない比べ方）。"""
    dj = d[d["jockey_id"] == jid]
    a, b = dj.loc[dj["dist4"] == level, "win"], dj.loc[dj["dist4"] != level, "win"]
    ma, na, _, va = _stat(a)
    mb, nb, _, vb = _stat(b)
    return {"in": ma, "out": mb, "n_in": na, "n_out": nb, "p": _p(va, vb, ma, mb)}


def lab(level: str) -> str:
    return level.split("\n")[0]


def pv(sym: str, v: float) -> str:
    """「p=0.012」「q<0.001」の形。"""
    t = pq(v)
    return f"{sym}{t}" if t.startswith("<") else f"{sym}={t}"


def pq(q: float) -> str:
    return "–" if q is None or (isinstance(q, float) and math.isnan(q)) else ("<0.001" if q < 0.001 else f"{q:.3f}")


def detail_table(t: pd.DataFrame, jid: str, name: str) -> dict[str, Any]:
    rows = []
    for _, r in t[t["jockey_id"] == jid].iterrows():
        rows.append([lab(r["level"]), int(r["n"]), f"{jc.pct(r['rate'])}（{jc.pct(r['ci'][0])}〜{jc.pct(r['ci'][1])}）", jc.pct(r["avg"]),
                     jc.pct(r["exp"]), f"{jc.pt(r['diff'])}（{jc.pt_ci(r['diff_ci'])}）", r["label"]])
    return {"type": "table", "columns": ["距離帯", "騎乗数", "勝率（95%区間）", "全騎手の平均", "人気からの見込み", "実際 − 見込み（95%区間）", "2 軸の判定"],
            "rows": rows, "caption": f"{name}。判定は (a) 全騎手の平均との差、(b) 同じ騎手のほかの距離帯との「実際 − 見込み」の差を、それぞれ q<{jc.Q_SIG}（BH 補正）で判定。"}


def all_table(t: pd.DataFrame, ids: list[str], names: dict[str, str]) -> dict[str, Any]:
    by = {(r["jockey_id"], r["level"]): r for _, r in t.iterrows()}
    rows = []
    for j in jc.kana_order(ids, names):
        row: list[Any] = [names[j]]
        for lv in jc.DIST4:
            r = by[(j, lv)]
            mark = "" if r["short"] in ("–",) else f" {r['short']}"
            row.append(f"{jc.pct(r['rate'])}（{jc.pt(r['diff'])}）{mark}")
        rows.append(row)
    return {"type": "table", "columns": ["騎手", *[lab(x) for x in jc.DIST4]], "rows": rows,
            "caption": "勝率（かっこ内は 実際 − 人気からの見込み）。印は 2 軸の判定（苦手・得意・相対的に上/下。– ははっきりした差なし）。50 音順。"}


CANDIDATE_P = 0.01       # 補正前の p がこれ未満の組を「候補」として表に出す


def notable_table(t: pd.DataFrame, names: dict[str, str], metric_label: str) -> dict[str, Any]:
    """(b) 騎手内の差が、補正前の p<CANDIDATE_P になった組（多重比較の補正後は有意でないものを含む候補）。"""
    x = t[(t["pb"] < CANDIDATE_P) & (t["short"] != "標本不足")].copy()
    x["name"] = x["jockey_id"].map(names)
    x = x.sort_values("pb")
    rows = [[r["name"], lab(r["level"]), int(r["n"]), jc.pct(r["rate"]), jc.pct(r["avg"]), jc.pt(r["diff"]), jc.pt(r["other_diff"]),
             pq(r["pb"]), pq(r["qb"]), ("上" if r["diff"] > r["other_diff"] else "下")] for _, r in x.iterrows()]
    return {"type": "table", "columns": ["騎手", "距離帯", "騎乗数", metric_label, "全騎手の平均", "実際 − 見込み", "ほかの距離帯の 実際 − 見込み", "p 値", "q 値（補正後）", "ほかの距離帯と比べて"],
            "rows": rows, "caption": f"{metric_label}で、(b) 騎手内の差の補正前の p が {CANDIDATE_P} 未満の組。補正後の q はどれも {jc.Q_SIG} 以上なら、偶然でも出る範囲。"}


def analyze(d: pd.DataFrame) -> dict[str, Any]:
    names = jc.jockey_names(d)
    tw, t3, ids = judge(d)
    inv = {v: k for k, v in names.items()}
    a: dict[str, Any] = {"names": names, "tw": tw, "t3": t3, "ids": ids, "chart": chart_block(d, names),
                         "n_runs": len(d), "n_jockeys": len(ids)}
    a["kawada"] = inv["川田将雅"]
    a["lemaire"] = inv.get("C.ルメール") or next(k for k, v in names.items() if "ルメール" in v)
    a["kawada_raw"] = raw_within(d, a["kawada"], jc.DIST4[3])
    a["lemaire_raw"] = raw_within(d, a["lemaire"], jc.DIST4[3])
    a["long_avg"] = float(d.loc[d["dist4"] == jc.DIST4[3], "win"].mean())
    cnt_long = tw[tw["level"] == jc.DIST4[3]].copy()
    a["long_rank"] = cnt_long.sort_values("rate", ascending=False)["jockey_id"].tolist()
    h = jc.holdout_frame()
    a["holdout"] = {m: jc.holdout_check(h, a["lemaire"], "dist4", jc.DIST4[3], m) for m in ("win", "top3")}
    return a


def holdout_line(a: dict[str, Any], ll: pd.Series, lt3: pd.Series) -> str:
    """ルメールJの長距離を、集計に使っていない年で確かめた一文。"""
    hw, h3 = a["holdout"]["win"], a["holdout"]["top3"]
    if "diff" not in h3:
        return f"\n\n{jc.HOLDOUT_YEAR} 年のデータでは騎乗数が足りず、確かめられませんでした。"
    return (f"\n\n集計に使っていない {jc.HOLDOUT_YEAR} 年（1〜10 月、{h3['n']}騎乗）で、ほかの距離帯と比べた 実際 − 見込み の差を見ると、"
            f"複勝率 {jc.pt(h3['diff'])}（集計期間 {jc.pt(lt3['diff'] - lt3['other_diff'])}）、勝率 {jc.pt(hw['diff'])}（集計期間 {jc.pt(ll['diff'] - ll['other_diff'])}）でした。"
            "騎乗数が少なく区間が広い（複勝率で " + jc.pt_ci(h3["ci"]) + "pt）ので、今のところ「長距離が得意」とは確かめられていません。")


def row(t: pd.DataFrame, jid: str, level: str) -> pd.Series:
    return t[(t["jockey_id"] == jid) & (t["level"] == level)].iloc[0]


def article(a: dict[str, Any]) -> dict[str, Any]:
    tw, t3, names = a["tw"], a["t3"], a["names"]
    K, LM = a["kawada"], a["lemaire"]
    long_ = jc.DIST4[3]
    kl, ll = row(tw, K, long_), row(tw, LM, long_)
    kt3 = row(t3, K, long_)
    k_other = tw[(tw["jockey_id"] == K) & (tw["level"] != long_)]
    kr = a["kawada_raw"]
    lr = a["lemaire_raw"]
    counts = tw[~tw["short"].isin(["標本不足"])]["short"].value_counts().to_dict()
    counts3 = t3[~t3["short"].isin(["標本不足"])]["short"].value_counts().to_dict()
    n_tests = int((~tw["short"].isin(["標本不足"])).sum())
    k_rank_long = a["long_rank"].index(K) + 1
    l_rank_long = a["long_rank"].index(LM) + 1
    lt3 = row(t3, LM, long_)
    judged = tw[tw["short"] != "標本不足"]
    judged3 = t3[t3["short"] != "標本不足"]
    sig_w, sig_3 = int((judged["pb"] < 0.05).sum()), int((judged3["pb"] < 0.05).sum())
    n_flag = sum(v for k, v in counts.items() if k not in ("–",)) + sum(v for k, v in counts3.items() if k not in ("–",))

    body: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "川田J 長距離の勝率", "value": jc.pct(kl["rate"]), "sub": f"{int(kl['n'])}騎乗。全騎手の平均 {jc.pct(kl['avg'])}"},
            {"label": "川田J 長距離の 実際 − 人気の見込み", "value": jc.pt(kl["diff"]), "sub": f"見込み {jc.pct(kl['exp'])}。ほかの距離帯は {jc.pt(kl['other_diff'])}"},
            {"label": "ルメールJ 長距離の勝率", "value": jc.pct(ll["rate"]), "sub": f"{int(ll['n'])}騎乗。見込みとの差 {jc.pt(ll['diff'])}"},
        ]},
        a["chart"],
        {"type": "heading", "text": "「苦手」は 2 つの物差しで見る"},
        {"type": "text", "text":
            "「○○は長距離が苦手」と言うとき、2 つの意味が混ざりがちです。\n\n"
            "(a) 騎手間の比較: その距離で、ほかの騎手より成績が悪い。\n"
            "(b) 騎手内の比較: その騎手のほかの距離と比べて、成績が落ちる。\n\n"
            "トップ騎手は強い馬に乗るので、どの距離でも (a) では平均を大きく上回ります。一方、(b) を生の勝率で比べると、"
            "距離ごとの騎乗馬の強さの違い（長距離では人気馬に乗る機会が少ない、など）が混ざります。"
            "そこで (b) は「実際の勝率 − 騎乗馬の人気（単勝オッズ）から見込まれる勝率」で比べ、馬の強さの違いを除きました。\n\n"
            "この記事では、(a)(b) の両方で下回るときだけ「苦手」と呼びます。片方だけのときは、"
            "「平均より上だが、ほかの距離ほどではない」（(b) だけ）、「全体に平均より下（この距離が特に悪いわけではない）」（(a) だけ）と書き分けます。"},
        {"type": "table", "columns": ["(a) 全騎手の平均と比べて", "(b) ほかの距離帯と比べて（人気の見込みとの差）", "表現"], "rows": [
            ["下", "下", "苦手"],
            ["上・差なし", "下", "ほかの距離より落ちる（平均より上 / 並み）"],
            ["下", "差なし", "全体に平均より下（この距離が特に悪いわけではない）"],
            ["上", "上", "得意"],
            ["下・差なし", "上", "この騎手の中では良い"],
            ["上", "差なし", "全体に平均より上（この距離が特に良いわけではない）"],
        ], "caption": f"差の判定は、騎手 × 距離帯の数だけ検定を繰り返すため、Benjamini–Hochberg 補正の q<{jc.Q_SIG} を「差あり」とした。"},
        {"type": "heading", "text": "川田Jの長距離"},
        detail_table(tw, K, "川田将雅J"),
        {"type": "text", "text":
            f"川田Jの長距離の勝率は {jc.pct(kl['rate'])}（{int(kl['n'])}騎乗）で、4 つの距離帯の中では最も低い値です"
            f"（ほかの距離帯をまとめると {jc.pct(kr['out'])}、生の勝率の差の p={kr['p']:.2f}）。"
            f"ただし全騎手の長距離の平均 {jc.pct(kl['avg'])} の約 {kl['rate'] / kl['avg']:.1f} 倍で、(a) では明らかに上です"
            f"（{a['n_jockeys']}人中 {k_rank_long} 位）。\n\n"
            f"騎乗馬の人気から見込まれる勝率は {jc.pct(kl['exp'])} で、実際との差は {jc.pt(kl['diff'])}（95%区間 {jc.pt_ci(kl['diff_ci'])}pt）。"
            f"ほかの距離帯の差 {jc.pt(kl['other_diff'])} とほとんど同じで、(b) でも差は見られません。"
            f"長距離の勝率が少し低いのは、長距離で乗る馬の人気がほかの距離ほど高くないためと考えられます。"
            f"複勝率で見ても、長距離の 実際 − 見込み は {jc.pt(kt3['diff'])}（ほかの距離帯 {jc.pt(kt3['other_diff'])}、{pv('q', kt3['qb'])}）で、はっきりした差はありません。\n\n"
            "したがって「川田Jは長距離が苦手」は、(a)(b) のどちらの物差しでも支持されません。"
            f"ただし長距離の騎乗は 6 年で {int(kl['n'])} 回と少なく、95% 区間は広い点に注意してください。"},
        {"type": "heading", "text": "ルメールJの長距離"},
        detail_table(tw, LM, "C.ルメールJ"),
        {"type": "text", "text":
            f"ルメールJの長距離の勝率 {jc.pct(ll['rate'])}（{int(ll['n'])}騎乗）は、{a['n_jockeys']}人中 {l_rank_long} 位です。"
            f"人気からの見込み {jc.pct(ll['exp'])} を {jc.pt(ll['diff'])} 上回り、ほかの距離帯（{jc.pt(ll['other_diff'])}）より高めですが、"
            f"勝率では {pv('p', ll['pb'])}（補正後 {pv('q', ll['qb'])}）です。"
            f"複勝率では 実際 − 見込み が {jc.pt(lt3['diff'])}（ほかの距離帯 {jc.pt(lt3['other_diff'])}）で、補正前は {pv('p', lt3['pb'])} と小さいものの、"
            f"{a['n_jockeys']} 人 × 4 距離帯の検定を繰り返したことを補正すると {pv('q', lt3['qb'])} で、"
            + ("(a)(b) の両方で上回り「得意」と判定されました。" if lt3["short"] == "得意" else
               "「得意」とまでは言い切れません。")
            + holdout_line(a, ll, lt3)},
        {"type": "heading", "text": "距離で成績が変わる騎手は、データでは見つからない"},
        {"type": "text", "text":
            f"{TABLE_MIN_RIDES}騎乗以上の {a['n_jockeys']} 人 × 4 距離帯（騎乗 50 以上の {n_tests} 組）を 2 軸で判定すると、"
            + (f"「苦手」{counts.get('苦手', 0)} 組・「得意」{counts.get('得意', 0)} 組・「相対的に下」{counts.get('相対的に下', 0)} 組・「相対的に上」{counts.get('相対的に上', 0)} 組"
               f"（複勝率では 苦手 {counts3.get('苦手', 0)}・得意 {counts3.get('得意', 0)}・相対的に下 {counts3.get('相対的に下', 0)}・相対的に上 {counts3.get('相対的に上', 0)}）で、残りははっきりした差なしでした。"
               if n_flag else "勝率・複勝率のどちらでも、(b) 騎手内の差が補正後に残る組は 1 つもありませんでした（「苦手」「得意」は 0 組）。")
            + f"\n\n補正前の p<0.05 の組は 勝率で {sig_w} 組、複勝率で {sig_3} 組です。差が本当は無くても、{n_tests} 組を調べれば 5% の約 {round(n_tests * 0.05)} 組は偶然 p<0.05 になるので、"
            "数は偶然で出る範囲とほぼ同じです。下の表の「候補」も、まず偶然を疑うべき値です。\n\n"
            "距離帯ごとの勝率の違いは、多くの騎手で、乗る馬の人気の違いでほぼ説明できます。"
            "グラフでも、騎手の勝率（青）は人気からの見込み（紫の点線）とほぼ同じ形で上下します。"
            "騎手の「得意距離」を馬券に使うなら、生の勝率ではなく、人気から見込まれる値との差で見るのが安全です。"},
        notable_table(tw, names, "勝率"),
        notable_table(t3, names, "複勝率"),
        {"type": "accordion", "title": f"全 {a['n_jockeys']} 人の距離帯別 勝率と判定（50 音順）", "blocks": [all_table(tw, a["ids"], names)]},
        {"type": "note", "text":
            "人気から見込まれる勝率は、単勝オッズから求めた勝率（1/オッズ をレース内で合計 1 に直したもの）を 50 段階に分け、"
            "頭数の帯ごとに実際の勝率をあてはめたもの（判定では、距離帯ごとに全騎手で平均した 実際 − 見込み が 0 になるようにずらした）。オッズには騎手の評価も含まれるため、「実際 − 見込み」は市場が織り込んだ以上の上積みで、"
            "騎手の腕そのものより小さく出ます。同じレースの出走は独立でないため、信頼区間はやや狭めに出ている可能性があります。"},
    ]
    return {
        "slug": "jockey-long-distance",
        "title": "川田Jは長距離が苦手？——勝率は最低でも、人気を考えると差はない",
        "claim": "川田は長距離が苦手",
        "verdict": "not_supported",
        "summary": (f"川田Jの長距離勝率 {jc.pct(kl['rate'])} は 4 距離帯で最も低いものの、全騎手の平均 {jc.pct(kl['avg'])} を大きく上回り、"
                    f"騎乗馬の人気から見込まれる勝率との差（{jc.pt(kl['diff'])}）もほかの距離帯と同じでした。"
                    "騎手間・騎手内の 2 つの物差しで判定すると、距離で「苦手」「得意」と言える騎手は、多重比較を補正すると見つかりませんでした。"),
        "categories": ["jockey/distance"],
        "published": "2026-10-28",
        "updated": "2026-11-03",
        "dataset": {"period": "2020〜2025年", "conditions": "JRA 平地（障害を除く）。取消・除外を除く",
                    "sample": f"{a['n_runs']:,}出走。グラフは{CHART_MIN_RIDES}騎乗以上、判定は{TABLE_MIN_RIDES}騎乗以上の騎手"},
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
        print(json.dumps([b for b in col["body"] if b["type"] in ("stats", "text")], ensure_ascii=False, indent=1))
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
