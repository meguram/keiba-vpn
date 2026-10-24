"""tp-59 の章 docs/html/modeling/chapters/jockey_trainer_roi.html を、結果の JSON から作る。

python -m src.scripts.data.tp59_report --in /tmp/tp59_local          # ホストの <TP59_OUT> から持ってきた JSON のディレクトリ
入力: profile.json / census.json / rules.json / ev_overlay.json / ev_final.json / ev_robust.json / importance.json / uplift_screen.json / uplift_audit.json、
      任意で lgb_rs59/result.json / lgb_drop_rider/result.json（LightGBM での増分。無ければ『実行待ち』と書く）。
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/jockey_trainer_roi.html"

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --good:#34d399; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1150px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.ok { color: var(--good); } .ng { color: var(--bad); } code { background: #0b1428; padding: 1px 5px; border-radius: 4px; }"""


def e(x) -> str:
    return html.escape(str(x))


def f1(x, nd=1) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def ci(d: dict, nd=0) -> str:
    return f"{d['roi']:.{nd}f}%（{d['lo']:.{nd}f}〜{d['hi']:.{nd}f}）"


def table(head: list[str], rows: list[list], num_from: int = 1) -> str:
    h = "".join("<th" + (' class="n"' if i >= num_from else '') + f">{e(c)}</th>" for i, c in enumerate(head))
    b = ""
    for r in rows:
        b += "<tr>" + "".join(f"<td{' class=\"n\"' if i >= num_from else ''}>{c if isinstance(c, str) and c.startswith('<') else e(c)}</td>" for i, c in enumerate(r)) + "</tr>"
    return f"<table><tr>{h}</tr>{b}</table>"


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sec_persistence(prof: dict) -> str:
    out = ""
    for nm, label in (("jockey", "騎手"), ("trainer", "調教師")):
        x = prof[f"persist_{nm}"]
        rows = []
        for k, v in x["pairs"].items():
            rows.append([k, v["entities"]] + [f1(v.get(m), 2) for m in ("win_rate", "top3_rate", "roi_win", "roi_place", "ex_roi_win", "ex_roi_place", "resid_q", "resid_p")])
        out += f"<h3>{label}: 前年 → 翌年の順位相関（それぞれ前年・翌年とも出走 100 以上。2026 年は部分年なので 70 以上）</h3>"
        out += table(["年", "人数", "勝率", "複勝率", "単勝回収率", "複勝回収率", "単勝 超過", "複勝 超過", "勝ち−市場", "勝ち−モデル"], rows)
        rel = []
        for y, v in x["reliability"].items():
            rel.append([y, v["entities"], f1(v["win"].get("obs_sd"), 3), f1(v["win"].get("sampling_sd"), 3), f1(v["win"].get("reliability"), 2), f1(v["place"].get("reliability"), 2)])
        out += f"<p class=\"muted\">{label}の回収率のばらつきのうち本物の差の割合（信頼度）。観測した標準偏差と、標本の揺れだけで説明できる標準偏差を比べる。</p>"
        out += table(["年", "人数", "単勝 観測 SD", "単勝 揺れ SD", "単勝 信頼度", "複勝 信頼度"], rel)
    return out


def sec_decile(prof: dict) -> str:
    out = ""
    for nm, label in (("jockey", "騎手"), ("trainer", "調教師")):
        x = prof[f"persist_{nm}"]["top_decile"]
        rows = []
        for k, v in x.items():
            r = [k]
            for m in ("roi_win", "roi_place", "ex_roi_win", "win_rate"):
                if m in v:
                    c = v[m]
                    r.append(f"単{c['win']['top']['roi']:.0f}/{c['win']['all_eligible_roi']:.0f}・複{c['place']['top']['roi']:.0f}/{c['place']['all_eligible_roi']:.0f}")
                else:
                    r.append("—")
            rows.append(r)
        out += f"<h3>{label}: 前年の指標が上位 10% の全出走を翌年に買った回収率（上位 / 候補全体、%）</h3>"
        out += table(["年", "単勝回収率で選ぶ", "複勝回収率で選ぶ", "単勝の超過で選ぶ", "勝率で選ぶ"], rows)
    return out


def sec_census(cen: dict) -> str:
    names = {"j": "騎手", "t": "調教師", "jt": "騎手×調教師", "jv": "騎手×競馬場", "tv": "調教師×競馬場", "jsd": "騎手×芝ダ距離帯", "tsd": "調教師×芝ダ距離帯"}
    rows = []
    for ent, v in cen.items():
        if ent not in names:
            continue
        for bet, x in v.items():
            td25, td26 = x.get("top_decile_roi_2025"), x.get("top_decile_roi_2026")
            rows.append([names[ent], "単勝" if bet == "win" else "複勝", x["entities"], f"{x['sig_5pct_onesided']}（偶然の期待 {x['expected_by_chance']:.1f}）", x["bh_q10"], f1(x["t_max"], 2), f"{x['roi_median']:.0f}%", f"{100 * x['frac_roi_gt_100']:.0f}%",
                         f"{td25['roi']:.0f} / {x['all_roi_2025']:.0f}" if td25 else "—", f"{td26['roi']:.0f} / {x['all_roi_2026']:.0f}" if td26 else "—",
                         f1(x["persistence"]["2025"]["spearman_roi"], 2) if x["persistence"]["2025"]["spearman_roi"] is not None else "—"])
    return table(["単位", "券種", "対象数", "100% 超と言える（片側 5%）", "BH 補正後", "t の最大", "回収率の中央値", "100% 超の割合", "2025: 上位 10% / 全体", "2026: 上位 10% / 全体", "選択年→2025 の順位相関"], rows, 2)


def sec_pop(cen: dict) -> str:
    out = ""
    for nm, label in (("jockey", "騎手"), ("trainer", "調教師")):
        rows = []
        for per, g in cen["popularity_pooled"][nm].items():
            for k in ("top10", "11-30", "rest", "new"):
                v = g.get(k)
                if not v:
                    continue
                rows.append([per, {"top10": "前年の出走数 上位 10", "11-30": "11〜30 位", "rest": "それ以外", "new": "前年出走なし"}[k], v["n"], f"{v['median_win_odds']:.1f}", f"{100 * v['win_rate']:.2f}% / {100 * v['q_rate']:.2f}%",
                             f"{v['win_roi']['roi']:.0f}%", f"{v['ex_win']['roi']:+.1f}（{v['ex_win']['lo']:+.1f}〜{v['ex_win']['hi']:+.1f}）", f"{v['ex_place']['roi']:+.1f}（{v['ex_place']['lo']:+.1f}〜{v['ex_place']['hi']:+.1f}）"])
        out += f"<h3>{label}（前年の出走数で分けた翌年の成績）</h3>" + table(["期間", "群", "出走", "単勝オッズ中央値", "勝率 / 市場の見込み", "単勝回収率", "単勝 超過（pt）", "複勝 超過（pt）"], rows, 2)
    return out


def sec_calibration(prof: dict) -> str:
    out = ""
    for nm, label in (("jockey", "騎手"), ("trainer", "調教師")):
        rows = []
        for per in ("2022-2024", "2025", "2026"):
            xs = [x for x in prof["calibration"][nm] if x["period"] == per]
            rows.append([per, "モデル"] + [f"{100 * (x['win_rate'] - x['model_p']):+.2f}" for x in xs])
            rows.append([per, "市場"] + [f"{100 * (x['win_rate'] - x['market_q']):+.2f}" for x in xs])
        out += f"<h3>{label}の強さ（当年より前の勝率の縮約値）の 10 分位ごとの『実際の勝率 − 予測の勝率』（pt）</h3>"
        out += table(["期間", "予測"] + [f"d{i}" for i in range(10)], rows, 2)
    return out


def sec_conditions(prof: dict) -> str:
    c = prof["conditions"]
    out = ""
    for key, label in (("jockey_move", "乗り替わり（up=前走より強い騎手へ、down=弱い騎手へ、flat=差が 1pt 未満、first=初出走）"), ("j_tier", "騎手の強さ 5 分位（j5=最強）"), ("t_tier", "調教師の強さ 5 分位")):
        rows = []
        for lv in sorted({x["level"] for per in c[key].values() for x in per}):
            r = [lv]
            for per in ("2022-2024", "2025", "2026"):
                x = next((y for y in c[key][per] if y["level"] == lv), None)
                r.append(f"単{x['win']['roi']:.0f}・複{x['place']['roi']:.0f}" if x else "—")
            rows.append(r)
        out += f"<h3>{label}</h3>" + table(["区分", "2022〜2024", "2025", "2026"], rows)
    return out


def sec_grid(prof: dict) -> str:
    g = prof["conditions"]["jtier_x_odds"]
    bands = sorted({x["band"] for x in g})
    rows = []
    for t in sorted({x["j_tier"] for x in g}):
        r = [t]
        for b in bands:
            x = next((y for y in g if y["j_tier"] == t and y["band"] == b), None)
            r.append(f"{x['win_roi']:.0f}/{x['place_roi']:.0f}" if x else "—")
        rows.append(r)
    return table(["騎手の強さ"] + [b[4:] for b in bands], rows)


def sec_rules(rules: dict) -> str:
    rows = []
    for x in rules["top10"][:6]:
        py = x["per_year"]
        rows.append([x["rule"], f"{x['t_sel']:.2f}"] + [f"{py[y].get('n', 0)} / {py[y].get('roi', 0):.0f}%" for y in ("2022", "2023", "2024", "2025", "2026")])
    t1 = table(["規則（族|券種|指標|件数下限|上位 K）", "t（選択年）", "2022 件数/回収率", "2023", "2024", "2025（確認）", "2026（テスト②）"], rows, 1)
    fam = []
    names = {"j": "騎手", "t": "調教師", "jt": "騎手×調教師", "jv": "騎手×競馬場", "tv": "調教師×競馬場", "jsd": "騎手×芝ダ距離帯", "tsd": "調教師×芝ダ距離帯", "tc": "調教師×クラス", "jo": "騎手×人気帯", "to": "調教師×人気帯", "jc": "騎手×クラス"}
    for k, v in rules["family_rc"].items():
        fam.append([names.get(k, k), f"{v['t_max']:.2f}", f"{v['p_value']:.3f}", v["best_rule"]])
    t2 = table(["族", "t の最大", "Reality Check の p（族内）", "族内で最良の規則"], fam, 1)
    return t1 + "<p class=\"muted\">族ごと（その族の規則だけで最良を選んだ場合）:</p>" + t2


def sec_confirm(rules: dict) -> str:
    rows = []
    for n, v in list(rules["confirm"].items())[:12]:
        def g(k):
            c = v.get(k, {})
            return f"{c['pooled']['roi']:.0f}%（{c['pooled']['lo']:.0f}〜{c['pooled']['hi']:.0f}）n={c['pooled']['n']}" if "pooled" in c else "—"
        rows.append([n, g("sel_2022_2024"), g("2025"), g("2026")])
    return table(["規則", "選択年 2022〜2024", "2025（確認 1 回）", "2026（テスト②）"], rows, 1)


def sec_overlay(ov: dict) -> str:
    import collections

    by = collections.defaultdict(dict)
    for x in ov["overlay"]:
        by[(x["bet"], x["tau"], x["cond"])][x["period"]] = x
    cond_names = {"j_str_hi": "騎手の強さが上位半分", "t_str_hi": "調教師の強さが上位半分", "j_ex_pos": "騎手の超過払戻が正（過去）", "t_ex_pos": "調教師の超過払戻が正", "jt_ex_pos": "騎手×調教師の超過払戻が正",
                  "j_res_pos": "騎手の勝ち数が市場の見込みを上回る（過去）", "t_res_pos": "調教師の勝ち数が市場の見込みを上回る", "j_up": "強い騎手への乗り替わり", "j_down": "弱い騎手への乗り替わり", "j_same": "騎手が前走と同じ"}
    rows = []
    for c in ov["conds"]:
        d = [by[("win", 1.2, c)].get(p) for p in ("sel_2022_2024", "2025", "2026")]
        if not all(d):
            continue
        rows.append([cond_names.get(c, c)] + [f"{x['diff']:+.1f}（p={x['p_le0']:.2f}）" for x in d])
    t = table(["条件（単勝 EV ≥ 1.2 の中での、条件あり − なし の回収率の差 pt）", "選択年 2022〜2024", "2025", "2026"], rows, 1)
    tot = same_pos = same_neg = 0
    for k, v in by.items():
        if len(v) == 3:
            tot += 1
            d = [v[p]["diff"] for p in ("sel_2022_2024", "2025", "2026")]
            same_pos += all(x > 0 for x in d)
            same_neg += all(x < 0 for x in d)
    return t, tot, same_pos, same_neg


def sec_final(ev: dict) -> str:
    rows = []
    for x in ev["rows"]:
        if x["bet"] != "win" or x["tau"] not in (1.0, 1.2, 1.5):
            continue
        def g(k):
            c = x.get(k, {})
            if "pooled" not in c:
                return "—"
            return f"{c['pooled']['roi']:.0f}%（{c['pooled']['lo']:.0f}〜{c['pooled']['hi']:.0f}）n={c['pooled']['n']}"
        rows.append([f"EV≥{x['tau']}", {"none": "条件なし", "j_res_pos": "騎手", "t_res_pos": "調教師", "either": "どちらか", "both": "両方"}[x["cond"]], f"{x['t_sel']:.2f}", g("sel_2022_2024"), g("2025"), g("2026")])
    return table(["EV", "市場に対する残差が正", "t（選択年）", "選択年 2022〜2024", "2025", "2026"], rows, 2)


def sec_robust(rb: dict) -> str:
    p = rb["pooled"]
    t1 = table(["期間", "件数", "回収率（95% 区間）", "的中率"], [[k, v["n"], ci(v), f"{100 * v['hit']:.1f}%"] for k, v in p.items()], 1)
    rows = [[k] + [f1(v.get(f"drop{m}"), 0) for m in (0, 1, 3, 5, 10)] for k, v in rb["drop_top"].items()]
    t2 = table(["期間", "全部", "最大 1 本を除く", "3 本", "5 本", "10 本"], rows, 1)
    rows = []
    for cap, v in rb["odds_cap"].items():
        rows.append([f"≤{cap} 倍"] + [f"{x['roi']:.0f}%（{x['lo']:.0f}〜{x['hi']:.0f}）" if x else "—" for x in (v["sel_2022_2024"], v["2025"], v["2026"])])
    t3 = table(["単勝オッズの上限", "選択年", "2025", "2026"], rows, 1)
    rows = [[k, v["n_obs"], f"{v['observed_roi']:.0f}%", f"{v['perm_mean']:.0f}%（SD {v['perm_sd']:.0f}）", f"{v['perm_q95']:.0f}%", f"{v['p_ge_obs']:.3f}"] for k, v in rb["flag_permutation"].items()]
    t4 = table(["期間", "件数", "実際の回収率", "条件を入れ替えたときの回収率", "入れ替えの上位 5%", "入れ替えで実際以上になる割合"], rows, 1)
    ys = [[y, v["n"], ci(v), f"{100 * v['hit']:.1f}%", f"{v['median_odds']:.0f}"] for y, v in rb["by_year"].items()]
    t5 = table(["年", "件数", "回収率", "的中率", "単勝オッズ中央値"], ys, 1)
    return t5, t1, t2, t3, t4


def sec_feat(imp: dict, scr: dict, rs: dict | None, drop: dict | None) -> str:
    rows = []
    for m in ("lgb_extras_ped", "lgb_extras", "lgb_softmax"):
        for tg in ("win", "top2", "top3"):
            x = imp.get(m, {}).get(tg)
            if not x:
                continue
            g = x["group_share_mean"]
            rows.append([m, tg, f"{100 * g.get('jockey', 0):.2f}%", f"{100 * g.get('trainer', 0):.2f}%", f"{100 * g.get('jockey_weight(斤量)', 0):.2f}%"])
    t_imp = table(["メンバー", "目的", "騎手の列（gain）", "調教師の列（gain）", "斤量（参考）"], rows, 2)
    cols = imp["lgb_extras_ped"]["win"]["columns"]
    rows = [[c, f"{100 * v['gain_share']:.3f}%", f"{v['rank_median']:.0f}"] for c, v in sorted(cols.items(), key=lambda kv: -kv[1]["gain_share"])]
    t_cols = table(["列（lgb_extras_ped の勝率）", "gain 比率", "順位（1,215 列中、5 seed の中央値）"], rows, 1)
    rows = []
    for k, v in scr["groups"].items():
        w, t2, t3 = v["win"], v["top2"], v["top3"]
        rows.append([k, len(v["cols"]), f"{w['delta']:+.5f}（{w['lo']:+.5f}〜{w['hi']:+.5f}）", f"{t2['delta']:+.5f}", f"{t3['delta']:+.5f}", "通過" if v["adopt"] else "—"])
    t_scr = table(["群", "列数", "勝率 Δ（95% 区間）", "連対 Δ", "複勝 Δ", "一次選別"], rows, 1)
    def lg(r, label):
        if not r:
            return (f"<p>{label}: <b>実行待ち</b>（ホストの共有スロット 3 本が他の評価で埋まっているため、<code>/tmp/heavy_run.sh</code> の待ち行列に入れてある）。"
                    "完了すると <code>/home/jovyan/work/tp59_out/lgb_*/result.json</code> に出る。章への反映は、その JSON をローカルに持ってきて <code>python -m src.scripts.data.tp59_report --in &lt;ディレクトリ&gt;</code>。</p>")
        c, py = r["compare"], r["per_year"]
        return (f"<p>{label}: 勝率 Δ {c['Δwin']:+.5f}（上限 {c['win_hi']:+.5f}）・連対 {c['Δtop2']:+.5f}・複勝 {c['Δtop3']:+.5f}、採否 {e(c['採用'])}。年別の勝率 Δ: "
                + "、".join(f"{y} {v['Δwin']:+.5f}" for y, v in py.items()) + f"。{r['n_cols']} 列。</p>")
    return t_imp, t_cols, t_scr, lg(rs, "V5（1,142 列）+ OOF 残差の群 4 列（LightGBM、検証 2022・2023・2024、seed 0・1）"), lg(drop, "V5 から騎手・調教師の通算成績 10 列を外す（同じ設定）")


def sec_flagfree(ff: dict) -> str:
    rows = []
    for per in ("2021", "2022", "2023", "2024", "2025", "2026", "2021-2024", "2025-2026", "2021-2026"):
        for band in ("all", ">=30"):
            x = next((r for r in ff["rows"] if r["period"] == per and r["band"] == band), None)
            if not x:
                continue
            w, pl = x["win"], x["place"]
            rows.append([per, "全オッズ" if band == "all" else "単勝 30 倍以上", x["n_flag"], f"{w['flag']['roi']:.0f}% / {w['other_roi']:.0f}%", f"{w['diff']['diff']:+.1f}（p={w['diff']['p_le0']:.2f}）",
                         f"{pl['flag']['roi']:.0f}% / {pl['other_roi']:.0f}%", f"{pl['diff']['diff']:+.1f}（p={pl['diff']['p_le0']:.2f}）"])
    return table(["期間", "オッズ帯", "条件ありの件数", "単勝 条件あり / なし", "差（pt）", "複勝 条件あり / なし", "差（pt）"], rows, 2)


def best_row_periods(evf: dict) -> dict:
    x = next(r for r in evf["rows"] if r["bet"] == "win" and r["tau"] == 1.5 and r["cond"] == "both")
    return {k: v for k, v in x.items() if isinstance(v, dict) and "pooled" in v}


def build(d: Path) -> str:
    prof, cen, rules = load(d, "profile.json"), load(d, "census.json"), load(d, "rules.json")
    ov, evf, rb = load(d, "ev_overlay.json"), load(d, "ev_final.json"), load(d, "ev_robust.json")
    imp, scr = load(d, "importance.json"), load(d, "uplift_screen.json")
    ff = load(d, "flag_modelfree.json")
    rs, drop = load(d, "lgb_rs59/result.json"), load(d, "lgb_drop_rider/result.json")
    pop = prof["population"]
    pj = prof["persist_jockey"]["pairs"]
    pt = prof["persist_trainer"]["pairs"]
    def rng(x, key, ys):
        v = [x[k][key] for k in ys if x[k].get(key) is not None]
        return f"{min(v):.2f}〜{max(v):.2f}"
    ys_sel = ["2020->2021", "2021->2022", "2022->2023", "2023->2024", "2024->2025"]
    units = [(ent, bet, x) for ent, v in cen.items() if ent != "popularity_pooled" for bet, x in v.items()]
    tot_tests = sum(x["entities"] for _, _, x in units)
    tot_exp = sum(x["expected_by_chance"] for _, _, x in units)
    tot_sig = sum(x["sig_5pct_onesided"] for _, _, x in units)
    tot_bh = sum(x["bh_q10"] for _, _, x in units)
    roi_corr = [v[m] for g in (pj, pt) for k, v in g.items() if k in ys_sel for m in ("roi_win", "roi_place") if v.get(m) is not None]
    rel_max = max(v["win"].get("reliability", 0) or 0 for g in (prof["persist_jockey"], prof["persist_trainer"]) for v in g["reliability"].values())
    frac_win = [x["frac_roi_gt_100"] for _, b, x in units if b == "win"]
    frac_pl = [x["frac_roi_gt_100"] for _, b, x in units if b == "place"]
    top10 = rules["top10"]
    def yr(y):
        v = [x["per_year"][y]["roi"] for x in top10 if x["per_year"][y].get("n", 0) > 0]
        return f"{min(v):.0f}〜{max(v):.0f}%"
    cf = list(rules["confirm"].values())[:10]
    def pooled_rng(k):
        v = [c[k]["pooled"]["roi"] for c in cf if "pooled" in c.get(k, {})]
        return f"{min(v):.0f}〜{max(v):.0f}%"
    dd_max = max(v["dd"]["max_drawdown_yen"] for k, v in best_row_periods(evf).items() if "dd" in v)
    dd_run = max(v["dd"]["longest_losing_races"] for k, v in best_row_periods(evf).items() if "dd" in v)
    t_ov, n_tr, n_pos, n_neg = sec_overlay(ov)
    t5, t1, t2, t3, t4 = sec_robust(rb)
    t_imp, t_cols, t_scr, lg_rs, lg_drop = sec_feat(imp, scr, rs, drop)
    best = next(x for x in evf["rows"] if x["bet"] == "win" and x["tau"] == 1.5 and x["cond"] == "both")
    base15 = next(x for x in evf["rows"] if x["bet"] == "win" and x["tau"] == 1.5 and x["cond"] == "none")
    rc = rules["reality_check"]
    body = f"""<a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>騎手・調教師の分析と、回収率として稼げるか（tp-59）</h1>
<p class="muted">2026-10-21、ユーザー質問『調教師、騎手の分析及び特徴量への寄与はすでに分析済みですか？勝率・複勝率に加えて、単勝・複勝の回収率としての扱いにも注目してほしい。最終的に稼げるかが問題』への回答。
資料: <a href="training_pipeline_decisions.html">決定ログ</a>（§33 G2 関係者、§17 tp-19、§29 tp-28）・<a href="betting_edge_search.html">馬券で回収率 100% を超える余地の探索（tp-36）</a>・<a href="lap_groups_and_trainer_stats.html">調教師別・コース別の調教成績（tp-35）</a>。
回収率は『1 出走（1 点）に 100 円』。区間はレース単位ブートストラップ。選択年は 2022〜2024、2025 は確認、2026 年はテストセット②（選択に使わない。報告のみ）。市場オッズは買い目の評価にだけ使い、特徴量には使わない。騎手・調教師は ID で扱う。</p>

<div class="note"><b>要点（先に答え）.</b>
<ul>
<li><b>分析済みか</b>: 特徴量としての寄与は<b>済み</b>（§33 の G2 で、騎手・調教師・騎手×調教師・騎手×馬など 12 の組を、競馬場・芝ダ距離帯・クラス・馬場・頭数・枠・間隔・出走回数・性齢の 11 条件、直近 7〜365 日、連勝・連敗、乗り替わりまで試し、<b>馬主系だけが採用で、騎手・調教師系は上乗せなし</b>）。一方、<b>騎手・調教師そのものの単勝・複勝回収率を主眼にした調査は今回が初め</b>。</li>
<li><b>勝率・複勝率は強く再現するが、回収率は再現しない</b>。前年→翌年の順位相関は、騎手の勝率 {rng(pj, 'win_rate', ys_sel)}・複勝率 {rng(pj, 'top3_rate', ys_sel)}、調教師は {rng(pt, 'win_rate', ys_sel)}・{rng(pt, 'top3_rate', ys_sel)}。単勝回収率は騎手 {rng(pj, 'roi_win', ys_sel)}、調教師 {rng(pt, 'roi_win', ys_sel)}、複勝回収率は {rng(pj, 'roi_place', ys_sel)}・{rng(pt, 'roi_place', ys_sel)}。回収率のばらつきのうち本物の差は単勝で最大 {rel_max:.2f}。</li>
<li><b>『回収率が 100% を超える』と言える騎手・調教師・組み合わせは 1 つも無い</b>（7 単位・延べ {tot_tests:,} の関係者×券種を全数調査。偶然に出る期待数 約 {tot_exp:.0f} 件に対し実際は {tot_sig} 件、BH 補正後 {tot_bh}）。前年の回収率上位 10% を翌年に買っても全体並み。</li>
<li><b>市場は騎手・調教師の強さを織り込み済み</b>。強さの 10 分位ごとの『実際の勝率 − 予測』は市場もモデルも ±0.5pt 以内。人気騎手（前年の出走数上位 10）は平均 5pt 前後<b>買われすぎ</b>（単勝の超過 −5.4pt、区間 −9.2〜−1.4）で、調教師は −6.8pt。ただし 2026 年は逆向きで安定しない。</li>
<li><b>回収率で買う規則 {rules['n_rules']} 通り</b>は Reality Check で p = {rc['p_value']:.4f}（t の最大 {rc['t_max']:.2f}）。選択年の合算で上位 10 規則が {pooled_rng('sel_2022_2024')} 出ても、2025 年は {pooled_rng('2025')}、2026 年は {pooled_rng('2026')} に戻る。</li>
<li><b>唯一の手がかり</b>: 現行の単勝 EV（V4 の勝率 × 確定オッズ）≥ 1.5 の買い目のうち、<b>騎手と調教師の両方が『過去の勝ち数が市場の見込みを上回る』</b>ものに絞ると、{ci(best['sel_2022_2024']['pooled'])}（選択年）・{ci(best['2025']['pooled'])}（2025）・{ci(best['2026']['pooled'])}（2026）で、条件なし（約 100%）より 30pt 前後高く、条件の付け替え検定でも偶然は考えにくい（選択年 p = {rb['flag_permutation']['sel_2022_2024']['p_ge_obs']:.3f}）。<b>ただし未確証</b>: 事後に絞った候補で（Reality Check p = {evf['reality_check']['p_value']:.2f}）、2025・2026 は絞り込みの途中で結果を見ており独立の確認にならず、上位の的中を数本外すと 100% を割り、モデルを使わない確認では独立の 2021 年が逆向き。規則を凍結して<b>新しいレースで前向きに追跡</b>する（確証には約 3,300 件、約 2 年）。</li>
<li><b>特徴量</b>: 既存の騎手・調教師の列は V5 の gain の約 1%。『馬ごとの期待値との差』の上積み群は一次選別で効果なし。一方、<b>V5 の OOF に対する関係者の残差</b>は一次選別を通る（勝率 −0.00026、区間 −0.00041〜−0.00012）。LightGBM での評価は下記。</li>
</ul></div>

<h2>1. すでに分析済みだったこと／今回新しくやったこと</h2>
{table(['領域', '内容', '結果', '出所'], [
['騎手・調教師の通算成績（B0）', 'jockey_prior_*・trainer_prior_*（件数・勝数・複勝数・勝率・複勝率）と、レース内の相対化（_rgap・_rrank・_rz）', 'V4 から V5 まで全モデルに入力。gain の割合は下記', 'tp-03 / V4 の列'],
['G2 関係者（騎手・調教師・組み合わせ）', '12 の組 × 11 条件 × {勝, 複勝, 正規化着順, 上がり, 位置}、直近 7〜365 日、連勝・連敗、規模、同日、乗り替わり', '馬主系 4 群 125 列だけ採用。騎手・調教師系は上乗せなし', '§33 G2'],
['調教師 × 追い切りラップ × コース', 'パターン別の勝率・回収率', '勝率・複勝率は翌年も再現、回収率は再現せず。区間下限 > 100% なし', '§15 tp-35 / §17 tp-19'],
['追い切りパターンの統計（調教師別を含む）', 'tp-28 の 7 群', '全群不採用', '§29 tp-28'],
['<b>今回: 騎手・調教師そのものの回収率</b>', '再現性・全数調査・市場への織り込み・528 規則・EV への重ね掛け・上積みと残差の特徴量', '本章', 'tp-59'],
])}

<h2>2. V5 の中の騎手・調教師</h2>
<p>V5 の LightGBM（5 seed 平均）で、騎手・調教師の列が使う gain の割合。1,215 列のうちの通算成績 10 列と、その相対化 3 列（_rgap・_rrank・_rz）、斤量を含む。</p>
{t_imp}
{t_cols}
<p class="muted">合計しても勝率で約 1%、複勝で約 0.7%。V5 は騎手・調教師の通算成績をほとんど使っておらず、馬の力・メンバー・コースの列が主。群を外した増分は下記（LightGBM）。</p>
{lg_drop}

<h2>3. 勝率・複勝率は再現するが、回収率は再現しない</h2>
<p>騎手・調教師を ID で集計し、各年で出走 100 以上（2026 年は部分年なので 70 以上）の関係者を、前年と翌年で順位相関（Spearman）。『超過』は、単勝オッズの帯ごとの平均払戻を引いた払戻（人気馬ばかり乗る関係者が高く見えるのを除く）。『勝ち−市場』は（勝った数 − 市場の勝率の合計）/ 出走数、『勝ち−モデル』は V4 の勝率で同じことをしたもの。</p>
{sec_persistence(prof)}
<p>勝率・複勝率の順位相関は 0.7〜0.9 と高く、騎手・調教師の腕は確かにある。一方、回収率の相関は 2024 年までの組で {min(roi_corr):.2f}〜{max(roi_corr):.2f} に留まり（騎手の 2025→2026 の単勝だけ 0.43 と高いが、部分年の 1 組）、<b>信頼度（本物の差の割合）は単勝で最大 {rel_max:.2f}</b>。同じ騎手の回収率が来年も高い、とは言えない。</p>

<h3>前年の上位 10% を翌年に買う</h3>
{sec_decile(prof)}
<p>勝率で選ぶと回収率は 72〜81% と全体（70〜75%）より少し高いが、これは勝率の高い騎手・調教師が人気馬に乗る（オッズが低い）ため払戻率が人気に寄るだけで、100% には遠い。回収率で選んでも全体並み。</p>

<h2>4. 100% を超える関係者はいるか（全数調査）</h2>
<p>選択年（2022〜2024 の合算）の単勝・複勝の回収率について、レース別の利益から t 値を出して『回収率が 100% を超える』を片側検定。件数の下限は単独 300 出走、組み合わせ 100 出走。Benjamini-Hochberg（q=10%）で多重性を補正。右の列は、選択年で t が上位 10% の関係者を 2025・2026 に買った回収率（全体との比較）と、順位相関。</p>
{sec_census(cen)}
<p>どの単位でも、100% を超えると言える関係者は偶然の期待数より少ない（回収率の中央値は 59〜74%。100% を超える関係者の割合は単勝で {100*min(frac_win):.0f}〜{100*max(frac_win):.0f}%、複勝で {100*min(frac_pl):.0f}〜{100*max(frac_pl):.0f}%）。選択年の上位は翌年以降に平均へ戻る。</p>

<h2>5. 市場が織り込んでいるか</h2>
<h3>強さ別の較正（モデルと市場のずれ）</h3>
{sec_calibration(prof)}
<p>強さの 10 分位のどこでも、実際の勝率と予測（モデル・市場とも）のずれは数 0.1pt 以内（年ごとのぶれは標本の揺れ）。<b>騎手・調教師の強さによる体系的な過小・過大評価は、モデルにも市場にも無い</b>。</p>
<h3>人気の騎手・厩舎は買われすぎか</h3>
{sec_pop(cen)}
<p>前年の出走数が上位 10 の騎手・調教師は、2021〜2025 年の合算で単勝の超過がマイナス（騎手 −5.4pt、調教師 −6.8pt。区間は 0 を含まない）で、<b>人気が回収率を押し下げている</b>。ただし 2026 年は区間が広く符号も逆で、買い続ける根拠にはならない。</p>
<h3>騎手の強さ × 単勝オッズ帯の回収率（2022〜2024 の合算、単勝/複勝 %）</h3>
{sec_grid(prof)}
<p>オッズ帯の影響が圧倒的（本命側 80% 台、100 倍超は 50% 前後）で、同じオッズ帯なら騎手の強さによる差はほぼ無い。『強い騎手の穴馬が妙味』といった形は出ていない。</p>
<h3>条件別（乗り替わり・強さ）の回収率</h3>
{sec_conditions(prof)}

<h2>6. 回収率で買う規則を全部試す</h2>
<p>関係者 11 種（騎手・調教師・騎手×調教師・騎手×競馬場・騎手×芝ダ距離帯・騎手×クラス・調教師×競馬場・調教師×芝ダ距離帯・調教師×クラス・騎手×人気帯・調教師×人気帯）× 券種 2（単勝・複勝）× 指標 2（払戻の縮約平均／超過払戻）× 件数下限 3 × 上位 K 4 ＝ <b>{rules['n_rules']} 規則</b>。毎年、<b>前年までの全期間の成績だけ</b>で上位 K のセルを選び、その年の該当する全出走に 100 円。選択年 2022〜2024 を Reality Check にかけ、上位を 2025・2026 で見る。</p>
<p><b>Reality Check: t の最大 {rc['t_max']:.2f}、p = {rc['p_value']:.4f}</b>（{rules['n_rules']} 規則の t の中央値は {rules['dist_t']['mean']:.1f}、t &gt; 2 の規則は 0 本）。選択年の上位 6 規則:</p>
{sec_rules(rules)}
<p>選択年の上位 10 規則の回収率は 2022 年に {yr('2022')}（直前の履歴が 2020〜2021 年の 2 年しかなく、セルの順位が不安定で、少数の的中が平均を押し上げた）、2023 年 {yr('2023')}、2024 年 {yr('2024')}、2025 年 {yr('2025')}、2026 年 {yr('2026')}。上位と族ごとの最良を、期間をまとめて見ると:</p>
{sec_confirm(rules)}

<h2>7. 現行の推奨（EV）に騎手・調教師の条件を重ねる</h2>
<p>V4 の勝率 × 確定単勝オッズ（EV）≥ τ の買い目に、騎手・調教師の条件（当年より前の成績だけで作る）を重ねたとき、『条件あり − なし』の回収率の差。10 条件 × τ 5 通り × 券種 2 ＝ 100 通りを、選択年・2025・2026 の 3 期間で見た（単勝 EV ≥ 1.2 の例）:</p>
{t_ov}
<p>100 通りのうち<b>3 期間とも差が正のもの {n_pos} 本・3 期間とも負のもの {n_neg} 本</b>（符号が独立なら各 12.5 本ずつ程度）。符号が安定して正なのは『騎手（調教師）の勝ち数が市場の見込みを上回る』系（<code>j_res_pos</code>・<code>t_res_pos</code>）に集中し、他の条件（強さ上位、乗り替わり、騎手が同じ）は期間で符号が入れ替わる。</p>
<h3>候補規則の族（Reality Check）</h3>
<p>上の観察から、事後に『単勝 EV ≥ τ × {{条件なし, 騎手, 調教師, どちらか, 両方}}』と複勝の同じ形、合計 {evf['n_rules']} 規則の族を作り、選択年で Reality Check。<b>t の最大 {evf['reality_check']['t_max']:.2f}、p = {evf['reality_check']['p_value']:.2f}</b>（最良は『単勝 EV ≥ 1.5 かつ両方』）。複勝は全規則で 100% 未満か同等。</p>
{sec_final(evf)}
<p class="muted">条件なしの EV ≥ 1.5 は選択年 {ci(base15['sel_2022_2024']['pooled'])}・2025 {ci(base15['2025']['pooled'])}・2026 {ci(base15['2026']['pooled'])} と 100% 付近（tp-36 と同じ）。</p>
<h3>最良の候補の頑健性（単勝 EV ≥ 1.5 かつ騎手・調教師の残差がともに正）</h3>
{t5}
{t1}
<p>的中の大きい順に k 本を除いた回収率（%）:</p>
{t2}
<p>単勝オッズの上限を付けた回収率:</p>
{t3}
<p><b>条件の付け替え検定</b>: その年の騎手どうし・調教師どうしで『残差が正』の印を無作為に入れ替えて同じ規則を評価（2,000 回）。印が騎手・調教師の実際の成績と結びついていなければ、入れ替えても同じ回収率になるはず。</p>
{t4}
<h3>モデルを使わない確認（2021 年は独立の標本）</h3>
<p>EV もモデルも使わず、『騎手・調教師の勝ち数が当年より前に市場の見込みを上回る（ともに）』という印だけを、全出走・単勝 30 倍以上で、印のない出走と比べる。<b>2021 年の印は 2020 年だけで作る</b>ので、これまでの選択年・確認年とは無関係の標本になる。</p>
{sec_flagfree(ff)}
<p>2021 年は<b>逆向き</b>（全体 −4.6pt、30 倍以上 −10.9pt）。2022 年以降は正が多く（30 倍以上で +9〜+25pt）、2021〜2026 年の合算では 30 倍以上で +12pt（p = 0.02）だが、印のある出走自体の回収率は 70% で、印だけでは 100% に届かない。効果があるとしても小さく、年ごとにぶれる。</p>
<div class="note"><b>この候補をどう扱うか.</b>
<ul>
<li>良い点: 5 年すべてで 122〜147%、条件の付け替え検定で選択年 p = {rb['flag_permutation']['sel_2022_2024']['p_ge_obs']:.3f}・2026 p = {rb['flag_permutation']['2026']['p_ge_obs']:.3f}（2025 は {rb['flag_permutation']['2025']['p_ge_obs']:.2f}）。モデルの勝率に対する残差でも同じ向き（診断: 騎手の残差が正のとき 2025・2026 とも差が正）。V5 の OOF に対する関係者の残差が小さいが有意な情報を持つ（第 9 節）ことと整合する。</li>
<li>弱い点: ①<b>事後に絞った</b>（100 通りの観察から族を作った）。族内 Reality Check の p は {evf['reality_check']['p_value']:.2f} で、観察の 100 通りを含めると更に緩い。②<b>2025・2026 は独立の確認ではない</b>（観察の段階で 3 期間を並べて見た）。③<b>長い穴馬に依存</b>（中央値のオッズは約 50 倍、的中率 3〜4%）: 的中の大きい 5 本を除くと 2025 で 87%、2026 で 83%。オッズ 30 倍以下に限ると約 100% で、優位は 30 倍超の穴馬にある。④確定オッズで評価している（購入時点のオッズは違う。穴馬は直前に人気が上がり下がりしやすい）。⑤区間の下限が 100% を超えるのは選択年の 100.9% だけ。⑥モデルを使わない確認で、独立の 2021 年（印を 2020 年だけで作った）は逆向き。</li>
<li>扱い: <b>採用しない。凍結して前向きに追跡する</b>。規則・条件表（2025 年までの成績で作った騎手 117 人・調教師 118 人。ID のみ）を <code>config/tp59_frozen_rule.json</code> に固定（2026-10-21）。2026-10-22 以降のレースで <code>python -m src.scripts.data.tp59_track track --from 20261022</code> を回すと、買い目・的中・回収率を集計する。区間の下限が 100% を超えるまでに約 3,300 件（真の回収率 130% の場合。年に約 1,700 件なので約 2 年）かかる。ドローダウンは各期間で最大 {dd_max/10000:.1f} 万円（100 円/点）、最長 {dd_run} レース連続マイナス。</li>
</ul></div>

<h2>8. 馬ごとの期待値との差（上積み）の特徴量</h2>
<p>既存の G2 は『結果そのものの率』で、馬の力を引いていない。そこで、馬ごとに『それまでの平均』を期待値にして今回の結果との差を各出走に付け、関係者ごとに as-of の縮約平均を取る（騎手・調教師・騎手×調教師・騎手×条件・直近 180 日・前走騎手との差・転厩直後）。<code>src/pipeline/features/tp59_uplift.py</code>、先読み検査: 結果をその日以降で入れ替えても値が変わらない（最大差 0）。V5 の OOF への一次選別（LOYO）:</p>
{t_scr}
<h2>9. 関係者の『モデルに対する残差』</h2>
<p>V5 の OOF（評価年より前の年だけで学習した予測）に対する、関係者の勝ち・複勝の残差の as-of 縮約平均（オッズは使わない）。上の表の <code>RS_*</code>。騎手・調教師どちらも単独で勝率 −0.00014、両方で <b>−0.00026（区間 −0.00041〜−0.00012）・連対 −0.00026・複勝 −0.00043</b> と、一次選別（勝率の区間上限 &lt; 0、効果 ≥ 0.0002、連対・複勝が悪化しない）を通る。第 7 節の『市場に対する残差が正』が効くことと同じ現象（モデルが騎手・調教師の癖を取り切れていない）の可能性がある。</p>
{lg_rs}
<p class="muted">本番に使うなら、過去のレースごとに『その時点のモデル』の予測が要る（2020〜2025 年は expanding の OOF、2026 年以降は保存済みの予測 predictions_v3 と backfill）。実装は予測の履歴ストアが前提で、v6 以降の設計事項。</p>

<h2>10. 結論</h2>
<ul>
<li>騎手・調教師の腕は勝率・複勝率として強く再現するが、<b>単勝・複勝の回収率としては再現せず</b>、100% を超えると言える関係者・組み合わせ・過去回収率の規則は無い（{rules['n_rules']} 規則 p = {rc['p_value']:.3f}、関係者の全数調査で有意 0）。</li>
<li>市場もモデルも騎手・調教師の強さを織り込んでおり、人気の騎手・調教師は平均 5〜7pt 買われすぎ（2026 年は逆向きで安定しない）。</li>
<li>特徴量としての既存列は V5 の gain の約 1%。馬の力を引いた上積みは効果なし。<b>V5 の残差に対する関係者の補正</b>は小さいが有意（−0.00026）で、v6 以降の候補。</li>
<li><b>稼げる可能性がある唯一の形</b>は『単勝 EV ≥ 1.5 かつ騎手・調教師がともに市場の見込みを上回る実績』（過去 5 年で 122〜147%）。ただし事後に絞った・穴馬依存・的中数本で崩れる、の弱点があり、<b>採用せず凍結して前向きに追跡</b>する。</li>
<li>複勝は、騎手・調教師の条件を重ねても 100% に届かない（EV ≥ 1.5 でも 97〜102%、2026 年は 79〜99%）。『安定して稼げる側』の根拠にはならない。</li>
</ul>

<h2>11. 再現手順</h2>
<pre>python -m src.scripts.data.tp59_data build          # 1 出走 1 行の表（2020〜2026）
python -m src.scripts.data.tp59_profile             # 再現性・信頼度・較正・条件別
python -m src.scripts.data.tp59_census              # 全数調査・人気の効果
python -m src.scripts.data.tp59_rules scan          # 528 規則
python -m src.scripts.data.tp59_rules ev            # EV への重ね掛け（100 通り）
python -m src.scripts.data.tp59_rules evfinal       # 候補の族（50 規則）の Reality Check
python -m src.scripts.data.tp59_rules evrobust      # 候補の頑健性・付け替え検定
python -m src.scripts.data.tp59_importance --bundle … --out …   # V5 の gain
python -m src.scripts.data.tp59_feat build|audit|screen|run     # 上積み・残差の特徴量
python -m src.scripts.data.tp59_track freeze|track  # 凍結した候補の前向き追跡</pre>
<p class="muted">関連: <a href="training_pipeline_decisions.html">決定ログ</a>・<a href="betting_edge_search.html">馬券で回収率 100% を超える余地の探索（tp-36）</a>・<a href="proven_methods_roi.html">回収率 100% 超の実績者の手法と検証（tp-44）</a>・<a href="ev_calibration_audit.html">期待値の水準の監査</a>・<a href="feature_gap_audit.html">特徴量のやり残しの棚卸し</a></p>"""
    return f'<!doctype html>\n<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>騎手・調教師の分析と回収率（tp-59）— keiba-vpn</title>\n<style>{CSS}</style></head>\n<body>{body}\n</body></html>\n'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    OUT.write_text(build(Path(a.inp)), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
