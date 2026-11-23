"""◎ からの馬券のパターンの成績表（docs/html/operations/final/bet_patterns_from_honmei.html）を、記録の JSON から作る（数字を手で写さない）。

入力（src/research/bet_patterns/bet_patterns_run.py が書く）:
  config/bet_patterns/selection.json（訓練だけで選んだ規則・試した数・交差検証・reality check・想定オッズの換算）
  docs/operations/retrain_trials/bet_patterns.json（選んだ規則の訓練・評価・2025・2026 の成績）
使い方: python -m src.scripts.docs.build_bet_patterns
費用 0 円（手元のファイルだけ）。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scripts.docs.build_latest_strategy_tables import CSS, esc, num, pct, rec, rng, roi_cell, row, table

ROOT = Path(__file__).resolve().parents[3]
SEL = ROOT / "config/bet_patterns/selection.json"
RES = ROOT / "docs/operations/retrain_trials/bet_patterns.json"
DST = ROOT / "docs/html/operations/final/bet_patterns_from_honmei.html"
SEL_EX = ROOT / "config/bet_patterns/selection_extra.json"           # 後から足した組み方（参考の行）
RES_EX = ROOT / "docs/operations/retrain_trials/bet_patterns_extra.json"
TITLE = "◎ からの馬券のパターンの成績"
PNAME = {"train": "訓練 2019〜22（選んだ期間）", "eval": "評価 2023〜24（確かめ）", "test": "2025（結果だけ）", "main2026": "2026 主 01/05〜10/12（結果だけ）",
         "pre2026": "2026 10/17〜11/20 本物の T-60（参考）"}
SNAME = {"a2": "設定 (i) A2 が参戦したレース", "all": "設定 (ii) 未勝利・新馬の全レース"}
SCORE_JP = {"p2": "連対率", "p3": "複勝率"}
STAKE_JP = {"equal": "均等", "payout": "払戻をそろえる（想定オッズに反比例）", "kelly": "Kelly の 1/4（資金 1 万円・1 点 500 円まで）"}


def struct_names(sel: dict) -> dict:
    return {s["id"]: s["name"] for s in sel["grid"]["structs"]}


def shape_txt(s: dict, kind: str, sh) -> str:
    n2, n3 = sh
    if kind in ("tan", "fuku"):
        return "相手なし"
    if kind in ("trio_form", "t1_form") and n2 != n3:
        return f"2 列目 上位 {n2} 頭・3 列目 上位 {n3} 頭"
    if kind.endswith("_box"):
        return f"◎＋上位 {n3} 頭のボックス"
    return f"相手 上位 {n3} 頭"


def spec_txt(spec: dict, kinds: dict) -> str:
    st = kinds[spec["s"]]
    parts = [shape_txt(spec, st["kind"], spec["sh"])]
    if spec.get("f"):
        pmin, rmin, dmin, omin, omax = spec["f"]
        sc = SCORE_JP.get(st.get("score"), "確率")
        c = [f"{sc}の順"]
        if pmin:
            c.append(f"{sc} ≥ {pmin:g}")
        if rmin:
            c.append(f"1 位との比 ≥ {rmin:g}")
        if dmin is not None:
            c.append(f"偏差値 ≥ {dmin:g}")
        if omin and omin > 1:
            c.append(f"想定単勝 ≥ {omin:g} 倍")
        if omax is not None:
            c.append(f"想定単勝 ≤ {omax:g} 倍")
        parts.append("候補: " + "・".join(c))
    if spec.get("ev") is not None:
        parts.append(f"組の期待値 ≥ {spec['ev']:g}")
    stake = spec.get("stake", "equal")
    if stake != "equal":
        parts.append("配分: " + STAKE_JP[stake])
    sk = spec.get("skip") or {}
    skp = []
    if sk.get("cap") is not None:
        skp.append(f"{sk['cap']} 点超")
    if sk.get("synth") is not None:
        skp.append(f"合成オッズ < {sk['synth']:g}")
    if sk.get("rev") is not None:
        skp.append(f"レースの期待値 < {sk['rev']:g}")
    if skp:
        parts.append("見送り: " + "・".join(skp))
    return "／".join(parts)


def pv_txt(p) -> str:
    return "－" if p is None else f"{p:.3f}"


def period_cells(m: dict | None) -> list[str]:
    if not m:
        return ["－"] * 7
    return [f'{num(m["bet_races"], 0)}／{num(m["races"])}', num(m.get("points_avg"), 1), rng(m.get("hit_rate"), m.get("hit_lo"), m.get("hit_hi")),
            roi_cell(m.get("roi"), m.get("roi_lo"), m.get("roi_hi")), pct(m.get("roi_ex_top5")), rec(m.get("honmei"), 1), diff_cell(m)]


DCOLS = ["期間", "参戦／対象レース", "平均点数", "的中率（95% 区間）", "回収率（95% 区間）", "上位 5 件を除く", "◎ 1-2-3-着外", "素の単勝との差（95% 区間）"]
SCOLS = ["組み方", "選んだ条件（訓練だけで選択）", "訓練", "1 年抜き", "評価 2023〜24", "評価: 素の単勝との差", "2025", "2026 主", "2026 本物の T-60", "評価の参戦・点数"]
EXTRA = {"base_tan": "基準: 素の A2（◎ の単勝 1 点・均等）", "base_fuku": "基準: ◎ の複勝 1 点・均等"}


def diff_cell(m: dict | None) -> str:
    if not m or m.get("diff_roi") is None:
        return "－"
    v, lo, hi = m["diff_roi"], m["diff_lo"], m["diff_hi"]
    if v == 0 and lo == 0 and hi == 0:
        return "（基準）"
    c = "ok" if lo > 0 else ("bad" if hi < 0 else "")
    return f'<span class="{c}">{v * 100:+.1f}pt（{lo * 100:+.0f}〜{hi * 100:+.0f}）</span>'


def key_name(k: str, names: dict) -> str:
    if k in EXTRA:
        return EXTRA[k]
    if k.startswith("pair:"):
        a, b = k[5:].split("+")
        return f"{names[a]} ＋ {names[b]}"
    if k.endswith("_cap20"):
        return names[k[: -len("_cap20")]] + "（20 点超は見送り・副）"
    return names[k]


def build() -> str:
    from src.research.bet_patterns.bet_patterns_run import _final_specs

    sel = json.loads(SEL.read_text(encoding="utf-8"))
    res = json.loads(RES.read_text(encoding="utf-8"))
    if res["selection_hash"] != sel["hash"]:
        raise SystemExit("評価の結果が今の selection.json と違う版のものです")
    names = struct_names(sel)
    kinds = {s["id"]: s for s in sel["grid"]["structs"]}
    P = res["periods"]
    reg = res.get("prereg") or {}
    body = [f"<h1>{TITLE}</h1>",
            f'<p class="sub">HTML の作成 {esc(datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes").replace("T", " ")[:16])}（JST）。選んだ版 {esc(sel["hash"])}。数字は記録から生成（<code>src/scripts/docs/build_bet_patterns.py</code>）。'
            f'A2（第 2 回の買い方・凍結 {esc(sel["frozen_r2_hash"])}）の ◎ を固定して、ほかの券種・組み方で買ったときの成績。</p>']
    # ---- 結論
    lines = []
    for s in ("a2", "all"):
        st = sel["settings"][s]
        rc = st["rc"]["_all"]
        pk = reg.get("primary", {}).get(s)
        g = lambda nm, k: (P.get(nm) or {}).get(s, {}).get("rules", {}).get(k) or {}
        ev, vd = g("eval", pk), (P.get("eval") or {}).get(s, {}).get("primary", {}).get("verdict", "－")
        cvb = st["cv"]["best"]
        cvt = st["cv"]["per"].get("tan", {})
        lines.append(f"<li><b>{SNAME[s]}</b>: 事前登録の主の規則「{esc(key_name(pk, names))}」は評価 2023〜24 で <b>{esc(vd)}</b>。"
                     f"回収率 {rng(ev.get('roi'), ev.get('roi_lo'), ev.get('roi_hi'))}、素の単勝 {pct(ev.get('base_roi'))} との差 {diff_cell(ev)}。"
                     f"選び方ごとの 1 年抜きの回収率 {pct(cvb.get('roi_cv'))}（単勝の組み方だけで選んだ場合 {pct(cvt.get('roi_cv'))}）。"
                     f"段 A の全規則の reality check の p 値 {pv_txt(rc['p_value'])}。</li>")
    n_tot = sel["settings"]["a2"]["n_tried"]
    body.append('<div class="box"><b>結論（数字は下の表から）</b><ul>' + "".join(lines) +
                f"<li>試した数（1 設定あたり）: 段 A {num(n_tot['A'])}・段 B {num(n_tot['B'])}・段 C {num(n_tot['C'])}（重なりを含む。下の注）。"
                "訓練の値は選んだ期間の値なので上に偏る。判断は事前登録の主の規則の評価の差と、1 年抜きの値で行う。ほかの行の評価の値は参考（評価を見て主を替えない）。</li>"
                "<li>どの期間の値も楽観に偏る。予測の p は確定オッズの offset を含み、T-60 の再現も確定オッズから作るので、期待値で絞る規則ほど偏りが大きい（下の注）。本物の T-60 の列は偏りの大きさを見る参考。</li>"
                "</ul></div>")
    # ---- 設定ごとの一覧
    for s in ("a2", "all"):
        st = sel["settings"][s]
        cnt = st["counts"]
        specs = _final_specs(st)
        body.append(f'<h2 id="{s}">{SNAME[s]}</h2>')
        body.append(f'<p class="sub">訓練で設定に当てはまったレース: {num(cnt["in_setting"] / cnt["k"], 0)}（5 版の平均）・◎ の候補なしで見送り {num(cnt["no_honmei"] / cnt["k"], 1)} レース。'
                    "回収率の色は 100% 以上が緑、90〜100% が黄。区間はレース単位のブートストラップ（1,000 回。T-60 の再現の期間は、同じレースの 5 版に同じ重み）。"
                    "差の区間は同じレース・同じ重みで取った対の差で、下限が 0 より上なら緑。「参戦／対象レース」の対象は、5 版のどれかで設定に当てはまり ◎ が置けたレースの数（上の版の平均とは別の数え方）。</p>")
        rows = []
        keys = [k for k in (P.get("eval") or {}).get(s, {}).get("rules", {}) if k in specs]   # 上限の効かない副の版（重複）は載せない
        for k in keys:
            cv = st["cv"]["per"].get(k, {})
            g = lambda nm: (P.get(nm) or {}).get(s, {}).get("rules", {}).get(k) or {}
            ev = g("eval")
            lab = esc(key_name(k, names)) + (' <span class="b ok">主</span>' if k == reg.get("primary", {}).get(s) else "")
            rows.append(row(lab, [esc(" ＋ ".join(spec_txt(x, kinds) for x in specs[k])), pct(g("train").get("roi")),
                                  pct(cv.get("roi_cv")) if cv else "－",
                                  roi_cell(ev.get("roi"), ev.get("roi_lo"), ev.get("roi_hi")), diff_cell(ev),
                                  roi_cell(g("test").get("roi"), g("test").get("roi_lo"), g("test").get("roi_hi")),
                                  roi_cell(g("main2026").get("roi"), g("main2026").get("roi_lo"), g("main2026").get("roi_hi")),
                                  roi_cell(g("pre2026").get("roi"), g("pre2026").get("roi_lo"), g("pre2026").get("roi_hi")),
                                  f'{num(ev.get("bet_races"))}R・{num(ev.get("points_avg"), 1)} 点' if ev else "－"]))
        body.append(table(rows, SCOLS))
        body.append("<h3>組み方ごとの期間の成績</h3>")
        for k in keys:
            drows = [row(esc(PNAME[nm]), period_cells((P.get(nm) or {}).get(s, {}).get("rules", {}).get(k))) for nm in PNAME if nm in P]
            body.append(f"<details><summary>{esc(key_name(k, names))}</summary><div class=\"dbody\">{table(drows, DCOLS)}</div></details>")
    # ---- 追加の組み方（参考）
    if SEL_EX.exists() and RES_EX.exists():
        ex = json.loads(SEL_EX.read_text(encoding="utf-8"))
        rx = json.loads(RES_EX.read_text(encoding="utf-8"))
        if rx["selection_extra_hash"] != ex["hash"]:
            raise SystemExit("追加の評価の結果が今の selection_extra.json と違う版のものです")
        xnames = {s["id"]: s["name"] for s in ex["structs"]}
        xkinds = {s["id"]: s for s in ex["structs"]}
        PX = rx["periods"]
        body.append('<h2 id="extra">追加の組み方（参考の行）</h2>')
        body.append(f'<p class="sub">主の選択・事前登録の後に、統括役の最終チェックで足した組み方（{esc("・".join(xnames.values()))}）。'
                    "同じ手順（訓練だけで段 A・B、1 年抜き、reality check）で選び、評価・2025・2026 には参考の行として 1 回だけ当てた（台帳は eval_only）。主の規則の合否は変えない。</p>")
        for s in ("a2", "all"):
            st = ex["settings"][s]
            rows = []
            for k, v in st["per"].items():
                if not v:
                    rows.append(row(esc(xnames[k]), ["各年 100 レース以上の規則なし"] + ["－"] * 9))
                    continue
                g = lambda nm: (PX.get(nm) or {}).get(s, {}).get("rules", {}).get(k) or {}
                ev = g("eval")
                rc = st["rc"].get(k, {})
                rows.append(row(esc(xnames[k]), [esc(spec_txt(v["final"], xkinds)), pct(g("train").get("roi")),
                                                 pct(st["cv"]["per"].get(k, {}).get("roi_cv")), pv_txt(rc.get("p_value")),
                                                 roi_cell(ev.get("roi"), ev.get("roi_lo"), ev.get("roi_hi")), diff_cell(ev),
                                                 roi_cell(g("test").get("roi"), g("test").get("roi_lo"), g("test").get("roi_hi")),
                                                 roi_cell(g("main2026").get("roi"), g("main2026").get("roi_lo"), g("main2026").get("roi_hi")),
                                                 roi_cell(g("pre2026").get("roi"), g("pre2026").get("roi_lo"), g("pre2026").get("roi_hi")),
                                                 f'{num(ev.get("bet_races"))}R・{num(ev.get("points_avg"), 1)} 点' if ev else "－"]))
            n = st["n_tried"]
            body.append(f"<h3>{SNAME[s]}（試した数: 段 A {num(n['A'])}・段 B {num(n['B'])}）</h3>")
            body.append(table(rows, ["組み方", "選んだ条件（訓練だけで選択）", "訓練", "1 年抜き", "reality check の p 値", "評価 2023〜24",
                                     "評価: 素の単勝との差", "2025", "2026 主", "2026 本物の T-60", "評価の参戦・点数"]))
            for k, v in st["per"].items():
                if v:
                    drows = [row(esc(PNAME[nm]), period_cells((PX.get(nm) or {}).get(s, {}).get("rules", {}).get(k))) for nm in PNAME if nm in PX]
                    body.append(f"<details><summary>{esc(xnames[k])}</summary><div class=\"dbody\">{table(drows, DCOLS)}</div></details>")
        body.append('<p class="sub">3連単の ◎ 2 着付けのフォーメーションは「1 着列（2 列目の頭数）→ ◎ → 3 着列（3 列目の頭数）」、◎ 3 着付けは「1 着列 → 2 着列 → ◎」。'
                    "列は同じ候補の並び（複勝率の順）の上位から取る。枠連のボックスは、◎ と相手の枠の組すべて（同じ枠に 2 頭いればゾロ目も）。"
                    "どれも上の楽観・割引きの限界の注記がそのまま当てはまる。</p>")
    # ---- 3連複フォーメーション
    body.append('<h2 id="trio_f">3連複 フォーメーション（必須の形）</h2>')
    fl = []
    for s in ("a2", "all"):
        v = sel["settings"][s]["per"].get("trio_f")
        if not v:
            fl.append(f"<li>{SNAME[s]}: 各年 100 レース以上の条件を満たす規則が無かった。</li>")
            continue
        ev = (P.get("eval") or {}).get(s, {}).get("rules", {}).get("trio_f") or {}
        folds = sel["settings"][s]["cv"]["per"].get("trio_f", {}).get("folds", [])
        fl.append(f"<li>{SNAME[s]}: {esc(spec_txt(v['final'], kinds))}。評価の平均点数 {num(ev.get('points_avg'), 1)}・回収率 {pct(ev.get('roi'))}。"
                  f"1 年抜きで選ばれた条件は {len({json.dumps(f.get('spec'), sort_keys=True) for f in folds})} 通り（4 回のうち）。</li>")
    body.append("<ul>" + "".join(fl) + "</ul>")
    body.append('<p class="sub">2 列目・3 列目は同じ候補の並び（予測の複勝率の順）の上位から取る（2 列目は 3 列目の上位）。点数は 2 列目 a 頭・3 列目 b 頭で a×(b−a)＋a(a−1)/2。'
                "選ばれた 3連複フォーメーションは設定 (i) が 2 列目 1 頭（3連複 2 頭軸流しと同じ規則で、表の値も同じ）、設定 (ii) が 2 列目 2 頭。2 列目 3 頭・3 列目 10 頭なら 24 点、3連単（◎ 1 着）は a×(b−1) で 27 点になり、UI の既定の上限（1 レース 2,000 円）を超える。上限は段 B の見送りの条件（10・20 点）として試し、上限が効く最良（3連単フォーメーション）には、20 点の上限を足した副の版（選び直しではない）も載せた。</p>")
    # ---- 割引き
    body.append('<h2 id="multiple">選び過ぎの割引き</h2>')
    for s in ("a2", "all"):
        st = sel["settings"][s]
        rcrows = []
        for k, v in st["rc"].items():
            if k == "_all":
                continue
            cv = st["cv"]["per"].get(k, {})
            rcrows.append(row(esc(names[k]), [pct(1 + v["best_roi_minus_1"]), f'{v["p_value"]:.3f}', pct(cv.get("roi_cv")), num(cv.get("stable"))]))
        a = st["rc"]["_all"]
        rcrows.append(row("<b>全部</b>", [pct(1 + a["best_roi_minus_1"]) if a["best_roi_minus_1"] is not None else "－",
                                          "－" if a["p_value"] is None else f'{a["p_value"]:.3f}', pct(st["cv"]["best"].get("roi_cv")), num(st["cv"]["best"].get("stable"))]))
        body.append(f"<h3>{SNAME[s]}</h3>")
        body.append(table(rcrows, ["組み方", "段 A の最高の回収率（訓練）", "reality check の p 値", "1 年抜きの回収率", "1 年抜きで選ばれた条件の数（4 回中）"]))
    body.append('<p class="sub">reality check: White（2000）の方法。各規則の回収率から訓練の値を引いて再中心化し、ブートストラップの最大値の分布と、実際の最大の「回収率 − 100%」を比べる（各年 100 レース以上の規則だけ）。'
                "1 年抜き: 訓練の 4 年のうち 3 年で段 A〜C をやり直して選び、抜いた 1 年の回収率を合計で見る（選び方ごとの外での値）。</p>")
    # ---- 想定オッズ
    om = sel["odds_map"]
    orow = []
    for bt, v in om["coefs"].items():
        o = om["original_on_train"].get(bt)
        orow.append(row(esc(bt), [num(v["n"]), f'{v["rmse_log"]:.3f}', f'{o["rmse_log"]:.3f}' if o else "（元の係数なし）", f'{o["bias_log"]:+.3f}' if o else "－"]))
    body.append('<h2 id="odds">想定オッズの換算</h2>')
    body.append(f'<p class="sub">{esc(om["note"])}。元の係数（odds_pred の poly2）は 2022〜25 の払戻で当てはめてあり評価・2025 が入るので、訓練 2019〜22 だけで当てはめ直した。</p>')
    body.append(table(orow, ["券種", "当たりの組の数（訓練）", "当てはめ直しの誤差（log の RMSE）", "元の係数の誤差（同じ組）", "元の係数の平均のずれ"]))
    ch = []
    for s in ("a2", "all"):
        w = sel["settings"][s].get("with_original_odds_map") or {}
        diff = [names[k] for k, v in w.items() if not v.get("same_final")]
        ch.append(f"<li>{SNAME[s]}: 元の係数で選び直すと、条件が変わる組み方は {len(diff)} / {len(w)}（{esc('・'.join(diff)) or 'なし'}）。</li>")
    body.append("<ul>" + "".join(ch) + "</ul>")
    # ---- 作り方
    g = sel["grid"]
    body.append(f"""<h2 id="method">作り方・注意</h2>
<ul>
<li>◎・対象: 凍結 <code>{esc(sel["frozen_r2"])}</code> の ◎ の規則（2 番人気以下・勝率 5% 以上で r 最大、1 番人気は r が高いときだけ）。設定 (i) は凍結の見送り（未勝利・新馬かつ q_r_max の分位 3 以上）、設定 (ii) は q_r_max の見送りだけ外した。</li>
<li>相手の候補の閾値の格子: 予測の確率（2 頭の券種は連対率、3 頭の券種とワイドは複勝率）の下限 {esc(g["P_MIN"])}、1 位との比 {esc(g["RATIO_MIN"])}、偏差値 {esc(g["DEV_MIN"])}、想定単勝オッズの下限 {esc(g["ODDS_MIN"])}・上限 {esc(g["ODDS_MAX"])}、組の期待値 {esc(g["EV_MIN"])}。候補を確率の順に並べ、上から必要な頭数を取る。</li>
<li>段 B: 配分 {esc(g["STAKES"])}、見送り（点数の上限 {esc(g["CAPS"])}・合成オッズの下限 {esc(g["SYNTH_MIN"])}・レースの期待値の下限 {esc(g["REV_MIN"])}）。ボックスは配分・期待値の条件を付けない（◎ を含まない組の確率を作っていない）。</li>
<li>段 C: 段 B の上位 {g["top_pairs"]} の組み方の 2 つずつと、決めておいた組（単勝＋ワイド、馬連＋3連複）。</li>
<li>物差し: min(上位 5 件を除いた回収率, 年ごとの回収率の最小)。訓練の各年 {g["MIN_BETS_PER_YEAR"]} レース以上の参戦が条件。</li>
<li>期待値・合成オッズの想定オッズは、単勝は T-60 の再現、連系は上の換算（市場の確率から Harville）。モデルの組の確率は MB60（<code>fit_blend_distribution</code>）。予測の p は確定オッズの offset を含むので楽観。</li>
<li>試していないもの: ◎ を含まない組（◎ 固定の依頼の外）、WIN5 などレースをまたぐ券。枠連のボックス・3連単の ◎ 2 着／3 着付けのフォーメーションは、後から足した（上の「追加の組み方」）。</li>
<li>2025・2026 年は結果を出すだけで、選び直しに使っていない（見た回数の台帳に eval_only で申告）。評価 2023〜24 は confirm。1 回目は行を作る途中（精算の前）で止まり、data.py を直して（払戻の組の馬が出走表に無いレースを数えて外す）流し直した。値を見たのは 2 回目だけ。流し直しは台帳の行でも止める。ただし A2 の ◎ と見送りは、A2 の作成のときに訓練 2019〜22 で選ばれ、評価 2023〜24 でも確かめ済み。ここで新しく問えるのは「同じ ◎ で、素の単勝より上か」だけなので、合否は対の差で決めた。</li>
<li><b>楽観の向き</b>: 予測の p（確定オッズの offset を含む）と、確定オッズから逆向きに作った T-60 の再現の両方に、締切までの情報が残る。組の期待値（モデルの組の確率 × 想定オッズ）・合成オッズ・Kelly・相手の想定単勝の範囲は、このオッズと p に強く頼るので、素の単勝より楽観が大きくなりうる（水準だけでなく、単勝との差も上に偏る）。期待値 1.3 などは較正した期待値ではなく、換算の上の目安。</li>
<li><b>割引きの限界</b>: reality check は段 A の規則の素の回収率だけが対象で、段 B（配分・見送り）・段 C（組み合わせ）・A2 自体の探索を含まない。分散の大きい 3連単に引きずられ、標準化もしていない。A2 の ◎ を同じ訓練で選んだので、p 値は実際より小さく出る。1 年抜きも、想定オッズの換算（訓練 4 年で当てはめ）・A2 の ◎・分位の区切りは抜いた年を含んだまま。組み方ごとの 1 年抜きの値も「その組み方を選ぶ」という選択を含む。手順全体の外の年での値は「全体の最良」の 1 年抜きだけ。</li>
<li><b>物差しの限界</b>: 上位 5 件を除く回収率は件数が固定で、払戻の裾が重い 3連単には甘い。年 100 レースの条件も、的中の少ない 3連単には少ない。</li>
<li><b>試した数の重なり</b>: 3連複 2 頭軸流し（trio_2）は 3連複フォーメーションの 2 列目 1 頭と同じ規則で、3連単 1 着流しも 3連単フォーメーションと一部重なる。試した数はこの重なりを含む。</li>
<li><b>想定オッズの換算</b>: 当たった組の払戻だけで log を当てはめるので、exp に戻すと平均を低めに見る（2〜8%）。確定の単勝からの P_hat で当てはめ、T-60 の再現からの P_hat に当てる（雑音の分だけ、期待値で絞ると高く見えた組を拾う）。全クラスで当てはめた。表の「当てはめ直しの誤差」は当てはめたデータの中の値で、元の係数との比べは公平ではない。元の係数は 2022〜25 の払戻で当てはめたもので、比べるためだけに使った（採否に使っていない）。</li>
<li><b>精算の扱い</b>: 枠連は払戻のあるレースだけを買える扱い（9 頭以上で払戻の無いレースは訓練で 0 件。8 頭以下でも取消の前に発売されたレースは払戻がある）。払戻の組の馬が出走表に無いレース（勝ち馬のオッズの欠けで取消とみなされたもの）は買わずに数えた（期間ごとの counts の pay_mismatch_races。pay_mismatch は版の数）。合成オッズは配分によらず、買った点を等しく扱った値。</li>
<li>2026 主の予測ファイルは A2 の事前登録の後に差し替わっている（2026 年の部分）。単勝 ◎ の行が A2 の記録と合うかは <code>bet_patterns.json</code> の <code>a2_reproduction</code>。</li>
</ul>""")
    rep = res.get("a2_reproduction") or {}
    mm = {nm: len(P[nm]["a2"]["counts"].get("pay_mismatch_races") or []) for nm in P}
    body.append("<p class=\"sub\">A2 の記録との照合（素の単勝 ◎・設定 (i)。行の作りの検算）: " +
                "・".join(f"{PNAME[k].split('（')[0]} {'一致' if v['same'] else '不一致'}（{pct(v['roi'])}・{num(v['bet_races'], 0)}R / 記録 {pct(v['roi_record'])}・{num(v['bet_races_record'])}R）"
                         for k, v in rep.items()) +
                ("。不一致は、払戻の組の馬が出走表に無いレースを外した分（" + "・".join(f"{PNAME[k].split('（')[0]} {n} レース" for k, n in mm.items() if n) +
                 "。A2 の道具はそのレースを外れとして数える）。" if any(mm.values()) else "。") +
                "本物の T-60 の期間は発走前オッズのある日だけで、設定 (i) は十数レースしかなく、区間は意味を持たない。</p>")
    return (f'<!doctype html>\n<html lang="ja">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{TITLE}</title>\n<style>{CSS}</style>\n</head>\n<body>\n<main>{''.join(body)}</main>\n</body>\n</html>\n")


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(build(), encoding="utf-8")
    print("書いた", DST)


if __name__ == "__main__":
    main()
