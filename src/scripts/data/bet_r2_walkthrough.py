"""第 2 回で凍結した買い方（frozen_20261117_r2.json）が、予測の値からどう買い目になるかを、2025 年のサンプルのレースで見せる HTML。

使い方:
    python -m src.scripts.data.bet_r2_walkthrough [--frozen config/bet_cycles/frozen_20261117_r2.json] [--seed 20251118]

- 数字はすべて、凍結の設定と 2025 年の test の予測（凍結に書いた予測のファイル）から、ここで作る。手で写さない。
- 規則は変えない。新しい評価は流さない: 凍結した買い方を test 2025 にもう一度当てるのは、記録（bet_test2025_r2_test_<hash>.json）と
  同じ値になることの確かめ（再現）だけ。値が違えば（予測のファイルが差し替わったなど）、止まって何も書かない。
- 2026 年の予測・着順は読まない（発走前のオッズの再現のずれの元として、凍結の作法どおり 2026 年の T-60 と確定のオッズの比だけを使う）。
- サンプルは、区分（参戦して的中・参戦して外れ・見送りの理由ごと）の中から、種を固定した乱数で選ぶ。当たったものだけを選ばない。
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "config/bet_cycles/frozen_20261117_r2.json"
OUT = ROOT / "docs/html/operations/bet_r2_walkthrough.html"
TRIALS = ROOT / "docs/operations/retrain_trials"

N_BET, N_SKIP = 10, 9          # サンプルの参戦・見送りの数。的中の数と見送りの理由ごとの数は、2025 年の実際の割合から生成で決める
CATS = ("hit", "miss", "skip_r", "skip_none", "skip_class", "skip_field", "skip_other")


def cat_labels(classes: list[str], p_min: float, top_share: float) -> dict:
    return {"hit": "参戦して的中", "miss": "参戦して外れ",
            "skip_r": f"見送り: r の最大が区切りより小さい（下位 {100 - top_share * 100:.0f}%）",
            "skip_none": f"見送り: ◎ の候補（2 番人気以下・勝率 {p_min * 100:.0f}% 以上）がいない",
            "skip_class": f"見送り: {'・'.join(classes)}でない（r は足りる）", "skip_field": "見送り: 頭数が 5 頭未満", "skip_other": "見送り: そのほか"}


def allocate(counts: dict, n: int) -> dict:
    """n 件を、区分の実際の件数に比例して配る（0 件の区分は 0。件数がある区分は最低 1、端数は大きい順）。"""
    keys = [c for c in counts if counts[c] > 0]
    if not keys:
        return {}
    tot = sum(counts[c] for c in keys)
    raw = {c: n * counts[c] / tot for c in keys}
    out = {c: max(1, int(np.floor(raw[c]))) for c in keys}
    for c in sorted(keys, key=lambda c: raw[c] - np.floor(raw[c]), reverse=True):
        if sum(out.values()) >= n:
            break
        out[c] += 1
    while sum(out.values()) > n:
        c = max(out, key=lambda c: out[c] - raw[c])
        out[c] -= 1
    return out


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def pct(x, d: int = 1) -> str:
    return "－" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def num(x, d: int = 1) -> str:
    return "－" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def table(head: list[str], rows: list[list], left: int = 1, cls: list[str] | None = None) -> str:
    th = "".join(f'<th class="l">{esc(h)}</th>' if i < left else f"<th>{esc(h)}</th>" for i, h in enumerate(head))
    body = ""
    for k, r in enumerate(rows):
        c = f' class="{cls[k]}"' if cls and cls[k] else ""
        body += f"<tr{c}>" + "".join(f'<td class="l">{esc(x)}</td>' if i < left else f"<td>{esc(x)}</td>" for i, x in enumerate(r)) + "</tr>"
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def classify(cost: float, ret: float, cls: str, field: int, q_r_max: int, reason: str, classes: list[str], q_min: int, min_field: int = 5) -> str:
    """1 レースの区分。見送りは、凍結の買い方の判定の順（bet_cycles._skip: 頭数 → when の r の分位 → in のクラス、その後 ◎ の候補）で
    最初に当たった理由。r の分位 0 は r の最大が無い（◎ の候補が 1 頭もいない）。"""
    if cost > 0:
        return "hit" if ret > 0 else "miss"
    if field < min_field:
        return "skip_field"
    if q_r_max == 0:
        return "skip_none"
    if q_r_max < q_min:
        return "skip_r"
    if str(cls) not in classes:
        return "skip_class"
    return "skip_none" if "候補なし" in str(reason) else "skip_other"


def horse_status(i: int, H: int | None, pw: float, pop: float, r: float, p_min: float, odds: float = 1.0) -> str:
    """凍結の ◎ の決め方（bet_cycles.pick_honmei の r_max: 2 番人気以下・勝率 p_min 以上・r とオッズが有限で r が最大）での、その馬の扱い。"""
    if H is not None and i == H:
        return "◎（候補の中で r が最大）"
    if pop == 1:
        return "対象外: 1 番人気（例外のしきい値なし）"
    if pop >= 99 or not np.isfinite(r) or not np.isfinite(odds):
        return "対象外: オッズ・r なし"
    if pw < p_min:
        return f"対象外: 勝率 {p_min * 100:.0f}% 未満"
    return "候補（r が ◎ より小さい）" if H is not None else "候補（このレースは見送り）"


_TOKS: list = []


def build(frozen: Path, seed: int | None = None) -> tuple[str, dict]:
    """台帳の行を、途中で落ちたときも閉じる（例外は error、Ctrl-C などは aborted。started のまま残さない）。"""
    try:
        return _build_inner(frozen, seed)
    except BaseException as exc:
        for t in _TOKS:
            t.__exit__(type(exc), exc, None)
        raise


def _build_inner(frozen: Path, seed: int | None = None) -> tuple[str, dict]:
    from src.config.data_paths import TABLES_DIR
    from src.scripts.data import bet_cycles as bc

    body = json.loads(frozen.read_text(encoding="utf-8"))
    spec, cuts, h = body["spec"], body["qcuts"], body["hash"]
    hon = spec["honmei"]
    base = hon.get("base") or {}
    p_min = float(base.get("p_min", 0.05))
    classes = [str(x) for x in spec["skip"]["in"]["c_class"]]
    q_min, q_hi = spec["skip"]["when"]["q_r_max"]
    q_min = int(q_min)
    if q_min < 2 or q_hi is not None or set(spec["skip"]) - {"in", "when"} or set(spec["skip"]["when"]) != {"q_r_max"}:
        raise SystemExit("この説明のページは、r の分位の下限（2 以上・上限なし）とクラスだけで絞る買い方を前提にしています")
    if abs(p_min - 0.05) > 1e-12:
        raise SystemExit("r の最大の特徴（bet_cycles._r_features）は勝率 5% 以上で計る。◎ の勝率の下限が違う買い方は前提外")
    r_cut = float(cuts["r_max"][q_min - 2])                      # q_r_max ≥ q_min ⇔ r_max ≥ 区切りの (q_min−1) 本目
    top_share = (5 - q_min + 1) / 5                              # 5 分位のうち参戦する分（上位の割合）
    tr_years = body["inputs"]["periods"]["train"]["years"]
    labels = cat_labels(classes, p_min, top_share)
    rec_path = TRIALS / f"bet_test2025_r2_test_{h}.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rm = rec["periods"]["test"]["metrics"]

    from src.ops import data_ledger as L

    tok = L.guard(tool="src/scripts/data/bet_r2_walkthrough.py", family="bet_cycles", version=h, period=("2025-01-01", "2025-12-31"),
                  purpose="reproduce", frozen_paths=[frozen], reason="実例のページ（test 2025 の記録の再現を確かめてから例を並べる。判断に使わない）")

    _TOKS.append(tok)
    per, _ = bc.prepare_periods(body["inputs"], ("test",), cuts=cuts)
    p = per["test"]
    res = bc.run_period(p, spec, detail=True)
    m = bc.metrics_run(res, p, p.W)
    if m.get("bet_races") != rm["bet_races"] or abs(m["roi"] - rm["roi"]) > 1e-9 or abs(m["hit_rate"] - rm["hit_rate"]) > 1e-9:
        raise SystemExit(f"凍結の test の記録と値が合いません（予測のファイルが差し替わった可能性）: 記録 {rm['bet_races']}・{rm['roi']:.4f} / "
                         f"いま {m.get('bet_races')}・{m.get('roi', float('nan')):.4f}。HTML は書きません。")
    k, n0 = p.k, len(p.base.races)
    now_h = bc._input_hashes(body["inputs"])
    changed = [Path(x).name for x, v in body["pred_sha256"].items() if v is not None and now_h.get(x) not in (None, v)]
    seed = seed if seed is not None else int(h[:8], 16)          # 種は凍結の hash から機械的に決める（結果を見て選び直さない）
    recs = res["records"]
    cost, ret = res["cost"], res["ret"]
    ft = p.feats.reset_index(drop=True)

    # 版 0（T-60 の再現の 1 通り目）でのレースの区分
    cats = [classify(cost[i], ret[i], ft.at[i, "c_class"], int(ft.at[i, "field"]), int(ft.at[i, "q_r_max"]), recs.at[i, "reason"], classes, q_min)
            for i in range(n0)]
    cats = np.array(cats)
    cnt = {c: int((cats == c).sum()) for c in np.unique(cats)}
    hit0 = cnt.get("hit", 0) / max(cnt.get("hit", 0) + cnt.get("miss", 0), 1)
    plan = {"hit": int(round(N_BET * hit0))}
    plan["miss"] = N_BET - plan["hit"]
    plan.update(allocate({c: cnt.get(c, 0) for c in CATS if c.startswith("skip")}, N_SKIP))
    def draw(sd: int) -> list:
        rng = np.random.default_rng(sd)
        out_ = []
        for key in CATS:
            idx = np.flatnonzero(cats == key)
            n = plan.get(key, 0)
            if len(idx) == 0 or n == 0:
                continue
            ch = rng.choice(idx, size=min(n, len(idx)), replace=False)
            out_ += [(int(i), key, labels[key]) for i in ch]
        return out_

    picks = draw(seed)
    alt = [draw(seed + d) for d in (1, 2, 3, 4, 5)]               # 種を変えたときの、サンプルの回収率の幅（選び直しの疑いを避けるため並べる）
    alt_roi = [sum(ret[i] for i, _, _ in a_) / max(sum(cost[i] for i, _, _ in a_), 1) for a_ in alt]
    picks.sort(key=lambda x: (p.base.races[x[0]].date, p.base.races[x[0]].race_id))

    yr = int(str(p.base.races[0].race_id)[:4])
    flat = pd.read_parquet(Path(TABLES_DIR) / str(yr) / "race_result_flat.parquet",
                           columns=["race_id", "horse_number", "horse_name", "venue", "round", "race_name", "race_class", "surface", "distance"])
    flat["race_id"] = flat["race_id"].astype(str)
    flat["horse_number"] = pd.to_numeric(flat["horse_number"], errors="coerce")
    names = {(r_, int(n_)): nm for r_, n_, nm in flat[["race_id", "horse_number", "horse_name"]].itertuples(index=False) if pd.notna(n_)}
    info = flat.drop_duplicates("race_id").set_index("race_id")

    races_html, sample_rows = [], []
    rfin = bc.run_strategy(p.base.races, p.base.frecs, spec, detail=True)      # 確定オッズで判定した場合（楽観。比べるためだけ）
    fin_H = [None if pd.isna(x) else int(x) for x in rfin["records"]["H_num"]]
    for no, (i, key, label) in enumerate(picks, 1):
        rs, rf = p.races[i], p.base.races[i]                    # 版 0（T-60 の再現）と確定
        H = None if recs.at[i, "H_rank"] < 0 else int(recs.at[i, "H_rank"])
        rows, cls = [], []
        rv = np.where(np.isfinite(rs.r), rs.r, -np.inf) if rs.r is not None else np.full(len(rs.hn), -np.inf)
        order = np.argsort(-rv, kind="stable")                  # r の大きい順
        for rk, j in enumerate(order, 1):
            st = horse_status(int(j), H, float(rs.pw[j]), float(rs.pop[j]), float(rs.r[j]) if rs.r is not None else np.nan, p_min, float(rs.odds[j]))
            rows.append([rk, int(rs.hn[j]), names.get((rs.race_id, int(rs.hn[j])), ""), num(rs.r[j], 2) if rs.r is not None else "－", pct(rs.pw[j]),
                         pct(rs.q[j]) if rs.q is not None else "－", int(rs.pop[j]) if rs.pop[j] < 99 else "－", num(rs.odds[j]),
                         num(rf.odds[j]), int(rf.pop[j]) if rf.pop[j] < 99 else "－", int(rf.fin[j]) if rf.fin[j] < 99 else "－", st])
            cls.append("hon" if st.startswith("◎") else ("cand" if st.startswith("候補") else ""))
        inf = info.loc[rs.race_id] if rs.race_id in info.index else None
        title = (f"{esc(rs.date)}　{esc(inf['venue']) if inf is not None else ''} {int(inf['round']) if inf is not None and pd.notna(inf['round']) else ''}R　"
                 f"{esc(inf['race_name']) if inf is not None else ''}（{esc(ft.at[i, 'c_class'])}・{esc(inf['surface']) if inf is not None else ''}"
                 f"{int(inf['distance']) if inf is not None and pd.notna(inf['distance']) else ''}m・{int(ft.at[i, 'field'])} 頭）")
        rmax = float(ft.at[i, "r_max"]) if np.isfinite(ft.at[i, "r_max"]) else np.nan
        r_ok = int(ft.at[i, "q_r_max"]) >= q_min
        c_ok = str(ft.at[i, "c_class"]) in classes
        steps = [
            f"① r の最大（2 番人気以下・勝率 {p_min * 100:.0f}% 以上の中）: {num(rmax, 3)}（5 分位の {int(ft.at[i, 'q_r_max'])}）。区切り {r_cut:.3f} 以上で次へ → "
            + ("<b class=\"good\">足りる</b>" if r_ok else ("<b class=\"bad\">見送り</b>（候補がいない）" if not np.isfinite(rmax) else "<b class=\"bad\">見送り</b>（足りない）")),
            ("" if r_ok else "（参考）") + f"② クラス: {esc(ft.at[i, 'c_class'])} → "
            + ("<b class=\"good\">参戦の対象</b>" if c_ok else f"<b class=\"bad\">{'見送り' if r_ok else '対象外'}</b>（{'・'.join(classes)}でない）"),
        ]
        if key in ("hit", "miss", "skip_none"):
            steps.append("③ ◎: " + (f"{int(rs.hn[H])} 番 {esc(names.get((rs.race_id, int(rs.hn[H])), ''))}（T-60 の再現で {int(rs.pop[H])} 番人気・"
                                     f"勝率 {pct(rs.pw[H])}・r {num(rs.r[H], 2)}）" if H is not None else "条件に合う馬がいない → 見送り"))
        if H is not None:
            bets = json.loads(recs.at[i, "bets"])
            payout = int(ret[i])
            steps.append(f"④ 買い目: 単勝 {int(rs.hn[H])} 番 {bets[0]['yen']} 円 → 着順 {int(rf.fin[H]) if rf.fin[H] < 99 else '－'} 着・"
                         + (f"<b class=\"good\">払戻 {payout:,} 円</b>（確定の払戻で精算）" if payout > 0 else "<b class=\"bad\">払戻 0 円</b>"))
        vb = [int(cost[i + v * n0] > 0) for v in range(k)]
        vh = [int(ret[i + v * n0] > 0) for v in range(k)]
        vH = ["見送り" if pd.isna(recs.at[i + v * n0, "H_num"]) else f"{int(recs.at[i + v * n0, 'H_num'])} 番" for v in range(k)]
        steps.append(f"<span class=\"muted\">T-60 の再現 {k} 通りのうち、参戦 {sum(vb)} 通り・的中 {sum(vh)} 通り。各通りの ◎: {' / '.join(vH)}。"
                     f"楽観（確定オッズで判定）では: {'参戦（◎ ' + str(fin_H[i]) + ' 番）' if fin_H[i] else '見送り'}。</span>")
        sample_rows.append([no, labels[key], f"{rs.date} {rs.race_id}", str(ft.at[i, "c_class"]), num(rmax, 2),
                            f"{int(rs.hn[H])} 番" if H is not None else "－", (int(rf.fin[H]) if rf.fin[H] < 99 else "－") if H is not None else "－",
                            f"{int(cost[i]):,}", f"{int(ret[i]):,}"])
        races_html.append(
            f'<section class="race"><h3><span class="tag t-{key}">{esc(label)}</span> {no}. {title}</h3>'
            f'<p class="muted">race_id {esc(rs.race_id)}</p><ol class="flow">' + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"
            + table(["r の順", "馬番", "馬名", "r", "モデル p", "市場 q", "人気(T-60再現)", "単勝(T-60再現)", "単勝(確定)", "人気(確定)", "着順", "◎ の判定"], rows, left=3, cls=cls)
            + "</section>")

    # 2025 年全体（再現 5 通りを並べた値）: ◎ の人気・r の大きさ・クラスごと
    bet = cost > 0
    hidx = res["hrank"]
    hr = np.array([p.races[j].r[hidx[j]] if hidx[j] >= 0 and p.races[j].r is not None else np.nan for j in range(len(p.races))])
    hp = res["hpop"]

    def grp(mask) -> list:
        c, r_ = cost[mask], ret[mask]
        nb = int((c > 0).sum())
        if nb == 0:
            return ["0", "－", "－", "－"]
        return [f"{nb / k:,.0f}", pct(float((r_[c > 0] > 0).mean())), pct(float(r_.sum() / c.sum())), f"{r_.sum() / k:,.0f}"]

    pop_bins = [(2, 3, "2〜3 番人気"), (4, 5, "4〜5 番人気"), (6, 8, "6〜8 番人気"), (9, 99, "9 番人気以下")]
    pop_rows = [[lab] + grp(bet & (hp >= lo) & (hp <= hi)) for lo, hi, lab in pop_bins]
    r_edges = [r_cut, 0.7, 1.0, 1.5, np.inf]
    r_rows = [[f"{lo:.2f} 以上 {hi:.2f} 未満" if np.isfinite(hi) else f"{lo:.2f} 以上"] + grp(bet & (hr >= lo) & (hr < hi)) for lo, hi in zip(r_edges[:-1], r_edges[1:])]
    cl_all = p.feats["c_class"].astype(str).to_numpy()
    cl_rows = [[c] + grp(bet & (cl_all == c)) for c in classes]

    ex = (m.get("roi_ex_top5"),)
    sum_rows = [[labels[c], f"{cnt.get(c, 0):,}", pct(cnt.get(c, 0) / n0), str(sum(1 for x in picks if x[1] == c))] for c in CATS]
    n_hit_s = sum(1 for x in picks if x[1] == "hit")
    n_bet_s = sum(1 for x in picks if x[1] in ("hit", "miss"))
    skip_share = sum(v for c, v in cnt.items() if c.startswith("skip")) / n0
    s_cost = sum(cost[i] for i, _, _ in picks)
    s_ret = sum(ret[i] for i, _, _ in picks)
    hit_pay = ret[(cost > 0) & (ret > 0)]
    # ◎ が確定の 1 番人気だった参戦（再現で人気を落とした確定の人気馬）と、確定の判定でも参戦するか
    base_pop = np.array([p.base.races[j % n0].pop[hidx[j]] if hidx[j] >= 0 else 99 for j in range(len(p.races))])
    fin_bet = np.tile(rfin["cost"] > 0, k)
    fav_rows = [["◎ が確定の 1 番人気"] + grp(bet & (base_pop == 1)), ["◎ が確定の 2 番人気以下"] + grp(bet & (base_pop > 1))]
    both_rows = [["楽観（確定オッズで判定）でも参戦するレース"] + grp(bet & fin_bet), ["T-60 の再現でだけ参戦するレース"] + grp(bet & ~fin_bet)]
    from src.scripts.data.bet_cycles import OUT_DIR
    exf = OUT_DIR / str(body.get("tag")) / "evidence_extra.json"
    n_cands = (json.loads(exf.read_text(encoding="utf-8")).get("rc") or {}).get("n_candidates") if exf.exists() else None
    bl = rec["periods"]["test"]["baselines"]
    b0 = next(v for k_, v in bl.items() if k_.startswith("B0"))

    css = """:root{color-scheme:light dark;--bg:#f7f8fa;--panel:#fff;--text:#1d2330;--muted:#5b6578;--border:#d8dde6;--accent:#2f6fde;--good:#1d8a4e;--bad:#b3261e;--hold:#a86400;--head:#eef1f6;--hon:#fff4d6;--cand:#eef5ff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--hold:#f0b65a;--head:#1d2638;--hon:#3a3016;--cand:#1a2740}}
:root[data-theme="dark"]{--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--hold:#f0b65a;--head:#1d2638;--hon:#3a3016;--cand:#1a2740}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2em 0}
h2{font-size:1.15rem;margin:1.8em 0 .5em;border-bottom:1px solid var(--border);padding-bottom:.2em}h3{font-size:1rem;margin:0 0 .3em}
.muted{color:var(--muted);font-size:.88rem}.lead{background:var(--panel);border:1px solid var(--border);border-left:4px solid var(--accent);border-radius:6px;padding:10px 14px}
.warn{border-left-color:var(--hold)}
.pipe{display:flex;flex-direction:column;gap:0;margin:.6em 0}.pipe .box{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px 12px}
.pipe .box b{color:var(--accent)}.pipe .arrow{text-align:center;color:var(--muted);line-height:1.4}
.race{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;margin:14px 0}
.flow{margin:.3em 0 .5em;padding-left:1.2em}.flow li{margin:.15em 0}
.tag{display:inline-block;font-size:.78rem;border-radius:4px;padding:0 .5em;margin-right:.3em;border:1px solid var(--border)}
.t-hit{background:var(--good);color:var(--bg);border-color:var(--good)}.t-miss{background:var(--bad);color:var(--bg);border-color:var(--bad)}
.t-skip_r,.t-skip_class,.t-skip_none,.t-skip_field,.t-skip_other{background:var(--head);color:var(--muted)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}table{border-collapse:collapse;width:100%;font-size:.84rem;margin:.5em 0}
th,td{border:1px solid var(--border);padding:3px 6px;text-align:right;white-space:nowrap}th{background:var(--head);font-weight:600}td.l,th.l{text-align:left}
tr.hon td{background:var(--hon);font-weight:600}tr.cand td{background:var(--cand)}
code{font-size:.82rem;background:var(--head);padding:0 .3em;border-radius:3px;word-break:break-all}.good{color:var(--good)}.bad{color:var(--bad)}
ul{padding-left:1.2em}"""
    head = (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>第2回の買い方の実例</title><style>{css}</style></head><body><main>")
    o = [head, "<h1>第 2 回の買い方の実例 — 予測の値から買い目まで（2025 年のサンプル）</h1>",
         f'<p class="muted">凍結 <code>{esc(frozen.relative_to(ROOT))}</code>（hash {esc(h)}）の買い方を、2025 年の test の予測にそのまま当てた。'
         f"数字はすべて <code>src/scripts/data/bet_r2_walkthrough.py</code> が凍結の設定・予測・着順から作る。生成のたびに、test 2025 の記録"
         f"（<code>{esc(rec_path.relative_to(ROOT))}</code>）と同じ値になることを確かめている（新しい評価は流していない）。"
         + (f"凍結の後に差し替わった入力: <code>{esc(', '.join(changed))}</code>（2026 年の分の差し替え）。2025 年の値が記録と同じになることで、test の部分が変わっていないことを確かめた。" if changed else "")
         + '過程は <a href="bet_optimization_story.html">推奨馬券の最適化の過程</a>、サイクルの表は <a href="bet_cycles.html">推奨馬券の最適化サイクル</a>。</p>',
         '<div class="lead warn"><ul>'
         "<li><b>本番ではまだ使っていない。</b>2026 年の手つかずの確かめ（1 回だけ）の前の、仮の候補。</li>"
         "<li><b>2025 年は何度か見た期間。</b>第 1 回の test で 1 回、model-improve-cycles の選び方の確かめで 1 回、"
         "そして未勝利・新馬の区切りは統括役が 2022〜25 年の値を見て渡したもの。2025 年の値は、手つかずの確かめではない。</li>"
         f"<li><b>本番の再現: T-60 のオッズで判定し、確定の払戻で精算。</b>過去の年には T-60 のオッズが無いので、2026 年の T-60 と確定の比を、"
         f"人気の順位ごとに 1 レース丸ごと借りて確定オッズから逆に作る（{k} 通り）。下の実例は、その 1 通り目。2025 年全体の値は {k} 通りの平均。</li>"
         "<li><b>表の T-60 のオッズ・人気は作り物。</b>ほかのレース（2026 年）のずれを当てた値で、そのレースの実際の T-60 ではない。"
         "参戦するか、どの馬が ◎ になるかも、この乱数で変わる（各レースに 5 通りの ◎ を並べた）。</li>"
         "<li><b>p と r の偏り（向きの違う 2 つ）。</b>モデルの p は確定の時点の予測（市場の確定オッズを offset に使う）で、過去の年に発走前の予測は無い。"
         "q は再現の T-60 のオッズから作り、r = log(p/q) は q の替わり分だけ動かす。そのため (1) p に確定オッズの情報が残る（楽観。"
         "確定で人気になった馬が、再現で人気を落とすと r が上がって ◎ になりやすい）、(2) 本番では p も q と一緒に動くので、人気薄の r を下げすぎる（悲観）。"
         "どちらが大きいかは、本物の T-60 の 2026 年で確かめるまで分からない。</li></ul></div>"]

    # 1. つながりの図
    o += ["<h2>1. 予測の値から買い目までのつながり</h2>", '<div class="pipe">',
          '<div class="box"><b>入力</b>　モデルの勝率 <code>p</code>（第 2 段のモデル。市場のオッズを offset に使う）と、市場の確率 <code>q</code>'
          '（T-60 の単勝オッズの逆数を合計 1 に直したもの。本番では本物の T-60、このページの 2025 年は再現の値）</div>',
          '<div class="arrow">↓</div>',
          '<div class="box"><b>ずれ</b>　<code>r = log(p / q)</code>。r が大きいほど「市場が低く見ている」馬。</div>', '<div class="arrow">↓</div>',
          f'<div class="box"><b>レースの絞り込み</b>　クラスが {"・".join(classes)} で、かつ「2 番人気以下・勝率 {p_min * 100:.0f}% 以上の馬の r の最大」が '
          f'{r_cut:.3f} 以上（訓練 {min(tr_years)}〜{max(tr_years)} の分布で上位 {top_share * 100:.0f}%、5 分位の {q_min} 以上）のレースだけ参戦。'
          '判定の順は、頭数 5 頭未満 → r の最大 → クラス。</div>',
          '<div class="arrow">↓</div>',
          f'<div class="box"><b>◎ の選び方</b>　1 番人気を除き、勝率 {p_min * 100:.0f}% 以上の馬の中で r が最大の馬。'
          "1 番人気の例外（r が高いときだけ ◎ にする）は、訓練で決まるしきい値が無く、使わない。人気は T-60 の時点。</div>",
          '<div class="arrow">↓</div>',
          '<div class="box"><b>買い目</b>　◎ の単勝 100 円を 1 点。条件に合う馬がいなければ見送り。精算は確定の払戻。</div></div>',
          f'<p class="muted">凍結の値: <code>{esc(json.dumps({k_: v for k_, v in spec.items() if k_ != "name"}, ensure_ascii=False))}</code>。'
          f"r の最大の 5 分位の区切り（訓練で決めた値）: {', '.join(f'{x:.3f}' for x in cuts['r_max'])}。"
          "相手（partners）の設定は単勝だけなので使われない。</p>"]

    # 2. サンプル
    o += [f"<h2>2. 2025 年のサンプルのレース（{len(picks)} レース）</h2>",
          f'<p class="muted">選び方: T-60 の再現の 1 通り目でレースを区分に分け、各区分の中から乱数（numpy の default_rng）で選んだ。'
          f"種 {seed} は凍結の hash の先頭 8 桁から機械的に決めた（結果を見て選び直していない）。"
          f"区分ごとの数も生成で決めた: 参戦 {N_BET}・見送り {N_SKIP}。参戦の中の的中の数は 2025 年の的中率から、見送りは理由ごとの実際の件数に比例して配った"
          f"（件数がある理由は最低 1）。見送りは実際には全体の {pct(skip_share, 0)} だが、ここでは参戦の例を多めに並べた。</p>",
          table(["区分", "2025 年（1 通り目）", "割合", "サンプル"], sum_rows),
          f'<p class="muted">参戦の中の的中: サンプル {n_hit_s}/{n_bet_s}（{pct(n_hit_s / max(n_bet_s, 1))}）・2025 年の 1 通り目 {pct(hit0)}・{k} 通りの平均 {pct(m["hit_rate"])}。'
          f"サンプルの払戻の合計 {int(s_ret):,} 円 / 購入 {int(s_cost):,} 円（回収率 {pct(s_ret / max(s_cost, 1))}）。"
          f"2025 年全体は {pct(m['roi'])}、的中 1 件の払戻は中央値 {np.median(hit_pay):.0f} 円・四分位 {np.percentile(hit_pay, 25):.0f}〜{np.percentile(hit_pay, 75):.0f} 円。"
          f"種を 1〜5 ずらしたときのサンプルの回収率: {' / '.join(pct(x, 0) for x in alt_roi)}。10 点のサンプルの回収率は 1〜2 件の払戻で大きく揺れるので、全体の値は 3 節を見る。</p>",
          table(["#", "区分", "日付・race_id", "クラス", "r の最大", "◎", "◎ の着順", "購入", "払戻"], sample_rows, left=4),
          f'<p class="muted">各レースの表は r の大きい順。色: ◎ は黄色、◎ の候補（2 番人気以下・勝率 {p_min * 100:.0f}% 以上）は青。'
          "「対象外」は ◎ の候補に入らない理由（馬券の外れではない）。人気・q・r は T-60 の再現の 1 通り目の値。</p>"]
    o += races_html

    # 3. 全体
    o += [f"<h2>3. 2025 年全体（T-60 の再現 {k} 通りの平均）</h2>",
          table(["項目", "値"], [["全レース（予測・払戻がそろうもの）", f"{m['races']:,}"], ["参戦", f"{m['bet_races']:,}（{pct(m['bet_races'] / m['races'])}）"],
                                 ["的中率", f"{pct(m['hit_rate'])}（95% 区間 {pct(m.get('hit_lo'))}〜{pct(m.get('hit_hi'))}）"],
                                 ["回収率", f"{pct(m['roi'])}（95% 区間 {pct(m.get('roi_lo'))}〜{pct(m.get('roi_hi'))}、{k} 通りの幅 {pct(m.get('sim_roi_min'))}〜{pct(m.get('sim_roi_max'))}）"],
                                 ["上位 5 件の払戻を除いた回収率", pct(ex[0])],
                                 ["平均回収額 / 中央値回収額（1 レース 100 円）", f"{m['avg_return']:.0f} 円 / {m['median_return']:.0f} 円（的中時の中央値 {m.get('median_hit_return', 0):.0f} 円）"],
                                 ["基準: 予測 1 位の単勝を全部買う", pct(b0['roi'])],
                                 ["楽観（確定オッズで判定）", f"参戦 {rec['periods']['test']['metrics_final_odds']['bet_races']:,}・回収率 {pct(rec['periods']['test']['metrics_final_odds']['roi'])}"]]),
          "<h3>◎ の人気（T-60 の再現）ごと</h3>", table(["◎ の人気", "参戦", "的中率", "回収率", "払戻の合計（円）"], pop_rows),
          "<h3>◎ の r の大きさごと</h3>", table(["◎ の r", "参戦", "的中率", "回収率", "払戻の合計（円）"], r_rows),
          "<h3>クラスごと</h3>", table(["クラス", "参戦", "的中率", "回収率", "払戻の合計（円）"], cl_rows),
          "<h3>◎ が確定の 1 番人気だったか（p に残る確定オッズの情報の効き）</h3>", table(["◎", "参戦", "的中率", "回収率", "払戻の合計（円）"], fav_rows),
          "<h3>楽観（確定オッズで判定）でも参戦するか</h3>", table(["レース", "参戦", "的中率", "回収率", "払戻の合計（円）"], both_rows),
          '<p class="muted">参戦の数・払戻の合計は、再現の通り数で割った 1 年分の値。区分ごとの値は件数が少なく、偶然の幅が大きい（区間は全体の行だけに付けた）。'
          "95% 区間は、選んだ後の 1 本についてのレース単位のブートストラップで、多くの候補から選んだことの効き（勝者の呪い）を含まない。実際の幅はもっと広い。"
          + (f"第 2 回では、訓練で {n_cands} 本の候補を比べて選んだ。" if n_cands else "") + "</p>"]

    # 4. 注意
    o += ["<h2>4. 読むときの注意</h2>", '<div class="lead warn"><ul>',
          "<li>2025 年は何度か見た期間で、区切りの選定にも使われた。手つかずの確かめは 2026 年の全日の版（1 回だけ、統括役の合図で流す）。</li>",
          f"<li>上位 5 件の払戻を除くと {pct(ex[0])}"
          + ("で、100% を割る。" if ex[0] is not None and ex[0] < 1 else "。") + "穴の単勝なので、少数の高配当に頼る形。"
          + (f"回収率の 95% 区間（{pct(m.get('roi_lo'))}〜{pct(m.get('roi_hi'))}）は 100% をまたぐ。" if m.get("roi_lo", 1) < 1 < m.get("roi_hi", 0) else "") + "</li>",
          f"<li>判定のオッズの時点で結果が大きく変わる。楽観（確定オッズで判定）では参戦 {rec['periods']['test']['metrics_final_odds']['bet_races']:,}・"
          f"回収率 {pct(rec['periods']['test']['metrics_final_odds']['roi'])}、T-60 の再現では参戦 {m['bet_races']:,}・{pct(m['roi'])}。本物の T-60 では、さらに下がる可能性がある。</li>",
          "<li>本番ではまだ使っていない。</li></ul></div>", "</main></body></html>"]
    tok.finish(result_path=str(OUT))
    meta = {"picks": [(p.base.races[i].race_id, key) for i, key, _ in picks], "metrics": {k_: m.get(k_) for k_ in ("bet_races", "roi", "hit_rate")}}
    return "\n".join(o), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", default=str(FROZEN))
    ap.add_argument("--seed", type=int, default=None, help="既定は凍結の hash の先頭 8 桁（16 進）")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    doc, meta = build(Path(a.frozen).resolve(), a.seed)
    Path(a.out).write_text(doc, encoding="utf-8")
    print(f"書いた: {a.out}  サンプル {len(meta['picks'])} レース・再現の確かめ {meta['metrics']}")


if __name__ == "__main__":
    main()
