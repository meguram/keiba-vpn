"""血統 × 条件替わりの検定（『○○産駒の距離延長』『○○系の初ダート』など）。通説の検証の章の第 11 節。

    python -m src.scripts.data.folklore_sire run   # → <FOLKLORE_OUT>/sire_tests.parquet

粒度（血統の単位）:
  父 / 父の小系統（smartrc）/ 父の大系統（smartrc）/ 父のタイプ（smartrc: 米国型ミスプロ系など）/ 母父 / 母父の小系統（smartrc）
  系統は data/meta/sire_lines/sire_smartrc_lines.json（種牡馬 ID → smartrc の小系統・大系統・タイプ）。馬 → 父・母父は
  data/page_reference/pedigree_race_index/horse_sire_map.parquet（ID で結合）。
条件替わり: 延長・短縮（同じ面で 100m 以上）・大幅延長・大幅短縮（400m 以上）・初ダート・初芝・休み明け初戦・叩き 2 戦目・
  芝の道悪・ダートの道悪・小回り（芝）・昇級初戦・前走から内回り→外回り。

2 つの比べ方（どちらも市場の見込み p_mkt をそろえた 3 着内の差）:
  (a) 条件の中で: その血統の馬 − 同じ条件の他の血統の馬（『延長なら○○産駒を買え』。オッズのゆがみの候補）
  (b) 血統の中で: 条件に当てはまる走 − 当てはまらない走（『○○産駒は延長で良くなる』）
検定は第 1 節と同じ作り（発見 2016〜2022 → 確認 2023〜2025 の 2 段の BH、標準誤差はレース・馬・調教師の単位のクラスタ頑健の最大、
信頼度 1−max(q) で S/A/B/C）。粒度 × 比べ方ごとに別の族として補正する。父のように数が多い単位は、族が大きいぶん補正が厳しくなる。
表示用に、条件 × 粒度の中で経験ベイズで縮めた推定値も出す（標本の少ない血統ほど 0 へ寄せる）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.scripts.data.folklore_verify import OUT, TIERS, bh, diff_test  # noqa: E402

SIRE_MAP = ROOT / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
LEDGER = ROOT / "data/meta/sire_lines/sire_smartrc_lines.json"
DISC, CONF = (2016, 2022), (2023, 2025)
MIN_N1, MIN_R1, MIN_CONF = 200, 150, 60
LEVELS = {"sire": "父", "line_s": "父の小系統", "line_l": "父の大系統", "line_t": "父のタイプ", "dsire": "母父", "dline_s": "母父の小系統",
          "placebo": "偽の血統（馬ごとに無作為な 300 群）"}


def conditions(d: pd.DataFrame) -> dict:
    """条件替わり: 名前 → (資格, 当てはまる)。"""
    same = d["p_dist"].notna() & (d["p_surf"] == d["surf"])
    diff = d["dist"] - d["p_dist"]
    has = d["starts_prev"] >= 1
    return {
        "延長（100m 以上）": (same, diff >= 100),
        "短縮（100m 以上）": (same, diff <= -100),
        "大幅延長（400m 以上）": (same, diff >= 400),
        "大幅短縮（400m 以上）": (same, diff <= -400),
        "初ダート": ((d["surf"] == "ダ") & has, d["surf_n_prev"] == 0),
        "初芝": ((d["surf"] == "芝") & has, d["surf_n_prev"] == 0),
        "休み明け初戦（70 日以上）": (has & d["since_rest"].notna(), d["since_rest"] == 1),
        "叩き 2 戦目": (has & d["since_rest"].notna(), d["since_rest"] == 2),
        "芝の道悪（稍重〜不良）": (d["surf"] == "芝", d["track"].isin(["稍", "重", "不"])),
        "ダートの道悪（稍重〜不良）": (d["surf"] == "ダ", d["track"].isin(["稍", "重", "不"])),
        "小回り（芝）": (d["surf"] == "芝", d["shape"] == "小回り"),
        "昇級初戦": (d["p_cls"].notna() & has, (d["p_y1"] == 1) & (d["cls"] > d["p_cls"])),
        "内回り → 外回り": (d["io"].isin(["内", "外"]) & d["p_io"].isin(["内", "外"]), (d["p_io"] == "内") & (d["io"] == "外")),
    }


def add_blood(d: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_parquet(SIRE_MAP, columns=["horse_id", "sire_id", "sire_name", "dam_sire_id", "dam_sire_name", "sire_line_name"])
    m["horse_id"] = m["horse_id"].astype(str)
    d = d.copy()
    d["horse_id"] = d["horse_id"].astype(str)
    d = d.merge(m, on="horse_id", how="left")
    led = json.load(open(LEDGER))["sires"]

    def lk(ids, field):
        return ids.map(lambda x: (led.get(str(x)) or {}).get(field) if pd.notna(x) else None)

    d["sire"] = d["sire_name"]
    d["line_s"] = lk(d["sire_id"], "small_name").fillna(d["sire_line_name"])
    d["line_l"] = lk(d["sire_id"], "large_name")
    d["line_t"] = lk(d["sire_id"], "type_name")
    d["dsire"] = d["dam_sire_name"]
    d["dline_s"] = lk(d["dam_sire_id"], "small_name")
    # 偽の血統: 馬ごとに無作為な群（結果と無関係。誤検出の確認用）
    hid = pd.factorize(d["horse_id"])[0]
    grp = np.random.default_rng(7).integers(0, 300, hid.max() + 1)
    d["placebo"] = "群" + pd.Series(grp[hid]).astype(str).to_numpy()
    return d


def _cell_se(v: np.ndarray, key: np.ndarray, clusters: dict, nk: int):
    """key ごとの v の平均と、クラスタ頑健 SE（クラスタの単位ごとに出して最大）。"""
    n = np.bincount(key, None, nk).astype(float)
    s = np.bincount(key, v, nk)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s / n
    se = np.zeros(nk)
    for cl in clusters.values():
        nc = cl.max() + 1
        comb = key.astype(np.int64) * nc + cl
        u, inv = np.unique(comb, return_inverse=True)
        z = np.bincount(inv, v - m[key])
        gk = (u // nc).astype(np.int64)
        G = np.bincount(gk, None, nk).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.bincount(gk, z * z, nk) / (n * n) * G / np.maximum(G - 1, 1)
        se = np.fmax(se, np.sqrt(var))
    return n, m, se


def run() -> None:
    d = pd.read_pickle(OUT / "_prep.pkl")
    d = add_blood(d)
    conds = conditions(d)
    years = d["year"].to_numpy()
    r3 = d["r3"].to_numpy(float)
    y3 = d["y3"].to_numpy(float)
    pm = d["p_mkt"].to_numpy(float)
    ret = d["ret_cap"].to_numpy(float)
    clusters = {c: pd.factorize(d[c].astype(str))[0] for c in ("race_id", "horse_id", "trainer_id")}
    race_codes = pd.factorize(d["race_id"])[0]
    rows = []
    for lv in LEVELS:
        codes, labels = pd.factorize(d[lv].astype(object).where(d[lv].notna(), None))
        nk = len(labels)
        valid = codes >= 0
        for cname, (e, f) in conds.items():
            e = np.asarray(e.fillna(False), bool) & valid
            f = np.asarray(f.fillna(False), bool) & e
            for per, (y0, y1) in (("disc", DISC), ("conf", CONF)):
                py = (years >= y0) & (years <= y1)
                # (a) 条件の中で: 血統の当てはまる馬の残差の平均 − 条件全体の残差の平均
                m_in = f & py
                base = r3[m_in].mean()
                kk = codes[m_in]
                cl = {k: v[m_in] for k, v in clusters.items()}
                n1, mr, se_a = _cell_se(r3[m_in], kk, cl, nk)
                nr1 = np.bincount(np.unique(kk.astype(np.int64) * (race_codes.max() + 1) + race_codes[m_in]) // (race_codes.max() + 1), None, nk)
                _, my, _ = _cell_se(y3[m_in], kk, {"x": cl["race_id"]}, nk)
                _, mp, _ = _cell_se(pm[m_in], kk, {"x": cl["race_id"]}, nk)
                _, mret, _ = _cell_se(ret[m_in], kk, {"x": cl["race_id"]}, nk)
                # (b) 血統の中で: 条件に当てはまる走 − 当てはまらない走（資格の中）
                m_e = e & py
                ses = []
                for k, v in clusters.items():
                    out = diff_test(codes[m_e], v[m_e], f[m_e], r3[m_e], nk)
                    ses.append(out[6])
                N1b, N0b, _, _, _, db, _ = out
                se_b = np.nanmax(np.vstack(ses), axis=0)
                for i in np.flatnonzero(n1 > 0):
                    rows.append({"level": lv, "cond": cname, "key": labels[i], "period": per, "n1": int(n1[i]), "r1": int(nr1[i]),
                                 "a": float(mr[i] - base), "a_se": float(se_a[i]), "y3": float(my[i]), "p_mkt": float(mp[i]), "roi_cap": float(mret[i]),
                                 "b": float(db[i]) if N0b[i] > 0 and N1b[i] > 0 else np.nan, "b_se": float(se_b[i]), "n0b": int(N0b[i])})
            print(lv, cname, flush=True)
    t = pd.DataFrame(rows)
    t.to_parquet(OUT / "sire_tests_raw.parquet", index=False)
    judge(t)


def judge(t: pd.DataFrame) -> None:
    disc = t[t["period"] == "disc"].set_index(["level", "cond", "key"])
    conf = t[t["period"] == "conf"].set_index(["level", "cond", "key"])
    res = disc[(disc["n1"] >= MIN_N1) & (disc["r1"] >= MIN_R1)].copy()
    c = conf.reindex(res.index)
    res["c_n1"] = c["n1"].to_numpy()
    for fam in ("a", "b"):
        res[fam + "_c"] = c[fam].to_numpy()
        res[fam + "_c_se"] = c[fam + "_se"].to_numpy()
        res[fam + "_conf"] = 0.0
        res[fam + "_tier"] = "無"
        res[fam + "_eb"] = np.nan
        for lv in LEVELS:
            m = (res.index.get_level_values("level") == lv) & res[fam].notna() & (res[fam + "_se"] > 0)
            if fam == "b":
                m &= res["n0b"] >= MIN_N1
            if not m.any():
                continue
            x = res.loc[m]
            p = 2 * norm.sf(np.abs(x[fam] / x[fam + "_se"]))
            q = bh(np.nan_to_num(np.asarray(p, float), nan=1.0))
            fwd = (q < 0.10) & (x["c_n1"] >= MIN_CONF).to_numpy() & x[fam + "_c"].notna().to_numpy()
            zc = np.sign(x[fam]) * x[fam + "_c"] / x[fam + "_c_se"]
            pc = norm.sf(zc.to_numpy())
            qc = np.full(len(x), np.nan)
            if fwd.any():
                qc[fwd] = bh(pc[fwd])
            confd = 1 - np.maximum(q, np.nan_to_num(qc, nan=1.0))
            tier = np.full(len(x), "無", dtype=object)
            for name, th in TIERS[::-1]:
                tier[confd >= th] = name
            res.loc[m, fam + "_conf"] = confd
            res.loc[m, fam + "_tier"] = tier
            # 経験ベイズの縮小（条件 × 粒度の中で、平均 0 の正規分布へ）
            for cond, g in x.groupby(level="cond"):
                wv = 1 / g[fam + "_se"] ** 2
                mu = float((g[fam] * wv).sum() / wv.sum())
                tau2 = max(float(((g[fam] - mu) ** 2).mean() - (g[fam + "_se"] ** 2).mean()), 1e-8)
                res.loc[g.index, fam + "_eb"] = (mu + (g[fam] - mu) * tau2 / (tau2 + g[fam + "_se"] ** 2)).to_numpy()
    res = res.reset_index()
    res.to_parquet(OUT / "sire_tests.parquet", index=False)
    for fam in ("a", "b"):
        print(fam, res.groupby("level")[fam + "_tier"].value_counts().unstack(fill_value=0).to_string())


# ---------------------------------------------------------------- 章（第 11 節）
def sire_section(w) -> None:
    f = OUT / "sire_tests.parquet"
    if not f.exists():
        return
    import html as _h

    t = pd.read_parquet(f)
    esc = _h.escape
    order = {"S": 0, "A": 1, "B": 2, "C": 3, "無": 4}

    def tier(tr, x):
        if tr == "無" or pd.isna(tr):
            return "<span class='t0'>無</span>"
        arrow = "＋" if x > 0 else "−"
        return f"<span class='t{tr}'>{tr}</span>{arrow}"

    def pt(x):
        return "" if pd.isna(x) else f"{x * 100:+.1f}"

    def conf_txt(tr, c):
        return "" if tr == "無" or pd.isna(c) else f" <span class='muted'>{c * 100:.1f}%</span>"

    real = t[t["level"] != "placebo"]
    pl = t[t["level"] == "placebo"]
    w("<h2 id='blood'>11. 血統 × 条件替わり（○○産駒の延長・短縮・初ダート・道悪など）</h2>")
    w("<p>血統の粒度は 6 つ: <b>父</b>・<b>父の小系統</b>・<b>父の大系統</b>・<b>父のタイプ</b>（いずれも smartrc の系統。例: Lサンデー系・ミスプロ系・米国型ミスプロ系）・<b>母父</b>・<b>母父の小系統</b>。"
      "馬 → 父・母父は ID で結合（horse_sire_map）、系統は smartrc の台帳（sire_smartrc_lines.json）。条件替わりは 13 種類（延長・短縮・大幅延長・大幅短縮・初ダート・初芝・休み明け初戦・叩き 2 戦目・芝の道悪・ダートの道悪・小回り・昇級初戦・内回り→外回り）。</p>"
      "<p>比べ方は 2 つ（どちらも人気をそろえた 3 着内の差）:</p><ul>"
      "<li><b>(a) 条件の中で</b>: その血統の馬 − 同じ条件の全馬。『延長なら○○産駒を買え』。市場が血統の適性を織り込めていない＝<b>オッズのゆがみの候補</b>。</li>"
      "<li><b>(b) 血統の中で</b>: 条件に当てはまる走 − 当てはまらない走。『○○産駒は延長で良くなる（悪くなる）』。</li></ul>"
      f"<p>検定は第 1 節と同じ（発見 2016〜2022 → 確認 2023〜2025 の 2 段の BH、レース・馬・調教師の単位のクラスタ頑健の最大）。粒度 × 比べ方ごとに別の族。"
      f"標本の下限は発見期間で当てはまる馬 {MIN_N1} 走・{MIN_R1} レース、確認期間で {MIN_CONF} 走（父は多くがこれに届かず、届いた父も族が大きいぶん補正が厳しい）。"
      "『縮めた推定』は、条件 × 粒度の中で経験ベイズで縮めた値（標本の少ない血統ほど全体へ寄せる。判定には使わない）。</p>")
    # 偽の血統
    w(f"<p class='muted'>偽の血統（馬ごとに無作為な 300 群）の検定 {len(pl):,} 本は、(a) S〜C {int((pl['a_tier'] != '無').sum())} 本・(b) {int((pl['b_tier'] != '無').sum())} 本。</p>")
    # 粒度ごとの数
    w("<div class='wrap'><table><thead><tr><th>粒度</th><th>検定した血統 × 条件</th><th>(a) 条件の中で S/A/B/C（＋は人気以上・−は人気以下）</th><th>(b) 血統の中で S/A/B/C</th></tr></thead><tbody>")
    for lv, nm in LEVELS.items():
        if lv == "placebo":
            continue
        x = real[real["level"] == lv]
        ca = "・".join(f"{k}{int((x['a_tier'] == k).sum())}" for k in "SABC")
        cb = "・".join(f"{k}{int((x['b_tier'] == k).sum())}" for k in "SABC")
        w(f"<tr><td>{nm}</td><td class='num'>{len(x):,}</td><td class='num'>{ca}</td><td class='num'>{cb}</td></tr>")
    w("</tbody></table></div>")
    w("<div class='ok-tools'><label>血統で絞り込み <input type='search' id=\"blood-q\" placeholder='例: ディープインパクト・ロベルト系・米国型'/></label>"
      "<span class='muted' id=\"blood-cnt\"></span></div>")
    head = ("<div class='wrap'><table><thead><tr><th>粒度</th><th>血統</th><th>(a) 条件の中で</th><th>Δ 発見/確認(pt)</th><th>(b) 血統の中で</th><th>Δ 発見/確認(pt)</th>"
            "<th>縮めた推定 (a)/(b)</th><th>走数（発見/確認）</th><th>3 着内率 / 市場</th><th>単勝回収率（50 倍頭打ち）</th></tr></thead><tbody>")

    def row(r):
        return (f"<tr class='brow'><td>{LEVELS[r['level']]}</td><td>{esc(str(r['key']))}</td><td>{tier(r['a_tier'], r['a'])}{conf_txt(r['a_tier'], r['a_conf'])}</td>"
                f"<td class='num'>{pt(r['a'])} / {pt(r['a_c'])}</td><td>{tier(r['b_tier'], r['b'])}{conf_txt(r['b_tier'], r['b_conf'])}</td><td class='num'>{pt(r['b'])} / {pt(r['b_c'])}</td>"
                f"<td class='num'>{pt(r['a_eb'])} / {pt(r['b_eb'])}</td><td class='num'>{int(r['n1']):,} / {0 if pd.isna(r['c_n1']) else int(r['c_n1']):,}</td>"
                f"<td class='num'>{r['y3'] * 100:.1f}% / {r['p_mkt'] * 100:.1f}%</td><td class='num'>{r['roi_cap'] * 100:.0f}%</td></tr>")

    conds = list(dict.fromkeys(real["cond"]))
    for c in conds:
        x = real[real["cond"] == c].copy()
        x["o"] = np.minimum(x["a_tier"].map(order), x["b_tier"].map(order))
        sig = x[x["o"] < 4].sort_values(["o", "level"])
        na = int((x["a_tier"] != "無").sum())
        nb = int((x["b_tier"] != "無").sum())
        w(f"<details class='okh bsec'><summary>{esc(c)}: 有意 (a) {na}・(b) {nb} 本（検定 {len(x):,}）</summary>")
        if len(sig):
            w("<p class='muted' style='margin:6px 0'>有意（S〜C）の血統</p>" + head + "".join(row(r) for _, r in sig.iterrows()) + "</tbody></table></div>")
        for lv in ("sire", "line_s", "dsire"):
            y = x[x["level"] == lv].dropna(subset=["a_eb"])
            if len(y) < 6:
                continue
            top = pd.concat([y.nlargest(5, "a_eb"), y.nsmallest(5, "a_eb")])
            w(f"<details class='okc'><summary>参考: {LEVELS[lv]}の縮めた推定 (a) の上位 5・下位 5（判定ではない）</summary>" + head + "".join(row(r) for _, r in top.iterrows()) + "</tbody></table></div></details>")
        w("</details>")
    w("""<script>(function(){var q=document.getElementById('blood-q'),cnt=document.getElementById('blood-cnt');
var secs=[].slice.call(document.querySelectorAll('details.bsec'));
function apply(){var s=(q.value||'').trim().toLowerCase(),n=0;secs.forEach(function(sec){var any=false;
[].slice.call(sec.querySelectorAll('tr.brow')).forEach(function(tr){var v=!s||tr.cells[1].textContent.toLowerCase().indexOf(s)>=0;tr.classList.toggle('hide',!v);if(v&&s){any=true;n++;}});
[].slice.call(sec.querySelectorAll('details.okc')).forEach(function(c){var a=!s||[].some.call(c.querySelectorAll('tr.brow'),function(tr){return !tr.classList.contains('hide');});if(s)c.open=a;});
if(s)sec.open=any;});cnt.textContent=s?(n+' 行'):'';}
q.addEventListener('input',apply);})();</script>""")
    w("<div class='note'><b>読み方.</b> (a) で＋の S〜C は『その条件のとき、その血統は人気より来る』＝市場のゆがみの候補（第 10 節の期待値の買い方に入れる候補）。"
      "(b) で＋は『その血統は条件替わりで良くなる』だが、人気が既に反映していれば (a) は無になる。父の粒度は標本が足りないことが多く、無は『差が無い』ではなく『あると言えない』。</div>")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "judge":
        judge(pd.read_parquet(OUT / "sire_tests_raw.parquet"))
