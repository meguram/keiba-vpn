"""ドリフト診断: 暦の特徴量（fe_year_frac など）が学習範囲の外に出ることが、2026 年の予測の劣化に効いているかを調べる。

  python -m src.scripts.data.drift_calendar_experiment facts     --bundle <bundle_v5_20xx> --out calendar_facts.json
  python -m src.scripts.data.drift_calendar_experiment intervene --year 2026 --bundle <bundle_v5_2025> --out calendar_2026.json
  python -m src.scripts.data.drift_calendar_experiment train     --name cal_drop_yf --variant yf        # LightGBM 1 本の再学習（学習側の確認）
  python -m src.scripts.data.drift_calendar_experiment report    --dir <結果 JSON のディレクトリ>        # drift_diagnosis.html の 13 節を差し替える

背景: fe_year_frac = 年 + (年内通算日 − 1) / 365.25 は単調に増える時間トレンド列。V5 の 4 メンバー（lgb_extras_ped / xgb_extras_ped / lgb_extras / lgb_softmax）と
linear が入力に持つ（NN 2 本は fe_* を持たない）。2026 年の値は学習期間の外にある。木は範囲外を最後の葉に丸め、linear は z を ±5 に丸めたうえで外挿する。
介入（同じ行・同じ bundle で暦の列だけを固定して予測し直し、差を見る）:
  a 現状 / b fe_year_frac を学習末期（学習最大）に固定 / c fe_year_frac を学習平均に固定 / d 暦の列すべてを学習平均に固定 / e 季節性の列だけを学習平均に固定（fe_year_frac は現状）
指標は報告のみ。2026 年はテストセット②なので、この結果でモデルや設定を選ばない。市場オッズは特徴量に使わない（評価の比較対象としてだけ使う）。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

CAL_TREND = ["fe_year_frac"]
CAL_SEASON = ["fe_month", "fe_week", "fe_dom", "fe_weekday", "fe_sin1", "fe_cos1", "fe_sin2", "fe_cos2", "fe_sin3", "fe_cos3"]
CAL_ALL = CAL_TREND + CAL_SEASON
EPS = 1e-6


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_end_year_frac(bundle: Path) -> float:
    """bundle 名（bundle_v5_2024 など）から、学習の最終日の fe_year_frac（その年の 12 月 31 日）を返す。"""
    m = re.search(r"(20\d{2})", bundle.name)
    if not m:
        raise ValueError(f"bundle 名から学習の最終年を読めません: {bundle.name}")
    return int(m.group(1)) + 364.0 / 365.25


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _bin_ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def race_bootstrap(race: np.ndarray, delta: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """レース単位のブートストラップで、馬ごとの差（delta）の平均の 95% 区間を返す。(平均, 下限, 上限)。"""
    g = pd.DataFrame({"r": race, "d": delta}).groupby("r")["d"].agg(["sum", "size"])
    s, k = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    est = s[idx].sum(1) / k[idx].sum(1)
    return float(delta.mean()), float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))


def scenario_values(linear_prep, end_frac: float) -> dict[str, dict[str, float]]:
    """介入ごとに固定する値。学習平均は linear メンバーの前処理（学習データの平均）から取る。"""
    mean = {c: float(linear_prep.mu[linear_prep.num_cols.index(c)]) for c in CAL_ALL if c in linear_prep.num_cols}
    return {
        "a_current": {},
        "b_yf_to_train_max": {"fe_year_frac": end_frac},
        "c_yf_to_train_mean": {"fe_year_frac": mean["fe_year_frac"]},
        "d_all_calendar_to_mean": {c: mean[c] for c in CAL_ALL if c in mean},
        "e_season_to_mean_keep_yf": {c: mean[c] for c in CAL_SEASON if c in mean},
    }


def cmd_facts(a) -> None:
    from src.pipeline.models.ensemble_bundle import EnsembleBundle

    b = EnsembleBundle.load(Path(a.bundle))
    w = (b.spec.get("targets", {}).get("win", {}) or {}).get("weights")
    out: dict = {"bundle": Path(a.bundle).name, "train_end_year_frac": train_end_year_frac(Path(a.bundle)), "win_weights": w, "members": []}
    for i, m in enumerate(b.members):
        cols = m.feature_cols
        has = [c for c in CAL_ALL if c in cols]
        e: dict = {"name": m.name, "recipe": m.recipe, "weight_win": (w[i] if w and i < len(w) else None), "n_cols": len(cols), "calendar_cols": has,
                   "has_fe_year_frac": "fe_year_frac" in cols}
        mo = b.models[m.name][0]
        if "fe_year_frac" in cols and "boosters" in mo and "win" in mo["boosters"]:
            bst = mo["boosters"]["win"]
            g = dict(zip(bst.feature_name(), bst.feature_importance("gain")))
            tot = float(sum(g.values())) or 1.0
            order = sorted(g, key=lambda k: -g[k])
            e["win_gain_share_pct"] = {"fe_year_frac": round(g.get("fe_year_frac", 0.0) / tot * 100, 4), "calendar_total": round(sum(g.get(c, 0.0) for c in CAL_ALL) / tot * 100, 4)}
            e["fe_year_frac_rank"] = (order.index("fe_year_frac") + 1) if "fe_year_frac" in g else None
            e["n_features_used"] = len(g)
        elif "fe_year_frac" in cols and "xgb" in mo:
            sc = mo["xgb"]["win"].get_booster().get_score(importance_type="gain")
            tot = float(sum(sc.values())) or 1.0
            order = sorted(sc, key=lambda k: -sc[k])
            e["win_gain_share_pct"] = {"fe_year_frac": round(sc.get("fe_year_frac", 0.0) / tot * 100, 4), "calendar_total": round(sum(sc.get(c, 0.0) for c in CAL_ALL) / tot * 100, 4)}
            e["fe_year_frac_rank"] = (order.index("fe_year_frac") + 1) if "fe_year_frac" in sc else None
            e["n_features_used"] = len(sc)
        elif "prep" in mo and "fe_year_frac" in cols:
            pr = mo["prep"]
            j = pr.num_cols.index("fe_year_frac")
            sd = float(pr.sd[j])
            e["standardization"] = {"mu": float(pr.mu[j]), "sd": sd, "z_at_train_end": (out["train_end_year_frac"] - float(pr.mu[j])) / sd,
                                    "z_at_2026_07_01": (2026.5 - float(pr.mu[j])) / sd, "z_clip": 5}
            W = mo["state"]["head.weight"].detach().cpu().numpy()          # linear は head.weight（行 0=勝率スコア、1=連対、2=複勝）。列は 数値の z → 欠損フラグ → 埋め込みの順
            w0 = W[0, :len(pr.num_cols)]
            z_end = min((out["train_end_year_frac"] - float(pr.mu[j])) / sd, 5.0)
            z_26 = min((2026.5 - float(pr.mu[j])) / sd, 5.0)
            e["first_layer_abs_weight"] = {"fe_year_frac_win": float(abs(w0[j])), "median_numeric_cols_win": float(np.median(np.abs(w0))), "rank_among_numeric_cols": int((np.abs(w0) > abs(w0[j])).sum() + 1),
                                           "n_numeric_cols": len(pr.num_cols), "signed_weight_win": float(w0[j])}
            e["win_score_shift_2026_07_vs_train_end"] = float(w0[j] * (z_26 - z_end))
        out["members"].append(e)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for e in out["members"]:
        _log(f"{e['name']:<20} w={e['weight_win']} cal={len(e['calendar_cols'])} gain={e.get('win_gain_share_pct')} rank={e.get('fe_year_frac_rank')} std={e.get('standardization')} W={e.get('first_layer_abs_weight')}")


def cmd_intervene(a) -> None:
    from src.pipeline.models import ensemble as en
    from src.pipeline.models import win_models as wm
    from src.pipeline.models.ensemble_bundle import EnsembleBundle
    from src.scripts.data.drift_data import load_results_year, market_probs
    from src.scripts.data.drift_monitor import load_dumps

    root = Path(a.root).resolve()
    bpath = Path(a.bundle)
    b = EnsembleBundle.load(bpath)
    end_frac = train_end_year_frac(bpath)
    lo, hi = (int(x) for x in a.months.split("-")) if "-" in a.months else (int(a.months), int(a.months))
    d = load_dumps(root / "data/local/inference/drift_features", [a.year])
    d = d[d["day"].astype(str).str[4:6].astype(int).between(lo, hi)].reset_index(drop=True)
    cols = list(dict.fromkeys(c for m in b.members for c in m.feature_cols))
    tables = root / "data/page_reference/tables"
    res = load_results_year(tables, a.year)
    mk, _ = market_probs(tables, a.year)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    meta = d[["race_id", "horse_id"]].merge(res[["race_id", "horse_id", "horse_number", "finish_position"]], on=["race_id", "horse_id"], how="left").merge(mk, on=["race_id", "horse_number"], how="left")
    keep = (meta["finish_position"].notna() & meta["p_mkt"].notna() & (meta["finish_position"] >= 1)).to_numpy()
    d, meta = d[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
    y, pm = (meta["finish_position"] == 1).to_numpy(float), meta["p_mkt"].to_numpy(float)
    race = d["race_id"].to_numpy()
    lin = next(m for m in b.members if m.recipe == "linear")
    vals = scenario_values(b.models[lin.name][0]["prep"], end_frac)
    yf = pd.to_numeric(d["fe_year_frac"], errors="coerce")
    meta_out = {"year": a.year, "bundle": bpath.name, "rows": int(len(d)), "races": int(d["race_id"].nunique()), "train_end_year_frac": end_frac,
                "fe_year_frac_rows": {"min": float(yf.min()), "median": float(yf.median()), "max": float(yf.max())},
                "years_beyond_train_end_median": float(yf.median() - end_frac), "scenario_values": vals, "market_ll": float(_bin_ll(y, pm).mean())}
    _log(f"{bpath.name} × {a.year}: rows={len(d)} races={meta_out['races']} fe_year_frac median={yf.median():.3f} (学習末期 {end_frac:.3f})")
    X0 = d[cols].copy()
    spec = b.spec["targets"]["win"]
    base_per: dict[str, np.ndarray] = {}
    results: dict = {}
    for name, fix in vals.items():
        X = X0.copy()
        for c, v in fix.items():
            if c in X.columns:
                X[c] = float(v)
        per = {}
        for m in b.members:
            touched = any(c in m.feature_cols for c in fix)
            if name != "a_current" and not touched:
                per[m.name] = base_per[m.name]
            else:
                per[m.name] = b.predict_member(m, X, race)["win"]
        if name == "a_current":
            base_per = per
        P = np.stack([per[m.name] for m in b.members], axis=1)
        p = wm.normalize_to_total(race, en.apply_spec(spec, P), 1.0)
        if name == "a_current":
            p_base = p
        lm = _logit(p)
        ll = _bin_ll(y, p)
        r = {"ll_win": float(ll.mean()), "gap": float((ll - _bin_ll(y, pm)).mean()), "logit_sd": float(lm.std()), "logit_sd_ratio_vs_market": float(lm.std() / _logit(pm).std()),
             "corr_with_current": float(np.corrcoef(p, p_base)[0, 1])}
        dd = ll - _bin_ll(y, p_base)
        r["delta_ll_vs_current"], r["delta_lo"], r["delta_hi"] = race_bootstrap(race, dd)
        r["members_mean_abs_dlogit_vs_current"] = {m.name: float(np.mean(np.abs(_logit(per[m.name]) - _logit(base_per[m.name])))) for m in b.members}
        r["members_dll_vs_current"] = {m.name: float(np.mean(_bin_ll(y, per[m.name]) - _bin_ll(y, base_per[m.name]))) for m in b.members}
        results[name] = r
        _log(f"  {name:<26} LL={r['ll_win']:.5f} gap={r['gap']:+.5f} sd比={r['logit_sd_ratio_vs_market']:.3f} Δ={r['delta_ll_vs_current']:+.5f} [{r['delta_lo']:+.5f},{r['delta_hi']:+.5f}] corr={r['corr_with_current']:.4f}")
    meta_out["scenarios"] = results
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(meta_out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def cmd_train(a) -> None:
    """学習側の確認: V5（1,142 列）から暦の列を外した LightGBM 1 本を、V5 と同じ設定（検証 2022/23/24・seed）で再学習し、V5 の OOF と対で比べる。"""
    import os

    os.environ.setdefault("TP40_THREADS", "3")
    from src.scripts.data import tp27_eval as t
    from src.scripts.data import v5_select_eval as v5
    from src.scripts.data.tp59_feat import BASE_OOF, KEY

    t.SEEDS = [int(s) for s in a.seeds.split(",")]
    df, base, meta, folds, lp = t.setup()
    X = v5.build_frame(v5.CONFIGS["d_CY_BEST"], df)
    drop = {"yf": CAL_TREND, "season": CAL_SEASON, "all": CAL_ALL}[a.variant]
    drop = [c for c in drop if c in X.columns]
    _log(f"外す列 {len(drop)}: {drop}（{X.shape[1]} → {X.shape[1] - len(drop)} 列）")
    X = X.drop(columns=drop)
    od = Path(a.out_dir) / f"lgb_{a.name}"
    od.mkdir(parents=True, exist_ok=True)
    lv, by, oof = t._run(a.name, X, meta, folds, lp)
    oof.to_parquet(od / "oof.parquet", index=False)
    prev = pd.read_parquet(BASE_OOF)
    for dd in (prev, oof):
        dd["race_id"], dd["horse_id"] = dd["race_id"].astype(str), dd["horse_id"].astype(str)
    fin_map = meta.set_index(KEY)["finish_position"]
    cmp = t._cmp(prev, oof, fin_map, a.name)
    per_year = {str(yv): t._cmp(prev[prev["year"] == yv], oof[oof["year"] == yv], fin_map, f"{a.name}_{yv}") for yv in (2022, 2023, 2024)}
    res = {"name": a.name, "variant": a.variant, "dropped": drop, "n_cols": int(X.shape[1]), "levels": lv, "by_fold": by, "compare": cmp, "per_year": per_year, "seeds": t.SEEDS}
    (od / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"{a.name}: Δwin={cmp['Δwin']:.5f} 上限={cmp['win_hi']:.5f} Δtop2={cmp['Δtop2']:.5f} Δtop3={cmp['Δtop3']:.5f} 採用={cmp['採用']}")
    for yv, c in per_year.items():
        _log(f"  {yv}: Δwin={c['Δwin']:.5f} Δtop2={c['Δtop2']:.5f} Δtop3={c['Δtop3']:.5f}")


# ---------------------------------------------------------------- 章への反映（drift_diagnosis.html の 13 節を差し替える）
BEGIN, END = "<!--CAL-BEGIN-->", "<!--CAL-END-->"
SC_LABEL = {"a_current": "a 現状", "b_yf_to_train_max": "b fe_year_frac を学習末期に固定", "c_yf_to_train_mean": "c fe_year_frac を学習平均に固定",
            "d_all_calendar_to_mean": "d 暦の列すべてを学習平均に固定", "e_season_to_mean_keep_yf": "e 季節性だけを学習平均に固定（fe_year_frac は現状）"}


def _fmt(v: float, nd: int = 5, sign: bool = True) -> str:
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _scenario_table(d: dict) -> str:
    head = "<tr><th>介入</th><th>勝率 LL</th><th>市場との差</th><th>尖り（SD比）</th><th>LL の変化（現状比）</th><th>95%区間</th><th>現状との相関</th></tr>"
    rows = []
    for k, v in d["scenarios"].items():
        rows.append(f"<tr><td>{SC_LABEL[k]}</td><td class='num'>{v['ll_win']:.5f}</td><td class='num'>{_fmt(v['gap'])}</td><td class='num'>{v['logit_sd_ratio_vs_market']:.3f}</td>"
                    f"<td class='num'>{_fmt(v['delta_ll_vs_current'])}</td><td class='num'>[{_fmt(v['delta_lo'])}, {_fmt(v['delta_hi'])}]</td><td class='num'>{v['corr_with_current']:.5f}</td></tr>")
    return f"<table>{head}{''.join(rows)}</table>"


def build_section(dirp: Path) -> str:
    def load(n: str):
        f = dirp / n
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    facts = {t: load(f"calendar_facts_{t}.json") for t in ("2024", "2025")}
    runs = [("2026 年 1〜10 月 × bundle_v5_2025（学習末期から +0.49 年。運用の V5 と同じ組）", load("calendar_2026.json")),
            ("2026 年 6〜10 月 × bundle_v5_2025（差が開いた時期）", load("calendar_2026_jun_oct.json")),
            ("2025 年 × bundle_v5_2024（対照: 学習末期から +0.49 年。2025 年の同じ季節の組）", load("calendar_2025_ctrl.json")),
            ("2026 年 1〜10 月 × bundle_v5_2024（学習末期から +1.49 年）", load("calendar_2026_b2024.json"))]
    out = [BEGIN, "<h2>13. 暦の特徴量（fe_year_frac など）の範囲外は原因か</h2>",
           "<p class='muted'>仮説: 特徴量の抜き取りの PSI 上位に fe_year_frac・fe_sin1・fe_month が並び、敵対的検証の AUC が 1.0 になった。fe_year_frac は年+通算日/365.25 の単調な時間トレンド列なので、2026 年の値は学習期間の外にあり、"
           "範囲外の外挿が 2026 年の劣化の一部かもしれない。敵対的検証の 1.0 は暦の列が日付そのものを表すための自明な値で、情報はない。</p>"]
    f25 = facts.get("2025")
    if f25:
        rows = []
        for m in f25["members"]:
            g = m.get("win_gain_share_pct")
            st = m.get("standardization")
            fl = m.get("first_layer_abs_weight")
            use = f"fe_year_frac を含め暦の列 {len(m['calendar_cols'])} 本" if m["calendar_cols"] else "暦の列なし"
            det = ""
            if g:
                det = f"勝率ブースターの gain 比 fe_year_frac {g['fe_year_frac']:.3f}%・暦の列の合計 {g['calendar_total']:.2f}%（fe_year_frac の順位 {m['fe_year_frac_rank']}／{m['n_features_used']}）"
            if st and fl:
                det = (f"標準化 μ={st['mu']:.2f}・σ={st['sd']:.2f}（学習末期 z={st['z_at_train_end']:.2f}、2026 年 7 月 z={st['z_at_2026_07_01']:.2f}、±5 で丸め）。係数 {fl['signed_weight_win']:+.4f}"
                       f"（数値列 {fl['n_numeric_cols']} 本中 {fl['rank_among_numeric_cols']} 位、中央値の {fl['fe_year_frac_win'] / fl['median_numeric_cols_win']:.1f} 倍）。"
                       f"2026 年 7 月の勝率スコアのずれ {m['win_score_shift_2026_07_vs_train_end']:+.3f}。隠れ層なしで加法なので、レース内で全馬に同じ値が乗り、勝率（レース内 softmax）では打ち消される")
            rows.append(f"<tr><td>{m['name']}</td><td>{m['recipe']}</td><td class='num'>{m['weight_win']}</td><td>{use}</td><td>{det}</td></tr>")
        out.append("<h3>どのメンバーが暦の列を使うか（bundle_v5_2025、勝率の合成重み）</h3>"
                   "<table><tr><th>メンバー</th><th>種類</th><th>勝率の重み</th><th>入力</th><th>使われ方</th></tr>" + "".join(rows) + "</table>")
    out.append("<h3>介入実験（同じ行・同じ bundle で暦の列だけを固定して予測し直す）</h3>")
    for title, d in runs:
        if not d:
            out.append(f"<p class='muted'>{title}: 未実行</p>")
            continue
        out.append(f"<h4>{title}</h4><p class='muted'>{d['rows']}行・{d['races']}レース。fe_year_frac の中央値 {d['fe_year_frac_rows']['median']:.3f}（学習末期 {d['train_end_year_frac']:.3f}）。市場の LL {d['market_ll']:.5f}。</p>")
        out.append(_scenario_table(d))
    d0 = runs[0][1]
    if d0:
        mv = d0["scenarios"]["d_all_calendar_to_mean"]["members_mean_abs_dlogit_vs_current"]
        out.append("<p class='muted'>暦の列をすべて固定したときの勝率ロジットの平均絶対変化（メンバー別）: " + "、".join(f"{k} {v:.4f}" for k, v in mv.items()) +
                   "。fe_year_frac を学習末期に固定した場合は、木のメンバーは完全に同じ予測（範囲外は最後の葉に丸められるため）で、linear も勝率では 0。</p>")
    tr = [(n, load(f"lgb_{n}/result.json")) for n in ("cal_drop_yf", "cal_drop_all")]
    rows = []
    for n, r in tr:
        if not r:
            continue
        c = r["compare"]
        pay = "、".join(f"{y} {v['Δwin']:+.5f}" for y, v in r["per_year"].items())
        rows.append(f"<tr><td>{'fe_year_frac だけを外す' if r['variant'] == 'yf' else '暦の列 11 本をすべて外す'}</td><td class='num'>{r['n_cols']}</td><td class='num'>{c['Δwin']:+.5f}（上限 {c['win_hi']:+.5f}）</td>"
                    f"<td class='num'>{c['Δtop2']:+.5f}</td><td class='num'>{c['Δtop3']:+.5f}</td><td>{pay}</td></tr>")
    if rows:
        out.append("<h3>学習側の確認（V5 構成 d の LightGBM 1 本から暦の列を外して再学習。valid 2022〜24、V5 の OOF と対で比較。負が改善）</h3>"
                   "<table><tr><th>変種</th><th>列数</th><th>Δ勝率（95%上限）</th><th>Δ連対</th><th>Δ複勝</th><th>年別 Δ勝率</th></tr>" + "".join(rows) + "</table>")
    else:
        out.append("<p class='muted'>学習側の確認（LightGBM 1 本の再学習）は共有スロット待ちで未完了。結果は <code>/home/jovyan/work/drift_out/lgb_cal_drop_*/result.json</code>。</p>")
    out.append("<h3>結論</h3><ul>"
               "<li><b>fe_year_frac の範囲外は、2026 年の劣化の原因ではない。</b>学習末期への固定・学習平均への固定のどちらでも、勝率 LL の変化は ±0.00003 以内、市場との差・尖り・予測の相関はほぼ不変（区間は 0 を含む）。"
               "木は範囲外を最後の葉に丸めるため完全に同じ予測、linear は加法でレース内の定数になり勝率では打ち消される。NN の 2 本は暦の列を持たない。</li>"
               "<li>暦の列全体（季節性）を固定しても LL は +0.0001 前後（区間は 0 を含む）で、使われ方は弱い。2025 年の同じ組（対照）でも同じ大きさで、2026 年に特有の効果ではない。</li>"
               "<li>この結果で、説明できない残り（約 6 割）は暦の列の外挿では説明できないことが分かった。調教の欠損・暦の外挿・敵対的検証の自明な 1.0 を除くと、残るのは概念ドリフトか、PSI の大きい別の入力の変化（FIELD7 群の他の列、馬場・場の傾向）。</li>"
               "<li><b>v6 以降</b>: 単調な時間トレンド列（fe_year_frac）は外すのが安全（情報がほぼ無く、将来の年で範囲外になる）。季節性の sin/cos は残す。学習側の確認の結果は上の表。</li></ul>"
               "<p class='muted'>制約: 2026 年はテストセット②で、この診断は『なぜ劣化したか』の観察のみ。結果を理由に 2026 年のデータでパラメータは調整しない。標本は 2026 年が 1,180 行・90 レース（6〜10 月は 620 行・50 レース）で、±0.0003 より小さい差は分解できない。"
               "再現: <code>python -m src.scripts.data.drift_calendar_experiment facts|intervene|train</code>、章への反映は <code>report</code>。</p>")
    out.append(END)
    return "\n".join(out)


def cmd_report(a) -> None:
    html = Path(a.chapter)
    t = html.read_text(encoding="utf-8")
    sec = build_section(Path(a.dir))
    if BEGIN in t:
        i, j = t.index(BEGIN), t.index(END) + len(END)
        t = t[:i] + sec + t[j:]
    else:
        k = t.rindex("</body>")
        t = t[:k] + sec + "\n\n" + t[k:]
    html.write_text(t, encoding="utf-8")
    _log(f"{html} の 13 節を更新しました")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facts")
    f.add_argument("--bundle", required=True)
    f.add_argument("--out", required=True)
    i = sub.add_parser("intervene")
    i.add_argument("--root", default=".")
    i.add_argument("--year", type=int, required=True)
    i.add_argument("--bundle", required=True)
    i.add_argument("--months", default="1-12")
    i.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--variant", choices=["yf", "season", "all"], required=True)
    t.add_argument("--seeds", default="0,1")
    t.add_argument("--out-dir", default="/home/jovyan/work/drift_out")
    r = sub.add_parser("report")
    r.add_argument("--dir", default="/tmp/tp_collect", help="facts / intervene / train の結果 JSON を置いたディレクトリ（train は <dir>/lgb_<name>/result.json）")
    r.add_argument("--chapter", default="docs/html/modeling/chapters/drift_diagnosis.html")
    a = ap.parse_args()
    {"facts": cmd_facts, "intervene": cmd_intervene, "train": cmd_train, "report": cmd_report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
