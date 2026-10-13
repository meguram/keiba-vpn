"""馬連・ワイド・3連複の確率モデル（割引 Harville）の検証。

  synthetic … 真の分布が分かるシミュレーション（Thurstone 型: 能力 + 正規ノイズで着順を決める）で、
              Harville（勝率のみ）と、勝率+連対率+複勝率に合わせた割引 Harville の誤差を比べる。
  market    … dev ストアの実データ（既定 20260906 の 36 レース）で、(a) 市場オッズから逆算した確率との乖離（KL）、
              (b) 実際の着順に対する対数損失、を比べる。市場の確率は払戻率で割り戻した 1/オッズ。

使い方:
    python3 -m src.scripts.data.validate_exotic_probabilities synthetic
    python3 -m src.scripts.data.validate_exotic_probabilities market --date 20260906
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.inference.exotic_probabilities import (  # noqa: E402
    RankDistribution,
    fit_rank_distribution,
    normalize_marginal,
)


# ── 1. シミュレーション ──────────────────────────────────────────────────────


def simulate_thurstone(strength: np.ndarray, sigma: float, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """能力 strength（大きいほど強い）+ N(0, sigma²) のノイズで着順を決め、各サンプルの上位 3 頭 index を返す。"""
    n = len(strength)
    perf = strength[None, :] + sigma * rng.standard_normal((n_samples, n))
    return np.argsort(-perf, axis=1)[:, :3]


def empirical_truth(top3: np.ndarray, n: int) -> dict[str, Any]:
    """シミュレーションの頻度から、勝率・連対率・複勝率と馬連・ワイド・3連複の真の確率を作る。"""
    m = len(top3)
    win = np.bincount(top3[:, 0], minlength=n) / m
    top2 = (np.bincount(top3[:, 0], minlength=n) + np.bincount(top3[:, 1], minlength=n)) / m
    top3m = top2 + np.bincount(top3[:, 2], minlength=n) / m
    umaren: dict = {}
    wide: dict = {}
    trio: dict = {}
    for a, b, c in top3:
        umaren[tuple(sorted((a, b)))] = umaren.get(tuple(sorted((a, b))), 0) + 1
        t = tuple(sorted((a, b, c)))
        trio[t] = trio.get(t, 0) + 1
        for x, y in combinations(t, 2):
            wide[(x, y)] = wide.get((x, y), 0) + 1
    return {"win": win, "top2": top2, "top3": top3m,
            "umaren": {k: v / m for k, v in umaren.items()}, "wide": {k: v / m for k, v in wide.items()},
            "sanrenpuku": {k: v / m for k, v in trio.items()}}


def _weighted_rel_error(model: dict, truth: dict) -> float:
    """真の確率で重み付けした相対誤差 Σ truth·|model/truth − 1|（頻出する組合せほど重視）。"""
    tot = 0.0
    for k, t in truth.items():
        tot += t * abs(model.get(k, 0.0) / t - 1.0)
    return tot / sum(truth.values())


def run_synthetic(n_races: int = 12, n_horses: int = 14, sigma: float = 1.0, n_samples: int = 200_000, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {f"{m}_{b}": [] for m in ("harville", "fitted") for b in ("umaren", "wide", "sanrenpuku")}
    alphas = []
    for _ in range(n_races):
        strength = rng.normal(0.0, 1.0, n_horses)
        truth = empirical_truth(simulate_thurstone(strength, sigma, n_samples, rng), n_horses)
        h = RankDistribution(truth["win"])
        fit, rep = fit_rank_distribution(truth["win"], truth["top2"], truth["top3"])
        alphas.append((rep.alpha2, rep.alpha3))
        for name, dist in (("harville", h), ("fitted", fit)):
            acc[f"{name}_umaren"].append(_weighted_rel_error(dist.all_umaren(), truth["umaren"]))
            acc[f"{name}_wide"].append(_weighted_rel_error(dist.all_wide(), truth["wide"]))
            acc[f"{name}_sanrenpuku"].append(_weighted_rel_error(dist.all_sanrenpuku(), truth["sanrenpuku"]))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["alpha2_mean"], out["alpha3_mean"] = (float(np.mean([a for a, _ in alphas])), float(np.mean([b for _, b in alphas])))
    return out


# ── 2. 実データ（市場オッズ + 実際の着順） ────────────────────────────────────


def load_race(root: Path, rid: str) -> dict[str, Any] | None:
    def j(cat: str) -> dict | None:
        p = root / cat / rid[:4] / f"{rid}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    odds, pair, res = j("race_odds"), j("race_pair_odds"), j("race_result")
    if not (odds and pair and res):
        return None
    ok = [e for e in odds["entries"] if (e.get("win_odds") or 0) > 0]
    nums = [int(e["horse_number"]) for e in ok]
    if len(nums) < 5:
        return None
    pos = {int(e["horse_number"]): int(e.get("finish_position") or 0) for e in res["entries"]}
    finish = sorted((hn for hn in nums if pos.get(hn, 0) > 0), key=lambda hn: pos[hn])
    if len(finish) < 3:
        return None
    return {"rid": rid, "nums": nums, "odds": {int(e["horse_number"]): e for e in ok}, "pair": pair, "top3": finish[:3]}


def market_probs(r: dict[str, Any]) -> dict[str, Any]:
    nums = r["nums"]
    idx = {hn: i for i, hn in enumerate(nums)}
    n = len(nums)
    win = np.array([1.0 / r["odds"][hn]["win_odds"] for hn in nums])
    win /= win.sum()
    plc = []
    for hn in nums:
        o = r["odds"][hn]
        lo, hi = o.get("place_odds_min") or 0, o.get("place_odds_max") or 0
        mid = (lo + hi) / 2 if lo and hi else (lo or hi or 10.0)
        plc.append(1.0 / mid)
    top3 = normalize_marginal(plc, 3.0)

    def table(entries, key_len, mid=False):
        t = {}
        for e in entries:
            ks = tuple(sorted(idx[x] for x in (e.get("pair") or e.get("combo") or []) if x in idx))
            o = ((e.get("odds_min", 0) + e.get("odds_max", 0)) / 2) if mid else e.get("odds", 0)
            if len(ks) == key_len and o and o > 0:
                t[ks] = 1.0 / o
        return t

    um, wd, tr = table(r["pair"].get("umaren", []), 2), table(r["pair"].get("wide", []), 2, mid=True), table(r["pair"].get("sanrenpuku", []), 3)
    for tbl, total in ((um, 1.0), (wd, 3.0), (tr, 1.0)):
        s = sum(tbl.values())
        for k in tbl:
            tbl[k] *= total / s
    top2 = np.zeros(n)
    for (a, b), p in um.items():
        top2[a] += p
        top2[b] += p
    return {"n": n, "idx": idx, "win": win, "top3": top3, "top2": normalize_marginal(top2, 2.0), "umaren": um, "wide": wd, "sanrenpuku": tr}


def _kl(market: dict, model: dict, total: float) -> float:
    """KL(市場 || モデル)。どちらも総和 total で割って確率分布にしてから比べる。"""
    s_m, s_p = sum(market.values()), sum(model.get(k, 0.0) for k in market) or 1.0
    return sum((m / s_m) * math.log((m / s_m) / max(model.get(k, 0.0) / s_p, 1e-12)) for k, m in market.items())


def run_market(root: Path, date: str) -> dict[str, Any]:
    from src.scripts.data.dev_fixture import _roots

    lst = _roots()["page_reference"] / "race_lists" / f"{date}.json"
    rids = [r["race_id"] for r in json.loads(lst.read_text(encoding="utf-8"))["races"]]
    return _run_market_ids(root, rids)


def _run_market_ids(root: Path, rids: list[str]) -> dict[str, Any]:
    models = ("harville", "fit_win_top3", "fit_win_top2_top3")
    kl = {f"{m}_{b}": [] for m in models for b in ("umaren", "wide", "sanrenpuku")}
    ll = {f"{m}_{b}": [] for m in (*models, "market") for b in ("umaren", "wide", "sanrenpuku")}
    alphas = {"fit_win_top3": [], "fit_win_top2_top3": []}
    used = 0
    for rid in rids:
        r = load_race(root, rid)
        if not r:
            continue
        mp = market_probs(r)
        if not (mp["umaren"] and mp["wide"] and mp["sanrenpuku"]):
            continue
        used += 1
        idx = mp["idx"]
        dists = {"harville": RankDistribution(mp["win"])}
        d, rep = fit_rank_distribution(mp["win"], None, mp["top3"])
        dists["fit_win_top3"] = d
        alphas["fit_win_top3"].append((rep.alpha2, rep.alpha3))
        d, rep = fit_rank_distribution(mp["win"], mp["top2"], mp["top3"])
        dists["fit_win_top2_top3"] = d
        alphas["fit_win_top2_top3"].append((rep.alpha2, rep.alpha3))
        a, b, c = (idx[x] for x in r["top3"])
        real = {"umaren": tuple(sorted((a, b))), "sanrenpuku": tuple(sorted((a, b, c)))}
        real_wide = [tuple(sorted(p)) for p in combinations((a, b, c), 2)]
        for name, dist in dists.items():
            um, wd, tr = dist.all_umaren(), dist.all_wide(), dist.all_sanrenpuku()
            kl[f"{name}_umaren"].append(_kl(mp["umaren"], um, 1.0))
            kl[f"{name}_wide"].append(_kl(mp["wide"], wd, 3.0))
            kl[f"{name}_sanrenpuku"].append(_kl(mp["sanrenpuku"], tr, 1.0))
            ll[f"{name}_umaren"].append(-math.log(max(um[real["umaren"]], 1e-12)))
            ll[f"{name}_wide"].append(float(np.mean([-math.log(max(wd[k], 1e-12)) for k in real_wide])))
            ll[f"{name}_sanrenpuku"].append(-math.log(max(tr[real["sanrenpuku"]], 1e-12)))
        ll["market_umaren"].append(-math.log(max(mp["umaren"].get(real["umaren"], 1e-12), 1e-12)))
        ll["market_wide"].append(float(np.mean([-math.log(max(mp["wide"].get(k, 1e-12), 1e-12)) for k in real_wide])))
        ll["market_sanrenpuku"].append(-math.log(max(mp["sanrenpuku"].get(real["sanrenpuku"], 1e-12), 1e-12)))
    f = lambda d: {k: (float(np.mean(v)) if v else None) for k, v in d.items()}  # noqa: E731
    al = {k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for k, v in alphas.items() if v}
    return {"races_used": used, "kl_vs_market": f(kl), "realized_logloss": f(ll), "alpha_mean(α2,α3)": al}


# ── 3. 学習済みモデルの出力で評価・調整（保存された予測 × 実際の結果） ───────────────────


def _is_mock_prediction(pred: dict) -> bool:
    m = pred.get("_meta") or {}
    return bool(m.get("mock_prediction") or m.get("dev_mock") or (pred.get("_cache_meta") or {}).get("source") == "dev_mock")


def collect_predicted_races(storage, race_ids: list[str], *, include_mock: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """保存された予測（race_predictions）と結果（race_result）を突き合わせる。モック予測は既定で除外。"""
    from src.pipeline.inference.race_prediction_service import load_cached

    races: list[dict[str, Any]] = []
    skipped = {"no_prediction": 0, "mock": 0, "no_result": 0, "incomplete": 0}
    for rid in race_ids:
        pred = load_cached(storage, rid)
        if not pred or not pred.get("predictions"):
            skipped["no_prediction"] += 1
            continue
        if _is_mock_prediction(pred) and not include_mock:
            skipped["mock"] += 1
            continue
        res = storage.load("race_result", rid) or {}
        pos = {int(e["horse_number"]): int(e.get("finish_position") or 0) for e in res.get("entries", []) if e.get("horse_number") is not None}
        rows = [p for p in pred["predictions"] if p.get("horse_number") is not None and p.get("win_prob") is not None]
        nums = [int(p["horse_number"]) for p in rows]
        order = sorted((hn for hn in nums if pos.get(hn, 0) > 0), key=lambda hn: pos[hn])
        if not pos:
            skipped["no_result"] += 1
            continue
        if len(rows) < 5 or len(order) < 3:
            skipped["incomplete"] += 1
            continue
        races.append({
            "rid": rid, "nums": nums, "win": [float(p["win_prob"]) for p in rows],
            "top2": [p.get("top2_prob") for p in rows], "top3": [p.get("top3_prob") for p in rows],
            "finish": order[:3], "pos": pos,
        })
    return races, skipped


def _stage_loglik(races: list[dict[str, Any]], stage: int, alpha: float, a2: float = 1.0) -> float:
    """実際の着順（2 着目 or 3 着目）の対数尤度。stage=2 は 1 着馬を除いた中での 2 着、stage=3 は 1・2 着を除いた中での 3 着。"""
    ll = 0.0
    for r in races:
        w = _normalize = np.clip(np.asarray(r["win"], dtype=float), 1e-12, None)
        w = w / w.sum()
        ix = {hn: i for i, hn in enumerate(r["nums"])}
        a, b, c = (ix[h] for h in r["finish"])
        if stage == 2:
            lam = w ** alpha
            ll += math.log(max(lam[b] / (lam.sum() - lam[a]), 1e-12))
        else:
            lam = w ** alpha
            ll += math.log(max(lam[c] / (lam.sum() - lam[a] - lam[b]), 1e-12))
    return ll


def mle_global_alpha(races: list[dict[str, Any]]) -> tuple[float, float]:
    """全レース共通の α2, α3 を、実際の 2・3 着の対数尤度最大で決める（勝率 win_prob を worth とする Benter 型）。"""
    from src.pipeline.inference.exotic_probabilities import ALPHA_MAX, ALPHA_MIN, _golden_min

    a2 = _golden_min(lambda a: -_stage_loglik(races, 2, a), ALPHA_MIN, ALPHA_MAX)
    a3 = _golden_min(lambda a: -_stage_loglik(races, 3, a), ALPHA_MIN, ALPHA_MAX)
    return a2, a3


def calibration(races: list[dict[str, Any]], key: str, k: int) -> dict[str, Any]:
    """win/top2/top3 の予測確率 vs 実際（k 着以内）。Brier・対数損失・10 分位の信頼性。"""
    ps, ys = [], []
    for r in races:
        vals = r["win"] if key == "win" else r[key]
        if any(v is None for v in vals):
            continue
        for hn, p in zip(r["nums"], vals):
            ps.append(float(p))
            ys.append(1.0 if 0 < r["pos"].get(hn, 0) <= k else 0.0)
    if not ps:
        return {}
    p, y = np.clip(np.array(ps), 1e-9, 1 - 1e-9), np.array(ys)
    order = np.argsort(p)
    bins = [
        {"pred": float(p[ix].mean()), "obs": float(y[ix].mean()), "n": int(len(ix))}
        for ix in np.array_split(order, 10) if len(ix)
    ]
    return {"n": int(len(p)), "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "bins": bins}


def evaluate_predictions(races: list[dict[str, Any]]) -> dict[str, Any]:
    """Harville / レースごとの当てはめ / 全レース共通 α（MLE）で、実際の馬連・ワイド・3連複の対数損失を比べる。"""
    g2, g3 = mle_global_alpha(races)
    ll = {f"{m}_{b}": [] for m in ("harville", "per_race_fit", "global_alpha") for b in ("umaren", "wide", "sanrenpuku")}
    for r in races:
        ix = {hn: i for i, hn in enumerate(r["nums"])}
        a, b, c = (ix[h] for h in r["finish"])
        t2 = r["top2"] if all(v is not None for v in r["top2"]) else None
        t3 = r["top3"] if all(v is not None for v in r["top3"]) else None
        dists = {
            "harville": RankDistribution(r["win"]),
            "per_race_fit": fit_rank_distribution(r["win"], t2, t3)[0],
            "global_alpha": RankDistribution(r["win"], g2, g3),
        }
        real_wide = [tuple(sorted(p)) for p in combinations((a, b, c), 2)]
        for name, d in dists.items():
            ll[f"{name}_umaren"].append(-math.log(max(d.umaren(a, b), 1e-12)))
            ll[f"{name}_wide"].append(float(np.mean([-math.log(max(d.wide(*k), 1e-12)) for k in real_wide])))
            ll[f"{name}_sanrenpuku"].append(-math.log(max(d.sanrenpuku(a, b, c), 1e-12)))
    n = len(races)
    mean = {k: float(np.mean(v)) for k, v in ll.items()}
    se = {k: float(np.std(v, ddof=1) / math.sqrt(n)) if n > 1 else None for k, v in ll.items()}
    return {"races": n, "global_alpha": (g2, g3), "realized_logloss": mean, "realized_logloss_se": se,
            "calibration": {k: calibration(races, k, kk) for k, kk in (("win", 1), ("top2", 2), ("top3", 3))}}


def write_alpha_config(alpha2: float, alpha3: float, n_races: int, source: str) -> Path:
    from datetime import datetime

    from src.pipeline.inference.exotic_probabilities import ALPHA_CONFIG

    ALPHA_CONFIG.write_text(json.dumps({
        "alpha2": round(alpha2, 4), "alpha3": round(alpha3, 4), "n_races": n_races, "source": source,
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "note": "連対率・複勝率が無いときの既定 α。validate_exotic_probabilities predictions --write-config で更新",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ALPHA_CONFIG


def run_predictions(storage, dates: list[str], *, include_mock: bool = False) -> dict[str, Any]:
    rids: list[str] = []
    for d in dates:
        rl = storage.load("race_lists", d) or {}
        rids += [r["race_id"] for r in rl.get("races", []) if r.get("race_id")]
    races, skipped = collect_predicted_races(storage, rids, include_mock=include_mock)
    out: dict[str, Any] = {"dates": dates, "race_ids": len(rids), "skipped": skipped}
    out.update(evaluate_predictions(races) if races else {"races": 0})
    return out


def _dates_between(start: str, end: str) -> list[str]:
    from datetime import date, timedelta

    d0, d1 = date(int(start[:4]), int(start[4:6]), int(start[6:])), date(int(end[:4]), int(end[4:6]), int(end[6:]))
    return [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]


def _print_table(title: str, d: dict[str, float | None]) -> None:
    print(f"\n{title}")
    keys = sorted({k.rsplit("_", 1)[1] for k in d})
    models = []
    for k in d:
        m = k.rsplit("_", 1)[0]
        if m not in models:
            models.append(m)
    print(f"  {'モデル':<20}" + "".join(f"{b:>14}" for b in keys))
    for m in models:
        print(f"  {m:<20}" + "".join(f"{(d.get(f'{m}_{b}') if d.get(f'{m}_{b}') is not None else float('nan')):>14.4f}" for b in keys))


def _cmd_predictions(args) -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    from src.scraper.storage import HybridStorage

    r = run_predictions(HybridStorage(), _dates_between(args.date_from, args.date_to), include_mock=args.include_mock)
    print(f"{args.date_from}〜{args.date_to}: 対象レース {r['race_ids']} 件 → 評価 {r['races']} レース / 除外 {r['skipped']}")
    if not r["races"]:
        print("評価できるレースがありません（保存済みの予測と結果が必要。モック予測は --include-mock で含められます）")
        return 1
    g2, g3 = r["global_alpha"]
    print(f"全レース共通 α（実際の 2・3 着の対数尤度が最大）: α2={g2:.3f} α3={g3:.3f}")
    _print_table("実際の着順に対する −log P（小さいほど良い）", r["realized_logloss"])
    se = r["realized_logloss_se"]
    print("  標準誤差（目安）:", {k: round(v, 3) for k, v in se.items() if v is not None and k.startswith("harville")})
    for name, c in r["calibration"].items():
        if c:
            print(f"\n較正 {name}: n={c['n']} Brier={c['brier']:.4f} logloss={c['logloss']:.4f}")
            print("   予測平均 → 実測:", "  ".join(f"{b['pred']:.2f}→{b['obs']:.2f}" for b in c["bins"]))
    if args.write_config:
        if args.include_mock:
            print("モック予測を含む評価では --write-config できません", file=sys.stderr)
            return 2
        if r["races"] < args.min_races:
            print(f"レース数 {r['races']} < {args.min_races}: 既定 α は更新しません（--min-races で変更可）", file=sys.stderr)
            return 2
        print("書き込み:", write_alpha_config(g2, g3, r["races"], f"{args.date_from}-{args.date_to}"))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--sigma", type=float, default=1.0)
    m = sub.add_parser("market")
    m.add_argument("--date", default="20260906")
    pr = sub.add_parser("predictions", help="学習済みモデルの保存済み予測 × 実際の結果で評価・α を調整")
    pr.add_argument("--from", dest="date_from", required=True, help="YYYYMMDD")
    pr.add_argument("--to", dest="date_to", required=True, help="YYYYMMDD")
    pr.add_argument("--include-mock", action="store_true", help="モック予測も含める（動作確認用。--write-config とは併用不可）")
    pr.add_argument("--write-config", action="store_true", help="全レース共通 α を config/exotic_alpha.json に書く（連対率・複勝率が無いときの既定値）")
    pr.add_argument("--min-races", type=int, default=200, help="--write-config に必要な最小レース数")
    args = ap.parse_args(argv)

    if args.cmd == "synthetic":
        r = run_synthetic(sigma=args.sigma)
        print(f"シミュレーション（Thurstone 型, σ={args.sigma}）— 真の確率で重み付けした平均相対誤差（小さいほど良い）")
        for b in ("umaren", "wide", "sanrenpuku"):
            print(f"  {b:<12} Harville(勝率のみ) {r[f'harville_{b}']:.4f}   割引Harville(勝率+連対率+複勝率) {r[f'fitted_{b}']:.4f}")
        print(f"  当てはめた α2={r['alpha2_mean']:.3f} α3={r['alpha3_mean']:.3f}")
    elif args.cmd == "predictions":
        return _cmd_predictions(args)
    else:
        from src.scraper.dev_store import dev_mock_root

        root = dev_mock_root(REPO_ROOT)
        r = run_market(root, args.date)
        print(f"実データ {args.date}: {r['races_used']} レースで評価")
        _print_table("市場オッズから逆算した確率との乖離 KL(市場||モデル)（小さいほど良い）", r["kl_vs_market"])
        _print_table("実際の着順に対する対数損失 −log P(実現)（小さいほど良い。market は参考値）", r["realized_logloss"])
        print("\n当てはめた α の平均 (α2, α3):", r["alpha_mean(α2,α3)"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
