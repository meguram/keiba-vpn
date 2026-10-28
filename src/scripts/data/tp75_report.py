"""tp-75 の結果から、章 HTML・図・凍結ファイル・台帳 CSV を作る。

python -m src.scripts.data.tp75_report --src <結果ディレクトリ> [--no-frozen]
  結果ディレクトリ = tp75_rules.py eval の出力（results.json, ledger.json）。ホストの /home/jovyan/work/tp75_out を手元へ集めて指定する。
  出力: docs/html/modeling/chapters/rule_personas.html、config/tp75_frozen_rule_ev.json / tp75_frozen_rule_place.json、docs/operations/tp75_rule_ledger.csv、tp75_chapter_inventory.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CH = ROOT / "docs/html/modeling/chapters/rule_personas.html"
FROZEN = {"ev_seeker": ROOT / "config/tp75_frozen_rule_ev.json", "place_seeker": ROOT / "config/tp75_frozen_rule_place.json"}
LEDGER_CSV = ROOT / "docs/operations/tp75_rule_ledger.csv"
INV_CSV = ROOT / "docs/operations/tp75_chapter_inventory.csv"
FAM = {"ev_buy": ("win", +1, "ev_seeker", "buy"), "ev_kill": ("win", -1, "ev_seeker", "kill"), "place_buy": ("top3", +1, "place_seeker", "buy"), "place_kill": ("top3", -1, "place_seeker", "kill")}
FAM_JA = {"ev_buy": "期待値追求家の買い（勝ちの過小評価）", "ev_kill": "期待値追求家の消し（勝ちの過大評価）", "place_buy": "複勝率追求家の買い（3着内の過小評価）", "place_kill": "複勝率追求家の消し（3着内の過大評価）"}
PERSONA_JA = {"ev_seeker": "期待値追求家", "place_seeker": "複勝率追求家"}
FROZEN_ON = "2026-10-22"


def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for cand in [*Path.home().glob(".fonts/*JP*.*"), *Path("/usr/share/fonts").glob("**/*CJK*JP*.*")]:
        try:
            fm.fontManager.addfont(str(cand))
        except Exception:  # noqa: BLE001
            pass
    have = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams["font.family"] = [n for n in ("Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "DejaVu Sans") if n in have][:1] or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _b64(fig) -> str:
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    buf.seek(0)
    im = Image.open(buf).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=96)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return base64.b64encode(out.getvalue()).decode()


def pc(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{x * 100:.{d}f}"


def ci(v, d: int = 1) -> str:
    if not v or v[0] is None:
        return "–"
    return f"{v[0] * 100:.{d}f}<span class='muted'> [{v[1] * 100:.{d}f}, {v[2] * 100:.{d}f}]</span>"


def rs(b) -> str:
    if not b or b.get("mean") is None:
        return "–"
    return f"{b['mean'] * 100:+.2f}<span class='muted'> (z={b['z']:.1f})</span>"


def esc(s: str) -> str:
    return html.escape(s, quote=False)


# ------------------------------------------------------------------------------------------------ 集計
def replication(ledger: list[dict]) -> dict:
    """採用ルールが 2025・2026 でも同じ向きか（同符号の割合、z が 1.645 を向きの方向へ越える数、偶然の期待数）。"""
    out = {}
    for fam, (outcome, sign, _p, _d) in FAM.items():
        z25 = "z_2025_win" if outcome == "win" else "z_2025_top3"
        z26 = "z_2026_win" if outcome == "win" else "z_2026_top3"
        ad = [r for r in ledger if r["adopt"].get(fam, {}).get("adopted")]
        o = {"n": len(ad), "A": sum(r["adopt"][fam]["tier"] == "A" for r in ad), "B": sum(r["adopt"][fam]["tier"] == "B" for r in ad), "robust": sum(r["adopt"][fam]["robust_bh"] for r in ad)}
        for tag, col in (("y2025", z25), ("y2026", z26)):
            zs = [sign * r[col] for r in ad if r[col] is not None]
            o[tag] = {"n": len(zs), "same_sign": int(sum(z > 0 for z in zs)), "z_gt_1645": int(sum(z > 1.645 for z in zs)), "expected_null": round(0.05 * len(zs), 1),
                      "z_lt_neg1645": int(sum(z < -1.645 for z in zs))}
        rb = [r for r in ad if r["adopt"][fam]["robust_bh"]]
        zs = [sign * r[z25] for r in rb if r[z25] is not None]
        o["robust_2025"] = {"n": len(zs), "same_sign": int(sum(z > 0 for z in zs)), "z_gt_1645": int(sum(z > 1.645 for z in zs))}
        out[fam] = o
    return out


def rule_table(ledger: list[dict], clusters: dict, fam: str, k: int = 12) -> str:
    outcome, sign, _p, _d = FAM[fam]
    z25 = "z_2025_win" if outcome == "win" else "z_2025_top3"
    z26 = "z_2026_win" if outcome == "win" else "z_2026_top3"
    eff = "win_resid_pt" if outcome == "win" else "top3_resid_pt"
    rep_of = {m: c["rep"] for c in clusters[fam] for m in c["members"]}
    ad = sorted([r for r in ledger if r["adopt"].get(fam, {}).get("adopted")], key=lambda r: -r["adopt"][fam]["z"])[:k]
    rows = ""
    for r in ad:
        a = r["adopt"][fam]
        zz = lambda v: "–" if v is None else f"{sign * v:+.1f}"
        rows += (f"<tr><td>{esc(r['id'])}</td><td class='expr'>{esc(r['expr'])}</td><td>{esc(r['source'])}</td><td class='num'>{r['n_disc']:,}</td><td class='num'>{r[eff]:+.2f}</td>"
                 f"<td>{a['tier']}{'・頑健' if a['robust_bh'] else ''}</td><td class='num'>{a['z']:.1f}</td><td class='num'>{zz(r[z25])}</td><td class='num'>{zz(r[z26])}</td><td>{esc(rep_of.get(r['id'], ''))}</td></tr>")
    return ("<table><tr><th>ルール</th><th>条件</th><th>出典</th><th>走数（発見）</th><th>残差（pt）</th><th>区分</th><th>z（発見）</th><th>z（2025）</th><th>z（2026）</th><th>クラスタ代表</th></tr>" + rows + "</table>")


# ------------------------------------------------------------------------------------------------ 図
def figures(R: dict, ledger: list[dict], rep: dict) -> dict:
    plt = _setup_plt()
    imgs = {}
    # 1 採用数と 2025 での再現
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    fams = list(FAM)
    x = np.arange(len(fams))
    axs[0].bar(x - 0.2, [rep[f]["A"] for f in fams], 0.4, label="A（95%）", color="#4f8bff")
    axs[0].bar(x + 0.2, [rep[f]["B"] for f in fams], 0.4, label="B（90%・暫定）", color="#fbbf24")
    axs[0].plot(x, [rep[f]["robust"] for f in fams], "k_", ms=18, mew=2, label="BH で頑健（A+B のうち）")
    axs[0].set_xticks(x); axs[0].set_xticklabels(["期待値\n買い", "期待値\n消し", "複勝率\n買い", "複勝率\n消し"], fontsize=9); axs[0].set_title("発見期間（2022–24）で採用されたルール数", fontsize=10); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.2)
    sh = [rep[f]["y2025"]["same_sign"] / max(rep[f]["y2025"]["n"], 1) for f in fams]
    sh6 = [rep[f]["y2026"]["same_sign"] / max(rep[f]["y2026"]["n"], 1) for f in fams]
    axs[1].bar(x - 0.2, sh, 0.4, label="2025（v6e）", color="#4f8bff"); axs[1].bar(x + 0.2, sh6, 0.4, label="2026（v4・記述）", color="#9aa7c7")
    axs[1].axhline(0.5, color="#c0392b", ls=":", lw=1); axs[1].set_ylim(0, 1); axs[1].set_xticks(x); axs[1].set_xticklabels(["期待値\n買い", "期待値\n消し", "複勝率\n買い", "複勝率\n消し"], fontsize=9)
    axs[1].set_title("採用ルールが後の年でも同じ向きの割合（赤点線 = 偶然の 50%）", fontsize=10); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.2)
    fig.tight_layout(); imgs["counts"] = _b64(fig); plt.close(fig)
    # 2 発見の z と 2025 の z（買い）
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, fam in zip(axs, ("ev_buy", "place_buy")):
        outcome, sign, _p, _d = FAM[fam]
        z25 = "z_2025_win" if outcome == "win" else "z_2025_top3"
        ad = [r for r in ledger if r["adopt"].get(fam, {}).get("adopted") and r[z25] is not None]
        ax.scatter([r["adopt"][fam]["z"] for r in ad], [sign * r[z25] for r in ad], s=14, c=["#e74c3c" if r["adopt"][fam]["robust_bh"] else "#4f8bff" for r in ad])
        ax.axhline(0, color="#999", lw=0.8); ax.axvline(1.645, color="#999", ls=":", lw=0.8); ax.axhline(1.645, color="#999", ls=":", lw=0.8)
        ax.set_xlabel("発見期間の z", fontsize=9); ax.set_ylabel("2025 年の z（買いの向き）", fontsize=9); ax.set_title(FAM_JA[fam] + "（赤 = BH で頑健）", fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); imgs["scatter"] = _b64(fig); plt.close(fig)
    # 3 スコア帯別の残差
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = ["neg", "zero", "pos_low", "pos_high"]; lab = ["消し優勢（<0）", "合致なし（=0）", "買い・弱（>0）", "買い・強（>0）"]
    for ax, (p, col) in zip(axs, (("ev_seeker", "resid_win"), ("place_seeker", "resid_top3"))):
        for j, pn in enumerate(("disc", "y2025", "y2026")):
            b = R["primary"]["personas"][p][f"bins/{pn}"]
            ys = [b[k][col]["mean"] * 100 if k in b and b[k][col]["mean"] is not None and b[k]["n"] >= 300 else np.nan for k in bins]
            es = [1.96 * b[k][col]["se"] * 100 if k in b and b[k][col]["se"] is not None and b[k]["n"] >= 300 else 0 for k in bins]
            ax.errorbar(np.arange(4) + (j - 1) * 0.15, ys, yerr=es, fmt="o-", label={"disc": "発見 2022–24（選択に使った）", "y2025": "2025（v6e）", "y2026": "2026（v4・記述）"}[pn], capsize=3, ms=4)
        ax.axhline(0, color="#999", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8)
        ax.set_ylabel("実績 − 現行モデルの予測（pt, 95% 区間）", fontsize=9); ax.set_title(f"{PERSONA_JA[p]}: スコア帯別の" + ("勝ち" if col == "resid_win" else "3着内") + "の残差", fontsize=9); ax.grid(alpha=0.2); ax.legend(fontsize=7)
    fig.tight_layout(); imgs["bins"] = _b64(fig); plt.close(fig)
    # 4 トップ群の成績
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    items = []
    for p in ("ev_seeker", "place_seeker"):
        for pn in ("y2025", "y2026"):
            items.append((f"{PERSONA_JA[p][:3]}\n{pn[1:]}", R["primary"]["personas"][p][f"def_b/{pn}/top_group"]))
    mt = [("top3_rate", "3着内率（%）"), ("roi_win", "単勝の回収率（%）"), ("roi_place", "複勝の回収率（%）")]
    for ax, (k, ttl) in zip(axs, mt):
        ys = [it[1][k][0] * 100 if it[1].get(k) and it[1][k][0] is not None else np.nan for it in items]
        lo = [(it[1][k][0] - it[1][k][1]) * 100 if it[1].get(k) and it[1][k][0] is not None else 0 for it in items]
        hi = [(it[1][k][2] - it[1][k][0]) * 100 if it[1].get(k) and it[1][k][0] is not None else 0 for it in items]
        ax.bar(range(len(items)), ys, yerr=[lo, hi], capsize=3, color=["#4f8bff", "#9aa7c7", "#fbbf24", "#d7b25a"])
        if k != "top3_rate":
            ax.axhline(100, color="#c0392b", ls=":", lw=1)
        ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], fontsize=8); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.2)
    fig.suptitle("スコア b の最上位グループ（各馬 1 単位）の成績。2025 は評価 1 回、2026 は記述", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); imgs["top"] = _b64(fig); plt.close(fig)
    return imgs


# ------------------------------------------------------------------------------------------------ 凍結ファイル・台帳
def write_frozen(R: dict, ledger: list[dict], rep: dict) -> None:
    by_id = {r["id"]: r for r in ledger}
    for persona, path in FROZEN.items():
        rules = []
        clusters_out = []
        for fam, (outcome, sign, p, direction) in FAM.items():
            if p != persona:
                continue
            for c in R["clusters"][fam]:
                clusters_out.append({"family": fam, "representative": c["rep"], "members": c["members"], "weight_pt": round(100 * c["weight"], 3), "cover_disc": round(c["cover"], 4), "tier": c["tier"], "robust_bh": c["robust_bh"]})
                for m in c["members"]:
                    r = by_id[m]
                    a = r["adopt"][fam]
                    z25 = r["z_2025_win"] if outcome == "win" else r["z_2025_top3"]
                    z26 = r["z_2026_win"] if outcome == "win" else r["z_2026_top3"]
                    rules.append({"id": r["id"], "expr": r["expr"], "source": r["source"], "direction": direction, "outcome": outcome, "family": fam, "tier": a["tier"], "robust_bh": a["robust_bh"],
                                  "market_free": r["market_free"], "cluster_representative": c["rep"], "n_disc": r["n_disc"], "effect_pt": round(r["win_resid_pt"] if outcome == "win" else r["top3_resid_pt"], 3),
                                  "z_disc": round(a["z"], 3), "q_bh": round(a["q"], 4), "z_2025": None if z25 is None else round(sign * z25, 3), "z_2026_v4": None if z26 is None else round(sign * z26, 3)})
        pr = R["primary"]["personas"][persona]
        spec = {
            "version": 1, "name": f"tp75_{persona}", "kind": "rule_persona", "persona": PERSONA_JA[persona], "market_free": all(r["market_free"] for r in rules),
            "_doc": ("市場情報（オッズ・人気・市場確率・想定オッズ・期待値）を条件・順位・採用に使わないルール集合。現行モデルの予測に対する残差で採用した。"
                     "章: docs/html/modeling/chapters/rule_personas.html、決定ログ §69。再評価: python -m src.scripts.data.tp75_rules reeval <このファイル> --years <新しい年>"),
            "frozen_on": FROZEN_ON, "track_from": FROZEN_ON, "base_model": "v6e（LOYO OOF 2022–24, test 2025）。2026 は v4（bundle_v4_2025）で記述のみ",
            "criteria": {"tiers": {"A": "両側 95%", "B": "両側 90%（暫定）"}, "discovery_years": [2022, 2023, 2024], "replication": "n≥100 の年が 2 年以上で過半が同じ向き", "robust": "BH FDR 10%",
                         "buy": "勝ち（期待値追求家）／3着内（複勝率追求家）の残差の下側限界>0" , "kill": "同・上側限界<0", "clusters": "Jaccard≥0.5 または包含≥0.9 は同一クラスタ"},
            "scoring": {"a": "買いクラスタ数−消しクラスタ数", "b": "縮約した効果（n/(n+300)）の合計（買い−消し）【主】", "c": "最強の買い−最強の消し", "kill_handling": ["exclude", "penalize"]},
            "n_rules": len(rules), "n_clusters": len(clusters_out), "clusters": clusters_out, "rules": rules,
            "evaluation_summary": {"top_group_def_b": {pn: {k: pr[f"def_b/{pn}/top_group"].get(k) for k in ("n_picks", "n_races", "win_rate", "top3_rate", "roi_win", "roi_place")} for pn in ("disc", "y2025", "y2026")},
                                   "replication": {f: rep[f] for f, v in FAM.items() if v[2] == persona}},
            "history": [{"date": FROZEN_ON, "action": "freeze v1", "added": len(rules), "expired": 0, "note": "発見 2022–24 で採用（A・B）されたルールすべて。使う側は tier=A かつ robust_bh を優先する"}],
            "operations": {"reeval_cadence": "毎月（新しいレースが 300 走以上たまったルールから）", "status_rule": "ok / watch（z<0）/ failing（z<−1.645）。failing が 2 か月続く、または凍結後の累積で z<−1.645 になったら失効に移し、history に記録して rules から外す（新版の version を上げる）",
                           "add_rule": "新しい条件は別の事前登録（型の数を記録）で検定し、発見期間で採用された場合だけ新版に追加する。結果を見てから条件を増やさない"},
        }
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")


def write_csv(ledger: list[dict], R: dict) -> None:
    rep_of = {fam: {m: c["rep"] for c in R["clusters"][fam] for m in c["members"]} for fam in FAM}
    cols = ["id", "family", "name", "expr", "source", "market_free", "race_level", "n_disc", "races_disc", "win_resid_pt", "win_resid_z", "top3_resid_pt", "top3_resid_z", "roi_win_disc", "roi_place_disc",
            "z_2025_win", "z_2025_top3", "n_2025", "z_2026_win", "z_2026_top3", "n_2026"]
    ext = [f"{f}_{k}" for f in FAM for k in ("tier", "adopted", "robust_bh", "q", "cluster")]
    with LEDGER_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ext)
        for r in ledger:
            row = [r.get(c) for c in cols]
            for f in FAM:
                a = r["adopt"].get(f, {})
                row += [a.get("tier"), a.get("adopted"), a.get("robust_bh"), None if a.get("q") is None else round(a["q"], 4), rep_of[f].get(r["id"])]
            w.writerow(row)


def write_inventory_csv() -> None:
    from src.scripts.data.tp75_inventory import INVENTORY
    with INV_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "kind", "ingest", "family", "rule_stats", "reason"])
        w.writeheader()
        w.writerows(INVENTORY)


# ------------------------------------------------------------------------------------------------ 章
def chapter(src: Path) -> None:
    from src.scripts.data.tp75_inventory import INVENTORY

    R = json.loads((src / "results.json").read_text(encoding="utf-8"))
    ledger = json.loads((src / "ledger.json").read_text(encoding="utf-8"))
    pre = json.loads((Path(__file__).with_name("tp75_prereg.json")).read_text(encoding="utf-8"))
    rep = replication(ledger)
    imgs = figures(R, ledger, rep)
    write_frozen(R, ledger, rep)
    write_csv(ledger, R)
    write_inventory_csv()
    css = re.search(r"<style>.*?</style>", (ROOT / "docs/html/modeling/chapters/equipment_and_change.html").read_text(encoding="utf-8"), re.S).group(0)
    P, N = R["primary"], R["narrow_posthoc"]
    s = R["adopt_summary"]
    ex = pre["excluded_market"]
    inv_ct: dict[str, int] = {}
    for it in INVENTORY:
        inv_ct[it["ingest"]] = inv_ct.get(it["ingest"], 0) + 1

    def top(p: str, pn: str, d: str = "b", base: bool = False) -> dict:
        return P["personas"][p][f"def_{d}/{pn}/" + ("model_top1_same_races" if base else "top_group")]

    # 表: ペルソナの最上位グループ
    trows = ""
    for p in ("ev_seeker", "place_seeker"):
        for dn in "abc":
            for pn, pl in (("disc", "発見（選択に使用）"), ("y2025", "2025（評価 1 回）"), ("y2026", "2026（記述）")):
                t, b = top(p, pn, dn), top(p, pn, dn, True)
                if not t.get("n_picks"):
                    continue
                trows += (f"<tr><td>{PERSONA_JA[p]}</td><td>{dn}</td><td>{pl}</td><td class='num'>{t['n_races']:,}</td><td class='num'>{ci(t['win_rate'])}</td><td class='num'>{ci(t['top3_rate'])}</td>"
                          f"<td class='num'>{pc(t['model_p_top3_mean'])}</td><td class='num'>{ci(t['roi_win'], 0)}</td><td class='num'>{ci(t['roi_place'], 0)}</td>"
                          f"<td class='num'>{pc(b['win_rate'][0])}</td><td class='num'>{pc(b['top3_rate'][0])}</td><td class='num'>{pc(b['roi_win'][0], 0)}</td></tr>")
    # 表: スコア帯別
    brows = ""
    lab = {"neg": "消し優勢（スコア<0）", "zero": "合致なし（=0）", "pos_low": "買い・弱", "pos_high": "買い・強"}
    for p in ("ev_seeker", "place_seeker"):
        for pn, pl in (("disc", "発見"), ("y2025", "2025"), ("y2026", "2026")):
            for k, v in P["personas"][p][f"bins/{pn}"].items():
                if v["n"] < 100:
                    continue
                brows += (f"<tr><td>{PERSONA_JA[p]}</td><td>{pl}</td><td>{lab[k]}</td><td class='num'>{v['n']:,}</td><td class='num'>{rs(v['resid_win'])}</td><td class='num'>{rs(v['resid_top3'])}</td>"
                          f"<td class='num'>{ci(v['roi_win'], 0)}</td><td class='num'>{ci(v['roi_place'], 0)}</td></tr>")
    # 消し条件
    krows = ""
    for p in ("ev_seeker", "place_seeker"):
        for pn, pl in (("disc", "発見"), ("y2025", "2025"), ("y2026", "2026")):
            a, b = P["personas"][p][f"kill_filter/{pn}/top3_by_ptop3_kill"], P["personas"][p][f"kill_filter/{pn}/top3_by_ptop3_other"]
            t1, t2 = P["personas"][p][f"kill_filter/{pn}/model_top1_all"], P["personas"][p][f"kill_filter/{pn}/model_top1_excluding_kill"]
            krows += (f"<tr><td>{PERSONA_JA[p]}</td><td>{pl}</td><td class='num'>{a['n_picks']:,}</td><td class='num'>{ci(a['top3_rate'])}</td><td class='num'>{b['n_picks']:,}</td><td class='num'>{ci(b['top3_rate'])}</td>"
                      f"<td class='num'>{t1['n_picks']:,}</td><td class='num'>{ci(t1['top3_rate'])}</td><td class='num'>{t2['n_picks']:,}</td><td class='num'>{ci(t2['top3_rate'])}</td></tr>")
    # overlay
    orow = ""
    for tag, blk in (("主（事前登録）", P), ("事後・狭いクラスタのみ", N)):
        for p in ("ev_seeker", "place_seeker"):
            ov = blk["personas"][p]["overlay"]
            cells = ""
            for pn in ("disc", "y2025", "y2026"):
                o = ov[pn]
                cells += f"<td class='num'>{o['delta'] * 1e4:+.1f}<span class='muted'> [{o['lo'] * 1e4:+.1f}, {o['hi'] * 1e4:+.1f}]</span></td>"
            orow += f"<tr><td>{tag}</td><td>{PERSONA_JA[p]}</td>{cells}</tr>"
    # 2 ペルソナ
    xrows = ""
    qlab = {"both_pos": "両方が買い（スコア>0）", "ev_only": "期待値追求家だけ買い", "place_only": "複勝率追求家だけ買い", "neither": "どちらも買いでない"}
    for pn, pl in (("disc", "発見"), ("y2025", "2025"), ("y2026", "2026")):
        for k in ("both_pos", "ev_only", "place_only", "neither"):
            v = P["cross"][pn][k]
            xrows += (f"<tr><td>{pl}</td><td>{qlab[k]}</td><td class='num'>{v['n']:,}</td><td class='num'>{ci(v['top3_rate'])}</td><td class='num'>{ci(v['win_rate'])}</td><td class='num'>{rs(v['resid_top3'])}</td><td class='num'>{ci(v['roi_win'], 0)}</td><td class='num'>{ci(v['roi_place'], 0)}</td></tr>")
        a, b = P["cross"][pn]["model_top3_both_pos"], P["cross"][pn]["model_top3_not_both"]
        xrows += f"<tr><td>{pl}</td><td>モデルの複勝確率上位3頭のうち、両方が買い</td><td class='num'>{a['n']:,}</td><td class='num'>{ci(a['top3_rate'])}</td><td></td><td></td><td></td><td class='num'>{ci(a['roi_place'], 0)}</td></tr>"
        xrows += f"<tr><td>{pl}</td><td>モデルの複勝確率上位3頭のうち、両方が買いでない</td><td class='num'>{b['n']:,}</td><td class='num'>{ci(b['top3_rate'])}</td><td></td><td></td><td></td><td class='num'>{ci(b['roi_place'], 0)}</td></tr>"
    # 採用の表
    arows = ""
    for fam in FAM:
        r_ = rep[fam]
        arows += (f"<tr><td>{FAM_JA[fam]}</td><td class='num'>{s[fam]['testable']}</td><td class='num'>{s[fam]['z_gt_1.645_before_replication']}（偶然の期待 {s[fam]['expected_false_pos_at_1.645']:.0f}）</td><td class='num'>{r_['A']}</td><td class='num'>{r_['B']}</td>"
                  f"<td class='num'>{r_['robust']}</td><td class='num'>{len(R['clusters'][fam])}</td><td class='num'>{r_['y2025']['same_sign']} / {r_['y2025']['n']}</td><td class='num'>{r_['y2025']['z_gt_1645']}（期待 {r_['y2025']['expected_null']}）</td>"
                  f"<td class='num'>{r_['robust_2025']['same_sign']} / {r_['robust_2025']['n']}</td><td class='num'>{r_['y2026']['same_sign']} / {r_['y2026']['n']}</td></tr>")
    # 台帳（出典の棚卸し）
    ing_ja = {"recomputed": "再計算して台帳に入れた", "excluded_market": "市場依存のため除外", "baseline": "比較の基準に使った", "none": "取り込まなかった"}
    irows = ""
    for it in sorted(INVENTORY, key=lambda x: ({"recomputed": 0, "excluded_market": 1, "baseline": 2, "none": 3}[x["ingest"]], x["source"])):
        irows += f"<tr><td>{esc(it['source'])}</td><td>{ing_ja[it['ingest']]}</td><td>{esc(it['family'])}</td><td>{esc(it['rule_stats'])}</td><td>{esc(it['reason'])}</td></tr>"
    exrows = "".join(f"<tr><td>{esc(e['id'])}</td><td class='expr'>{esc(e['expr'])}</td><td>{esc(e['reason'])}</td></tr>" for e in ex)
    fam_rows = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in pre["counts"]["by_family"].items())
    eb, pb = P["personas"]["ev_seeker"], P["personas"]["place_seeker"]
    ev25b, pl25b = eb["bins/y2025"], pb["bins/y2025"]
    ev26b, pl26b = eb["bins/y2026"], pb["bins/y2026"]
    t_ev25, t_pl25 = top("ev_seeker", "y2025"), top("place_seeker", "y2025")
    t_ev26, t_pl26 = top("ev_seeker", "y2026"), top("place_seeker", "y2026")
    m_ev25 = top("ev_seeker", "y2025", "b", True)
    ov25_ev, ov25_pl = eb["overlay"]["y2025"], pb["overlay"]["y2025"]
    hi25 = max(P["personas"][p][f"def_{d}/y2025/top_group"][k][2] for p in ("ev_seeker", "place_seeker") for d in "abc" for k in ("roi_win", "roi_place") if P["personas"][p][f"def_{d}/y2025/top_group"].get(k) and P["personas"][p][f"def_{d}/y2025/top_group"][k][2] is not None)
    cr26 = [(PERSONA_JA[p], d, k, P["personas"][p][f"def_{d}/y2026/top_group"][k]) for p in ("ev_seeker", "place_seeker") for d in "abc" for k in ("roi_win", "roi_place") if P["personas"][p][f"def_{d}/y2026/top_group"].get(k)]
    n_cross26 = sum(1 for x in cr26 if x[3][1] is not None and x[3][1] <= 1.0 <= x[3][2])
    share_kill = P["personas"]["ev_seeker"]["kill_filter/y2025/changed_races_before"]["n_picks"] / max(P["personas"]["ev_seeker"]["kill_filter/y2025/model_top1_all"]["n_picks"], 1)
    xa, xb = P["cross"]["y2025"]["model_top3_both_pos"], P["cross"]["y2025"]["model_top3_not_both"]
    html_doc = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>ルールベースのペルソナ: 期待値追求家・複勝率追求家（tp-75）— keiba-vpn</title>
{css}
<style>td.expr {{ font-family: monospace; font-size: 12px; max-width: 360px; word-break: break-all; }} details {{ margin: 8px 0 16px; }} summary {{ cursor: pointer; color: var(--accent-2); }}</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>ルールベースのペルソナ: 期待値追求家・複勝率追求家（tp-75）</h1>
<p class="muted">ユーザーの依頼（2026-10-22）: 『期待値だけを追うモデル（集計結果・過去統計をもとに、それに該当する馬を買うモデル）のペルソナを追加したい。合致率で順位を定め、「この条件は来ない」という消しの条件も加味する。統計的に優位なものを採用し、95% が難しければ 90% でもよい』『複勝率という軸のペルソナも用意する』『大衆情報（オッズ・人気・市場確率）はルールに組み込まない』。コード: src/scripts/data/tp75_rules.py（prereg / eval / reeval）・tp75_report.py・tp75_inventory.py。ルール集合は評価の実行前にコミット（b8a7541）。モデルの学習はしていない（集計と規則だけ）。ペルソナ調査（<a href="persona_ensemble_survey.html">persona_ensemble_survey.html</a>）の 6・7 つ目のペルソナの実体。</p>

<div class="note"><b>要点.</b>
<ul>
<li><b>市場に依存しないルール台帳を作った.</b> 過去の分析（去勢明け・母の年齢と初仔・乗り替わり・休養・馬具と替わり・コース・脚質と位置取り・買う要素の網羅探索 881 型）から、条件を共通の表の上で定義し直した {pre['counts']['n_rules']:,} 型（検定 {pre['counts']['n_tests']:,} 本）。単勝・複勝オッズ、人気、市場確率、想定オッズ、期待値に依存する条件・既存規則 {len(ex)} 型と {inv_ct.get('excluded_market', 0)} 件の出典は除外した（第 2 節）。回収率の測定にだけ確定払戻を使う。</li>
<li><b>採用されたルール（発見 2022–24、A=95%・B=90% 暫定、年をまたぐ再現）.</b> 期待値追求家の買い {rep['ev_buy']['n']}・消し {rep['ev_kill']['n']}、複勝率追求家の買い {rep['place_buy']['n']}・消し {rep['place_kill']['n']}。BH（FDR 10%）でも残る頑健なものは {rep['ev_buy']['robust']}・{rep['ev_kill']['robust']}・{rep['place_buy']['robust']}・{rep['place_kill']['robust']}。重なりを束ねたクラスタは {len(R['clusters']['ev_buy'])}・{len(R['clusters']['ev_kill'])}・{len(R['clusters']['place_buy'])}・{len(R['clusters']['place_kill'])}。</li>
<li><b>買い条件は後の年でも7割ほどが同じ向き、消し条件は半々に近い.</b> 2025 年に同じ向きだった採用ルールの割合は、期待値の買い {rep['ev_buy']['y2025']['same_sign']}/{rep['ev_buy']['y2025']['n']}、複勝率の買い {rep['place_buy']['y2025']['same_sign']}/{rep['place_buy']['y2025']['n']} に対し、期待値の消し {rep['ev_kill']['y2025']['same_sign']}/{rep['ev_kill']['y2025']['n']}、複勝率の消し {rep['place_kill']['y2025']['same_sign']}/{rep['place_kill']['y2025']['n']}（偶然なら半々）。2025 年に z が 1.645 を採用の向きへ越えた数（括弧は偶然の期待数）は、期待値の買い {rep['ev_buy']['y2025']['z_gt_1645']}（{rep['ev_buy']['y2025']['expected_null']}）、複勝率の買い {rep['place_buy']['y2025']['z_gt_1645']}（{rep['place_buy']['y2025']['expected_null']}）、期待値の消し {rep['ev_kill']['y2025']['z_gt_1645']}（{rep['ev_kill']['y2025']['expected_null']}）、複勝率の消し {rep['place_kill']['y2025']['z_gt_1645']}（{rep['place_kill']['y2025']['expected_null']}）。</li>
<li><b>ルールの合致で『来る・来ない』の順序は付くが小さい.</b> スコア帯別の3着内の残差（実績−モデル）は、2025 年で、期待値追求家 消し優勢 {rs(ev25b['neg']['resid_top3'])}pt・買い強 {rs(ev25b['pos_high']['resid_top3'])}pt、複勝率追求家 消し優勢 {rs(pl25b['neg']['resid_top3'])}pt・買い強 {rs(pl25b['pos_high']['resid_top3'])}pt。2026 年（v4 との差なので大きめに出る）では買い強 {rs(ev26b['pos_high']['resid_top3'])}pt・{rs(pl26b['pos_high']['resid_top3'])}pt。</li>
<li><b>ルールだけで選ぶと弱い.</b> スコア最上位グループの 2025 年の3着内率は 期待値追求家 {pc(t_ev25['top3_rate'][0])}%・複勝率追求家 {pc(t_pl25['top3_rate'][0])}%（同じレースのモデル本命 {pc(m_ev25['top3_rate'][0])}%）、単勝の回収率は {pc(t_ev25['roi_win'][0], 0)}%・{pc(t_pl25['roi_win'][0], 0)}%。ルールは『モデルとのずれ』を測る道具で、能力の順位ではない。</li>
<li><b>回収率は 100% を超えない.</b> 2025 年は、2 ペルソナ・3 定義・単勝と複勝のすべてで、区間の上限が {hi25 * 100:.0f}% 以下。2026 年（記述）は、{len(cr26)} 通りのうち {n_cross26} 通りで区間が 100% を跨ぐ（例: 期待値追求家の単勝 {pc(t_ev26['roi_win'][0], 0)}%、{pc(t_ev26['roi_win'][1], 0)}〜{pc(t_ev26['roi_win'][2], 0)}）が、点推定が 100% を超えるのは単勝の一部で、v4 の予測との差で選んだ群でもあり、確認にならない。</li>
<li><b>確率への補正に使うと 2025 年は悪化.</b> 合致したクラスタのロジットのずれを足した LogLoss の差は、2025 年で 期待値追求家 {ov25_ev['delta'] * 1e4:+.1f}×10⁻⁴、複勝率追求家 {ov25_pl['delta'] * 1e4:+.1f}×10⁻⁴（正が悪化）。発見期間では改善（選択に使ったので当然）、標本外では改善しない。</li>
<li><b>第 2 ラウンド（第 12・13 節）.</b> 原子条件の連言（2・3 条件）と、完全な母方 10 世代表での血統の系統の再測定。事前登録の基準だけで採用するとルールが数千個になるが標本外では偶然とほぼ同じで、BH 頑健なものだけが再現した。凍結ファイルは v3（BH 頑健な連言を追加）。順位付けの成績は第 1 ラウンドと同水準で、回収率は 100% を超えない。</li>
<li><b>結論.</b> 『ルールの合致率で馬を選ぶ』だけでは期待値は取れない。使い道は(1) 消し・買いの傾向を表示や監視の根拠にする、(2) 凍結して月次で再評価し、失効を除いていく（第 8 節）、の2つ。買い条件の複勝率側には標本外でも残る弱い信号があるが、モデルの確率を補正して使うには足りない。</li>
</ul></div>

<h2>1. 設計（評価の実行前に固定）</h2>
<ul>
<li><b>市場非依存.</b> 条件・順位付け・採用の判断に、単勝・複勝オッズ、人気、市場確率、オッズ由来の想定オッズ・期待値、オッズの変動、投票比率を使わない。条件の式は <code>market_free()</code> で検査し、ルールごと・凍結ファイルごとに印（market_free）を持つ。確定オッズ・払戻は回収率の測定（評価）にだけ使う。</li>
<li><b>基準 = 現行モデルの予測に対する残差（実績−予測）.</b> モデルは v6e（発見 2022–24 は LOYO の OOF、2025 は bundle_v6_2024 の test 予測）。市場の確率との差は使わない。2026 年は v4（bundle_v4_2025）で、記述のみ（v6e の 2026 年全体の予測は未生成）。発見期間が 2022–24 なのは、残差の基準になる OOF 予測がその年からしか無いため（指示の 2020–24 から、予測の無い 2020–21 を外した）。</li>
<li><b>ペルソナの意味.</b> <b>期待値追求家</b> = 過去統計から見て、モデルが勝ちを過小評価している条件に該当する馬を選ぶ（回収率はその結果を測る指標で、採用の基準ではない）。<b>複勝率追求家</b> = 3着以内に入りやすい条件（複勝率がモデルより高い）に該当する馬を選ぶ。どちらも消し（モデルが過大評価している条件）に当たる馬は避ける。</li>
<li><b>採用の基準.</b> 買い = 発見期間の残差の下側信頼限界 &gt; 0、消し = 上側信頼限界 &lt; 0。区間はレース単位のクラスタ頑健。A = 両側 95%、B = 両側 90%（暫定）。追加条件: 発見の3年のうち n≥100 の年が2年以上あり、過半が同じ向き（年をまたぐ再現）。検定可能 = 発見期間で300走・200レース以上かつレース単位でない条件。BH（FDR 10%）で頑健かどうかを併記（族 = 買い・消し × 勝ち・3着内の4つ、それぞれ検定可能なルール全体）。回収率の区間は参考表示で、採用には使わない。</li>
<li><b>重複の整理.</b> 発見期間の該当行の Jaccard≥0.5 または包含率≥0.9 のルールは同じクラスタ（代表 = z 最大）にまとめ、1頭が数えるのはクラスタごとに1回。</li>
<li><b>合致率（スコア）の定義（3通りを事前に固定）.</b> a = 買いクラスタの合致数 − 消しクラスタの合致数、<b>b = 縮約した効果（pt、n/(n+300)）の合計（買い−消し）【主】</b>、c = 最強の買い − 最強の消し。レースごとにスコア最大（&gt;0）の馬（同点は全員）を選び、各馬 1 単位を賭けた場合の成績を測る。消しの扱いは『除外』『減点（スコアに含む）』の2通り。</li>
<li><b>期間の使い分け.</b> 選択は発見 2022–24 のみ。2025 は選んだ規則集合の評価 1 回。2026 は記述統計のみ（選択に使わない）。</li>
</ul>

<h2>2. ルール台帳: 出典の棚卸し</h2>
<p>docs/html/modeling/chapters/ の全 {sum(1 for i in INVENTORY if i['kind'] == 'chapter')} 章、ホストの <code>*_out/</code>、リポジトリ内の分析スクリプトと設定について、『ルール化できる統計があるか』と、取り込んだ／取り込まなかった理由を記録した（出典ごと。下の折りたたみ。CSV: <code>docs/operations/tp75_chapter_inventory.csv</code>）。内訳: 再計算して台帳に入れた {inv_ct.get('recomputed', 0)}、市場依存のため除外 {inv_ct.get('excluded_market', 0)}、比較の基準 {inv_ct.get('baseline', 0)}、取り込まなかった {inv_ct.get('none', 0)}（設計・仕様・モデル構成の章など、条件別の統計表を持たないもの）。</p>
<p><b>ルールの族.</b> T72 = 買う要素の網羅探索の事前登録 {pre['counts']['n_rules_tp72']} 型（市場由来の前走人気の9型を除く）、G = 去勢明け N 戦目、D = 母の出産時年齢・初仔・種牡馬の供用年、J = 乗り替わりの型、R = 休養日数、E = 馬具・替わり、C = コース、P = 脚質・位置取り・前走の上がり、H = 見出しの条件×面。新規の {pre['counts']['n_rules_new']} 型は『既存の分析で測られた条件を共通の表で定義し直したもの』で、結果を見る前に登録した。</p>
<table><tr><th>族</th><th>型の数</th></tr>{fam_rows}</table>
<h3>市場依存のため除外したもの</h3>
<ul>
<li>印の規則（◎勝率順位は比較の基準にだけ使い、▲単穴・★期待値はオッズ由来）、推奨ポートフォリオ（bet_optimization）、paid_zone の B/C（想定オッズ・実オッズ）、tp-36 の混合 EV（Benter 型）、EV 較正・確率の市場補正、tp-67 の消印（人気と複勝確率）、tp-44 の人気帯・『函館の3番人気』、tp-56 の穴馬（人気薄）、tp-59 の凍結規則（EV≥1.5）。</li>
<li>市場に依存しない部分（paid_zone の A＝確信度）は、レース単位の指標で馬の条件ではないため台帳に入れなかった。</li>
<li>穴馬（人気薄）を条件にする分析結果は、ルールにしていない。</li>
</ul>
<table><tr><th>除外した型</th><th>条件</th><th>理由</th></tr>{exrows}</table>
<details><summary>出典の棚卸し（{len(INVENTORY)} 件）</summary>
<table><tr><th>出典</th><th>取り込み</th><th>族</th><th>ルール化できる統計</th><th>理由</th></tr>{irows}</table></details>

<h2>3. 採用されたルール</h2>
<table><tr><th>族</th><th>検定可能</th><th>発見で z&gt;1.645（再現前）</th><th>A</th><th>B</th><th>BH 頑健</th><th>クラスタ</th><th>2025 同符号</th><th>2025 で z&gt;1.645 の向き</th><th>頑健のうち 2025 同符号</th><th>2026 同符号</th></tr>{arows}</table>
<p class="muted">『発見で z&gt;1.645』は検定可能 {s['ev_buy']['testable']} 型のうちの数で、偶然なら約 5%（{s['ev_buy']['expected_false_pos_at_1.645']:.0f} 型）。それを超える分は、共通の原因（性別×季節など）で多数のルールが同時に動くためでもある（クラスタが {len(R['clusters']['ev_buy'])} に縮むのはその表れ）。</p>
<img alt="採用数と再現" style="max-width:100%" src="data:image/png;base64,{imgs['counts']}" />
<img alt="発見と2025のz" style="max-width:100%" src="data:image/png;base64,{imgs['scatter']}" />
<h3>上位のルール（発見の z が大きい順）</h3>
<p class="muted">残差 = 実績 − 現行モデルの予測（pt）。z（2025・2026）は採用の向きを正にしてある。全ルールの一覧は <code>docs/operations/tp75_rule_ledger.csv</code>。</p>
""" + "".join(f"<h3>{FAM_JA[f]}</h3>" + rule_table(ledger, R["clusters"], f) for f in FAM) + f"""

<h2>4. 合致率による順位付けと成績</h2>
<h3>4.1 スコア帯別の残差（順序性）</h3>
<table><tr><th>ペルソナ</th><th>期間</th><th>スコア b の帯</th><th>走数</th><th>勝ちの残差（pt）</th><th>3着内の残差（pt）</th><th>単勝の回収率 %</th><th>複勝の回収率 %</th></tr>{brows}</table>
<img alt="スコア帯別の残差" style="max-width:100%" src="data:image/png;base64,{imgs['bins']}" />
<h3>4.2 スコア最上位グループを選んだ場合</h3>
<p class="muted">各レースでスコアが最大（&gt;0）の馬（同点は全員）に1単位ずつ。ルールに当たる馬がいないレースは選ばない（選んだレース数を併記）。右の3列は『同じレースのモデル本命（勝率最大）』。定義 a/b/c の比較: b（効果の合計）が主、a（合致数）、c（最強の根拠）。</p>
<table><tr><th>ペルソナ</th><th>定義</th><th>期間</th><th>選んだレース</th><th>勝率 %</th><th>3着内率 %</th><th>モデルの複勝確率の平均 %</th><th>単勝の回収率 %</th><th>複勝の回収率 %</th><th>モデル本命の勝率 %</th><th>同 3着内率 %</th><th>同 単勝回収率 %</th></tr>{trows}</table>
<img alt="最上位グループの成績" style="max-width:100%" src="data:image/png;base64,{imgs['top']}" />

<h2>5. 消し条件の効果</h2>
<p>消しのクラスタに当たる馬の扱いを2通り見た。(1) 複勝確率の上位3頭のうち、消しに当たる馬と当たらない馬の3着内率、(2) モデル本命（勝率最大）が消しに当たるとき、それを除いて次の馬を選ぶ場合。</p>
<table><tr><th>ペルソナ</th><th>期間</th><th>上位3頭・消しに当たる 頭数</th><th>3着内率 %</th><th>上位3頭・消しに当たらない 頭数</th><th>3着内率 %</th><th>モデル本命（全レース）</th><th>3着内率 %</th><th>消しを除いた本命</th><th>3着内率 %</th></tr>{krows}</table>
<ul>
<li>消しに当たる馬の3着内率は、当たらない馬より 2025 年で約 {abs((P['personas']['ev_seeker']['kill_filter/y2025/top3_by_ptop3_other']['top3_rate'][0] - P['personas']['ev_seeker']['kill_filter/y2025/top3_by_ptop3_kill']['top3_rate'][0]) * 100):.1f}pt（期待値追求家の消し）低い。ただし<b>消しのクラスタは広い</b>（性別×季節など、発見期間の行の 4 割近くに当たるものがある）ため、2025 年のモデル本命のうち {share_kill * 100:.0f}% が期待値追求家の消しに当たり、『消しを除いた本命』は選択的でない（別の馬に替わると 3着内率が大きく下がる）。</li>
<li>後の年でも同じ向きの消しルールは、2025 年で 半分前後（上の第3節）。消し条件は『この条件は来ない』と言い切れる強さではなく、表示用の目安にとどまる。</li>
</ul>

<h2>6. 確率への補正として使う場合</h2>
<p>合致したクラスタごとに、代表ルールの発見期間でのロジットのずれ（n/(n+300) で縮約）を現行モデルの確率に足した。LogLoss の差（×10⁻⁴、負が改善。[ ] は95%区間）。2026 年は v4 の予測にそのまま当てた参考値。</p>
<table><tr><th>集合</th><th>ペルソナ</th><th>発見（選択に使用）</th><th>2025</th><th>2026（v4・参考）</th></tr>{orow}</table>
<p>発見期間では改善する（選択に使ったので当然）が、2025 年は悪化または改善なし。標本外で確率を補正できるほどの強さはない。</p>

<h2>7. 2つのペルソナの一致・食い違い</h2>
<table><tr><th>期間</th><th>区分</th><th>走数</th><th>3着内率 %</th><th>勝率 %</th><th>3着内の残差（pt）</th><th>単勝の回収率 %</th><th>複勝の回収率 %</th></tr>{xrows}</table>
<ul>
<li>両方が買いの馬は、3着内率が高く（2025 年 {pc(P['cross']['y2025']['both_pos']['top3_rate'][0])}%、残差 {rs(P['cross']['y2025']['both_pos']['resid_top3'])}pt）、どちらも買いでない馬は低い。<b>片方だけ買い</b>の馬は残差が 0 付近で、2つの買いが食い違うときは信頼できない。</li>
<li>2つの消しは重なりが大きい（消しの族は同じ『3着内/勝ちの過大評価』を別の列で測る）。</li>
<li>モデルの複勝確率の上位3頭のうち、両方が買いの馬は、そうでない馬より 3着内率が {pc(xa['top3_rate'][0] - xb['top3_rate'][0])}pt 高い（2025）が、複勝の回収率は {pc(xa['roi_place'][0], 0)}% と {pc(xb['roi_place'][0], 0)}% で差がない（市場が同じ差を織り込んでいる）。</li>
</ul>

<h2>8. 規則集合のバージョン管理と月次の再評価（運用案）</h2>
<ul>
<li><b>凍結ファイル.</b> <code>config/tp75_frozen_rule_ev.json</code>（期待値追求家）と <code>config/tp75_frozen_rule_place.json</code>（複勝率追求家）。ペルソナごとに分け、version・frozen_on・基準モデル・採用基準・スコアの定義・クラスタ・ルール（条件の式、出典、向き、区分 A/B、BH 頑健、発見の統計、2025・2026 の z、market_free 印）・評価の要約・history を持つ。v1 は発見期間で採用された（A・B）ルールすべて。使う側は tier=A かつ robust_bh を優先する。</li>
<li><b>追加・失効の履歴（history）.</b> 追加は『別の事前登録（型の数を記録）で検定し、発見期間で採用されたものだけ新版へ』。失効はルールを rules から外し、history に日付・理由・統計を残して version を上げる。結果を見てから条件を増やさない原則は変えない。</li>
<li><b>月次の再評価.</b> <code>python -m src.scripts.data.tp75_rules reeval config/tp75_frozen_rule_ev.json --years 2026</code>。凍結後に新しく出たレースでルールごとの z（採用の向きを正）を計算する。状態: ok（z≥0）／watch（z&lt;0）／failing（z&lt;−1.645）。n が 300 走に満たないルールは判定を保留。failing が2か月続く、または凍結後の累積で z&lt;−1.645 になったら失効に移す。</li>
<li><b>前向き追跡.</b> 既存の週次追跡（<code>scripts/cron/frozen_tracking_weekly.sh</code>、tp-59・tp-72 の凍結規則）に載せる案: 月次の reeval の結果を同じ枠に出す。cron の登録はしていない（承認後）。</li>
<li><b>注意.</b> 『2026 年に作った予測』は v4 の予測のため、v6e での再確認（v6e の 2026 年全体の予測の生成後）が先に要る。</li>
</ul>

<h2>9. 事後の感度分析（結果を見た後に追加したもの）</h2>
<p>主の結果では、性別×季節などの広いクラスタ（発見期間の行の 25〜70% に当たる）が多く、『消しに当たらない馬』を選ぶ操作が選択的でなかった。そこで、<b>事後に</b>、発見期間の行の 15% 以下にしか当たらない（特定の）クラスタだけを使う感度分析を足した（事前登録の外。結果を見てから足したことを明記する）。クラスタ数: 期待値の買い {N['personas']['ev_seeker']['n_buy_clusters']}・消し {N['personas']['ev_seeker']['n_kill_clusters']}、複勝率の買い {N['personas']['place_seeker']['n_buy_clusters']}・消し {N['personas']['place_seeker']['n_kill_clusters']}。スコア帯別の残差や補正の結論は主と同じ向きで、補正の LogLoss は 2025 年で 期待値追求家 {N['personas']['ev_seeker']['overlay']['y2025']['delta'] * 1e4:+.1f}×10⁻⁴、複勝率追求家 {N['personas']['place_seeker']['overlay']['y2025']['delta'] * 1e4:+.1f}×10⁻⁴（正が悪化）。広いクラスタを除いても、標本外で確率を補正する力は出なかった。</p>

<h2>10. 他のペルソナとの役割の違い</h2>
<ul>
<li>ラップ・馬場質・能力・血統・トラックバイアスの各ペルソナ（<a href="persona_ensemble_survey.html">persona_ensemble_survey.html</a>）は<b>馬の力・適性を見積もる側</b>（確率を出す）で、現行の単一モデルがほぼ内包している。<b>期待値追求家・複勝率追求家は、モデルが外す傾向を過去の統計から拾う側</b>（確率の補正・消しの根拠）で、モデルの出力を前提にする。</li>
<li>不一致が出る場面: 能力のペルソナが本命でも、ルールが『消し』に当たる（例: 去勢明け1戦目、休養明けで馬体重が大きく減った古馬、夏の牡馬）。このとき実績はモデルより低くなる傾向があるが、その差は 1〜3pt で、本命を覆すほどではない（第5節）。逆にルールが買いに当たる穴の馬は、残差は正でも回収率は 100% に届かない。</li>
</ul>

<h2>11. 限界</h2>
<ul>
<li>基準モデルは v6e（発見 2022–24 は LOYO の OOF、2025 は bundle_v6_2024）。LOYO の OOF は学習に将来の年を含むため、発見期間の残差の構造は『将来を知っているモデル』に対するものになる。2025 年の標本外で消しの再現が弱い一因かもしれない。</li>
<li>2026 年は v4 の予測のため、v6e で再確認が必要。</li>
<li>馬主・地方や海外帰り・展開利・追い切りの標準化など、共通の表に無い条件は台帳に入っていない（第2節の理由）。次の版で結合を検討する。</li>
<li>主な採用ルールは性別×季節など広い条件に偏る。個別の特定の条件（去勢明け1戦目・母の年齢など）は検定可能でも、後の年での再現は半々。</li>
<li>多数の検定（{pre['counts']['n_tests']:,} 本）の中で、BH で頑健なのは一部。B（90%）は暫定で、使う側は A かつ頑健を優先する。</li>
</ul>
</body></html>
"""
    extra_html = ""
    if (src / "results_r2.json").exists() and (src / "results_r2b.json").exists():
        from src.scripts.data import tp75b_report as RB

        R2, L2 = RB.load(src)
        X = RB.load_extra(src)
        RB.write_all(src)
        extra_html += RB.section_r2(R, R2, L2, X)
        if X["rc"] and X["rcp"] and X["lrc"]:
            extra_html += RB.section_rc(R, R2, X)
    html_doc = html_doc.replace("</body></html>", extra_html + "</body></html>")
    CH.write_text(html_doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    chapter(Path(a.src))
    print("wrote", CH, *FROZEN.values(), LEDGER_CSV, INV_CSV, sep="\n  ")


if __name__ == "__main__":
    main()
