"""tp-48 の続き: 的中率が高い券種（複勝・ワイド・馬連）の『期待値が高いものだけ買う』を、未使用の 2026 年で 1 回だけ評価する。

python -m src.scripts.data.tp48_2026 register                      # 規則を固定してハッシュを記録（2026 年を見る前に実行し、コミットする）
python -m src.scripts.data.tp48_2026 run --out <dir>/eval_2026.json  # 2024〜2026 を同じ処理で評価（ホストで実行）。2026 は 1 回だけ（出力があれば拒否）
python -m src.scripts.data.tp48_2026 report --json <eval_2026.json>  # 章 trifecta_true_odds.html に節を挿入

入力: data/local/modeling/betting/tp36/frame_all.parquet（2026 年は複勝オッズ・複勝払戻が無い）に、
data/local/modeling/betting_2026/{odds_win_place,odds_pair}.parquet の 2026 年の確定オッズを足す。払戻は race_result_flat（補正行つき）。
EV の式は tp-47 の修正後（複勝は下限 + 0.3 ×（上限 − 下限）、馬連・ワイドは実オッズ）。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp36_single as t36
from src.scripts.data import tp48_eval as te

ROOT = Path(__file__).resolve().parents[3]
RULES_FILE = ROOT / "src/scripts/data/tp48_2026_rules.json"
B26 = ROOT / "data/local/modeling/betting_2026"
CHAPTER = ROOT / "docs/html/modeling/chapters/trifecta_true_odds.html"
MARK_BEGIN, MARK_END = "<!-- tp48-2026 begin -->", "<!-- tp48-2026 end -->"

# 2026 年を見る前に固定する規則（tp48_eval の方策名と同じ）。ユーザーの質問『的中率が高い券種を期待値 1.2 以上だけ買う』に直結するもの。
RULES = [
    {"id": "F1", "policy": "fukusho|model|ev>=2.0|top5", "note": "複勝: 市場なしモデルの EV≥2.0（tp-48 で最も近かった規則: 2024〜25 年 113%）"},
    {"id": "F2", "policy": "fukusho|model|ev>=1.5|top5", "note": "複勝: モデル EV≥1.5"},
    {"id": "F3", "policy": "fukusho|model|ev>=1.2|top5", "note": "複勝: モデル EV≥1.2（ユーザーの質問の水準）"},
    {"id": "F4", "policy": "fukusho|model|shrunk1.05(raw>=1.84)|top50", "note": "複勝: 縮約後 EV≥1.05 に相当する生の EV≥1.84（縮約係数 k=0.24）"},
    {"id": "F5", "policy": "fukusho|mixed|ev>=1.2|top5", "note": "複勝: 混合（Benter 型）EV≥1.2"},
    {"id": "F6", "policy": "fukusho|mixed|ev>=1.0|top5", "note": "複勝: 混合 EV≥1.0"},
    {"id": "W1", "policy": "wide|model|top1", "note": "ワイド: モデル EV 最大の 1 点"},
    {"id": "W2", "policy": "wide|mixed|top1", "note": "ワイド: 混合 EV 最大の 1 点"},
    {"id": "W3", "policy": "wide|model|ev>=2.0|top1", "note": "ワイド: モデル EV 最大の 1 点が EV≥2.0 のときだけ"},
    {"id": "U1", "policy": "umaren|model|top1", "note": "馬連: モデル EV 最大の 1 点"},
    {"id": "U2", "policy": "umaren|mixed|top1", "note": "馬連: 混合 EV 最大の 1 点"},
    {"id": "U3", "policy": "umaren|model|ev>=2.0|top1", "note": "馬連: モデル EV 最大の 1 点が EV≥2.0 のときだけ"},
]


# ───────────────────────── 規則の固定 ─────────────────────────
def rules_payload(registered_at: str | None = None) -> dict:
    return {"registered_at": registered_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "2026 年（未使用）の評価。評価前にこの一覧を固定する。各方策名は tp48_eval の policy_rows と同じ。",
            "periods": {"design": "2024〜2025（参考）", "final": "2026（1 回だけ）"}, "rules": RULES}


def rules_hash(payload: dict) -> str:
    body = json.dumps({"rules": payload["rules"], "periods": payload["periods"]}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def cmd_register(a) -> None:
    p = rules_payload()
    p["hash"] = rules_hash(p)
    RULES_FILE.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    print("registered", RULES_FILE, p["hash"], len(RULES), "rules")


# ───────────────────────── 2026 年の入力 ─────────────────────────
def attach_2026_place(frame: pd.DataFrame, hits: dict) -> pd.DataFrame:
    """2026 年の行に、確定オッズの複勝の幅と複勝の払戻（100 円あたり）を足す。"""
    ow = pd.read_parquet(B26 / "odds_win_place.parquet", columns=["race_id", "horse_number", "place_odds_min", "place_odds_max"])
    ow = ow.drop_duplicates(["race_id", "horse_number"])
    m = frame["year"] == 2026
    f26 = frame[m].drop(columns=["place_odds_min", "place_odds_max"]).merge(ow, on=["race_id", "horse_number"], how="left")
    rp = []
    for rid, hn in zip(f26["race_id"], f26["horse_number"]):
        rp.append(hits.get(str(rid), {}).get("fukusho", {}).get((int(hn),), 0.0))
    f26["ret_place"] = rp
    return pd.concat([frame[~m], f26], ignore_index=True)


def load_pair_real_from(path: Path, race_ids: set, year: int | None) -> dict:
    pair = pd.read_parquet(path, columns=["race_id", "bet_type", "combo", "odds", "odds_min", "odds_max"])
    if year is not None:
        pair = pair[(pair["race_id"] >= f"{year}") & (pair["race_id"] < f"{year + 1}")]
    pair = pair[pair["race_id"].isin(race_ids) & pair["bet_type"].isin(["umaren", "wide"])].copy()
    pair["o"] = np.where(pair["bet_type"] == "wide", te.place_adj(pair["odds_min"], pair["odds_max"]), pair["odds"])
    pair = pair[pair["o"] > 0]
    out: dict = {}
    for (rid, bt), g in pair.groupby(["race_id", "bet_type"], sort=False):
        out.setdefault(rid, {})[bt] = {t36.parse_combo(c): float(o) for c, o in zip(g["combo"], g["o"])}
    return out


# ───────────────────────── 評価 ─────────────────────────
def evaluate_policy(stake: np.ndarray, ret: np.ndarray, mask: np.ndarray, n_boot: int = 1000) -> dict:
    s, r = stake[mask], ret[mask]
    bets = int(s.sum() / 100)
    if s.sum() <= 0:
        return {"roi": None, "lo": None, "hi": None, "bets": 0, "races_bet": 0, "hit_race_rate": None, "roi_without_top3": None}
    roi, lo, hi = te.boot(s, r, n_boot)
    bet_idx = np.where(s > 0)[0]
    rb = r[bet_idx]
    top3 = np.argsort(-rb)[:3]
    keep = np.ones(len(bet_idx), bool)
    keep[top3] = False
    ss = s[bet_idx]
    return {"roi": float(roi), "lo": float(lo), "hi": float(hi), "bets": bets, "races_bet": int(len(bet_idx)), "hit_race_rate": float((rb > 0).mean()),
            "roi_without_top3": float(rb[keep].sum() / ss[keep].sum()) if ss[keep].sum() > 0 else None}


def cmd_run(a) -> None:
    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} が既にある。2026 年の評価は 1 回だけ（再実行は --force を明示）")
    payload = json.loads(RULES_FILE.read_text(encoding="utf-8"))
    h = rules_hash(payload)
    if h != payload.get("hash"):
        raise SystemExit("規則のハッシュが固定時と違う（規則を変えていないか確認）")
    print("rules hash", h, "registered_at", payload["registered_at"], flush=True)

    frame = pd.read_parquet(te.BET / "tp36/frame_all.parquet")
    frame = frame[frame["year"] >= 2022].copy()
    inv = (1.0 / frame["win_odds"].where(frame["win_odds"] >= 1.0)).groupby(frame["race_id"]).sum()
    bad_odds = set(inv[~inv.between(1.0, 1.6)].index) | set(frame.loc[~(frame["win_odds"] >= 1.0), "race_id"])
    frame = frame[~frame["race_id"].isin(bad_odds)]
    hits, bad_pay = te.load_hits_all()
    pay26 = pd.read_parquet(B26 / "payoff.parquet")
    bad_pay = set(bad_pay) | set(pay26.loc[pay26["ambiguous"] | ~pay26["parsed"], "race_id"])
    frame = frame[~frame["race_id"].isin(bad_pay)]
    frame = attach_2026_place(frame, hits)
    pair22 = t36.load_pair_real(set(frame.loc[frame["year"] == 2022, "race_id"]), 2022)
    alpha_m = t36.fit_market_alpha(frame[frame["year"] == 2022], pair22)
    del pair22
    recs, meta = [], []
    for year in (2024, 2025, 2026):
        dy = frame[frame["year"] == year]
        ids = set(dy["race_id"])
        pair_real = te.load_pair_real2(ids, year) if year <= 2025 else load_pair_real_from(B26 / "odds_pair.parquet", ids, None)
        for k, (rid, g) in enumerate(dy.groupby("race_id", sort=False)):
            r = te.process_race(g, hits.get(rid, {}), None, pair_real.get(rid), alpha_m, False)
            recs.append(r)
            meta.append((rid, str(g["date"].iloc[0]), year))
            if k % 500 == 0:
                print(year, k, flush=True)
        del pair_real
    order = sorted(range(len(meta)), key=lambda i: (meta[i][1], meta[i][0]))
    recs = [recs[i] for i in order]
    meta = [meta[i] for i in order]
    years = np.array([m[2] for m in meta])
    store: dict = {}
    for bt in ("fukusho", "wide", "umaren"):
        for sch in ("model", "mixed"):
            ev, od, py = te.fam_arrays(recs, lambda r, bt=bt, sch=sch: r.get("E", {}).get((bt, sch)), t36.KMAX)
            okm = np.isfinite(ev[:, :1]).any(1)
            te.policy_rows(f"{bt}|{sch}", ev, py, years, okm, bt, store, [], [])
    res: dict = {"hash": h, "registered_at": payload["registered_at"], "races": {str(y): int((years == y).sum()) for y in (2024, 2025, 2026)}, "rules": []}
    P26 = []
    for rule in RULES:
        st = store.get(rule["policy"])
        if st is None:
            raise SystemExit("方策が無い: " + rule["policy"])
        stake, ret = st
        row = {**rule}
        for lab, m in (("y2024", years == 2024), ("y2025", years == 2025), ("design", (years == 2024) | (years == 2025)), ("y2026", years == 2026)):
            row[lab] = evaluate_policy(stake, ret, m)
        res["rules"].append(row)
        P26.append((ret - stake)[years == 2026])
    P = np.stack(P26, 1)
    ok = P.std(0) > 0
    rc = te.reality_check(P[:, ok], n_boot=1000)
    res["reality_check_2026"] = {"t_max": rc["t_max"], "p_value": rc["p_value"], "n_strategies": rc["n_strategies"],
                                  "best": [r["id"] for r, o in zip(RULES, ok) if o][rc["best"]]}
    res["place_odds_coverage_2026"] = float(frame.loc[frame["year"] == 2026, "place_odds_min"].notna().mean())
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


# ───────────────────────── 章への節 ─────────────────────────
def pc(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.0f}%"


def cell_ev(d: dict | None) -> str:
    if not d or d.get("roi") is None:
        return "<td class='num'>—</td>"
    return f"<td class='num'>{pc(d['roi'])}<br><span class='muted'>[{pc(d['lo'])}–{pc(d['hi'])}] {d['bets']:,}点</span></td>"


def section_html(res: dict) -> str:
    rows = []
    for r in res["rules"]:
        y26 = r["y2026"]
        rows.append(f"<tr><td>{html.escape(r['id'])}</td><td>{html.escape(r['policy'])}<br><span class='muted'>{html.escape(r['note'])}</span></td>"
                    + cell_ev(r["y2024"]) + cell_ev(r["y2025"]) + cell_ev(y26)
                    + f"<td class='num'>{pc(y26.get('hit_race_rate'))}</td><td class='num'>{pc(y26.get('roi_without_top3'))}</td></tr>")
    rc = res["reality_check_2026"]
    best = max((r for r in res["rules"] if r["y2026"]["lo"] is not None), key=lambda r: r["y2026"]["lo"], default=None)
    over = [r["id"] for r in res["rules"] if r["y2026"]["lo"] is not None and r["y2026"]["lo"] > 1.0]
    concl = ("区間下限が 100% を超えた規則は " + "・".join(over) + "。" if over else "<b>区間下限が 100% を超えた規則は無かった。</b>")
    if best:
        concl += f"下限が最も高いのは {best['id']}（{pc(best['y2026']['lo'])}）。"
    return (f"{MARK_BEGIN}\n<h2>2026 年の最終評価（未使用データ、1 回だけ）</h2>\n"
            f"<p>的中率が高い券種（複勝・ワイド・馬連）で『期待値が高いものだけ買う』を、2026 年の確定オッズ（複勝の幅・馬連・ワイド）が揃ったので、"
            f"<b>評価前に規則 {len(res['rules'])} 本を固定して</b>（固定時刻 {html.escape(res['registered_at'])}、ハッシュ {res['hash']}。リポジトリの <code>tp48_2026_rules.json</code> にコミット済み）、"
            f"2026 年を 1 回だけ評価した。予測は 2026 年を見ない V4（bundle_v4_2025）。オッズは確定後の値で、購入時点の値ではない。"
            f"2026 年は評価できたレース {res['races']['2026']:,}（2024 年 {res['races']['2024']:,}、2025 年 {res['races']['2025']:,}）。</p>\n"
            "<table><tr><th>ID</th><th>規則</th><th>2024</th><th>2025</th><th>2026（1 回）</th><th>2026 的中レース率</th><th>2026 上位 3 的中を除く</th></tr>"
            + "".join(rows) + "</table>\n"
            f"<p>{concl} 12 規則に対する Reality Check（2026 年）は p = {rc['p_value']:.2f}（最大の t = {rc['t_max']:.2f}）。</p>\n"
            "<p class='muted'>2026 年は、モデルが市場に大きく負けた年（モデルと市場の差 0.100、2022〜2025 年は 0.036〜0.075）。"
            "この年の結果は『混合の効果が 2026 年では再現しなかった』という弱い検証であり、規則そのものの良し悪しの決定打にはならない。</p>\n"
            f"{MARK_END}\n")


def cmd_report(a) -> None:
    res = json.loads(Path(a.json).read_text(encoding="utf-8"))
    sec = section_html(res)
    t = CHAPTER.read_text(encoding="utf-8")
    if MARK_BEGIN in t:
        t = re.sub(re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END) + r"\n?", lambda m: sec, t, flags=re.S)
    else:
        i = t.rfind("</body>")
        t = t[:i] + sec + t[i:] if i >= 0 else t + sec
    CHAPTER.write_text(t, encoding="utf-8")
    print("updated", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "run", "report"])
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"register": cmd_register, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
