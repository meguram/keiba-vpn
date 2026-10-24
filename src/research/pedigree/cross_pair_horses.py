"""指定の (父 × 母父系統) ペアに含まれる馬群を列挙し、
   ペア lift と各馬の実戦績を比較して妥当性を検証する。"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
IDX_DIR = ROOT / "data/page_reference/pedigree_race_index"
ART_DIR = ROOT / "data/page_reference/note_aptitude_race"

DIST_BINS = [0, 1400, 1800, 2400, 9999]
DIST_LABELS = ["短距離","マイル","中距離","長距離"]
STEEP_VENUES = ("中山","阪神","中京")
EB_PRIOR = 30
GLOBAL_WIN = 0.139


def _eb(w, n):
    if n < 5: return None
    return (w + EB_PRIOR * GLOBAL_WIN) / (n + EB_PRIOR)


def main():
    print("[load]")
    bms = pd.read_parquet(IDX_DIR / "horse_bms.parquet", columns=["horse_id","sire_id","sire_name","bms_id","bms_name"])
    bms["horse_id"] = bms["horse_id"].astype(str)
    bms["sire_id"] = bms["sire_id"].astype(str)
    bms["bms_id"] = bms["bms_id"].astype(str)

    race = pd.read_parquet(IDX_DIR / "race_result_slim.parquet")
    race["horse_id"] = race["horse_id"].astype(str)
    race = race[race["finish_position"].notna() & (race["finish_position"] > 0)].copy()
    race["win"] = (race["finish_position"] == 1).astype(int)
    race["dist_cat"] = pd.cut(race["distance"], bins=DIST_BINS, labels=DIST_LABELS).astype(str)
    race["is_steep"] = race["venue"].astype(str).isin(STEEP_VENUES)
    race["is_heavy"] = race["track_condition"].astype(str).isin(["重","不良"])

    sid_to_main = json.loads((ART_DIR / "sid_to_main.json").read_text(encoding="utf-8"))
    sid_to_name: dict[str,str] = {}
    for fn in ("sid_to_name_full.json","sid_to_name.json"):
        p = ART_DIR / fn
        if p.exists():
            sid_to_name.update(json.loads(p.read_text(encoding="utf-8")))

    bms["sire_main"] = bms["sire_id"].map(sid_to_main)
    bms["bms_main"] = bms["bms_id"].map(sid_to_main)

    # 馬名マスタ
    horse_name_df = race[["horse_id","horse_name"]].dropna().drop_duplicates("horse_id").set_index("horse_id")["horse_name"]
    horse_name = horse_name_df.to_dict()

    print(f"  bms={len(bms):,}, race={len(race):,}")

    def _profile_for(horse_ids: set[str]) -> dict[str, float] | None:
        mask = race["horse_id"].isin(horse_ids)
        sub = race[mask]
        if len(sub) < 30: return None
        eb_total = _eb(sub["win"].sum(), len(sub))
        if eb_total is None: return None
        out = {"_n_total": int(len(sub)), "_n_horses": int(len(horse_ids)), "_eb_total": float(eb_total)}
        # venue
        for v in ["東京","中山","阪神","京都","中京","新潟","小倉","福島","札幌","函館"]:
            sc = race[mask & (race["venue"].astype(str) == v)]
            e = _eb(sc["win"].sum(), len(sc))
            if e is not None:
                out[f"win_v_{v}"] = e / eb_total
                out[f"n_v_{v}"] = int(len(sc))
        for d in DIST_LABELS:
            sc = race[mask & (race["dist_cat"] == d)]
            e = _eb(sc["win"].sum(), len(sc))
            if e is not None:
                out[f"win_d_{d}"] = e / eb_total
                out[f"n_d_{d}"] = int(len(sc))
        sc = race[mask & race["is_heavy"]]
        e = _eb(sc["win"].sum(), len(sc))
        if e is not None:
            out["win_heavy"] = e / eb_total
            out["n_heavy"] = int(len(sc))
        return out

    def _horse_record(horse_id: str) -> dict[str, dict]:
        sub = race[race["horse_id"] == horse_id]
        if sub.empty: return {}
        rec: dict[str, dict] = {"total": {"n": int(len(sub)), "win": int(sub["win"].sum()),
                                          "win_rate": float(sub["win"].mean())}}
        for v in ["東京","中山","阪神","京都"]:
            s = sub[sub["venue"].astype(str) == v]
            if len(s) > 0:
                rec[f"v_{v}"] = {"n": int(len(s)), "win": int(s["win"].sum()),
                                  "win_rate": float(s["win"].mean())}
        for d in DIST_LABELS:
            s = sub[sub["dist_cat"] == d]
            if len(s) > 0:
                rec[f"d_{d}"] = {"n": int(len(s)), "win": int(s["win"].sum()),
                                  "win_rate": float(s["win"].mean())}
        s = sub[sub["is_heavy"]]
        if len(s) > 0:
            rec["heavy"] = {"n": int(len(s)), "win": int(s["win"].sum()),
                              "win_rate": float(s["win"].mean())}
        return rec

    # ────────────────────────────────────────────
    # ターゲット: ドゥラメンテ × 各母父系統
    # ────────────────────────────────────────────
    SIRE_ID = "2012104511"   # ドゥラメンテ
    SIRE_NAME = sid_to_name.get(SIRE_ID, "ドゥラメンテ")

    print(f"\n══════════════════════════════════════════════")
    print(f"  父 = {SIRE_NAME} ({SIRE_ID})")
    print(f"══════════════════════════════════════════════")

    sub_sire = bms[bms["sire_id"] == SIRE_ID]
    print(f"  ドゥラメンテ産駒 総数: {len(sub_sire)} 頭")
    print(f"  母父系統別の頭数: {dict(sub_sire['bms_main'].value_counts(dropna=False))}")

    MAIN_GROUPS = ["Turn-To系", "Native Dancer系", "Northern Dancer系", "Nasrullah系", "非主流"]

    # ペア lift
    print(f"\n  ── ペア集団 lift ──")
    print(f"  {'母父系統':<22} {'n_horses':>10} {'n_records':>10} {'阪神':>6} {'東京':>6} {'長距離':>7} {'重':>6}")
    pair_profiles: dict[str, dict] = {}
    for mg in MAIN_GROUPS + [None]:
        if mg is None:
            continue  # NaN の母父は飛ばす
        horses = set(sub_sire[sub_sire["bms_main"] == mg]["horse_id"])
        prof = _profile_for(horses)
        if prof is None: continue
        pair_profiles[mg] = prof
        print(f"  {mg:<22} {prof['_n_horses']:>10} {prof['_n_total']:>10} "
              f"{prof.get('win_v_阪神', float('nan')):>6.2f} {prof.get('win_v_東京', float('nan')):>6.2f} "
              f"{prof.get('win_d_長距離', float('nan')):>7.2f} {prof.get('win_heavy', float('nan')):>6.2f}")

    # ── 「ドゥラメンテ × 非主流母父」集団の馬リスト ──
    print(f"\n  ── ドゥラメンテ × 非主流母父 の所属馬 (出走数 上位 30) ──")
    target_set = sub_sire[sub_sire["bms_main"] == "非主流"]
    horses_in_pair = set(target_set["horse_id"])
    # 各馬の出走数 + 主要場所別勝ちを集計
    rec_in_pair = race[race["horse_id"].isin(horses_in_pair)]
    sums = rec_in_pair.groupby("horse_id").agg(
        n=("win","count"),
        win=("win","sum"),
        n_hanshin=("venue", lambda s: int((s.astype(str)=="阪神").sum())),
        win_hanshin=("win", lambda s: int(s[(rec_in_pair.loc[s.index,"venue"].astype(str)=="阪神")].sum())),
    ).sort_values("n", ascending=False).head(30)

    # 各馬の母父名・名前を結合
    sums["horse_name"] = sums.index.map(lambda h: horse_name.get(h, "?"))
    sums["bms_name"] = sums.index.map(lambda h: target_set.set_index("horse_id")["bms_name"].get(h, "?"))

    print(f"  {'馬名':<24} {'母父':<22} {'n':>4} {'win':>4} {'勝率':>6} {'阪神戦':>6} {'阪神勝':>6}")
    for hid, r in sums.iterrows():
        print(f"  {r['horse_name'][:22]:<24} {r['bms_name'][:20]:<22} {r['n']:>4} {r['win']:>4} "
              f"{r['win']/r['n']:>5.1%} {r['n_hanshin']:>6} {r['win_hanshin']:>6}")

    # タイトルホルダー特化
    TITLE = "2018103559"
    print(f"\n  ── タイトルホルダーの個別実績 ──")
    rec = _horse_record(TITLE)
    print(f"    通算: {rec.get('total', {})}")
    for k in ["v_阪神","v_東京","v_中山","v_京都","d_長距離","d_中距離","d_マイル","heavy"]:
        if k in rec:
            print(f"    {k}: {rec[k]}")

    # ── 比較: 「ドゥラメンテ × Turn-To系母父」「× Northern Dancer系母父」の上位馬 ──
    for mg in ["Turn-To系", "Northern Dancer系"]:
        print(f"\n  ── ドゥラメンテ × {mg} 母父 の所属馬 (出走数 上位 12) ──")
        sub_mg = sub_sire[sub_sire["bms_main"] == mg]
        h_set = set(sub_mg["horse_id"])
        rec_pair = race[race["horse_id"].isin(h_set)]
        s = rec_pair.groupby("horse_id").agg(n=("win","count"), win=("win","sum")).sort_values("n", ascending=False).head(12)
        s["name"] = s.index.map(lambda h: horse_name.get(h, "?"))
        s["bms"] = s.index.map(lambda h: sub_mg.set_index("horse_id")["bms_name"].get(h, "?"))
        for hid, r in s.iterrows():
            print(f"    {r['name'][:22]:<24} {r['bms'][:20]:<22} n={r['n']:>3} win={r['win']:>3}")

    # ────────────────────────────────────────────
    # 別のペア検証: ロードカナロア × 母父系統
    # ────────────────────────────────────────────
    print(f"\n══════════════════════════════════════════════")
    SIRE2 = "2008103552"   # ロードカナロア
    print(f"  父 = ロードカナロア ({SIRE2})")
    print(f"══════════════════════════════════════════════")

    sub2 = bms[bms["sire_id"] == SIRE2]
    print(f"  ロードカナロア産駒 総数: {len(sub2)} 頭")
    print(f"  母父系統別: {dict(sub2['bms_main'].value_counts(dropna=False))}")
    print(f"  {'母父系統':<22} {'n_horses':>10} {'n_records':>10} {'阪神':>6} {'東京':>6} {'長距離':>7} {'重':>6}")
    for mg in MAIN_GROUPS:
        h = set(sub2[sub2["bms_main"] == mg]["horse_id"])
        prof = _profile_for(h)
        if prof is None: continue
        print(f"  {mg:<22} {prof['_n_horses']:>10} {prof['_n_total']:>10} "
              f"{prof.get('win_v_阪神', float('nan')):>6.2f} {prof.get('win_v_東京', float('nan')):>6.2f} "
              f"{prof.get('win_d_長距離', float('nan')):>7.2f} {prof.get('win_heavy', float('nan')):>6.2f}")

    # ロードカナロア × 非主流 の所属馬上位
    print(f"\n  ── ロードカナロア × 非主流母父 の所属馬 (出走数 上位 15) ──")
    h_set = set(sub2[sub2["bms_main"] == "非主流"]["horse_id"])
    rec_p = race[race["horse_id"].isin(h_set)]
    s = rec_p.groupby("horse_id").agg(n=("win","count"), win=("win","sum"),
                                       n_h=("venue", lambda s: int((s.astype(str)=="阪神").sum())),
                                       win_h=("win", lambda s: int(s[rec_p.loc[s.index,"venue"].astype(str)=="阪神"].sum())),
                                       ).sort_values("n", ascending=False).head(15)
    s["name"] = s.index.map(lambda h: horse_name.get(h, "?"))
    s["bms"] = s.index.map(lambda h: sub2.set_index("horse_id")["bms_name"].get(h, "?"))
    print(f"  {'馬名':<24}{'母父':<22}{'n':>4}{'win':>4}{'勝率':>6}{'阪神戦':>6}{'阪神勝':>6}")
    for hid, r in s.iterrows():
        print(f"  {r['name'][:22]:<24}{r['bms'][:20]:<22}{r['n']:>4}{r['win']:>4}{r['win']/r['n']:>5.1%}{r['n_h']:>6}{r['win_h']:>6}")

    # ────────────────────────────────────────────
    # 別のペア検証: キズナ × 母父系統
    # ────────────────────────────────────────────
    print(f"\n══════════════════════════════════════════════")
    SIRE3 = "2010105827"
    print(f"  父 = キズナ ({SIRE3})")
    print(f"══════════════════════════════════════════════")
    sub3 = bms[bms["sire_id"] == SIRE3]
    print(f"  キズナ産駒 総数: {len(sub3)}")
    print(f"  母父系統別: {dict(sub3['bms_main'].value_counts(dropna=False))}")
    print(f"  {'母父系統':<22} {'n_horses':>10} {'n_records':>10} {'阪神':>6} {'東京':>6} {'長距離':>7} {'重':>6}")
    for mg in MAIN_GROUPS:
        h = set(sub3[sub3["bms_main"] == mg]["horse_id"])
        prof = _profile_for(h)
        if prof is None: continue
        print(f"  {mg:<22} {prof['_n_horses']:>10} {prof['_n_total']:>10} "
              f"{prof.get('win_v_阪神', float('nan')):>6.2f} {prof.get('win_v_東京', float('nan')):>6.2f} "
              f"{prof.get('win_d_長距離', float('nan')):>7.2f} {prof.get('win_heavy', float('nan')):>6.2f}")


if __name__ == "__main__":
    main()
