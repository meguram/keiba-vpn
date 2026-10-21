"""tp-60 血統の静的表: 10 世代までの血量（Σ 2^-世代）と、父・母父などの ID を馬ごとに作る（結果・オッズは使わない。as-of の問題なし）。

python -m src.scripts.data.tp60_ped static        # ホストで実行。出力 <TP60_OUT>/ped/*.parquet, *.json
python -m src.scripts.data.tp60_ped check         # 被覆の要約

入力: tp-38 の祖先表（<TP38_OUT>/static/{anc,horses,ancestors}.parquet。5 世代まで）と、馬別 JSON（data/local/horse_pedigree_5gen）。
6〜10 世代は『5 世代目の祖先 X の自分の 5 世代表（horse_pedigree_5gen/<X の先頭 4 桁>/<X>.json）』をつなげて作る（10gen の連結 JSON は 14% の馬にしか無い）。
X のファイルが無い／祖先が空のとき、その枝の 6〜10 世代は欠ける（被覆は馬ごとに deep_cov = 解決できた 5 世代目の枠の割合で持つ）。

出力:
  horse_static.parquet   horse_id, 親の ID（sire/dam/ssire/dsire/sdsire/ddsire/dsire_sire）, 父系統(sire_line / dsire_line), 近交（x_F, x_n, x_max）, deep_cov, birth_year
  blood_share.parquet    horse_id, anc_id, w5（1〜5 世代の血量）, wdeep（6〜10 世代の血量）。男系祖先のうち 10 世代以内に 200 頭以上に出るものだけ
  anc_names.json         祖先 ID → 名前（表示用。公開の章には ID と代表名のみ）
  resolve_stats.json     被覆
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
TP38_OUT = Path(os.environ.get("TP38_OUT") or "/home/jovyan/work/tp38_out")
OUT = Path(os.environ.get("TP60_OUT") or "/home/jovyan/work/tp60_out") / "ped"
PED5 = ROOT / "data/local/horse_pedigree_5gen"
MIN_HORSES = int(os.environ.get("TP60_MIN_HORSES", 200))

# (gen, pos) → 役割名
ROLES = {"sire": (1, 0), "dam": (1, 1), "ssire": (2, 0), "dsire": (2, 2), "sdsire": (3, 2), "ddsire": (3, 6), "dsire_sire": (3, 4)}

# 父系統（尾雄の鎖の中で最初に一致した創始者）。優先順に並べる
LINE_PATTERNS = [
    ("SS系", r"サンデーサイレンス|Sunday Silence"),
    ("ロベルト系", r"ロベルト|Roberto"),
    ("MrP系", r"ミスタープロスペクター|Mr\.? ?Prospector"),
    ("ND系", r"ノーザンダンサー|Northern Dancer"),
    ("ナスルーラ系", r"ボールドルーラー|Bold Ruler|ナスルーラ|Nasrullah"),
    ("ネイティヴダンサー系", r"ネイティヴダンサー|Native Dancer|レイズアネイティヴ|Raise a Native"),
    ("ヘイルトゥリーズン系", r"ヘイルトゥリーズン|Hail to Reason"),
    ("ブランドフォード系", r"ブランドフォード|Blandford|ヘロド|Herod|エクリプス|Eclipse"),
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read5(aid: str) -> dict | None:
    f = PED5 / aid[:4] / f"{aid}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def classify_line(chain_names: list[str]) -> str:
    """尾雄の鎖（本馬 → 父 → 父父 …）の名前から、最初に一致した創始者の系統。"""
    for nm in chain_names:
        for lab, pat in LINE_PATTERNS:
            if re.search(pat, nm or ""):
                return lab
    return "その他"


def cmd_static(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    S = TP38_OUT / "static"
    anc = pd.read_parquet(S / "anc.parquet")
    horses = pd.read_parquet(S / "horses.parquet")["horse_id"].astype(str).to_numpy()
    uniq = pd.read_parquet(S / "ancestors.parquet")["anc_id"].astype(str).to_numpy()
    H = pd.read_parquet(S / "cross_H.parquet")
    _log(f"馬 {len(horses):,}、祖先 {len(uniq):,}、祖先表 {len(anc):,} 行")

    # 名前（馬・5 世代目の祖先・その祖先の鎖）
    names: dict[str, str] = {}
    g5 = anc[anc["gen"] == 5]
    xs = np.unique(g5["a"].to_numpy())
    _log(f"5 世代目の祖先（ユニーク）{len(xs):,}")
    xtab_rows: list[tuple[str, int, int, str]] = []
    resolved_x: set[str] = set()
    for a_code in xs:
        x = str(uniq[a_code])
        js = _read5(x)
        if not js or not js.get("ancestors"):
            continue
        resolved_x.add(x)
        for r in js["ancestors"]:
            g, p, i = r.get("generation"), r.get("position"), str(r.get("horse_id") or "")
            if g and g <= 5 and i:
                xtab_rows.append((x, int(g), int(p), i))
                nm = str(r.get("name") or "")
                if nm and i not in names:
                    names[i] = nm
    xt = pd.DataFrame(xtab_rows, columns=["x", "g2", "p2", "b_id"])
    _log(f"解決できた X {len(resolved_x):,}/{len(xs):,}、X の祖先行 {len(xt):,}")

    # 馬自身のファイルから名前・近交・生年
    for h in horses:
        js = _read5(h)
        if not js:
            continue
        for r in js.get("ancestors") or []:
            i = str(r.get("horse_id") or "")
            if i and i not in names:
                names[i] = str(r.get("name") or "")
    (OUT / "anc_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")

    # ---- 馬ごとの静的表
    hs = pd.DataFrame({"h": np.arange(len(horses)), "horse_id": horses})
    pos_tab = anc.assign(anc_id=uniq[anc["a"].to_numpy()])
    for role, (g, p) in ROLES.items():
        sub = pos_tab[(pos_tab["gen"] == g) & (pos_tab["pos"] == p)].drop_duplicates("h").set_index("h")["anc_id"]
        hs[f"{role}_id"] = hs["h"].map(sub)
    hs["birth_year"] = pd.to_numeric(hs["horse_id"].str[:4], errors="coerce")
    hs.loc[hs["horse_id"].str.startswith("000"), "birth_year"] = np.nan
    hs["horse_foreign"] = hs["horse_id"].str.startswith("000").astype(int)

    # 父系統: 父・母父の尾雄の鎖（本馬自身を含む）で分類。鎖は X のファイルから（父→父父→…）
    chain_cache: dict[str, str] = {}

    def line_of(sid: str | None) -> str | None:
        if not sid:
            return None
        if sid in chain_cache:
            return chain_cache[sid]
        chain = [names.get(sid, "")]
        js = _read5(sid)
        if js and js.get("ancestors"):
            nm_by = {(int(r["generation"]), int(r["position"])): str(r.get("name") or "") for r in js["ancestors"]}
            for g in range(1, 6):
                chain.append(nm_by.get((g, 0), ""))
        lab = classify_line(chain)
        chain_cache[sid] = lab
        return lab

    hs["sire_line"] = hs["sire_id"].map(line_of)
    hs["dsire_line"] = hs["dsire_id"].map(line_of)
    hs = hs.merge(H.reset_index().rename(columns={"h": "h"}), on="h", how="left")
    _log(f"系統の内訳（父）: {hs['sire_line'].value_counts(dropna=False).head(10).to_dict()}")

    # ---- 6〜10 世代の血量
    xcode = {str(u): i for i, u in enumerate(uniq)}
    b_ids = pd.Index(pd.unique(np.concatenate([uniq, xt["b_id"].to_numpy()])))
    bcode = {b: i for i, b in enumerate(b_ids)}
    xt["x_code"] = xt["x"].map(xcode).astype("int32")
    xt["b"] = xt["b_id"].map(bcode).astype("int32")
    xt["w2"] = (0.5 ** (5 + xt["g2"].astype(float))).astype("float64")
    xt["male"] = (xt["p2"] % 2 == 0)
    xtp = pl.from_pandas(xt[xt["male"]][["x_code", "b", "w2"]])
    g5p = pl.from_pandas(g5[["h", "a"]].rename(columns={"a": "x_code"}).astype("int32"))
    # 解決率（馬ごと）
    ok_x = np.zeros(len(uniq), dtype=bool)
    ok_x[[xcode[x] for x in resolved_x]] = True
    g5 = g5.assign(ok=ok_x[g5["a"].to_numpy()])
    cov = g5.groupby("h")["ok"].agg(["mean", "size"]).rename(columns={"mean": "deep_cov", "size": "n_g5"})
    hs = hs.merge(cov, left_on="h", right_index=True, how="left")
    parts = []
    chunk = 4000
    for lo in range(0, len(horses), chunk):
        sub = g5p.filter((pl.col("h") >= lo) & (pl.col("h") < lo + chunk))
        j = sub.join(xtp, on="x_code", how="inner").group_by(["h", "b"]).agg(pl.col("w2").sum().alias("wdeep"))
        parts.append(j)
    deep = pl.concat(parts)
    _log(f"深い血量（男系）{deep.height:,} 行")
    # 1〜5 世代（男系のみ）
    a5 = anc[anc["male"]].copy()
    a5["b"] = pd.Series(uniq[a5["a"].to_numpy()]).map(bcode).to_numpy()
    sh5 = pl.from_pandas(a5[["h", "b", "w"]].astype({"h": "int32", "b": "int32"})).group_by(["h", "b"]).agg(pl.col("w").sum().alias("w5"))
    full = sh5.join(deep, on=["h", "b"], how="full", coalesce=True).with_columns(pl.col("w5").fill_null(0.0), pl.col("wdeep").fill_null(0.0))
    cnt = full.group_by("b").agg(pl.len().alias("n_h")).filter(pl.col("n_h") >= MIN_HORSES)
    full = full.join(cnt, on="b", how="inner")
    pdf = full.to_pandas()
    pdf["horse_id"] = horses[pdf["h"].to_numpy()]
    pdf["anc_id"] = b_ids[pdf["b"].to_numpy()]
    pdf[["horse_id", "anc_id", "w5", "wdeep"]].astype({"w5": "float32", "wdeep": "float32"}).to_parquet(OUT / "blood_share.parquet", index=False)
    hs.drop(columns=["h"]).to_parquet(OUT / "horse_static.parquet", index=False)
    stats = {"horses": int(len(horses)), "x_total": int(len(xs)), "x_resolved": int(len(resolved_x)), "deep_cov_mean": float(hs["deep_cov"].mean()),
             "deep_cov_by_birth": {str(int(k)): float(v) for k, v in hs.groupby("birth_year")["deep_cov"].mean().items() if k == k},
             "male_ancestors_kept": int(pdf["anc_id"].nunique()), "share_rows": int(len(pdf)), "min_horses": MIN_HORSES}
    (OUT / "resolve_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"完了 {stats}")


def cmd_check(a) -> None:
    st = json.loads((OUT / "resolve_stats.json").read_text(encoding="utf-8"))
    print(json.dumps(st, ensure_ascii=False, indent=1))
    hs = pd.read_parquet(OUT / "horse_static.parquet")
    print(hs.shape)
    print(hs[["sire_id", "dsire_id", "sire_line", "dsire_line", "x_F", "deep_cov"]].describe(include="all").T)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("static")
    sub.add_parser("check")
    a = ap.parse_args()
    {"static": cmd_static, "check": cmd_check}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
