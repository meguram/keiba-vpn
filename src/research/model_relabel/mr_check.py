"""データセットの点検の表（担当 model-relabel、11/27 夜のユーザーの言葉「データセットに不備があると元も子もない」）。

- 年ごとの行数・レース数・ラベルの割合・重複・1 レースの勝ち馬の数・結果表との突き合わせ（除いた行の理由別の数）・
  欠損の多い列・値が一定の列・名前が結果や市場に見える列・血統の 4 列と市場の q の網羅を、docs/operations/model_relabel/dataset_check.md に書く。
- 2016〜24 の行だけ（2025 年以降は読まない）。モデルの成績は見ない。費用 0 円（手元の表だけ）。
  KEIBA_TABLES_DIR=<本体>/data/page_reference/tables KEIBA_SESSION=model-relabel flock /tmp/claude-1000/heavy.lock nice -n 19 python -m src.research.model_relabel.mr_check
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.model_relabel import mr_labels as LB
from src.research.model_relabel import mr_train as TR

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "docs/operations/model_relabel/dataset_check.md"
RESULT_LIKE = r"finish|^fin$|_fin$|margin|time_sec|payoff|odds|popular|ninki|tansho|fukusho|ret_|_res_today|y1|y3"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer() else
                                     (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.to_numpy()) + " |")
    return "\n".join(out)


def main() -> None:
    lines = ["# データセットの点検（model-relabel）", "",
             "作り: `python -m src.research.model_relabel.mr_check`（2016〜24 の行だけ。モデルの成績は見ない）。", ""]
    # ---- 行とラベル
    X, meta, names = TR.load_matrix("v5bag")
    assert pd.to_numeric(meta["year"]).max() <= 2024
    lab = TR.attach_labels(meta)
    fin = pd.to_numeric(meta["finish_position"], errors="coerce")
    lab["fin"] = fin.to_numpy()
    g = lab.groupby("year")
    w = lab[lab["fin"] == 1].groupby("race_id").size()
    nwin = lab[["race_id", "year"]].drop_duplicates().assign(nw=lambda d: d["race_id"].map(w).fillna(0).astype(int))
    t = pd.DataFrame({
        "行": g.size(), "レース": g["race_id"].nunique(),
        "着順が無い・0 以下": g["fin"].apply(lambda s: int((s.isna() | (s <= 0)).sum())),
        "y1": g["y1"].mean(), "y_w": g["y_w"].mean(), "y3": g["y3"].mean(), "y_p": g["y_p"].mean(), "y_p1": g["y_p1"].mean(),
        "大敗": g["big"].mean(), "本命決着の行": g["chalk_race"].mean(),
        "拡張しないレース": g.apply(lambda d: int((~d.groupby("race_id")["ext_ok"].all()).sum())),
        "勝ち馬 0 頭のレース": nwin.groupby("year")["nw"].apply(lambda s: int((s == 0).sum())),
        "勝ち馬 2 頭以上のレース": nwin.groupby("year")["nw"].apply(lambda s: int((s >= 2).sum())),
    }).reset_index()
    lines += ["## 1. 年ごとの行・レース・ラベル（学習の表 = cycles16_fix3 の meta と X）", "", md_table(t), ""]
    dup = int(meta.duplicated(["race_id", "horse_id"]).sum())
    lines += [f"- (race_id, horse_id) の重複: {dup} 行。X と meta の行数: {len(X)}・{len(meta)}。", ""]
    # ---- 結果表との突き合わせ
    res = LB.load_results()          # 2022〜24 は開発の検証の期間なので台帳に残す
    res["race_id"], res["horse_id"] = res["race_id"].astype(str), res["horse_id"].astype(str)
    import pyarrow.parquet as pq
    st = []
    for y in range(2016, 2025):
        p = LB.TABLES / str(y) / "race_result_flat.parquet"
        cols = [c for c in ("race_id", "horse_id", "finish_status", "surface") if c in pq.ParquetFile(p).schema_arrow.names]
        st.append(pd.read_parquet(p, columns=cols))
    st = pd.concat(st, ignore_index=True)
    st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
    key_m = set(zip(meta["race_id"].astype(str), meta["horse_id"].astype(str)))
    st["in_meta"] = [(a, b) in key_m for a, b in zip(st["race_id"], st["horse_id"])]
    st["year"] = st["race_id"].str[:4].astype(int)
    miss = st[~st["in_meta"]]
    fpm = res.drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])["finish_position"]
    fp_miss = pd.to_numeric(fpm.reindex(list(zip(miss["race_id"], miss["horse_id"]))).to_numpy(), errors="coerce")
    by_fp = np.select([np.isnan(fp_miss), fp_miss <= 0], ["結果表に鍵なし", "着順 0 以下（中止・除外・取消。2020 年以降は区別の列が無い）"], "着順あり")
    st_label = miss["finish_status"].fillna(pd.Series(by_fp, index=miss.index))
    reason = miss.assign(理由=st_label + np.where(miss["surface"] == "障", "・障害", ""))
    rt = reason.groupby(["year", "理由"]).size().unstack(fill_value=0).reset_index()
    lines += ["## 2. 結果表にあって学習の表に無い走（除いた行の理由別）", "",
              "理由: finish_status（scratched = 取消、excluded = 除外、dnf = 中止、finished = 完走）。2020 年以降はこの列が無いので着順で分ける。"
              "「・障害」は障害のレースの走（障害の完走は学習の表に入っている）。", "",
              md_table(rt), ""]
    key_r = set(zip(st["race_id"], st["horse_id"]))
    extra = sum((a, b) not in key_r for a, b in zip(meta["race_id"].astype(str), meta["horse_id"].astype(str)))
    lines += [f"- 学習の表にあって結果表に無い走: {extra} 行。", ""]
    fin_ok = miss[(miss["finish_status"] == "finished") | (fp_miss > 0)]
    lines += [f"- 結果表で完走なのに学習の表に無い走: {len(fin_ok)} 行（{fin_ok['race_id'].nunique()} レース。障害を除くと "
              f"{int((fin_ok['surface'] != '障').sum())} 行）。", ""]
    if len(fin_ok):
        dd = pd.concat([pd.read_parquet(LB.TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_id", "date"])
                        for y in sorted(fin_ok["year"].unique())])
        dd["race_id"], dd["horse_id"] = dd["race_id"].astype(str), dd["horse_id"].astype(str)
        for r in sorted(fin_ok["race_id"].unique()):
            x = dd[dd["race_id"] == r]
            other = dd[(dd["date"] == x["date"].iloc[0]) & dd["horse_id"].isin(x["horse_id"]) & (dd["race_id"] != r)]
            lines.append(f"  - {r}: {len(x)} 頭のうち {other['horse_id'].nunique()} 頭が同じ日に {', '.join(sorted(other['race_id'].unique()))} にも出ている"
                         "（取り込みの誤りの写し。pedigree_lab の data._prep と同じく、学習の表から丸ごと外れているのが正しい）")
        lines.append("")
    # ---- 列の欠損・一定・名前
    yr = pd.to_numeric(meta["year"]).to_numpy()
    na = X.isna()
    by = pd.DataFrame({y: na[yr == y].mean() for y in range(2016, 2025)})
    by["全体"] = na.mean()
    top = by.sort_values("全体", ascending=False).head(25).round(3).reset_index().rename(columns={"index": "列"})
    lines += ["## 3. 欠損の多い列（v5bag、上位 25）", "", md_table(top), ""]
    const = [c for c in names if X[c].nunique(dropna=True) <= 1]
    allna = [c for c in names if na[c].all()]
    lines += [f"- 値が一定の列（{len(const)}）: {', '.join(const[:30])}", f"- 全部が欠けの列（{len(allna)}）: {', '.join(allna[:30])}", ""]
    import json
    v6 = json.loads(TR.V6E_BUNDLE.read_text(encoding="utf-8"))
    v6 = next(m for m in v6["members"] if m["name"] == "xgb_extras_ped")["feature_cols"]
    sus5 = [c for c in names if re.search(RESULT_LIKE, c, re.I)]
    sus6 = [c for c in v6 if re.search(RESULT_LIKE, c, re.I)]
    lines += ["## 4. 名前が結果・市場に見える列（名前だけの篩。作りは longshot-ai の market_free_columns.md で判定済み）", "",
              f"- v5bag（{len(sus5)}）: {', '.join(sus5)}", f"- v6e の代理（{len(sus6)}）: {', '.join(sus6)}",
              "- どれも過去の走の値（前走の着順・着差の履歴。as-of）で、その走の結果ではない（horse_history の lag・seq の作り）。"
              "その走の着順・着差・時計・払戻・オッズ・人気の列は無い。", ""]
    # ---- 血統の 4 列・市場の q の網羅
    from src.research.model_relabel import mr_ped
    ped = pd.read_parquet(mr_ped.DEST)
    k = meta[["race_id", "horse_id"]].astype(str).merge(ped, on=["race_id", "horse_id"], how="left")
    q = pd.read_parquet("/home/jovyan/work/megu_cycles_out/runners.parquet", columns=["race_id", "horse_id", "year", "q_final"],
                        filters=[("year", "<=", 2024)])
    q["race_id"], q["horse_id"] = q["race_id"].astype(str), q["horse_id"].astype(str)
    kq = meta[["race_id", "horse_id"]].astype(str).merge(q.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    cov = pd.DataFrame({"year": yr, "dirt_mat5": k["dirt_mat5"].notna().to_numpy(), "q_final": kq["q_final"].notna().to_numpy()})
    lines += ["## 5. 血統の 4 列と市場の q の網羅（行の割合）", "", md_table(cov.groupby("year").mean().round(4).reset_index()), "",
              "- 血統の 4 列の欠けは、pedigree_lab の feat_v4 に無い走（障害のレース）。市場の q の欠けは runners.parquet に無い走。", ""]
    # ---- 障害のレース・特徴量のほとんど無い馬（11/27 統括役・longshot-ai の指摘）
    rs = res.drop_duplicates(["race_id", "horse_id"])[["race_id", "horse_id", "surface"]]
    sm = meta[["race_id", "horse_id"]].astype(str).merge(rs, on=["race_id", "horse_id"], how="left")
    obst = (sm["surface"] == "障").to_numpy()
    nan_share = na.mean(axis=1).to_numpy()
    debut = pd.to_numeric(X["horse_prior_races"], errors="coerce").fillna(0).to_numpy() == 0
    odds = pd.read_parquet("/home/jovyan/work/megu_cycles_out/runners.parquet", columns=["race_id", "horse_id", "year", "odds_final"],
                           filters=[("year", "<=", 2024)])
    odds["race_id"], odds["horse_id"] = odds["race_id"].astype(str), odds["horse_id"].astype(str)
    ko = meta[["race_id", "horse_id"]].astype(str).merge(odds.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    o = ko["odds_final"].to_numpy(float)
    t6 = pd.DataFrame({"year": yr, "障害の行": obst, "障害のレース": np.where(obst, meta["race_id"].astype(str), None),
                       "初出走": debut, "q の欠け": kq["q_final"].isna().to_numpy()})
    t6g = t6.groupby("year").agg(行=("year", "size"), 障害の行=("障害の行", "sum"), 障害のレース=("障害のレース", "nunique"),
                                 初出走=("初出走", "sum"), q_欠け=("q の欠け", "sum")).reset_index()
    lines += ["## 6. 障害のレース・特徴量のほとんど無い馬・市場の q の欠け", "", md_table(t6g), "",
              f"- 欠けの割合の分布（行）: ≥50% {float(np.mean(nan_share >= 0.5)):.4%}・≥80% {float(np.mean(nan_share >= 0.8)):.4%}・"
              f"≥90% {float(np.mean(nan_share >= 0.9)):.4%}。最大 {float(nan_share.max()):.3f}。",
              "- 2016 年の初出走が多いのは、履歴が 2016 年の始めから数え始めるため（2015 年以前の走が無い）。2017 年以降は年約 4,600〜4,900 頭。",
              f"- 初出走（horse_prior_races が 0 か欠け）: {float(np.mean(debut)):.4%}"
              f"（オッズのある行のうち確定 10 倍以上の割合 {float(np.mean(o[debut & np.isfinite(o)] >= 10)):.4f}）。"
              "longshot-ai の言う「特徴量の無い馬 0.28%」は、数え方（どの列の欠けか）を突き合わせる。",
              f"- 障害の行のうち q が欠ける割合: {float(np.mean(kq['q_final'].isna().to_numpy()[obst])):.4f}。"
              f"q が欠ける行のうち障害の割合: {float(np.mean(obst[kq['q_final'].isna().to_numpy()])):.4f}。",
              f"- 平地で q の欠けた行: {int((kq['q_final'].isna().to_numpy() & ~obst).sum())}（評価からはそのレースを丸ごと除く）。",
              "- 扱い（spec mr-7）: 学習は今の v5bag と同じく全行（障害・初出走を含む）。評価は平地で全頭の q がそろったレースにそろえる。"
              "初出走など特徴量の少ない馬は落とさない（推論でも同じ馬が出るので、学習と評価から外すと本番とずれる）。", ""]
    import hashlib
    lp = LB.OUT / "labels_convert_A.parquet"
    lines += ["## 7. 中間物の sha256", "",
              f"- labels_convert_A.parquet: {hashlib.sha256(lp.read_bytes()).hexdigest()}（chalk_race の列を足した最後の版。これより後は上書きしない）",
              f"- spec.json: {hashlib.sha256(TR.SPEC.read_bytes()).hexdigest()}", ""]
    # ---- longshot-ai の「特徴量の無い馬」との突き合わせ（runners にあって学習の表に行が無い馬）
    rn = pd.read_parquet("/home/jovyan/work/megu_cycles_out/runners.parquet", columns=["race_id", "horse_id", "year", "fin"], filters=[("year", "<=", 2024)])
    rn["race_id"], rn["horse_id"] = rn["race_id"].astype(str), rn["horse_id"].astype(str)
    rn["in"] = [(a, b) in key_m for a, b in zip(rn["race_id"], rn["horse_id"])]
    xo = rn[~rn["in"]]
    lines += ["## 8. runners（平地・オッズのある全馬）にあって学習の表に行が無い馬（longshot-ai の「特徴量の無い馬」）", "",
              f"- {len(xo)} 頭（{xo['race_id'].nunique()} レース）。うち着順が無い（中止・除外など）: {int(xo['fin'].isna().sum())}、"
              f"着順がある: {int(xo['fin'].notna().sum())}（レース {', '.join(sorted(xo.loc[xo['fin'].notna(), 'race_id'].unique()))}）。",
              f"- 着順がある馬のレースが §2 の写しのレースの中にあるか: "
              f"{set(xo.loc[xo['fin'].notna(), 'race_id']) <= set(fin_ok['race_id'])}。", ""]
    DEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
