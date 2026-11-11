"""厩舎コメントの語ごとの好走傾向（形態素解析 → 3 着以内=good / 4 着以降=bad）。

    PYTHONPATH=<fugashi と unidic-lite の入ったディレクトリ> python -m src.scripts.data.trainer_comment_words \
        [--out 出力ディレクトリ] [--min-df 60] [--years 2020-2025]

- コメント: race_trainer_comment_flat（レース前に公開）。結果を示す語（prerace_extras.RESULT_WORDS）を含む行は除く。〈師名〉・先頭の Ｑ も除く。
- 結果: race_result_flat の finish_position（数値にならない行＝中止・除外などは除く）。good = 1〜3 着、bad = 4 着以降。
- 語: fugashi + unidic-lite。名詞・動詞・形容詞・形状詞・副詞・感動詞・連体詞の見出し語。直後が「ない」「ず」なら『語(否)』にする。
  1 語（uni）と、隣り合う内容語の 2 語（bi）を数える。1 コメントに何回出ても 1 回と数える（有無）。
- 指標:
  - 生の lift = その語を含むコメントの good 率 / 全体の good 率。検定は 2 群の比率の差（語あり vs 語なし）、BH で FDR 補正。
  - 人気補正の lift = 実際の good 数 / 期待 good 数。期待は『その馬の単勝人気順位ごとの全体の 3 着以内率』の合計（人気に織り込み済みかを見る）。
    人気は結果側の値で、説明のための補正にだけ使う（特徴量ではない）。
  - 前後の確認: 前半の年で見つけた語が後半の年でも同じ向きか（符号一致率・lift の相関）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
RESULT_WORDS = ("勝てた", "勝った", "勝利", "優勝", "制した", "2着", "3着", "着に", "圏内に入", "ゴール", "快勝", "惜敗", "完敗", "敗れ")
CONTENT_POS = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "感動詞", "連体詞"}
SKIP_POS2 = {"数詞", "代名詞", "非自立可能", "助数詞"}
NEG_LEMMAS = {"ない", "ず", "ぬ"}


def load_pairs(years: list[int], tables: Path = TABLES) -> pd.DataFrame:
    import polars as pl

    cs, rs = [], []
    for y in years:
        fc = tables / str(y) / "race_trainer_comment_flat.parquet"
        fr = tables / str(y) / "race_result_flat.parquet"
        if not (fc.exists() and fr.exists()):
            continue
        c = pl.read_parquet(fc, columns=["race_id", "horse_id", "comment"]).to_pandas()
        r = pl.read_parquet(fr, columns=["race_id", "horse_id", "finish_position", "popularity", "field_size", "date", "surface"]).to_pandas()
        c["year"] = y
        cs.append(c)
        rs.append(r)
    C = pd.concat(cs, ignore_index=True)
    R = pd.concat(rs, ignore_index=True)
    C = C[C["horse_id"].notna() & C["comment"].notna()].copy()
    C["race_id"], C["horse_id"] = C["race_id"].astype(str), C["horse_id"].astype(str)
    R["race_id"], R["horse_id"] = R["race_id"].astype(str), R["horse_id"].astype(str)
    t = C["comment"].astype(str).str.strip()
    t = t.str.replace(r"〈[^〉]*〉", "", regex=True).str.replace(r"^Ｑ[　 ]*", "", regex=True).str.strip()
    C["text"] = t
    C = C[(C["text"] != "") & ~C["text"].map(lambda x: any(w in x for w in RESULT_WORDS))]
    C = C.drop_duplicates(["race_id", "horse_id"], keep="last")
    D = C.merge(R.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="inner")
    D["pos"] = pd.to_numeric(D["finish_position"], errors="coerce")
    D = D[D["pos"].notna()].copy()
    D["good"] = (D["pos"] <= 3).astype(int)
    D["pop"] = pd.to_numeric(D["popularity"], errors="coerce")
    return D.reset_index(drop=True)


def _lemma(w) -> str:
    f = w.feature
    lem = getattr(f, "lemma", None) or w.surface
    return lem.split("-")[0]


def tokenize(texts: list[str]) -> list[list[str]]:
    try:
        import fugashi
    except ImportError:
        sys.exit("fugashi / unidic-lite が見つかりません。PYTHONPATH に入れたディレクトリを指定してください。")
    tagger = fugashi.Tagger()
    out = []
    for s in texts:
        ws = list(tagger(s))
        seq: list[str] = []
        for i, w in enumerate(ws):
            f = w.feature
            pos1 = getattr(f, "pos1", None)
            pos2 = getattr(f, "pos2", None)
            if pos1 not in CONTENT_POS or pos2 in SKIP_POS2:
                if pos1 == "補助記号" and seq and seq[-1] != "|":
                    seq.append("|")
                continue
            lem = _lemma(w)
            if len(lem) == 0 or lem in {"*"}:
                continue
            nxt = ws[i + 1] if i + 1 < len(ws) else None
            if nxt is not None and getattr(nxt.feature, "pos1", None) == "助動詞" and _lemma(nxt) in NEG_LEMMAS:
                lem += "(否)"
            seq.append(lem)
        uni = [t for t in seq if t != "|"]
        bi = []
        for a, b in zip(seq[:-1], seq[1:]):
            if a != "|" and b != "|":
                bi.append(f"{a}+{b}")
        out.append(sorted(set(uni)) + sorted(set(bi)))
    return out


def build_matrix(tokens: list[list[str]], min_df: int):
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(analyzer=lambda d: d, binary=True, min_df=min_df)
    X = cv.fit_transform(tokens).tocsc()
    return X, np.array(cv.get_feature_names_out())


def term_table(X, terms: np.ndarray, good: np.ndarray, p_exp: np.ndarray) -> pd.DataFrame:
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    N = len(good)
    G = good.sum()
    base = G / N
    n = np.asarray(X.sum(axis=0)).ravel().astype(float)
    g = np.asarray(X.T @ good).ravel().astype(float)
    E = np.asarray(X.T @ p_exp).ravel()
    V = np.asarray(X.T @ (p_exp * (1 - p_exp))).ravel()
    rate = g / n
    n0, g0 = N - n, G - g
    rate0 = g0 / n0
    pool = G / N
    se = np.sqrt(pool * (1 - pool) * (1 / n + 1 / n0))
    z = (rate - rate0) / se
    p = 2 * stats.norm.sf(np.abs(z))
    zadj = (g - E) / np.sqrt(V)
    padj = 2 * stats.norm.sf(np.abs(zadj))
    se_t = np.sqrt(rate * (1 - rate) / n)
    df = pd.DataFrame({
        "term": terms, "kind": np.where(pd.Series(terms).str.contains(r"\+", regex=True), "bi", "uni"),
        "n": n.astype(int), "good": g.astype(int), "good_rate": rate, "lift": rate / base,
        "lift_lo": (rate - 1.96 * se_t) / base, "lift_hi": (rate + 1.96 * se_t) / base,
        "z": z, "p": p, "exp_good": E, "adj_lift": g / E, "z_adj": zadj, "p_adj": padj,
    })
    df["q"] = multipletests(df["p"], method="fdr_bh")[1]
    df["q_adj"] = multipletests(df["p_adj"], method="fdr_bh")[1]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/work/trainer_comment_words"))
    ap.add_argument("--min-df", type=int, default=60)
    ap.add_argument("--tables", default=str(TABLES))
    ap.add_argument("--years", default="2020-2025")
    ap.add_argument("--split", type=int, default=2023, help="この年までを発見側、翌年以降を確認側にする")
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    years = list(range(y0, y1 + 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    D = load_pairs(years, Path(a.tables))
    n_nopop = int((~(D["pop"] > 0)).sum())
    D = D[D["pop"] > 0].reset_index(drop=True)
    print(f"単勝人気が不明(0 または欠損)の {n_nopop} 頭を除く")
    print(f"対象: {len(D):,} 頭  good率 {D['good'].mean():.4f}  年別 {D.groupby('year').size().to_dict()}")

    # 人気順位ごとの 3 着以内率（期待値）。14 位以降は 1 つにまとめる
    pk = D["pop"].clip(upper=14).astype(int)
    rate_by_pop = D.groupby(pk)["good"].mean()
    D["p_exp"] = pk.map(rate_by_pop).to_numpy()
    rate_by_pop.rename("top3_rate").to_csv(out / "top3_rate_by_popularity.csv")

    tokens = tokenize(D["text"].tolist())
    X, terms = build_matrix(tokens, a.min_df)
    good = D["good"].to_numpy().astype(float)
    p_exp = D["p_exp"].to_numpy()
    T = term_table(X, terms, good, p_exp)

    # 前後の確認
    disc = (D["year"] <= a.split).to_numpy()
    Td = term_table(X[disc], terms, good[disc], p_exp[disc]).set_index("term")
    Tv = term_table(X[~disc], terms, good[~disc], p_exp[~disc]).set_index("term")
    T = T.set_index("term")
    for c in ("n", "lift", "adj_lift", "z", "z_adj"):
        T[f"{c}_disc"] = Td[c]
        T[f"{c}_val"] = Tv[c]
    T["q_disc"] = Td["q"]
    T["same_dir_raw"] = np.sign(T["lift_disc"] - 1) == np.sign(T["lift_val"] - 1)
    T["same_dir_adj"] = np.sign(T["adj_lift_disc"] - 1) == np.sign(T["adj_lift_val"] - 1)
    T = T.reset_index().sort_values("z", ascending=False)
    T.to_csv(out / "term_stats.csv", index=False, encoding="utf-8-sig")

    summ = {"n_rows": int(len(D)), "good_rate": float(D["good"].mean()), "n_terms": int(len(T)), "min_df": a.min_df, "split_year": a.split}
    sig = T[T["q"] < 0.05]
    summ["n_sig_raw"] = int(len(sig))
    summ["n_sig_adj"] = int((T["q_adj"] < 0.05).sum())
    d_sig = T[T["q_disc"] < 0.05]
    summ["disc_sig_n"] = int(len(d_sig))
    summ["disc_sig_same_dir_raw_in_val"] = float(d_sig["same_dir_raw"].mean())
    summ["disc_sig_same_dir_adj_in_val"] = float(d_sig["same_dir_adj"].mean())
    summ["corr_lift_disc_val"] = float(T[["lift_disc", "lift_val"]].corr().iloc[0, 1])
    summ["corr_adj_lift_disc_val"] = float(T[["adj_lift_disc", "adj_lift_val"]].corr().iloc[0, 1])
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
