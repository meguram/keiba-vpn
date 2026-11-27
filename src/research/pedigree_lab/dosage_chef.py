"""本家の Dosage（公開の chef-de-race の一覧・2019-10 時点 を 4 代の牡の祖先に当てる。着順は使わない段 0 の数え上げ）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.dosage_chef <一覧のテキスト>

- 一覧は teamvalor の公開の PDF（ChefDeRace_Oct2019.pdf）から取り出した「馬名 (区分)」の行。区分は B・I・C・S・P、2 区分は半分ずつ。
- 祖先の英語の名前は、手元の 10 代の血統表（data/local/horse_pedigree_10gen）の name の英字の部分。大文字にし、記号を除いて突き合わせる。
- 点: 1 代 16・2 代 8・3 代 4・4 代 2（牡の祖先だけ）。DI = (B + I + C/2) / (C/2 + S + P)、CD = (2B + I − S − 2P) / 全点。
- 注: 2019-10 の一覧は、その時点までの産駒の成績を見て選ばれた指定で、評価の期間の一部と重なる（軽い先読み）。
- 出力: OUT/dosage_chef.parquet（horse_id・dp_B〜dp_P・dp_total・DI・CD）と、当たる割合の要約（標準出力）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data

PED10 = Path(__file__).resolve().parents[3] / "data/local/horse_pedigree_10gen"
if not PED10.exists():                                   # worktree には data/local が無いので、元の作業ツリーの手元の表を読む（GCS は読まない）
    PED10 = Path("/home/jovyan/work/keiba-vpn/data/local/horse_pedigree_10gen")
PTS = {1: 16.0, 2: 8.0, 3: 4.0, 4: 2.0}


def norm(s: str) -> str:
    s = re.sub(r"\s+(II|III|IV)\s*$", "", s.strip().upper())        # 一覧のローマ数字（NIJINSKY II など）を落とす（critic）
    return re.sub(r"[^A-Z0-9]", "", s.replace("’", "'").replace("'", ""))


def load_chef(path: str) -> dict[str, list[str]]:
    out = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(.+?)\s*\(([BICSP](?:/[BICSP])?)\)\s*$", line)
        if m:
            out[norm(m.group(1))] = m.group(2).split("/")
    return out


def english(name: str) -> str:
    m = re.search(r"[A-Za-z][A-Za-z0-9 .'’\-]*", name or "")
    return norm(m.group(0)) if m else ""


def main() -> None:
    chef = load_chef(sys.argv[1])
    print("chef の頭数", len(chef), flush=True)
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "dosage_chef",
                                   "chef-de-race の一覧を 4 代に当てる数え上げ（着順を使わない）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["horse_id", "date"])
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)]
    rows, miss = [], 0
    for h in X["horse_id"].unique():
        f = PED10 / h[:4] / f"{h}.json"
        if not f.exists():
            miss += 1
            continue
        anc = json.loads(f.read_text()).get("ancestors", [])
        pt = dict.fromkeys("BICSP", 0.0)
        pm = dict.fromkeys("BICSP", 0.0)                    # 母の側（p ≥ 2^(g−1)）だけの点（1 号）
        for a in anc:
            g, p = int(a.get("generation", 0)), int(a.get("position", 0))
            if 1 <= g <= 4 and p % 2 == 0:
                c = chef.get(english(a.get("name", "")))
                if c:
                    for k in c:
                        pt[k] += PTS[g] / len(c)
                        if p >= 2 ** (g - 1):
                            pm[k] += PTS[g] / len(c)
        rows.append((h, *pt.values(), *pm.values()))
    D = pd.DataFrame(rows, columns=["horse_id", "dp_B", "dp_I", "dp_C", "dp_S", "dp_P", "dm_B", "dm_I", "dm_C", "dm_S", "dm_P"])
    tm = D[["dm_B", "dm_I", "dm_C", "dm_S", "dm_P"]].sum(axis=1)
    denm = D["dm_C"] / 2 + D["dm_S"] + D["dm_P"]
    D["dm_total"] = tm
    # 分母が 0 で点がある（B・I だけ）ときは DI を上限 64 にする（点の無い馬の NaN と区別する。critic）
    D["DI_m"] = np.where(denm > 0, (D["dm_B"] + D["dm_I"] + D["dm_C"] / 2) / denm.replace(0, np.nan), np.where(tm > 0, 64.0, np.nan))
    D["CD_m"] = np.where(tm > 0, (2 * D["dm_B"] + D["dm_I"] - D["dm_S"] - 2 * D["dm_P"]) / tm.replace(0, np.nan), np.nan)
    D["dp_total"] = D[["dp_B", "dp_I", "dp_C", "dp_S", "dp_P"]].sum(axis=1)
    den = D["dp_C"] / 2 + D["dp_S"] + D["dp_P"]
    D["DI"] = np.where(den > 0, (D["dp_B"] + D["dp_I"] + D["dp_C"] / 2) / den.replace(0, np.nan), np.where(D["dp_total"] > 0, 64.0, np.nan))
    D["CD"] = np.where(D["dp_total"] > 0, (2 * D["dp_B"] + D["dp_I"] - D["dp_S"] - 2 * D["dp_P"]) / D["dp_total"].replace(0, np.nan), np.nan)
    path = data.OUT / "dosage_chef.parquet"
    D.to_parquet(path, index=False)
    print("10 代の表の無い馬", miss, "馬", len(D), flush=True)
    print("点が 1 つ以上の馬の割合", round(float((D["dp_total"] > 0).mean()), 3), flush=True)
    print("点の合計の分布", D["dp_total"].describe().round(2).to_dict(), flush=True)
    print("点が 8 以上の馬の割合", round(float((D["dp_total"] >= 8).mean()), 3), flush=True)
    print("DI の分布", D["DI"].describe().round(2).to_dict(), "CD の分布", D["CD"].describe().round(2).to_dict(), flush=True)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
