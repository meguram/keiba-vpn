"""父の系統・系統配合 × クラス・馬場・開催状態・会場・コース の成績マトリクス（勝率・複勝率・1-2-3-4着以下）の HTML を作る。

使い方: python src/scripts/docs/build_sire_line_course_matrix_html.py [OUT.html]

テーブルは 2 つ（画面のタブ）:
  1. 父の系統 … 列 = 大系統（1 階層目）→ 小系統（2 階層目。○ に印を付けた種牡馬が半数以上の小系統は緑の名前）
  2. 系統配合 … 列 = 父の大系統（1 階層目）→ 母方の型（2 階層目）。母方 = 母父・母母父・母母母父のいずれか（5 世代血統の位置から）。
     母方の型は smartrc の国別タイプ（日本型サンデー系・米国型ミスプロ系・欧州型ノーザンダンサー系など 10 種）。母方の祖先が複数の型に当たる馬は、それぞれの列に数える（列どうしは重なる）。
行（どちらのテーブルも共通）:
  - クラス別（芝／ダート。新馬・未勝利・1勝・2勝・3勝・OP・G3・G2・G1 と、上位クラスの集計: 3勝以上・OP以上・重賞）
  - 馬場状態別（芝・ダ × 良・稍重・重/不良。全会場）
  - 芝の開催状態: 開催日数（1-2・3-4・5-6・7日目以降。全年）、柵位置（A〜D）を考慮した馬場の新しさ（開幕／柵替わり × 1-2・3-4・5日目〜）、コース区分 A〜D
  - 会場別 → コース別（同じ会場・同じ芝/ダートで、レース数が少ない距離は隣の距離とまとめる。例: 芝 2500m 以上）
画面の操作（すべて画面側の JS。データは JSON で埋め込む）:
  - 色（既定は相対）: 系統ごとの「ふだん」（行の芝/ダート全体）の成績と行の全系統の率から見込んだ率との差。明るいほど得意（緑）、暗いほど苦手。
    どの条件でも強い系統は見込みも高いので緑にならない（2026-11-13 ユーザー指示: 絶対の水準で強調しない）。切り替えで従来の「同じ行の全系統の率との差」。
  - ★ / ☆: 30 走以上で、見込みを複勝率 3pt・勝率 1.5pt 以上上回るセル（正確な二項の上側の確率。★ = z≥3 相当、☆ = z≥2 相当の参考）。黄色の枠（☆は点線）と、見込みからの差（pt）。
    系統のふだんはそのセル自身を除いた残り。会場の全コースの行は芝ダの割合が混ざるので印なし。
    出走数が最少に満たないセルも NaN にせず、縮めた率で色を付け、「※」で低信頼を示す。
  - 並べ替え: 列見出し（小系統・大系統・「全系統」）を押すと、その列の率で行を並べ替える（降順 → 昇順 → 解除）。行ラベルを押すと、その行の率で列を並べ替える。
入力:
  data/page_reference/tables/<年>/race_result_flat.parquet （2016〜2026 の平地。障害を除く）
  data/page_reference/pedigree_race_index/horse_sire_map.parquet （馬 → 父）
  data/local/horse_pedigree_5gen/<上位4桁>/<馬ID>.json （母方の祖先）
  data/meta/sire_lines/sire_smartrc_colors.csv （種牡馬 → smartrc の大系統・小系統・型。系統が空欄の種牡馬は集計から外す）
  data/meta/sire_lines/euro_kick_marks.json （○ の種牡馬）
  data/page_reference/course_position/course_position_<年>.json （開催日ごとの芝のコース区分 A〜D。netkeiba のレース一覧から src/scraper/course_position.py が作る）
出力: data/meta/sire_lines/sire_line_course_matrix.html（docs/ 配下ではないので本には載せない）
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FLAT = REPO / "data/page_reference/tables"
SIRE_MAP = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
COLORS = REPO / "data/meta/sire_lines/sire_smartrc_colors.csv"
MARKS = REPO / "data/meta/sire_lines/euro_kick_marks.json"
PED5 = REPO / "data/local/horse_pedigree_5gen"
CPOS = REPO / "data/page_reference/course_position"
OUT = REPO / "data/meta/sire_lines/sire_line_course_matrix.html"
JST = timezone(timedelta(hours=9))
YEARS = range(2016, 2027)
VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
MIN_LINE_RUNS = 300          # 列に出す小系統の最少出走数
MIN_PAIR_RUNS = 600          # 列に出す配合（父の大系統 × 母方の型）の最少出走数
MIN_COURSE_RACES = 100       # コースの行の最少レース数（これに満たない距離は、同じ会場・同じ芝/ダートの隣の距離とまとめる）
_JUMP = r"障害|ジャンプ|JS\(|JG|J・G"
CLASSES = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "G3", "G2", "G1"]
CLASS_SUMS = [("3勝以上", ["3勝", "OP", "G3", "G2", "G1"]), ("OP以上（OP・重賞）", ["OP", "G3", "G2", "G1"]), ("重賞（G1〜G3）", ["G3", "G2", "G1"])]
COND_ROWS = [("芝", "芝", "良"), ("芝", "芝", "稍重"), ("芝", "芝", "重・不良"), ("ダート", "ダ", "良"), ("ダート", "ダ", "稍重"), ("ダート", "ダ", "重・不良")]
DAY_CATS = ["開催1-2日目", "開催3-4日目", "開催5-6日目", "開催7日目以降"]
RAIL_CATS = ["開幕 1-2日目", "開幕 3-4日目", "開幕 5日目〜", "柵替わり 1-2日目", "柵替わり 3-4日目", "柵替わり 5日目〜"]
LETTERS = ["A", "B", "C", "D"]
MAT_CACHE = REPO / "data/local/running_profile/horse_mother_types.json"
MAT_PATHS = [(2, 2), (3, 6), (4, 14)]      # 母父・母母父・母母母父の (世代, 位置)。位置の 2 進数の並び（0=父側・1=母側）が 母→父 / 母→母→父 / 母→母→母→父


# ── 読み込み ────────────────────────────────────────────────────────────────
def load_rail() -> pd.DataFrame:
    """開催日ごとの芝のコース区分: (date, venue) → letter。netkeiba のレース一覧（src/scraper/course_position.py。日ごとの値）だけを使う。
    cushion の表の course_position は使わない: 開催の最後の柵の位置が全日に入っており、日ごとの値ではない（2026 京都 第 4 回は、netkeiba が A（1〜9 日目）→ B（10 日目〜）のところ、全日 B）。"""
    rows: dict[tuple[str, str], str] = {}
    for f in sorted(glob.glob(str(CPOS / "course_position_*.json"))):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))["dates"]
        except (OSError, ValueError, KeyError):
            continue
        for date, recs in d.items():
            for r in recs:
                if r.get("turf_course") in LETTERS:
                    rows[(date, r["venue"])] = r["turf_course"]
    return pd.DataFrame([{"date": k[0], "venue": k[1], "letter": v} for k, v in rows.items()], columns=["date", "venue", "letter"])


def rail_categories(runs_days: pd.DataFrame, rail: pd.DataFrame) -> pd.DataFrame:
    """開催（年・会場・開催回）ごとに、柵の位置（A〜D）が同じ連続した開催日を 1 まとまり（ブロック）にし、最初のまとまりを「開幕」、以降を「柵替わり」とする。
    runs_days: date・venue・kai・day を持つ（その開催の全開催日。コース区分が分からない日も含める）。rail: date・venue・letter（分かる日だけ）。
    返す列: date・venue・letter・rail_cat。柵が替わった日が特定できないブロック（前日のコース区分が分からない）は、特定できるまで rail_cat を付けない（誤った日目にしない）。
    注: 「開幕」は開催回の最初の柵。前の開催から同じ柵のまま続く場合も、開催回の最初を開幕とみなす。"""
    d = runs_days.drop_duplicates(["date", "venue"]).copy()
    d["year"] = d["date"].str[:4]
    d = d.merge(rail, on=["date", "venue"], how="left").sort_values(["year", "venue", "kai", "day"])
    out = []
    for _, g in d.groupby(["year", "venue", "kai"], sort=False):
        days, letters = list(g["day"]), list(g["letter"])
        start = None
        first = True
        for i, (day, L) in enumerate(zip(days, letters)):
            cat = None
            if isinstance(L, str):
                prev = letters[i - 1] if i > 0 else None
                if i == 0:
                    start, first = day, True                                  # 開催の最初の日（開幕）
                elif isinstance(prev, str) and prev != L:
                    start, first = day, False                                 # 前日と柵が違う（柵替わり）。前日が分かるので特定できる
                elif not isinstance(prev, str):
                    start = None                                              # 前日が不明: 替わったか分からない。特定できるまで付けない
                if start is not None:
                    pos = day - start + 1
                    cat = ("開幕 " if first else "柵替わり ") + ("1-2日目" if pos <= 2 else "3-4日目" if pos <= 4 else "5日目〜")
            elif i == 0:
                start = None
            else:
                start = None                                                  # 不明な日をまたぐブロックは特定できない
            out.append({"date": g.iloc[i]["date"], "venue": g.iloc[i]["venue"], "letter": L if isinstance(L, str) else None, "rail_cat": cat})
    return pd.DataFrame(out, columns=["date", "venue", "letter", "rail_cat"])


def merged_courses(runs: pd.DataFrame) -> dict[tuple[str, str, int], tuple[str, int]]:
    """(venue, surface, distance) → (まとめた行の名前, 並び順)。同じ会場・同じ芝/ダートの中で、レース数が MIN_COURSE_RACES に満たない距離は隣とまとめる。
    まとめた行に、レース数が十分な距離が 1 つあれば、その距離の名前に「（○○mを含む）」を付ける（少ない距離が主力の距離の名前を範囲に変えない）。
    全て少ない距離のまとまりは、最も長い距離を含むなら「○○m以上」、それ以外は「○○〜○○m」。最後の足りない分は独立の行にする。"""
    races = runs.drop_duplicates("race_id").groupby(["venue", "surface", "distance"]).size()
    out: dict[tuple[str, str, int], tuple[str, int]] = {}
    for (v, sf), g in races.groupby(level=[0, 1]):
        ds = sorted(int(x) for x in g.index.get_level_values(2))
        n = {int(x): int(c) for x, c in zip(g.index.get_level_values(2), g.values)}
        groups: list[list[int]] = []
        cur: list[int] = []
        for dist in ds:
            cur.append(dist)
            if sum(n[x] for x in cur) >= MIN_COURSE_RACES:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        for i, grp in enumerate(groups):
            big = [x for x in grp if n[x] >= MIN_COURSE_RACES]
            if len(grp) == 1:
                lab = f"{sf}{grp[0]}m"
            elif big:
                main = max(big, key=lambda x: n[x])
                others = "・".join(f"{x}m" for x in grp if x != main)
                lab = f"{sf}{main}m（{others}を含む）"
            elif i == len(groups) - 1:
                lab = f"{sf}{grp[0]}m以上"
            else:
                lab = f"{sf}{grp[0]}〜{grp[-1]}m"
            for x in grp:
                out[(v, sf, x)] = (lab, grp[0])
    return out


def _mother_types_of(h: str, ledger_type: dict[str, str]) -> list[str] | None:
    p = PED5 / h[:4] / f"{h}.json"
    try:
        anc = json.loads(p.read_text(encoding="utf-8")).get("ancestors", [])
    except (OSError, ValueError):
        return None
    want = set(MAT_PATHS)
    ts = {ledger_type[a["horse_id"]] for a in anc if (a.get("generation"), a.get("position")) in want and a.get("horse_id") in ledger_type}
    return sorted(ts)


def mother_types(horse_ids: list[str], ledger_type: dict[str, str], cache: Path | None = MAT_CACHE) -> dict[str, list[str]]:
    """馬 → 母方（母父・母母父・母母母父）の型の一覧（重複を除く）。5 世代血統の位置から。台帳に無い祖先は飛ばし、1 つも分からない馬は入れない。
    読み込みが遅い（約 4 万 5 千頭で 10 分）ので、結果をローカルにキャッシュし（台帳の更新時刻が変わったら作り直す）、無い馬だけを並列で読む。"""
    from concurrent.futures import ThreadPoolExecutor

    key = (f"{COLORS.stat().st_mtime:.0f}" if COLORS.exists() else "") + "|" + ",".join(f"{g}.{p}" for g, p in MAT_PATHS)      # 台帳の更新・対象の位置が変わったら作り直す
    known: dict[str, list[str]] = {}
    if cache is not None and cache.exists():
        try:
            c = json.loads(cache.read_text(encoding="utf-8"))
            if c.get("key") == key:
                known = c["types"]
        except (OSError, ValueError, KeyError):
            known = {}
    todo = [h for h in horse_ids if h not in known]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for h, ts in zip(todo, ex.map(lambda x: _mother_types_of(x, ledger_type), todo)):
                if ts is not None:                      # 5 世代血統のファイルが無い馬は保存しない（あとで取得されたら付く）
                    known[h] = ts
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"key": key, "types": known}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, cache)
    return {h: known[h] for h in horse_ids if known.get(h)}


def load_runs() -> tuple[pd.DataFrame, dict[str, dict], dict]:
    """走ごとの表と、小系統ごとの情報、その他の情報（ledger など）。"""
    parts = []
    for y in YEARS:
        p = FLAT / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "date", "venue", "surface", "distance", "race_name", "finish_position", "horse_id", "track_condition", "grade"]))
    r = pd.concat(parts, ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"]) & ~r["race_name"].fillna("").str.contains(_JUMP)].copy()
    r["pos"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r["distance"] = pd.to_numeric(r["distance"], errors="coerce")
    r = r[(r["pos"] > 0) & r["distance"].notna()].copy()       # 着順が付いた走だけ（競走中止・取消・除外は -1 で外れる）
    r["distance"] = r["distance"].astype(int)
    r["horse_id"] = r["horse_id"].astype(str)
    r["race_id"] = r["race_id"].astype(str)
    r["kai"] = r["race_id"].str[6:8].astype(int)
    r["day"] = r["race_id"].str[8:10].astype(int)
    r["date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    m = pd.read_parquet(SIRE_MAP, columns=["horse_id", "sire_id"]).astype(str).drop_duplicates("horse_id")
    c = pd.read_csv(COLORS, dtype=str)
    ok = c[c["status"] == "ok"].set_index("sire_id")
    r = r.merge(m, on="horse_id", how="left")
    r["line"] = r["sire_id"].map(ok["small_name"])
    r["father_large"] = r["sire_id"].map(ok["large_name"])
    r = r[r["line"].notna()].copy()
    marks = json.loads(MARKS.read_text(encoding="utf-8"))["marks"] if MARKS.exists() else {}
    o = {k for k, v in marks.items() if v.get("mark") == "o"}
    used = set(r["sire_id"])                                   # 母数は、この期間に産駒が走った父だけ（表の集計範囲に合わせる）
    cu = c[(c["status"] == "ok") & c["sire_id"].isin(used)].assign(o=lambda d: d["sire_id"].isin(o))
    meta = {s: {"sires": int(len(g)), "o": int(g["o"].sum()), "large": str(g["large_name"].mode().iat[0])} for s, g in cu.groupby("small_name")}
    r["cond"] = r["track_condition"].map({"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"})
    r["cls"] = r["grade"].replace({"L": "OP"})
    r.loc[~r["cls"].isin(CLASSES), "cls"] = None
    ledger_type = ok["type_name"].dropna().to_dict()
    return r, meta, {"ledger_type": ledger_type}


# ── 集計 ────────────────────────────────────────────────────────────────────
def _with_counts(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(c1=(d["pos"] == 1).astype(int), c2=(d["pos"] == 2).astype(int), c3=(d["pos"] == 3).astype(int), c4=(d["pos"] >= 4).astype(int))


def _sum(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return d.groupby(keys, as_index=False)[["c1", "c2", "c3", "c4"]].sum()


def _cnt(g: pd.DataFrame | None) -> list[int]:
    return [0, 0, 0, 0] if g is None or len(g) == 0 else [int(x) for x in g[["c1", "c2", "c3", "c4"]].sum()]


def prepare(runs: pd.DataFrame, rail: pd.DataFrame | None = None) -> pd.DataFrame:
    """行のキー（クラス・馬場状態・開催状態・会場・コース）を走ごとに付ける。"""
    r = _with_counts(runs)
    course = merged_courses(r)
    r["course"] = [course[(v, s, d)][0] for v, s, d in zip(r["venue"], r["surface"], r["distance"])]
    r["course_order"] = [course[(v, s, d)][1] for v, s, d in zip(r["venue"], r["surface"], r["distance"])]
    turf = r["surface"] == "芝"
    r["day_cat"] = np.where(~turf, None, np.where(r["day"] <= 2, DAY_CATS[0], np.where(r["day"] <= 4, DAY_CATS[1], np.where(r["day"] <= 6, DAY_CATS[2], DAY_CATS[3]))))
    r["letter"] = None
    r["rail_cat"] = None
    if rail is not None and len(rail):
        rc = rail_categories(r[turf][["date", "venue", "kai", "day"]], rail)                  # 芝のある日（コース区分が不明の日も含めて渡す）
        r = r.drop(columns=["letter", "rail_cat"]).merge(rc, on=["date", "venue"], how="left")
        r.loc[r["surface"] != "芝", ["letter", "rail_cat"]] = None      # 柵位置は芝だけ（同じ日・会場のダートに付かないように）
    return r


def build_rows(frames: list[tuple[pd.DataFrame, pd.DataFrame, str, list[str]]]) -> tuple[list[dict], list[dict]]:
    """行の定義と、各テーブルのセル。frames = [(列を持つ走, 合計用の走, 列名, 列の並び)]（テーブルごと）。返す: (rows, sections)。"""
    rows: list[dict] = []
    sections: list[dict] = []
    main = frames[0][1]

    def add(rid: str, kind: str, label: str, mask_fn, extra: dict | None = None) -> None:
        k = []
        for b, t, colcol, cols in frames:
            sb, st = mask_fn(b), mask_fn(t)
            by = {s: g for s, g in sb.groupby(colcol)} if len(sb) else {}
            gl = {fl: _cnt(g) for fl, g in st.groupby("father_large")} if "father_large" in st.columns and len(st) else {}
            k.append({"t": _cnt(st), "c": [_cnt(by.get(s)) for s in cols], "g": gl})
        if all(x["t"] == [0, 0, 0, 0] for x in k):
            return
        rows.append({"id": rid, "kind": kind, "label": label, "k": k, **(extra or {})})

    # クラス別
    for sf, sfl in (("芝", "芝"), ("ダート", "ダ")):
        ids = []
        for cl in CLASSES:
            add(f"class:{sf}:{cl}", "class", f"{sfl} {cl}" + ("（OP・L）" if cl == "OP" else ""), lambda d, sf=sf, cl=cl: d[(d["surface"] == sf) & (d["cls"] == cl)], {"sf": sf})
            ids.append(f"class:{sf}:{cl}")
        for lab, members in CLASS_SUMS:
            add(f"class:{sf}:{lab}", "class", f"{sfl} {lab}", lambda d, sf=sf, members=members: d[(d["surface"] == sf) & d["cls"].isin(members)], {"sf": sf, "sum": True})
            ids.append(f"class:{sf}:{lab}")
        sections.append({"title": f"クラス別（{sf}）", "rows": ids, "sf": sf})
    # 馬場状態別
    ids = []
    for sf, sfl, cond in COND_ROWS:
        add(f"cond:{sf}:{cond}", "cond", f"{sfl} {cond}", lambda d, sf=sf, cond=cond: d[(d["surface"] == sf) & (d["cond"] == cond)], {"sf": sf})
        ids.append(f"cond:{sf}:{cond}")
    sections.append({"title": "馬場状態別（全会場。良・稍重・重／不良）", "rows": ids})
    # 芝の開催状態
    ids = []
    for cat in DAY_CATS:
        add(f"day:{cat}", "state", cat, lambda d, cat=cat: d[d["day_cat"] == cat], {"sf": "芝"})
        ids.append(f"day:{cat}")
    sections.append({"title": "芝の開催状態（開催何日目。全年・全会場）", "rows": ids, "sf": "芝"})
    ids = []
    for cat in RAIL_CATS:
        add(f"rail:{cat}", "state", cat, lambda d, cat=cat: d[d["rail_cat"] == cat], {"sf": "芝"})
        ids.append(f"rail:{cat}")
    for L in LETTERS:
        add(f"letter:{L}", "state", f"{L}コース（全日程）", lambda d, L=L: d[d["letter"] == L], {"sf": "芝"})
        ids.append(f"letter:{L}")
    sections.append({"title": "芝の馬場の新しさ（柵位置 A〜D を考慮。開幕＝開催の最初の柵、柵替わり＝柵の位置が変わった後。コース区分の分かる日のみ）", "rows": ids, "sf": "芝"})
    # 会場別
    vsec = []
    for vi, v in enumerate(VENUES):
        if not (main["venue"] == v).any():
            continue
        for vsf, vlab in (("all", "全コース"), ("芝", "芝全体"), ("ダート", "ダート全体")):
            add(f"venue:{v}:{vsf}", "venue", f"{v} {vlab}", lambda d, v=v, vsf=vsf: d[(d["venue"] == v) & ((d["surface"] == vsf) if vsf != "all" else True)], {"v": vi, "vname": v, "vsf": vsf})
        courses = main[main["venue"] == v][["surface", "course", "course_order"]].drop_duplicates().sort_values(["surface", "course_order"], ascending=[False, True])
        for sf, cname in zip(courses["surface"], courses["course"]):
            add(f"course:{v}:{sf}:{cname}", "course", cname, lambda d, v=v, sf=sf, cname=cname: d[(d["venue"] == v) & (d["surface"] == sf) & (d["course"] == cname)], {"v": vi, "vname": v, "sf": sf})
        vsec.append(v)
    sections.append({"title": "会場別（＋で開くとコース別。距離の少ないコースは同じ会場・同じ芝/ダートの隣とまとめた）", "venues": True})
    return rows, sections


def col_refs(b: pd.DataFrame, t: pd.DataFrame, colcol: str, cols: list[str]) -> dict[str, dict]:
    """相対の印の基準: 列（系統）ごとの、芝全体・ダート全体・全体の [1着, 2着, 3着, 4着以下] と、行の合計に当たる全系統の値。
    行に芝/ダートがあればその芝/ダート全体、無ければ（会場の全コース）全体を、その系統の「ふだんの成績」とみなす。"""
    out = {}
    for key in ("芝", "ダート", "all"):
        sb = b if key == "all" else b[b["surface"] == key]
        st = t if key == "all" else t[t["surface"] == key]
        by = {s: g for s, g in sb.groupby(colcol)} if len(sb) else {}
        out[key] = {"t": _cnt(st), "c": [_cnt(by.get(s)) for s in cols]}
    return out


def model(runs: pd.DataFrame, meta: dict[str, dict], ledger_type: dict[str, str], rail: pd.DataFrame | None = None, mat: dict[str, list[str]] | None = None) -> dict:
    """画面に埋め込むデータ。tables = [父の系統, 系統配合]、rows = 行（各行にテーブルごとのセル k）、sections = 行の並び。"""
    r = prepare(runs, rail)
    # テーブル 1: 父の小系統（大系統ごとに、大系統の出走数の多い順 → 小系統の出走数の多い順）
    tot = r.groupby("line").size()
    lines = [s for s in tot.index if tot[s] >= MIN_LINE_RUNS]
    large = {s: meta.get(s, {}).get("large", "（不明）") for s in lines}
    gtot: dict[str, int] = {}
    for s in lines:
        gtot[large[s]] = gtot.get(large[s], 0) + int(tot[s])
    lines.sort(key=lambda s: (-gtot[large[s]], large[s], -int(tot[s])))
    cols1 = [{"name": s, "large": large[s], "sires": meta.get(s, {}).get("sires", 0), "o": meta.get(s, {}).get("o", 0), "runs": int(tot[s])} for s in lines]
    t1 = r[r["line"].isin(lines)]
    tables = [{"id": "line", "title": "父の系統", "cols": cols1}]
    frames = [(t1, r, "line", lines)]
    # テーブル 2: 父の大系統 × 母方の型
    if mat:
        rm = r[r["horse_id"].isin(mat)].copy()
        rm["mt"] = rm["horse_id"].map(mat)
        ex = rm.explode("mt")
        ex["pair"] = ex["father_large"] + "\t" + ex["mt"]
        pt = ex.groupby("pair").size()
        pairs = [p for p in pt.index if pt[p] >= MIN_PAIR_RUNS]
        fl_tot: dict[str, int] = {}
        for p in pairs:
            fl_tot[p.split("\t")[0]] = fl_tot.get(p.split("\t")[0], 0) + int(pt[p])
        pairs.sort(key=lambda p: (-fl_tot[p.split("\t")[0]], p.split("\t")[0], -int(pt[p])))
        cols2 = [{"name": p.split("\t")[1], "large": p.split("\t")[0], "sires": 0, "o": 0, "runs": int(pt[p])} for p in pairs]
        tables.append({"id": "mix", "title": "系統配合（父の大系統 × 母方の型）", "cols": cols2, "groups": {k: int(v) for k, v in rm.groupby("father_large").size().items()}, "note": "母方 = 母父・母母父・母母母父のいずれか。複数の型に当たる馬は、それぞれの列に数える（列どうしは重なる）。行の合計・基準は、母方の型が分かる馬の全走。"})
        frames.append((ex[ex["pair"].isin(pairs)], rm, "pair", pairs))
    rows, sections = build_rows(frames)
    for tb, fr in zip(tables, frames):
        tb["ref"] = col_refs(*fr)
    return {"tables": tables, "rows": rows, "sections": sections, "min_line_runs": MIN_LINE_RUNS, "classes": CLASSES}


# ── 画面の JS ─────────────────────────────────────────────────────────────
# 純粋な計算（率・縮め・色・並べ替え）。DOM に触れないので、単体でテストできる（tests/scripts/…）。
JS_PURE = r"""
var PURE=(function(){
  var K=30;                                                    // 出走数が少ないセルの率を、行の基準へ縮める強さ（疑似標本数）
  function cnt(c){return c[0]+c[1]+c[2]+c[3];}
  function pl(c){return c[0]+c[1]+c[2];}
  function xv(c,m){return m==='p'?pl(c):c[0];}
  function rate(c,m){var n=cnt(c);return n?xv(c,m)/n:NaN;}
  function shrunk(c,b,m){var n=cnt(c);return n?(xv(c,m)+K*b)/(n+K):NaN;}
  // 色・並べ替えの値。データが少なくても NaN にしない（行の基準へ縮める。出走 0 は基準そのもの）。行に全くデータが無いときだけ NaN
  function val(c,b,m){var n=cnt(c);return n?shrunk(c,b,m):b;}
  function level(c,b,m){var n=cnt(c);if(!n||!(b===b))return null;var d=shrunk(c,b,m)-b,s=(m==='p'?0.10:0.05);return Math.max(-1,Math.min(1,d/s));}
  function sum(cs){var t=[0,0,0,0];cs.forEach(function(c){for(var i=0;i<4;i++)t[i]+=c[i];});return t;}
  function sortIdx(vals,dir){                                    // dir=1 降順、-1 昇順。NaN は常に末尾。同値は元の並び
    var ix=vals.map(function(v,i){return i;});
    ix.sort(function(a,b){var x1=vals[a],y1=vals[b],nx=!(x1===x1),ny=!(y1===y1);
      if(nx&&ny)return a-b; if(nx)return 1; if(ny)return -1; return (dir*(y1-x1))||(a-b);});
    return ix;
  }
  var NEU=[36,45,66], BAD=[58,10,20], GOOD=[80,240,150];          // 中立(暗い青灰) → 不振(さらに暗い赤) / 好成績(明るい緑)。RGB の線形補間なので 0 で途切れない
  function mix(a,b,t){return [0,1,2].map(function(i){return Math.round(a[i]+(b[i]-a[i])*t);});}
  function color(s){                                             // s: -1(不振)〜+1(好成績)。明るいほど好成績、暗いほど不振
    var c=s>=0?mix(NEU,GOOD,s):mix(NEU,BAD,-s), lum=0.299*c[0]+0.587*c[1]+0.114*c[2];
    return {bg:'rgb('+c.join(',')+')',fg:lum>=120?'#0b1b12':'#e5e9f5',hi:s>=0.5,lum:lum};
  }
  // 列の並び: 大系統の枠を保つ。rowCounts = 基準の行の各列の [c1..c4]、base = その行の基準の率（無ければ元の並び）
  function orderCols(cols,rowCounts,base,m,dir,groupCounts,flat){
    var gs=[],gi={};cols.forEach(function(c,i){if(!(c.large in gi)){gi[c.large]=gs.length;gs.push({large:c.large,idx:[]});}gs[gi[c.large]].idx.push(i);});
    if(!rowCounts)return gs;
    if(flat){                                                    // 大系統の枠をこえて、全ての列を 1 つの並びにする。1 階層目は、隣り合う同じ大系統ごとにまとめる
      var all=cols.map(function(c,i){return i;}), v2=all.map(function(i){return val(rowCounts[i],base,m);}), ord=sortIdx(v2,dir).map(function(k){return all[k];});
      var segs=[];ord.forEach(function(i){var L=cols[i].large;if(segs.length&&segs[segs.length-1].large===L)segs[segs.length-1].idx.push(i);else segs.push({large:L,idx:[i]});});
      return segs;
    }
    gs.forEach(function(g){var v=g.idx.map(function(i){return val(rowCounts[i],base,m);});var o=sortIdx(v,dir);g.idx=o.map(function(k){return g.idx[k];});});
    var gv=gs.map(function(g){return val(groupCounts?(groupCounts[g.large]||[0,0,0,0]):sum(g.idx.map(function(i){return rowCounts[i];})),base,m);});   // groupCounts があれば（配合）重ならない実数で大系統を並べる
    return sortIdx(gv,dir).map(function(k){return gs[k];});
  }
  // 相対の印: その系統が「ふだん（芝全体など）」どれだけ走るかと、この行の全系統の率から、この行での見込みの率を出す（対数オッズで足し引き）。
  // 見込み = 行の全系統の率 × (その系統のふだんの率 ÷ 全系統のふだんの率) をオッズで。どの条件でも強い系統は見込みも高いので、印は付かない。
  function lg(p){p=Math.min(Math.max(p,1e-4),1-1e-4);return Math.log(p/(1-p));}
  function expect(rowBase,colRef,refBase){ if(!(rowBase===rowBase&&colRef===colRef&&refBase===refBase))return NaN; return 1/(1+Math.exp(-(lg(rowBase)+lg(colRef)-lg(refBase)))); }
  // 見込み e からの上振れ: lift = 見込みへ縮めた率 − e（色に使う）、raw = 実際の率 − e、z = 二項の標準化（印に使う）
  // 上側の裾の確率 P(X ≥ x)、X ~ 二項(n, e)。見込みの数（n·e）が小さいセルでも正規近似のように甘くならないよう、正確に足す
  function lgam(z){var g=[76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5],x=z,y=z,t=x+5.5;t-=(x+0.5)*Math.log(t);var a=1.000000000190015;for(var j=0;j<6;j++)a+=g[j]/++y;return -t+Math.log(2.5066282746310005*a/x);}
  function upper(x,n,e){ if(x<=0)return 1; if(x>n||e<=0)return 0; if(e>=1)return 1;
    var lc=lgam(n+1)-lgam(x+1)-lgam(n-x+1)+x*Math.log(e)+(n-x)*Math.log(1-e), t=Math.exp(lc), s=t, r=e/(1-e);
    for(var k=x;k<n;k++){ t*=(n-k)/(k+1)*r; s+=t; if(t<s*1e-12)break; } return Math.min(1,s); }
  // 見込み e からの上振れ: lift = 見込みへ縮めた率 − e（色・並べ替え）、raw = 実際の率 − e、pv = 上側の確率（印）
  function rel(c,e,m){ var n=cnt(c); if(!n||!(e===e))return null; var x=xv(c,m);
    return {e:e,lift:(x+K*e)/(n+K)-e,raw:x/n-e,pv:upper(x,n,e),n:n}; }
  function relLevel(r,m){ if(!r)return null; var s=(m==='p'?0.06:0.03); return Math.max(-1,Math.min(1,r.lift/s)); }
  return {K:K,cnt:cnt,rate:rate,shrunk:shrunk,val:val,level:level,sum:sum,sortIdx:sortIdx,color:color,orderCols:orderCols,expect:expect,rel:rel,relLevel:relLevel,upper:upper,PV:{2:0.02275,3:0.00135,4:0.0000317}};
})();
"""

JS_UI = r"""
(function(){
  var D=JSON.parse(document.getElementById('data').textContent), P=PURE;
  var st={tab:0,m:'p',sm:'',flat:true,minN:30,sf:'',onlyO:false,open:{},rs:null,cs:null,showW:true,showP:true,showK:true,cm:'rel',zt:2};
  var tbl=document.getElementById('m'), VN=[]; D.rows.forEach(function(r){ if(r.vname&&VN.indexOf(r.vname)<0) VN.push(r.vname); });
  function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
  function T(){return D.tables[st.tab];}
  function SM(){return st.sm||st.m;}                 // 並べ替えの率（勝率 'w'／複勝率 'p'）。色の基準とは別に選べる（空なら色の基準と同じ）
  function SML(){return SM()==='p'?'複勝率':'勝率';}
  function isO(c){return c.sires>0&&c.o*2>=c.sires;}
  function visCols(){ var vis={}; T().cols.forEach(function(c,i){ if(!st.onlyO||st.tab>0||isO(c)) vis[i]=1; }); return vis; }
  function rowById(id){ for(var i=0;i<D.rows.length;i++) if(D.rows[i].id===id) return D.rows[i]; return null; }
  function K(r){return r.k[st.tab];}
  // 列で行を並べ替えるときの値（key: 列の添字 / 'g:大系統' / 'tot'）。データが少なくても、行の基準へ縮めた値で並べる
  function rowVal(r,key){
    var m=SM(),k=K(r),b=P.rate(k.t,m),vis=visCols();
    if(key==='tot') return b;
    if(typeof key==='string'&&key.indexOf('c:')===0){                       // 割れた見出し: 押した見出しの下の列だけの合計
      var ids=key.slice(2).split(',').map(Number).filter(function(i){return vis[i];});
      return P.val(P.sum(ids.map(function(i){return k.c[i];})),b,m); }
    if(typeof key==='string'&&key.indexOf('g:')===0){ var L=key.slice(2),cs=[];
      if(st.tab>0&&k.g) return P.val(k.g[L]||[0,0,0,0],b,m);              // 配合: 列は重なるので、父の大系統の実数（展開前）で並べる
      T().cols.forEach(function(c,i){ if(c.large===L&&vis[i]) cs.push(k.c[i]); }); return P.val(P.sum(cs),b,m); }
    if(st.cm==='rel'&&typeof key==='number'){ var rl=relOfRow(r,key,k.c[key],b,m); return rl?rl.lift:NaN; }
    return P.val(k.c[key],b,m);
  }
  function relOfRow(r,i,c,b,m){ var R=T().ref&&T().ref[r.sf||(r.vsf&&r.vsf!=='all'?r.vsf:'all')]; if(!R) return null;
    var rc=[0,1,2,3].map(function(j){return R.c[i][j]-c[j];}), rt=[0,1,2,3].map(function(j){return R.t[j]-K(r).t[j];}); if(!P.cnt(rc)||!P.cnt(rt)){ rc=R.c[i]; rt=R.t; }
    return P.rel(c,P.expect(b,P.rate(rc,m),P.rate(rt,m)),m); }
  function fixState(){
    var vis=visCols(), want=st.sf||'all';
    if(st.rs&&typeof st.rs.key==='number'&&!vis[st.rs.key]) st.rs=null;
    if(st.rs&&typeof st.rs.key==='string'&&st.rs.key.indexOf('c:')===0&&!st.rs.key.slice(2).split(',').every(function(i){return vis[+i];})) st.rs=null;
    if(st.rs&&typeof st.rs.key==='string'&&st.rs.key.indexOf('g:')===0&&!T().cols.some(function(c,i){return vis[i]&&'g:'+c.large===st.rs.key;})) st.rs=null;
    if(st.rs&&typeof st.rs.key==='number'&&st.rs.key>=T().cols.length) st.rs=null;
    if(st.cs){
      var id=st.cs.id, p=id.split(':'), r=rowById(id);
      if(!r) st.cs=null;
      else if(p[0]==='venue'&&p[2]!==want){ id='venue:'+p[1]+':'+want; if(!rowById(id)) st.cs=null; else st.cs.id=id; }
      else if(r.sf&&want!=='all'&&r.sf!==want) st.cs=null;
    }
  }
  function orderRows(rows){
    if(!st.rs) return rows.slice();
    var v=rows.map(function(r){return rowVal(r,st.rs.key);}), o=P.sortIdx(v,st.rs.dir); return o.map(function(i){return rows[i];});
  }
  function layout(){
    var want=st.sf||'all', out=[];
    D.sections.forEach(function(sec){
      if(sec.venues){
        out.push({title:sec.title});
        var blocks=VN.map(function(v){ return {v:v,head:D.rows.filter(function(r){return r.kind==='venue'&&r.vname===v&&r.vsf===want;})[0],
            courses:D.rows.filter(function(r){return r.kind==='course'&&r.vname===v&&(want==='all'||r.sf===want);})}; }).filter(function(b){return b.head;});
        if(st.rs){ var bv=blocks.map(function(b){return rowVal(b.head,st.rs.key);}); blocks=P.sortIdx(bv,st.rs.dir).map(function(i){return blocks[i];}); }
        blocks.forEach(function(b){ out.push({row:b.head,toggle:true,v:b.v}); if(st.open[b.v]) orderRows(b.courses).forEach(function(r){out.push({row:r});}); });
        return;
      }
      if(sec.sf&&want!=='all'&&sec.sf!==want) return;
      var rs=sec.rows.map(rowById).filter(function(r){return r&&(want==='all'||!r.sf||r.sf===want);});
      if(!rs.length) return;
      out.push({title:sec.title});
      // クラス別は、並べ替え中でも「基準のクラス行」と「集計行」を分けて並べる（集計行は末尾に固定）
      var base=rs.filter(function(r){return !r.sum;}), sums=rs.filter(function(r){return r.sum;});
      orderRows(base).concat(sums).forEach(function(r){out.push({row:r});});
    });
    return out;
  }
  function arrow(on,dir){ return on?(dir===1?' ▼':' ▲'):''; }
  function render(){
    fixState();
    var cols=T().cols, rowC=st.cs?rowById(st.cs.id):null, vis=visCols();
    var vi=cols.map(function(c,i){return i;}).filter(function(i){return vis[i];});
    var gs=P.orderCols(vi.map(function(i){return cols[i];}),rowC?vi.map(function(i){return K(rowC).c[i];}):null,rowC?P.rate(K(rowC).t,SM()):NaN,SM(),st.cs?st.cs.dir:1,(rowC&&st.tab>0&&K(rowC).g)?K(rowC).g:null,st.flat)
      .map(function(g){return {large:g.large,idx:g.idx.map(function(k){return vi[k];})};});
    var h1='<tr><th class="sticky" rowspan="2">行<div class="hint">行名を押す＝列を並べ替え</div></th><th class="tot" rowspan="2" data-key="tot">全体'+arrow(st.rs&&st.rs.key==='tot',st.rs&&st.rs.dir)+'</th>';
    var h2='<tr>', order=[];
    gs.forEach(function(g){
      var ix=g.idx.filter(function(i){return vis[i];}); if(!ix.length) return;
      var whole=ix.length===cols.filter(function(c,i){return c.large===g.large&&vis[i];}).length;      // 見えている大系統の全列が並んでいるときだけ「全体」（割れた見出しは、その枠の列だけ）
      var runs=(T().groups&&T().groups[g.large]!==undefined&&whole&&!st.onlyO)?T().groups[g.large]:ix.reduce(function(a,i){return a+cols[i].runs;},0), gk=whole?'g:'+g.large:'c:'+ix.join(',');
      h1+='<th class="grp" colspan="'+ix.length+'" data-key="'+esc(gk)+'">'+esc(g.large)+'<span class="sub"> '+((st.tab>0&&!whole)?'':(whole?'全体 ':'この枠 ')+runs.toLocaleString()+'走')+'</span>'+arrow(st.rs&&st.rs.key===gk,st.rs&&st.rs.dir)+'</th>';
      ix.forEach(function(i){ var c=cols[i]; order.push(i);
        h2+='<th class="line'+(isO(c)?' o':'')+'" data-key="'+i+'"><div class="nm">'+esc(c.name)+arrow(st.rs&&st.rs.key===i,st.rs&&st.rs.dir)+'</div><div class="sub">'+(c.sires?'○ '+c.o+'／'+c.sires+'頭<br>':'')+'全体 '+c.runs.toLocaleString()+'走</div></th>'; });
    });
    h1+='</tr>';h2+='</tr>';
    var b='', lay=layout(), n=2+order.length;
    var REFN={'芝':'芝全体','ダート':'ダート全体','all':'全体'};
    function refKey(r){ return r.sf||(r.vsf&&r.vsf!=='all'?r.vsf:'all'); }
    function sub(a,b){ return [a[0]-b[0],a[1]-b[1],a[2]-b[2],a[3]-b[3]]; }
    // 相対: その系統のふだん（行の芝/ダート全体）の成績から見込んだ率と比べる。印は、出走数が足りて、見込みを z 以上に上回るセル
    // 系統の「ふだん」は、このセル自身の走を除いた残り（大きい行で見込みが実際に引き寄せられないように）。残りが無ければ除かない
    function relOf(r,i,c,base){ var R=T().ref&&T().ref[refKey(r)]; if(!R) return null;
      var rc=sub(R.c[i],c), rt=sub(R.t,K(r).t); if(!P.cnt(rc)||!P.cnt(rt)){ rc=R.c[i]; rt=R.t; }
      return P.rel(c,P.expect(base,P.rate(rc,st.m),P.rate(rt,st.m)),st.m); }
    function cell(c,base,tot,r,i){
      var nn=P.cnt(c); if(!nn) return '<td class="c e"></td>';
      var rl=tot?null:relOf(r,i,c,base), s=tot?null:(st.cm==='rel'?P.relLevel(rl,st.m):P.level(c,base,st.m)), col=s===null?null:P.color(s), w=c[0]/nn, p=P.rate(c,'p'), low=!tot&&nn<st.minN;
      // ★ = 上側の確率 ≤ 0.00135（z≥3 相当。確か）、☆ = ≤ 0.023（z≥2 相当。参考）。出走 30 走以上（画面の低信頼の設定より下げない）。
      // 会場の全コースの行は、芝・ダートの割合の違いが混ざるので印を付けない（会場の芝全体・ダート全体の行で見る）
      // 差の大きさも求める（複勝率 3pt・勝率 1.5pt 以上）。出走の多い系統で、わずかな差まで確かになって印が並ばないように
      var mk=!tot&&!low&&nn>=30&&r.vsf!=='all'&&rl&&rl.raw>=(st.m==='p'?0.03:0.015)&&rl.pv<=P.PV[st.zt], strong=mk&&rl.pv<=P.PV[3];
      var style=col?' style="background:'+col.bg+';color:'+col.fg+'"':'';
      var cls='c'+(tot?' t':'')+(low?' low':'')+(mk?(strong?' rel':' rel2'):'')+((st.cm==='row'&&col&&col.hi&&!low)?' hi':'');
      var ML=st.m==='p'?'複勝率':'勝率';
      var tip=nn+'走'+(low?'（出走数が少ない。率は基準へ寄せた色）':'')+(rl?'。'+ML+': この系統の'+REFN[refKey(r)]+'の成績から見込む '+(100*rl.e).toFixed(1)+'% → 実際 '+(100*(rl.raw+rl.e)).toFixed(1)+'%（'+(rl.raw>=0?'+':'')+(100*rl.raw).toFixed(1)+'pt、偶然でこれ以上になる確率 '+(rl.pv<0.001?rl.pv.toExponential(1):rl.pv.toFixed(3))+'）':'');
      return '<td class="'+cls+'"'+style+' title="'+esc(tip)+'"><div class="r">'+(mk?'<b class="mk">'+(strong?'★':'☆')+'+'+(100*rl.raw).toFixed(1)+'pt</b>':'')+(st.showW?'<span>勝 '+(100*w).toFixed(1)+'%</span>':'')+(st.showP?'<span>複 '+(100*p).toFixed(1)+'%</span>':'')+'</div>'
        +(st.showK?'<div class="k">'+c.join('-')+'<i>（'+nn+'）</i>'+(low?'<b class="lw">※</b>':'')+'</div>':'')+'</td>';
    }
    lay.forEach(function(e){
      if(e.title){ b+='<tr class="gt"><th class="sticky" colspan="'+n+'">'+esc(e.title)+'</th></tr>'; return; }
      var r=e.row, k=K(r), base=P.rate(k.t,st.m), sorted=st.cs&&st.cs.id===r.id;
      var lab=(e.toggle?'<button type="button" class="tg" data-v="'+esc(e.v)+'">'+(st.open[e.v]?'－':'＋')+'</button> ':'')+esc(r.label)+arrow(sorted,st.cs&&st.cs.dir);
      b+='<tr class="'+r.kind+(r.sum?' sumrow':'')+'"><th class="sticky rl" data-row="'+esc(r.id)+'">'+lab+'</th>'+cell(k.t,base,true);
      order.forEach(function(i){ b+=cell(k.c[i],base,false,r,i); });
      b+='</tr>';
    });
    tbl.innerHTML='<thead>'+h1+h2+'</thead><tbody>'+b+'</tbody>';
    var th1=tbl.querySelector('thead tr'); tbl.style.setProperty('--h1', th1.offsetHeight+'px');
    var info=[]; if(st.rs){ var kk=st.rs.key; info.push('行の並べ替え（'+SML()+'）: '+(kk==='tot'?'全体':(typeof kk==='string'&&kk.indexOf('c:')===0)?'見出しの枠の列（'+kk.slice(2).split(',').map(function(i){return T().cols[+i].name;}).join('・')+'）':typeof kk==='number'?(st.tab>0?T().cols[kk].large+' × ':'')+T().cols[kk].name:String(kk).slice(2))+(st.rs.dir===1?' ▼降順':' ★昇順')); }
    if(st.cs){ var rr=rowById(st.cs.id); info.push('列の並べ替え（'+SML()+(st.flat?'・大系統をまたぐ':'・大系統の枠の中')+'）: '+(rr?((rr.kind==='course'?rr.vname+' ':'')+rr.label):'')+(st.cs.dir===1?' ▼降順':' ★昇順')); }
    document.getElementById('sortinfo').textContent=info.join(' ／ ');
    var note=T().note||''; document.getElementById('tabnote').textContent=note;
    [].slice.call(document.querySelectorAll('.tab')).forEach(function(t){ t.classList.toggle('on', +t.dataset.tab===st.tab); });
  }
  function cycle(cur,key,idKey){ // 降順 → 昇順 → 解除
    var o={}; if(cur&&cur[idKey]===key){ if(cur.dir===1){ o.dir=-1; o[idKey]=key; return o; } return null; }
    o.dir=1; o[idKey]=key; return o;
  }
  tbl.addEventListener('click',function(e){
    var tg=e.target.closest('.tg'); if(tg){ var v=tg.dataset.v; st.open[v]=!st.open[v]; render(); return; }
    var th=e.target.closest('th[data-key]'); if(th){ var k=th.dataset.key; if(/^\d+$/.test(k)) k=+k; st.rs=cycle(st.rs,k,'key'); render(); return; }
    var rl=e.target.closest('th[data-row]'); if(rl){ st.cs=cycle(st.cs,rl.dataset.row,'id'); render(); return; }
  });
  [].slice.call(document.querySelectorAll('.tab')).forEach(function(t){ t.addEventListener('click',function(){ st.tab=+t.dataset.tab; st.rs=null; st.cs=null; render(); }); });
  function bind(id,fn,ev){ document.getElementById(id).addEventListener(ev||'change',function(){fn(this);render();}); }
  bind('metric',function(x){st.m=x.value;}); bind('cmode',function(x){st.cm=x.value;}); bind('zt',function(x){st.zt=+x.value;}); bind('sortmetric',function(x){st.sm=x.value;}); bind('minn',function(x){st.minN=Math.max(1,+x.value||1);},'input');
  bind('sw',function(x){st.showW=x.checked;}); bind('sp',function(x){st.showP=x.checked;}); bind('sk',function(x){st.showK=x.checked;});
  bind('onlyo',function(x){st.onlyO=x.checked;}); bind('flat',function(x){st.flat=x.checked;}); bind('sf',function(x){st.sf=x.value;});
  document.getElementById('open').addEventListener('click',function(){ VN.forEach(function(v){st.open[v]=true;}); render(); });
  document.getElementById('close').addEventListener('click',function(){ st.open={}; render(); });
  document.getElementById('reset').addEventListener('click',function(){ st.rs=null; st.cs=null; render(); });
  render();
})();
"""

CSS = """
:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e5e9f5; --muted:#9aa7c7; --accent:#7aa9ff; --o:#34d399; }
body { margin:0; background:var(--bg); color:var(--text); font:13px/1.45 "Noto Sans JP",system-ui,sans-serif; }
main { padding:16px 14px 60px; } h1 { font-size:20px; margin:0 0 4px; } .muted { color:var(--muted); } p.lead { max-width:1240px; margin:4px 0 8px; color:var(--muted); }
code { background:#1c2a4d; padding:1px 6px; border-radius:5px; font-size:12px; }
.tabs { display:flex; gap:6px; margin:8px 0 0; } .tab { background:var(--panel); color:var(--muted); border:1px solid var(--border); border-bottom:none; border-radius:8px 8px 0 0; padding:6px 16px; cursor:pointer; font-size:13.5px; } .tab.on { color:var(--text); font-weight:700; background:#1d2d52; }
#tabnote { color:var(--muted); font-size:12px; margin:2px 0 4px; min-height:16px; }
.bar { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin:8px 0; position:sticky; top:0; z-index:20; background:var(--bg); padding:6px 0; border-bottom:1px solid var(--border); }
.bar select,.bar input[type=number],.bar button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:13px; }
.bar button { cursor:pointer; } .bar button:hover { border-color:var(--accent); }
.legend { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); } .legend i { display:inline-block; width:150px; height:12px; border-radius:6px; background:linear-gradient(90deg,rgb(58,10,20),rgb(36,45,66),rgb(80,240,150)); } .sortinfo { color:var(--accent); font-size:12px; }
.wrap { overflow:auto; max-height:calc(100vh - 200px); border:1px solid var(--border); border-radius:8px; }
table { border-collapse:separate; border-spacing:0; }
th,td { border-bottom:1px solid var(--border); border-right:1px solid var(--border); padding:3px 6px; }
thead th { position:sticky; background:var(--panel); vertical-align:bottom; z-index:3; cursor:pointer; user-select:none; }
thead tr:first-child th { top:0; } thead tr:nth-child(2) th { top:var(--h1,30px); }
thead th.sticky { left:0; z-index:6; width:210px; min-width:210px; max-width:210px; cursor:default; } thead th.tot { left:210px; z-index:5; min-width:96px; }
thead th.grp { text-align:center; font-size:13px; } thead th:hover { background:#1d2d52; }
th.line { min-width:96px; } th.line .nm { font-weight:700; } th.line .sub,.grp .sub { font-weight:400; color:var(--muted); font-size:11px; } th.line.o .nm { color:var(--o); } th.line.o { box-shadow:inset 0 -3px 0 var(--o); }
.hint { font-weight:400; color:var(--muted); font-size:11px; }
tbody th.sticky { position:sticky; left:0; background:var(--panel); z-index:2; text-align:left; white-space:nowrap; cursor:pointer; width:210px; min-width:210px; max-width:210px; overflow:hidden; text-overflow:ellipsis; }
tbody th.sticky:hover { background:#1d2d52; } tr.gt th { background:var(--bg) !important; color:var(--accent); font-size:12px; padding:8px 6px 3px; cursor:default; white-space:normal !important; max-width:none !important; width:auto !important; overflow:visible !important; }
td.c { text-align:center; min-width:96px; font-variant-numeric:tabular-nums; } td.c .k { font-size:11px; opacity:.85; white-space:nowrap; } td.c .k i { font-style:normal; opacity:.8; }
td.c .r span { margin:0 3px; } td.c.low { outline:1px dashed rgba(229,233,245,.45); outline-offset:-3px; } td.c.e { background:transparent; } .lw { color:#fbbf24; margin-left:3px; }
td.c.t { position:sticky; left:210px; z-index:1; background:#17264a; border-right:2px solid var(--border); }
td.c.hi { box-shadow:inset 0 0 0 2px rgba(190,255,215,.9); font-weight:700; } td.c.rel { box-shadow:inset 0 0 0 2px #fbbf24; font-weight:700; } td.c.rel2 { outline:1px dashed #fbbf24; outline-offset:-3px; } .mk { display:block; color:#fbbf24; font-size:11px; text-shadow:0 0 2px #000; }
tr.venue th.sticky,tr.venue td { font-weight:600; } tr.course th.sticky { padding-left:28px; font-weight:400; } tr.cond th.sticky,tr.class th.sticky,tr.state th.sticky { padding-left:14px; font-weight:600; } tr.sumrow th.sticky { color:var(--accent); }
.tg { width:22px; height:20px; padding:0; line-height:1; background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:5px; cursor:pointer; }
"""

TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>父の系統・系統配合 × クラス・舞台 成績マトリクス</title><style>%CSS%</style></head><body><main>
<h1>父の系統・系統配合 × クラス・舞台 成績マトリクス</h1>
<p class="lead">作成: %NOW%。2016〜2026 年の平地 %NRUNS% 走（父の系統が空欄の馬は除外）。タブで「父の系統」（大系統→小系統）と「系統配合」（父の大系統→母方の型）を切り替える。
行は<b>クラス別（芝・ダート。上位クラスの集計行つき）</b>→馬場状態→芝の開催状態（開催日数・柵位置を考慮した馬場の新しさ）→会場→コース。「上のクラスで戦えるレベルの馬を出せるか」は、クラス別の行（3勝以上・OP以上・重賞）で見る。
セルは「勝率・複勝率」と「1着-2着-3着-4着以下（出走数）」。<b>色と ★ は「その系統の中での得意・不得意」</b>: 系統ごとに、ふだん（同じ芝/ダートの全体。会場の全コースの行は全体）の成績と、その行の全系統の率から、その行での見込みの率を出し（対数オッズで足し引き）、見込みとの差で色を付ける（明るい緑ほど得意、暗いほど苦手。複勝率は ±6pt、勝率は ±3pt で最大）。
どの条件でも強い系統（例: ディープ系）は見込みも高いので、強いだけでは緑・★にならない。<b>★</b>（黄色の枠と「★+差pt」）は、30 走以上で、見込みを複勝率 3pt（勝率 1.5pt）以上、確かに上回るセル（正確な二項の上側の確率 ≤ 0.00135。z≥3 相当）。<b>☆</b>（黄色の点線）は ≤ 0.023（z≥2 相当）の参考（セルが数千あるので、偶然のものも混じる）。「得意の印」で ★ だけにできる。
会場の「全コース」の行は芝・ダートの割合の違いが混ざるので印を付けない（会場の芝全体・ダート全体の行で見る）。系統の「ふだん」は、そのセル自身の走を除いた残り。色が相対のとき、列の見出しで行を並べ替えると、見込みからの上振れの順になる（行名で列を並べ替えるときは率の順）。セルにマウスを載せると見込みと実際の率が出る。
「色」を「行の全系統との差（絶対）」にすると、従来の色（同じ行の全体の率との差。複勝率 ±10pt・勝率 ±5pt。水準を明確に上回るセルは白い枠）。出走数が少ないセルは率を基準に寄せて色を付け、<b>※</b>と点線で低信頼を示す（★は付けない）。<br>
<b>並べ替え</b>: 列の見出し（小系統・大系統・「全体」）を押すと、その列の率で<b>行</b>を並べ替え（降順 → 昇順 → 解除）。行名を押すと、その行の率で<b>列</b>を並べ替え（既定は大系統をまたいで全ての列を 1 つの並びに。「大系統をまたいで列を並べ替える」を外すと、大系統の枠の中で並べ、大系統どうしも並べる）。並べ替えの率は「並べ替えの基準」（複勝率／勝率。色の基準とは別に選べる）。データが少ないセルも、行の基準に寄せた値で並べる（NaN にしない）。<br>
<b>集計の範囲</b>: 競走中止・取消・除外は数えない（着順が付いた走だけ。全体の約 0.4%）。コースは「芝・ダートと距離」で、同じ会場の内回り・外回りは区別できず 1 行にまとめている。レース数が少ない距離は、同じ会場・同じ芝/ダートの隣の距離とまとめた（例: 芝 2500m 以上）。
「全体」の列と色の基準は、列に出していない小系統（出走数 %MINRUNS% 走未満）を含み、父の系統が空欄の馬を含まない。柵位置（A〜D）は %RAILNOTE%。「開幕」は開催回の最初の柵（前の開催から同じ柵のまま続く場合も、開催回の最初を開幕とみなす）。開催の初日のコース区分が取れなかった回は、柵が替わった日が特定できるまで行に入らない。生成: <code>src/scripts/docs/build_sire_line_course_matrix_html.py</code>。</p>
<div class="tabs">%TABS%</div><div id="tabnote"></div>
<div class="bar">
<label>色の基準 <select id="metric"><option value="p">複勝率</option><option value="w">勝率</option></select></label>
<label title="相対: その系統のふだん（同じ芝/ダート全体）の成績から見込んだ率との差。どの条件でも強い系統は緑になりにくい。行の全系統との差: 従来の色（強い系統はどの行でも緑）">色 <select id="cmode"><option value="rel">系統の中での得意・不得意（相対）</option><option value="row">行の全系統との差（絶対）</option></select></label>
<label title="見込みをどれだけ確かに上回ったら印を付けるか（正確な二項の上側の確率）。★ = z≥3 相当、☆ = z≥2 相当（行 × 系統で数千セルあるので、☆には偶然のものも混じる）">得意の印 <select id="zt"><option value="2" selected>★（z≥3）と ☆（z≥2、参考）</option><option value="3">★だけ（z≥3）</option><option value="4">z≥4 だけ（厳しい）</option></select></label>
<label>並べ替えの基準 <select id="sortmetric"><option value="">色の基準と同じ</option><option value="p">複勝率</option><option value="w">勝率</option></select></label>
<label>低信頼の目印を付ける出走数 <input id="minn" type="number" min="1" value="30" style="width:64px"></label>
<label><input type="checkbox" id="sw" checked> 勝率</label><label><input type="checkbox" id="sp" checked> 複勝率</label><label><input type="checkbox" id="sk" checked> 着別カウント</label>
<label><input type="checkbox" id="onlyo"> ○の小系統だけ（父の系統）</label>
<label title="行名を押して列を並べ替えるとき、大系統の枠をこえて全ての列を 1 つの並びにする（1 階層目の見出しは、隣り合う同じ大系統ごと）"><input type="checkbox" id="flat" checked> 大系統をまたいで列を並べ替える</label>
<label>芝・ダート <select id="sf"><option value="">芝・ダート</option><option value="芝">芝のみ</option><option value="ダート">ダートのみ</option></select></label>
<button type="button" id="open">会場をすべて開く</button><button type="button" id="close">すべて閉じる</button><button type="button" id="reset">並べ替えを解除</button>
<span class="legend">苦手・不振<i></i>得意・好成績</span><span id="sortinfo" class="sortinfo"></span>
</div>
<div class="wrap"><table id="m"></table></div>
<script type="application/json" id="data">%DATA%</script>
<script>%PURE%</script>
<script>%UI%</script>
</main></body></html>
"""


def build(runs: pd.DataFrame, meta: dict[str, dict], ledger_type: dict[str, str], rail: pd.DataFrame | None = None, mat: dict[str, list[str]] | None = None) -> str:
    """走ごとの表から HTML を作る。"""
    d = model(runs, meta, ledger_type, rail, mat)
    data = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")      # <script> や <!-- の途中で HTML の解析が切れないように（JSON としては同じ値）
    tabs = "".join(f'<button type="button" class="tab{" on" if i == 0 else ""}" data-tab="{i}">{t["title"]}</button>' for i, t in enumerate(d["tables"]))
    n_rail = int(rail["date"].nunique()) if rail is not None and len(rail) else 0
    by_year = {} if not n_rail else {y: int(g["date"].nunique()) for y, g in rail.assign(y=rail["date"].str[:4]).groupby("y")}
    return (TEMPLATE.replace("%CSS%", CSS).replace("%NOW%", datetime.now(JST).strftime("%Y-%m-%d %H:%M")).replace("%NRUNS%", f"{len(runs):,}")
            .replace("%MINRUNS%", str(MIN_LINE_RUNS)).replace("%RAILNOTE%", ("netkeiba のレース一覧から取った開催日ごとの値で、コース区分が分かる開催日だけ使う（日数: " + "、".join(f"{y}年 {c}" for y, c in sorted(by_year.items())) + "）。コース区分が取れなかった日の走は、柵位置の行に入らない") if n_rail else "未取得（柵位置の行は空）")
            .replace("%TABS%", tabs).replace("%PURE%", JS_PURE).replace("%UI%", JS_UI).replace("%DATA%", data))     # データは最後に差し込む（データ内の %…% を置換しない）


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    runs, meta, extra = load_runs()
    rail = load_rail()
    mat = mother_types(sorted(runs["horse_id"].unique()), extra["ledger_type"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(runs, meta, extra["ledger_type"], rail, mat), encoding="utf-8")
    print(json.dumps({"runs": len(runs), "horses": int(runs["horse_id"].nunique()), "mat_horses": len(mat), "rail_dates": int(rail["date"].nunique()), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
