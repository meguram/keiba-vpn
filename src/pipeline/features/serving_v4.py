"""v4 の推論用: 馬主系・馬自身 8 群・出走メンバー/コース環境 7 群・種牡馬 ID×条件 の列を、当日の出走馬に付ける。

方針（二重実装にしない）: 学習側（factory_conn / factory_self / factory_field / factory_ped）の関数を、そのまま
「履歴（その日より厳密に前の確定結果）+ 当日の出走馬の行（結果は未確定）」に対して呼び、当日の行だけを返す。どの関数も
『日付が厳密に前』の as-of なので、当日の行（結果なし）を足しても履歴側の値は変わらない（tests/pipeline/test_serving_v4.py が検査）。
履歴は起動時に読み、日次の状態更新は ``update_serving_state`` が書く表（馬主・産地のプロフィール、a_style / f3_r の履歴）で行う。

学習時の定数（g_attrs の産地コード上位 15 と平均）は学習期間（2020〜2025）の値に固定する（``frozen``）。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.features import factory_conn as fc
from src.pipeline.features import factory_field as ff
from src.pipeline.features import factory_ped as fp
from src.pipeline.features import factory_self as fs

logger = logging.getLogger("pipeline.features.serving_v4")

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
WORK = ROOT / "data/local/modeling/training_pipeline"
FAC = WORK / "factory"
STATE_DIR = Path(__import__("os").environ.get("KEIBA_INFERENCE_DIR") or ROOT / "data/local/inference") / "state"
KEY = ["race_id", "horse_id"]

SELF8 = ["change", "experience", "lags_margin", "pair", "rest", "seq_stats", "timedev", "attrs"]
FIELD7_TARGET_ONLY = ["mem_strength", "mem_demo", "mem_conn", "calendar_env"]      # 当日の行だけで決まる（calendar_env は前日の馬場の履歴も使う）
TRAIN_LAST_YEAR = 2025


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s)


def make_fast_asof(day: pd.Timestamp):
    """``stage_models.asof_group_mean`` の、当日 1 日分の行専用の高速版。

    当日の行（date >= day）の値は『date < day の全行の合計・件数』と『その全行の全体平均への縮約』だけで決まるので、
    履歴全体の累積表を作らずに、キーごとの合計を 1 回引くだけで同じ値になる（学習側と一致することは verify_serving_v4 が確認）。
    履歴の行（date < day）の戻り値は使われないので欠損にする。"""
    day = pd.Timestamp(day)

    def fast(df: pd.DataFrame, group: str, value: str, k: float = 20.0, name: str | None = None) -> pd.DataFrame:
        name = name or f"{group}_{value}"
        d = pd.to_datetime(df["date"]).to_numpy()
        pre = d < np.datetime64(day)
        v = pd.to_numeric(df[value], errors="coerce").to_numpy(dtype=float)
        key = df[group].to_numpy()
        g = pd.DataFrame({"g": key[pre], "v": v[pre]}).groupby("g")["v"].agg(S="sum", N="count")
        ok = ~np.isnan(v[pre])
        GS, GN = float(v[pre][ok].sum()), int(ok.sum())
        gm = GS / GN if GN else np.nan
        S = g["S"].reindex(key).fillna(0.0).to_numpy()
        N = g["N"].reindex(key).fillna(0).to_numpy(dtype=float)
        mean = np.where(pre, np.nan, (S + k * gm) / (N + k))
        n = np.where(pre, np.nan, N)
        return pd.DataFrame({f"{name}_mean": mean.astype("float32"), f"{name}_n": n.astype("float32")}, index=df.index)

    return fast


class fast_asof:
    """with fast_asof(day): の間だけ、学習側の as-of 平均（asof_group_mean）を高速版に差し替える。"""

    def __init__(self, day):
        self.fn = make_fast_asof(day)

    def __enter__(self):
        from src.pipeline.models import stage_models as sm

        self._orig = (sm.asof_group_mean, ff.asof_group_mean)
        sm.asof_group_mean = self.fn
        ff.asof_group_mean = self.fn
        return self

    def __exit__(self, *exc):
        from src.pipeline.models import stage_models as sm

        sm.asof_group_mean, ff.asof_group_mean = self._orig


class V4History:
    """履歴の読み込み（起動時に 1 回）と、状態表（プロフィール・a_style / f3_r）。"""

    def __init__(self, tables_dir: str | Path | None = None, train_table: str | Path | None = None, years: list[int] | None = None):
        self.tables = Path(tables_dir) if tables_dir else TABLES
        self.years = years or sorted(int(p.name) for p in self.tables.iterdir() if p.is_dir() and p.name.isdigit())
        self.train_path = Path(train_table) if train_table else None
        self.conn_raw: pd.DataFrame | None = None
        self.self_runs: pd.DataFrame | None = None
        self.field_hist: pd.DataFrame | None = None
        self.attrs: pd.DataFrame | None = None
        self.frozen: dict | None = None
        self.pace: pd.DataFrame | None = None
        self.ped_A: pd.DataFrame | None = None
        self.ped_H: pd.DataFrame | None = None
        self.graded: dict[str, str] = {}
        self.profile: pd.DataFrame | None = None            # 馬ごとの owner / breeder / birthplace（静的な表 + 日次で足す分）

    # ------------------------------------------------------------------ 読み込み
    def warm(self) -> dict:
        t0 = time.perf_counter()
        self.profile = pd.read_parquet(fc.PROFILE) if fc.PROFILE.exists() else pd.DataFrame(columns=["horse_id", "owner", "breeder", "birthplace"])
        self.conn_raw = fc.load_history(self.tables, self.years, profile=self.profile if len(self.profile) else None)
        self.conn_raw["date"] = pd.to_datetime(self.conn_raw["date"])
        self.self_runs = fs.load_runs(self.tables, self.years)
        at_path = FAC / "self/horse_attrs.parquet"
        self.attrs = pd.read_parquet(at_path).set_index("horse_id") if at_path.exists() else pd.DataFrame(columns=["birthplace", "birthday"])
        self.frozen = self._frozen_constants()
        self.field_hist = self._field_history()
        self.pace = ff.load_pace_table(self.tables, self.years)
        from src.pipeline.features.horse_history.inference import graded_race_names

        self.graded = graded_race_names(self.tables, self.years)
        A, self.ped_H = fp.build_ped_tables([])
        self.ped_A = A[A["gen"] <= 2].reset_index(drop=True)          # 父・母・父父・父母・母父・母母だけ使う（PED_sire_cond）
        info = {"conn_rows": len(self.conn_raw), "self_rows": len(self.self_runs), "field_rows": len(self.field_hist), "last_date": str(self.self_runs["date"].max().date()),
                "warm_sec": round(time.perf_counter() - t0, 1)}
        logger.info("v4 の履歴: %s", info)
        return info

    def _frozen_constants(self) -> dict:
        p = FAC / "self/attrs_frozen.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        r = self.self_runs[self.self_runs["date"].dt.year <= TRAIN_LAST_YEAR]
        c = fs.attrs_constants(r, self.attrs)
        try:
            p.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            logger.warning("attrs_frozen.json を書けなかった（読み取り専用）: %s", p)
        return c

    FIELD_COLS = ["date", "race_id", "horse_id", "horse_number", "venue_code", "surface", "distance", "track_condition", "field_size", "bracket_number", "finish_position",
                  "sex_age", "weight", "weight_change", "race_name", "race_class", "grade", "direction", "weather", "round", "start_time", "jockey_id", "trainer_id", "jockey_name",
                  "a_style", "f3_r"]

    def _field_history(self) -> pd.DataFrame:
        """course_draw / race_type / calendar_env / 種牡馬×条件 の履歴行。学習側は学習表（出馬表の列 + 結果の着順）の行で作っているので、
        2025 年までは学習表そのもの、それ以降は出馬表と結果表を同じ形（出馬表の頭数・馬番・枠番 + 結果の着順）に結合して足す。"""
        import polars as pl

        parts = []
        last = None
        if self.train_path is not None and self.train_path.exists():
            sch = pl.read_parquet_schema(self.train_path)
            t = pl.read_parquet(self.train_path, columns=[c for c in self.FIELD_COLS if c in sch]).to_pandas()
            t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
            t["date"] = pd.to_datetime(t["date"])
            parts.append(t)
            last = t["date"].max()
        for y in self.years:
            sb, rr = self.tables / str(y) / "race_shutuba_flat.parquet", self.tables / str(y) / "race_result_flat.parquet"
            if not (sb.exists() and rr.exists()) or (last is not None and y < last.year):
                continue
            s_sch = pl.read_parquet_schema(sb)
            a = pl.read_parquet(sb, columns=[c for c in self.FIELD_COLS if c in s_sch and c not in ("finish_position", "a_style", "f3_r")]).to_pandas()
            b = pl.read_parquet(rr, columns=["race_id", "horse_number", "finish_position"]).to_pandas()
            a["race_id"], b["race_id"] = a["race_id"].astype(str), b["race_id"].astype(str)
            m = a.merge(b, on=["race_id", "horse_number"], how="inner")
            m = m[(pd.to_numeric(m["finish_position"], errors="coerce") > 0) & pd.to_numeric(m["venue_code"], errors="coerce").between(1, 10)]
            m["date"] = pd.to_datetime(m["date"])
            if last is not None:
                m = m[m["date"] > last]
            m["horse_id"] = m["horse_id"].astype(str)
            parts.append(m)
        h = pd.concat(parts, ignore_index=True)
        st = self._style_table()
        h = h.drop(columns=[c for c in ("a_style", "f3_r") if c in h.columns and h[c].isna().all()]).merge(st, on=KEY, how="left", suffixes=("", "__s"))
        for c in ("a_style", "f3_r"):
            if c + "__s" in h.columns:
                h[c] = h[c].where(h[c].notna(), h[c + "__s"]) if c in h.columns else h[c + "__s"]
                h = h.drop(columns=[c + "__s"])
        return h

    def _style_table(self) -> pd.DataFrame:
        """2026 年以降の a_style / f3_r は日次蓄積（feature-accumulate の features_<年>.parquet）。"""
        import polars as pl

        from src.pipeline.features.horse_history.accumulate import store_dir

        parts = []
        for f in sorted(store_dir().glob("features_*.parquet")):
            parts.append(pl.read_parquet(f, columns=["race_id", "horse_id", "a_style", "f3_r"]).to_pandas())
        st = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEY + ["a_style", "f3_r"])
        st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
        return st.drop_duplicates(KEY, keep="last")


# ---------------------------------------------------------------------- 当日の行
def _card_rows(card: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    t = card.copy()
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    for c in cols:
        if c not in t.columns:
            t[c] = np.nan
    t["finish_position"] = np.nan
    return t[list(dict.fromkeys(cols + ["finish_position"]))]


def _pick(F: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    m = keys[KEY].merge(F.drop_duplicates(KEY), on=KEY, how="left")
    return m


class V4Groups:
    def __init__(self, hist: V4History):
        self.h = hist

    def normalize(self, card: pd.DataFrame) -> pd.DataFrame:
        """出馬表と結果表の表記の違い（馬場の略称、格が空欄・OP のまま）を結果表（学習時）に合わせる。"""
        import polars as pl

        from src.pipeline.features.horse_history.inference import normalize_card

        c = card.copy()
        for col in ("grade", "race_class", "race_name", "track_condition"):
            if col in c.columns:
                c[col] = c[col].astype("string").astype(object).where(c[col].notna(), None)
        out = normalize_card(pl.from_pandas(c), self.h.graded).to_pandas()
        return out

    @staticmethod
    def as_result_surface(card: pd.DataFrame) -> pd.DataFrame:
        """学習側の履歴（race_result_flat）は障害レースの面を「芝」で持っているので、結果表から作る群（馬主・馬自身）は条件のキーをそろえる。"""
        c = card.copy()
        if "surface" in c.columns:
            c["surface"] = c["surface"].replace({"障": "芝"})
        return c

    # ------------------------------------------------------------------ 馬主系
    def owner(self, card: pd.DataFrame, profile: pd.DataFrame | None = None) -> pd.DataFrame:
        card = self.as_result_surface(self.normalize(card))
        D = _to_dt(card["date"]).min()
        raw = self.h.conn_raw
        hist = raw[raw["date"] < D]
        T = _card_rows(card, fc.HIST_COLS)
        profile = profile if profile is not None else self.h.profile
        if profile is not None and len(profile):
            p = profile.drop_duplicates("horse_id").assign(horse_id=lambda d: d["horse_id"].astype(str)).set_index("horse_id")
            for c in ("owner", "breeder", "birthplace"):
                if c in p.columns:
                    T[c] = fc._norm_name(p[c].reindex(T["horse_id"]).reset_index(drop=True)).to_numpy()
        h = pd.concat([hist, T], ignore_index=True)
        hp = fc.prepare(h)
        eng = fc.Engine(hp)
        out: dict[str, np.ndarray] = {}
        for ent in ("o", "ob"):
            out.update(fc.entity_features(eng, ent, fc.CONDS))
        out.update(fc.window_features(eng, "o"))
        out.update(fc.streak_features(eng, "o"))
        mask = (hp["date"] >= D).to_numpy()
        F = pd.DataFrame({c: np.asarray(a, dtype="float32")[mask] for c, a in out.items()})
        F[KEY[0]], F[KEY[1]] = hp.loc[mask, "race_id"].to_numpy(), hp.loc[mask, "horse_id"].to_numpy()
        return F

    # ------------------------------------------------------------------ 馬自身 8 群
    def self8(self, card: pd.DataFrame, attrs_extra: pd.DataFrame | None = None) -> pd.DataFrame:
        card = self.as_result_surface(self.normalize(card))
        D = _to_dt(card["date"]).min()
        runs = self.h.self_runs
        hist = runs[runs["date"] < D]
        T = _card_rows(card, fs.RUN_COLS)
        T["distance"] = pd.to_numeric(T["distance"], errors="coerce").astype(float)
        allr = pd.concat([hist, T], ignore_index=True).sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
        at = self.h.attrs
        if attrs_extra is not None and len(attrs_extra):
            ex = attrs_extra.drop_duplicates("horse_id").set_index("horse_id")
            at = pd.concat([at, ex[[c for c in at.columns if c in ex.columns]].loc[lambda d: ~d.index.isin(at.index)]])
        groups = fs.build_groups(allr, birthday=None, attrs=at, only=SELF8, frozen=self.h.frozen)
        tk = pd.MultiIndex.from_frame(T[KEY])
        parts = []
        for name, g in groups.items():
            g = g.set_index(KEY)
            parts.append(g.loc[g.index.isin(tk)].drop(columns=[c for c in g.columns if c in KEY], errors="ignore"))
        return pd.concat(parts, axis=1).reset_index() if parts else pd.DataFrame(columns=KEY)

    # ------------------------------------------------------------------ 出走メンバー・コース環境 7 群
    def field7(self, target: pd.DataFrame) -> pd.DataFrame:
        """target: 当日の行（card の列 + V0 の列 + race_name など）。mem_* は当日の行だけ、course_draw / race_type / calendar_env は履歴も使う。"""
        D = _to_dt(target["date"]).min()
        hist = self.h.field_hist
        hist = hist[hist["date"] < D]
        T = target.copy()
        T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
        T["date"] = pd.to_datetime(T["date"])
        T["finish_position"] = np.nan
        out = []
        tdf = T.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
        for g in ("mem_strength", "mem_demo", "mem_conn"):
            f = ff.build_group(g, tdf)
            out.append(pd.concat([tdf[KEY], f], axis=1))
        # course_draw / race_type は asof_group_mean（縮約先が「その日より前の全行の平均」）を使うので、履歴は絞らない
        out.append(self._with_history("course_draw", hist, tdf, cols_re=r"^fdr_(course|br|hn|third|cond_third|fs_third)_"))
        out.append(self._with_history("race_type", hist, tdf, pace_table=self.h.pace))
        # 前日の馬場: 同じ場・面の過去の開催日（絞っても結果は同じ）
        vs = set(zip(tdf["venue_code"].astype(str), tdf["surface"].astype(str)))
        hv = pd.Series(list(zip(hist["venue_code"].astype(str), hist["surface"].astype(str))), index=hist.index)
        out.append(self._with_history("calendar_env", hist[hv.isin(vs)], tdf))
        F = out[0]
        for o in out[1:]:
            F = F.merge(o, on=KEY, how="left")
        return F

    @staticmethod
    def _with_history(name: str, hist: pd.DataFrame, tdf: pd.DataFrame, cols_re: str | None = None, **kw) -> pd.DataFrame:
        import re

        cols = list(dict.fromkeys(list(hist.columns) + list(tdf.columns)))
        B = pd.concat([hist.reindex(columns=cols), tdf.reindex(columns=cols)], ignore_index=True)
        B = B.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
        with fast_asof(tdf["date"].min()):
            f = ff.build_group(name, B, **kw)
        if cols_re:
            f = f[[c for c in f.columns if re.search(cols_re, c)]]
        tk = pd.MultiIndex.from_frame(tdf[KEY])
        sel = pd.MultiIndex.from_frame(B[KEY]).isin(tk) & (B["date"] >= tdf["date"].min()).to_numpy()
        return pd.concat([B.loc[sel, KEY].reset_index(drop=True), f.loc[sel].reset_index(drop=True)], axis=1)

    # ------------------------------------------------------------------ 種牡馬 ID × 条件
    def ped_cond(self, card: pd.DataFrame) -> pd.DataFrame:
        card = self.normalize(card)
        D = _to_dt(card["date"]).min()
        hist = self.h.field_hist
        hist = hist[hist["date"] < D][["race_id", "horse_id", "date", "venue_code", "surface", "distance", "track_condition", "finish_position", "field_size"]]
        T = _card_rows(card, ["race_id", "horse_id", "date", "venue_code", "surface", "distance", "track_condition", "field_size"])
        A, H = self.h.ped_A, self.h.ped_H
        miss = [x for x in T["horse_id"].unique() if x not in set(A["horse_id"])]
        if miss:                                                    # 新馬など（血統 JSON が保存済みなら足す）
            A = pd.concat([A, self._ped_extra(miss)], ignore_index=True)
        B = pd.concat([hist, T], ignore_index=True).sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
        with fast_asof(D):
            f = fp.ped_sire_cond(B, A, H)      # 縮約先が全行の平均（asof_rate）なので、履歴は絞らず、当日の行は合計を引くだけの高速版で
        sel = (B["date"] >= D).to_numpy()
        return pd.concat([B.loc[sel, KEY].reset_index(drop=True), f.loc[sel].reset_index(drop=True)], axis=1)

    @staticmethod
    def _ped_extra(horse_ids: list[str]) -> pd.DataFrame:
        import json as _json

        anc = []
        for h in horse_ids:
            for d in fp.PED_DIRS:
                f = d / h[:4] / f"{h}.json"
                if f.exists():
                    try:
                        data = _json.loads(f.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    for a in data.get("ancestors") or []:
                        g, p, i = a.get("generation"), a.get("position"), str(a.get("horse_id") or "")
                        if g and g <= fp.MAX_GEN and i:
                            anc.append((h, int(g), int(p), i))
                    break
        return pd.DataFrame(anc, columns=["horse_id", "gen", "pos", "anc_id"])


# ====================================================================== 組み立て（V3Serving から呼ぶ）
GROUP_COLUMNS_JSON = Path(__file__).with_name("serving_v4_columns.json")
EXTRA_CARD_COLS = ["race_name", "jockey_name", "sex_age", "jockey_id", "trainer_id", "direction", "start_time", "weight_rule", "grade", "race_class", "round", "weather"]


def group_columns() -> dict[str, list[str]]:
    """群 → 列名（学習時の保存物から作った一覧。``update_serving_state columns`` で作り直す）。"""
    return json.loads(GROUP_COLUMNS_JSON.read_text(encoding="utf-8")) if GROUP_COLUMNS_JSON.exists() else {}


def needed_groups(feature_cols: list[str]) -> set[str]:
    cols = set(feature_cols)
    return {g for g, cs in group_columns().items() if cols & set(cs)}


class V4Features:
    """起動時に履歴を読み、当日の出走馬に V4 の群の列を付ける。馬主・馬自身の群は『その日の全レース』を 1 回作って日単位でキャッシュする
    （条件・体重に依存しないので、馬体重の発表ごとに作り直さない）。メンバー構成・コース環境・血統の群は発表のたびに作る（数秒）。"""

    def __init__(self, tables_dir: str | Path | None = None, train_table: str | Path | None = None, state_dir: str | Path | None = None):
        self.hist = V4History(tables_dir, train_table)
        self.groups = V4Groups(self.hist)
        self.state_dir = Path(state_dir) if state_dir else STATE_DIR
        self._day: dict[str, pd.DataFrame] = {}
        self.warmed = False

    def warm(self) -> dict:
        info = self.hist.warm()
        dyn = self.state_dir / "profile_dyn.parquet"
        if dyn.exists():
            d = pd.read_parquet(dyn)
            self.hist.profile = pd.concat([self.hist.profile, d[~d["horse_id"].isin(self.hist.profile["horse_id"])]], ignore_index=True)
            at = d.dropna(subset=["birthplace"]).drop_duplicates("horse_id").set_index("horse_id")[["birthplace"]]
            if "birthday" in d.columns:
                at["birthday"] = d.drop_duplicates("horse_id").set_index("horse_id")["birthday"]
            self.hist.attrs = pd.concat([self.hist.attrs, at.loc[~at.index.isin(self.hist.attrs.index)]])
            info["profile_dyn"] = len(d)
        self.warmed = True
        return info

    # ------------------------------------------------------------------ プロフィール（馬主・生産者・産地）
    def profile_for(self, horse_ids: list[str], storage: Any | None) -> pd.DataFrame:
        """未登録の馬（新馬など）を horse_result から補い、状態にも追加する。"""
        known = set(self.hist.profile["horse_id"].astype(str))
        miss = [h for h in dict.fromkeys(map(str, horse_ids)) if h not in known]
        if miss and storage is not None:
            rows = []
            for h in miss:
                try:
                    rec = storage.load("horse_result", h) or storage.load("horse_profile", h) or {}
                except Exception:
                    rec = {}
                rows.append({"horse_id": h, "owner": rec.get("owner"), "breeder": rec.get("breeder"), "birthplace": rec.get("birthplace"), "birthday": rec.get("birthday")})
            add = pd.DataFrame(rows)
            self.hist.profile = pd.concat([self.hist.profile, add[["horse_id", "owner", "breeder", "birthplace"]]], ignore_index=True)
            at = add.dropna(subset=["birthplace"]).set_index("horse_id")[["birthplace", "birthday"]]
            self.hist.attrs = pd.concat([self.hist.attrs, at.loc[~at.index.isin(self.hist.attrs.index)]])
        return self.hist.profile

    # ------------------------------------------------------------------ 日単位（馬主・馬自身）
    def day_groups(self, card_all: pd.DataFrame, storage: Any | None = None) -> pd.DataFrame:
        """card_all: その日の全レースの出馬表。結果は日単位でキャッシュ（メモリ + state/<日>/v4_day.parquet）。"""
        D = str(pd.to_datetime(card_all["date"]).min().date())
        path = self.state_dir / D / "v4_day.parquet"
        have = self._day.get(D)
        if have is None and path.exists():
            have = pd.read_parquet(path)
        want = card_all[KEY].astype(str)
        if have is not None and len(want.merge(have[KEY], on=KEY, how="left", indicator=True).query("_merge == 'left_only'")) == 0:
            self._day[D] = have
            return have
        prof = self.profile_for(card_all["horse_id"].astype(str).tolist(), storage)
        t0 = time.perf_counter()
        own = self.groups.owner(card_all, prof)
        slf = self.groups.self8(card_all)
        F = own.merge(slf, on=KEY, how="outer")
        logger.info("v4 日単位の群: %d 行 %.1fs", len(F), time.perf_counter() - t0)
        self._day[D] = F
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            F.to_parquet(path, index=False)
        except OSError:
            logger.warning("v4 日単位の群を保存できなかった: %s", path)
        return F

    # ------------------------------------------------------------------ 本体
    def features(self, card: pd.DataFrame, base: pd.DataFrame, feature_cols: list[str], storage: Any | None = None, card_all: pd.DataFrame | None = None) -> pd.DataFrame:
        """card: 出馬表（card_frame）、base: V3 部分の特徴量（race_id・horse_id + V0 の列）。戻り値: race_id・horse_id + V4 の群の列（バンドルが要る分）。"""
        need = needed_groups(feature_cols)
        out = base[KEY].astype(str).copy()
        if not need:
            return out
        t0 = time.perf_counter()
        card = card.copy()
        card["race_id"], card["horse_id"] = card["race_id"].astype(str), card["horse_id"].astype(str)
        if need & {"owner", "self8"}:
            day = self.day_groups(card_all if card_all is not None else card, storage)
            out = out.merge(day, on=KEY, how="left")
        if need & {"field7"}:
            extra = [c for c in EXTRA_CARD_COLS if c in card.columns and c not in base.columns]
            T = base.merge(card[KEY + ([] if "date" in base.columns else ["date"]) + extra].drop_duplicates(KEY), on=KEY, how="left")
            out = out.merge(self.groups.field7(T), on=KEY, how="left")
        if need & {"ped"}:
            out = out.merge(self.groups.ped_cond(card), on=KEY, how="left")
        logger.info("v4 の群: 要る群 %s %.1fs", sorted(need), time.perf_counter() - t0)
        return out
