"""v4 の推論用の状態を更新する（夜間の結果確定後・開催日の朝に実行）。

python -m src.scripts.data.update_serving_state columns                 # 群 → 列名の一覧（serving_v4_columns.json）を学習時の保存物から作る
python -m src.scripts.data.update_serving_state profile [--limit N]     # 履歴にいて馬主などが未登録の馬を horse_result から補う（state/profile_dyn.parquet）
python -m src.scripts.data.update_serving_state precompute --date YYYYMMDD   # その日の全レースの馬主・馬自身の群を作ってキャッシュ（state/<日>/v4_day.parquet）

履歴（race_result_flat、features_<年>.parquet）は夜間のエクスポートと feature-accumulate が更新する。ここで更新する状態は、プロフィールと日単位のキャッシュだけ。
precompute は、その日に GCS から補った新しい馬（新馬など。profile の対象は走った後の馬だけ）を profile_dyn に書き戻す。後から起動する
weight-runner・影のランナー（本体の profile_dyn を読む。serving_v4.profile_local_dir）は、その馬を GCS から読み直さない。
止めるとき: KEIBA_PROFILE_LOCAL=0 か data/local/meta/profile_local.off（書き戻しと、手元を先に読む段の両方が止まる）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import pandas as pd

from src.pipeline.features import serving_v4 as sv

KEY = ["race_id", "horse_id"]


def cmd_columns() -> None:
    cols: dict[str, list[str]] = {}

    def names(path: Path, pat: str | None = None) -> list[str]:
        import re

        import polars as pl

        c = [x for x in pl.read_parquet_schema(path) if x not in KEY]
        return [x for x in c if re.search(pat, x)] if pat else c

    cols["owner"] = [c for n in ("ent_o", "ent_ob", "win_o", "streak_o") for c in names(sv.FAC / "conn/groups" / f"{n}.parquet")]
    cols["self8"] = [c for n in sv.SELF8 for c in names(sv.FAC / "self" / f"{n}.parquet")]
    cols["field7"] = (names(sv.FAC / "field/mem_strength.parquet") + names(sv.FAC / "field/mem_demo.parquet") + names(sv.FAC / "field/mem_conn.parquet")
                      + names(sv.FAC / "field/course_draw.parquet", r"^fdr_(course|br|hn|third|cond_third|fs_third)_") + names(sv.FAC / "field/calendar_env.parquet")
                      + names(sv.FAC / "field/race_type.parquet"))
    cols["ped"] = names(sv.FAC / "src/cand_PED_sire_cond.parquet")
    cols = {g: list(dict.fromkeys(c)) for g, c in cols.items()}
    sv.GROUP_COLUMNS_JSON.write_text(json.dumps(cols, ensure_ascii=False, indent=1), encoding="utf-8")
    print({g: len(c) for g, c in cols.items()})


def _storage():
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.scraper.storage import HybridStorage

    return HybridStorage()


def attach_birthday(new: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """profile の行に誕生日を補う。すでに birthday 列があるときは、欠けている行だけ attrs の値で埋める（重複列にしない）。"""
    if "birthday" not in attrs.columns:
        return new
    b = attrs.rename_axis("horse_id").reset_index()[["horse_id", "birthday"]].rename(columns={"birthday": "_birthday_attr"})
    out = new.merge(b, on="horse_id", how="left")
    out["birthday"] = out["birthday"].where(out["birthday"].notna(), out["_birthday_attr"]) if "birthday" in out.columns else out["_birthday_attr"]
    return out.drop(columns="_birthday_attr")


def cmd_profile(limit: int) -> None:
    f = sv.V4Features()
    f.skip_absent = False                    # 走った後の馬を補うので、朝の『文書が無い』の記録では飛ばさない（None の行も profile_dyn に残し、bp_retry の対象にする）
    f.warm()
    hist_horses = set(f.hist.self_runs.loc[f.hist.self_runs["date"] >= "2026-01-01", "horse_id"].astype(str))
    miss = sorted(hist_horses - set(f.hist.profile["horse_id"].astype(str)))
    if limit:
        miss = miss[:limit]
    print(f"未登録 {len(miss)} 頭", flush=True)
    st = _storage()
    t0 = time.time()
    f.profile_for(miss, st)
    new = f.hist.profile[f.hist.profile["horse_id"].isin(miss)]
    p = sv.STATE_DIR / "profile_dyn.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    new = attach_birthday(new, f.hist.attrs.reindex(miss))
    dyn = pd.concat([old, new], ignore_index=True).drop_duplicates("horse_id", keep="last")
    dyn = retry_birthplace(dyn, hist_horses, st, limit)
    save_profile_dyn(p, dyn, prefer_new=True)
    print(f"保存 {p}（{len(new)} 頭、{time.time() - t0:.0f}s）")


DYN_COLS = ["horse_id", "owner", "breeder", "birthplace", "birthday", "bp_retry"]


def _norm_dyn(d: pd.DataFrame) -> pd.DataFrame:
    """profile_dyn の形にそろえる（列は DYN_COLS に固定、horse_id は文字列、誕生日は『2024年2月2日』の文字列、bp_retry は bool）。
    昔の結合で混ざった birthday_x・birthday_y や、全部空の color・sex は落とす（読む側は horse_id・馬主・生産者・産地・誕生日だけを使う）。"""
    out = d.copy()
    for c in DYN_COLS:
        if c not in out.columns:
            out[c] = False if c == "bp_retry" else None
    out = out[DYN_COLS]
    out["horse_id"] = out["horse_id"].astype(str)
    out["birthday"] = birthday_as_text(out["birthday"])
    for c in ("owner", "breeder", "birthplace"):
        out[c] = out[c].astype(object).where(out[c].notna(), None)
    out["bp_retry"] = out["bp_retry"].astype("boolean").fillna(False).astype(bool)
    return out


def save_profile_dyn(path: Path, rows: pd.DataFrame, *, prefer_new: bool) -> int:
    """profile_dyn.parquet に行を足す（ロック → 読み直して結合 → 一時ファイル → os.replace）。書くのは root で動く profile・precompute だけ。
    prefer_new=True（profile）: 渡した行で同じ馬の行を置き換える。False（precompute）: 既にある馬は残し、無い馬（と、値がすべて空の行）だけ足す。
    戻り値: 足した（置き換えた）行数。"""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = _norm_dyn(rows).drop_duplicates("horse_id", keep="last")
    with open(path.parent / ".profile_dyn.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = _norm_dyn(pd.read_parquet(path)) if path.exists() else _norm_dyn(pd.DataFrame(columns=DYN_COLS))
        if not prefer_new:                   # 既にある馬は残す。ただし馬主・生産者・産地がすべて空の行は、取れた値で置き換える
            have = cur.loc[cur[["owner", "breeder", "birthplace"]].notna().any(axis=1), "horse_id"]
            new = new[~new["horse_id"].isin(have)]
        if not len(new) and path.exists():
            return 0
        out = pd.concat([cur[~cur["horse_id"].isin(new["horse_id"])], new], ignore_index=True)
        _replace_parquet(out, path)
    return len(new)


def _replace_parquet(df: pd.DataFrame, path: Path) -> None:
    """一時ファイルに書いてから os.replace で置き換える（書きかけを読ませない）。失敗したら一時ファイルを消す。"""
    import os

    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


ABSENT_KEEP_SEC = 7 * 86400                   # profile_absent の行を残す長さ（読むときの期限は serving_v4.profile_absent_ttl_sec）


def save_profile_absent(path: Path, absent: dict[str, float]) -> int:
    """『GCS に文書が無かった馬』の記録（horse_id・checked_at〔UNIX 秒〕）を足す。同じ馬は新しい時刻にする。7 日より古い行は落とす。"""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame({"horse_id": [str(h) for h in absent], "checked_at": [float(t) for t in absent.values()]})
    with open(path.parent / ".profile_absent.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["horse_id", "checked_at"])
        cur = cur.astype({"horse_id": str, "checked_at": float})
        out = pd.concat([cur, new], ignore_index=True).sort_values("checked_at").drop_duplicates("horse_id", keep="last")
        out = out[out["checked_at"] >= time.time() - ABSENT_KEEP_SEC].reset_index(drop=True)
        _replace_parquet(out, path)
    return len(new)


def persist_fetched(f, path: Path | None = None) -> tuple[int, int]:
    """precompute が GCS から補った馬（V4Features.fetched）のうち、馬主・生産者・産地のどれかが取れた馬を profile_dyn に足す。
    文書が無かった馬（404 だけ。V4Features.absent）は、同じ置き場（本体の state）の profile_absent.parquet に時刻つきで記録する。後から起動する
    weight-runner・影のランナーは、期限（既定 3 時間。KEIBA_PROFILE_ABSENT_TTL_SEC）の間は読み直さない。GCS の不調で読めなかった馬はどちらにも書かない
    （不調の None を固定しない。走った後の profile が記録する）。
    手元を読まない設定（KEIBA_PROFILE_LOCAL=0・目印のファイル）では書かない。戻り値: (profile_dyn に足した頭数, 文書なしと記録した頭数)。"""
    if not sv.profile_local_enabled():
        return 0, 0
    n_dyn = n_abs = 0
    if f.fetched:
        d = pd.DataFrame(f.fetched)
        d = d[d[["owner", "breeder", "birthplace"]].notna().any(axis=1)]
        if len(d):
            try:
                n_dyn = save_profile_dyn(path or (f.state_dir / "profile_dyn.parquet"), d, prefer_new=False)
            except Exception as exc:  # noqa: BLE001  （権限・壊れたファイルなど。v4_day は保存済みなので precompute は落とさない）
                print(f"profile_dyn に書き戻せなかった（推論は続く）: {exc!r}")
    if f.absent:
        try:
            n_abs = save_profile_absent((path.parent if path else f.state_dir) / "profile_absent.parquet", f.absent)
        except Exception as exc:  # noqa: BLE001
            print(f"profile_absent に書けなかった（推論は続く）: {exc!r}")
    return n_dyn, n_abs


def birthday_as_text(v: pd.Series) -> pd.Series:
    """profile_dyn の誕生日を保存の形（『2024年2月2日』の文字列）にそろえる。推論の表で日付に変えた値が混ざると、parquet に書けない。"""
    def one(x):
        if x is None or (isinstance(x, float) and pd.isna(x)) or x is pd.NaT:
            return None
        if isinstance(x, (pd.Timestamp, datetime.date)):
            t = pd.Timestamp(x)
            return f"{t.year}年{t.month}月{t.day}日"
        return str(x)
    return v.map(one).astype(object)


def retry_birthplace(dyn: pd.DataFrame, ran: set[str], storage, limit: int = 0) -> pd.DataFrame:
    """成績の文書（horse_result）に産地が無かった馬だけ、プロフィールの文書（horse_profile）を 1 頭 1 回だけ読み直して産地・誕生日を足す。
    読み直した馬には bp_retry=True を付け、二度と読まない（ホストから GCS を繰り返し読まない。docs/operations/horse_attrs_coverage.md）。
    対象は 2026 年に走った馬だけ。読み出しは 1 頭 1 回で、新しい馬のうち成績の文書に産地が無い馬の数だけ（週 数十回程度）。"""
    d = dyn.copy()
    if "bp_retry" not in d.columns:
        d["bp_retry"] = False
    d["bp_retry"] = d["bp_retry"].astype("boolean").fillna(False).astype(bool)
    bp = d["birthplace"] if "birthplace" in d.columns else pd.Series(index=d.index, dtype=object)
    lack = bp.isna() | (bp.astype(str).str.strip() == "")
    idx = d.index[lack & ~d["bp_retry"] & d["horse_id"].astype(str).isin(ran)]
    if limit:
        idx = idx[:limit]
    for i in idx:
        try:
            rec = storage.load("horse_profile", str(d.at[i, "horse_id"])) or {}
        except Exception:
            rec = {}
        if str(rec.get("birthplace") or "").strip():
            d.at[i, "birthplace"] = rec.get("birthplace")
            if "birthday" in d.columns and (pd.isna(d.at[i, "birthday"]) or not str(d.at[i, "birthday"]).strip()):
                d.at[i, "birthday"] = rec.get("birthday")
        d.at[i, "bp_retry"] = True
    if len(idx):
        print(f"産地の読み直し（horse_profile）: {len(idx)} 頭、埋まった {int((d.loc[idx, 'birthplace'].astype(str).str.strip().replace('nan', '') != '').sum())} 頭")
    return d


def cmd_precompute(date: str) -> None:
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference.race_day_workflow import load_day_schedule

    st = _storage()
    sched = load_day_schedule(st, date)
    ids = [r["race_id"] for r in sched]
    if not ids:
        print(f"{date}: 開催なし（日程が空）。何もしない")
        return
    card = wtp.card_frame(st, ids).to_pandas()
    if card.empty:
        print(f"{date}: 出馬表がまだ保存されていない。何もしない")
        return
    print(f"{date}: {len(ids)} レース / {len(card)} 頭", flush=True)
    f = sv.V4Features()
    f.warm()
    F = f.day_groups(card, st)
    print(f"day cache: {len(F)} 行 × {F.shape[1]} 列")
    n_dyn, n_abs = persist_fetched(f)
    print(f"新しい馬の補い: {f.profile_stats}、profile_dyn に足した {n_dyn} 頭・文書なしと記録した {n_abs} 頭")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("columns")
    p = sp.add_parser("profile")
    p.add_argument("--limit", type=int, default=0)
    q = sp.add_parser("precompute")
    q.add_argument("--date", required=True)
    a = ap.parse_args()
    if a.cmd == "columns":
        cmd_columns()
    elif a.cmd == "profile":
        cmd_profile(a.limit)
    else:
        cmd_precompute(a.date)


if __name__ == "__main__":
    main()
