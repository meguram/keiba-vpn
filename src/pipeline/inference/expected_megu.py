"""出走馬ごとの『想定めぐ指数』（過去のめぐ指数からの想定。AI の予測ではない）。

前日に確定している情報だけで作る: その馬の過去走の絶対指数（v3a の ``megu_index_v3a_abs_robust``、斤量 55kg 換算、1 点 = 0.1 秒）を、
直近ほど重い指数加重平均（直近 5 走）にまとめる。対象レース日より前の走だけを使う。

今回の条件が過去走と大きく違うとき（芝↔ダート、距離、クラスの昇降級、休み明け、競馬場、馬齢）は、
(1) 同じ面・近い距離の走を重くし、(2) 学習した補正量（``config/expected_megu/model.json``。2020〜2024 年で学習、2025 年で検証）を足し、
(3) 条件変化が大きいほどレンジ（±）を広げて理由を付ける。補正係数が無いときは単純な加重平均だけを返す。

データは ``data/page_reference/tables/<年>/megu_index_flat.parquet``（馬×レース）と ``race_result_flat.parquet``（レースの条件）。
年ごとに 1 回だけ読み、ファイルの更新時刻でキャッシュする（1 レース 16 頭 × 週 36 レースでも読み直さない）。
"""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.pipeline.inference import horse_growth as hg

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "config" / "expected_megu" / "model.json"

N_RUNS = 5                  # 直近何走を使うか
DECAY = 0.65                # 1 走古くなるごとの重み（最新 = 1）
LOW_CONF_WEIGHT = 0.5       # 信頼度が低い走の重み
REFERENCE_MAX_RUNS = 2      # この走数以下は『参考』
STALE_DAYS = 240            # 直近の反映走がこれより古いと『間隔が空いている』
SEC_PER_POINT = 0.1         # 指数 1 点 = 0.1 秒
MIN_RANGE = 1.5             # 表示するレンジ（±）の下限
GRADE_LEVEL = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "OP": 5, "L": 6, "G3": 7, "G2": 8, "G1": 9}
LEVEL_NAME = {0: "新馬", 1: "未勝利", 2: "1勝クラス", 3: "2勝クラス", 4: "3勝クラス", 5: "オープン", 6: "リステッド", 7: "G3", 8: "G2", 9: "G1"}

_lock = threading.Lock()
_cache: dict[str, tuple[tuple, dict[str, list[tuple]], str]] = {}
_model_cache: dict[str, Any] = {}

# 走の記録: (YYYYMMDD, 指数, 低信頼, 面, 距離, 競馬場, クラス水準, 斤量, 馬齢)。面〜馬齢は不明なら None
Run = tuple


def _digits(s: Any) -> str:
    return re.sub(r"\D", "", str(s or ""))[:8]


def _fmt(ymd: str) -> str:
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}" if len(ymd) == 8 else ymd


def _days(a: str, b: str) -> int | None:
    from datetime import date

    try:
        return (date(int(a[:4]), int(a[4:6]), int(a[6:8])) - date(int(b[:4]), int(b[4:6]), int(b[6:8]))).days
    except (ValueError, IndexError):
        return None


def class_level(grade: Any) -> int | None:
    return GRADE_LEVEL.get(str(grade or "").strip())


def surface_of(s: Any) -> str | None:
    t = str(s or "")
    return "芝" if t.startswith("芝") else "ダ" if t.startswith("ダ") else None


def age_of(sex_age: Any) -> int | None:
    m = re.search(r"(\d+)", str(sex_age or ""))
    return int(m.group(1)) if m else None


def load_model(path: Path | None = None) -> dict | None:
    """学習した補正量（小さな JSON）。無ければ None（単純な加重平均だけ）。ファイルの更新時刻でキャッシュ。"""
    p = path or MODEL_PATH
    m = hg._mtime(p)
    with _lock:
        hit = _model_cache.get(str(p))
        if hit and hit[0] == m:
            return hit[1]
    try:
        model = json.loads(p.read_text()) if m else None
    except (OSError, ValueError):
        model = None
    with _lock:
        _model_cache[str(p)] = (m, model)
    return model


# ── 純粋関数（テスト対象・学習スクリプトと共有）────────────────────────────

def _pad(r: Run) -> Run:
    return tuple(r) + (None,) * (10 - len(r)) if len(r) < 10 else tuple(r)


def run_weights(runs: list[Run], today: dict | None, wcfg: dict | None) -> list[float]:
    """直近ほど重く、低信頼は半分。今回と面が違う走・距離が遠い走は軽くする（係数は wcfg。無ければ使わない）。"""
    w = []
    for i, r in enumerate(runs):
        wi = DECAY ** i * (LOW_CONF_WEIGHT if r[2] else 1.0)
        if wcfg and today:
            surf, dist = r[3], r[4]
            if today.get("surface") and surf and surf != today["surface"]:
                wi *= float(wcfg.get("surface_mismatch", 1.0))
            if today.get("distance") and dist:
                wi *= math.exp(-abs(float(today["distance"]) - float(dist)) / float(wcfg.get("distance_scale", 1e9)))
        w.append(max(wi, 1e-6))
    return w


def condition_features(runs: list[Run], w: list[float], today: dict | None, race_date: str) -> dict[str, float] | None:
    """今回の条件と、過去走（重み w）の差を数値にする。today に面・距離が無ければ None。リーク無し（今回のレース前に分かるものだけ）。"""
    if not today or not today.get("surface") or not today.get("distance"):
        return None
    tw = sum(w)

    def wmean(vals: list[float | None]) -> float | None:
        pairs = [(wi, v) for wi, v in zip(w, vals) if v is not None]
        s = sum(wi for wi, _ in pairs)
        return sum(wi * v for wi, v in pairs) / s if s > 0 else None

    surf_share = {s: sum(wi for wi, r in zip(w, runs) if r[3] == s) / tw for s in ("芝", "ダ")}
    f: dict[str, float] = {
        "to_dirt": (1.0 - surf_share["ダ"]) if today["surface"] == "ダ" else 0.0,
        "to_turf": (1.0 - surf_share["芝"]) if today["surface"] == "芝" else 0.0,
    }
    pd_ = wmean([r[4] for r in runs])
    dd = (float(today["distance"]) - pd_) / 100.0 if pd_ is not None else 0.0
    f["dist_up"], f["dist_down"] = max(dd, 0.0), max(-dd, 0.0)
    pl = wmean([r[6] for r in runs])
    tl = today.get("level")
    cd = (float(tl) - pl) if (tl is not None and pl is not None) else 0.0
    f["class_up"], f["class_down"] = max(cd, 0.0), max(-cd, 0.0)
    rest = _days(race_date, runs[0][0]) if race_date and runs else None
    f["rest_long"] = max(0.0, (rest - 90)) / 30.0 if rest is not None else 0.0
    f["rest_short"] = 1.0 if (rest is not None and rest < 14) else 0.0
    age = today.get("age")
    f["age"] = float(min(max(age, 2), 8)) if age else 4.0
    venues = {r[5] for r in runs if r[5]}
    f["venue_new"] = 1.0 if (today.get("venue") and venues and today["venue"] not in venues) else 0.0
    pw = wmean([r[7] for r in runs])
    tw_ = today.get("weight")
    f["weight_diff"] = (float(tw_) - pw) if (tw_ and pw is not None) else 0.0
    return f


def change_score(f: dict[str, float] | None) -> float:
    """条件変化の大きさ（0 = 同じ条件）。面の切り替え・距離 ±100m・昇降級 1 段などを同じ尺度に寄せた単純な和。"""
    if not f:
        return 0.0
    return (3.0 * (f["to_dirt"] + f["to_turf"]) + 0.5 * (f["dist_up"] + f["dist_down"]) + 1.0 * (f["class_up"] + f["class_down"])
            + 1.0 * f["rest_long"] + 1.0 * f["venue_new"])


def change_group(score: float, model: dict | None) -> str:
    th = (model or {}).get("group_thresholds") or {"some": 1.0, "large": 3.0}
    return "large" if score >= th["large"] else "some" if score >= th["some"] else "none"


CENTERED = {"age", "weight_diff"}
FEATURE_LABELS = {
    "to_dirt": "ダート替わり", "to_turf": "芝替わり", "dist_up": "距離延長", "dist_down": "距離短縮",
    "class_up": "昇級", "class_down": "降級", "rest_long": "休み明け", "rest_short": "間隔が短い",
    "age": "馬齢", "venue_new": "初めての競馬場", "weight_diff": "斤量差",
}


def _reasons(f: dict[str, float], today: dict, runs: list[Run], rest: int | None) -> list[str]:
    out: list[str] = []
    if f["to_dirt"] >= 0.5:
        out.append("ダート初出走" if not any(r[3] == "ダ" for r in runs) else "ダートは久しぶり")
    if f["to_turf"] >= 0.5:
        out.append("芝初出走" if not any(r[3] == "芝" for r in runs) else "芝は久しぶり")
    if f["dist_up"] >= 4 or f["dist_down"] >= 4:
        d = f["dist_up"] if f["dist_up"] >= 4 else -f["dist_down"]
        out.append(f"距離 {int(round(d)) * 100:+d}m")
    if f["class_up"] >= 1.0:
        out.append(f"{int(round(f['class_up']))}クラス昇級")
    if f["class_down"] >= 1.0:
        out.append(f"{int(round(f['class_down']))}クラス降級")
    if rest is not None and rest >= 120:
        out.append(f"休み明け {rest} 日")
    if f["venue_new"] >= 1.0:
        out.append(f"{today.get('venue')}は初めて")
    if abs(f["weight_diff"]) >= 3.0:
        out.append(f"斤量 {f['weight_diff']:+.0f}kg（指数は55kg換算済み）")
    return out


def expected_from_runs(runs: Iterable[Run], race_date: str | None, today: dict | None = None, model: dict | None = None) -> dict | None:
    """runs: 走の記録（先頭 3 要素は (YYYYMMDD, 指数, 低信頼)、以降は面・距離・競馬場・クラス水準・斤量）。
    race_date より前の直近 N_RUNS 走から想定を返す。使える走が 0 なら None。today / model があれば条件変化の補正とレンジの拡大を行う。"""
    rd = _digits(race_date)
    usable = [_pad(r) for r in runs if r[0] and r[1] == r[1] and not (rd and r[0] >= rd)]
    usable.sort(key=lambda x: x[0], reverse=True)
    usable = usable[:N_RUNS]
    if not usable:
        return None
    wcfg = (model or {}).get("weights")
    w = run_weights(usable, today, wcfg)
    vals = [float(r[1]) for r in usable]
    tw = sum(w)
    base = sum(wi * v for wi, v in zip(w, vals)) / tw
    n = len(usable)
    sd = (sum(wi * (v - base) ** 2 for wi, v in zip(w, vals)) / tw) ** 0.5 if n >= 2 else None
    latest = usable[0][0]
    gap = _days(rd, latest) if rd else None
    f = condition_features(usable, w, today, rd) if (today and rd) else None
    value = base
    adjust: list[dict] = []
    reasons: list[str] = []
    group = "none"
    if f is not None:
        score = change_score(f)
        group = change_group(score, model)
        if model and model.get("coef"):
            cap = float(model.get("cap", 8.0))
            center = model.get("center", {})
            parts: dict[str, float] = {}
            const = float(model.get("intercept", 0.0))
            for k, c_ in model["coef"].items():
                if k not in f:
                    continue
                if k in CENTERED:                           # 馬齢・斤量差は『平均的な値』からの差で効く
                    parts[k] = float(c_) * (f[k] - float(center.get(k, 0.0)))
                else:                                       # 面の切り替えなどは『変化なし(0)』が基準。平均との差は定数に回す
                    parts[k] = float(c_) * f[k]
                    const -= float(c_) * float(center.get(k, 0.0))
            total = sum(parts.values()) + const
            clipped = max(-cap, min(cap, total))
            scale = (clipped / total) if total else 1.0
            for k, v in parts.items():
                if abs(v * scale) >= 0.3:
                    adjust.append({"key": k, "label": FEATURE_LABELS.get(k, k), "delta": round(v * scale, 1)})
            if abs(const * scale) >= 0.3:
                adjust.append({"key": "const", "label": "全体の傾向", "delta": round(const * scale, 1)})
            value = base + clipped
        reasons = _reasons(f, today, usable, _days(rd, latest))
    if n <= REFERENCE_MAX_RUNS:
        conf = "reference"
    elif gap is not None and gap > STALE_DAYS:
        conf = "stale"
    else:
        conf = "normal"
    rng = max(sd, MIN_RANGE) if sd is not None else None
    extra = ((model or {}).get("resid_sd") or {}).get(group)
    if extra is not None:
        rng = max(rng or 0.0, float(extra))                 # 過去走のばらつきだけでなく、実際の誤差（検証済みの標準偏差。条件変化が大きいほど大きい）まで広げる
    out = {
        "value": round(value, 1),
        "range": round(rng, 1) if rng is not None else None,
        "n_runs": n,
        "latest_run_date": _fmt(latest),
        "confidence": conf,
        "low_conf_runs": sum(1 for r in usable if r[2]),
        "gap_days": gap,
        "base_value": round(base, 1),
        "change": group,
        "reasons": reasons,
        "adjust": adjust,
        "recent": [{"date": _fmt(r[0]), "value": round(float(r[1]), 1), "low": bool(r[2]), **({"race_id": r[9]} if r[9] else {})} for r in usable],   # 実測の直近走（新しい順）
    }
    return out


def rank_and_gap(items: dict[Any, dict | None]) -> None:
    """レース内で、最上位との差（秒。1 点 = 0.1 秒）と順位を足す（高い指数ほど速い想定）。items の値を直接更新する。"""
    known = {k: v for k, v in items.items() if v}
    if not known:
        return
    top = max(v["value"] for v in known.values())
    order = sorted(known, key=lambda k: -known[k]["value"])
    for rank, k in enumerate(order, 1):
        known[k]["rank"] = rank
        known[k]["gap_sec"] = round((top - known[k]["value"]) * SEC_PER_POINT, 2)


# ── データの読み込み ─────────────────────────────────────────────────

def _load_year(year: int, base: Path) -> tuple[dict[str, list[Run]], str]:
    """その年の {horse_id: [走の記録]} と、その年の最新の走の日付。ファイルの更新時刻でキャッシュ。"""
    ax = hg.AXES["megu_v3a_abs"]
    legacy = hg.AXES["megu"]
    p = base / str(year) / ax.file
    rr = base / str(year) / "race_result_flat.parquet"
    sig = (str(p), hg._mtime(p), str(rr), hg._mtime(rr))
    key = f"{base}|{year}"
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1], hit[2]
    out: dict[str, list[Run]] = {}
    latest = ""
    if p.exists():
        import pyarrow.parquet as pq

        names = set(pq.read_schema(p).names)
        use = ax if ax.value_col in names else legacy       # dev のモックは従来の列だけ
        ctx_cols = ["venue", "surface", "distance", "grade", "jockey_weight", "sex_age"]
        cols = ["race_id", "horse_id", use.value_col] + [c for c in (use.status_col, use.conf_col, "date", *ctx_cols) if c and c in names]
        df = hg._read(p, columns=cols)
        if rr.exists():
            rnames = set(pq.read_schema(rr).names)
            mkey = ["race_id", "horse_id"] if "horse_id" in rnames else ["race_id"]       # 馬ごとの行が無い古い形式はレース単位で結合
            rcols = mkey + [c for c in ctx_cols if c in rnames and c not in df.columns] + (["date"] if "date" not in df.columns and "date" in rnames else [])
            ctx = hg._read(rr, columns=rcols).drop_duplicates(mkey)
            df = df.merge(ctx, on=mkey, how="left")
        if "date" not in df.columns:
            df["date"] = None
        for c in ctx_cols:
            if c not in df.columns:
                df[c] = None
        df = df[df[use.value_col].notna() & df["date"].notna()]
        latest = max((_digits(d) for d in df["date"]), default="")      # 状態に関係なく、指数が計算されている最新の日
        if use.status_col and use.status_col in df.columns:
            df = df[df[use.status_col].isin(use.valid_set)]
        lows = df[use.conf_col] if use.conf_col and use.conf_col in df.columns else [False] * len(df)
        for hid, rid_, d, v, lo, ve, su, di, gr, jw, sa in zip(df["horse_id"], df["race_id"], df["date"], df[use.value_col], lows, df["venue"], df["surface"],
                                                                df["distance"], df["grade"], df["jockey_weight"], df["sex_age"]):
            ymd = _digits(d)
            if len(ymd) == 8:
                out.setdefault(str(hid), []).append((
                    ymd, float(v), lo is True or lo == 1, surface_of(su), (float(di) if di == di and di is not None else None),
                    (str(ve) if ve is not None and ve == ve else None), class_level(gr), (float(jw) if jw == jw and jw is not None else None), age_of(sa),
                    str(rid_),
                ))
    with _lock:
        _cache[key] = (sig, out, latest)
    return out, latest


DETAIL_COLS = ["race_id", "horse_id", "race_name", "race_class", "round", "venue", "surface", "distance", "track_condition", "grade", "field_size",
               "finish_position", "popularity", "jockey_name", "jockey_weight", "last_3f", "passing_order", "weight", "weight_change", "margin"]


def _num(v: Any, nd: int | None = None) -> float | int | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(f, nd) if nd is not None else (int(f) if f == int(f) else f)


def _txt(v: Any) -> str | None:
    return None if v is None or v != v or str(v).strip() in ("", "nan", "None") else str(v).strip()


def _class_label(row: Mapping[str, Any]) -> str | None:
    """『2歳未勝利』『3歳以上1勝クラス』『G3』など、年齢つきのクラス名。読めなければ None（画面はレース名で代える）。"""
    try:
        from src.pipeline.inference import class_benchmark as cb

        age, tier = cb.cell_of(row.get("race_class"), row.get("grade"), row.get("race_name"))
        return cb.label_of(age, tier) if tier else None
    except Exception:  # noqa: BLE001
        return None


def race_details(pairs: Iterable[tuple[str, str]], base: Path | None = None) -> dict[tuple[str, str], dict]:
    """(race_id, horse_id) → 過去走のレース情報（馬柱用）。実績の表を年ごとに、必要な馬だけ絞って読む。読めない年・列は飛ばす。"""
    base = base or hg.tables_dir()
    need: dict[int, dict[str, set[str]]] = {}
    for rid, hid in pairs:
        if rid and hid and rid[:4].isdigit():
            need.setdefault(int(rid[:4]), {}).setdefault(hid, set()).add(rid)
    out: dict[tuple[str, str], dict] = {}
    for year, by_horse in need.items():
        p = base / str(year) / "race_result_flat.parquet"
        if not p.exists():
            continue
        try:
            import pyarrow.parquet as pq

            names = set(pq.read_schema(p).names)
            cols = [c for c in DETAIL_COLS if c in names]
            if "race_id" not in cols or "horse_id" not in cols:
                continue
            df = hg._read(p, columns=cols, filters=[("horse_id", "in", list(by_horse))])
        except Exception:
            continue
        for row in df.to_dict("records"):
            key = (str(row.get("race_id")), str(row.get("horse_id")))
            if key[0] not in by_horse.get(key[1], ()):
                continue
            out[key] = {
                "cls": _class_label(row),
                "race_name": _txt(row.get("race_name")), "round": _num(row.get("round")), "venue": _txt(row.get("venue")),
                "surface": _txt(row.get("surface")), "distance": _num(row.get("distance")), "going": _txt(row.get("track_condition")),
                "grade": _txt(row.get("grade")), "field": _num(row.get("field_size")), "finish": _num(row.get("finish_position")),
                "pop": _num(row.get("popularity")), "jockey": _txt(row.get("jockey_name")), "jw": _num(row.get("jockey_weight"), 1),
                "l3f": _num(row.get("last_3f"), 1), "passing": _txt(row.get("passing_order")), "bw": _num(row.get("weight")),
                "bw_diff": _num(row.get("weight_change")), "margin": _txt(row.get("margin")),
            }
    return out


def runs_for_horse(horse_id: str, race_date: str | None, base: Path | None = None) -> tuple[list[Run], str]:
    """対象レース日の年とその前々年までの走（直近 5 走を取るのに十分）と、データ全体の最新の走の日付。"""
    base = base or hg.tables_dir()
    rd = _digits(race_date)
    y = int(rd[:4]) if len(rd) >= 4 else None
    years = [y, y - 1, y - 2] if y else []
    runs: list[Run] = []
    through = ""
    for yy in years:
        if yy < hg.FIRST_YEAR:
            continue
        m, latest = _load_year(yy, base)
        through = max(through, latest)
        runs.extend(m.get(horse_id, []))
    return runs, through


def today_context(card: dict | None, entry: dict | None) -> dict | None:
    """今回のレースの条件（前日に分かるもの）。出馬表カードと、その馬の行から作る。"""
    card = card or {}
    entry = entry or {}
    su = surface_of(card.get("surface"))
    try:
        di = float(card.get("distance"))
    except (TypeError, ValueError):
        di = None
    if not su or not di:
        return None
    jw = entry.get("jockey_weight")
    try:
        jw = float(jw) if jw else None
    except (TypeError, ValueError):
        jw = None
    return {"surface": su, "distance": di, "venue": card.get("venue") or None, "level": class_level(card.get("grade")),
            "weight": jw, "age": age_of(entry.get("sex_age"))}


def attach_expected_megu(resp: dict, race_date: str | None, base: Path | None = None, past_doc: dict | None = None,
                         card: dict | None = None, model: dict | None = None) -> dict:
    """予測 API の応答（predictions[]）の各馬に expected_megu を、レースに expected_megu_meta を足す。他の項目は変えない。

    past_doc（馬柱）があれば、馬柱にあって指数に反映されていない直近の走を検出し、『直近の走は未反映』を付ける。"""
    preds = resp.get("predictions") or []
    if not preds:
        return resp
    base = base or hg.tables_dir()
    model = model if model is not None else load_model()
    last_past: dict[str, str] = {}
    for e in (past_doc or {}).get("entries") or []:
        if isinstance(e, dict):
            ds = [_digits(p.get("date")) for p in (e.get("past_races") or []) if isinstance(p, dict)]
            ds = [d for d in ds if len(d) == 8 and (not _digits(race_date) or d < _digits(race_date))]
            if ds:
                last_past[str(e.get("horse_id") or "")] = max(ds)
    card = card or {k: resp.get(k) for k in ("venue", "surface", "distance")}
    by_id = {str(e.get("horse_id") or ""): e for e in (card.get("entries") or []) if isinstance(e, dict)}
    items: dict[Any, dict | None] = {}
    runs_by_i: dict[int, list] = {}
    through = ""
    for i, p in enumerate(preds):
        hid = str(p.get("horse_id") or "")
        runs, t = runs_for_horse(hid, race_date, base) if hid else ([], "")
        runs_by_i[i] = runs
        through = max(through, t)
        ex = expected_from_runs(runs, race_date, today_context(card, {**by_id.get(hid, {}), **{k: v for k, v in p.items() if v}}), model)
        if ex is not None and last_past.get(hid) and last_past[hid] > _digits(ex["latest_run_date"]):
            ex["latest_unreflected"] = True
            ex["latest_past_date"] = _fmt(last_past[hid])
        items[i] = ex
    rank_and_gap(items)
    pairs = [(r["race_id"], str(preds[i].get("horse_id") or "")) for i, ex in items.items() if ex for r in ex.get("recent") or [] if r.get("race_id")]
    details = race_details(pairs, base) if pairs else {}
    for i, ex in items.items():
        for r in (ex or {}).get("recent") or []:
            d = details.get((r.get("race_id", ""), str(preds[i].get("horse_id") or "")))
            if d:
                r.update({k: v for k, v in d.items() if v is not None})
    try:                                                                  # v3b の集約値と、直近走の『無視してよい低下』の判定（tp-69。参考表示用で、集約から除かない）
        from src.pipeline.inference.megu_excusable_attach import annotate as _annotate

        for i, ex in items.items():
            _annotate(ex, runs_by_i.get(i) or [])
    except Exception:  # noqa: BLE001
        pass
    for i, p in enumerate(preds):
        p["expected_megu"] = items[i]
    n_known = sum(1 for v in items.values() if v)
    resp["expected_megu_meta"] = {
        "axis": "megu_v3a_abs", "name": "想定めぐ指数（過去のめぐ指数から。AI予測ではありません）",
        "method": f"過去走の絶対指数（斤量55kg換算）の直近{N_RUNS}走を、直近ほど重い加重平均（1点＝{SEC_PER_POINT}秒）。"
                  + ("条件の変化（面・距離・クラス・休み明け・競馬場・馬齢）は、2020〜2024年で学習した補正を足す" if model and model.get("coef") else "条件の変化の補正は未適用"),
        "data_through": _fmt(through) if through else None, "n_horses": len(preds), "n_known": n_known,
        "model": ({k: model.get(k) for k in ("version", "trained_on", "validated_on", "eval")} if model else None),
    }
    try:                                                                  # このクラスの基準（突破ライン）。表が無い・クラスが読めないレースは付けない
        from src.pipeline.inference.class_benchmark import lookup as _bench

        bm = _bench(card.get("race_class"), card.get("grade"), card.get("race_name") or resp.get("race_name"),
                    card.get("surface") or resp.get("surface"), card.get("distance") or resp.get("distance"))
        if bm:
            resp["class_benchmark"] = bm
    except Exception:  # noqa: BLE001
        bm = None
    try:                                                                  # めぐ印（tp-66）。表が無い・過去走が無い・出走が少ないときは印を付けず理由だけ返す
        from src.pipeline.inference.megu_mark import compute_marks

        marks = compute_marks(
            [{"key": i, "horse_number": p.get("horse_number"), "runs": runs_by_i.get(i) or [], "race_date": race_date} for i, p in enumerate(preds)],
            (bm or {}).get("line") if isinstance(bm, dict) else None)
        for i, p in enumerate(preds):
            if marks.get(i):
                p["megu_mark"] = marks[i]
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.pipeline.inference.growth_attach import attach_growth       # 成長度（KEIBA_GROWTH_INDEX=1 のときだけ。既定は何もしない）

        resp = attach_growth(resp, race_date or "", items, base)
    except Exception:  # noqa: BLE001
        pass
    return resp
