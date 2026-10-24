"""気象庁「過去の気象データ」から、中央競馬場 10 か所の最寄り観測所の時別値（風向・風速・気温・降水量など）を取る。

  python -m src.scraper.jma_weather stations                     # 各競馬場に最も近い（現行・風観測あり）観測所を選び config/jma_stations.json に保存
  python -m src.scraper.jma_weather fetch --from 20160101 [--to 20261022] [--venues 05 06] [--max-requests 3000]
  python -m src.scraper.jma_weather daily                        # 直近 3 日分（昨日まで）を補う。cron から毎日
  python -m src.scraper.jma_weather report [--years 2026]        # 競馬場×年の取得済み日数
  python -m src.scraper.jma_weather export [--out PATH]          # 分析用のフラット表（Parquet。1 行＝競馬場×観測時刻）

保存: HybridStorage のカテゴリ jma_weather_hourly、キー <競馬場コード>_<YYYYMM>（stg/prod は GCS の others/jma_weather_hourly/、dev は dev_mock）。中身は 日付 → {station, fetched_at, hours[24]}。取得済みの日は取り直さない。
方針: 間隔は約 1.5 秒、HTTP が 200 以外・表が読めない場合は 1 件で即中断（再開は同じコマンド）。出典: 気象庁ホームページ（過去の気象データ検索）。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
STATIONS_PATH = ROOT / "config" / "jma_stations.json"
CATEGORY = "jma_weather_hourly"
SOURCE = "気象庁 過去の気象データ検索（https://www.data.jma.go.jp/obd/stats/etrn/）"
BASE = "https://www.data.jma.go.jp/obd/stats/etrn"
UA = "keiba-vpn-research/1.0 (personal research; contact via repository owner)"
INTERVAL = (1.2, 1.8)

# 競馬場（JRA の場コード）。座標は住所からの概数（観測所の選定にだけ使う。数百 m の誤差は選定に影響しない想定）。prec_no は気象庁の府県予報区番号。
COURSES = [
    {"venue_code": "01", "name": "札幌", "address": "北海道札幌市中央区北16条西16丁目", "lat": 43.0726, "lon": 141.3338, "prec_no": 14},
    {"venue_code": "02", "name": "函館", "address": "北海道函館市湯川町1丁目8-2", "lat": 41.7752, "lon": 140.7777, "prec_no": 23},
    {"venue_code": "03", "name": "福島", "address": "福島県福島市松木町1-1", "lat": 37.7585, "lon": 140.4521, "prec_no": 36},
    {"venue_code": "04", "name": "新潟", "address": "新潟県新潟市北区笹山前913-1", "lat": 37.9565, "lon": 139.2061, "prec_no": 54},
    {"venue_code": "05", "name": "東京", "address": "東京都府中市日吉町1-1", "lat": 35.6618, "lon": 139.4856, "prec_no": 44},
    {"venue_code": "06", "name": "中山", "address": "千葉県船橋市古作1-1-1", "lat": 35.7261, "lon": 139.9603, "prec_no": 45, "extra_prec": [44]},
    {"venue_code": "07", "name": "中京", "address": "愛知県豊明市間米町敷田1180", "lat": 35.0620, "lon": 136.9905, "prec_no": 51},
    {"venue_code": "08", "name": "京都", "address": "京都府京都市伏見区葭島渡場島町32", "lat": 34.9043, "lon": 135.7283, "prec_no": 61, "extra_prec": [62, 60]},
    {"venue_code": "09", "name": "阪神", "address": "兵庫県宝塚市駒の町1-1", "lat": 34.7788, "lon": 135.3611, "prec_no": 63, "extra_prec": [62]},
    {"venue_code": "10", "name": "小倉", "address": "福岡県北九州市小倉南区北方3丁目11-1", "lat": 33.8478, "lon": 130.8736, "prec_no": 82, "extra_prec": [81]},
]

COMPASS16 = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東", "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
WIND_DEG = {n: i * 22.5 for i, n in enumerate(COMPASS16)}


class JmaError(RuntimeError):
    pass


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


# ------------------------------------------------------------------ 観測所の選定
_VP = re.compile(r"viewPoint\('([as])',((?:'[^']*',?)+)\)")


def parse_stations(html: str, prec_no: int) -> list[dict[str, Any]]:
    """府県ページの viewPoint(...) から観測所を取り出す。a（アメダス）は block_no が 4 桁、s（気象台等）は 5 桁。"""
    out = []
    for m in re.finditer(r"viewPoint\(('[^)]*)\)", html):
        a = [x.strip("'") for x in re.findall(r"'([^']*)'", m.group(1))]
        if len(a) < 15 or a[0] not in ("a", "s"):
            continue
        try:
            lat = float(a[4]) + float(a[5]) / 60
            lon = float(a[6]) + float(a[7]) / 60
        except ValueError:
            continue
        out.append({"type": a[0], "prec_no": prec_no, "block_no": a[1], "name": a[2], "lat": round(lat, 5), "lon": round(lon, 5), "height": a[8],
                    "has_precip": a[9] == "1", "has_wind": a[10] == "1", "has_temp": a[11] == "1", "ended": len(a) > 15 and a[15].strip() not in ("", "9999")})      # 終了年が 9999 なら現行の観測所
    uniq = {(s["type"], s["block_no"]): s for s in out}
    return list(uniq.values())


def nearest_station(course: dict[str, Any], stations: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [s for s in stations if s["has_wind"] and not s["ended"]]
    if not ok:
        return None
    best = min(ok, key=lambda s: haversine_km(course["lat"], course["lon"], s["lat"], s["lon"]))
    return {**best, "distance_km": round(haversine_km(course["lat"], course["lon"], best["lat"], best["lon"]), 2)}


def _get(url: str, session: requests.Session) -> str:
    time.sleep(random.uniform(*INTERVAL))
    r = session.get(url, timeout=30, headers={"User-Agent": UA})
    if r.status_code != 200:
        raise JmaError(f"HTTP {r.status_code}: {url}")
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def select_stations(session: requests.Session | None = None) -> dict[str, Any]:
    s = session or requests.Session()
    sel = {}
    for c in COURSES:
        cand = []
        for pn in [c["prec_no"], *c.get("extra_prec", [])]:      # 隣接府県の観測所のほうが近いことがある
            cand += parse_stations(_get(f"{BASE}/select/prefecture.php?prec_no={pn}", s), pn)
        st = nearest_station(c, cand)
        if not st:
            raise JmaError(f"{c['name']}: 風を観測している現行の観測所が見つかりません（prec_no={c['prec_no']}）")
        sel[c["venue_code"]] = {"venue": c["name"], "address": c["address"], "course_lat": c["lat"], "course_lon": c["lon"], "station": st, "candidates": len(cand)}
    return sel


def load_stations() -> dict[str, Any]:
    return json.loads(STATIONS_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 時別値の取得・解析
def hourly_url(st: dict[str, Any], d: date) -> str:
    kind = "s1" if st["type"] == "s" else "a1"
    return f"{BASE}/view/hourly_{kind}.php?prec_no={st['prec_no']}&block_no={st['block_no']}&year={d.year}&month={d.month}&day={d.day}&view="


def _num(s: str) -> tuple[float | None, str]:
    """値と品質印（')'=準正常値, ']'=資料不足値）。×・///・空は欠測、-- は現象なし（0）。"""
    t = s.strip()
    flag = ""
    while t and t[-1] in ") ]":
        if t[-1] in ")]":
            flag = t[-1]
        t = t[:-1].rstrip()
    if t in ("", "×", "///", "#", "*"):
        return None, flag
    if t == "--":
        return 0.0, flag
    try:
        return float(t), flag
    except ValueError:
        return None, flag


def parse_hourly(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tablefix1")
    if table is None:
        raise JmaError("時別値の表（tablefix1）が見つかりません")
    rows = table.find_all("tr")
    head = [r for r in rows if r.find("th") and not r.find("td")][:2]
    if len(head) < 2:
        raise JmaError("表の見出しが想定と違います")
    top = [(c.get_text(strip=True), int(c.get("colspan", 1))) for c in head[0].find_all(["th", "td"])]
    subs = iter(c.get_text(strip=True) for c in head[1].find_all(["th", "td"]))
    cols: list[str] = []
    for name, span in top:
        if span > 1:
            cols += [f"{name}::{next(subs, '')}" for _ in range(span)]
        else:
            cols.append(name)
    out = []
    for r in rows:
        tds = [c.get_text(strip=True) for c in r.find_all("td")]
        if not tds or not re.fullmatch(r"\d{1,2}", tds[0]) or len(tds) != len(cols):
            continue
        rec: dict[str, Any] = {"hour": int(tds[0])}
        for name, raw in zip(cols[1:], tds[1:]):
            sub = name.split("::")[-1]
            if sub == "風向":
                w = raw.strip()
                rec["wind_dir"] = w if w in WIND_DEG or w == "静穏" else None
                rec["wind_dir_deg"] = WIND_DEG.get(w)
                continue
            key = (
                "wind_speed_ms" if "風速" in sub else
                "precip_mm" if name.startswith("降水量") else "temp_c" if name.startswith("気温") else
                "dewpoint_c" if name.startswith("露点") else "humidity_pct" if name.startswith("湿度") else
                "pressure_hpa" if name.endswith("::現地") else "pressure_sea_hpa" if name.endswith("::海面") else
                "sunshine_h" if name.startswith("日照") else "solar_mj" if name.startswith("全天日射") else
                "snowfall_cm" if sub.startswith("降雪") else "snow_depth_cm" if sub.startswith("積雪") else
                "visibility_km" if name.startswith("視程") else None
            )
            if key:
                v, flag = _num(raw)
                rec[key] = v
                if flag:
                    rec.setdefault("flags", {})[key] = flag
            elif name.startswith("天気"):
                rec["weather_code_text"] = raw or None
            elif name.startswith("雲量"):
                rec["cloud_cover"] = raw or None
        out.append(rec)
    if not out:
        raise JmaError("時別値の行が 1 件も読めません（ページ構造の変更の可能性）")
    return out


# ------------------------------------------------------------------ 保存（HybridStorage。stg/prod は GCS）
def _default_store():
    from src.scraper.storage import HybridStorage

    return HybridStorage()


def load_month(venue_code: str, ym: str, store=None) -> dict[str, Any]:
    d = (store or _default_store()).load(CATEGORY, f"{venue_code}_{ym}", bypass_cache=True)
    return d if isinstance(d, dict) and "days" in d else {"venue_code": venue_code, "month": f"{ym[:4]}-{ym[4:]}", "source": SOURCE, "days": {}}


def save_month(data: dict[str, Any], store=None) -> None:
    key = f"{data['venue_code']}_{data['month'].replace('-', '')}"
    if not (store or _default_store()).save(CATEGORY, key, data):
        raise JmaError(f"保存に失敗しました（GCS 未接続など）: {key}")


def with_observed_at(recs: list[dict[str, Any]], d: date) -> list[dict[str, Any]]:
    """hour（1〜24）= その時刻の観測（直前 10 分の平均風など）。24 時は翌日 0 時。結合しやすいよう ISO 日時（+09:00）を付ける。"""
    base = datetime(d.year, d.month, d.day)
    return [{**r, "observed_at": (base + timedelta(hours=r["hour"])).strftime("%Y-%m-%dT%H:%M:00+09:00")} for r in recs]


def fetch_range(d_from: date, d_to: date, venues: list[str] | None = None, max_requests: int = 0, store=None,
                session: requests.Session | None = None, newest_first: bool = True) -> dict[str, int]:
    st_all = load_stations()
    s = session or requests.Session()
    days = [d_from + timedelta(n) for n in range((d_to - d_from).days + 1)]
    if newest_first:
        days.reverse()
    done = skipped = 0
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    dirty: set[tuple[str, str]] = set()
    try:
        for d in days:
            for vc, info in st_all.items():
                if venues and vc not in venues:
                    continue
                key = (vc, d.strftime("%Y%m"))
                data = cache.get(key) or cache.setdefault(key, load_month(vc, key[1], store))
                if d.isoformat() in data["days"]:
                    skipped += 1
                    continue
                if max_requests and done >= max_requests:
                    raise StopIteration
                recs = parse_hourly(_get(hourly_url(info["station"], d), s))
                if d >= date.today() - timedelta(days=1) and len(recs) < 24:      # 昨日以降は観測が出そろうまで保存しない（翌日分の取りこぼしを防ぐ）
                    continue
                data["days"][d.isoformat()] = {
                    "station": {k: info["station"][k] for k in ("type", "prec_no", "block_no", "name", "lat", "lon", "distance_km")},
                    "fetched_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
                    "hours": with_observed_at(recs, d),
                }
                dirty.add(key)
                done += 1
                if done % 50 == 0:
                    for k in dirty:
                        save_month(cache[k], store)
                    dirty.clear()
    except StopIteration:
        pass
    finally:
        for k in dirty:
            save_month(cache[k], store)
    return {"fetched": done, "skipped": skipped}


def _iter_months(years: list[int]):
    today = date.today()
    for y in years:
        for m in range(1, 13):
            if (y, m) <= (today.year, today.month):
                yield f"{y}{m:02d}"


def report(store=None, years: list[int] | None = None) -> dict[str, dict[int, int]]:
    """競馬場×年の取得済み日数（other カテゴリは list_keys が使えないので、キーを直接読む）。"""
    st = store or _default_store()
    rep: dict[str, dict[int, int]] = {}
    for vc in sorted(c["venue_code"] for c in COURSES):
        for ym in _iter_months(years or [date.today().year]):
            d = st.load(CATEGORY, f"{vc}_{ym}", bypass_cache=True)
            if isinstance(d, dict) and d.get("days"):
                y = int(ym[:4])
                rep.setdefault(vc, {})[y] = rep.get(vc, {}).get(y, 0) + len(d["days"])
    return rep


def export_flat(out: Path, store=None, years: list[int] | None = None) -> int:
    """分析用のフラット表。1 行＝競馬場×観測時刻。観測所・取得時刻の列つき。レースとの結合は venue_code と observed_at（発走の直前の時刻）で。"""
    import pandas as pd

    st = store or _default_store()
    rows = []
    for vc in sorted(c["venue_code"] for c in COURSES):
        for ym in _iter_months(years or list(range(2016, date.today().year + 1))):
            d = st.load(CATEGORY, f"{vc}_{ym}", bypass_cache=True)
            for day, v in ((d or {}).get("days") or {}).items():
                for h in v["hours"]:
                    rows.append({"venue_code": vc, "date": day, **h, "station_name": v["station"]["name"], "station_distance_km": v["station"]["distance_km"], "fetched_at": v["fetched_at"]})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("stations")
    f = sp.add_parser("fetch")
    f.add_argument("--from", dest="d_from", default="20160101")
    f.add_argument("--to", dest="d_to", default="")
    f.add_argument("--venues", nargs="*")
    f.add_argument("--max-requests", type=int, default=0)
    f.add_argument("--oldest-first", action="store_true")
    sp.add_parser("daily")
    rp = sp.add_parser("report")
    rp.add_argument("--years", type=int, nargs="*")
    ex = sp.add_parser("export")
    ex.add_argument("--out", default=str(ROOT / "data" / "page_reference" / "tables" / "jma_weather_hourly_flat.parquet"))
    ex.add_argument("--years", type=int, nargs="*")
    a = ap.parse_args()
    ymd = lambda s: datetime.strptime(s, "%Y%m%d").date()      # noqa: E731
    yesterday = date.today() - timedelta(days=1)
    try:
        if a.cmd == "stations":
            sel = select_stations()
            STATIONS_PATH.write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
            for vc, v in sel.items():
                st = v["station"]
                print(f"{vc} {v['venue']}: {st['name']}（{'アメダス' if st['type'] == 'a' else '気象台等'} {st['prec_no']}/{st['block_no']}）{st['distance_km']} km")
        elif a.cmd == "fetch":
            r = fetch_range(ymd(a.d_from), ymd(a.d_to) if a.d_to else yesterday, a.venues, a.max_requests, newest_first=not a.oldest_first)
            print(json.dumps(r))
        elif a.cmd == "daily":
            print(json.dumps(fetch_range(yesterday - timedelta(days=2), yesterday)))
        elif a.cmd == "export":
            print(json.dumps({"rows": export_flat(Path(a.out), years=a.years), "out": a.out}))
        else:
            print(json.dumps(report(years=a.years), ensure_ascii=False))
    except JmaError as e:
        print(f"中断: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
