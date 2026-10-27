"""負荷テスト: 同時接続数を段階的に増やし、どこまで問題なく捌けるか／超えたらどうなるか／戻るまで何秒かを測る。

シナリオ:
  browse  利用者の画面遷移を真似る（週間予測ページ → 開催日一覧 → 推奨度 → 数レースの AI 予測と詳細。考える時間つき）。
          段の数 = 同時にいる利用者数。複数のページ・API が混ざる。
  single  1 つのアドレスだけに、待ち時間なしで繰り返し当てる。段の数 = 同時リクエスト数。

安全装置（中断して負荷を止める）:
  ホストの 1 分平均負荷が コア数×--max-load-per-core を超えた／空きメモリが --min-mem-gb を下回った／ヘルスチェックが連続で失敗。
  --stop-on-collapse で、崩壊（エラー率や遅さが基準を超えた）を 1 段観測したらそこで増やすのをやめる。
  負荷を止めたあと、ヘルスチェックと実エンドポイントが元の速さに戻るまでの秒数（回復時間）を測る。

評価の基準（段ごと）: 快適 = p95 < 1 秒かつエラー 0.5% 未満 / 許容 = p95 < 3 秒かつエラー 1% 未満 / 崩壊 = エラー 10% 以上または p95 ≥ 15 秒。

例（stg ホストで。公開経路は通さず、ホスト内のポートに直接当てる）:
  python -m src.scripts.ops.load_monkey browse --pools /tmp/lm/pools.json --steps 2,5,10,25,50,100,200 --hold 25 --out /tmp/lm/load_browse
  python -m src.scripts.ops.load_monkey single --path /api/race/202604040301/predictions --steps 1,5,20,50,100,200,400 --out /tmp/lm/load_single
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import resource
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

SLO_GOOD = (1000.0, 0.005)
SLO_OK = (3000.0, 0.01)
COLLAPSE = (15_000.0, 0.10)


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(len(s) * p / 100))], 1)


def classify(p95: float | None, err: float) -> str:
    if p95 is None or err >= COLLAPSE[1] or p95 >= COLLAPSE[0]:
        return "崩壊"
    if p95 < SLO_GOOD[0] and err < SLO_GOOD[1]:
        return "快適"
    if p95 < SLO_OK[0] and err < SLO_OK[1]:
        return "許容"
    return "悪化"


def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 1e9


def rss_mb(pids: list[int]) -> float:
    tot = 0.0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    tot += int(line.split()[1]) / 1024
        except OSError:
            pass
    return round(tot, 1)


class Stats:
    def __init__(self) -> None:
        self.lat_ok: list[float] = []
        self.lat_all: list[float] = []
        self.status: Counter[str] = Counter()
        self.by_path: dict[str, list[float]] = {}

    def add(self, key: str, status: str, ms: float, ok: bool) -> None:
        self.status[status] += 1
        self.lat_all.append(ms)
        if ok:
            self.lat_ok.append(ms)
            self.by_path.setdefault(key, []).append(ms)

    def summary(self, secs: float) -> dict[str, Any]:
        n = len(self.lat_all)
        ok = len(self.lat_ok)
        err = (n - ok) / n if n else 1.0
        p95 = percentile(self.lat_all, 95)           # 失敗（時間切れ含む）も遅さに数える
        return {
            "requests": n, "ok": ok, "rps": round(n / secs, 1), "error_rate": round(err, 4),
            "p50_ms": percentile(self.lat_all, 50), "p95_ms": p95, "p99_ms": percentile(self.lat_all, 99), "max_ms": round(max(self.lat_all), 1) if n else None,
            "status": dict(self.status.most_common()), "verdict": classify(p95, err),
        }


class Target:
    def __init__(self, a: argparse.Namespace) -> None:
        self.bases = {"F": a.fastapi, "K": a.flask, "N": a.next}
        self.all_base = a.all_base
        self.headers = {"User-Agent": "keiba-load-monkey/1", "Accept": "*/*"}
        if a.all_base:
            key = os.environ.get(a.key_env, "")
            if key:
                self.headers["X-Keiba-Proxy-Key"] = key
        else:
            from src.api.auth import COOKIE_NAME, ROLE_DEV, _make_token

            self.headers["Cookie"] = f"{COOKIE_NAME}={_make_token(role=ROLE_DEV)}"

    def url(self, path: str) -> str:
        if self.all_base:
            return self.all_base.rstrip("/") + path
        kind = "K" if path.startswith("/api/v1") else "F" if path.startswith(("/api", "/docs")) else "N"
        return self.bases[kind].rstrip("/") + path


async def fetch(client: Any, target: Target, path: str, stats: Stats, timeout: float) -> int:
    t0 = time.perf_counter()
    try:
        r = await client.get(target.url(path), headers=target.headers, timeout=timeout)
        ms = (time.perf_counter() - t0) * 1000
        code = r.status_code
        stats.add(_generic(path), str(code), ms, 200 <= code < 300)
        return code
    except Exception as exc:  # noqa: BLE001
        ms = (time.perf_counter() - t0) * 1000
        stats.add(_generic(path), type(exc).__name__, ms, False)
        return 0


def _generic(path: str) -> str:
    import re

    return re.sub(r"\d{8,12}", "N", path.split("?")[0])


def browse_plan(pools: dict[str, Any], rng: random.Random) -> list[tuple[str, float]]:
    """1 人の利用者の操作列: (パス, その後の考える時間)。直近の開催日を多めに選ぶ。"""
    items = pools["items"]
    hot = [i for i in items if i["hot"]] or items
    pick = rng.choice(hot if rng.random() < 0.7 else items)
    same = [i for i in items if i["date"] == pick["date"]] or [pick]
    d = pick["date"]
    plan: list[tuple[str, float]] = [
        ("/weekly-predictions", rng.uniform(0.5, 1.5)),
        ("/api/auth/status", 0.1),
        ("/api/scrape-dates?picker_past_days=30&picker_future_days=14", 0.2),
        (f"/api/race-list/{d}", 0.3),
        (f"/api/v1/betting/race-values?date={d}", rng.uniform(1.0, 3.0)),
    ]
    for it in rng.sample(same, min(len(same), rng.randint(2, 4))):
        plan.append((f"/api/race/{it['race_id']}/predictions", rng.uniform(1.0, 3.0)))
        if rng.random() < 0.5:
            plan.append((f"/api/race/{it['race_id']}", rng.uniform(1.0, 3.0)))
    return plan


async def vu_browse(client: Any, target: Target, pools: dict[str, Any], stats: Stats, stop: asyncio.Event, seed: int, timeout: float, no_pages: bool) -> None:
    rng = random.Random(seed)
    await asyncio.sleep(rng.uniform(0, 2.0))           # 一斉に始めない
    while not stop.is_set():
        for path, think in browse_plan(pools, rng):
            if stop.is_set():
                return
            if no_pages and not path.startswith("/api"):
                continue
            await fetch(client, target, path, stats, timeout)
            try:
                await asyncio.wait_for(stop.wait(), timeout=think)
                return
            except asyncio.TimeoutError:
                pass


async def vu_single(client: Any, target: Target, path: str, stats: Stats, stop: asyncio.Event, timeout: float, pools: dict[str, Any], seed: int) -> None:
    """path に {race_id} / {date} があれば、リクエストごとにサンプルからランダムに選ぶ（誰も開いていないキーを当て続ける）。"""
    rng = random.Random(seed)
    items = pools.get("items") or []
    while not stop.is_set():
        p = path
        if items and ("{race_id}" in p or "{date}" in p):
            it = rng.choice(items)
            p = p.replace("{race_id}", it["race_id"]).replace("{date}", it["date"])
        await fetch(client, target, p, stats, timeout)


async def monitor(a: argparse.Namespace, target: Target, samples: list[dict[str, Any]], abort: asyncio.Event, reason: list[str], stop: asyncio.Event) -> None:
    import httpx

    cores = os.cpu_count() or 1
    fails = 0
    pids = [int(x) for x in a.pids.split(",") if x]
    async with httpx.AsyncClient() as c:
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                r = await c.get(target.url(a.health), headers=target.headers, timeout=10)
                ok = r.status_code < 500
            except Exception:  # noqa: BLE001
                ok = False
            ms = (time.perf_counter() - t0) * 1000
            fails = 0 if ok else fails + 1
            load1, mem = os.getloadavg()[0], mem_available_gb()
            samples.append({"t": round(time.time(), 1), "load1": round(load1, 2), "mem_gb": round(mem, 1), "health_ms": round(ms, 1), "health_ok": ok, "rss_mb": rss_mb(pids)})
            if load1 > cores * a.max_load_per_core:
                reason.append(f"ホスト負荷 {load1:.1f} > {cores * a.max_load_per_core:.1f}")
            elif mem < a.min_mem_gb:
                reason.append(f"空きメモリ {mem:.1f}GB < {a.min_mem_gb}GB")
            elif fails >= a.health_fail_limit:
                reason.append(f"ヘルスチェックが {fails} 回連続で失敗")
            if reason:
                abort.set()
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def run_step(a: argparse.Namespace, target: Target, pools: dict[str, Any], n: int, hold: float, abort: asyncio.Event) -> dict[str, Any]:
    import httpx

    if getattr(a, "procs", 1) > 1 and n >= a.procs * 4:
        return await run_step_multi(a, pools, n, hold, abort)

    stats = Stats()
    stop = asyncio.Event()
    limits = httpx.Limits(max_connections=n + 20, max_keepalive_connections=n + 20)
    async with httpx.AsyncClient(limits=limits, verify=False) as client:
        if a.mode == "browse":
            tasks = [asyncio.create_task(vu_browse(client, target, pools, stats, stop, a.seed * 1000 + i, a.timeout, a.no_pages)) for i in range(n)]
        else:
            tasks = [asyncio.create_task(vu_single(client, target, a.path, stats, stop, a.timeout, pools, a.seed * 1000 + i)) for i in range(n)]
        t0 = time.time()
        while time.time() - t0 < hold and not abort.is_set():
            await asyncio.sleep(0.5)
        stop.set()
        await asyncio.wait(tasks, timeout=a.timeout + 5)
        for t in tasks:
            t.cancel()
        secs = max(time.time() - t0, 0.001)
    out = stats.summary(secs)
    out["level"] = n
    out["slowest_paths"] = sorted(({"path": k, "p95_ms": percentile(v, 95), "n": len(v)} for k, v in stats.by_path.items()), key=lambda x: -(x["p95_ms"] or 0))[:4]
    return out


def _child(args: dict[str, Any], n: int, hold: float, pools: dict[str, Any], idx: int) -> dict[str, Any]:
    """別プロセスで n 人分の負荷をかける（1 プロセスの Python は 400〜500 req/s 前後で頭打ちになり、サーバーの限界と区別できないため）。"""
    a = argparse.Namespace(**args)
    a.seed = a.seed * 100 + idx
    a.procs = 1
    st: dict[str, Any] = {}

    async def go() -> None:
        import httpx

        target = Target(a)
        stats = Stats()
        stop = asyncio.Event()
        limits = httpx.Limits(max_connections=n + 20, max_keepalive_connections=n + 20)
        async with httpx.AsyncClient(limits=limits, verify=False) as client:
            if a.mode == "browse":
                tasks = [asyncio.create_task(vu_browse(client, target, pools, stats, stop, a.seed * 1000 + i, a.timeout, a.no_pages)) for i in range(n)]
            else:
                tasks = [asyncio.create_task(vu_single(client, target, a.path, stats, stop, a.timeout, pools, a.seed * 1000 + i)) for i in range(n)]
            await asyncio.sleep(hold)
            stop.set()
            await asyncio.wait(tasks, timeout=a.timeout + 5)
            for t in tasks:
                t.cancel()
        st.update(lat_all=stats.lat_all, lat_ok=stats.lat_ok, status=dict(stats.status), by_path=stats.by_path)

    asyncio.run(go())
    return st


async def run_step_multi(a: argparse.Namespace, pools: dict[str, Any], n: int, hold: float, abort: asyncio.Event) -> dict[str, Any]:
    import concurrent.futures
    import multiprocessing

    k = max(1, min(a.procs, n))
    per = [n // k + (1 if i < n % k else 0) for i in range(k)]
    args = {kk: vv for kk, vv in vars(a).items()}
    loop = asyncio.get_running_loop()
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=k, mp_context=multiprocessing.get_context("spawn")) as ex:
        futs = [loop.run_in_executor(ex, _child, args, per[i], hold, pools, i) for i in range(k)]
        parts = await asyncio.gather(*futs)
    secs = time.time() - t0 - 2.0                      # 子プロセスの起動・終了の分を除く
    stats = Stats()
    stats.lat_all = [x for p in parts for x in p["lat_all"]]
    stats.lat_ok = [x for p in parts for x in p["lat_ok"]]
    for p in parts:
        stats.status.update(p["status"])
        for key, v in p["by_path"].items():
            stats.by_path.setdefault(key, []).extend(v)
    out = stats.summary(max(secs, hold))
    out["level"] = n
    out["procs"] = k
    out["slowest_paths"] = sorted(({"path": kk, "p95_ms": percentile(v, 95), "n": len(v)} for kk, v in stats.by_path.items()), key=lambda x: -(x["p95_ms"] or 0))[:4]
    return out


async def probe_recovery(a: argparse.Namespace, target: Target, baseline_ms: float, probe_path: str) -> dict[str, Any]:
    """負荷を止めたあと、ヘルスチェックと実エンドポイントが基準の 2 倍以内（かつ 500ms 未満）に戻るまでの秒数。"""
    import httpx

    t0 = time.time()
    rows = []
    async with httpx.AsyncClient(verify=False) as c:
        while time.time() - t0 < a.recovery_max:
            res = {}
            for name, path in (("health", a.health), ("probe", probe_path)):
                s = time.perf_counter()
                try:
                    r = await c.get(target.url(path), headers=target.headers, timeout=20)
                    res[name] = (r.status_code, (time.perf_counter() - s) * 1000)
                except Exception as exc:  # noqa: BLE001
                    res[name] = (0, (time.perf_counter() - s) * 1000)
                    res[name + "_err"] = type(exc).__name__
            good = all(res[k][0] and res[k][0] < 500 and res[k][1] < max(baseline_ms * 2, 500) for k in ("health", "probe"))
            rows.append({"t": round(time.time() - t0, 1), **{k: (round(v[1]), v[0]) for k, v in res.items() if isinstance(v, tuple)}})
            if good:
                return {"recovered": True, "seconds": round(time.time() - t0, 1), "probes": rows[-5:]}
            await asyncio.sleep(2)
    return {"recovered": False, "seconds": a.recovery_max, "probes": rows[-5:]}


async def amain(a: argparse.Namespace) -> int:
    target = Target(a)
    pools = json.loads(Path(a.pools).read_text(encoding="utf-8")) if a.pools else {"items": []}
    if a.mode == "browse" and not pools["items"]:
        print("browse には --pools が必要です", file=sys.stderr)
        return 2
    probe_path = a.path.replace("{race_id}", pools["items"][0]["race_id"]).replace("{date}", pools["items"][0]["date"]) if a.mode == "single" and pools["items"] else a.path if a.mode == "single" else (f"/api/race/{pools['items'][0]['race_id']}/predictions" if pools["items"] else a.health)
    abort, stop_mon, reason = asyncio.Event(), asyncio.Event(), []
    samples: list[dict[str, Any]] = []
    mon = asyncio.create_task(monitor(a, target, samples, abort, reason, stop_mon))
    results = []
    print(f"[基準] 1 並列で測定中…", flush=True)
    base = await run_step(a, target, pools, 1, max(8.0, a.hold / 2), abort)
    base_ms = base["p50_ms"] or 100.0
    print(f"  基準 p50 {base_ms}ms  p95 {base['p95_ms']}ms  エラー {base['error_rate']:.1%}", flush=True)
    results.append(base)
    collapsed_at = None
    for n in [int(x) for x in a.steps.split(",") if x]:
        if abort.is_set():
            break
        r = await run_step(a, target, pools, n, a.hold, abort)
        results.append(r)
        print(f"  {n:>4} 並列: {r['rps']:>7} req/s  p50 {r['p50_ms']}  p95 {r['p95_ms']}  p99 {r['p99_ms']}ms  エラー {r['error_rate']:.1%}  [{r['verdict']}]  {r['status']}", flush=True)
        if r["verdict"] == "崩壊" and collapsed_at is None:
            collapsed_at = n
            if a.stop_on_collapse:
                break
        await asyncio.sleep(a.cooldown)
    abort_reason = reason[:1]
    abort.clear()
    print("[回復] 負荷を止めて、元の速さに戻るまでを測定…", flush=True)
    rec = await probe_recovery(a, target, base_ms, probe_path)
    print(f"  回復: {rec['recovered']}  {rec['seconds']}秒", flush=True)
    stop_mon.set()
    await asyncio.wait([mon], timeout=5)
    good = max((r["level"] for r in results[1:] if r["verdict"] == "快適"), default=0)
    okk = max((r["level"] for r in results[1:] if r["verdict"] in ("快適", "許容")), default=0)
    summary = {"mode": a.mode, "path": a.path if a.mode == "single" else "(browse)", "steps": results, "good_max": good, "acceptable_max": okk,
               "collapsed_at": collapsed_at, "aborted": abort_reason, "recovery": rec,
               "peak": {"load1": max((s["load1"] for s in samples), default=None), "min_mem_gb": min((s["mem_gb"] for s in samples), default=None),
                        "max_rss_mb": max((s["rss_mb"] for s in samples), default=None), "worst_health_ms": max((s["health_ms"] for s in samples), default=None),
                        "health_failures": sum(1 for s in samples if not s["health_ok"])}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "monitor.jsonl").write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    print(f"[結果] 快適 {good} / 許容 {okk} / 崩壊 {collapsed_at} / 中断 {abort_reason or 'なし'}  → {out}/summary.json", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["browse", "single"])
    ap.add_argument("--path", help="single: 当てるパス（例 /api/race/202604040301/predictions）")
    ap.add_argument("--pools", help="latency_monkey pools が作ったサンプル（browse 用）")
    ap.add_argument("--steps", default="2,5,10,25,50,100,200")
    ap.add_argument("--hold", type=float, default=25.0, help="1 段の秒数")
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=30.0, help="1 リクエストの待ち上限（秒）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=1, help="負荷発生プロセス数（段の並列数がこの 4 倍以上のときに分割。1 プロセスは 400〜500 req/s で頭打ち）")
    ap.add_argument("--no-pages", action="store_true", help="browse で Next.js のページを除く")
    ap.add_argument("--fastapi", default="http://127.0.0.1:23220")
    ap.add_argument("--flask", default="http://127.0.0.1:23210")
    ap.add_argument("--next", default="http://127.0.0.1:23200")
    ap.add_argument("--all-base", help="1 つの URL に全部当てる（Cloud Run のタグ付き URL など）。鍵は --key-env の環境変数から X-Keiba-Proxy-Key で付ける")
    ap.add_argument("--key-env", default="KEIBA_PROXY_KEY")
    ap.add_argument("--health", default="/api/health")
    ap.add_argument("--pids", default="", help="メモリ(RSS)を見るプロセス ID（カンマ区切り）")
    ap.add_argument("--max-load-per-core", type=float, default=0.8)
    ap.add_argument("--min-mem-gb", type=float, default=15.0)
    ap.add_argument("--health-fail-limit", type=int, default=5)
    ap.add_argument("--stop-on-collapse", action="store_true")
    ap.add_argument("--recovery-max", type=float, default=180.0)
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "keiba-load-monkey"))
    a = ap.parse_args(argv)
    if a.mode == "single" and not a.path:
        ap.error("single には --path が必要です")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(hard, soft), 8192), hard))
    except (ValueError, OSError):
        pass
    return asyncio.run(amain(a))


if __name__ == "__main__":
    sys.exit(main())
