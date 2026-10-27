"""Cloud Run の API の動作確認と、ホストの API との出力比較（デプロイ後・切替前に実行）。

    export KEIBA_PROXY_KEY=...            # 共有ヘッダ鍵（表示しない）。ホスト側は --compare-header で Access のヘッダを渡す
    python -m src.scripts.ops.cloudrun_smoke --base https://rNNN---keiba-api-xxxx-an.a.run.app --race-id 202608020311 \
        [--compare-base https://api-internal.megu-keiba.com --compare-header CF-Access-Client-Id=CF_ID --compare-header CF-Access-Client-Secret=CF_SECRET]

確認する項目: ①鍵なしの /api/health が 200 ②鍵なしの /api/race/.../predictions が 401 ③鍵ありで 200 ④/api/v1/health ⑤/ops/heartbeat/status
⑥（--compare-base）勝率・連対率・複勝率が馬番ごとに一致（浮動小数の許容 1e-6）。値（鍵）は出力しない。終了コード 0=全部 OK、1=NG。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

PROB_KEYS = ("win_prob", "top2_prob", "top3_prob")


def fetch(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "keiba-cloudrun-smoke/1", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()
    except Exception as exc:  # noqa: BLE001
        return 0, type(exc).__name__.encode()


def fetch_warm(url: str, headers: dict[str, str] | None = None, timeout: float = 75.0, retries: int = 2) -> tuple[int, bytes]:
    """初回の予測は、モデルや履歴の読み込みで 30〜60 秒かかることがある(Cloud Run の要求上限は 60 秒で、超えると 504)。
    待ち時間を延ばし、応答が無かった(0)・502/503/504 のときは取り直す（最初の要求でインスタンスが温まる）。"""
    code, body = fetch(url, headers, timeout)
    for _ in range(retries):
        if code not in (0, 502, 503, 504):
            break
        code, body = fetch(url, headers, timeout)
    return code, body


def prob_table(body: bytes) -> dict[int, tuple[float | None, ...]]:
    try:
        d = json.loads(body)
    except ValueError:
        return {}
    out: dict[int, tuple[float | None, ...]] = {}
    for p in d.get("predictions") or []:
        hn = p.get("horse_number")
        if isinstance(hn, int):
            out[hn] = tuple(p.get(k) for k in PROB_KEYS)
    return out


def diff_tables(a: dict, b: dict, tol: float = 1e-6) -> list[str]:
    msgs = []
    if set(a) != set(b):
        msgs.append(f"馬番の集合が違う: only_a={sorted(set(a) - set(b))} only_b={sorted(set(b) - set(a))}")
    for hn in sorted(set(a) & set(b)):
        for k, x, y in zip(PROB_KEYS, a[hn], b[hn]):
            if (x is None) != (y is None) or (x is not None and abs(float(x) - float(y)) > tol):
                msgs.append(f"馬番{hn} {k}: {x} != {y}")
    return msgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--key-env", default="KEIBA_PROXY_KEY", help="共有ヘッダ鍵が入った環境変数名")
    ap.add_argument("--compare-base")
    ap.add_argument("--compare-header", action="append", default=[], help="NAME=ENVVAR（比較先に付けるヘッダ。値は環境変数から）")
    a = ap.parse_args(argv)
    base = a.base.rstrip("/")
    key = os.environ.get(a.key_env, "")
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"[{'OK' if cond else 'NG'}] {name}{(' — ' + detail) if detail else ''}")

    pred = f"/api/race/{a.race_id}/predictions"
    check("鍵なしの /api/health が 200", fetch(base + "/api/health")[0] == 200)
    code = fetch(base + pred)[0]
    check("鍵なしの予測 API が 401", code == 401, f"HTTP {code}")
    if not key:
        check("鍵あり確認", False, f"環境変数 {a.key_env} が空です")
    else:
        code, body = fetch_warm(base + pred, {"X-Keiba-Proxy-Key": key})
        check("鍵ありの予測 API が 200", code == 200, f"HTTP {code}")
        check("鍵ありの /api/v1/health が 200", fetch(base + "/api/v1/health", {"X-Keiba-Proxy-Key": key})[0] == 200)
        sc, sb = fetch(base + "/ops/heartbeat/status", {"X-Keiba-Proxy-Key": key})
        check("ハートビートの状態が読める", sc == 200, (json.loads(sb).get("status") if sc == 200 else f"HTTP {sc}"))
        if a.compare_base and code == 200:
            hdrs = {}
            for spec in a.compare_header:
                name, _, env = spec.partition("=")
                hdrs[name] = os.environ.get(env, "")
            c2, b2 = fetch(a.compare_base.rstrip("/") + pred, hdrs)
            check("比較先の予測 API が 200", c2 == 200, f"HTTP {c2}")
            if c2 == 200:
                t1, t2 = prob_table(body), prob_table(b2)
                msgs = diff_tables(t1, t2)
                check(f"確率が一致（{len(t1)} 頭）", not msgs and bool(t1), "; ".join(msgs[:5]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
