#!/usr/bin/env python3
"""API キャッシュ設定(nginx-api-cache.conf + nginx-zones.conf)を、root なし・本番に触れずに検証する。

自前の nginx を一時ディレクトリで高ポートに立て、疑似の上流に対して次を確かめる:
  匿名の public 応答が MISS→HIT になる / s-maxage が TTL に効く / Cookie・Authorization 付きは読まず保存せず
  private・no-store・Set-Cookie 付き・POST は保存しない / 上流が誤って public を付けても会員の本文が匿名に出ない
  合言葉つき(X-Keiba-Refresh)の要求だけ、読まずに取り直して保存し直す（合言葉なしは効かない・上流には渡さない）
  取り直しのときだけ上流に X-Keiba-Fresh: 1 が付く / stg（別ポート）は本番が保存した分を読むが、自分の取得は保存しない
  Vary で保存先を分けない: 圧縮なし（事前ロード）で保存した物を gzip の利用者が HIT で読み、その逆は展開して返す / 上流へは常に gzip で頼む
  stale-while-revalidate の秒数の内は古いものを即座に返して裏で取り直し、秒数を過ぎた物は古いものを返さずに取り直す（古さの上限）
  swr の内の同時の要求は上流 1 本・swr の外は同時の本数だけ上流へ行く（群がりの記録）/ stg は swr の裏の取り直しでも保存しない
  Cookie も Authorization も無い会員だけの API は nginx が 401 を返す（上流へ行かない）/ 127.0.0.1 は API の別枠（120 回/分で止まらない）
使い方: python3 scripts/ops/vps/verify_api_cache.py   （nginx が PATH に必要。VPS 上で実行する）
"""
import gzip, os, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "templates")
hits: dict[str, int] = {}
fresh_seen: list[str] = []      # 上流に届いた X-Keiba-Fresh の値（届いた順）
ae_seen: list[str] = []         # 上流に届いた Accept-Encoding（届いた順）


class Up(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, body: str, headers: dict, gz: bool = False):
        hits[self.path.split("?")[0]] = hits.get(self.path.split("?")[0], 0) + 1
        b = body.encode()
        if gz and "gzip" in (self.headers.get("Accept-Encoding") or ""):     # 上流（GZipMiddleware）と同じく、頼まれたときだけ圧縮
            b = gzip.compress(b); headers = {**headers, "Content-Encoding": "gzip"}
        self.send_response(200)
        for k, v in headers.items(): self.send_header(k, v)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        if self.command != "HEAD": self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if self.headers.get("X-Keiba-Refresh") is not None: hits["leaked-refresh-header"] = hits.get("leaked-refresh-header", 0) + 1
        fresh_seen.append(self.headers.get("X-Keiba-Fresh") or "")
        ae_seen.append(self.headers.get("Accept-Encoding") or "")
        who = "member" if (self.headers.get("Cookie") or self.headers.get("Authorization")) else "anon"
        pub = {"Cache-Control": "public, max-age=300, s-maxage=60", "Vary": "Accept-Encoding, Cookie"}
        if p == "/api/pub": self._send(f"pub-{who}", pub)
        elif p == "/api/sm": self._send(f"sm-{who}", {**pub, "Cache-Control": "public, max-age=300, s-maxage=2"})
        elif p == "/api/ver": self._send(f"ver-{hits.get('/api/ver', 0) + 1}", pub)         # 呼ぶたびに本文が変わる（上書きの確認用）
        elif p == "/api/long": self._send(f"long-{who}", {**pub, "Cache-Control": "public, max-age=60, s-maxage=604800"})
        elif p == "/api/leak": self._send(f"leak-{who}", pub)                      # 上流の取りこぼし想定: 会員にも public を付ける
        elif p == "/api/private": self._send(f"private-{who}", {"Cache-Control": "private, max-age=60", "Vary": "Cookie"})
        elif p == "/api/nostore": self._send(f"nostore-{who}", {"Cache-Control": "no-store, max-age=0"})
        elif p == "/api/setcookie": self._send(f"setcookie-{who}", {**pub, "Set-Cookie": "a=b"})
        elif p == "/api/nohdr": self._send(f"nohdr-{who}", {})
        elif p == "/api/stgonly": self._send(f"stgonly-{who}", pub)
        elif p == "/api/gz": self._send(f"gz-{who}-{hits.get('/api/gz', 0) + 1}-" + "x" * 2000, pub, gz=True)     # 圧縮される大きさ・呼ぶたびに変わる
        elif p == "/api/stale": self._send(f"stale-{hits.get('/api/stale', 0) + 1}", {**pub, "Cache-Control": "public, max-age=60, s-maxage=2"}, gz=True)
        elif p.startswith("/api/swr"):
            if "slow" in p: time.sleep(2.0)                   # Cloud Run の起動し直し相当
            self._send(f"swr-{hits.get(p, 0) + 1}-" + "x" * 2000, {**pub, "Cache-Control": "public, max-age=2, s-maxage=2, stale-while-revalidate=4"}, gz=True)
        elif p.startswith("/api/herd"):
            time.sleep(1.0)
            cc = "public, max-age=2, s-maxage=2, stale-while-revalidate=30" if p.endswith("in") else "public, max-age=1, s-maxage=1, stale-while-revalidate=1"
            self._send(f"herd-{hits.get(p, 0) + 1}", {**pub, "Cache-Control": cc})
        elif p == "/api/maintenance/status": self._send(f"mnt-{hits.get(p, 0) + 1}", {**pub, "Cache-Control": "public, max-age=0, s-maxage=120"})
        elif p.startswith("/api/v1/betting/") or p.startswith("/api/race/1/result") or p.startswith("/api/growth-curve/"):
            self._send(f"member-path-{who}", {"Cache-Control": "private, max-age=0"})
        else: self._send("other", {})

    do_HEAD = do_GET

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send("post", {"Cache-Control": "public, max-age=60, s-maxage=60"})


def req_raw(port, path, headers=None):
    """本文を展開せずに返す: (本文のバイト列, Content-Encoding, X-Cache-Status)。"""
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    with urllib.request.urlopen(r, timeout=10) as resp:
        return resp.read(), resp.headers.get("Content-Encoding") or "", resp.headers.get("X-Cache-Status", "-")


def text_of(raw: bytes, enc: str) -> str:
    return (gzip.decompress(raw) if enc == "gzip" else raw).decode()


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def req(port, path, method="GET", headers=None):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, headers=headers or {}, data=b"x" if method == "POST" else None)
    with urllib.request.urlopen(r, timeout=10) as resp:
        return resp.read().decode(), resp.headers.get("X-Cache-Status", "-")


def main() -> int:
    nginx = shutil.which("nginx") or "/usr/sbin/nginx"
    if not os.path.exists(nginx):
        print("nginx が見つかりません"); return 2
    tmp = tempfile.mkdtemp(prefix="apicache_")
    up_port, ng_port, stg_port = free_port(), free_port(), free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", up_port), Up)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    zones = open(os.path.join(TPL, "nginx-zones.conf")).read().replace("/var/cache/nginx/keiba_api", f"{tmp}/cache")
    zones = "\n".join(l for l in zones.splitlines() if "keiba_img" not in l)
    token = "tok-" + os.urandom(8).hex()
    with open(f"{tmp}/refresh.conf", "w") as f: f.write(f'"{token}" 1;\n')
    zones = zones.replace("/etc/nginx/keiba-refresh.conf", f"{tmp}/refresh.conf")
    with open(f"{tmp}/readonly.conf", "w") as f: f.write(f'"{stg_port}" 1;\n')      # stg のポートは読むだけ（vps_09 が書くのと同じ形）
    zones = zones.replace("/etc/nginx/keiba-cache-readonly.conf", f"{tmp}/readonly.conf")
    gate = open(os.path.join(TPL, "nginx-api-gate.conf")).read()
    loc = f"""location /api/ {{{{
            limit_req zone=keiba_api_ext burst=60 nodelay;
            limit_req zone=keiba_api_lo burst=300 nodelay;
            proxy_pass http://127.0.0.1:{up_port};
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            include {TPL}/nginx-api-cache.conf;
        }}}}"""
    conf = f"""pid {tmp}/nginx.pid;
error_log {tmp}/error.log;
worker_processes 1;
events {{ worker_connections 64; }}
http {{
    access_log off;
    client_body_temp_path {tmp}/b; proxy_temp_path {tmp}/p; fastcgi_temp_path {tmp}/f; uwsgi_temp_path {tmp}/u; scgi_temp_path {tmp}/s;
{zones}
{gate}
    server {{
        listen 127.0.0.1:{ng_port};
        set_real_ip_from 127.0.0.1;
        real_ip_header CF-Connecting-IP;
        {loc.format()}
    }}
    server {{
        listen 127.0.0.1:{stg_port};
        {loc.format()}
    }}
}}
"""
    open(f"{tmp}/nginx.conf", "w").write(conf)
    t = subprocess.run([nginx, "-t", "-c", f"{tmp}/nginx.conf", "-p", tmp], capture_output=True, text=True)
    if t.returncode != 0:
        print(t.stderr); return 1
    subprocess.run([nginx, "-c", f"{tmp}/nginx.conf", "-p", tmp], check=True)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))
        ok = ok and cond

    try:
        time.sleep(0.3)
        b1, s1 = req(ng_port, "/api/pub"); b2, s2 = req(ng_port, "/api/pub")
        check("匿名 public: MISS→HIT で上流は1回", (s1, s2) == ("MISS", "HIT") and hits["/api/pub"] == 1 and b1 == b2 == "pub-anon", f"{s1},{s2},{hits.get('/api/pub')}")
        n = hits["/api/pub"]
        b3, s3 = req(ng_port, "/api/pub", headers={"Cookie": "keiba_dev_session=abc"})
        check("Cookie 付きは BYPASS で上流へ", s3 == "BYPASS" and b3 == "pub-member" and hits["/api/pub"] == n + 1, f"{s3},{b3}")
        b4, s4 = req(ng_port, "/api/pub", headers={"Authorization": "Bearer x"})
        check("Authorization 付きは BYPASS", s4 == "BYPASS" and b4 == "pub-member", f"{s4},{b4}")
        b5, s5 = req(ng_port, "/api/pub")
        check("会員要求のあとも匿名は匿名の本文(HIT)", b5 == "pub-anon" and s5 == "HIT", f"{b5},{s5}")
        b6, s6 = req(ng_port, "/api/leak", headers={"Cookie": "keiba_dev_session=abc"})
        b7, s7 = req(ng_port, "/api/leak")
        check("上流が会員に public を付けても保存されず、匿名に会員の本文が出ない", b6 == "leak-member" and b7 == "leak-anon" and s7 == "MISS", f"{b6},{b7},{s7}")
        b8, s8 = req(ng_port, "/api/leak")
        check("その後の匿名は HIT", s8 == "HIT" and b8 == "leak-anon", f"{b8},{s8}")
        for p in ("private", "nostore", "setcookie", "nohdr"):
            a = req(ng_port, f"/api/{p}"); b = req(ng_port, f"/api/{p}")
            check(f"{p} は保存しない", a[1] != "HIT" and b[1] != "HIT" and hits[f"/api/{p}"] == 2, f"{a[1]},{b[1]},{hits.get(f'/api/{p}')}")
        n = hits.get("/api/pub", 0)
        req(ng_port, "/api/pub", method="POST"); req(ng_port, "/api/pub", method="POST")
        check("POST は保存しない", hits.get("/api/pub", 0) == n + 2)
        req(ng_port, "/api/sm"); _, s = req(ng_port, "/api/sm")
        time.sleep(3)
        _, s_after = req(ng_port, "/api/sm")
        check("s-maxage が TTL に効く(max-age=300 でなく 2 秒で失効)", s == "HIT" and s_after in ("EXPIRED", "MISS"), f"{s},{s_after}")
        req(ng_port, "/api/pub?x=1"); _, sq = req(ng_port, "/api/pub?x=1"); _, sq2 = req(ng_port, "/api/pub?x=2")
        check("クエリ違いは別キー", sq == "HIT" and sq2 == "MISS", f"{sq},{sq2}")
        # Vary で保存先を分けない（事前ロード = 圧縮なし、利用者 = gzip で同じ保存物）
        GZ = {"Accept-Encoding": "gzip, deflate, br"}
        k = len(ae_seen)
        r1, e1, c1 = req_raw(ng_port, "/api/gz")                      # 事前ロードと同じ圧縮なしの要求で保存
        r2, e2, c2 = req_raw(ng_port, "/api/gz", headers=GZ)          # 利用者（gzip）
        check("圧縮なしで保存した物を gzip の利用者が HIT で読む", c1 == "MISS" and c2 == "HIT" and e2 == "gzip" and text_of(r2, e2) == text_of(r1, e1) and hits["/api/gz"] == 1, f"{c1},{c2},{e2},{hits.get('/api/gz')}")
        check("圧縮なしの相手には展開して返す", e1 == "" and text_of(r1, e1).startswith("gz-anon-1-"), f"{e1}")
        check("上流へは常に Accept-Encoding: gzip", ae_seen[k:] == ["gzip"], str(ae_seen[k:]))
        r3, e3, c3 = req_raw(ng_port, "/api/gz", headers={"X-Keiba-Refresh": token})       # 事前ロードの取り直し（圧縮なし）
        r4, e4, c4 = req_raw(ng_port, "/api/gz", headers=GZ)
        r5, e5, c5 = req_raw(ng_port, "/api/gz")
        check("事前ロードの取り直しが gzip の利用者にも圧縮なしの相手にも届く", c3 == "BYPASS" and text_of(r3, e3).startswith("gz-anon-2-")
              and c4 == "HIT" and text_of(r4, e4).startswith("gz-anon-2-") and c5 == "HIT" and text_of(r5, e5).startswith("gz-anon-2-"), f"{c3},{c4},{c5}")
        r6, e6, c6 = req_raw(ng_port, "/api/gz", headers={**GZ, "Cookie": "keiba_dev_session=abc"})
        r7, e7, c7 = req_raw(ng_port, "/api/gz", headers=GZ)
        check("Cookie 付きは BYPASS のまま・匿名の保存物は変わらない", c6 == "BYPASS" and text_of(r6, e6).startswith("gz-member-") and c7 == "HIT" and text_of(r7, e7).startswith("gz-anon-2-"), f"{c6},{c7}")
        # swr の無い物は、期限切れの後の最初の要求が古いもの（STALE）を返さずに取り直す（use_stale に updating が無いこと）
        req_raw(ng_port, "/api/stale", headers=GZ); time.sleep(3)
        ra, ea, ca = req_raw(ng_port, "/api/stale", headers=GZ)
        rb, eb, cb = req_raw(ng_port, "/api/stale", headers=GZ)
        check("swr の無い物は期限切れの後の最初の要求で取り直す（STALE で古いものを返さない）", ca == "EXPIRED" and text_of(ra, ea) == "stale-2"
              and cb == "HIT" and text_of(rb, eb) == "stale-2", f"{ca},{text_of(ra, ea)},{cb},{text_of(rb, eb)}")
        # stale-while-revalidate: 秒数の内は即座に古いものを返し、裏で取り直す。同じ接続の次の要求を待たせない。秒数を過ぎたら取り直す
        import http.client
        req_raw(ng_port, "/api/swr/slow", headers=GZ); time.sleep(3.5)             # s-maxage=2 を過ぎ、swr=4 の内
        hc = http.client.HTTPConnection("127.0.0.1", ng_port, timeout=10)
        t0 = time.monotonic(); hc.request("GET", "/api/swr/slow", headers=GZ); r0 = hc.getresponse(); b0 = r0.read(); d1 = time.monotonic() - t0
        t0 = time.monotonic(); hc.request("GET", "/api/swr/fast"); r1 = hc.getresponse(); r1.read(); d2 = time.monotonic() - t0
        check("swr の内は STALE で古い本文を即座に返す（gzip で保存した物）", r0.getheader("X-Cache-Status") == "STALE" and text_of(b0, r0.getheader("Content-Encoding") or "").startswith("swr-1-")
              and d1 < 0.5, f"{r0.getheader('X-Cache-Status')},{d1:.2f}")
        check("裏の取り直しを起こした接続でも、続く要求は待たない（keepalive_timeout 0）", d2 < 0.5 and (r0.getheader("Connection") or "").lower() == "close", f"{d2:.2f},{r0.getheader('Connection')}")
        rc_, ec_, cc_ = req_raw(ng_port, "/api/swr/slow", headers={"Cookie": "keiba_dev_session=abc"})
        rf_, ef_, cf_ = req_raw(ng_port, "/api/swr/slow", headers={"X-Keiba-Refresh": token})
        check("swr の内でも Cookie 付き・取り直しは BYPASS（古いものを返さない）", cc_ == "BYPASS" and cf_ == "BYPASS", f"{cc_},{cf_}")
        # 取り直し（BYPASS）は待って保存し終えている。保存から s-maxage（2 秒）の内に読むと、その本文が HIT（圧縮なしの相手には展開して返す）
        rt, et, ct = req_raw(ng_port, "/api/swr/slow")
        check("取り直した本文が保存され、次は HIT・展開して返す", ct == "HIT" and text_of(rt, et) == text_of(rf_, ef_) and et == "", f"{ct},{text_of(rt, et)[:8]},{text_of(rf_, ef_)[:8]},{et}")
        time.sleep(10.0)                  # 最後の保存（裏の取り直しが残っていればその保存も）から s-maxage＋swr（6 秒）を十分に過ぎる。間に要求を出さない
        n0 = hits.get("/api/swr/slow", 0)
        t0 = time.monotonic(); ru, eu, cu = req_raw(ng_port, "/api/swr/slow"); d3 = time.monotonic() - t0
        check("s-maxage＋swr を過ぎた物は古いものを返さず取り直す（古さの上限）", cu == "EXPIRED" and text_of(ru, eu).startswith(f"swr-{n0 + 1}-") and d3 > 1.5, f"{cu},{text_of(ru, eu)[:8]},{d3:.2f}")
        # 群がりの記録: 期限切れの鍵に同時に 10 本。swr の内は上流 1 本、swr の外は 10 本（docs の費用の見積もりの根拠）
        def burst(path):
            out: list = []
            ths = [threading.Thread(target=lambda: out.append(req(ng_port, path)[1])) for _ in range(10)]
            b = hits.get(path, 0); [x.start() for x in ths]; [x.join() for x in ths]; time.sleep(1.5)
            return hits.get(path, 0) - b, sorted(set(out))
        burst("/api/herd/in"); burst("/api/herd/out"); time.sleep(3.5)
        n_in, st_in = burst("/api/herd/in")
        n_out, st_out = burst("/api/herd/out")
        print(f"INFO 群がり: swr の内 上流 {n_in} 本 {st_in} / swr の外 上流 {n_out} 本 {st_out}")
        check("swr の内の同時 10 本は上流 1 本（裏の取り直しだけ）", n_in == 1, f"{n_in},{st_in}")
        # stg（読むだけ）が swr の内で裏の取り直しを起こしても、保存しない（検証中のコードの応答を本番の保存物に入れない）
        req(ng_port, "/api/swr/stg"); time.sleep(3.5)
        _, s_stg = req(stg_port, "/api/swr/stg"); time.sleep(1.5)       # まだ swr の内（保存から約 5 秒）
        b_prod, s_prod = req(ng_port, "/api/swr/stg")
        check("stg が swr の内で起こした裏の取り直しは保存されない（本番の保存物は古いまま STALE）", s_stg == "STALE" and s_prod == "STALE" and b_prod.startswith("swr-1-"), f"{s_stg},{s_prod},{b_prod[:8]}")
        # 会員だけの API: Cookie も Authorization も無ければ nginx が 401（上流へ行かない）。Cookie 付きは上流へ
        n0 = sum(v for k, v in hits.items() if k.startswith(("/api/v1/betting/", "/api/race/1/result", "/api/growth-curve/")))
        try:
            req(ng_port, "/api/v1/betting/race-values?date=20261123"); g = None
        except urllib.error.HTTPError as e:
            g = (e.code, e.headers.get("X-Keiba-Gate"), e.headers.get("Content-Type"), e.read().decode())
        import json as _json
        body_ok = g is not None and _json.loads(g[3]) == {"error": "この情報は会員限定です。ログインしてください。", "login_url": "/login", "reason": "member_required"}
        n1 = sum(v for k, v in hits.items() if k.startswith(("/api/v1/betting/", "/api/race/1/result", "/api/growth-curve/")))
        check("匿名の会員だけの API は nginx が 401（本文は API と同じ・上流へ行かない）", g is not None and g[0] == 401 and g[1] == "member-401" and body_ok
              and (g[2] or "").startswith("application/json") and n1 == n0, f"{g},{n1 - n0}")
        bm, _ = req(ng_port, "/api/v1/betting/race-values?date=20261123", headers={"Cookie": "keiba_dev_session=abc"})
        check("Cookie 付きは会員だけの API も上流へ（判定は API）", bm == "member-path-member", bm)
        try:
            req(ng_port, "/api/race/1/result"); r401 = False
        except urllib.error.HTTPError as e:
            r401 = e.code == 401
        b_rs, _ = req(ng_port, "/api/race/1/result-status")
        b_gs, _ = req(ng_port, "/api/growth-curve/status")
        check("一覧の形どおり（result は 401・result-status と growth-curve/status は上流へ）", r401 and b_rs == "member-path-anon" and b_gs == "member-path-anon", f"{r401},{b_rs},{b_gs}")
        # 127.0.0.1 は別枠: 150 本続けても 429 にならない（旧の 120 回/分・burst 60 なら 61 本目から 429）
        codes = []
        for _ in range(150):
            try:
                req(ng_port, "/api/pub"); codes.append(200)
            except urllib.error.HTTPError as e:
                codes.append(e.code)
        check("127.0.0.1 は API の別枠（150 本続けても 429 にならない）", 429 not in codes, str(codes.count(429)))
        # 外の要求（CF-Connecting-IP が実 IP になる）は実 IP ごとの 120 回/分・burst 60 の枠
        codes = []
        for _ in range(75):
            try:
                req(ng_port, "/api/pub", headers={"CF-Connecting-IP": "203.0.113.1"}); codes.append(200)
            except urllib.error.HTTPError as e:
                codes.append(e.code)
        check("外の要求は実 IP ごとの枠（burst を越えると 429）", 429 not in codes[:60] and codes.count(429) >= 5, f"{codes.count(429)}")
        b_other, _ = req(ng_port, "/api/pub", headers={"CF-Connecting-IP": "203.0.113.2"})
        check("別の実 IP は別の枠（429 にならない）", b_other == "pub-anon", b_other)
        # stg（読むだけ）の保守の状態は、本番の保存物を読まない（Cloud Run ごとの臨時メンテを stg で見られる）
        _, m1 = req(ng_port, "/api/maintenance/status"); _, m2 = req(ng_port, "/api/maintenance/status")
        bs1, ms1 = req(stg_port, "/api/maintenance/status")
        _, b_ph = req(stg_port, "/api/pub")
        check("stg の保守の状態は本番の保存物を読まない（ほかのパスは今までどおり共有）", (m1, m2) == ("MISS", "HIT") and ms1 == "MISS" and b_ph == "HIT", f"{m1},{m2},{ms1},{b_ph}")
        # 事前ロードの上書き取得
        b, s_ = req(ng_port, "/api/ver"); b2, s2_ = req(ng_port, "/api/ver")
        check("上書きの前: 同じ本文が HIT", b == b2 == "ver-1" and s2_ == "HIT", f"{b},{b2},{s2_}")
        bw, sw = req(ng_port, "/api/ver", headers={"X-Keiba-Refresh": "wrong"})
        check("合言葉が違えば通常どおり HIT（上流へ届かない）", sw == "HIT" and bw == "ver-1", f"{sw},{bw}")
        br, sr = req(ng_port, "/api/ver", headers={"X-Keiba-Refresh": token})
        check("合言葉つきは読まずに上流から取り直す", sr == "BYPASS" and br == "ver-2", f"{sr},{br}")
        ba, sa_ = req(ng_port, "/api/ver")
        check("取り直した本文が保存されて匿名にも HIT で出る", ba == "ver-2" and sa_ == "HIT", f"{ba},{sa_}")
        n = hits.get("/api/ver", 0)
        bm, sm_ = req(ng_port, "/api/ver", headers={"X-Keiba-Refresh": token, "Cookie": "keiba_dev_session=abc"})
        ba2, _ = req(ng_port, "/api/ver")
        check("合言葉があっても Cookie 付きは保存しない（匿名の保存分は ver-2 のまま）", bm == "ver-3" and ba2 == "ver-2" and hits.get("/api/ver") == n + 1, f"{bm},{ba2},{hits.get('/api/ver')}")
        check("合言葉の見出しは上流へ渡さない", hits.get("leaked-refresh-header", 0) == 0, str(hits.get("leaked-refresh-header")))
        # 取り直しのときだけ、API 内の使い回しも飛ばさせる見出し（X-Keiba-Fresh: 1）が上流に届く
        k = len(fresh_seen)
        req(ng_port, "/api/ver", headers={"X-Keiba-Refresh": token})
        req(ng_port, "/api/nohdr")
        check("X-Keiba-Fresh は取り直しのときだけ 1", fresh_seen[k:] == ["1", "0"], str(fresh_seen[k:]))
        # stg と本番の共有: stg は本番が保存した分を読むが、自分の取得は保存しない
        bs, ss = req(stg_port, "/api/pub")
        check("stg は本番が保存した分を HIT で読む", ss == "HIT" and bs == "pub-anon", f"{ss},{bs}")
        _, s1s = req(stg_port, "/api/stgonly"); _, s2s = req(stg_port, "/api/stgonly"); _, sps = req(ng_port, "/api/stgonly")
        check("stg の取得は保存しない（stg も本番も MISS のまま）", (s1s, s2s, sps) == ("MISS", "MISS", "MISS"), f"{s1s},{s2s},{sps}")
        _, sp2 = req(ng_port, "/api/stgonly"); _, sst = req(stg_port, "/api/stgonly")
        check("本番が保存したあとは stg も HIT", sp2 == "HIT" and sst == "HIT", f"{sp2},{sst}")
        # 確定済みレース用: 1か月の s-maxage は長く保持され、--match の消去は対象だけに効く
        req(ng_port, "/api/long?a=1"); req(ng_port, "/api/long?b=2")
        time.sleep(3)
        _, sl = req(ng_port, "/api/long?a=1")
        check("s-maxage=1か月は数秒後も HIT（長く保持）", sl == "HIT", sl)
        purge = os.path.join(HERE, "purge_api_cache.sh")
        shim = os.path.join(tmp, "bin"); os.makedirs(shim, exist_ok=True)
        with open(os.path.join(shim, "nginx"), "w") as f: f.write("#!/bin/sh\nexit 0\n")      # reload は本番に触れないよう無効化
        os.chmod(os.path.join(shim, "nginx"), 0o755)
        env = {**os.environ, "PATH": shim + os.pathsep + os.environ["PATH"]}
        r = subprocess.run(["bash", purge, "--match", "api/long\\?a=1", f"{tmp}/cache"], capture_output=True, text=True, env=env)
        _, sa = req(ng_port, "/api/long?a=1"); _, sb = req(ng_port, "/api/long?b=2")
        check("--match は条件に合うキャッシュだけ消す", r.returncode == 0 and sa == "MISS" and sb == "HIT", f"{r.returncode},{sa},{sb},{r.stderr.strip()}")
        r = subprocess.run(["bash", purge, f"{tmp}/cache"], capture_output=True, text=True, env=env)
        _, sc = req(ng_port, "/api/long?b=2")
        check("引数なしは全部消す", r.returncode == 0 and sc == "MISS", f"{r.returncode},{sc}")
    finally:
        subprocess.run([nginx, "-c", f"{tmp}/nginx.conf", "-p", tmp, "-s", "quit"], capture_output=True)
        srv.shutdown()
        time.sleep(0.5)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
