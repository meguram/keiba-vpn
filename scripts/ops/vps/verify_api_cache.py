#!/usr/bin/env python3
"""API キャッシュ設定(nginx-api-cache.conf + nginx-zones.conf)を、root なし・本番に触れずに検証する。

自前の nginx を一時ディレクトリで高ポートに立て、疑似の上流に対して次を確かめる:
  匿名の public 応答が MISS→HIT になる / s-maxage が TTL に効く / Cookie・Authorization 付きは読まず保存せず
  private・no-store・Set-Cookie 付き・POST は保存しない / 上流が誤って public を付けても会員の本文が匿名に出ない
  合言葉つき(X-Keiba-Refresh)の要求だけ、読まずに取り直して保存し直す（合言葉なしは効かない・上流には渡さない）
  取り直しのときだけ上流に X-Keiba-Fresh: 1 が付く / stg（別ポート）は本番が保存した分を読むが、自分の取得は保存しない
  Vary で保存先を分けない: 圧縮なし（事前ロード）で保存した物を gzip の利用者が HIT で読み、その逆は展開して返す / 上流へは常に gzip で頼む
  期限切れの後の最初の要求は古いものを返さず、上流から取り直す（EXPIRED。background_update を入れていない）
使い方: python3 scripts/ops/vps/verify_api_cache.py   （nginx が PATH に必要。VPS 上で実行する）
"""
import gzip, os, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request
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
    loc = f"""location /api/ {{{{
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
    server {{
        listen 127.0.0.1:{ng_port};
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
        # 期限切れの後の最初の要求は、古いもの（STALE）を返さずに取り直す（background_update が無いこと）
        req_raw(ng_port, "/api/stale", headers=GZ); time.sleep(3)
        ra, ea, ca = req_raw(ng_port, "/api/stale", headers=GZ)
        rb, eb, cb = req_raw(ng_port, "/api/stale", headers=GZ)
        check("期限切れの後の最初の要求は取り直した新しい本文（STALE で古いものを返さない）", ca == "EXPIRED" and text_of(ra, ea) == "stale-2"
              and cb == "HIT" and text_of(rb, eb) == "stale-2", f"{ca},{text_of(ra, ea)},{cb},{text_of(rb, eb)}")
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
