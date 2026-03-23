"""
netkeiba.com / smartrc.jp 用 HTTPクライアント

- .env からの認証情報読み込み + ログインセッション管理
- 自動リトライ (指数バックオフ)
- レートリミット (robots.txt 遵守)
- EUC-JP 自動判定
- レスポンスキャッシュ (同一セッション内)
- smartrc.jp アクセス対応
- スロットリング機構 (429/503 即時停止 + 指数バックオフリトライ)

Anti-Bot 対策:
- User-Agent ローテーション (Chrome/Firefox/Edge × Windows/Mac)
- リアルなブラウザヘッダーセット (sec-ch-ua, sec-fetch-*, etc.)
- Referer チェーン (遷移元を自動付与)
- バースト制限 (短期間の連続アクセス防止)
- セッションあたりのクールダウン (一定リクエスト数ごとに長めの休止)
- セッションリフレッシュ (定期的に TLS/Cookie を再構築)

環境変数で間隔を広げられる（ブロック回避時）:
NETKEIBA_BURST_WINDOW, NETKEIBA_BURST_COOLDOWN_MIN/MAX,
NETKEIBA_SESSION_COOLDOWN_INTERVAL, NETKEIBA_SESSION_COOLDOWN_MIN/MAX,
NETKEIBA_SESSION_REFRESH_INTERVAL, NETKEIBA_THROTTLE_MIN/MAX
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("scraper.client")

_ENCODING_MAP = {
    "db.netkeiba.com": "EUC-JP",
    "race.netkeiba.com": "EUC-JP",
    "regist.netkeiba.com": "EUC-JP",
}

# ---------------------------------------------------------------------------
# User-Agent プール
# ---------------------------------------------------------------------------
_UA_POOL: list[dict[str, str]] = [
    {
        "ua": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua": '"Chromium";v="131", "Google Chrome";v="131", "Not?A_Brand";v="99"',
        "platform": '"Windows"',
    },
    {
        "ua": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua": '"Chromium";v="131", "Google Chrome";v="131", "Not?A_Brand";v="99"',
        "platform": '"macOS"',
    },
    {
        "ua": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
            "Gecko/20100101 Firefox/133.0"
        ),
        "sec_ch_ua": "",
        "platform": "",
    },
    {
        "ua": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/18.2 Safari/605.1.15"
        ),
        "sec_ch_ua": "",
        "platform": "",
    },
    {
        "ua": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0"
        ),
        "sec_ch_ua": '"Chromium";v="131", "Microsoft Edge";v="131", "Not?A_Brand";v="99"',
        "platform": '"Windows"',
    },
    {
        "ua": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        "platform": '"Windows"',
    },
    {
        "ua": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        "platform": '"macOS"',
    },
    {
        "ua": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "sec_ch_ua": '"Chromium";v="131", "Google Chrome";v="131", "Not?A_Brand";v="99"',
        "platform": '"Linux"',
    },
]

# ---------------------------------------------------------------------------
# Referer マッピング
# ---------------------------------------------------------------------------
_REFERER_MAP: dict[str, str] = {
    "db.netkeiba.com": "https://db.netkeiba.com/",
    "race.netkeiba.com": "https://race.netkeiba.com/",
    "regist.netkeiba.com": "https://regist.netkeiba.com/",
    "www.smartrc.jp": "https://www.smartrc.jp/",
}

_LOGIN_URL = "https://regist.netkeiba.com/account/"

SMARTRC_BASE_URL = "https://www.smartrc.jp/"

# ---------------------------------------------------------------------------
# スロットリング設定
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_THROTTLE_CONFIG: dict[str, dict[str, float]] = {
    "netkeiba": {
        "min_interval": _env_float("NETKEIBA_THROTTLE_MIN", 3.0),
        "max_interval": _env_float("NETKEIBA_THROTTLE_MAX", 6.0),
    },
    "smartrc": {"min_interval": 3.0, "max_interval": 6.0},
}

_SMARTRC_REQUEST_LIMIT = 200

_BACKOFF_INITIAL = 5.0
_BACKOFF_FACTOR = 2.5
_BACKOFF_MAX_RETRIES = 3

# バースト制限: 連続 N リクエスト後に追加休止（.env / 環境変数で上書き可）
_BURST_WINDOW = _env_int("NETKEIBA_BURST_WINDOW", 10)
_BURST_COOLDOWN_MIN = _env_float("NETKEIBA_BURST_COOLDOWN_MIN", 8.0)
_BURST_COOLDOWN_MAX = _env_float("NETKEIBA_BURST_COOLDOWN_MAX", 15.0)

# セッションクールダウン: N リクエストごとに長い休止
_SESSION_COOLDOWN_INTERVAL = _env_int("NETKEIBA_SESSION_COOLDOWN_INTERVAL", 50)
_SESSION_COOLDOWN_MIN = _env_float("NETKEIBA_SESSION_COOLDOWN_MIN", 30.0)
_SESSION_COOLDOWN_MAX = _env_float("NETKEIBA_SESSION_COOLDOWN_MAX", 60.0)

# セッションリフレッシュ: N リクエストごとに新しいセッション（TLS/接続の張り直し）
_SESSION_REFRESH_INTERVAL = _env_int("NETKEIBA_SESSION_REFRESH_INTERVAL", 150)


def _load_env(env_path: str | None = None) -> dict[str, str]:
    """
    .env ファイルから netkeiba_id / netkeiba_pw を読み込む。
    python-dotenv が入っていればそれを使い、なければ手動パースする。
    """
    if env_path is None:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]
        for c in candidates:
            if c.exists():
                env_path = str(c)
                break

    env_vars: dict[str, str] = {}

    if env_path and Path(env_path).exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                env_vars[key.strip()] = val.strip().strip("'\"")

    for k, v in env_vars.items():
        if k not in os.environ:
            os.environ[k] = v

    return env_vars


def _source_from_url(url: str) -> str:
    """URLからソース名を判定する。"""
    if "smartrc.jp" in url:
        return "smartrc"
    return "netkeiba"


def _build_browser_headers(ua_entry: dict[str, str]) -> dict[str, str]:
    """選択された UA エントリからリアルなブラウザヘッダーセットを構築する。"""
    headers: dict[str, str] = {
        "User-Agent": ua_entry["ua"],
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    if ua_entry.get("sec_ch_ua"):
        headers["Sec-Ch-Ua"] = ua_entry["sec_ch_ua"]
        headers["Sec-Ch-Ua-Mobile"] = "?0"
        headers["Sec-Ch-Ua-Platform"] = ua_entry.get("platform", '"Windows"')
        headers["Sec-Fetch-Dest"] = "document"
        headers["Sec-Fetch-Mode"] = "navigate"
        headers["Sec-Fetch-Site"] = "same-origin"
        headers["Sec-Fetch-User"] = "?1"

    return headers


class NetkeibaClient:
    """
    netkeiba.com / smartrc.jp 専用 HTTPクライアント。

    Parameters:
        source: アクセス対象 ('netkeiba' または 'smartrc')
        interval: リクエスト間隔 (秒) ※後方互換のため残存、throttle設定が優先
        max_retries: 最大リトライ回数
        timeout: タイムアウト (秒)
        cache_dir: HTMLキャッシュディレクトリ (Noneでキャッシュ無効)
        env_path: .env ファイルパス (Noneで自動検索)
        auto_login: Trueなら初回リクエスト前に自動ログイン
    """

    def __init__(
        self,
        source: str = "netkeiba",
        interval: float = 2.0,
        max_retries: int = 3,
        timeout: int = 20,
        cache_dir: str | None = None,
        env_path: str | None = None,
        auto_login: bool = True,
    ):
        if source not in _THROTTLE_CONFIG:
            raise ValueError(f"source は 'netkeiba' または 'smartrc' を指定してください。got: {source!r}")

        self.source = source
        self.interval = interval
        self.timeout = timeout
        self.cache_dir = cache_dir
        self._max_retries = max_retries
        self._last_request_time: float = 0
        self._logged_in = False
        self._env_path = env_path

        self._request_count: int = 0
        self._burst_counter: int = 0
        self._last_url: str = ""
        self._client_lock = threading.Lock()

        _load_env(env_path)

        self._current_ua = random.choice(_UA_POOL)
        self._session = self._create_session()

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if auto_login:
            self.login()

    # ------------------------------------------------------------------
    # セッション構築
    # ------------------------------------------------------------------

    def _create_session(self) -> requests.Session:
        """新しいセッションを構築してブラウザヘッダーを設定する。"""
        session = requests.Session()
        session.headers.update(_build_browser_headers(self._current_ua))

        retry = Retry(
            total=self._max_retries,
            backoff_factor=1.5,
            status_forcelist=[500, 502, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=5)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _refresh_session(self) -> None:
        """セッションを破棄して新しいものに切り替える。UA もローテーションする。
        認証クッキー (netkeiba, nkauth) は login() で再設定されるためコピーしない。"""
        _AUTH_COOKIES = {"netkeiba", "nkauth"}
        old_cookies = {k: v for k, v in self._session.cookies.items()
                       if k not in _AUTH_COOKIES}
        self._session.close()

        self._current_ua = random.choice(_UA_POOL)
        self._session = self._create_session()

        for name, value in old_cookies.items():
            self._session.cookies.set(name, value)

        self._burst_counter = 0
        logger.info(
            "セッションリフレッシュ完了 (UA: %s...)",
            self._current_ua["ua"][:50],
        )

    # ------------------------------------------------------------------
    # Referer 管理
    # ------------------------------------------------------------------

    def _get_referer(self, url: str) -> str:
        """URL に応じた適切な Referer ヘッダーを返す。"""
        if self._last_url:
            parsed_last = urlparse(self._last_url)
            parsed_new = urlparse(url)
            if parsed_last.netloc == parsed_new.netloc:
                return self._last_url

        parsed = urlparse(url)
        return _REFERER_MAP.get(parsed.netloc, f"{parsed.scheme}://{parsed.netloc}/")

    # ------------------------------------------------------------------
    # スロットリング
    # ------------------------------------------------------------------

    def _throttle(self, source: str | None = None) -> None:
        """ソースに応じたランダムスリープを挿入する。"""
        src = source if source is not None else self.source
        config = _THROTTLE_CONFIG.get(src, _THROTTLE_CONFIG["netkeiba"])
        sleep_time = random.uniform(config["min_interval"], config["max_interval"])

        jitter = random.gauss(0, sleep_time * 0.15)
        sleep_time = max(config["min_interval"], sleep_time + jitter)

        logger.debug("スロットリング: %.2f 秒待機 (source=%s)", sleep_time, src)
        time.sleep(sleep_time)

    def _rate_limit(self):
        """後方互換用: 最低限の interval を保証する。"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # バースト / クールダウン管理
    # ------------------------------------------------------------------

    def _check_burst(self) -> None:
        """バースト制限とクールダウンを適用する。"""
        self._burst_counter += 1

        if self._burst_counter % _BURST_WINDOW == 0:
            cooldown = random.uniform(_BURST_COOLDOWN_MIN, _BURST_COOLDOWN_MAX)
            logger.info(
                "バースト制限: %d リクエスト到達、%.1f 秒クールダウン",
                self._burst_counter, cooldown,
            )
            time.sleep(cooldown)

        if self._request_count > 0 and self._request_count % _SESSION_COOLDOWN_INTERVAL == 0:
            cooldown = random.uniform(_SESSION_COOLDOWN_MIN, _SESSION_COOLDOWN_MAX)
            logger.info(
                "セッションクールダウン: %d リクエスト到達、%.1f 秒休止",
                self._request_count, cooldown,
            )
            time.sleep(cooldown)

        if (self._request_count > 0
                and self._request_count % _SESSION_REFRESH_INTERVAL == 0):
            logger.info("セッションリフレッシュ: %d リクエスト到達", self._request_count)
            was_logged_in = self._logged_in
            self._refresh_session()
            if was_logged_in:
                self.login()

    # ------------------------------------------------------------------
    # リクエスト数管理
    # ------------------------------------------------------------------

    def _increment_request_count(self, url: str) -> None:
        """リクエスト数をインクリメントし、上限チェックを行う。"""
        self._request_count += 1
        src = _source_from_url(url)
        if src == "smartrc" and self._request_count >= _SMARTRC_REQUEST_LIMIT:
            logger.warning(
                "smartrc へのリクエスト数が上限 %d に達しました (現在: %d)。"
                "セッションを切り替えることを検討してください。",
                _SMARTRC_REQUEST_LIMIT,
                self._request_count,
            )

    # ------------------------------------------------------------------
    # 429/503 対応の指数バックオフリトライ
    # ------------------------------------------------------------------

    def _get_with_backoff(self, url: str) -> requests.Response:
        """
        GETリクエストを実行し、429/503 の場合は指数バックオフでリトライする。
        """
        wait = _BACKOFF_INITIAL
        last_resp: requests.Response | None = None

        for attempt in range(_BACKOFF_MAX_RETRIES + 1):
            src = _source_from_url(url)
            self._throttle(src)
            self._last_request_time = time.time()

            referer = self._get_referer(url)
            self._session.headers["Referer"] = referer

            if random.random() < 0.3:
                self._session.headers.pop("Cache-Control", None)
            else:
                self._session.headers["Cache-Control"] = "max-age=0"

            logger.info("GET %s (attempt %d/%d)", url, attempt + 1, _BACKOFF_MAX_RETRIES + 1)
            resp = self._session.get(url, timeout=self.timeout)
            last_resp = resp

            if resp.status_code in (429, 503):
                if attempt < _BACKOFF_MAX_RETRIES:
                    jitter = random.uniform(0, wait * 0.3)
                    total_wait = wait + jitter
                    logger.warning(
                        "HTTP %d 受信。%.1f 秒後にリトライします (attempt %d/%d): %s",
                        resp.status_code,
                        total_wait,
                        attempt + 1,
                        _BACKOFF_MAX_RETRIES,
                        url,
                    )
                    time.sleep(total_wait)
                    wait *= _BACKOFF_FACTOR
                    continue
                else:
                    logger.error(
                        "HTTP %d: 最大リトライ回数 %d に達しました: %s",
                        resp.status_code,
                        _BACKOFF_MAX_RETRIES,
                        url,
                    )
                    resp.raise_for_status()

            elif resp.status_code == 403:
                logger.warning(
                    "HTTP 403 (アクセス拒否) — UA ローテーション実施: %s", url,
                )
                self._current_ua = random.choice(_UA_POOL)
                self._session.headers.update(_build_browser_headers(self._current_ua))
                if attempt < _BACKOFF_MAX_RETRIES:
                    time.sleep(wait + random.uniform(5, 15))
                    wait *= _BACKOFF_FACTOR
                    continue
                resp.raise_for_status()
            else:
                resp.raise_for_status()
                self._last_url = url
                return resp

        assert last_resp is not None
        last_resp.raise_for_status()
        return last_resp

    # ------------------------------------------------------------------
    # ログイン
    # ------------------------------------------------------------------

    def login(self) -> bool:
        """
        netkeiba.com にログインする。
        認証情報は環境変数 netkeiba_id / netkeiba_pw から取得する。
        スレッドセーフ。
        """
        if self._logged_in:
            return True
        uid = os.environ.get("netkeiba_id", "")
        pw = os.environ.get("netkeiba_pw", "")

        if not uid or not pw:
            logger.warning("ログイン情報なし (.env に netkeiba_id / netkeiba_pw を設定してください)")
            return False

        self._throttle()
        logger.info("ログイン中... (%s)", uid[:6] + "***")

        self._session.headers["Referer"] = "https://regist.netkeiba.com/account/"

        try:
            resp = self._session.post(
                _LOGIN_URL,
                data={
                    "pid": "login",
                    "action": "auth",
                    "return_url2": "",
                    "mem_tp": "",
                    "login_id": uid,
                    "pswd": pw,
                },
                timeout=self.timeout,
                allow_redirects=True,
            )
            cookies = list(self._session.cookies.keys())
            if "nkauth" in cookies:
                self._logged_in = True
                logger.info("ログイン成功 (cookies: %s)", cookies)
                return True
            else:
                logger.error("ログイン失敗 (nkauth cookie なし, cookies: %s)", cookies)
                return False
        except Exception as e:
            logger.error("ログイン失敗: %s", e)
            return False

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def is_logged_in(self) -> bool:
        return self._logged_in

    @property
    def request_count(self) -> int:
        """現在のセッションで発行したリクエスト総数。"""
        return self._request_count

    # ------------------------------------------------------------------
    # エンコーディング / キャッシュ
    # ------------------------------------------------------------------

    def _detect_encoding(self, url: str) -> str:
        for domain, enc in _ENCODING_MAP.items():
            if domain in url:
                return enc
        return "UTF-8"

    def _cache_key(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        suffix = "_auth" if self._logged_in else ""
        h = hashlib.md5((url + suffix).encode()).hexdigest()
        return Path(self.cache_dir) / f"{h}.html"

    # ------------------------------------------------------------------
    # フェッチ
    # ------------------------------------------------------------------

    def fetch(self, url: str, use_cache: bool = True, encoding: str | None = None) -> str:
        """
        URLからHTMLを取得して文字列で返す。
        キャッシュがあればそれを返す。
        429/503 受信時は指数バックオフでリトライする。
        スレッドセーフ: 複数ジョブが同一クライアントを共有しても安全。

        encoding:
            指定時はレスポンスの解釈にそのエンコーディングを使う（例: top の race_list 断片は UTF-8）。
            未指定時はホストに応じた既定（race.netkeiba.com は EUC-JP など）。
        """
        cache_path = self._cache_key(url)

        if use_cache and cache_path and cache_path.exists():
            logger.debug("キャッシュヒット: %s", url)
            return cache_path.read_text(encoding="utf-8")

        with self._client_lock:
            self._increment_request_count(url)
            self._check_burst()
            resp = self._get_with_backoff(url)

        enc = encoding or self._detect_encoding(url)
        resp.encoding = enc
        html = resp.text

        if cache_path:
            cache_path.write_text(html, encoding="utf-8")

        return html

    # ------------------------------------------------------------------
    # クリーンアップ
    # ------------------------------------------------------------------

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
