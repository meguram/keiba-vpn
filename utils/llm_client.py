"""
LLM クライアント

エージェントが自律的に推論するための LLM API 抽象化レイヤー。

対応プロバイダー:
  1. anthropic_bedrock — .claude/settings.json の Bedrock プロキシ経由で Claude を呼ぶ (推奨)
  2. openai            — OpenAI API (GPT-4o-mini 等)

プロバイダー選択:
  .env に LLM_PROVIDER を指定 (省略時は自動判定: bedrock 設定があれば bedrock、なければ openai)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger("LLM")

load_dotenv()

_provider: str = ""
_available: bool = False

_openai_client: Any = None
_openai_model: str = "gpt-4o-mini"

_bedrock_base_url: str = ""
_bedrock_token: str = ""
_bedrock_model: str = "us.anthropic.claude-sonnet-4-6-20260101-v1:0"


def _load_claude_settings() -> dict:
    """~/.claude/settings.json から Bedrock プロキシ設定を読み込む。"""
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        return data.get("env", {})
    except Exception:
        return {}


def init_llm() -> bool:
    """LLM クライアントを初期化する。利用可能なら True を返す。"""
    global _provider, _available

    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()

    if explicit == "openai":
        return _init_openai()
    if explicit in ("anthropic_bedrock", "bedrock", "claude"):
        return _init_bedrock()

    if _init_bedrock():
        return True
    return _init_openai()


def _init_bedrock() -> bool:
    """Bedrock プロキシ (Claude) を初期化する。"""
    global _provider, _available, _bedrock_base_url, _bedrock_token, _bedrock_model

    base_url = os.getenv("ANTHROPIC_BEDROCK_BASE_URL", "")
    token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")

    if not base_url or not token:
        claude_env = _load_claude_settings()
        base_url = base_url or claude_env.get("ANTHROPIC_BEDROCK_BASE_URL", "")
        token = token or claude_env.get("AWS_BEARER_TOKEN_BEDROCK", "")

    if not base_url or not token:
        return False

    try:
        import httpx  # noqa: F401
    except ImportError:
        logger.warning("httpx がインストールされていません (pip install httpx)")
        return False

    _bedrock_base_url = base_url.rstrip("/")
    _bedrock_token = token
    _bedrock_model = os.getenv("BEDROCK_MODEL", _bedrock_model)
    _provider = "bedrock"
    _available = True
    logger.info("LLM 初期化完了: provider=bedrock (Claude), model=%s", _bedrock_model)
    return True


def _init_openai() -> bool:
    """OpenAI クライアントを初期化する。"""
    global _provider, _available, _openai_client, _openai_model

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("LLM API キーが未設定です。LLM 推論は無効化されます。")
        _available = False
        return False

    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
        _openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        _provider = "openai"
        _available = True
        logger.info("LLM 初期化完了: provider=openai, model=%s", _openai_model)
        return True
    except ImportError:
        logger.warning("openai パッケージがインストールされていません。")
        _available = False
        return False
    except Exception as e:
        logger.warning("OpenAI 初期化失敗: %s", e)
        _available = False
        return False


def is_available() -> bool:
    return _available


def get_provider() -> str:
    return _provider


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    raw: Any = None
    usage: dict = field(default_factory=dict)

    def as_json(self) -> dict | None:
        text = self.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


def chat(
    messages: list[LLMMessage],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    response_format: str | None = None,
) -> LLMResponse:
    """LLM にメッセージを送って応答を得る。"""
    if not _available:
        return LLMResponse(content="", raw=None)

    if _provider == "bedrock":
        return _chat_bedrock(messages, temperature, max_tokens)
    return _chat_openai(messages, temperature, max_tokens, response_format)


def _chat_bedrock(
    messages: list[LLMMessage],
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    """Bedrock プロキシ経由で Claude を呼ぶ。"""
    import httpx

    system_parts = []
    api_messages = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        else:
            api_messages.append({"role": m.role, "content": m.content})

    if not api_messages:
        return LLMResponse(content="", raw=None)

    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": api_messages,
        "temperature": temperature,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)

    url = f"{_bedrock_base_url}/model/{_bedrock_model}/invoke"

    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": _bedrock_token,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )

        if resp.status_code != 200:
            logger.error("Bedrock API エラー [%d]: %s", resp.status_code, resp.text[:300])
            return LLMResponse(content="", raw=None)

        data = resp.json()
        content_blocks = data.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        usage_data = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("input_tokens", 0),
            "completion_tokens": usage_data.get("output_tokens", 0),
            "total_tokens": usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        }
        return LLMResponse(content=text, raw=data, usage=usage)

    except httpx.TimeoutException:
        logger.error("Bedrock API タイムアウト")
        return LLMResponse(content="", raw=None)
    except Exception as e:
        logger.error("Bedrock API エラー: %s", e)
        return LLMResponse(content="", raw=None)


def _chat_openai(
    messages: list[LLMMessage],
    temperature: float,
    max_tokens: int,
    response_format: str | None = None,
) -> LLMResponse:
    """OpenAI API で推論する。"""
    if _openai_client is None:
        return LLMResponse(content="", raw=None)

    try:
        kwargs: dict[str, Any] = {
            "model": _openai_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        resp = _openai_client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(content=content, raw=resp, usage=usage)

    except Exception as e:
        logger.error("OpenAI API エラー: %s", e)
        return LLMResponse(content="", raw=None)


def think(system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
    """ショートカット: system + user の2メッセージで推論する。"""
    return chat(
        [LLMMessage("system", system_prompt), LLMMessage("user", user_prompt)],
        **kwargs,
    )


def think_json(system_prompt: str, user_prompt: str, **kwargs) -> dict | None:
    """JSON 形式のレスポンスを期待する推論。"""
    if _provider == "bedrock":
        kwargs.pop("response_format", None)
        if "system_prompt" not in kwargs:
            system_prompt += "\n\n必ず有効な JSON のみを返してください。マークダウンや説明テキストは不要です。"
        resp = think(system_prompt, user_prompt, **kwargs)
    else:
        resp = think(system_prompt, user_prompt, response_format="json", **kwargs)
    return resp.as_json()
