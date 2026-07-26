"""Rerank 提供商代理配置的测试（#9383）。

验证 aiohttp 系 Rerank 提供商（vLLM / 百炼 / NVIDIA / TEI）能够读取
provider 配置中的 `proxy` 字段，并在每次请求中传递给 aiohttp。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from astrbot.core.provider.sources.bailian_rerank_source import BailianRerankProvider
from astrbot.core.provider.sources.nvidia_rerank_source import NvidiaRerankProvider
from astrbot.core.provider.sources.tei_rerank_source import TEIRerankProvider
from astrbot.core.provider.sources.vllm_rerank_source import VLLMRerankProvider

PROXY = "http://127.0.0.1:7890"


class _FakeResponse:
    def __init__(self, json_data) -> None:
        self._json_data = json_data
        self.status = 200

    async def json(self):
        return self._json_data

    def raise_for_status(self) -> None:
        return None

    async def text(self) -> str:
        return ""


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info) -> None:
        return None


def _make_fake_client(json_data) -> tuple[MagicMock, dict]:
    """构造伪 aiohttp 客户端，记录每次请求实际传入的关键字参数。"""
    captured: dict = {}
    client = MagicMock()
    client.closed = False

    def _request(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeRequestContext(_FakeResponse(json_data))

    client.post = MagicMock(side_effect=_request)
    client.get = MagicMock(side_effect=_request)
    return client, captured


async def _swap_client(provider, fake_client) -> None:
    """替换 provider 持有的真实 ClientSession，避免真实网络请求。"""
    real_client = getattr(provider, "client", None)
    if real_client is not None and hasattr(real_client, "close"):
        await real_client.close()
    provider.client = fake_client


@pytest.mark.asyncio
async def test_vllm_rerank_passes_proxy():
    provider = VLLMRerankProvider(
        {"rerank_model": "m", "proxy": PROXY}, provider_settings={}
    )
    client, captured = _make_fake_client(
        {"results": [{"index": 0, "relevance_score": 0.5}]}
    )
    await _swap_client(provider, client)

    results = await provider.rerank("q", ["doc1"])

    assert provider.proxy == PROXY
    assert captured["proxy"] == PROXY
    assert results and results[0].index == 0


@pytest.mark.asyncio
async def test_bailian_rerank_passes_proxy():
    provider = BailianRerankProvider(
        {"rerank_api_key": "test-key", "proxy": PROXY}, provider_settings={}
    )
    client, captured = _make_fake_client(
        {
            "output": {"results": [{"index": 0, "relevance_score": 0.9}]},
            "usage": {"total_tokens": 1},
        }
    )
    await _swap_client(provider, client)

    results = await provider.rerank("q", ["doc1"])

    assert captured["proxy"] == PROXY
    assert results and results[0].index == 0


@pytest.mark.asyncio
async def test_nvidia_rerank_passes_proxy():
    provider = NvidiaRerankProvider(
        {"nvidia_rerank_api_key": "k", "proxy": PROXY}, provider_settings={}
    )
    client, captured = _make_fake_client(
        {"rankings": [{"index": 0, "relevance_score": 0.7}], "usage": {}}
    )
    provider.client = client  # NVIDIA 的客户端是惰性创建的，直接注入

    results = await provider.rerank("q", ["doc1"])

    assert captured["proxy"] == PROXY
    assert results and results[0].index == 0


@pytest.mark.asyncio
async def test_tei_rerank_passes_proxy_for_rerank_and_health_check():
    provider = TEIRerankProvider({"proxy": PROXY}, provider_settings={})
    client, captured = _make_fake_client([{"index": 0, "score": 0.6}])
    await _swap_client(provider, client)

    results = await provider.rerank("q", ["doc1"])
    assert captured["proxy"] == PROXY
    assert results and results[0].index == 0

    captured.clear()
    await provider.test()
    assert captured["proxy"] == PROXY


@pytest.mark.asyncio
async def test_rerank_without_proxy_defaults_to_none():
    provider = VLLMRerankProvider({"rerank_model": "m"}, provider_settings={})
    client, captured = _make_fake_client(
        {"results": [{"index": 0, "relevance_score": 0.5}]}
    )
    await _swap_client(provider, client)

    await provider.rerank("q", ["doc1"])

    assert provider.proxy == ""
    assert captured["proxy"] is None
