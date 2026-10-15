"""The Lemonade client's backend calls, against a stand-in HTTP server."""

from __future__ import annotations

import json

import httpx
import pytest

from ai_buddy.lemonade import Lemonade, LemonadeError


def client(handler) -> Lemonade:
    lm = Lemonade("http://lemonade.test/v1")
    lm._http = httpx.AsyncClient(base_url=lm.base_url, transport=httpx.MockTransport(handler))
    return lm


def sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


async def test_backends_reads_system_info():
    def handler(req: httpx.Request):
        assert req.url.path == "/v1/system-info"
        return httpx.Response(200, json={"recipes": {"flm": {"backends": {"npu": {"state": "update_required"}}}}})
    assert await client(handler).backends() == {"flm": {"npu": {"state": "update_required"}}}


async def test_install_streams_progress_until_complete():
    def handler(req: httpx.Request):
        assert req.url.path == "/v1/install"
        assert json.loads(req.content) == {"recipe": "flm", "backend": "npu", "stream": True}
        return httpx.Response(200, content=sse(("progress", {"file": "flm.tar.gz", "percent": 40}),
                                               ("progress", {"file": "flm.tar.gz", "percent": 100}),
                                               ("complete", {})),
                              headers={"content-type": "text/event-stream"})
    events = [e async for e in client(handler).install_backend("flm", "npu")]
    assert [e["percent"] for e in events] == [40, 100]


async def test_install_error_event_raises():
    def handler(req):
        return httpx.Response(200, content=sse(("progress", {"percent": 3}), ("error", {"error": "disk full"})))
    with pytest.raises(LemonadeError, match="disk full"):
        [e async for e in client(handler).install_backend("flm", "npu")]


async def test_install_without_completion_raises():
    def handler(req):
        return httpx.Response(200, content=sse(("progress", {"percent": 3})))
    with pytest.raises(LemonadeError, match="without completing"):
        [e async for e in client(handler).install_backend("flm", "npu")]
