from __future__ import annotations

import ssl

import aiohttp
import certifi

_ssl_context = ssl.create_default_context(cafile=certifi.where())


def new_client_session(*, timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientSession:
    kwargs: dict[str, object] = {"connector": aiohttp.TCPConnector(ssl=_ssl_context)}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return aiohttp.ClientSession(**kwargs)  # type: ignore[arg-type]
