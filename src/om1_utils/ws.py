"""Websocket helper stub used by OM1 for development only.

This implements a minimal `Client` class with the methods used in the
repository: `connect`, `send`, `recv`, and `close`. Methods are safe no-ops
or simple placeholders.
"""
from typing import Any, Optional
import asyncio


class Client:
    def __init__(self, url: str, *args: Any, **kwargs: Any) -> None:
        self.url = url
        self._open = False

    async def connect(self) -> None:
        """Async connect (no-op)."""
        self._open = True

    async def send(self, data: Any) -> None:
        """Send data (no-op)."""
        if not self._open:
            raise RuntimeError("Client not connected")

    async def recv(self, timeout: Optional[float] = None) -> Any:
        """Return empty bytes to indicate no data."""
        if not self._open:
            raise RuntimeError("Client not connected")
        await asyncio.sleep(0)
        return b""

    async def close(self) -> None:
        self._open = False


# Convenience synchronous helpers
class SyncClient(Client):
    def connect(self) -> None:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(super().connect())

    def send(self, data: Any) -> None:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(super().send(data))

    def recv(self, timeout: Optional[float] = None) -> Any:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(super().recv(timeout=timeout))

    def close(self) -> None:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(super().close())


__all__ = ["Client", "SyncClient"]
