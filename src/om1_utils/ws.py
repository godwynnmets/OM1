"""Websocket helper stub used by OM1 for development only.

This implements a minimal `Client` class with the methods used in the
repository: `connect`, `send`, `recv`, and `close`. Methods are safe no-ops
or simple placeholders.
"""
from typing import Any, Optional, Callable
import asyncio
import logging


class Client:
    def __init__(self, url: str, *args: Any, **kwargs: Any) -> None:
        self.url = url
        self._open = False
        self._message_callback: Optional[Callable[[str], None]] = None
        self._loop = asyncio.get_event_loop()
        self._recv_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Async connect (no-op)."""
        self._open = True

    async def send(self, data: Any) -> None:
        """Send data (no-op)."""
        if not self._open:
            raise RuntimeError("Client not connected")

    def send_message(self, data: Any) -> None:
        """Convenience sync method expected by callers.

        Schedules the async send on the event loop so callers don't need to be async.
        """
        try:
            if asyncio.get_event_loop().is_running():
                # we're in an event loop; schedule
                asyncio.get_event_loop().create_task(self.send(data))
            else:
                self._loop.run_until_complete(self.send(data))
        except RuntimeError:
            # If no loop is available, just ignore in stub
            pass

    async def recv(self, timeout: Optional[float] = None) -> Any:
        """Return empty bytes to indicate no data."""
        if not self._open:
            raise RuntimeError("Client not connected")
        await asyncio.sleep(0)
        return b""

    def register_message_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        """Register a callback to be invoked when messages are received."""
        self._message_callback = cb

    def _recv_loop(self) -> None:
        """Background task that polls recv and dispatches to the callback."""
        async def _task():
            while self._open:
                try:
                    msg = await self.recv()
                    if msg and self._message_callback:
                        try:
                            self._message_callback(msg)
                        except Exception as e:
                            logging.warning(f"om1_utils.ws Client callback error: {e}")
                except Exception:
                    await asyncio.sleep(0.1)

        self._recv_task = asyncio.create_task(_task())

    def start(self) -> None:
        """Start the client (no-op but begins recv loop in environments with a running loop)."""
        self._open = True
        try:
            # try to schedule the recv loop if a running loop exists
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._recv_loop()
        except RuntimeError:
            # no running loop available in this environment
            pass

    def stop(self) -> None:
        """Stop the client and cancel background tasks."""
        self._open = False
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()


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
