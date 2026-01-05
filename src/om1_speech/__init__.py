"""Minimal local stub for `om1_speech` used for development.

This provides lightweight stand-ins for the real audio stream classes so the
project can import them during development. Replace or extend these with the
real implementation when available.
"""
from typing import Optional, Any


class AudioOutputStream:
    """A minimal audio output stream stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False

    def write(self, data: bytes) -> None:
        """Write audio data to the output stream (no-op)."""
        if self.closed:
            raise RuntimeError("Stream is closed")

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "AudioOutputStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class AudioInputStream:
    """A minimal audio input stream stub."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.closed = False

    def read(self, size: Optional[int] = None) -> bytes:
        """Read audio data from the input stream (returns empty bytes)."""
        if self.closed:
            raise RuntimeError("Stream is closed")
        return b""

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "AudioInputStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class AudioRTSPInputStream(AudioInputStream):
    """Stub for an RTSP audio input stream."""

    def __init__(self, rtsp_url: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rtsp_url = rtsp_url


__all__ = ["AudioOutputStream", "AudioInputStream", "AudioRTSPInputStream"]
