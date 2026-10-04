"""Whisper speech-to-text client interface (Milestone 3).

The platform core supervises the whisper-server process (services.py); this
application-layer client is the audio-in -> transcript-out seam. As of Milestone
3 it is a real client: WhisperClient.transcribe posts the audio file to a running
whisper-server /inference endpoint and returns the transcript the server reports.
No fake transcription is ever produced -- a transport or server error propagates
as an exception the caller must handle honestly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from whisper import transcribe_file


class SpeechToText(ABC):
    """Interface for transcribing audio to text."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Return the transcript of the given audio file."""


class WhisperClient(SpeechToText):
    """Real whisper-server client: audio file -> transcript over HTTP /inference.

    Constructed with the resolved loopback port of a running whisper-server (the
    launcher supplies it after starting the service), so this client holds no
    machine-specific path and never starts a process itself.
    """

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self._port = port
        self._host = host

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file via the running whisper-server. Real output only."""
        return transcribe_file(audio_path, self._port, self._host)
