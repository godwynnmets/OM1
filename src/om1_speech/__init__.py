"""Minimal local stub for `om1_speech` used for development.

This provides lightweight stand-ins for the real audio stream classes so the
project can import them during development. Replace or extend these with the
real implementation when available.
"""
from typing import Optional, Any
import logging
import os
import json

import torch
import torchaudio

# Make sure you have installed modelscope, SpeechT5 and transformers
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.audio.tts.speech_adapter import SpeechAdapter
from modelscope.utils.audio.tts_exceptions import NoValidRequestException
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import save_wav


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
        self.running = False
        self._audio_callback = None
        self.remote_input = False

    def read(self, size: Optional[int] = None) -> bytes:
        """Read audio data from the input stream (returns empty bytes)."""
        if self.closed:
            raise RuntimeError("Stream is closed")
        return b""

    def register_audio_data_callback(self, cb):
        """Register a callback that will receive audio chunks."""
        self._audio_callback = cb

    def fill_buffer_remote(self, data: bytes) -> None:
        """Fill the internal remote buffer (no-op for stub)."""
        # For remote input tests, this method can be used to inject audio
        pass

    def start(self) -> None:
        """Start the audio input stream (no-op)."""
        self.running = True

    def stop(self) -> None:
        """Stop the audio input stream (no-op)."""
        self.running = False

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


class StepAudioTTS:
    def __init__(self, model_dir, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.cosyvoice = CosyVoice(model_dir=self.model_dir, device=self.device)
        self.cosyvoice.list_avaliable_spks()
        self.model_id = 'damo/speech_adapter-v2-16k-paimo_8k'
        try:
            self.model_dir = snapshot_download(self.model_id,
                                          cache_dir='/tmp/modelscope',
                                          ignore_file_pattern=[r'\..*py$'])
        except Exception as e:
            logging.warning(f'snapshot_download failed, using last version, error: {e}')
        self.speech_adapter = SpeechAdapter(self.model_dir, device=self.device)
        self.speakers = {}

    def register_speakers(self, spks_json):
        with open(spks_json, 'r') as f:
            spks_info = json.load(f)
        for spk, spk_file in spks_info.items():
            if not os.path.exists(spk_file):
                logging.warning(f'spk_file {spk_file} not found, download from modelscope')
                try:
                    spk_file_new = snapshot_download('manyeyes/manyeyes_cosyvoice_backup',
                                                     cache_dir='/tmp/modelscope',
                                                     allow_file_pattern=f'spk_new/{os.path.basename(spk_file)}')
                    spk_file = spk_file_new
                except Exception as e:
                    logging.error(f'failed to download spk file, error: {e}')
                    continue
            self.speakers[spk] = self.cosyvoice.get_spk_emb(spk_file)

    def synthesize(self, text, spk, out_wav):
        if spk not in self.speakers:
            raise NoValidRequestException(f'speaker {spk} not registered')
        if text.startswith('[') and text.endswith(']'):
            try:
                output = self.cosyvoice.inference_sft(text, spk, self.speakers[spk])
            except NoValidRequestException:
                logging.warning('Execute sft inference failed, maybe the text is not a valid sft request')
                output = self.cosyvoice.inference_zero_shot(text, spk, self.speakers[spk])
        else:
            output = self.cosyvoice.inference_zero_shot(text, spk, self.speakers[spk])
        output = self.speech_adapter.forward(output['tts_speech'])
        save_wav(output, out_wav)


__all__ = ["AudioOutputStream", "AudioInputStream", "AudioRTSPInputStream", "StepAudioTTS"]
