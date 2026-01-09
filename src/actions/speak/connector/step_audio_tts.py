import json
import logging
import time
from typing import Optional
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.asr_rtsp_provider import ASRRTSPProvider
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import (
    AudioStatus,
    String,
    TTSStatusRequest,
    TTSStatusResponse,
    open_zenoh_session,
    prepare_header,
)


class SpeakStepAudioTTSConfig(ActionConfig):
    """
    Configuration for StepAudio TTS connector.

    Parameters:
    ----------
    """


class SpeakStepAudioTTSConnector(
    ActionConnector[SpeakStepAudioTTSConfig, SpeakInput]
):
    """
    A "Speak" connector that uses the StepAudio TTS Provider to perform Text-to-Speech.
    This connector is compatible with the standard SpeakInput interface.
    """

    def __init__(self, config: SpeakStepAudioTTSConfig):
        """
        Initializes the connector and its underlying TTS provider.

        Parameters
        ----------
        config : SpeakStepAudioTTSConfig
            Configuration for the connector.
        """
        super().__init__(config)


    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Process a speak action by sending text to StepAudio TTS.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.
        """
        pass

    def stop(self) -> None:
        """
        Stop the StepAudio TTS connector and cleanup resources.
        """
        pass
