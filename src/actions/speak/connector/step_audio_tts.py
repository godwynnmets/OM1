import json
import logging
import time
from typing import Optional
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.step_audio_tts_provider import StepAudioTTSProvider
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
    pass


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

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # IO Provider
        self.io_provider = IOProvider()

        # Initialize Step Audio TTS Provider
        self.tts = StepAudioTTSProvider()
        self.tts.start()

        # TTS status
        self.tts_enabled = True

        # Initialize conversation provider
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Process a speak action by sending text to StepAudio TTS.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.
        """
        if self.tts_enabled is False:
            logging.info("TTS is disabled, skipping TTS action")
            return

        # Add pending message to TTS
        pending_message = self.tts.create_pending_message(output_interface.action)

        # Store robot message to conversation history only if there was ASR input
        if (
            self.io_provider.llm_prompt is not None
            and "INPUT: Voice" in self.io_provider.llm_prompt
        ):
            self.conversation_provider.store_robot_message(output_interface.action)

        self.tts.add_pending_message(pending_message)


    def stop(self) -> None:
        """
        Stop the StepAudio TTS connector and cleanup resources.
        """
        if self.tts:
            self.tts.stop()
