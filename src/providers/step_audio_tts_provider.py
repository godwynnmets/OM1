import logging
from typing import Callable, Optional, Union

from om1_speech import StepAudioTTS

from .singleton import singleton


@singleton
class StepAudioTTSProvider:
    """
    Text-to-Speech Provider that manages an audio output stream.

    A singleton class that handles text-to-speech conversion and audio output
    through a dedicated thread.
    """

    def __init__(
        self,
    ):
        """
        Initialize the TTS provider.
        """
        self.running: bool = False
        self._audio_stream: StepAudioTTS = StepAudioTTS(model_dir="/tmp/modelscope")
        self._audio_stream.register_speakers('speakers/speakers_info.json')

    def create_pending_message(self, text: str) -> dict:
        """
        Create a pending message for TTS processing.

        Parameters
        ----------
        text : str
            Text to be converted to speech

        Returns
        -------
        dict
            A dictionary containing the TTS request parameters.
        """
        logging.info(f"audio_stream: {text}")
        return {
            "text": text,
        }

    def add_pending_message(self, message: Union[str, dict], spk: str, out_wav: str):
        """
        Add a pending message to the TTS provider.

        Parameters
        ----------
        message : Union[str, dict]
            The message to be added, typically containing text and TTS parameters.
        spk : str
            The speaker to use for TTS.
        out_wav : str
            The output wav file.
        """
        if not self.running:
            logging.warning(
                "TTS provider is not running. Call start() before adding messages."
            )
            return

        if isinstance(message, str):
            message = self.create_pending_message(message)
        
        self._audio_stream.synthesize(message["text"], spk=spk, out_wav=out_wav)

    def get_pending_message_count(self) -> int:
        """
        Get the count of pending messages in the TTS provider.

        Returns
        -------
        int
            The number of pending messages.
        """
        return 0

    def start(self):
        """
        Start the TTS provider and its audio stream.
        """
        if self.running:
            logging.warning("Step Audio TTS provider is already running")
            return

        self.running = True

    def stop(self):
        """
        Stop the TTS provider and cleanup resources.
        """
        if not self.running:
            logging.warning("Step Audio TTS provider is not running")
            return

        self.running = False
