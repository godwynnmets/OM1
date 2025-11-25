import logging

from backgrounds.base import Background, BackgroundConfig
from providers.vlm_ingest_rtsp_provider import VLMIngestRTSPProvider


class LostAndFoundIngestRTSP(Background):
    """
    Background task that starts the simple Lost & Found ingest (VLMIngestRTSPProvider).

    It subscribes to an RTSP video stream, runs YOLO, and stores detections
    and frames in SQLite for later "lost and found" queries.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        """
        Initialize the Lost & Found ingest background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration for the background task.

            Optional fields on config:
              - cfg_path: str (default "config.yaml")
              - rtsp_url: str (default "rtsp://localhost:8554/top_camera")
              - decode_format: str (default "H264")
              - fps: int (default 30)
              - ingest_stride: int (default 3)
              - queue_max: int (default 10)
        """
        super().__init__(config)

        cfg_path = getattr(self.config, "cfg_path", None)
        rtsp_url = getattr(self.config, "rtsp_url", "rtsp://localhost:8554/top_camera")
        decode_format = getattr(self.config, "decode_format", "H264")
        fps = getattr(self.config, "fps", 30)
        ingest_stride = getattr(self.config, "ingest_stride", 3)
        queue_max = getattr(self.config, "queue_max", 10)

        self.ingest_provider = VLMIngestRTSPProvider(
            cfg_path=cfg_path,
            rtsp_url=rtsp_url,
            decode_format=decode_format,
            fps=fps,
            ingest_stride=ingest_stride,
            queue_max=queue_max,
        )
        self.ingest_provider.start()

        logging.info(
            "LostAndFoundIngestRTSP background initialized "
            f"(cfg_path: {cfg_path}, rtsp_url: {rtsp_url}, "
            f"stride: {ingest_stride}, queue_max: {queue_max})"
        )
