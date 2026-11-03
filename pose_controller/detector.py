"""MediaPipe Tasks PoseLandmarker wrapper for Python 3.14."""

from collections.abc import Sequence
from pathlib import Path
import time
from types import TracebackType
from typing import Any, Self
import urllib.request

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .config import ControllerConfig, FrameArray


class PoseDetector:
    """Context-managed wrapper around MediaPipe Tasks PoseLandmarker (VIDEO mode)."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._landmarker: vision.PoseLandmarker | None = None
        self._start_time_ns: int = time.monotonic_ns()
        self._last_timestamp_ms: int = -1

    def ensure_model(self) -> Path:
        """Ensure the PoseLandmarker .task model file exists locally."""
        model_path = self._config.model_path
        if model_path.is_file():
            return model_path

        fallback_root = Path(model_path.name)
        if fallback_root.is_file():
            return fallback_root

        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"📥 Downloading MediaPipe pose model to '{model_path}'...")
        urllib.request.urlretrieve(self._config.model_url, model_path)
        print(f"✅ Successfully downloaded '{model_path}'.")
        return model_path

    def open(self) -> None:
        """Initialize the MediaPipe PoseLandmarker instance."""
        model_path = self.ensure_model()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self._config.min_detection_confidence,
            min_pose_presence_confidence=self._config.min_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._start_time_ns = time.monotonic_ns()
        self._last_timestamp_ms = -1

    def detect(self, rgb_frame: FrameArray) -> Sequence[Any] | None:
        """Run video pose detection on an RGB frame and return the first pose's landmarks."""
        if self._landmarker is None:
            raise RuntimeError("PoseDetector is not initialized. Call open() or use a 'with' block.")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(
            self._last_timestamp_ms + 1,
            (time.monotonic_ns() - self._start_time_ns) // 1_000_000,
        )
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            return result.pose_landmarks[0]
        return None

    def close(self) -> None:
        """Release the underlying MediaPipe PoseLandmarker resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            finally:
                self._landmarker = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
