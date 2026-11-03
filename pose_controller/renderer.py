"""On-Screen Display (HUD), skeleton landmark drawing, and school logo overlay."""

from collections.abc import Sequence
from typing import Any

import cv2
from mediapipe.tasks.python import vision
import numpy as np

from .commands import ControlCommand
from .config import (
    BLACK_COLOR,
    GREEN_COLOR,
    YELLOW_COLOR,
    ControllerConfig,
    FrameArray,
)

mp_drawing = vision.drawing_utils
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS


class HudRenderer:
    """Renders pose landmarks, school logo overlay, and status HUD onto video frames."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._logo: FrameArray | None = self._load_logo()
        self._draw_spec_landmarks = mp_drawing.DrawingSpec(
            color=GREEN_COLOR, thickness=3, circle_radius=3
        )
        self._draw_spec_connections = mp_drawing.DrawingSpec(
            color=GREEN_COLOR, thickness=2, circle_radius=2
        )

    def _load_logo(self) -> FrameArray | None:
        """Load and resize the school logo if available."""
        logo_path = self._config.resolve_logo_path()
        if not logo_path.is_file():
            print(
                f"⚠️ Warning: Logo file '{logo_path}' not found. "
                "The script will run without the logo."
            )
            return None

        try:
            raw_logo = cv2.imread(str(logo_path))
            if raw_logo is None:
                raise ValueError(f"OpenCV could not decode '{logo_path}'.")
            resized_logo: FrameArray = np.asarray(
                cv2.resize(
                    raw_logo,
                    (self._config.logo_size, self._config.logo_size),
                ),
                dtype=np.uint8,
            )
            print(f"✅ Successfully loaded and resized school logo '{logo_path}'.")
            return resized_logo
        except Exception as exc:
            print(
                f"⚠️ Warning: Could not process logo. Error: {exc}. "
                "The script will run without the logo."
            )
            return None

    def draw_landmarks(
        self,
        frame: FrameArray,
        landmarks: Sequence[Any],
    ) -> None:
        """Draw detected pose landmarks and skeleton connections onto the frame."""
        mp_drawing.draw_landmarks(
            frame,
            list(landmarks),
            POSE_CONNECTIONS,
            landmark_drawing_spec=self._draw_spec_landmarks,
            connection_drawing_spec=self._draw_spec_connections,
        )

    def overlay_logo(self, frame: FrameArray) -> None:
        """Overlay the school logo in the top-left corner of the frame."""
        if self._logo is None:
            return

        margin = self._config.logo_margin
        end_pos = margin + self._config.logo_size
        h, w = frame.shape[:2]
        if h >= end_pos and w >= end_pos:
            frame[margin:end_pos, margin:end_pos] = self._logo

    def draw_hud(self, frame: FrameArray, command: ControlCommand) -> None:
        """Draw the top-right status panel showing USB command and action text."""
        _h, w = frame.shape[:2]
        cv2.rectangle(frame, (w - 340, 10), (w - 10, 95), BLACK_COLOR, -1)
        cv2.putText(
            frame,
            f"USB Sent: {command.value}",
            (w - 330, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            YELLOW_COLOR,
            2,
        )
        cv2.putText(
            frame,
            f"Action  : {command.action_text}",
            (w - 330, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            GREEN_COLOR,
            2,
        )

    def render_frame(
        self,
        frame: FrameArray,
        landmarks: Sequence[Any] | None,
        command: ControlCommand,
    ) -> None:
        """Render skeleton landmarks (if present), logo, and HUD onto the frame."""
        if landmarks is not None:
            self.draw_landmarks(frame, landmarks)
        self.overlay_logo(frame)
        self.draw_hud(frame, command)
