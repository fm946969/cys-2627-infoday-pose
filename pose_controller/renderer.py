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
    BGRColor,
    ControllerConfig,
    FrameArray,
)

HUD_TRANSPARENCY = 0.3

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

    def _draw_translucent_box(
        self,
        frame: FrameArray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: BGRColor = BLACK_COLOR,
        transparency: float = HUD_TRANSPARENCY,
    ) -> None:
        """Draw a filled rectangle blended with the underlying frame (30% transparent by default)."""
        h, w = frame.shape[:2]
        x1 = max(0, min(w, top_left[0]))
        y1 = max(0, min(h, top_left[1]))
        x2 = max(0, min(w, bottom_right[0]))
        y2 = max(0, min(h, bottom_right[1]))
        if x2 <= x1 or y2 <= y1:
            return

        alpha = max(0.0, min(1.0, 1.0 - transparency))
        roi = frame[y1:y2, x1:x2]
        overlay = np.full_like(roi, color, dtype=np.uint8)
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    def draw_hud(
        self,
        frame: FrameArray,
        command: ControlCommand,
        serial_status: str = "",
        serial_connected: bool = False,
    ) -> None:
        """Draw the top-right status panel showing USB command, action text, and serial status."""
        _h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        status_label = serial_status or (
            f"{self._config.serial_port} (Connected)"
            if serial_connected
            else (
                f"{self._config.serial_port} (Disconnected)"
                if self._config.serial_port
                else "Test Mode (No COM)"
            )
        )
        serial_line = f"Serial  : {status_label}"
        serial_text_width = cv2.getTextSize(serial_line, font, 0.6, 2)[0][0]
        panel_width = min(max(330, serial_text_width + 24), w - 20)
        left = w - panel_width - 10

        self._draw_translucent_box(
            frame,
            (left, 10),
            (w - 10, 130),
            BLACK_COLOR,
        )
        cv2.putText(
            frame,
            f"USB Sent: {command.value}",
            (left + 10, 40),
            font,
            0.8,
            YELLOW_COLOR,
            2,
        )
        cv2.putText(
            frame,
            f"Action  : {command.action_text}",
            (left + 10, 78),
            font,
            0.7,
            GREEN_COLOR,
            2,
        )
        cv2.putText(
            frame,
            serial_line,
            (left + 10, 114),
            font,
            0.6,
            GREEN_COLOR if serial_connected else YELLOW_COLOR,
            2,
        )

    def draw_operation_hints(
        self,
        frame: FrameArray,
        camera_index: int,
        capture_mode: str,
        available_modes: Sequence[str] = (),
    ) -> None:
        """Draw compact camera, available modes, and keyboard hints at the bottom-left."""
        height, width = frame.shape[:2]
        status = f"Camera {camera_index} | Mode: {capture_mode}"
        modes_line = (
            "Modes: "
            + "  ".join(
                f"{idx}:{name.split(' @')[0]}"
                for idx, name in enumerate(available_modes, start=1)
            )
            if available_modes
            else ""
        )
        mode_key_hint = (
            f"1-{len(available_modes)}/M: mode" if available_modes else "M: mode"
        )
        hints = f"C: next camera  {mode_key_hint}  Q: quit"

        lines: list[tuple[str, BGRColor]] = [(status, YELLOW_COLOR)]
        if modes_line:
            lines.append((modes_line, (220, 220, 220)))
        lines.append((hints, GREEN_COLOR))

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        max_text_width = max(
            cv2.getTextSize(text, font, scale, 1)[0][0] for text, _ in lines
        )
        panel_width = min(max_text_width + 16, width - 20)
        line_height = 18
        panel_height = 8 + line_height * len(lines)
        left = 10
        top = height - panel_height - 10
        self._draw_translucent_box(
            frame,
            (left, top),
            (left + panel_width, top + panel_height),
            BLACK_COLOR,
        )
        for index, (text, color) in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (left + 8, top + 16 + index * line_height),
                font,
                scale,
                color,
                1,
            )

    def render_frame(
        self,
        frame: FrameArray,
        landmarks: Sequence[Any] | None,
        command: ControlCommand,
        camera_index: int = 0,
        capture_mode: str = "Auto",
        serial_status: str = "",
        serial_connected: bool = False,
        available_modes: Sequence[str] = (),
    ) -> None:
        """Render pose, logo, command HUD, and compact operation hints."""
        if landmarks is not None:
            self.draw_landmarks(frame, landmarks)
        self.overlay_logo(frame)
        self.draw_hud(
            frame,
            command,
            serial_status=serial_status,
            serial_connected=serial_connected,
        )
        self.draw_operation_hints(
            frame,
            camera_index,
            capture_mode,
            available_modes=available_modes,
        )
