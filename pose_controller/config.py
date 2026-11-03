"""Configuration and shared type aliases for the AI Pose Controller."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

# Python 3.12+ / 3.14 `type` statement aliases
type FrameArray = npt.NDArray[np.uint8]
type BGRColor = tuple[int, int, int]

GREEN_COLOR: Final[BGRColor] = (0, 255, 0)
YELLOW_COLOR: Final[BGRColor] = (0, 255, 255)
BLACK_COLOR: Final[BGRColor] = (0, 0, 0)


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Immutable configuration for the AI Pose Controller."""

    # Serial Port (Modify to your micro:bit's COM Port in Windows Device Manager)
    serial_port: str = "COM5"
    baud_rate: int = 115200
    serial_timeout: float = 0.1
    send_interval: float = 0.15

    # Logo Setup (checks assets/ first, with fallback to workspace root)
    logo_path: Path = Path("assets/3cys-schoolLogo.png")
    logo_size: int = 120
    logo_margin: int = 15

    # Camera & Pose Detection (MediaPipe Tasks API for Python 3.14 / MediaPipe 1.x)
    camera_index: int = 0
    model_path: Path = Path("models/pose_landmarker_full.task")
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )
    min_detection_confidence: float = 0.7
    min_presence_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    turn_x_threshold: float = 0.25

    # Window & HUD
    window_title: str = "CYS AI Pose Controller (Maqueen Plus V2)"

    def resolve_logo_path(self) -> Path:
        """Resolve the logo path, falling back to the workspace root if needed."""
        if self.logo_path.is_file():
            return self.logo_path
        fallback = Path(self.logo_path.name)
        if fallback.is_file():
            return fallback
        return self.logo_path
