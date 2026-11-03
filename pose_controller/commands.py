"""Control commands and pose gesture classification logic."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from mediapipe.tasks.python import vision

PoseLandmark = vision.PoseLandmark


class ControlCommand(StrEnum):
    """Serial command codes and on-screen action descriptions."""

    FORWARD = "F"
    BACKWARD = "B"
    LEFT = "L"
    RIGHT = "R"
    STOP = "S"

    @property
    def action_text(self) -> str:
        match self:
            case ControlCommand.FORWARD:
                return "FORWARD (↑)"
            case ControlCommand.BACKWARD:
                return "BACKWARD (↓)"
            case ControlCommand.LEFT:
                return "TURN LEFT (←)"
            case ControlCommand.RIGHT:
                return "TURN RIGHT (→)"
            case ControlCommand.STOP:
                return "STOP (🛑)"


class CommandClassifier:
    """Classifies MediaPipe pose landmarks into Maqueen Plus V2 commands."""

    def __init__(self, turn_x_threshold: float = 0.25) -> None:
        self.turn_x_threshold = turn_x_threshold

    def classify(self, landmarks: Sequence[Any]) -> ControlCommand:
        """Map pose landmarks to a control command."""
        left_wrist = landmarks[PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[PoseLandmark.LEFT_HIP]
        right_hip = landmarks[PoseLandmark.RIGHT_HIP]

        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            return ControlCommand.FORWARD
        if left_wrist.y > left_hip.y and right_wrist.y > right_hip.y:
            return ControlCommand.BACKWARD
        if (
            abs(right_wrist.x - right_shoulder.x) > self.turn_x_threshold
            and right_wrist.y < right_hip.y
            and left_wrist.y > left_shoulder.y
        ):
            return ControlCommand.LEFT
        if (
            abs(left_wrist.x - left_shoulder.x) > self.turn_x_threshold
            and left_wrist.y < left_hip.y
            and right_wrist.y > right_shoulder.y
        ):
            return ControlCommand.RIGHT
        return ControlCommand.STOP
