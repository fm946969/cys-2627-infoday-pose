"""Control commands and pose gesture classification logic."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any
import math

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


def _calculate_angle(point_a: Any, vertex: Any, point_b: Any) -> float:
    """Return the angle in degrees (0.0-180.0) at `vertex` formed by `point_a` and `point_b`."""
    radians = math.atan2(point_b.y - vertex.y, point_b.x - vertex.x) - math.atan2(
        point_a.y - vertex.y, point_a.x - vertex.x
    )
    angle = abs(math.degrees(radians))
    return 360.0 - angle if angle > 180.0 else angle


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
        left_eye = landmarks[PoseLandmark.LEFT_EYE]
        right_eye = landmarks[PoseLandmark.RIGHT_EYE]
        left_elbow = landmarks[PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[PoseLandmark.RIGHT_ELBOW]

        left_angle = _calculate_angle(left_elbow, left_shoulder, left_hip)
        right_angle = _calculate_angle(right_elbow, right_shoulder, right_hip)

        left_hand_down = left_angle < 45 and left_wrist.y > 0.85 * left_hip.y + 0.15 * left_shoulder.y
        right_hand_down = right_angle < 45 and right_wrist.y > 0.85 * right_hip.y + 0.15 * right_shoulder.y

        left_hand_up = left_wrist.y < left_eye.y
        right_hand_up = right_wrist.y < right_eye.y


        print(f"{left_angle=}, {right_angle=}")
        left_hand_extended = 45 < left_angle < 135
        right_hand_extended = 45 < right_angle < 135

        if left_hand_up and right_hand_up:
            return ControlCommand.FORWARD
        if left_hand_down and right_hand_down:
            return ControlCommand.BACKWARD
        if (
                135 > left_angle > 30 and right_angle < 45
        ):
            return ControlCommand.RIGHT
        if (
                135 > right_angle > 30 and left_angle < 45
        ):
            return ControlCommand.LEFT
        return ControlCommand.STOP
