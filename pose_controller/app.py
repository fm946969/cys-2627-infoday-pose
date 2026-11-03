"""Application orchestrator for the CYS AI Pose Controller."""

from contextlib import ExitStack

import cv2
import numpy as np

from .commands import CommandClassifier, ControlCommand
from .config import ControllerConfig, FrameArray
from .detector import PoseDetector
from .renderer import HudRenderer
from .serial_client import SerialCommunicator


class PoseControllerApp:
    """Coordinates video capture, pose detection, command classification, serial TX, and HUD."""

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        self.classifier = CommandClassifier(
            turn_x_threshold=self.config.turn_x_threshold
        )

    def run(self) -> None:
        """Start the video capture loop and process pose commands until 'q' is pressed."""
        renderer = HudRenderer(self.config)

        with ExitStack() as stack:
            serial_client = stack.enter_context(SerialCommunicator(self.config))
            detector = stack.enter_context(PoseDetector(self.config))

            cap = cv2.VideoCapture(self.config.camera_index)
            stack.callback(cap.release)
            stack.callback(cv2.destroyAllWindows)

            print(
                "\n🚀 AI Pose Control System Activated! Please stand in front of the camera."
            )
            print("💡 Tip: Press the 'q' key on the video window to quit.")

            while cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    print("⚠️ Failed to grab frame from camera.")
                    break

                frame: FrameArray = np.asarray(cv2.flip(raw_frame, 1), dtype=np.uint8)
                rgb_frame: FrameArray = np.asarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8
                )

                landmarks = detector.detect(rgb_frame)
                current_cmd = (
                    self.classifier.classify(landmarks)
                    if landmarks is not None
                    else ControlCommand.STOP
                )

                serial_client.send(current_cmd)
                renderer.render_frame(frame, landmarks, current_cmd)

                cv2.imshow(self.config.window_title, frame)

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break


def main(config: ControllerConfig | None = None) -> None:
    """Run the AI Pose Controller application."""
    app = PoseControllerApp(config=config)
    app.run()
