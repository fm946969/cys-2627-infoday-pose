"""Application orchestrator for the CYS AI Pose Controller."""

from contextlib import ExitStack
from dataclasses import replace

import cv2
import numpy as np
from serial.tools import list_ports

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
    if config is None:
        config = _select_serial_port(ControllerConfig())
    app = PoseControllerApp(config=config)
    app.run()


def _select_serial_port(config: ControllerConfig) -> ControllerConfig:
    """Prompt the user to select an available serial device."""
    ports = sorted(list_ports.comports(), key=lambda port: port.device)
    if not ports:
        print("⚠️ No serial devices found. Continuing in No-Hardware Test Mode.")
        return config

    print("\nAvailable serial devices:")
    for index, port in enumerate(ports, start=1):
        print(f"  {index}. {port.device} - {port.description}")

    while True:
        choice = input(f"Select a COM port [1-{len(ports)}]: ")
        try:
            selected_index = int(choice) - 1
        except ValueError:
            selected_index = -1

        if 0 <= selected_index < len(ports):
            selected_port = ports[selected_index].device
            print(f"Selected serial port: {selected_port}")
            return replace(config, serial_port=selected_port)

        print(f"Please enter a number from 1 to {len(ports)}.")
