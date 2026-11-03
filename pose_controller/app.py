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

CAMERA_INDEX_LIMIT = 10
CAPTURE_MODES: tuple[tuple[str, int | None, int | None, float | None], ...] = (
    ("Auto", None, None, None),
    ("VGA 640x480 @30", 640, 480, 30.0),
    ("HD 1280x720 @30", 1280, 720, 30.0),
    ("Full HD 1920x1080 @30", 1920, 1080, 30.0),
)


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

            cameras = _find_available_cameras()
            camera_index = (
                self.config.camera_index
                if self.config.camera_index in cameras
                else (cameras[0] if cameras else self.config.camera_index)
            )
            if cameras:
                print(f"Available cameras: {', '.join(map(str, cameras))}")
            else:
                print("⚠️ No cameras detected; trying the configured camera.")

            cap = cv2.VideoCapture(camera_index)
            stack.callback(lambda: cap.release())
            stack.callback(cv2.destroyAllWindows)
            capture_mode_index = 0
            original_capture_properties = (
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap.get(cv2.CAP_PROP_FPS),
            )

            print(
                "\n🚀 AI Pose Control System Activated! Please stand in front of the camera."
            )
            print("💡 Use C to change camera, M to change capture mode, and Q to quit.")

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
                mode_name = CAPTURE_MODES[capture_mode_index][0]
                renderer.render_frame(
                    frame,
                    landmarks,
                    current_cmd,
                    camera_index=camera_index,
                    capture_mode=mode_name,
                )

                cv2.imshow(self.config.window_title, frame)

                key = cv2.waitKey(5) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c") and len(cameras) > 1:
                    next_camera = _next_camera(cameras, camera_index)
                    new_cap = cv2.VideoCapture(next_camera)
                    if new_cap.isOpened():
                        cap.release()
                        cap = new_cap
                        camera_index = next_camera
                        original_capture_properties = (
                            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                            cap.get(cv2.CAP_PROP_FPS),
                        )
                        if capture_mode_index:
                            if not _apply_capture_mode(
                                cap,
                                CAPTURE_MODES[capture_mode_index],
                                original_capture_properties,
                            ):
                                capture_mode_index = 0
                                print(
                                    "⚠️ Capture mode is unsupported by this camera; "
                                    "using Auto."
                                )
                        print(f"Switched to camera {camera_index}.")
                    else:
                        new_cap.release()
                        print(f"⚠️ Could not open camera {next_camera}.")
                if key == ord("m"):
                    next_mode_index = (capture_mode_index + 1) % len(CAPTURE_MODES)
                    next_mode = CAPTURE_MODES[next_mode_index]
                    if _apply_capture_mode(
                        cap, next_mode, original_capture_properties
                    ):
                        capture_mode_index = next_mode_index
                        print(f"Capture mode: {next_mode[0]}")
                    else:
                        print(f"⚠️ Camera does not support capture mode {next_mode[0]}.")


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


def _find_available_cameras() -> list[int]:
    """Return camera indexes that OpenCV can open."""
    available: list[int] = []
    for camera_index in range(CAMERA_INDEX_LIMIT):
        camera = cv2.VideoCapture(camera_index)
        try:
            if camera.isOpened():
                available.append(camera_index)
        finally:
            camera.release()
    return available


def _next_camera(cameras: list[int], current_camera: int) -> int:
    """Return the next available camera index, wrapping at the end."""
    return cameras[(cameras.index(current_camera) + 1) % len(cameras)]


def _apply_capture_mode(
    camera: cv2.VideoCapture,
    mode: tuple[str, int | None, int | None, float | None],
    original_properties: tuple[float, float, float] | None = None,
) -> bool:
    """Apply a resolution/FPS preset, restoring the prior settings if unsupported."""
    name, width, height, fps = mode
    if name == "Auto":
        if original_properties is None:
            return True
        width, height, fps = original_properties
    elif width is None or height is None or fps is None:
        return False

    previous_properties = (
        camera.get(cv2.CAP_PROP_FRAME_WIDTH),
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
        camera.get(cv2.CAP_PROP_FPS),
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    camera.set(cv2.CAP_PROP_FPS, float(fps))

    if name == "Auto":
        return True

    actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    if (
        abs(actual_width - float(width)) < 1
        and abs(actual_height - float(height)) < 1
        and abs(actual_fps - float(fps)) < 1
    ):
        return True

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, previous_properties[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, previous_properties[1])
    camera.set(cv2.CAP_PROP_FPS, previous_properties[2])
    return False
