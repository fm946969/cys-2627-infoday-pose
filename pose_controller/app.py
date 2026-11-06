"""Application orchestrator for the CYS AI Pose Controller."""

from contextlib import ExitStack
from dataclasses import replace
from typing import Any

import cv2
import numpy as np
from serial.tools import list_ports
from serial.tools.list_ports_common import ListPortInfo

from .commands import CommandClassifier, ControlCommand
from .config import BLACK_COLOR, GREEN_COLOR, YELLOW_COLOR, ControllerConfig, FrameArray
from .detector import PoseDetector
from .renderer import HudRenderer
from .serial_client import SerialCommunicator

CAMERA_INDEX_LIMIT = 10
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
SELECTOR_WIDTH = 800
SELECTOR_HEIGHT = 520
MAX_VISIBLE_PORTS = 6
UP_KEY_CODES = {2490368, 65362, 63232}
DOWN_KEY_CODES = {2621440, 65364, 63233}
ENTER_KEY_CODES = {13, 10, 32}
ESC_KEY_CODE = 27


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
            cv2.namedWindow(self.config.window_title, cv2.WINDOW_NORMAL)
            _configure_capture_resolution(cap, camera_index)
            resolution_pending = True

            print(
                "\n🚀 AI Pose Control System Activated! Please stand in front of the camera."
            )
            print("💡 Use C to change camera and Q to quit.")

            while cap.isOpened():
                ret, raw_frame = _safe_read_frame(cap)
                if not ret or raw_frame is None:
                    print("⚠️ Failed to grab frame from camera.")
                    break

                if resolution_pending:
                    _report_capture_resolution(raw_frame, camera_index)
                    resolution_pending = False

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
                renderer.render_frame(
                    frame,
                    landmarks,
                    current_cmd,
                    camera_index=camera_index,
                    serial_status=serial_client.status_text,
                    serial_connected=serial_client.is_connected,
                )

                _present_frame(self.config.window_title, frame)

                key = cv2.waitKey(5) & 0xFF
                try:
                    if (
                        cv2.getWindowProperty(
                            self.config.window_title, cv2.WND_PROP_VISIBLE
                        )
                        < 1
                    ):
                        break
                except cv2.error:
                    break
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("c"), ord("C")) and len(cameras) > 1:
                    next_camera = _next_camera(cameras, camera_index)
                    new_cap = cv2.VideoCapture(next_camera)
                    if new_cap.isOpened():
                        _configure_capture_resolution(new_cap, next_camera)
                        cap.release()
                        cap = new_cap
                        camera_index = next_camera
                        resolution_pending = True
                        print(f"Switched to camera {camera_index}.")
                    else:
                        new_cap.release()
                        print(f"⚠️ Could not open camera {next_camera}.")


def main(config: ControllerConfig | None = None) -> None:
    """Run the AI Pose Controller application."""
    if config is None:
        config = _select_serial_port(ControllerConfig())
        if config is None:
            return
    app = PoseControllerApp(config=config)
    app.run()


def _list_serial_ports() -> list[ListPortInfo]:
    """Return available COM port devices sorted by device name."""
    ports = sorted(list_ports.comports(), key=lambda port: port.device)
    if ports:
        print("\nAvailable serial devices:")
        for index, port in enumerate(ports, start=1):
            print(f"  {index}. {_format_port_label(port)}")
    else:
        print("\n⚠️ No serial devices found.")
    return ports


def _format_port_label(port: ListPortInfo) -> str:
    """Format a COM port entry with device name and human-readable description."""
    description = (port.description or "").strip()
    if description and description != port.device:
        label = f"{port.device} - {description}"
    else:
        label = port.device
    manufacturer = (port.manufacturer or "").strip()
    if manufacturer and manufacturer.lower() not in label.lower():
        label = f"{label} ({manufacturer})"
    return label


def _default_port_selection(ports: list[ListPortInfo], default_port: str) -> int:
    """Determine the initial highlighted option index."""
    for index, port in enumerate(ports):
        if port.device.upper() == default_port.upper():
            return index
    return 0


def _select_serial_port(config: ControllerConfig) -> ControllerConfig | None:
    """Display an OpenCV startup screen to let the user select a COM port device."""
    ports = _list_serial_ports()
    selected_index = _default_port_selection(ports, config.serial_port)

    row_bounds: list[tuple[int, int, int, int, int]] = []
    button_bounds: dict[str, tuple[int, int, int, int]] = {}
    mouse_action: dict[str, str | None] = {"trigger": None}
    viewport: dict[str, float | int] = {"scale": 1.0, "offset_x": 0, "offset_y": 0}

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal selected_index
        scale = float(viewport["scale"])
        if scale <= 0:
            return
        fx = int((x - int(viewport["offset_x"])) / scale)
        fy = int((y - int(viewport["offset_y"])) / scale)

        if event == cv2.EVENT_MOUSEMOVE:
            for option_idx, left, top, right, bottom in row_bounds:
                if left <= fx <= right and top <= fy <= bottom:
                    selected_index = option_idx
                    break
        elif event == cv2.EVENT_LBUTTONDOWN:
            for option_idx, left, top, right, bottom in row_bounds:
                if left <= fx <= right and top <= fy <= bottom:
                    selected_index = option_idx
                    mouse_action["trigger"] = "confirm"
                    return
            for name, (left, top, right, bottom) in button_bounds.items():
                if left <= fx <= right and top <= fy <= bottom:
                    mouse_action["trigger"] = name
                    return

    try:
        cv2.namedWindow(config.window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_title, SELECTOR_WIDTH, SELECTOR_HEIGHT)
        cv2.setMouseCallback(config.window_title, _on_mouse)
    except cv2.error:
        return _select_serial_port_console(config, ports)

    while True:
        option_count = len(ports) + 1  # ports + No-Hardware Test Mode
        selected_index = max(0, min(selected_index, option_count - 1))

        frame, row_bounds, button_bounds = _render_port_selector_frame(
            ports, selected_index
        )
        scale, offset_x, offset_y = _present_frame(config.window_title, frame)
        viewport["scale"] = scale
        viewport["offset_x"] = offset_x
        viewport["offset_y"] = offset_y

        raw_key = cv2.waitKeyEx(30)
        try:
            if cv2.getWindowProperty(config.window_title, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return None
        except cv2.error:
            return None

        action = mouse_action["trigger"]
        mouse_action["trigger"] = None

        if raw_key in UP_KEY_CODES:
            selected_index = (selected_index - 1) % option_count
            continue
        if raw_key in DOWN_KEY_CODES:
            selected_index = (selected_index + 1) % option_count
            continue

        key = raw_key & 0xFF if 0 <= raw_key < 256 else -1
        if key in (ord("w"), ord("W")):
            selected_index = (selected_index - 1) % option_count
            continue
        if key in (ord("s"), ord("S")):
            selected_index = (selected_index + 1) % option_count
            continue
        if key == ESC_KEY_CODE or key in (ord("q"), ord("Q")) or action == "quit":
            cv2.destroyAllWindows()
            return None
        if key in (ord("r"), ord("R")) or action == "refresh":
            ports = _list_serial_ports()
            selected_index = _default_port_selection(ports, config.serial_port)
            continue
        if key in (ord("0"), ord("n"), ord("N")):
            selected_index = len(ports)
            action = "confirm"
        elif ord("1") <= key <= ord("9"):
            number_index = key - ord("1")
            if number_index < len(ports):
                selected_index = number_index
                action = "confirm"
        elif key in ENTER_KEY_CODES:
            action = "confirm"

        if action == "confirm":
            try:
                cv2.setMouseCallback(config.window_title, lambda *_args: None)
            except cv2.error:
                pass
            if selected_index < len(ports):
                selected_port = ports[selected_index].device
                print(f"Selected serial port: {selected_port}")
                return replace(config, serial_port=selected_port)
            print("Selected: No-Hardware Test Mode")
            return replace(config, serial_port="")


def _render_port_selector_frame(
    ports: list[ListPortInfo],
    selected_index: int,
) -> tuple[
    FrameArray,
    list[tuple[int, int, int, int, int]],
    dict[str, tuple[int, int, int, int]],
]:
    """Render the OpenCV COM port selection screen and interactive hitboxes."""
    frame: FrameArray = np.full(
        (SELECTOR_HEIGHT, SELECTOR_WIDTH, 3), 24, dtype=np.uint8
    )
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Header bar
    cv2.rectangle(frame, (0, 0), (SELECTOR_WIDTH, 88), BLACK_COLOR, -1)
    cv2.putText(
        frame,
        "Select Serial (COM) Device",
        (28, 42),
        font,
        0.9,
        YELLOW_COLOR,
        2,
    )
    cv2.putText(
        frame,
        "Click a device or use Up/Down + Enter  |  1-9: Select  |  R: Refresh  |  Q: Quit",
        (28, 72),
        font,
        0.45,
        GREEN_COLOR,
        1,
    )

    row_bounds: list[tuple[int, int, int, int, int]] = []
    left = 28
    right = SELECTOR_WIDTH - 28
    row_top = 108
    row_height = 46
    row_gap = 8

    if not ports:
        cv2.rectangle(frame, (left, row_top), (right, row_top + 56), (40, 40, 40), -1)
        cv2.putText(
            frame,
            "No COM port devices detected. Connect micro:bit and press R to refresh.",
            (left + 16, row_top + 34),
            font,
            0.52,
            YELLOW_COLOR,
            1,
        )
        row_top += 72
    else:
        start_idx = 0
        if len(ports) > MAX_VISIBLE_PORTS and selected_index < len(ports):
            start_idx = min(
                max(0, selected_index - MAX_VISIBLE_PORTS // 2),
                len(ports) - MAX_VISIBLE_PORTS,
            )
        visible_ports = ports[start_idx : start_idx + MAX_VISIBLE_PORTS]

        for offset, port in enumerate(visible_ports):
            option_idx = start_idx + offset
            top = row_top + offset * (row_height + row_gap)
            bottom = top + row_height
            is_selected = option_idx == selected_index

            bg_color = (45, 95, 45) if is_selected else (42, 42, 42)
            border_color = GREEN_COLOR if is_selected else (85, 85, 85)
            cv2.rectangle(frame, (left, top), (right, bottom), bg_color, -1)
            cv2.rectangle(frame, (left, top), (right, bottom), border_color, 2 if is_selected else 1)

            label = _truncate_text(f"{option_idx + 1}. {_format_port_label(port)}", 72)
            cv2.putText(
                frame,
                label,
                (left + 16, top + 30),
                font,
                0.58,
                YELLOW_COLOR if is_selected else (235, 235, 235),
                2 if is_selected else 1,
            )
            row_bounds.append((option_idx, left, top, right, bottom))

        row_top += len(visible_ports) * (row_height + row_gap) + 6

    # No-Hardware Test Mode option
    no_hw_idx = len(ports)
    no_hw_top = min(row_top, SELECTOR_HEIGHT - 125)
    no_hw_bottom = no_hw_top + row_height
    is_no_hw_selected = selected_index == no_hw_idx
    cv2.rectangle(
        frame,
        (left, no_hw_top),
        (right, no_hw_bottom),
        (45, 95, 45) if is_no_hw_selected else (38, 38, 38),
        -1,
    )
    cv2.rectangle(
        frame,
        (left, no_hw_top),
        (right, no_hw_bottom),
        GREEN_COLOR if is_no_hw_selected else (85, 85, 85),
        2 if is_no_hw_selected else 1,
    )
    cv2.putText(
        frame,
        "0 / N. No-Hardware Test Mode (Continue without COM device)",
        (left + 16, no_hw_top + 30),
        font,
        0.55,
        YELLOW_COLOR if is_no_hw_selected else (210, 210, 210),
        2 if is_no_hw_selected else 1,
    )
    row_bounds.append((no_hw_idx, left, no_hw_top, right, no_hw_bottom))

    # Footer buttons
    button_bounds: dict[str, tuple[int, int, int, int]] = {}
    btn_top = SELECTOR_HEIGHT - 60
    btn_bottom = SELECTOR_HEIGHT - 18
    buttons = (
        ("confirm", "[Enter] Start", left, left + 220, GREEN_COLOR),
        ("refresh", "[R] Refresh Devices", left + 240, left + 490, YELLOW_COLOR),
        ("quit", "[Q] Quit", left + 510, right, (180, 180, 180)),
    )
    for name, text, b_left, b_right, color in buttons:
        cv2.rectangle(frame, (b_left, btn_top), (b_right, btn_bottom), BLACK_COLOR, -1)
        cv2.rectangle(frame, (b_left, btn_top), (b_right, btn_bottom), color, 1)
        cv2.putText(
            frame,
            text,
            (b_left + 18, btn_top + 27),
            font,
            0.52,
            color,
            1,
        )
        button_bounds[name] = (b_left, btn_top, b_right, btn_bottom)

    return frame, row_bounds, button_bounds


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate long device descriptions to fit inside the selector row."""
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _select_serial_port_console(
    config: ControllerConfig,
    ports: list[ListPortInfo],
) -> ControllerConfig:
    """Fallback console prompt when an OpenCV GUI window cannot be created."""
    if not ports:
        print("⚠️ No serial devices found. Continuing in No-Hardware Test Mode.")
        return replace(config, serial_port="")

    print("  0. No-Hardware Test Mode")
    while True:
        choice = input(f"Select a COM port [1-{len(ports)}, or 0 for test mode]: ")
        try:
            selected_number = int(choice)
        except ValueError:
            selected_number = -1

        if selected_number == 0:
            print("Selected: No-Hardware Test Mode")
            return replace(config, serial_port="")
        if 1 <= selected_number <= len(ports):
            selected_port = ports[selected_number - 1].device
            print(f"Selected serial port: {selected_port}")
            return replace(config, serial_port=selected_port)

        print(f"Please enter a number from 0 to {len(ports)}.")


def _present_frame(window_title: str, frame: FrameArray) -> tuple[float, int, int]:
    """Scale and letterbox a frame to fit the current OpenCV window while preserving aspect ratio."""
    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        cv2.imshow(window_title, frame)
        return 1.0, 0, 0

    try:
        _x, _y, win_w, win_h = cv2.getWindowImageRect(window_title)
    except cv2.error:
        win_w, win_h = 0, 0

    if win_w <= 0 or win_h <= 0 or (win_w == src_w and win_h == src_h):
        cv2.imshow(window_title, frame)
        return 1.0, 0, 0

    scale = min(win_w / src_w, win_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    offset_x = (win_w - scaled_w) // 2
    offset_y = (win_h - scaled_h) // 2

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized: FrameArray = np.asarray(
        cv2.resize(frame, (scaled_w, scaled_h), interpolation=interpolation),
        dtype=np.uint8,
    )
    canvas: FrameArray = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = resized
    cv2.imshow(window_title, canvas)
    return scale, offset_x, offset_y


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


def _configure_capture_resolution(camera: cv2.VideoCapture, camera_index: int) -> None:
    """Request 720p once without probing modes or changing the camera's FPS."""
    for prop, value in (
        (cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH),
        (cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT),
    ):
        try:
            accepted = camera.set(prop, float(value))
        except cv2.error as exc:
            print(f"⚠️ Camera {camera_index}: resolution property {prop} failed: {exc}")
        else:
            if not accepted:
                print(
                    f"⚠️ Camera {camera_index}: resolution property {prop} "
                    "was rejected; continuing with the actual resolution."
                )


def _report_capture_resolution(frame: FrameArray, camera_index: int) -> None:
    """Report the actual resolution from the first ordinary frame of each activation."""
    height, width = frame.shape[:2]
    if (width, height) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
        print(
            f"⚠️ Camera {camera_index}: requested {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}, "
            f"received {width}x{height}; continuing at the actual resolution."
        )
    else:
        print(f"Camera {camera_index}: {width}x{height}.")


def _safe_read_frame(
    camera: cv2.VideoCapture,
) -> tuple[bool, FrameArray | None]:
    """Read a frame from the camera, catching OpenCV assertion/buffer errors gracefully."""
    try:
        ret, frame = camera.read()
        if not ret or frame is None:
            return False, None
        return True, np.asarray(frame, dtype=np.uint8)
    except (cv2.error, OSError, RuntimeError) as exc:
        print(f"⚠️ OpenCV frame capture error: {exc}")
        return False, None
