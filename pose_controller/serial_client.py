"""USB Serial communication client for micro:bit (Maqueen Plus V2)."""

import time
from types import TracebackType
from typing import Self

import serial
from serial.tools import list_ports

from .commands import ControlCommand
from .config import ControllerConfig


class SerialCommunicator:
    """Manages micro:bit serial connection and rate-limited command transmission."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._serial: serial.Serial | None = None
        self._last_cmd: str = ""
        self._last_send_time: float = time.monotonic()
        self._next_port_scan_at: float = 0.0

    @property
    def is_connected(self) -> bool:
        """Return whether a physical serial connection is active."""
        return self._serial is not None and self._serial.is_open

    @property
    def status_text(self) -> str:
        """Return a concise human-readable serial connection status for the HUD."""
        if not self._config.serial_port:
            return "Test Mode (No COM)"
        if self.is_connected:
            return f"{self._config.serial_port} (Connected)"
        return f"{self._config.serial_port} (Disconnected)"

    def connect(self) -> None:
        """Open the micro:bit USB serial port or fall back to No-Hardware Test Mode."""
        if not self._config.serial_port:
            print("⚠️ No serial port selected. Running in 'No-Hardware Test Mode'.")
            self._serial = None
            return

        try:
            self._serial = serial.Serial(
                self._config.serial_port,
                self._config.baud_rate,
                timeout=self._config.serial_timeout,
            )
            print(
                f"✅ Successfully connected to micro:bit (Port: {self._config.serial_port})"
            )
            self._next_port_scan_at = time.monotonic() + 1.0
        except (serial.SerialException, OSError):
            print(
                f"⚠️ Could not connect to serial port {self._config.serial_port}. "
                "Running in 'No-Hardware Test Mode'."
            )
            self._serial = None
            self._next_port_scan_at = time.monotonic() + 1.0

    def send(self, command: ControlCommand) -> None:
        """Transmit the command over USB serial if changed or interval elapsed."""
        now = time.monotonic()
        self._maintain_connection(now)
        if (command.value != self._last_cmd) or (
            now - self._last_send_time > self._config.send_interval
        ):
            if self._serial is not None:
                try:
                    self._serial.write(f"{command.value}\n".encode("utf-8"))
                except (serial.SerialException, OSError) as exc:
                    self._mark_disconnected(now, exc)
            self._last_cmd = command.value
            self._last_send_time = now

    def _maintain_connection(self, now: float) -> None:
        """Check the selected port and reconnect no more than once per second."""
        if not self._config.serial_port or now < self._next_port_scan_at:
            return
        self._next_port_scan_at = now + 1.0
        port_available = any(
            port.device == self._config.serial_port for port in list_ports.comports()
        )

        if self._serial is not None and (
            not self._serial.is_open or not port_available
        ):
            self._mark_disconnected(now)

        if self._serial is None and port_available:
            try:
                self._serial = serial.Serial(
                    self._config.serial_port,
                    self._config.baud_rate,
                    timeout=self._config.serial_timeout,
                )
            except (serial.SerialException, OSError):
                return

            self._last_cmd = ""
            print(
                f"✅ Serial connection restored (Port: {self._config.serial_port})"
            )

    def _mark_disconnected(
        self, now: float, error: serial.SerialException | OSError | None = None
    ) -> None:
        """Close a lost connection and schedule the next scan."""
        if self._serial is None:
            return

        try:
            self._serial.close()
        except (serial.SerialException, OSError) as close_error:
            print(f"⚠️ Error closing disconnected serial port: {close_error}")
        finally:
            self._serial = None

        message = (
            f"⚠️ Serial connection lost on {self._config.serial_port}. "
            "Retrying every second."
        )
        if error is not None:
            message += f" Error: {error}"
        print(message)
        self._next_port_scan_at = now + 1.0

    def close(self) -> None:
        """Safely close the serial port if open."""
        if self._serial is not None:
            try:
                self._serial.close()
                print("🔌 Serial port closed safely. System shutting down.")
            finally:
                self._serial = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
