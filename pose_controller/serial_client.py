"""USB Serial communication client for micro:bit (Maqueen Plus V2)."""

import time
from types import TracebackType
from typing import Self

import serial

from .commands import ControlCommand
from .config import ControllerConfig


class SerialCommunicator:
    """Manages micro:bit serial connection and rate-limited command transmission."""

    def __init__(self, config: ControllerConfig) -> None:
        self._config = config
        self._serial: serial.Serial | None = None
        self._last_cmd: str = ""
        self._last_send_time: float = time.monotonic()

    @property
    def is_connected(self) -> bool:
        """Return whether a physical serial connection is active."""
        return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        """Open the micro:bit USB serial port or fall back to No-Hardware Test Mode."""
        try:
            self._serial = serial.Serial(
                self._config.serial_port,
                self._config.baud_rate,
                timeout=self._config.serial_timeout,
            )
            print(
                f"✅ Successfully connected to micro:bit (Port: {self._config.serial_port})"
            )
        except Exception:
            print(
                f"⚠️ Could not connect to serial port {self._config.serial_port}. "
                "Running in 'No-Hardware Test Mode'."
            )
            self._serial = None

    def send(self, command: ControlCommand) -> None:
        """Transmit the command over USB serial if changed or interval elapsed."""
        now = time.monotonic()
        if (command.value != self._last_cmd) or (
            now - self._last_send_time > self._config.send_interval
        ):
            if self._serial is not None:
                try:
                    self._serial.write(f"{command.value}\n".encode("utf-8"))
                except Exception as exc:
                    print(f"⚠️ Error sending data to serial port: {exc}")
            self._last_cmd = command.value
            self._last_send_time = now

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
