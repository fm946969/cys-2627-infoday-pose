"""CYS AI Pose Controller (Maqueen Plus V2) — Python 3.14 Entry Point.

Delegates to the modular `pose_controller` package.
"""

from pose_controller import (
    CommandClassifier,
    ControlCommand,
    ControllerConfig,
    HudRenderer,
    PoseControllerApp,
    PoseDetector,
    SerialCommunicator,
    main,
)

__all__ = [
    "CommandClassifier",
    "ControlCommand",
    "ControllerConfig",
    "HudRenderer",
    "PoseControllerApp",
    "PoseDetector",
    "SerialCommunicator",
    "main",
]


if __name__ == "__main__":
    main()

