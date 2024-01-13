import os
from pathlib import Path
from typing import Final


class Paths:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Final[str] = os.path.join(BASE_DIR, 'data')
    ASSETS_DIR: Final[str] = os.path.join(BASE_DIR, 'assets')
    LOGS_DIR: Final[str] = os.path.join(BASE_DIR, 'logs')
    MODEL_CHECKPOINT_DIR: Final[str] = os.path.join(ASSETS_DIR, 'model_checkpoints')