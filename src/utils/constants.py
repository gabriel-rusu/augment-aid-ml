import os
from pathlib import Path
from typing import Final


class Paths:
    BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Final[str] = os.path.join(BASE_DIR, 'data')
    ASSETS_DIR: Final[str] = os.path.join(BASE_DIR, 'assets')
    LOGS_DIR: Final[str] = os.path.join(BASE_DIR, 'logs')
    MODEL_CHECKPOINT_DIR: Final[str] = os.path.join(ASSETS_DIR, 'model_checkpoints')
    GAN_WRAPPER_CHECKPOINT_FILE_PATH: Final[str] = os.path.join(MODEL_CHECKPOINT_DIR, 'gan-wrapper.ckpt')
    VAE_WRAPPER_CHECKPOINT_FILE_PATH: Final[str] = os.path.join(MODEL_CHECKPOINT_DIR, 'vae-wrapper.ckpt')


    STATS_DIR: Final[str] = os.path.join(ASSETS_DIR, 'stats')
    STATISTICS_CSV: Final[str] = os.path.join(STATS_DIR, 'run-version_{}-tag-{}_{}.csv')