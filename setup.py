import os.path
from setuptools import setup, find_packages
from src.utils.constants import Paths

temp_dirs = [Paths.LOGS_DIR, Paths.MODEL_CHECKPOINT_DIR, Paths.STATS_DIR]

for dir in temp_dirs:
    if not os.path.exists(dir):
        print(f'Creating "{dir}" directory')
        os.makedirs(dir)
    else:
        print(f'Directory "{dir}" exists, skipping creation')

setup(name='augment-aid',
      packages=find_packages(where='src'),
      package_dir={'': '.'},
      version='1.0.0')
