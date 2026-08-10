"""
BIRE config package exports.

Keeps older imports working:
from bire.config import SIGNAL_COLS, VALID_RANGES, RESAMPLE_FREQ, WINDOW_SIZE
"""

from bire.config.settings import *  # noqa: F403


SIGNAL_COLS = [
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "sbp",
    "dbp",
]

VALID_RANGES = {
    "heart_rate": (25, 250),
    "resp_rate": (3, 80),
    "spo2": (50, 100),
    "temperature": (30, 43),
    "sbp": (40, 260),
    "dbp": (20, 160),
}

RESAMPLE_FREQ = "5min"

WINDOW_SIZE = 6