RESAMPLE_FREQ = "5min"
WINDOW_SIZE = 6

SIGNAL_COLS = [
    "heart_rate",
    "resp_rate",
    "spo2",
    "temperature",
    "sbp",
    "dbp"
]

VALID_RANGES = {
    "heart_rate": (20, 260),
    "resp_rate": (5, 80),
    "spo2": (40, 100),
    "temperature": (30, 43),
    "sbp": (50, 250),
    "dbp": (30, 150)
}
