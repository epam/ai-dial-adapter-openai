import time


class Timer:
    _start: float

    def __init__(self):
        self._start = time.perf_counter()

    def get_elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start
