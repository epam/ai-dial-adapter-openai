from contextlib import ContextDecorator
from time import perf_counter


class Timer(ContextDecorator):
    def __init__(self, msg: str, verbose: bool = False):
        self.msg = msg
        self.verbose = verbose
        self.time = -1

    def __enter__(self):
        print(f"Starting {self.msg.lower()}...")
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        elapsed = perf_counter() - self.time
        self.time = elapsed
        if self.verbose:
            print(f"{self.msg} took {elapsed:.3f} seconds")
