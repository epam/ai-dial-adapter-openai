import os


def pytest_configure(config):
    """
    Setting up fake environment variables for unit tests.
    """
    os.environ["DIAL_URL"] = "dummy_url"
