from openai import Timeout

# connect timeout and total timeout
DEFAULT_TIMEOUT = Timeout(600, connect=10)
