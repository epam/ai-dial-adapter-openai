import time

import requests


def ping_server(url: str) -> bool:
    healthcheck_url = f"{url}/healthcheck"

    try:
        response = requests.get(healthcheck_url)
        return response.status_code == 200 and response.text == "OK"
    except requests.ConnectionError:
        return False


def wait_for_server(url: str, timeout=10) -> bool:
    start_time = time.time()

    while True:
        if ping_server(url):
            return True

        if time.time() - start_time > timeout:
            return False

        time.sleep(0.1)
