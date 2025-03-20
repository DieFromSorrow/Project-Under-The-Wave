import requests
import socket
import json
import time
from datetime import datetime
from utils import running_print


date = None
data_path = "../json/data.json"

with open(data_path, 'r') as f:
    data = json.load(f)


def auto_connect(wait_seconds=0):
    ip = socket.gethostbyname(socket.gethostname())
    url = data['url'].format(data['username'], data['password'], ip)
    response = requests.get(url)
    status_code = response.status_code
    running_print(f'Reconnection status code: {status_code}')
    time.sleep(wait_seconds)


def stable_travel():
    while True:
        for hours in range(24):
            current_time = datetime.now()
            formatted_time = current_time.strftime("%d-%H-%M")
            running_print(f'Now: {formatted_time}, process has been running for {hours} hours.')
            if hours == 23:
                time.sleep(60 * 59)
                for _ in range(60):
                    auto_connect(wait_seconds=1)
            else:
                time.sleep(60 * 60)


def test_stable_travel():
    while True:
        auto_connect()
        time.sleep(1)


if __name__ == "__main__":
    stable_travel()
