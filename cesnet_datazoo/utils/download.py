import os

import requests
from tqdm import tqdm


def simple_download(url: str, file_path: str):
    r = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(r.content)

def resumable_download(url: str, file_path: str, chunk_size: int = 1024**2, silent: bool = False):
    r1 = requests.get(url, stream=True)
    redirected_url = r1.url
    content_size = int(r1.headers["Content-Length"])

    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)
    else:
        temp_size = 0
    if temp_size == content_size:
        return

    headers = {"Range": f"bytes={temp_size}-"}
    r2 = requests.get(redirected_url, stream=True, headers=headers)
    if not silent:
        print(f"File size: {content_size / (1024**3):0.2f}GB")
        print(f"Remaining: {(content_size - temp_size) / (1024**3):0.2f}GB")

    progress_bar = tqdm(total=content_size - temp_size, unit="B", unit_scale=True, unit_divisor=1024, disable=silent)
    with open(file_path, "ab") as file:
        for data in r2.iter_content(chunk_size=chunk_size):
            file.write(data)
            progress_bar.update(len(data))
    progress_bar.close()
