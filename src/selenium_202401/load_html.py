import os
import requests
import time
import const


def fetch_html(year: str, month: str):
    save_dir = os.path.join(const.html_loc, str(year), str(month))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(str(year) + "-" + str(month) + ".txt", "r") as f:
        urls = f.read().splitlines()
        for url in urls:
            list = url.split("/")
            race_id = list[-2]
            save_file_path = os.path.join(save_dir, race_id + ".html")
            response = requests.get(url)
            response.encoding = response.apparent_encoding
            html = response.text
            time.sleep(5)
            with open(save_file_path, "w") as file:
                file.write(html)
