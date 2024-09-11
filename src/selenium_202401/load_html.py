import os
import requests
import time
import const
from datetime import datetime


def fetch_html(year: int, month: int):
    save_dir = os.path.join(const.html_loc, str(year), str(month))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    url_filename = os.path.join(const.url_loc, str(year) + "-" + str(month) + ".txt")
    with open(url_filename, "r") as f:
        urls = f.read().splitlines()
        for url in urls:
            list = url.split("/")
            race_id = list[-2]
            save_file_path = os.path.join(save_dir, race_id + ".html")

            if os.path.isfile(save_file_path):
                continue

            response = requests.get(url)
            response.encoding = response.apparent_encoding
            html = response.text
            time.sleep(5)
            # if os.path.isfile(save_file_path):
            #     os.remove(save_file_path)
            with open(save_file_path, "w") as file:
                file.write(html)


def request_html(
    start_year: int,
    start_month: int,
    end_year: int = datetime.today().year,
    end_month: int = datetime.today().month,
):
    for year in range(start_year, end_year + 1):
        for month in range(start_month, 13):
            if year == end_year and month > end_month:
                break
            fetch_html(year, month)
