import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os
import const

from typing import Tuple
from datetime import datetime

race_data_columns = [
    "race_id",
    "race_date",
    "race_place",
    "race_number",
    "race_condition",
    "race_cource",
    "weather",
    "race_type",
    "track_condition",
]
horse_data_columns = [
    "race_id",  # レースIDを関連付けます
    "horse_number",
    "horse_name",
    "age",  # '牝3'のような形式から'3'を抽出します
    "sex",  # '牝3'のような形式から'牝'を抽出します
    "jockey",
    "finish_time",
    "finish_position",
    "odds",
    "popularity",
    "horse_weight",
]


def request_data(
    start_year: int,
    start_month: int,
    end_year: int = datetime.today().year,
    end_month: int = datetime.today().month,
):
    for year in range(start_year, end_year + 1):
        for month in range(start_month, 13):
            if year == end_year and month > end_month:
                break
            exploit_race_data(year, month)


def exploit_race_data(year: int, month: int):
    save_race_csv = const.csv_loc + "/race-" + str(year) + "-" + str(month) + ".csv"
    save_horse_csv = const.csv_loc + "/horse-" + str(year) + "-" + str(month) + ".csv"

    if os.path.isfile(save_race_csv):
        os.remove(save_race_csv)
    if os.path.isfile(save_horse_csv):
        os.remove(save_horse_csv)

    # race_data_columns, horse_data_columnsは長くなるので省略
    race_df = pd.DataFrame(columns=race_data_columns)
    horse_df = pd.DataFrame(columns=horse_data_columns)

    html_dir = os.path.join(const.html_loc, str(year), str(month))

    print(html_dir)
    if os.path.isdir(html_dir):
        file_list = os.listdir(html_dir)
        for file_name in file_list:
            with open(html_dir + "/" + file_name, "r") as f:
                html = f.read()
                list = file_name.split(".")
                race_id = list[-2]
                race_list, horse_list_list = get_rade_and_horse_data_by_html(
                    race_id, html
                )
                for horse_list in horse_list_list:
                    horse_se = pd.Series(horse_list.values(), index=horse_df.columns)
                    horse_df.append(horse_se, ignore_index=True)
                race_se = pd.Series(race_list.values(), index=race_df.columns)
                race_df.append(race_se, ignore_index=True)

                raise
    else:
        print(f"no html directory for {year}-{month}")

    race_df.to_csv(save_race_csv, header=True, index=False)
    horse_df.to_csv(save_horse_csv, header=True, index=False)


def get_rade_and_horse_data_by_html(race_id: str, html: str) -> Tuple[dict, list[dict]]:
    # BeautifulSoupオブジェクトを作成します
    soup = BeautifulSoup(html, "html.parser")

    # レースデータを抽出します
    race_head = soup.find("div", class_="race_head_inner")
    race_data = {
        "race_id": race_id,
        "race_date": race_head.find("p", class_="smalltxt")
        .get_text(strip=True)
        .split()[0],
        "race_place": race_head.find("ul", class_="race_place").a.get_text(strip=True),
        "race_number": race_head.find("dt").get_text(strip=True),
        "race_condition": race_head.find("span").get_text(strip=True),
        "race_cource": race_head.find("span")
        .get_text(strip=True)
        .split("/")[0]
        .strip(),
        "weather": race_head.find("span")
        .get_text(strip=True)
        .split("/")[1]
        .split(":")[1]
        .strip(),
        "race_type": race_head.find("h1").get_text(strip=True),
        "track_condition": race_head.find("span")
        .get_text(strip=True)
        .split("/")[2]
        .split(":")[1]
        .strip(),
    }
    print(race_data)
    raise

    # 馬のデータを抽出します
    results_table = soup.find("table", class_="race_table_01 nk_tb_common")
    rows = results_table.find_all("tr")[1:]  # 最初の行はヘッダーなのでスキップします
    horses_data = []

    for row in rows:
        columns = row.find_all("td")
        horse_data = {
            "race_id": race_data["race_id"],  # レースIDを関連付けます
            "horse_number": columns[2].get_text(strip=True),
            "horse_name": columns[3].get_text(strip=True),
            "age": columns[4].get_text(strip=True)[
                1:
            ],  # '牝3'のような形式から'3'を抽出します
            "sex": columns[4].get_text(strip=True)[
                0
            ],  # '牝3'のような形式から'牝'を抽出します
            "jockey": columns[6].get_text(strip=True),
            "finish_time": columns[7].get_text(strip=True),
            "finish_position": columns[0].get_text(strip=True),
            "odds": columns[12].get_text(strip=True),
            "popularity": columns[13].get_text(strip=True),
            "horse_weight": columns[14].get_text(strip=True),
        }
        horses_data.append(horse_data)

    # 結果を表示
    print("Race Data:", race_data)
    print("\nHorse Data:")
    for horse in horses_data:
        print(horse)
    return race_data, horses_data
