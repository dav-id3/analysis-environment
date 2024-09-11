import time
import os

from selenium import webdriver
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import const
from datetime import datetime


def fetch_url(start_year: int, start_month: int):
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, 10)

    # 月ごとに検索
    today_month = datetime.today().month
    today_year = datetime.today().year

    for year in range(start_year, today_year + 1):
        for month in range(start_month, 13):
            if year == today_year and month > today_month:
                break
            driver.get("https://db.netkeiba.com/?pid=race_search_detail")
            time.sleep(1)
            wait.until(EC.presence_of_all_elements_located)

            # 期間を選択
            start_year_element = driver.find_element(By.NAME, "start_year")
            start_year_select = Select(start_year_element)
            start_year_select.select_by_value(str(year))
            start_mon_element = driver.find_element(By.NAME, "start_mon")
            start_mon_select = Select(start_mon_element)
            start_mon_select.select_by_value(str(month))
            end_year_element = driver.find_element(By.NAME, "end_year")
            end_year_select = Select(end_year_element)
            end_year_select.select_by_value(str(year))
            end_mon_element = driver.find_element(By.NAME, "end_mon")
            end_mon_select = Select(end_mon_element)
            end_mon_select.select_by_value(str(month))

            # 中央競馬場をチェック
            for i in range(1, 11):
                terms = driver.find_element(By.ID, f"check_Jyo_{str(i).zfill(2)}")
                terms.click()

            # 表示件数を選択(20,50,100の中から最大の100へ)
            list_element = driver.find_element(By.NAME, "list")
            list_select = Select(list_element)
            list_select.select_by_value("100")

            # フォームを送信
            frm = driver.find_element(By.CSS_SELECTOR, "#db_search_detail_form > form")
            frm.submit()
            time.sleep(5)
            wait.until(EC.presence_of_all_elements_located)

            # fetch url
            filename = os.path.join(
                const.curpath, const.url_loc, str(year) + "-" + str(month) + ".txt"
            )
            if os.path.isfile(filename):
                os.remove(filename)

            with open(filename, mode="w") as f:
                while True:
                    time.sleep(5)

                    wait.until(EC.presence_of_all_elements_located)
                    all_rows = driver.find_element(
                        By.CLASS_NAME, "race_table_01"
                    ).find_elements(By.TAG_NAME, "tr")
                    for row in range(1, len(all_rows)):
                        race_href = (
                            all_rows[row]
                            .find_elements(By.TAG_NAME, "td")[4]
                            .find_element(By.TAG_NAME, "a")
                            .get_attribute("href")
                        )
                        f.write(race_href + "\n")
                    try:
                        target = driver.find_elements(By.LINK_TEXT, "次")[0]
                        driver.execute_script(
                            "arguments[0].click();", target
                        )  # javascriptでクリック処理
                    except IndexError:
                        break
