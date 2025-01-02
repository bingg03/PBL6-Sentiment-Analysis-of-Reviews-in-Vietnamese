import random
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime

chrome_driver_path = r'C:\chromedriver-win64\chromedriver.exe'

def crawl_comments(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--start-maximized")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    )

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    # Kiểm tra URL
    if not url.startswith("http"):
        print("URL không hợp lệ:", url)
        driver.quit()
        return pd.DataFrame()

    # Truy cập URL
    driver.get(url)
    time.sleep(3)  # Đợi để kiểm tra tải trang
    if driver.current_url == "data:,":
        print("URL không tải được, thử refresh lại...")
        driver.refresh()
        time.sleep(2)

    try:
        WebDriverWait(driver, random.uniform(21, 25)).until(EC.presence_of_element_located((By.CLASS_NAME, 'item')))
    except Exception as e:
        print(f"Lỗi khi tải trang: {e}")
        driver.quit()
        return pd.DataFrame()

    all_comments = []

    for page in range(3):
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            WebDriverWait(driver, random.uniform(10, 14)).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'item')))
            comment_elements = driver.find_elements(By.CLASS_NAME, 'item')
            all_comments.extend([comment.text for comment in comment_elements])

            try:
                next_button = WebDriverWait(driver, random.uniform(10, 12)).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="module_product_review"]/div/div/div[3]/div[2]/div/button[2]/i'))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(random.uniform(2, 4))
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(random.uniform(3, 5))
            except Exception as e:
                print(f"Không tìm thấy nút 'Next': {e}. Dừng lại ở trang {page + 1}.")
                break
        except Exception as e:
            print(f"Lỗi khi xử lý trang {page + 1}: {e}")
            break

    driver.quit()

    data = []
    for comment in all_comments:
        split_comment = comment.split('\n')
        if len(split_comment) < 4:
            continue
        try:
            date = split_comment[0]
            name = split_comment[1]
            liked = split_comment[-1]
            product = split_comment[-2]
            comment_text = "\n".join(split_comment[2:-2])
        except IndexError:
            continue
        data.append([date, name, comment_text, product, liked])

    df = pd.DataFrame(data, columns=['Date', 'Name', 'Comment', 'Product', 'Liked'])

     # Tạo tên file CSV duy nhất
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Thời gian hiện tại
    file_name = f"comments_{timestamp}.csv"

    # Lưu DataFrame thành file CSV
    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"Dữ liệu đã được lưu vào file: {file_name}")

    return df, file_name