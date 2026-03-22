from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

os.makedirs("test_data", exist_ok=True)

def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def scrape(url, filename):
    print(f"Scraping: {url}")
    driver = get_driver()
    try:
        driver.get(url)
        time.sleep(4)
        # Try clicking expandable tabs / accordions
        clickable_texts = [
            "Սակագներ",
            "Փաստաթղթեր",
            "Հարց-պատասխան",
            "Ընդլայնել բոլորը",
            "Հիմնական"
        ]

        for label in clickable_texts:
            try:
                elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{label}')]")
                for el in elements:
                    try:
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
                        time.sleep(0.5)
                        driver.execute_script("arguments[0].click();", el)
                        time.sleep(1)
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            buttons = driver.find_elements(By.XPATH, "//*[@aria-expanded='false']")
            for btn in buttons[:20]:
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                    time.sleep(0.3)
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.8)
                except Exception:
                    pass
        except Exception:
            pass

        # Scroll down slowly to trigger lazy loading
        for i in range(8):
            driver.execute_script(f"window.scrollTo(0, {i * 800});")
            time.sleep(0.5)

        time.sleep(2)
        text = driver.find_element(By.TAG_NAME, "body").text
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)

        with open(f"test_data/{filename}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {len(text)} chars -> test_data/{filename}.txt")
    except Exception as e:
        print(f"Failed {url}: {e}")
    finally:
        driver.quit()
    time.sleep(2)


# AMERIABANK
scrape("https://ameriabank.am/personal/saving/deposits/ameria-deposit", "ameriabank_deposits")
scrape("https://ameriabank.am/personal/loans/consumer-loans/credit-line", "ameriabank_credits")
scrape("https://ameriabank.am/service-network", "ameriabank_branches")

# ARDSHINBANK
scrape("https://ardshinbank.am/for-you/avand?lang=hy", "ardshinbank_deposits")
scrape("https://ardshinbank.am/for-you/loans-ardshinbank?lang=hy", "ardshinbank_credits")
scrape("https://ardshinbank.am/Information/branch-atm?lang=hy", "ardshinbank_branches")

# IDBANK
scrape("https://idbank.am/deposits/", "idbank_deposits")
scrape("https://idbank.am/credits/", "idbank_credits")
scrape("https://idbank.am/information/about/branches-and-atms/", "idbank_branches")\


print("\nAll done!")