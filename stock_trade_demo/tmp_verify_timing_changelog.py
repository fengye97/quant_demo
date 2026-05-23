from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os, time

out_dir = '/tmp/quant_verify'
os.makedirs(out_dir, exist_ok=True)

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-gpu')
opts.add_argument('--window-size=2200,3200')
opts.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

driver = webdriver.Chrome(options=opts)
try:
    driver.get('http://localhost:8080/timing')
    time.sleep(8)
    section = driver.find_element(By.ID, 'changelog-section')
    driver.execute_script('arguments[0].scrollIntoView({block: "center"});', section)
    time.sleep(1)
    out = os.path.join(out_dir, 'timing_changelog_section.png')
    driver.save_screenshot(out)
    print(out)
    print(section.text)
finally:
    driver.quit()
