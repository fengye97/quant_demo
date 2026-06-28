---
name: frontend-screenshot-verify
description: Verify frontend changes in the running quant web app using a real browser and screenshots. Use this skill whenever debugging, reviewing, or confirming frontend fixes so code/API checks are cross-checked against the rendered UI.
---

Use this skill for any frontend debug, review, or fix verification work in `quant`, especially under `stock_trade_demo/web/` and `stock_trade_demo/web_app.py`.

## Rule

Do not declare a frontend issue fixed based only on reading code, template HTML, console output, or API responses. You must cross-check the rendered UI in a real browser and capture at least one screenshot as evidence.

## Required workflow

1. Ensure the Flask app is running at `http://localhost:8080`.
2. Use a real browser automation path, preferring Selenium with the machine-local Chrome + Chromedriver.
3. Drive the page to the surface where the change is visible.
4. Capture screenshots to a temporary directory such as `/tmp/quant_verify/`.
5. If relevant, also inspect console/runtime errors, but screenshots are mandatory.
6. In the final report, cite:
   - the page URL
   - the screenshot path(s)
   - the specific UI element or behavior confirmed by the screenshot

## Recommended Selenium pattern

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os, time

out_dir = '/tmp/quant_verify'
os.makedirs(out_dir, exist_ok=True)

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-gpu')
opts.add_argument('--window-size=1600,1400')
opts.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

driver = webdriver.Chrome(options=opts)
try:
    driver.get('http://localhost:8080/')
    time.sleep(4)
    driver.save_screenshot(os.path.join(out_dir, 'page.png'))
finally:
    driver.quit()
```

## quant-specific reminders

- For the stock selection page, confirm the live strategy list and key rendered text, not just `/api/strategy_list`.
- For the timing page, confirm the three rendered columns, current signal badges, charts, and any control buttons such as index refresh.
- If the UI looks stale, consider cache effects and re-check after restarting the Flask server.
- If browser automation is blocked by environment issues, report BLOCKED rather than claiming PASS from code inspection alone.

## Final report expectation

A valid verification report should mention the screenshot path explicitly, for example:

- `Screenshot: /tmp/quant_verify/timing.png`
- `Confirmed the timing page renders CSI 1000 / 科创50 / 创业板 columns with updated metrics and visible signal badges.`
