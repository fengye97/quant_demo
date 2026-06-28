# Verify Frontend With Screenshot

对 `quant` 项目中的前端改动做运行中验证，并**必须**用真实浏览器截图交叉核验，不能只看代码、模板或 API 返回。

## 执行要求

1. 确认 Flask 服务运行在 `http://localhost:8080`
2. 用 Selenium + Chrome/Chromedriver 打开实际页面
3. 驱动到变更可见的位置
4. 截图保存到 `/tmp/quant_verify/`
5. 必要时再补充 API/HTML/console 证据，但**截图必须存在**
6. 最终结论里必须写出截图路径，以及截图中确认到的 UI 结果

## 重点页面

- 首页选股页：`http://localhost:8080/`
- 择时页：`http://localhost:8080/timing`

## 失败处理

- 如果浏览器自动化环境异常，结论应为 `BLOCKED`
- 不允许在没有截图的情况下用“看代码没问题”或“API 正常”替代前端验证

## 推荐 Selenium 模板

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
    driver.get('http://localhost:8080/timing')
    time.sleep(5)
    driver.save_screenshot(os.path.join(out_dir, 'timing.png'))
finally:
    driver.quit()
```
