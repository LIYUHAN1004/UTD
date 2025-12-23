import requests
from requests.exceptions import Timeout, RequestException

try:
    response = requests.get(
        'https://jsonplaceholder.typicode.com/users',
        timeout=5  # 5 秒超時
    )
    response.raise_for_status()  # 如果狀態碼不是 2xx，會拋出異常

    users = response.json()
    print(f"成功取得 {len(users)} 位使用者")

except Timeout:
    print("請求超時！")
except RequestException as e:
    print(f"請求發生錯誤：{e}")
