"""
Redis 基本操作示範
"""
import redis

# 建立 Redis 連線
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
# 建立 Redis 連線（有密碼）
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, password = "你的密碼")

try:
    # ==========================================
    # 1. String 操作
    # ==========================================
    print("=== String 操作 ===")

    # 設定值
    r.set('name', '圖書管理系統')
    r.set('version', '1.0')

    # 取得值
    print(f"name: {r.get('name')}")
    print(f"version: {r.get('version')}")

    # 設定過期時間（5 秒後自動刪除）
    r.setex('temp_data', 5, '這是暫存資料')
    print(f"temp_data: {r.get('temp_data')}")

    # ==========================================
    # 2. Hash 操作（類似 Python dict）
    # ==========================================
    print("\n=== Hash 操作 ===")

    # 儲存書籍資料
    r.hset('book:1', mapping={
        'title': 'Python 入門',
        'price': '350',
        'stock': '10'
    })

    # 取得單一欄位
    print(f"書名: {r.hget('book:1', 'title')}")

    # 取得所有欄位
    book_data = r.hgetall('book:1')
    print(f"完整書籍資料: {book_data}")

    # ==========================================
    # 3. List 操作
    # ==========================================
    print("\n=== List 操作 ===")

    # 清除舊資料
    r.delete('recent_books')

    # 新增到列表
    r.lpush('recent_books', '書籍A', '書籍B', '書籍C')

    # 取得列表內容
    books = r.lrange('recent_books', 0, -1)
    print(f"最近書籍: {books}")

    # ==========================================
    # 4. 清理測試資料
    # ==========================================
    print("\n=== 清理資料 ===")
    r.delete('name', 'version', 'book:1', 'recent_books')
    print("測試資料已清理")

except KeyboardInterrupt:
    print("\n\n程式被使用者中斷（Ctrl+C）")
    print("正在清理測試資料...")
    try:
        r.delete('name', 'version', 'book:1', 'recent_books', 'temp_data')
        print("測試資料已清理")
    except:
        pass
    exit(0)
except Exception as e:
    print(f"\n發生錯誤：{e}")
    exit(1)
