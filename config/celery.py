"""
Celery 設定檔

這個檔案設定 Celery 與 Django 的整合
"""
import os
from celery import Celery

# 設定 Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')

# 建立 Celery app
app = Celery('config')

# 從 Django settings 讀取 CELERY_ 開頭的設定
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自動發現所有 Django app 中的 tasks.py
app.autodiscover_tasks()


@app.task(bind=True, ignore_result=True)
def debug_task(self):
    """測試用的任務"""
    print(f'Request: {self.request!r}')
