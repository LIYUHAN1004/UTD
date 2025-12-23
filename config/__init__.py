# 讓 Django 啟動時自動載入 Celery
from config.celery import app as celery_app

__all__ = ('celery_app',)

