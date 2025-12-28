from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path("", lambda request: redirect("/uma/upload/")),  # 👈 新增首頁
    path("admin/", admin.site.urls),
    path("uma/", include("apps.uma.urls")),
]

# 僅在 DEBUG=True 時提供 media（本機用）
if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )
