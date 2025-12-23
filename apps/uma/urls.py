# apps/uma/urls.py
from django.urls import path
from . import views

app_name = "uma"

urlpatterns = [
    path("upload/", views.upload_training_results, name="upload_training_results"),
    path("bulk-update/", views.bulk_update_results, name="bulk_update_results"),
    path("result/<int:pk>/", views.result_detail, name="result_detail"),

]
