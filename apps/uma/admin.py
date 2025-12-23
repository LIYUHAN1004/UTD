from django.contrib import admin
from .models import TrainingResult


@admin.register(TrainingResult)
class TrainingResultAdmin(admin.ModelAdmin):
    list_display = ("uma_name", "rank", "score", "terrain", "distance", "strategy", "total_skills", "created_at")
    search_fields = ("uma_name", "rank", "title")
    list_filter = ("rank", "terrain", "distance", "strategy", "created_at")
    readonly_fields = ("created_at", "updated_at")
