from django.db import models
from django.contrib.auth.models import User

class TrainingResult(models.Model):
    """
    記錄每次賽馬娘的育成結果（自動或手動輸入）
    """

    # 基本資料
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="training_results")
    uma_name = models.CharField("賽馬娘名稱", max_length=50)
    title = models.CharField("稱號", max_length=100, blank=True, null=True)
    rank = models.CharField("評級", max_length=10)
    score = models.PositiveIntegerField("分數")
    level = models.PositiveIntegerField("等級", default=1)
    portrait_crop = models.ImageField(upload_to="uma/portrait_crop/", blank=True, null=True)

    # 能力值
    speed = models.PositiveIntegerField(default=0)
    stamina = models.PositiveIntegerField(default=0)
    power = models.PositiveIntegerField(default=0)
    guts = models.PositiveIntegerField(default=0)
    intelligence = models.PositiveIntegerField(default=0)

    # 資質
    terrain = models.CharField("場地資質", max_length=20, blank=True, null=True)
    distance = models.CharField("距離資質", max_length=20, blank=True, null=True)
    strategy = models.CharField("腳質資質", max_length=20, blank=True, null=True)

    # 頭像圖片
    portrait = models.ImageField("頭像圖片", upload_to="uma/portraits/", blank=True, null=True)

    # 技能（用 JSON 儲存）
    unique_skills = models.JSONField(default=list, blank=True)
    gold_skills = models.JSONField(default=list, blank=True)
    white_skills = models.JSONField(default=list, blank=True)
    blue_skills = models.JSONField(default=list, blank=True)
    red_skills = models.JSONField(default=list, blank=True)

    # 時間戳
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    screenshot = models.ImageField("訓練截圖", upload_to="uma/screenshots/", blank=True, null=True)


    
    def total_skills(self):
        """回傳所有技能總數"""
        return sum([
            len(self.unique_skills or []),
            len(self.gold_skills or []),
            len(self.white_skills or []),
            len(self.blue_skills or []),
            len(self.red_skills or []),
        ])

    def __str__(self):
        return f"{self.uma_name} ({self.rank}) - {self.score}"

    class Meta:
        verbose_name = "育成結果"
        verbose_name_plural = "育成結果"
