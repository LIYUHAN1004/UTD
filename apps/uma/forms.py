# apps/uma/forms.py
from __future__ import annotations

from django import forms


class MultiFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultiImageField(forms.FileField):
    """
    讓 Django Form 正確接受 request.FILES.getlist("images")
    """
    widget = MultiFileInput

    def clean(self, data, initial=None):
        # data 可能是單個 UploadedFile，也可能是 list
        if data is None:
            data = []
        if not isinstance(data, (list, tuple)):
            data = [data]

        cleaned = []
        for f in data:
            cleaned.append(super().clean(f, initial))
        return cleaned


class TrainingImageUploadForm(forms.Form):
    images = MultiImageField(
        required=True,
        widget=MultiFileInput(attrs={"accept": "image/*", "multiple": True}),
        label="上傳圖片",
    )




# 這些 key 會同時用來找 static 圖檔檔名（key.png）
UMA_CHOICES = [
    ("portrait_101.jpg", "大和赤驥"),
    ("portrait_102.jpg", "舞會第一紅寶"),
    ("portrait_103.jpg", "里見皇冠"),
    ("portrait_104.jpg", "目白多伯"),
]

class ManualTrainingResultForm(forms.Form):
    # 由「點頭像」寫進 hidden input
    uma_key = forms.ChoiceField(choices=UMA_CHOICES, required=True)

    # 分數 + 五維
    score = forms.IntegerField(min_value=0, required=True)
    speed = forms.IntegerField(min_value=0, required=True)
    stamina = forms.IntegerField(min_value=0, required=True)
    power = forms.IntegerField(min_value=0, required=True)
    guts = forms.IntegerField(min_value=0, required=True)
    intelligence = forms.IntegerField(min_value=0, required=True)

    # 三資質：由按鈕寫進 hidden input（允許空白=未選）
    terrain = forms.CharField(required=False)
    distance = forms.CharField(required=False)
    strategy = forms.CharField(required=False)
