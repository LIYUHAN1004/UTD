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
