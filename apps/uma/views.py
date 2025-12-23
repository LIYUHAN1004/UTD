# apps/uma/views.py
from __future__ import annotations

import io
import os
import re

import cv2
import numpy as np
from PIL import Image
import pytesseract

from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.utils import timezone
from .forms import TrainingImageUploadForm
from .models import TrainingResult
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from .models import TrainingResult

# ====================================================
# Windows Tesseract 路徑
# ====================================================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ====================================================
# 你提供的截圖基準尺寸：858 x 1907
# ====================================================
BASE_W = 858
BASE_H = 1907

# ✅ 開著：會輸出裁切/二值化圖片到 MEDIA_ROOT/ocr_debug/
DEBUG_OCR = False

# ====================================================
# ROI（用指定座標）
# ====================================================
STAT_BOXES = {
    "speed": (15, 610, 200, 685),
    "stamina": (190, 610, 340, 685),
    "power": (350, 610, 520, 685),
    "guts": (510, 610, 660, 685),
    "intelligence": (640, 610, 840, 685),
}

# ✅ 你確認合適的頭像座標
PORTRAIT_BOX = (80, 350, 260, 575)

# ✅ 分數座標：你之後如果要微調，就改這一個就好
# 建議先讓它「只包含數字那個白色膠囊區」最穩
SCORE_BOX = (165, 545, 445, 620)


# ====================================================
# Helpers
# ====================================================
def _scale_box(w: int, h: int, x1: int, y1: int, x2: int, y2: int):
    sx = w / BASE_W
    sy = h / BASE_H
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _save_debug_images(prefix: str, bgr: np.ndarray, th: np.ndarray | None = None):
    if not DEBUG_OCR:
        return
    out_dir = os.path.join(getattr(settings, "MEDIA_ROOT", os.getcwd()), "ocr_debug")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}.png"), bgr)
    if th is not None:
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_th.png"), th)


def _prep_digits_cv(crop_bgr: np.ndarray) -> np.ndarray:
    crop_bgr = cv2.resize(crop_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def _ocr_digits_cv(th_img: np.ndarray) -> str:
    cfg7 = r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    cfg8 = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"

    text = pytesseract.image_to_string(th_img, config=cfg7).strip()
    if re.search(r"\d", text):
        return text
    return pytesseract.image_to_string(th_img, config=cfg8).strip()


def _parse_stat_int(val) -> int:
    """能力值：優先抓 1000~2999，其次 100~999"""
    if not val:
        return 0
    nums = re.findall(r"\d+", str(val))
    if not nums:
        return 0
    ints = [int(n) for n in nums]

    c4 = [n for n in ints if 1000 <= n <= 2999]
    if c4:
        return max(c4)
    c3 = [n for n in ints if 100 <= n <= 999]
    if c3:
        return max(c3)
    return 0


def _parse_score_int(val) -> int:
    """
    ✅ 分數：回到「至少 4 位數才算」避免抓到小數字/雜訊
    """
    if not val:
        return 0
    digits = re.sub(r"\D+", "", str(val))
    if len(digits) < 4:
        return 0
    try:
        return int(digits)
    except Exception:
        return 0


def _pick_best_value(candidates: list[tuple[float, int, float, str]]) -> int:
    if not candidates:
        return 0
    candidates_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)
    return candidates_sorted[0][1]


def _ocr_stat_value(key: str, cell_bgr: np.ndarray) -> int:
    _, w2 = cell_bgr.shape[:2]
    left_cuts = [0.42, 0.36, 0.30, 0.24, 0.18]
    candidates: list[tuple[float, int, float, str]] = []

    for cut in left_cuts:
        lx = int(w2 * cut)
        digit_bgr = cell_bgr[:, lx:w2]

        th = _prep_digits_cv(digit_bgr)
        text = _ocr_digits_cv(th)
        val = _parse_stat_int(text)

        score = 0.0
        if 1000 <= val <= 2999:
            score += 100.0
        elif 100 <= val <= 999:
            score += 60.0
        elif val > 0:
            score += 10.0

        score += cut * 5.0
        digit_count = len(re.findall(r"\d", text))
        score += min(digit_count, 4) * 2.0

        candidates.append((cut, val, score, text))

        if DEBUG_OCR:
            _save_debug_images(f"{key}_cut{int(cut*100):02d}_digit", digit_bgr, th)

    return _pick_best_value(candidates)


def _ocr_score_value(cell_bgr: np.ndarray) -> int:
    """
    ✅ 分數：跟能力值一樣做 left_cut 掃描，但用「至少 4 位數」判斷
    """
    _, w2 = cell_bgr.shape[:2]
    left_cuts = [0.60, 0.52, 0.44, 0.36, 0.28, 0.20]
    candidates: list[tuple[float, int, float, str]] = []

    for cut in left_cuts:
        lx = int(w2 * cut)
        digit_bgr = cell_bgr[:, lx:w2]

        th = _prep_digits_cv(digit_bgr)
        text = _ocr_digits_cv(th)
        val = _parse_score_int(text)

        score = 0.0
        digit_count = len(re.findall(r"\d", text))
        score += min(digit_count, 7) * 5.0
        score += cut * 3.0
        if val > 0:
            score += 10.0

        candidates.append((cut, val, score, text))

        if DEBUG_OCR:
            _save_debug_images(f"score_cut{int(cut*100):02d}_digit", digit_bgr, th)

    return _pick_best_value(candidates)


def analyze_mobile_image(image: Image.Image) -> dict:
    """
    只抓：分數 + 五維能力
    馬名/稱號/評級不做 OCR（之後完全手動 or 用頭像替代顯示）
    """
    w, h = image.size
    img_bgr = _pil_to_bgr(image)

    data: dict = {}

    # --- 分數（指定座標） ---
    sx1, sy1, sx2, sy2 = _scale_box(w, h, *SCORE_BOX)
    score_cell = img_bgr[sy1:sy2, sx1:sx2]
    score_val = _ocr_score_value(score_cell)

    if DEBUG_OCR:
        _save_debug_images("score_cell", score_cell, None)

    data["score"] = int(score_val or 0)

    # --- 能力值（指定座標） ---
    for key, (x1, y1, x2, y2) in STAT_BOXES.items():
        bx1, by1, bx2, by2 = _scale_box(w, h, x1, y1, x2, y2)
        cell_bgr = img_bgr[by1:by2, bx1:bx2]
        val = _ocr_stat_value(key, cell_bgr)
        data[key] = int(val or 0)

        if DEBUG_OCR:
            _save_debug_images(f"{key}_cell", cell_bgr, None)

    return data


def crop_portrait(image: Image.Image) -> Image.Image:
    """裁切頭像（用你確認的 PORTRAIT_BOX）"""
    w, h = image.size
    x1, y1, x2, y2 = _scale_box(w, h, *PORTRAIT_BOX)
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    return crop


# ====================================================
# Upload View（多圖）
# ====================================================
@login_required
def upload_training_results(request):
    if request.method == "POST":
        form = TrainingImageUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            messages.error(request, f"表單驗證失敗：{form.errors}")
            return render(request, "uma/upload.html", {"form": form, "results": []})

        # ✅ 這裡拿到的是「list[UploadedFile]」
        images = form.cleaned_data["images"]
        if not images:
            messages.error(request, "沒有收到圖片檔案")
            return render(request, "uma/upload.html", {"form": form, "results": []})

        results = []
        for f in images:
            # 讀圖
            pil_img = Image.open(f).convert("RGB")

            # OCR
            data = analyze_mobile_image(pil_img)

            # 裁頭像（你原本有 crop_portrait）
            portrait_crop = crop_portrait(pil_img)

            # ✅ 存 DB（這段一定要做，前端才會顯示 results）
            r = TrainingResult.objects.create(
                user=request.user,
                score=int(data.get("score", 0) or 0),
                speed=int(data.get("speed", 0) or 0),
                stamina=int(data.get("stamina", 0) or 0),
                power=int(data.get("power", 0) or 0),
                guts=int(data.get("guts", 0) or 0),
                intelligence=int(data.get("intelligence", 0) or 0),
                # terrain/distance/strategy 你本來就手動選，所以先空著
            )

            # 原圖存到 portrait（你 model 欄位名如果不是 portrait 請改成你的）
            # 用 ContentFile 存 PIL
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            r.portrait.save(f"upload_{r.id}.jpg", ContentFile(buf.getvalue()), save=False)

            buf2 = io.BytesIO()
            portrait_crop.save(buf2, format="JPEG", quality=95)
            r.portrait_crop.save(f"portrait_{r.id}.jpg", ContentFile(buf2.getvalue()), save=False)

            r.save()
            results.append(r)

        messages.success(request, f"✅ 已分析完成：{len(results)} 張")
        return render(request, "uma/upload.html", {"form": TrainingImageUploadForm(), "results": results})

    # GET
    return render(request, "uma/upload.html", {"form": TrainingImageUploadForm(), "results": []})

# ====================================================
# Bulk Update（能力值 + 分數 + 資質）— 資質必填版
# ====================================================
@login_required
def bulk_update_results(request):
    if request.method != "POST":
        return redirect("uma:upload_training_results")

    def clamp_stat(v: str) -> int:
        try:
            n = int(str(v).strip())
        except Exception:
            return 0
        return max(0, min(2999, n))

    def clamp_score(v: str) -> int:
        try:
            n = int(str(v).strip())
        except Exception:
            return 0
        return max(0, min(9_999_999, n))

    ALLOWED_TERRAIN = {"草地", "沙地"}
    ALLOWED_DISTANCE = {"短距離", "一哩", "中距離", "長距離"}

    STRATEGY_MAP = {
        "領頭": "領頭",
        "前": "前",
        "前列": "前",
        "中": "中",
        "居中": "中",
        "後": "後",
        "後追": "後",
    }
    ALLOWED_STRATEGY_CANON = {"領頭", "前", "中", "後"}

    def clean_required_choice(v: str, allowed: set[str]) -> str:
        v = (v or "").strip()
        return v if v in allowed else ""

    def clean_required_strategy(v: str) -> str:
        v = (v or "").strip()
        canon = STRATEGY_MAP.get(v, "")
        return canon if canon in ALLOWED_STRATEGY_CANON else ""

    ids = request.POST.getlist("result_ids")
    if not ids:
        messages.error(request, "❌ 沒有收到要更新的資料")
        return redirect("uma:upload_training_results")

    qs = TrainingResult.objects.filter(user=request.user, id__in=ids)
    found = {str(obj.id): obj for obj in qs}

    updated = 0
    skipped = 0
    invalid_required = 0

    for rid in ids:
        obj = found.get(str(rid))
        if not obj:
            skipped += 1
            continue

        obj.score = clamp_score(request.POST.get(f"score_{rid}", obj.score))

        obj.speed = clamp_stat(request.POST.get(f"speed_{rid}", obj.speed))
        obj.stamina = clamp_stat(request.POST.get(f"stamina_{rid}", obj.stamina))
        obj.power = clamp_stat(request.POST.get(f"power_{rid}", obj.power))
        obj.guts = clamp_stat(request.POST.get(f"guts_{rid}", obj.guts))
        obj.intelligence = clamp_stat(request.POST.get(f"intelligence_{rid}", obj.intelligence))

        terrain_raw = request.POST.get(f"terrain_{rid}", obj.terrain)
        distance_raw = request.POST.get(f"distance_{rid}", obj.distance)
        strategy_raw = request.POST.get(f"strategy_{rid}", obj.strategy)

        terrain = clean_required_choice(terrain_raw, ALLOWED_TERRAIN)
        distance = clean_required_choice(distance_raw, ALLOWED_DISTANCE)
        strategy = clean_required_strategy(strategy_raw)

        if not terrain or not distance or not strategy:
            invalid_required += 1
            continue

        obj.terrain = terrain
        obj.distance = distance
        obj.strategy = strategy

        obj.save(
            update_fields=[
                "score",
                "speed",
                "stamina",
                "power",
                "guts",
                "intelligence",
                "terrain",
                "distance",
                "strategy",
            ]
        )
        updated += 1

    if updated:
        messages.success(request, f"✅ 已儲存 {updated} 筆（分數 + 能力值 + 資質）")
    if invalid_required:
        messages.error(request, f"❌ 有 {invalid_required} 筆未選完整資質（場地/距離/作戰必填），已跳過未儲存")
    if skipped:
        messages.warning(request, f"⚠️ 有 {skipped} 筆無法更新（可能不是你的或已不存在）")

    return redirect(request.META.get("HTTP_REFERER", "uma:upload_training_results"))

@login_required
def result_detail(request, pk: int):
    r = get_object_or_404(TrainingResult, pk=pk, user=request.user)
    return render(request, "uma/result_detail.html", {"r": r})

@login_required
def result_list(request):
    qs = TrainingResult.objects.filter(user=request.user)

    # ===== 篩選 =====
    terrain = request.GET.get("terrain", "")
    distance = request.GET.get("distance", "")
    strategy = request.GET.get("strategy", "")

    if terrain:
        qs = qs.filter(terrain=terrain)
    if distance:
        qs = qs.filter(distance=distance)
    if strategy:
        qs = qs.filter(strategy=strategy)

    # ===== 排序 =====
    sort = request.GET.get("sort", "-score")  # 預設分數由高到低

    ALLOWED_SORTS = {
        "score", "-score",
        "speed", "-speed",
        "stamina", "-stamina",
        "power", "-power",
        "guts", "-guts",
        "intelligence", "-intelligence",
    }

    if sort not in ALLOWED_SORTS:
        sort = "-score"

    qs = qs.order_by(sort)

    context = {
        "results": qs,
        "selected": {
            "terrain": terrain,
            "distance": distance,
            "strategy": strategy,
            "sort": sort,
        },
        "choices": {
            "terrain": ["草地", "沙地"],
            "distance": ["短距離", "一哩", "中距離", "長距離"],
            "strategy": ["領頭", "前列", "居中", "後追"],
        },
    }
    return render(request, "uma/result_list.html", context)

    qs = TrainingResult.objects.filter(user=request.user).order_by("-created_at")

    terrain = request.GET.get("terrain", "")
    distance = request.GET.get("distance", "")
    strategy = request.GET.get("strategy", "")

    if terrain:
        qs = qs.filter(terrain=terrain)
    if distance:
        qs = qs.filter(distance=distance)
    if strategy:
        qs = qs.filter(strategy=strategy)

    context = {
        "results": qs,
        "selected": {"terrain": terrain, "distance": distance, "strategy": strategy},
        "choices": {
            "terrain": ["草地", "沙地"],        # 用你專案 whitelist 的中文
            "distance": ["短距離", "一哩", "中距離", "長距離"],
            "strategy": ["領頭", "前列", "居中", "後追"],
        },
    }
    return render(request, "uma/result_list.html", context)