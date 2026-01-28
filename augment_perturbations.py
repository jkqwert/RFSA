#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-level perturbation generator for images (com folder) and text (create.jsonl).

Levels:
  1. Mild
  2. Moderate
  3. Severe

Image perturbations (randomly pick ONE per image within the level set):
  Level 1: gaussian/salt-pepper noise, brightness/contrast, small rotation/scale/shift, downsample-upsample.
  Level 2: local occlusion / local blur / scratches, stronger geom (rot/scale/shear/flip), gray/hue shift.
  Level 3: heavy noise/blur/motion, large occlusion/scratches/crop, extreme geom.

Text perturbations (per text field subject/object/second/relation):
  Level 1: homophone/shape-like replace (10-15%), synonym replace, punctuation add/remove.
  Level 2: keyword drop (15-20%), length adjust (+1~2 related words), order shuffle (mild).
  Level 3: keyword replace 20-30%, order shuffle (strong), drop 30-40%.

Outputs:
  Images: keep subfolder structure under --image-output-root/level{1,2,3}
  Text:   create_level{1,2,3}.jsonl under --text-output-dir

Usage (PowerShell):
  python augment_perturbations.py ^
    --image-root "D:\\python project\\sorclip\\com" ^
    --image-output-root "D:\\python project\\sorclip\\comblur" ^
    --text-json "create.jsonl" ^
    --text-output-dir "outputs/text_aug"
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# ---------------------- Image utilities ---------------------- #
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gaussian_noise(img: Image.Image, sigma):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def salt_pepper_noise(img: Image.Image, density):
    arr = np.array(img)
    h, w, c = arr.shape
    num = int(h * w * density)
    coords = (
        np.random.randint(0, h, num),
        np.random.randint(0, w, num),
    )
    mask = np.random.rand(num) > 0.5
    arr[coords[0][mask], coords[1][mask]] = 255
    arr[coords[0][~mask], coords[1][~mask]] = 0
    return Image.fromarray(arr)


def random_brightness_contrast(img: Image.Image, b_range=(0.8, 1.2), c_range=(0.85, 1.15)):
    b = random.uniform(*b_range)
    c = random.uniform(*c_range)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img


def affine_small(img: Image.Image, rot_deg=5, scale_range=(0.8, 1.2), translate_ratio=0.05):
    """Small affine: rotate + scale + translate."""
    w, h = img.size
    angle = math.radians(random.uniform(-rot_deg, rot_deg))
    scale = random.uniform(*scale_range)
    tx = random.uniform(-translate_ratio, translate_ratio) * w
    ty = random.uniform(-translate_ratio, translate_ratio) * h
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    # Affine matrix for PIL (a, b, c, d, e, f):
    # x' = a*x + b*y + c
    # y' = d*x + e*y + f
    matrix = (
        cos_a,
        -sin_a,
        tx,
        sin_a,
        cos_a,
        ty,
    )
    return img.transform((w, h), Image.AFFINE, matrix, resample=Image.BICUBIC, fillcolor=(0, 0, 0))


def down_up(img: Image.Image, ratio=0.5):
    w, h = img.size
    new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
    small = img.resize((new_w, new_h), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def local_occlusion(img: Image.Image, area_ratio=(0.1, 0.2)):
    w, h = img.size
    area = random.uniform(*area_ratio) * w * h
    side = int(math.sqrt(area))
    x = random.randint(0, max(0, w - side))
    y = random.randint(0, max(0, h - side))
    arr = np.array(img)
    arr[y:y+side, x:x+side] = 0
    return Image.fromarray(arr)


def local_blur(img: Image.Image, area_ratio=(0.15, 0.2), radius=2.0):
    w, h = img.size
    area = random.uniform(*area_ratio) * w * h
    side = int(math.sqrt(area))
    x = random.randint(0, max(0, w - side))
    y = random.randint(0, max(0, h - side))
    patch = img.crop((x, y, x + side, y + side)).filter(ImageFilter.GaussianBlur(radius))
    img = img.copy()
    img.paste(patch, (x, y))
    return img


def scratches(img: Image.Image, num_lines=(5, 8), width=(1, 2), length=(5, 10)):
    arr = np.array(img)
    h, w, _ = arr.shape
    lines = random.randint(*num_lines)
    for _ in range(lines):
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)
        ang = random.uniform(0, 2 * math.pi)
        l = random.randint(*length)
        x1 = int(x0 + l * math.cos(ang))
        y1 = int(y0 + l * math.sin(ang))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        ww = random.randint(*width)
        color = random.randint(180, 255)
        # Bresenham-like draw directly on numpy array
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        cx, cy = x0, y0
        while True:
            arr[max(0, cy-ww):cy+ww, max(0, cx-ww):cx+ww] = color
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                cx += sx
            if e2 <= dx:
                err += dx
                cy += sy
    return Image.fromarray(arr)


def hue_shift(img: Image.Image, shift_deg=10):
    img = img.convert("HSV")
    arr = np.array(img, dtype=np.int16)  # allow negative safely
    shift = random.randint(-shift_deg, shift_deg)
    arr[..., 0] = np.mod(arr[..., 0] + shift, 256)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="HSV").convert("RGB")


def motion_blur(img: Image.Image, k=7):
    # simple linear kernel
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    arr = np.array(img).astype(np.float32)
    from scipy.signal import convolve2d  # scipy may be available; if not, fallback to PIL blur
    try:
        out = np.stack([convolve2d(arr[..., c], kernel, mode="same", boundary="symm") for c in range(3)], axis=-1)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    except Exception:
        return img.filter(ImageFilter.GaussianBlur(radius=3))


def heavy_crop(img: Image.Image, ratio=(0.2, 0.3)):
    w, h = img.size
    rx = random.uniform(*ratio)
    ry = random.uniform(*ratio)
    left = int(w * rx)
    top = int(h * ry)
    right = int(w * (1 - rx))
    bottom = int(h * (1 - ry))
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.BICUBIC)


def affine_medium(img: Image.Image, rot_deg=10, scale_range=(0.7, 1.3), shear_deg=3, flip_prob=0.3):
    w, h = img.size
    angle = random.uniform(-rot_deg, rot_deg)
    scale = random.uniform(*scale_range)
    shear = random.uniform(-shear_deg, shear_deg)
    img = img.transform(
        (w, h),
        Image.AFFINE,
        (
            scale * math.cos(math.radians(angle)),
            math.tan(math.radians(shear)),
            0,
            -math.tan(math.radians(shear)),
            scale * math.cos(math.radians(angle)),
            0,
        ),
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    )
    if random.random() < flip_prob:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < flip_prob / 2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def affine_heavy(img: Image.Image, rot_deg=15, scale_range=(0.6, 1.0), shear_deg=5):
    w, h = img.size
    angle = random.uniform(-rot_deg, rot_deg)
    scale_x = random.uniform(*scale_range)
    scale_y = random.uniform(*scale_range)
    shear = random.uniform(-shear_deg, shear_deg)
    img = img.transform(
        (w, h),
        Image.AFFINE,
        (
            scale_x * math.cos(math.radians(angle)),
            math.tan(math.radians(shear)),
            0,
            -math.tan(math.radians(shear)),
            scale_y * math.cos(math.radians(angle)),
            0,
        ),
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0),
    )
    return img


# ---------------------- Text utilities ---------------------- #
# Simple homophone / synonym dictionaries (extend as needed)
HOMO_MAP = {
    "石": ["时", "十"],
    "画": ["划", "话"],
    "像": ["象", "相"],
    "人": ["仁"],
    "马": ["码"],
    "车": ["撤", "尺"],
    "鼓": ["古"],
}

SYN_MAP = {
    "攻击": ["进攻", "袭击", "打击"],
    "捕猎": ["狩猎", "猎捕"],
    "聚会": ["集会", "相聚"],
    "跪拜": ["朝拜", "膜拜"],
    "出行": ["出发", "行进"],
    "挑战": ["对抗", "交战"],
}

FILL_WORDS = ["然后", "同时", "可能", "大概", "似乎", "可以", "也许"]
PUNCT = ["，", "。", "！", "？", "、"]


def perturb_text_field(text: str, level: int) -> str:
    if not text:
        return text
    chars = list(text)

    def replace_map(mapping, ratio_range):
        ratio = random.uniform(*ratio_range)
        k = max(1, int(len(chars) * ratio))
        idxs = random.sample(range(len(chars)), min(k, len(chars)))
        for i in idxs:
            c = chars[i]
            if c in mapping and mapping[c]:
                chars[i] = random.choice(mapping[c])

    def punct_op():
        if random.random() < 0.5 and chars:
            # add
            insert_pos = random.randint(0, len(chars))
            chars.insert(insert_pos, random.choice(PUNCT))
        elif chars:
            # delete punctuation
            for i, c in enumerate(chars):
                if c in PUNCT:
                    chars.pop(i)
                    break

    def drop_keywords(ratio_range):
        ratio = random.uniform(*ratio_range)
        k = max(1, int(len(chars) * ratio))
        idxs = sorted(random.sample(range(len(chars)), min(k, len(chars))), reverse=True)
        for i in idxs:
            chars.pop(i)

    def shuffle_keywords(max_swap=3):
        swaps = random.randint(1, max_swap)
        for _ in range(swaps):
            if len(chars) < 2:
                break
            i, j = random.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]

    if level == 1:
        replace_map(HOMO_MAP, (0.10, 0.15))
        replace_map(SYN_MAP, (0.10, 0.15))
        punct_op()
    elif level == 2:
        drop_keywords((0.15, 0.20))
        if random.random() < 0.7:
            chars.extend(random.sample(FILL_WORDS, k=1))
        shuffle_keywords(max_swap=2)
    else:  # level 3
        replace_map(SYN_MAP, (0.20, 0.30))
        drop_keywords((0.30, 0.40))
        shuffle_keywords(max_swap=4)
        if random.random() < 0.8:
            chars.extend(random.sample(FILL_WORDS, k=random.randint(1, 2)))

    return "".join(chars)


def perturb_text_record(rec: dict, level: int) -> dict:
    out = dict(rec)
    for key in ["subject", "object", "second", "relation"]:
        if key in rec:
            out[key] = perturb_text_field(str(rec[key]), level)
    return out


# ---------------------- Processing ---------------------- #

def pick_image_op(level: int):
    if level == 1:
        return random.choice([
            lambda img: gaussian_noise(img, sigma=random.uniform(0.01 * 255, 0.03 * 255)),
            lambda img: salt_pepper_noise(img, density=random.uniform(0.05, 0.10)),
            lambda img: random_brightness_contrast(img, (0.8, 1.2), (0.85, 1.15)),
            lambda img: affine_small(img, rot_deg=5, scale_range=(0.8, 1.2), translate_ratio=0.05),
            lambda img: down_up(img, ratio=random.uniform(0.4, 0.7)),
        ])
    elif level == 2:
        return random.choice([
            lambda img: local_occlusion(img, (0.10, 0.20)),
            lambda img: local_blur(img, (0.15, 0.20), radius=2.5),
            lambda img: scratches(img, (5, 8), (1, 2), (5, 10)),
            lambda img: affine_medium(img, rot_deg=10, scale_range=(0.7, 1.3), shear_deg=3, flip_prob=0.5),
            lambda img: hue_shift(img, 10),
            lambda img: img.convert("L").convert("RGB"),
        ])
    else:
        return random.choice([
            lambda img: gaussian_noise(img, sigma=random.uniform(0.05 * 255, 0.08 * 255)),
            lambda img: salt_pepper_noise(img, density=random.uniform(0.25, 0.30)),
            lambda img: motion_blur(img, k=7),
            lambda img: local_occlusion(img, (0.30, 0.40)),
            lambda img: scratches(img, (15, 20), (1, 2), (5, 15)),
            lambda img: heavy_crop(img, (0.20, 0.30)),
            lambda img: affine_heavy(img, rot_deg=15, scale_range=(0.6, 1.0), shear_deg=5),
        ])


def process_images(args, level: int):
    src_root = Path(args.image_root)
    dst_root = Path(args.image_output_root) / f"level{level}"
    files = [p for p in src_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"[Level {level}] Found {len(files)} images")
    op_picker = lambda: pick_image_op(level)

    for i, src in enumerate(files, 1):
        try:
            img = Image.open(src).convert("RGB")
            op = op_picker()
            out = op(img)
            dst = dst_root / src.relative_to(src_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            out.save(dst)
        except Exception as e:
            print(f"Failed on {src}: {e}")
        if i % 200 == 0:
            print(f"[Level {level}] Processed {i}/{len(files)}")


def process_text(args, level: int):
    src = Path(args.text_json)
    out_dir = Path(args.text_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"create_level{level}.jsonl"
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line.strip())
            rec_aug = perturb_text_record(rec, level)
            fout.write(json.dumps(rec_aug, ensure_ascii=False) + "\n")
    print(f"[Level {level}] Text saved to {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, required=True, help="Root dir of original images (com)")
    parser.add_argument("--image-output-root", type=str, required=True, help="Output root for perturbed images")
    parser.add_argument("--text-json", type=str, required=True, help="create.jsonl path")
    parser.add_argument("--text-output-dir", type=str, required=True, help="Output dir for perturbed text jsonl")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3], help="Levels to run")
    args = parser.parse_args()

    for level in args.levels:
        print(f"\n=== Running Level {level} ===")
        process_images(args, level)
        process_text(args, level)
    print("\nAll done.")


if __name__ == "__main__":
    main()

