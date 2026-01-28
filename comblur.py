import random
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# 输入与输出目录
src_root = Path(r"D:\python project\sorclip\com")
dst_root = Path(r"/comblur")

# 支持的图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def random_crop(img: Image.Image, min_scale=0.8):
    w, h = img.size
    scale = random.uniform(min_scale, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < 1 or new_h < 1:
        return img
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BICUBIC)

def add_gaussian_noise(img: Image.Image, std=15):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def random_blur(img: Image.Image):
    radius = random.uniform(0.8, 2.0)
    return img.filter(ImageFilter.GaussianBlur(radius))

def random_brightness(img: Image.Image):
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor)

augment_ops = [
    ("crop", random_crop),
    ("noise", add_gaussian_noise),
    ("blur", random_blur),
    ("brightness", random_brightness),
]

def process_image(src_path: Path, dst_path: Path):
    img = Image.open(src_path).convert("RGB")
    _, op = random.choice(augment_ops)
    aug = op(img)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    aug.save(dst_path)

def main():
    files = [p for p in src_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(files)} images.")
    for i, src in enumerate(files, 1):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        process_image(src, dst)
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)}")
    print("Done.")

if __name__ == "__main__":
    main()