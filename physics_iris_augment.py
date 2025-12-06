"""
Physics-aware iris augmentation script.

Creates augmented copies of polar-unwrapped iris images using feasible eye-physics transforms:
- Horizontal translation (polar shift): circular horizontal roll to simulate eye rotation.
- Vertical scaling (polar stretch): vertical resizing to simulate pupil dilation/constriction.
- Eyelid occlusion simulation: randomly mask top/bottom rows of the polar strip.
- Specular reflection injection: add small bright Gaussian blobs to mimic NIR reflections.

Usage (example):
    python3 physics_iris_augment.py \
    --input_dir masked_dataset/train \
    --output_dir masked_dataset_with_physics_aug/train \
    --per_image 10 \
    --seed 42 \
    --limit_subjects 411

The script preserves the per-subject folder layout and writes augmented images alongside originals.
"""

import argparse
import os
import random
import math
from glob import glob

import numpy as np
try:
    import cv2
    CV2 = True
except Exception:
    from PIL import Image
    CV2 = False


def load_image(path):
    if CV2:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read {path}")
        # convert BGRA->BGR if needed
        return img
    else:
        img = Image.open(path).convert('RGB')
        return np.array(img)


def save_image(path, img):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    if CV2:
        # cv2 expects BGR for color images; if img is RGB, convert
        cv2.imwrite(path, img)
    else:
        Image.fromarray(img).save(path)


def polar_horizontal_shift(img, max_shift_frac=0.2):
    # shift fraction of width, wrap-around (circular)
    h, w = img.shape[:2]
    max_shift = int(w * max_shift_frac)
    if max_shift < 1:
        return img
    shift = random.randint(-max_shift, max_shift)
    return np.roll(img, shift, axis=1)


def polar_vertical_scale(img, scale_min=0.8, scale_max=1.25):
    # scale vertically and then center-crop or pad back to original height
    h, w = img.shape[:2]
    scale = random.uniform(scale_min, scale_max)
    new_h = max(1, int(round(h * scale)))
    if CV2:
        scaled = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scaled = np.array(Image.fromarray(img).resize((w, new_h), Image.BILINEAR))
    if new_h == h:
        return scaled
    if new_h > h:
        # crop center
        start = (new_h - h) // 2
        return scaled[start:start+h, :]
    else:
        # pad equally top and bottom with zeros (or replicate border)
        pad_total = h - new_h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        if scaled.ndim == 3:
            pad_shape_top = ((pad_top,0),(0,0),(0,0))
        else:
            pad_shape_top = ((pad_top,0),(0,0))
        padded = np.pad(scaled, pad_shape_top, mode='constant', constant_values=0)
        if pad_bottom>0:
            if padded.ndim == 3:
                padded = np.pad(padded, ((0,pad_bottom),(0,0),(0,0)), mode='constant', constant_values=0)
            else:
                padded = np.pad(padded, ((0,pad_bottom),(0,0)), mode='constant', constant_values=0)
        return padded


def eyelid_occlusion(img, max_occl_frac=0.25):
    # mask top and/or bottom rows of the polar strip to simulate eyelids
    h, w = img.shape[:2]
    # decide top and bottom occlusion sizes (fraction of height)
    top_frac = random.uniform(0, max_occl_frac)
    bottom_frac = random.uniform(0, max_occl_frac)
    top_rows = int(round(h * top_frac))
    bottom_rows = int(round(h * bottom_frac))
    occluded = img.copy()
    if top_rows > 0:
        occluded[:top_rows, :] = 0
    if bottom_rows > 0:
        occluded[h-bottom_rows:, :] = 0
    return occluded


def inject_specular_reflection(img, max_blobs=3, sigma_range=(1.0,6.0), intensity_range=(180,255)):
    # add a few bright Gaussian blobs (white-ish) to simulate NIR LED reflections
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    num = random.randint(0, max_blobs)
    for _ in range(num):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        intensity = random.uniform(intensity_range[0], intensity_range[1])
        # create a small gaussian patch and blend additively
        size = int(math.ceil(sigma * 6))
        if size % 2 == 0:
            size += 1
        radius = size // 2
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        if len(xs) == 0 or len(ys) == 0:
            continue
        # create meshgrid where g has shape (len(ys), len(xs)) matching image slice
        xv, yv = np.meshgrid(xs - cx, ys - cy)
        g = np.exp(-(xv**2 + yv**2) / (2 * sigma * sigma))
        g = g / g.max()
        g = g * intensity
        # add to all channels; g already matches the (y, x) slice shape so no transpose
        if out.ndim == 3:
            for c in range(out.shape[2]):
                out[y0:y1, x0:x1, c] += g
        else:
            out[y0:y1, x0:x1] += g
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def apply_random_augmentations(img, cfg):
    a = img.copy()
    # order: shift -> vertical scale -> eyelid occlusion -> specular
    if random.random() < cfg['shift_prob']:
        a = polar_horizontal_shift(a, cfg['max_shift_frac'])
    if random.random() < cfg['vertical_prob']:
        a = polar_vertical_scale(a, cfg['scale_min'], cfg['scale_max'])
    if random.random() < cfg['eyelid_prob']:
        a = eyelid_occlusion(a, cfg['max_occl_frac'])
    if random.random() < cfg['specular_prob']:
        a = inject_specular_reflection(a, max_blobs=cfg['max_blobs'], sigma_range=cfg['sigma_range'], intensity_range=cfg['intensity_range'])
    return a


def is_image_file(p):
    ext = os.path.splitext(p)[1].lower()
    return ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')


def process_dataset(input_dir, output_dir, per_image=5, cfg=None, limit_subjects=None, dry_run=False):
    subjects = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if limit_subjects is not None:
        subjects = subjects[:limit_subjects]
    total = 0
    for s in subjects:
        in_sub = os.path.join(input_dir, s)
        out_sub = os.path.join(output_dir, s)
        os.makedirs(out_sub, exist_ok=True)
        images = sorted([os.path.join(in_sub, f) for f in os.listdir(in_sub) if is_image_file(f)])
        for img_path in images:
            try:
                img = load_image(img_path)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]
            # copy original into output_dir for convenience
            save_image(os.path.join(out_sub, base + '.jpg'), img)
            for i in range(per_image):
                aug = apply_random_augmentations(img, cfg)
                out_name = f"{base}_aug_{i+1}.jpg"
                save_image(os.path.join(out_sub, out_name), aug)
            total += 1
            if dry_run and total >= 10:
                return
    print(f"Wrote augmented dataset to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Physics-aware iris augmentations (polar domain)')
    parser.add_argument('--input_dir', required=True, help='Root input dir with per-subject subfolders')
    parser.add_argument('--output_dir', required=True, help='Output root dir to write augmented dataset')
    parser.add_argument('--per_image', type=int, default=3, help='Number of augmented images to generate per original image')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_shift_frac', type=float, default=0.25, help='Max horizontal shift fraction of width')
    parser.add_argument('--shift_prob', type=float, default=0.9)
    parser.add_argument('--scale_min', type=float, default=0.85)
    parser.add_argument('--scale_max', type=float, default=1.15)
    parser.add_argument('--vertical_prob', type=float, default=0.75)
    parser.add_argument('--eyelid_prob', type=float, default=0.6)
    parser.add_argument('--max_occl_frac', type=float, default=0.22)
    parser.add_argument('--specular_prob', type=float, default=0.6)
    parser.add_argument('--max_blobs', type=int, default=3)
    parser.add_argument('--limit_subjects', type=int, default=None)
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = {
        'max_shift_frac': args.max_shift_frac,
        'shift_prob': args.shift_prob,
        'scale_min': args.scale_min,
        'scale_max': args.scale_max,
        'vertical_prob': args.vertical_prob,
        'eyelid_prob': args.eyelid_prob,
        'max_occl_frac': args.max_occl_frac,
        'specular_prob': args.specular_prob,
        'max_blobs': args.max_blobs,
        'sigma_range': (1.0, 5.0),
        'intensity_range': (190, 255),
    }
    print('Config:', cfg)
    process_dataset(args.input_dir, args.output_dir, per_image=args.per_image, cfg=cfg, limit_subjects=args.limit_subjects, dry_run=args.dry_run)
