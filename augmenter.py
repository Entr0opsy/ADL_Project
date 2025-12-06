import os, random, math
import numpy as np
from PIL import Image
try:
    import cv2
    CV2 = True
except Exception:
    cv2 = None
    CV2 = False

# HYPERPARAMETERS (edit as needed)
INPUT_DIR = "./masked_dataset/"
OUTPUT_DIR = "./Masked_dataset_augmented/"
N_AUG_PER_IMAGE = 3


def _read_image(p):
    if CV2:
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(p)
        if im.ndim == 2:
            return im
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        return np.array(Image.open(p))


def _write_image(p, arr):
    arr = np.asarray(arr)
    if CV2:
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr2 = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            arr2 = arr
        cv2.imwrite(p, arr2)
    else:
        Image.fromarray(arr).save(p)


class BaseAug:
    def __call__(self, img):
        raise NotImplementedError


def polar_horizontal_shift(img, max_shift_frac=0.2):
    h, w = img.shape[:2]
    max_shift = int(w * max_shift_frac)
    if max_shift < 1:
        return img
    shift = random.randint(-max_shift, max_shift)
    return np.roll(img, shift, axis=1)


def polar_vertical_scale(img, scale_min=0.8, scale_max=1.25):
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
        start = (new_h - h) // 2
        return scaled[start:start+h, :]
    pad_total = h - new_h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    if scaled.ndim == 3:
        padded = np.pad(scaled, ((pad_top, pad_bottom),(0,0),(0,0)), mode='constant', constant_values=0)
    else:
        padded = np.pad(scaled, ((pad_top, pad_bottom),(0,0)), mode='constant', constant_values=0)
    return padded


def eyelid_occlusion(img, max_occl_frac=0.25):
    h, w = img.shape[:2]
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
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    num = random.randint(0, max_blobs)
    for _ in range(num):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        intensity = random.uniform(intensity_range[0], intensity_range[1])
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
        xv, yv = np.meshgrid(xs - cx, ys - cy)
        g = np.exp(-(xv**2 + yv**2) / (2 * sigma * sigma))
        g = g / g.max()
        g = g * intensity
        if out.ndim == 3:
            for c in range(out.shape[2]):
                out[y0:y1, x0:x1, c] += g
        else:
            out[y0:y1, x0:x1] += g
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def apply_random_augmentations(img, cfg):
    a = img.copy()
    if random.random() < cfg['shift_prob']:
        a = polar_horizontal_shift(a, cfg['max_shift_frac'])
    if random.random() < cfg['vertical_prob']:
        a = polar_vertical_scale(a, cfg['scale_min'], cfg['scale_max'])
    if random.random() < cfg['eyelid_prob']:
        a = eyelid_occlusion(a, cfg['max_occl_frac'])
    if random.random() < cfg['specular_prob']:
        a = inject_specular_reflection(a, max_blobs=cfg['max_blobs'], sigma_range=cfg['sigma_range'], intensity_range=cfg['intensity_range'])
    return a



# Change Here for the different augmentation
DEFAULT_CFG = {
    'shift_prob': 0.5,
    'max_shift_frac': 0.2,
    'vertical_prob': 0.4,
    'scale_min': 0.9,
    'scale_max': 1.1,
    'eyelid_prob': 0.3,
    'max_occl_frac': 0.25,
    'specular_prob': 0.4,
    'max_blobs': 2,
    'sigma_range': (1.0,4.0),
    'intensity_range': (180,255),
}


from tqdm import tqdm

def process(in_dir=INPUT_DIR, out_dir=OUTPUT_DIR, n_aug=N_AUG_PER_IMAGE, cfg=DEFAULT_CFG):
    if not os.path.exists(in_dir):
        raise SystemExit(f"input dir not found: {in_dir}")

    # Count total image files for a single overall progress bar
    total = 0
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                total += 1

    with tqdm(total=total, desc="Processing images", unit="image") as pbar:
        for root, dirs, files in os.walk(in_dir):
            rel = os.path.relpath(root, in_dir)
            out_root = os.path.join(out_dir, rel) if rel != '.' else out_dir
            os.makedirs(out_root, exist_ok=True)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    src = os.path.join(root, f)
                    try:
                        img = _read_image(src)
                    except Exception:
                        pbar.update(1)
                        continue
                    dst0 = os.path.join(out_root, f)
                    _write_image(dst0, img)
                    name, ext = os.path.splitext(f)
                    for i in range(n_aug):
                        aug = apply_random_augmentations(img, cfg)
                        dst = os.path.join(out_root, f"{name}_aug{i}{ext}")
                        _write_image(dst, aug)
                    pbar.update(1)

if __name__ == '__main__':
    process()
