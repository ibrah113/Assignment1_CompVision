import os
import glob
import time
from collections import deque

import cv2
import numpy as np


# ============================================================
# 1) Histogram + automatic threshold (Otsu from scratch)
# ============================================================

def otsu_threshold(gray: np.ndarray) -> int:
    """
    Finds a threshold T (0..255) by trying all values and picking the one
    that best separates the histogram into 2 groups (background vs object).
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    p = hist / total

    w0 = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]

    denom = w0 * (1.0 - w0)
    denom[denom == 0] = 1e-12
    sigma_b2 = (mu_t * w0 - mu) ** 2 / denom

    return int(np.argmax(sigma_b2))


def threshold_auto(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Uses Otsu to get T, then decides which side is foreground automatically.
    """
    T = otsu_threshold(gray)

    mask_dark = (gray < T).astype(np.uint8)
    mask_bright = (gray >= T).astype(np.uint8)

    frac_dark = float(mask_dark.mean())
    frac_bright = float(mask_bright.mean())

    def score(frac: float) -> float:
        if frac < 0.02 or frac > 0.70:
            return 1e9
        return abs(frac - 0.20)

    if score(frac_dark) <= score(frac_bright):
        return mask_dark, T
    else:
        return mask_bright, T


# ============================================================
# 2) Binary morphology (closing) to fill small holes
# ============================================================

def dilate(bin_img: np.ndarray, k: int = 3) -> np.ndarray:
    """Expands white pixels (1s)."""
    pad = k // 2
    h, w = bin_img.shape
    padded = np.pad(bin_img, pad, mode="constant", constant_values=0)
    out = np.zeros_like(bin_img)

    for y in range(h):
        for x in range(w):
            window = padded[y:y + k, x:x + k]
            out[y, x] = 1 if np.any(window) else 0
    return out


def erode(bin_img: np.ndarray, k: int = 3) -> np.ndarray:
    """Shrinks white pixels (1s)."""
    pad = k // 2
    h, w = bin_img.shape
    padded = np.pad(bin_img, pad, mode="constant", constant_values=0)
    out = np.zeros_like(bin_img)

    for y in range(h):
        for x in range(w):
            window = padded[y:y + k, x:x + k]
            out[y, x] = 1 if np.all(window) else 0
    return out


def closing(bin_img: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    """
    Closing = dilate then erode.
    Helps close tiny interior holes in the rubber region.
    """
    out = bin_img.copy()
    for _ in range(iters):
        out = dilate(out, k)
    for _ in range(iters):
        out = erode(out, k)
    return out


# ============================================================
# 3) Connected Component Labelling (extract regions)
# ============================================================

def connected_components(bin_img: np.ndarray, connectivity: int = 8):
    """
    Labels each separate white region with a different integer label.
    Returns a label image and a list of regions (area + bounding box).
    """
    h, w = bin_img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comps = []
    cur_label = 0

    neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neigh8 = neigh4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    neigh = neigh8 if connectivity == 8 else neigh4

    for y in range(h):
        for x in range(w):
            if bin_img[y, x] == 1 and labels[y, x] == 0:
                cur_label += 1
                q = deque([(y, x)])
                labels[y, x] = cur_label

                area = 0
                miny = maxy = y
                minx = maxx = x

                while q:
                    cy, cx = q.popleft()
                    area += 1

                    if cy < miny: miny = cy
                    if cy > maxy: maxy = cy
                    if cx < minx: minx = cx
                    if cx > maxx: maxx = cx

                    for dy, dx in neigh:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if bin_img[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = cur_label
                                q.append((ny, nx))

                comps.append({"label": cur_label, "area": area, "bbox": (miny, minx, maxy, maxx)})

    return labels, comps


def largest_component(bin_img: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Picks the biggest white region (the O-ring)."""
    labels, comps = connected_components(bin_img, connectivity=8)
    if not comps:
        return np.zeros_like(bin_img), None
    largest = max(comps, key=lambda c: c["area"])
    return (labels == largest["label"]).astype(np.uint8), largest


# ============================================================
# 4) PASS/FAIL analysis helpers
# ============================================================

def count_holes(ring_mask: np.ndarray) -> int:
    """
    Counts enclosed background areas inside the ring.
    A good O-ring should have exactly 1 hole (the center).
    """
    ys, xs = np.where(ring_mask == 1)
    if ys.size == 0:
        return 0

    miny, maxy = int(ys.min()), int(ys.max())
    minx, maxx = int(xs.min()), int(xs.max())

    roi = ring_mask[miny:maxy + 1, minx:maxx + 1]
    inv = (1 - roi).astype(np.uint8)

    labels, comps = connected_components(inv, connectivity=8)
    h, w = inv.shape

    border_labels = set()
    for x in range(w):
        if inv[0, x] == 1: border_labels.add(labels[0, x])
        if inv[h - 1, x] == 1: border_labels.add(labels[h - 1, x])
    for y in range(h):
        if inv[y, 0] == 1: border_labels.add(labels[y, 0])
        if inv[y, w - 1] == 1: border_labels.add(labels[y, w - 1])

    holes = [c for c in comps if c["label"] not in border_labels]
    return len(holes)


def boundary_angular_gap(ring_mask: np.ndarray, bins: int = 360) -> float:
    """
    Measures the largest missing run of boundary angles (in degrees).
    Good for detecting fully broken/open rings.
    """
    ys, xs = np.where(ring_mask == 1)
    if ys.size == 0:
        return 360.0

    cy = float(ys.mean())
    cx = float(xs.mean())
    h, w = ring_mask.shape

    occ = np.zeros(bins, dtype=np.uint8)

    for y, x in zip(ys, xs):
        # boundary pixel if any neighbor is background
        is_boundary = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and ring_mask[ny, nx] == 0:
                    is_boundary = True
                    break
            if is_boundary:
                break

        if not is_boundary:
            continue

        ang = np.arctan2(y - cy, x - cx)
        ang = (ang + 2.0 * np.pi) % (2.0 * np.pi)
        b = int((ang / (2.0 * np.pi)) * bins) % bins
        occ[b] = 1

    empty = (occ == 0).astype(np.uint8)
    doubled = np.concatenate([empty, empty])

    best = cur = 0
    for v in doubled:
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    best = min(best, bins)
    return best * (360.0 / bins)


def ray_thickness_stats(ring_mask: np.ndarray, n_angles: int = 360) -> dict:
    """
    Measures ring thickness around 360 degrees.
    For each angle: go outward from centroid and measure the first continuous run of 1s.
    If a chunk is missing, thickness drops at those angles.
    """
    ys, xs = np.where(ring_mask == 1)
    if ys.size == 0:
        return {"median": 0.0, "min": 0.0, "bad_frac": 1.0}

    cy = float(ys.mean())
    cx = float(xs.mean())

    h, w = ring_mask.shape
    max_r = int(max(cy, cx, h - cy, w - cx)) + 2

    thickness = np.zeros(n_angles, dtype=np.float64)

    for i in range(n_angles):
        theta = 2.0 * np.pi * i / n_angles

        # sample along ray
        vals = []
        for r in range(max_r):
            y = int(round(cy + r * np.sin(theta)))
            x = int(round(cx + r * np.cos(theta)))
            if y < 0 or x < 0 or y >= h or x >= w:
                break
            vals.append(ring_mask[y, x])

        if not vals:
            thickness[i] = 0.0
            continue

        vals = np.array(vals, dtype=np.uint8)
        idx = np.where(vals == 1)[0]
        if idx.size == 0:
            thickness[i] = 0.0
            continue

        # first run of 1s (enter ring and stay until leaving)
        start = int(idx[0])
        end = start
        while end + 1 < vals.size and vals[end + 1] == 1:
            end += 1

        thickness[i] = float(end - start + 1)

    nonzero = thickness[thickness > 0]
    if nonzero.size == 0:
        return {"median": 0.0, "min": 0.0, "bad_frac": 1.0}

    med = float(np.median(nonzero))
    mn = float(nonzero.min())

    # "bad" if thickness is much smaller than typical thickness
    bad_frac = float(np.mean(thickness < 0.70 * med))

    return {"median": med, "min": mn, "bad_frac": bad_frac}


def classify_oring(ring_mask: np.ndarray) -> tuple[str, str]:
    """
    Explainable rules:
    1) Must have exactly 1 hole (donut)
    2) Must not have a huge boundary angular gap (open/broken ring)
    3) Must not have a big thickness drop over a chunk of angles (ripped/bite defect)
    """
    holes = count_holes(ring_mask)
    max_gap_deg = boundary_angular_gap(ring_mask, bins=360)
    thick = ray_thickness_stats(ring_mask, n_angles=360)

    reasons = []
    fail = False

    if holes != 1:
        fail = True
        reasons.append(f"holes={holes} (expected 1)")

    # Open/broken ring (large missing arc)
    if max_gap_deg > 18.0:   # you can tune: 12..25 depending on dataset
        fail = True
        reasons.append(f"max_gap_deg={max_gap_deg:.1f}")

    # Missing material / ripped chunk (thickness drop)
    # These two checks catch "bite" defects like your #15.
    if thick["median"] > 0:
        if thick["min"] < 0.50 * thick["median"]:
            fail = True
            reasons.append(f"min_thick={thick['min']:.1f} (<0.5*med)")
        if thick["bad_frac"] > 0.03:  # >3% of angles too thin
            fail = True
            reasons.append(f"thin_frac={thick['bad_frac']:.3f}")

    label = "FAIL" if fail else "PASS"
    reason_text = "OK" if not reasons else " | ".join(reasons)
    return label, reason_text


# ============================================================
# 5) Visual output + timing
# ============================================================

def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay ring mask in red for easy viewing."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = bgr.copy()
    out[mask == 1, 2] = 255
    out[mask == 1, 0] = (out[mask == 1, 0] * 0.5).astype(np.uint8)
    out[mask == 1, 1] = (out[mask == 1, 1] * 0.5).astype(np.uint8)
    return out


def process_image(path: str, morph_k: int = 3, morph_iters: int = 1):
    """
    read -> threshold -> morphology -> largest component -> classify -> annotate
    """
    t0 = time.perf_counter()

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {path}")

    bin_mask, T = threshold_auto(gray)

    # Keep morphology minimal (don’t over-fix defects)
    bin_clean = closing(bin_mask, k=morph_k, iters=morph_iters)

    ring_mask, _ = largest_component(bin_clean)

    label, reason = classify_oring(ring_mask)

    dt_ms = (time.perf_counter() - t0) * 1000.0

    vis = overlay_mask(gray, ring_mask)
    line1 = f"{label}  T={T}  time={dt_ms:.1f}ms"
    line2 = reason

    cv2.putText(vis, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis, {
        "path": path,
        "T": T,
        "label": label,
        "reason": reason,
        "time_ms": dt_ms
    }


def main():
    INPUT_DIR = "."
    OUTPUT_DIR = os.path.join(os.getcwd(), "output_annotated")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.png")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
    )

    if not images:
        print("No images found. Put this script in the image folder or change INPUT_DIR.")
        return

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for p in images:
            vis, info = process_image(p, morph_k=3, morph_iters=1)

            out_name = os.path.splitext(os.path.basename(p))[0] + "_result.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, vis)

            line = (
                f"{os.path.basename(p)}\t{info['label']}\t"
                f"T={info['T']}\t{info['reason']}\t"
                f"time={info['time_ms']:.1f}ms"
            )
            print(line)
            f.write(line + "\n")

    print(f"\nSaved results in: {OUTPUT_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()