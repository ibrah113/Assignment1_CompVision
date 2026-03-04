import numpy as np
from components import connected_components

# This file contains the PASS/FAIL checks


def count_holes(ring_mask: np.ndarray) -> int:
    """
    Counts enclosed background regions inside the ring.
    Good ring should have exactly 1 hole (the center).
    """
    ys, xs = np.where(ring_mask == 1)
    if ys.size == 0:
        return 0

    miny, maxy = int(ys.min()), int(ys.max())
    minx, maxx = int(xs.min()), int(xs.max())

    roi = ring_mask[miny:maxy + 1, minx:maxx + 1]
    inv = (1 - roi).astype(np.uint8)  # background becomes 1

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
    Checks how much of the ring boundary is missing around 360 degrees.
    Large missing section means ring is open/broken.
    """
    ys, xs = np.where(ring_mask == 1)
    if ys.size == 0:
        return 360.0

    cy = float(ys.mean())
    cx = float(xs.mean())
    h, w = ring_mask.shape

    occ = np.zeros(bins, dtype=np.uint8)

    for y, x in zip(ys, xs):
        # boundary pixel = touches background
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
    Missing chunk makes thickness drop for some angles.
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
    bad_frac = float(np.mean(thickness < 0.70 * med))

    return {"median": med, "min": mn, "bad_frac": bad_frac}


def classify_oring(ring_mask: np.ndarray) -> tuple[str, str]:
    """
    Simple rules:
    - hole count must be 1
    - boundary gap must be small
    - thickness must not drop too much
    """
    holes = count_holes(ring_mask)
    max_gap_deg = boundary_angular_gap(ring_mask, bins=360)
    thick = ray_thickness_stats(ring_mask, n_angles=360)

    reasons = []
    fail = False

    if holes != 1:
        fail = True
        reasons.append(f"holes={holes} (expected 1)")

    if max_gap_deg > 18.0:
        fail = True
        reasons.append(f"max_gap_deg={max_gap_deg:.1f}")

    if thick["median"] > 0:
        if thick["min"] < 0.50 * thick["median"]:
            fail = True
            reasons.append(f"min_thick={thick['min']:.1f} (<0.5*med)")
        if thick["bad_frac"] > 0.03:
            fail = True
            reasons.append(f"thin_frac={thick['bad_frac']:.3f}")

    label = "FAIL" if fail else "PASS"
    reason_text = "OK" if not reasons else " | ".join(reasons)
    return label, reason_text