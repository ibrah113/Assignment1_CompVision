import numpy as np
from collections import deque

# This file finds connected regions, connected components


def connected_components(bin_img: np.ndarray, connectivity: int = 8):
    """
    Labels each connected white region (1s) with a unique label number.
    Uses BFS flood fill.
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

                comps.append({
                    "label": cur_label,
                    "area": area,
                    "bbox": (miny, minx, maxy, maxx)
                })

    return labels, comps


def largest_component(bin_img: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """
    Keeps only the largest connected component.
    We assume the O-ring is the biggest foreground region.
    """
    labels, comps = connected_components(bin_img, connectivity=8)
    if not comps:
        return np.zeros_like(bin_img), None

    largest = max(comps, key=lambda c: c["area"])
    mask = (labels == largest["label"]).astype(np.uint8)
    return mask, largest