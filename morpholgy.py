import numpy as np

# This file cleans the binary image using morphology (written from scratch)


def dilate(bin_img: np.ndarray, k: int = 3) -> np.ndarray:
    """Dilation: expands white pixels (1s)."""
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
    """Erosion: shrinks white pixels (1s)."""
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
    Closing = dilation then erosion.
    It fills small holes and closes small gaps.
    """
    out = bin_img.copy()
    for _ in range(iters):
        out = dilate(out, k)
    for _ in range(iters):
        out = erode(out, k)
    return out