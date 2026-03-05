import numpy as np

# This file finds the threshold using the histogram


def otsu_threshold(gray: np.ndarray) -> int:
    """
    Otsu threshold (implemented from scratch).
    It tries all thresholds 0..255 and picks the one that best separates 2 groups.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    p = hist / total  # probability of each intensity

    w0 = np.cumsum(p)                         # prob of class 0 (<=T)
    mu = np.cumsum(p * np.arange(256))        # mean up to T
    mu_t = mu[-1]                             # total mean

    denom = w0 * (1.0 - w0)
    denom[denom == 0] = 1e-12                 # avoid divide by zero
    sigma_b2 = (mu_t * w0 - mu) ** 2 / denom  # between-class variance

    return int(np.argmax(sigma_b2))


def threshold_auto(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Uses Otsu to get threshold T.
    Then chooses foreground side (dark or bright) automatically.
    """
    T = otsu_threshold(gray)

    mask_dark = (gray < T).astype(np.uint8)
    mask_bright = (gray >= T).astype(np.uint8)

    # Choose the mask that looks like a realistic "object size"
    frac_dark = float(mask_dark.mean())
    frac_bright = float(mask_bright.mean())

    def score(frac: float) -> float:
        if frac < 0.02 or frac > 0.70:
            return 1e9
        return abs(frac - 0.20)  # prefer around 20% foreground

    if score(frac_dark) <= score(frac_bright):
        return mask_dark, T
    else:
        return mask_bright, T