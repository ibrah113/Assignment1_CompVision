import os
import glob
import time

import cv2
import numpy as np

from thresholding import threshold_auto
from morpholgy import closing
from components import largest_component
from inspection import classify_oring


# This file runs the whole pipeline and saves the results


def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shows the ring pixels in red so we can see what was detected."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = bgr.copy()
    out[mask == 1, 2] = 255
    out[mask == 1, 0] = (out[mask == 1, 0] * 0.5).astype(np.uint8)
    out[mask == 1, 1] = (out[mask == 1, 1] * 0.5).astype(np.uint8)
    return out


def process_image(path: str, morph_k: int = 3, morph_iters: int = 1):
    """
    Steps:
    1) read image
    2) threshold
    3) morphology clean
    4) largest component (ring)
    5) classify pass/fail
    6) annotate output
    """
    t0 = time.perf_counter()

    # OpenCV used only for reading
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {path}")

    # 1) Threshold (auto from histogram)
    bin_mask, T = threshold_auto(gray)

    # 2) Morphology (closing)
    bin_clean = closing(bin_mask, k=morph_k, iters=morph_iters)

    # 3) Keep only largest region (the ring)
    ring_mask, _ = largest_component(bin_clean)

    # 4) PASS/FAIL
    label, reason = classify_oring(ring_mask)

    # Timing
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # OpenCV used only for showing/saving
    vis = overlay_mask(gray, ring_mask)
    line1 = f"{label}  T={T}  time={dt_ms:.1f}ms"
    line2 = reason

    cv2.putText(vis, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis, {"T": T, "label": label, "reason": reason, "time_ms": dt_ms}


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
        print("No images found. Put main.py in the image folder or change INPUT_DIR.")
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