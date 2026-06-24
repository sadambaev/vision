import cv2
import numpy as np

orig = cv2.imread("image3.png")

for i, path in enumerate([
    "sequence_9_17_13_1.png"
], 1):
    recon = cv2.imread(path)
    if orig.shape != recon.shape:
        print(f"[{i}] shapes différentes : orig={orig.shape} recon={recon.shape}")
        continue

    diff = cv2.absdiff(orig, recon)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    nonzero = np.count_nonzero(diff_gray)
    total = diff_gray.size
    print(f"[{i}] {path}")
    print(f"     pixels différents : {nonzero} / {total} ({100*nonzero/total:.2f}%)")
    print(f"     diff max={diff_gray.max()}, mean={diff_gray.mean():.3f}")

    diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f"diff_{i}.png", diff_vis)
    print(f"     Sauvegardé : diff_{i}.png")
