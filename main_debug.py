import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from tkinter import Label, Tk, Toplevel, messagebox, simpledialog
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np
from sklearn.cluster import KMeans

root = Tk()
root.withdraw()
image_path = askopenfilename(
    title="Choose an image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
)

if not image_path:
    print("No image selected")
    exit()

apply_filters = messagebox.askyesno(
    "Filters",
    "Apply filters to the image ?\n(denoising, contrast enhancement, binarization)",
)
root.destroy()

os.makedirs("chars", exist_ok=True)

img = cv2.imread(image_path)
img_original = img.copy()

max_display = 1200
scale = min(max_display / img.shape[1], max_display / img.shape[0], 1.0)
img_display = cv2.resize(img, None, fx=scale, fy=scale)

x, y, w, h = cv2.selectROI(
    "Select zone", img_display, fromCenter=False, showCrosshair=True
)

x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

img[y : y + h, x : x + w] = 255
mask_x, mask_y, mask_w, mask_h = x, y, w, h

cv2.imwrite("image_modified.png", img)

cv2.destroyAllWindows()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

if apply_filters:
    # denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # binarization
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
else:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("image_clean.png", bw)

subprocess.run(
    ["kraken", "-i", "image_clean.png", "image.json", "segment", "-b"], check=True
)

with open("image.json") as f:
    data = json.load(f)

glyphs = []  # (abs_x, abs_y, gw, gh, char_img, line_id)

for line_id, line in enumerate(data["lines"]):
    x1, y1, x2, y2 = line["bbox"]
    x, y, w, h = x1, y1, x2 - x1, y2 - y1

    line_img = gray[y : y + h, x : x + w]
    _, bw = cv2.threshold(line_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw)

    raw_boxes = []
    for j in range(1, num):
        gx, gy, gw, gh, area = stats[j]
        if area < 20:
            continue
        raw_boxes.append((x + gx, y + gy, gw, gh))

    # Merge vertically aligned components (e.g. dot of i/j, accents)
    raw_boxes.sort(key=lambda b: b[0])
    merged = []
    used = [False] * len(raw_boxes)
    for a in range(len(raw_boxes)):
        if used[a]:
            continue
        ax, ay, aw, ah = raw_boxes[a]
        for b in range(a + 1, len(raw_boxes)):
            if used[b]:
                continue
            bx, by, bw, bh = raw_boxes[b]
            # Check horizontal overlap (centers within half-width of each other)
            a_cx = ax + aw / 2
            b_cx = bx + bw / 2
            overlap_x = abs(a_cx - b_cx) < max(aw, bw) * 0.7
            # Check vertical proximity (gap less than half the line height)
            gap_y = max(0, max(ay, by) - min(ay + ah, by + bh))
            close_y = gap_y < h * 0.3
            if overlap_x and close_y:
                # Merge into one bounding box
                nx = min(ax, bx)
                ny = min(ay, by)
                nx2 = max(ax + aw, bx + bw)
                ny2 = max(ay + ah, by + bh)
                ax, ay, aw, ah = nx, ny, nx2 - nx, ny2 - ny
                used[b] = True
        merged.append((ax, ay, aw, ah))

    line_glyphs = []
    for abs_x, abs_y, gw, gh in merged:
        char_img = gray[abs_y : abs_y + gh, abs_x : abs_x + gw]
        line_glyphs.append((abs_x, abs_y, gw, gh, char_img, line_id))

    # Sort left to right so advance widths are meaningful
    line_glyphs.sort(key=lambda g: g[0])
    glyphs.extend(line_glyphs)

def overlaps_mask_vertically(ay, gh):
    return ay < mask_y + mask_h and ay + gh > mask_y


left_neighbor = None  # closest glyph whose right edge <= mask_x
right_neighbor = None  # closest glyph whose left edge >= mask_x + mask_w

for glyph in glyphs:
    abs_x, abs_y, gw, gh, _, _ = glyph
    if not overlaps_mask_vertically(abs_y, gh):
        continue
    right_edge = abs_x + gw
    left_edge = abs_x
    if right_edge <= mask_x:
        if left_neighbor is None or right_edge > left_neighbor[0] + left_neighbor[2]:
            left_neighbor = glyph
    elif left_edge >= mask_x + mask_w:
        if right_neighbor is None or left_edge < right_neighbor[0]:
            right_neighbor = glyph

# Measure inter-glyph gaps per line to extract space width
all_gaps = []
for i in range(len(glyphs) - 1):
    ax, ay, aw, ah, _, a_line = glyphs[i]
    bx, by, bw, bh, _, b_line = glyphs[i + 1]
    if a_line == b_line:
        gap = bx - (ax + aw)
        if gap > 0:
            all_gaps.append(gap)

avg_space = 0
avg_letter_gap = 0
letter_gap_max = 0
word_gaps = []
letter_gaps = []

if all_gaps:
    gaps_array = np.array(all_gaps).reshape(-1, 1)
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(gaps_array)
    centers = km.cluster_centers_.flatten()
    labels = km.labels_

    word_cluster = int(np.argmax(centers))
    letter_cluster = 1 - word_cluster

    word_gaps = [g for g, l in zip(all_gaps, labels) if l == word_cluster]
    letter_gaps = [g for g, l in zip(all_gaps, labels) if l == letter_cluster]

    if word_gaps:
        wg_sorted = sorted(word_gaps)
        q1 = wg_sorted[len(wg_sorted) // 4]
        q3 = wg_sorted[3 * len(wg_sorted) // 4]
        iqr = q3 - q1
        word_gaps = [g for g in word_gaps if q1 - 1.5 * iqr <= g <= q3 + 1.5 * iqr]

    avg_space = int(round(np.median(word_gaps))) if word_gaps else 0
    avg_letter_gap = int(round(np.median(letter_gaps))) if letter_gaps else 0
    letter_gap_max = (centers[letter_cluster] + centers[word_cluster]) / 2

    print(
        f"Cluster centers: letters={centers[letter_cluster]:.1f}px, words={centers[word_cluster]:.1f}px"
    )
    print(
        f"Inter-letter gaps: min={min(letter_gaps)}, max={max(letter_gaps)}, avg={avg_letter_gap} ({len(letter_gaps)} samples)"
    )
    if word_gaps:
        print(
            f"Inter-word gaps: min={min(word_gaps)}, max={max(word_gaps)}, avg={avg_space} ({len(word_gaps)} samples)"
        )
    print(f"Extracted space: {avg_space}px")

# Deduplicate glyphs: compare images, keep only unique ones
MATCH_SIZE = (32, 32)
SIMILARITY_THRESHOLD = 0.95


def normalize_glyph(char_img):
    return cv2.resize(char_img, MATCH_SIZE, interpolation=cv2.INTER_AREA)


unique_glyphs = []  # (abs_x, abs_y, gw, gh, char_img, line_id)
unique_normals = []
glyph_to_unique = {}  # index in glyphs → index in unique_glyphs

for i, (abs_x, abs_y, gw, gh, char_img, line_id) in enumerate(glyphs):
    norm = normalize_glyph(char_img)
    matched_uid = None
    for uid, existing_norm in enumerate(unique_normals):
        score = cv2.matchTemplate(norm, existing_norm, cv2.TM_CCOEFF_NORMED)[0][0]
        if score >= SIMILARITY_THRESHOLD:
            matched_uid = uid
            break
    if matched_uid is None:
        matched_uid = len(unique_glyphs)
        unique_glyphs.append((abs_x, abs_y, gw, gh, char_img, line_id))
        unique_normals.append(norm)
    glyph_to_unique[i] = matched_uid

print(f"Glyphs: {len(glyphs)} total → {len(unique_glyphs)} uniques")

uid_counts = {}
for uid in glyph_to_unique.values():
    uid_counts[uid] = uid_counts.get(uid, 0) + 1

for idx, (abs_x, abs_y, gw, gh, char_img, line_id) in enumerate(unique_glyphs):
    cv2.imwrite(f"chars/char_{idx}.png", char_img)

with open("unique_glyphs.txt", "w") as f:
    for idx in range(len(unique_glyphs)):
        f.write(f"{idx} ({uid_counts.get(idx, 1)} occ)\n")

# Record pair-specific gaps
letter_pair_gaps = {}  # intra-word
word_pair_gaps = {}  # inter-word

for i in range(len(glyphs) - 1):
    ax, ay, aw, ah, _, a_line = glyphs[i]
    bx, by, bw, bh, _, b_line = glyphs[i + 1]

    if a_line != b_line:
        continue

    gap = bx - (ax + aw)

    if gap < 0:
        continue

    uid_a = glyph_to_unique[i]
    uid_b = glyph_to_unique[i + 1]

    # intra-word
    if gap <= letter_gap_max:
        if (uid_a, uid_b) not in letter_pair_gaps:
            letter_pair_gaps[(uid_a, uid_b)] = gap

    # inter-word
    else:
        if (uid_a, uid_b) not in word_pair_gaps:
            word_pair_gaps[(uid_a, uid_b)] = gap


def get_word_gap(left_uid, right_uid):
    """
    Returns the observed word spacing between two glyphs if known.
    Fallback to avg_space otherwise.
    """

    if (left_uid, right_uid) in word_pair_gaps:
        return word_pair_gaps[(left_uid, right_uid)]

    return avg_space


for idx, (abs_x, abs_y, gw, gh, char_img, line_id) in enumerate(glyphs):
    if (left_neighbor and abs_x == left_neighbor[0] and abs_y == left_neighbor[1]) or (
        right_neighbor and abs_x == right_neighbor[0] and abs_y == right_neighbor[1]
    ):
        color = (255, 0, 0)  # blue
    else:
        color = (0, 0, 255)  # red
    cv2.rectangle(img, (abs_x, abs_y), (abs_x + gw, abs_y + gh), color, 1)

# Animate words between the two blue neighbors:
# [left_blue] + [word_gap] + [g1 pair_gap g2 ... gN] + [word_gap] + [right_blue]
if left_neighbor and right_neighbor and unique_glyphs:
    left_uid = None
    right_uid = None

    for i, glyph in enumerate(glyphs):
        if glyph[:4] == left_neighbor[:4]:
            left_uid = glyph_to_unique[i]

        if glyph[:4] == right_neighbor[:4]:
            right_uid = glyph_to_unique[i]

    ln_x, ln_y, ln_w, ln_h = (
        left_neighbor[0],
        left_neighbor[1],
        left_neighbor[2],
        left_neighbor[3],
    )
    rn_x, rn_y, rn_w, rn_h = (
        right_neighbor[0],
        right_neighbor[1],
        right_neighbor[2],
        right_neighbor[3],
    )
    ln_line = left_neighbor[5]

    # Compute baseline for this line
    line_glyphs_info = [(g[1], g[3]) for g in glyphs if g[5] == ln_line]
    heights = [h for _, h in line_glyphs_info]
    median_h = sorted(heights)[len(heights) // 2]
    normal_bottoms = [y + h for y, h in line_glyphs_info if h < median_h * 1.2]
    baseline_y = (
        sorted(normal_bottoms)[len(normal_bottoms) // 2]
        if normal_bottoms
        else ln_y + ln_h
    )

    available_width = rn_x - (ln_x + ln_w)
    # avg_space used as a budget estimate for DFS, actual gaps are computed per sequence via get_word_gap
    inner_width = available_width - avg_space - avg_space

    TOLERANCE = 10
    MAX_FREE = 8

    # Build adjacency from letter_pair_gaps (covers both letter and word transitions
    # large word-boundary gaps are naturally pruned by the inner_width budget)
    adjacency = {}
    for uid_a, uid_b in letter_pair_gaps:
        if uid_a not in adjacency:
            adjacency[uid_a] = []
        adjacency[uid_a].append(uid_b)

    # DFS: find sequences [g1, ..., gN] where every consecutive pair exists in
    # letter_pair_gaps and total width (glyphs + inter-letter gaps) ≈ inner_width.
    valid_sequences = []

    def dfs(uid, placed_width, path, depth):
        if depth > MAX_FREE:
            return
        if abs(placed_width - inner_width) <= TOLERANCE:
            valid_sequences.append(list(path))
        for next_uid in adjacency.get(uid) or []:
            gap = letter_pair_gaps[(uid, next_uid)]
            glyph_w = unique_glyphs[next_uid][2]
            new_width = placed_width + gap + glyph_w
            if new_width <= inner_width + TOLERANCE:
                path.append(next_uid)
                dfs(next_uid, new_width, path, depth + 1)
                path.pop()

    for first_uid in range(len(unique_glyphs)):
        glyph_w = unique_glyphs[first_uid][2]
        if glyph_w <= inner_width + TOLERANCE:
            dfs(first_uid, glyph_w, [first_uid], 1)

    print(f"Valid sequences found: {len(valid_sequences)}")

    if not valid_sequences:
                messagebox.showinfo("No Valid Sequences", "No valid combinations found (missing letter pairs or incompatible space)")
    else:
        img_clean = img.copy()

        def place_glyph(frame, glyph_img, px, base_y):
            gh, gw = glyph_img.shape[:2]
            py = base_y - gh
            glyph_bgr = cv2.cvtColor(glyph_img, cv2.COLOR_GRAY2BGR)
            end_x = min(px + gw, frame.shape[1])
            end_y = min(py + gh, frame.shape[0])
            if py < 0 or px < 0:
                return
            frame[py:end_y, px:end_x] = np.minimum(
                frame[py:end_y, px:end_x], glyph_bgr[: end_y - py, : end_x - px]
            )

        rn_img = right_neighbor[4]
        orig_rn_bgr = img_original[rn_y : rn_y + rn_h, rn_x : rn_x + rn_w]
        orig_rn_gray = (
            cv2.cvtColor(orig_rn_bgr, cv2.COLOR_BGR2GRAY)
            if orig_rn_bgr.ndim == 3
            else orig_rn_bgr
        )

        def score_seq(seq):
            first_uid = seq[0]
            left_space = get_word_gap(left_uid, first_uid)

            cursor = ln_x + ln_w + left_space
            for k, uid in enumerate(seq):
                cursor += unique_glyphs[uid][2]
                if k < len(seq) - 1:
                    gap = letter_pair_gaps.get((seq[k], seq[k + 1]), None)
                    if gap is None:
                        print(
                            f"[FALLBACK] score_seq: pair ({seq[k]}, {seq[k + 1]}) missing → avg_letter_gap={avg_letter_gap}px"
                        )
                        gap = avg_letter_gap
                    cursor += gap

            last_uid = seq[-1]
            right_space = get_word_gap(last_uid, right_uid)

            cursor += right_space
            patch = np.full((rn_h, rn_w), 255, dtype=np.uint8)
            dst_x = cursor - rn_x
            dst_y = (baseline_y - rn_img.shape[0]) - rn_y
            sx1 = max(0, -dst_x)
            sy1 = max(0, -dst_y)
            dx1 = max(0, dst_x)
            dy1 = max(0, dst_y)
            cw = min(rn_img.shape[1] - sx1, rn_w - dx1)
            ch = min(rn_img.shape[0] - sy1, rn_h - dy1)
            if cw > 0 and ch > 0:
                patch[dy1 : dy1 + ch, dx1 : dx1 + cw] = np.minimum(
                    patch[dy1 : dy1 + ch, dx1 : dx1 + cw],
                    rn_img[sy1 : sy1 + ch, sx1 : sx1 + cw],
                )
            if orig_rn_gray.shape != patch.shape or orig_rn_gray.size == 0:
                return None
            return float(
                cv2.matchTemplate(patch, orig_rn_gray, cv2.TM_CCOEFF_NORMED)[0][0]
            )


        animate = messagebox.askyesno(
            "Animation",
            f"Animate the {len(valid_sequences)} possible sequences ?\n(key to skip, ESC or Q to stop)",
        )
        specific_seq = None
        if not animate:
            raw = simpledialog.askstring(
                "Specific Sequence",
                "Enter the UIDs to display (e.g., 9,18,14,1) or leave empty to ignore:",
            )
            if raw and raw.strip():
                try:
                    specific_seq = [int(x.strip()) for x in raw.split(",")]
                except ValueError:
                    messagebox.showerror("Error", "Invalid format, sequence ignored")
        root.destroy()

        def show_zoomable(win_name, frame, state=None):
            """Display frame with mouse-wheel zoom and left-drag pan. Returns the pressed key.
            Keyboard fallback: +/= zoom in, - zoom out, arrows pan.
            Pass a shared state dict to persist zoom/pan across calls."""
            h, w = frame.shape[:2]
            if state is None:
                state = {"zoom": 1.0, "ox": 0, "oy": 0, "drag": False, "lx": 0, "ly": 0}

            def clamp():
                z = state["zoom"]
                vw = max(1, int(w / z))
                vh = max(1, int(h / z))
                state["ox"] = max(0, min(state["ox"], w - vw))
                state["oy"] = max(0, min(state["oy"], h - vh))

            def render():
                clamp()
                z = state["zoom"]
                ox, oy = state["ox"], state["oy"]
                vw = max(1, int(w / z))
                vh = max(1, int(h / z))
                crop = frame[oy : oy + vh, ox : ox + vw]
                cv2.imshow(
                    win_name, cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                )

            def zoom_at(factor, cx, cy):
                z = state["zoom"]
                vw = max(1, int(w / z))
                vh = max(1, int(h / z))
                new_z = max(1.0, min(z * factor, 20.0))
                img_x = state["ox"] + int(cx * vw / w)
                img_y = state["oy"] + int(cy * vh / h)
                state["zoom"] = new_z
                new_vw = max(1, int(w / new_z))
                new_vh = max(1, int(h / new_z))
                state["ox"] = img_x - int(cx * new_vw / w)
                state["oy"] = img_y - int(cy * new_vh / h)
                render()

            def mouse_cb(event, x, y, flags, _):
                if event == cv2.EVENT_MOUSEWHEEL:
                    # delta encoded in high 16 bits as a signed short
                    raw = (flags >> 16) & 0xFFFF
                    delta = raw if raw < 0x8000 else raw - 0x10000
                    zoom_at(1.15 if delta > 0 else 1 / 1.15, x, y)
                elif event == cv2.EVENT_LBUTTONDOWN:
                    state["drag"] = True
                    state["lx"], state["ly"] = x, y
                elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
                    z = state["zoom"]
                    vw = max(1, int(w / z))
                    vh = max(1, int(h / z))
                    state["ox"] -= int((x - state["lx"]) * vw / w)
                    state["oy"] -= int((y - state["ly"]) * vh / h)
                    state["lx"], state["ly"] = x, y
                    render()
                elif event == cv2.EVENT_LBUTTONUP:
                    state["drag"] = False

            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(win_name, mouse_cb)
            render()

            while True:
                key = cv2.waitKey(20)
                if key == -1:
                    continue
                ch = key & 0xFF
                if ch in (ord("+"), ord("=")):
                    zoom_at(1.2, w // 2, h // 2)
                elif ch == ord("-"):
                    zoom_at(1 / 1.2, w // 2, h // 2)
                elif key in (0, 82, 63232, 65362):  # up
                    step = max(1, int(50 / state["zoom"]))
                    state["oy"] -= step
                    render()
                elif key in (1, 84, 63233, 65364):  # down
                    step = max(1, int(50 / state["zoom"]))
                    state["oy"] += step
                    render()
                elif key in (2, 81, 63234, 65361):  # left
                    step = max(1, int(50 / state["zoom"]))
                    state["ox"] -= step
                    render()
                elif key in (3, 83, 63235, 65363):  # right
                    step = max(1, int(50 / state["zoom"]))
                    state["ox"] += step
                    render()
                elif ch == 0:
                    pass  # other special key (F-keys, etc.) ignored
                else:
                    return ch

        anim_scale = min(1200 / img.shape[1], 1200 / img.shape[0], 1.0)

        if animate:
            _zoom_state = {
                "zoom": 1.0,
                "ox": 0,
                "oy": 0,
                "drag": False,
                "lx": 0,
                "ly": 0,
            }
            for seq_i, seq in enumerate(valid_sequences, 1):
                seq_score = score_seq(seq)
                score_str = f"{seq_score:.4f}" if seq_score is not None else "N/A"
                print(
                    f"[{seq_i}/{len(valid_sequences)}] UIDs: {seq} | Score: {score_str}"
                )
                np.copyto(img, img_clean)
                wg_left = (
                    get_word_gap(left_uid, seq[0])
                    if left_uid is not None
                    else avg_space
                )
                cursor = ln_x + ln_w + wg_left
                for k, uid in enumerate(seq):
                    place_glyph(img, unique_glyphs[uid][4], cursor, baseline_y)
                    cursor += unique_glyphs[uid][2]
                    if k < len(seq) - 1:
                        cursor += letter_pair_gaps[(seq[k], seq[k + 1])]
                wg_right = (
                    get_word_gap(seq[-1], right_uid)
                    if right_uid is not None
                    else avg_space
                )
                cursor += wg_right
                place_glyph(img, right_neighbor[4], cursor, baseline_y)
                label = f"[{seq_i}/{len(valid_sequences)}] {seq} | Score: {score_str}"
                cv2.putText(
                    img,
                    label,
                    (ln_x, baseline_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )
                if show_zoomable(
                    "Sequence Animation",
                    cv2.resize(img, None, fx=anim_scale, fy=anim_scale),
                    _zoom_state,
                ) in (27, ord("q")):
                    break
            cv2.destroyWindow("Sequence Animation")
        elif specific_seq is not None:
            np.copyto(img, img_clean)
            save_img = img_original.copy()
            save_img[mask_y : mask_y + mask_h, mask_x : mask_x + mask_w] = 255
            wg_left = (
                get_word_gap(left_uid, specific_seq[0])
                if left_uid is not None and specific_seq
                else avg_space
            )
            cursor = ln_x + ln_w + wg_left
            for k, uid in enumerate(specific_seq):
                if uid < len(unique_glyphs):
                    place_glyph(img, unique_glyphs[uid][4], cursor, baseline_y)
                    place_glyph(save_img, unique_glyphs[uid][4], cursor, baseline_y)
                    cursor += unique_glyphs[uid][2]
                    if k < len(specific_seq) - 1:
                        gap = letter_pair_gaps.get(
                            (specific_seq[k], specific_seq[k + 1]), None
                        )
                        if gap is None:
                            print(
                                f"[FALLBACK] specific_seq: pair ({specific_seq[k]}, {specific_seq[k + 1]}) not found → avg_letter_gap={avg_letter_gap}px"
                            )
                            gap = avg_letter_gap
                        cursor += gap
            wg_right = (
                get_word_gap(specific_seq[-1], right_uid)
                if right_uid is not None and specific_seq
                else avg_space
            )
            print("wg_left:", wg_left, "wg_right:", wg_right)
            cursor += wg_right
            place_glyph(img, right_neighbor[4], cursor, baseline_y)
            place_glyph(save_img, right_neighbor[4], cursor, baseline_y)
            seq_label = "_".join(str(u) for u in specific_seq)
            cv2.imwrite(f"sequence_{seq_label}.png", save_img)
            print(f"Sequence saved : sequence_{seq_label}.png")
            specific_score = score_seq(specific_seq)
            score_str = f"{specific_score:.4f}" if specific_score is not None else "N/A"
            print(
                f"Sequence {specific_seq} | Score : {score_str} — press a key to continue..."
            )
            show_zoomable(
                "Specific sequence",
                cv2.resize(img, None, fx=anim_scale, fy=anim_scale),
            )
            cv2.destroyWindow("Specific sequence")

        np.copyto(img, img_clean)

        def render_seq_patch(seq):
            """Render the glyphs of a sequence into a patch the size of the masked area"""
            patch = np.full((mask_h, mask_w), 255, dtype=np.uint8)
            wg_left = (
                get_word_gap(left_uid, seq[0]) if left_uid is not None else avg_space
            )
            cursor = ln_x + ln_w + wg_left
            for k, uid in enumerate(seq):
                g = unique_glyphs[uid]
                gw, gh, glyph = g[2], g[3], g[4]
                dst_x = cursor - mask_x
                dst_y = (baseline_y - gh) - mask_y
                sx1 = max(0, -dst_x)
                sy1 = max(0, -dst_y)
                dx1 = max(0, dst_x)
                dy1 = max(0, dst_y)
                cw = min(gw - sx1, mask_w - dx1)
                ch = min(gh - sy1, mask_h - dy1)
                if cw > 0 and ch > 0:
                    patch[dy1 : dy1 + ch, dx1 : dx1 + cw] = np.minimum(
                        patch[dy1 : dy1 + ch, dx1 : dx1 + cw],
                        glyph[sy1 : sy1 + ch, sx1 : sx1 + cw],
                    )
                cursor += gw
                if k < len(seq) - 1:
                    cursor += letter_pair_gaps[(seq[k], seq[k + 1])]
            return patch

        all_scores = []
        for seq_idx, seq in enumerate(valid_sequences):
            score = score_seq(seq)
            if score is not None:
                all_scores.append((score, seq_idx, list(seq)))

        perfect = [x for x in all_scores if x[0] >= 1.0 - 1e-5]
        print(f"Sequences with score 1.0 : {len(perfect)}")
        top_results = sorted(perfect, key=lambda x: x[1])

        use_ocr = messagebox.askyesno(
            "OCR and dictionary",
            "Do you want to continue with OCR and dictionary filtering?",
        )

        if use_ocr:
            TESSERACT = shutil.which("tesseract") or "tesseract"
            _dict_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "dictionary.txt"
            )
            dictionary = set()
            if os.path.exists(_dict_path):
                with open(_dict_path, encoding="utf-8") as _f:
                    dictionary = set(w.strip().lower() for w in _f if w.strip())
            else:
                print(
                    f"Warning: Dictionary not found ({_dict_path}), filtering disabled"
                )

            def ocr_patch(patch):
                up = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                try:
                    cv2.imwrite(tmp_path, up)
                    res = subprocess.run(
                        [TESSERACT, tmp_path, "stdout", "--psm", "8"],
                        capture_output=True,
                        text=True,
                    )
                    text = res.stdout.strip().lower()
                    return "".join(c for c in text if c.isalpha())
                finally:
                    os.unlink(tmp_path)

            ocr_dlg = Toplevel(root)
            ocr_dlg.title("Processing in progress")
            ocr_dlg.resizable(False, False)
            Label(
                ocr_dlg,
                text="OCR and dictionary filtering in progress…\nPlease wait.",
                padx=20,
                pady=20,
            ).pack()
            ocr_dlg.update()

            dict_results = []
            glyph_to_letter = {}

            for score, seq_idx, seq in top_results:
                glyph_ids = seq

                cached_word = []
                missing = False

                for g in glyph_ids:
                    if g in glyph_to_letter:
                        cached_word.append(glyph_to_letter[g])
                    else:
                        missing = True
                        break

                if not missing:
                    word = "".join(cached_word)
                else:
                    word = ocr_patch(render_seq_patch(glyph_ids))

                    if word and len(word) == len(glyph_ids):
                        for g, c in zip(glyph_ids, word):
                            glyph_to_letter[g] = c
                print(f"  OCR: '{word}' | UIDs: {seq}")
                if word and (not dictionary or word in dictionary):
                    dict_results.append((score, seq_idx, seq, word))

            ocr_dlg.destroy()

            print(f"Valid words in the dictionary : {len(dict_results)}")
            for rank, (score, _, seq, word) in enumerate(dict_results, 1):
                print(f"  #{rank} Score: {score:.3f} | UIDs: {seq} | Word: '{word}'")
        else:
            print(
                "OCR and dictionary filtering ignored — displaying all sequences with score 1.0"
            )
            dict_results = [
                (score, seq_idx, seq, "") for score, seq_idx, seq in top_results
            ]

        # LLM ranking stage: OCR the full image then ask Ollama to pick the best candidate
        OLLAMA_MODEL = "qwen3.5:4b"
        candidates = [word for _, _, _, word in dict_results]
        llm_ranking = []

        if use_ocr:
            use_llm = messagebox.askyesno(
                "LLM Ranking", "Do you want to continue with LLM ranking?"
            )
        else:
            use_llm = False

        if use_llm and len(dict_results) > 1:
            llm_dlg = Toplevel(root)
            llm_dlg.title("Processing in progress")
            llm_dlg.resizable(False, False)
            Label(
                llm_dlg,
                text="LLM ranking in progress…\nPlease wait.",
                padx=20,
                pady=20,
            ).pack()
            llm_dlg.update()

            print("\nOCR of the complete image for context...")
            tsv_res = subprocess.run(
                [TESSERACT, "image_clean.png", "stdout", "--psm", "6", "tsv"],
                capture_output=True,
                text=True,
            )
            word_entries = []
            for row in tsv_res.stdout.splitlines()[1:]:
                parts = row.split("\t")
                if len(parts) < 12 or int(parts[0]) != 5:
                    continue
                text = parts[11].strip()
                if not text:
                    continue
                word_entries.append(
                    {
                        "left": int(parts[6]),
                        "top": int(parts[7]),
                        "width": int(parts[8]),
                        "height": int(parts[9]),
                        "text": text,
                        "line_key": (int(parts[2]), int(parts[3]), int(parts[4])),
                    }
                )
            word_entries.sort(key=lambda w: (w["line_key"], w["left"]))
            result_words = []
            here_inserted = False
            for w in word_entries:
                on_mask_line = (
                    w["top"] < mask_y + mask_h and w["top"] + w["height"] > mask_y
                )
                if not here_inserted and on_mask_line and w["left"] >= mask_x + mask_w:
                    result_words.append("[HERE]")
                    here_inserted = True
                result_words.append(w["text"])
            if not here_inserted:
                result_words.append("[HERE]")
            full_text = " ".join(result_words)

            prompt = (
                f"You must select exactly one word from this list — no other answer is acceptable:\n"
                f"{candidates}\n\n"
                "Rules:\n"
                "1. Your answer MUST be one of the words above. No exceptions.\n"
                "2. Do not output a word that is not in the list.\n"
                "3. The blank is marked [HERE] in the text below.\n\n"
                f'Text: "{full_text}"\n\n'
                "Which word from the list fits [HERE]? Reply with only that word."
            )
            payload = json.dumps(
                {
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "think": True,
                    "options": {"num_ctx": 16384},
                }
            ).encode()
            try:
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    llm_response = (
                        json.loads(resp.read()).get("response", "").strip().lower()
                    )
                print("===>", llm_response)
                candidates_set = set(candidates)
                for line in llm_response.splitlines():
                    word = "".join(c for c in line.strip().lower() if c.isalpha())
                    if word in candidates_set and word not in llm_ranking:
                        llm_ranking.append(word)
                print(f"LLM ranking : {llm_ranking}")
            except urllib.error.HTTPError as e:
                sys.exit(f"LLM HTTP error {e.code} : {e.read().decode()}")
            except urllib.error.URLError as e:
                sys.exit(f"Error : Ollama unreachable  — {e.reason}")
            except Exception as e:
                sys.exit(f"LLM error : {e}")
            llm_dlg.destroy()
        elif len(dict_results) == 1:
            llm_ranking = [dict_results[0][3]]
            print(f"\nUnique valid word : '{llm_ranking[0]}'")

        # Build display order with LLM-ranked first
        llm_set = set(llm_ranking)
        word_to_result = {
            word: (score, idx, seq) for score, idx, seq, word in dict_results
        }
        display_order = []
        for word in llm_ranking:
            if word in word_to_result:
                display_order.append((*word_to_result[word], word))
        for score, idx, seq, word in dict_results:
            if word not in llm_set:
                display_order.append((score, idx, seq, word))

        display_order.sort(key=lambda x: x[0], reverse=True)

        if display_order:
            print(
                f"\n{len(display_order)} matches found"
            )
            for rank, (score, _, seq, word) in enumerate(display_order, 1):
                llm_rank = llm_ranking.index(word) + 1 if word in llm_set else None
                tag = f" ★ LLM #{llm_rank}" if llm_rank else ""
                print(f"#{rank} Score: {score:.3f} | UIDs: {seq} | Word: '{word}'{tag}")

            display_scale_anim = min(1200 / img.shape[1], 1200 / img.shape[0], 1.0)
            zoom_state = {
                "zoom": 1.0,
                "ox": 0,
                "oy": 0,
                "drag": False,
                "lx": 0,
                "ly": 0,
            }
            for rank, (score, _, seq, word) in enumerate(display_order, 1):
                llm_rank = llm_ranking.index(word) + 1 if word in llm_set else None
                np.copyto(img, img_clean)
                wg_left = (
                    get_word_gap(left_uid, seq[0])
                    if left_uid is not None
                    else avg_space
                )
                cursor = ln_x + ln_w + wg_left
                for k, uid in enumerate(seq):
                    place_glyph(img, unique_glyphs[uid][4], cursor, baseline_y)
                    cursor += unique_glyphs[uid][2]
                    if k < len(seq) - 1:
                        cursor += letter_pair_gaps[(seq[k], seq[k + 1])]
                wg_right = (
                    get_word_gap(seq[-1], right_uid)
                    if right_uid is not None
                    else avg_space
                )
                cursor += wg_right
                place_glyph(img, right_neighbor[4], cursor, baseline_y)
                label = f"#{rank} match: {score:.2f} | '{word}'"
                if llm_rank:
                    label += f" ★ LLM #{llm_rank}"
                cv2.putText(
                    img,
                    label,
                    (ln_x, baseline_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )
                display = cv2.resize(
                    img, None, fx=display_scale_anim, fy=display_scale_anim
                )
                print(
                    f"Display #{rank} '{word}' press a key to continue"
                )
                show_zoomable("Top sequences", display, zoom_state)

            cv2.destroyAllWindows()
            cv2.imwrite("visualisation_glyphs.png", img)
            if sys.platform == "win32":
                os.startfile("visualisation_glyphs.png")
            elif sys.platform == "darwin":
                subprocess.run(["open", "visualisation_glyphs.png"])
            else:
                subprocess.run(["xdg-open", "visualisation_glyphs.png"])
        else:
            print("No valid scores calculated.")
