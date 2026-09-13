import cv2
import numpy as np
import random
import os
import glob
import json
import yaml

SOURCE_DIR = "source_images"
OUTPUT_DIR = "dataset"
CLASS_NAMES = ["green_crab", "rock_crab", "jonah_crab"]
BOXES_CACHE = os.path.join(SOURCE_DIR, "boxes.json")

COPIES_PER_SOURCE = 150
VAL_SPLIT = 0.15
IMG_SIZE = 640


def infer_class_id(filename):
    lower = filename.lower()
    if "green" in lower:
        return 0
    if "rock" in lower:
        return 1
    if "jonah" in lower:
        return 2
    print(f"\nCould not guess the class for {filename} from its name.")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} = {name}")
    while True:
        choice = input(f"Type the class number for {filename}: ").strip()
        if choice in ("0", "1", "2"):
            return int(choice)
        print("Please type 0, 1, or 2.")


def draw_box_interactively(image_path):
    image = cv2.imread(image_path)
    clone = image.copy()
    state = {"start": None, "end": None, "drawing": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["drawing"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["end"] = (x, y)
            state["drawing"] = False

    window_name = f"Draw a box around the crab in {os.path.basename(image_path)}  (s=save, r=redraw, q=quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        display = clone.copy()
        if state["start"] and state["end"]:
            cv2.rectangle(display, state["start"], state["end"], (0, 0, 255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s") and state["start"] and state["end"]:
            x1, y1 = state["start"]
            x2, y2 = state["end"]
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            cv2.destroyWindow(window_name)
            return box
        elif key == ord("r"):
            state["start"], state["end"] = None, None
        elif key == ord("q"):
            cv2.destroyWindow(window_name)
            raise SystemExit("Quit early, no dataset was built.")


def load_or_create_boxes():
    if os.path.exists(BOXES_CACHE):
        with open(BOXES_CACHE) as f:
            cache = json.load(f)
    else:
        cache = {}

    image_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.jpg")) +
                          glob.glob(os.path.join(SOURCE_DIR, "*.jpeg")) +
                          glob.glob(os.path.join(SOURCE_DIR, "*.png")))
    if not image_paths:
        raise SystemExit(f"No images found in {SOURCE_DIR}/. Put your crab photos there first.")

    changed = False
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        if fname in cache:
            continue
        print(f"\nLabeling {fname} ...")
        class_id = infer_class_id(fname)
        box = draw_box_interactively(img_path)
        cache[fname] = {"class_id": class_id, "box": list(box)}
        changed = True

    if changed:
        with open(BOXES_CACHE, "w") as f:
            json.dump(cache, f, indent=2)

    return cache


def rotate_point(pt, matrix):
    x, y = pt
    new_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    new_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return new_x, new_y


def augment_one(image, box, angle, scale, brightness, contrast):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    x_min, y_min, x_max, y_max = box
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    transformed = [rotate_point(c, matrix) for c in corners]
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    new_box = (max(min(xs), 0), max(min(ys), 0), min(max(xs), w), min(max(ys), h))

    jittered = cv2.convertScaleAbs(rotated, alpha=contrast, beta=brightness)
    return jittered, new_box


def to_yolo_format(box, img_w, img_h):
    x_min, y_min, x_max, y_max = box
    xc = ((x_min + x_max) / 2) / img_w
    yc = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return xc, yc, w, h


def main():
    boxes = load_or_create_boxes()

    for split in ["train", "val"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    total_written = 0
    for fname, info in boxes.items():
        img_path = os.path.join(SOURCE_DIR, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue

        base_name = os.path.splitext(fname)[0]
        class_id = info["class_id"]
        box = tuple(info["box"])

        for i in range(COPIES_PER_SOURCE):
            angle = random.uniform(0, 360)
            scale = random.uniform(0.7, 1.3)
            brightness = random.uniform(-40, 40)
            contrast = random.uniform(0.8, 1.2)

            aug_img, aug_box = augment_one(image, box, angle, scale, brightness, contrast)
            resized = cv2.resize(aug_img, (IMG_SIZE, IMG_SIZE))

            h, w = aug_img.shape[:2]
            scale_x, scale_y = IMG_SIZE / w, IMG_SIZE / h
            final_box = (aug_box[0] * scale_x, aug_box[1] * scale_y,
                         aug_box[2] * scale_x, aug_box[3] * scale_y)

            split = "val" if random.random() < VAL_SPLIT else "train"
            out_name = f"{base_name}_{i:04d}"

            cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{out_name}.jpg", resized)
            yolo_box = to_yolo_format(final_box, IMG_SIZE, IMG_SIZE)
            with open(f"{OUTPUT_DIR}/labels/{split}/{out_name}.txt", "w") as f:
                f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_box)}\n")

            total_written += 1

    data_yaml = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nDone. Wrote {total_written} augmented images across train and val.")
    print(f"Train with: model.train(data='{OUTPUT_DIR}/data.yaml', epochs=100, imgsz={IMG_SIZE})")


if __name__ == "__main__":
    main()