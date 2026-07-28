#!/usr/bin/env python3
"""
unified_masker.py

Unifies:
1) YOLOv8 segmentation to detect PERSON instances + left/middle/right labeling
2) InsightFace FaceAnalysis for accurate FACE masks using 106-landmark convex hull
3) CLI inputs:
   --choose   : who to target (left|middle|right|p0|p1|... or index 0..N-1)
   --mask     : what to mask (face|body|both)
   --effect   : how to apply (blur|pixelate|black) [optional; default blur]
Outputs:
- out/overlay.png                    (debug overlay)
- out/person_<label>_mask.png        (instance body mask)
- out/face_<label>_mask.png          (accurate face mask per person if available)
- out/chosen_mask.png                (final chosen mask based on --mask)
- out/masked.png                     (image with masking effect applied)

Notes:
- Requires: ultralytics, insightface, onnxruntime-gpu (optional but recommended), opencv-python, numpy
- If you don't have GPU providers, it will fallback to CPU.

Example:
  python unified_masker.py --image TestImage.png --choose middle --mask both --effect blur --outdir out

"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Dict, Optional, Tuple

import cv2
import numpy as np
import json

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis
    from ultralytics import YOLO


# ---------------------------
# Data models
# ---------------------------

@dataclass
class PersonInstance:
    idx: int
    label: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]
    mask: np.ndarray  # HxW uint8 (0/255)


@dataclass
class FaceInstance:
    bbox_xyxy: Tuple[int, int, int, int]
    score: float
    landmarks_106: Optional[np.ndarray]  # (106,2) float
    mask: np.ndarray  # HxW uint8 (0/255)


# ---------------------------
# Utilities
# ---------------------------


def letterbox_contain(image, target_size, interp):
    """
    Resize image to fit inside target_size x target_size with letterboxing.
    Returns: resized_image, transform_meta
    """
    h, w = image.shape[:2]
    T = target_size

    scale = min(T / w, T / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((T, T, image.shape[2]), dtype=image.dtype)
    pad_x = (T - new_w) // 2
    pad_y = (T - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "mode": "contain",
        "orig_w": w,
        "orig_h": h,
        "target": T,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "new_w": new_w,
        "new_h": new_h
    }

    return canvas, meta


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def label_people_left_to_right(people: List[PersonInstance]) -> List[PersonInstance]:
    if not people:
        return people

    def cx(p: PersonInstance) -> float:
        x1, _, x2, _ = p.bbox_xyxy
        return (x1 + x2) / 2.0

    people_sorted = sorted(people, key=cx)

    if len(people_sorted) == 3:
        labels = ["left", "middle", "right"]
    else:
        labels = [f"p{i}" for i in range(len(people_sorted))]

    relabeled: List[PersonInstance] = []
    for new_i, (p, lab) in enumerate(zip(people_sorted, labels)):
        relabeled.append(
            PersonInstance(
                idx=new_i,
                label=lab,
                score=p.score,
                bbox_xyxy=p.bbox_xyxy,
                mask=p.mask,
            )
        )
    return relabeled


def parse_person_choice(choice: str, people: List[PersonInstance]) -> Optional[PersonInstance]:
    c = choice.strip().lower()

    for p in people:
        if p.label.lower() == c:
            return p

    if c.isdigit():
        idx = int(c)
        if 0 <= idx < len(people):
            return people[idx]

    return None


# ---------------------------
# YOLO: person instance segmentation
# ---------------------------

def yolo_person_instances(
    model: Any,
    image_bgr: np.ndarray,
    conf: float = 0.4,
    device: str | None = None,
) -> List[PersonInstance]:
    H, W = image_bgr.shape[:2]
    predict_options: dict[str, object] = {
        "conf": conf,
        "verbose": False,
    }
    if device is not None:
        predict_options["device"] = device
    results = model.predict(image_bgr, **predict_options)

    if not results or results[0] is None:
        return []

    r = results[0]
    if r.boxes is None or r.masks is None:
        return []

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    # Ultralytics seg polygons per instance
    polys = r.masks.xy

    people: List[PersonInstance] = []
    for i in range(len(boxes)):
        if classes[i] != 0:  # 0 == person
            continue

        x1, y1, x2, y2 = boxes[i]
        x1i = clamp(int(round(x1)), 0, W - 1)
        y1i = clamp(int(round(y1)), 0, H - 1)
        x2i = clamp(int(round(x2)), 0, W - 1)
        y2i = clamp(int(round(y2)), 0, H - 1)

        mask_u8 = np.zeros((H, W), dtype=np.uint8)
        poly = polys[i]
        if poly is not None and len(poly) >= 3:
            pts = np.round(poly).astype(np.int32)
            cv2.fillPoly(mask_u8, [pts], 255)

        people.append(
            PersonInstance(
                idx=len(people),
                label=f"p{len(people)}",
                score=float(scores[i]),
                bbox_xyxy=(x1i, y1i, x2i, y2i),
                mask=mask_u8,
            )
        )

    return people


# ---------------------------
# InsightFace: face detection + landmark mask
# ---------------------------

def landmarks_to_mask(landmarks: np.ndarray, h: int, w: int, expand: float = 1.15) -> np.ndarray:
    """
    landmarks: (106,2) float
    Returns HxW uint8 mask (0/255)
    """
    pts = landmarks.astype(np.int32)
    hull = cv2.convexHull(pts)

    # Expand hull slightly (privacy-safe)
    cx, cy = hull.mean(axis=0)[0]
    hull = ((hull - [cx, cy]) * expand + [cx, cy]).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def init_face_app(prefer_gpu: bool = True) -> Any:
    """
    Attempts GPU if available, else CPU.
    """
    providers = None
    if prefer_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    from insightface.app import FaceAnalysis

    insightface_root = Path(
        os.environ.get(
            "INSIGHTFACE_HOME",
            str(Path.home() / ".insightface"),
        )
    ).expanduser()
    local_model = insightface_root / "models" / "buffalo_l"
    if not local_model.is_dir():
        raise RuntimeError(
            "The optional InsightFace buffalo_l model is not installed "
            "locally. DataVeil will not download it automatically."
        )

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(insightface_root),
        providers=providers,
    )
    # det_size can affect speed/quality; 1024 is decent for mid/high-res images
    app.prepare(ctx_id=0 if prefer_gpu else -1, det_size=(1024, 1024))
    return app


def detect_faces_insightface(
    app: Any,
    image_bgr: np.ndarray,
    expand: float = 1.15,
) -> List[FaceInstance]:
    H, W = image_bgr.shape[:2]
    faces = app.get(image_bgr)
    out: List[FaceInstance] = []

    for f in faces:
        # bbox
        x1, y1, x2, y2 = map(int, f.bbox)
        x1 = clamp(x1, 0, W - 1)
        y1 = clamp(y1, 0, H - 1)
        x2 = clamp(x2, 0, W - 1)
        y2 = clamp(y2, 0, H - 1)

        # score (may be in f.det_score)
        score = float(getattr(f, "det_score", 0.0))

        lm = getattr(f, "landmark_2d_106", None)
        if lm is None:
            # fallback: no landmarks -> empty mask
            mask = np.zeros((H, W), dtype=np.uint8)
            out.append(FaceInstance((x1, y1, x2, y2), score, None, mask))
            continue

        mask = landmarks_to_mask(lm, H, W, expand=expand)
        out.append(FaceInstance((x1, y1, x2, y2), score, lm, mask))

    return out


def assign_faces_to_people(
    people: List[PersonInstance],
    faces: List[FaceInstance],
    min_iou: float = 0.02,
) -> Dict[str, Optional[FaceInstance]]:
    """
    Assign each face to the person with highest IoU (face bbox vs person bbox).
    Keeps the highest-score face per person.
    """
    assigned: Dict[str, Optional[FaceInstance]] = {p.label: None for p in people}

    for face in faces:
        best_label = None
        best_iou = 0.0
        for p in people:
            iou = compute_iou(p.bbox_xyxy, face.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_label = p.label

        if best_label is None or best_iou < min_iou:
            continue

        curr = assigned.get(best_label)
        if curr is None or face.score > curr.score:
            assigned[best_label] = face

    return assigned


# ---------------------------
# Mask composition + application
# ---------------------------

def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("No masks to combine.")
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out = cv2.bitwise_or(out, m)
    return out


def apply_effect(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    effect: str = "blur",
    strength: int = 65,
    color: str = "#101815",
) -> np.ndarray:
    """
    Applies masking effect on pixels where mask is 255.
    effect: blur | pixelate | black
    """
    if mask_u8.ndim != 2:
        raise ValueError("mask_u8 must be HxW uint8")

    strength = max(1, min(100, int(strength)))
    out = image_bgr.copy()
    m = mask_u8.astype(bool)

    if effect in {"black", "silhouette"}:
        opacity = strength / 100.0
        dark = np.zeros_like(out)
        blended = cv2.addWeighted(out, 1.0 - opacity, dark, opacity, 0)
        out[m] = blended[m]
        return out

    if effect in {"pixelate", "mosaic"}:
        H, W = out.shape[:2]
        divisor = 3 + int(strength * 0.75)
        if effect == "mosaic":
            divisor = 5 + int(strength * 1.1)
        small_w = max(3, W // max(3, divisor))
        small_h = max(3, H // max(3, divisor))
        small = cv2.resize(out, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        if effect == "mosaic":
            pix = (pix // 32) * 32
        out[m] = pix[m]
        return out

    if effect == "solid":
        value = color.lstrip("#")
        if len(value) != 6:
            value = "101815"
        try:
            red, green, blue = (
                int(value[0:2], 16),
                int(value[2:4], 16),
                int(value[4:6], 16),
            )
        except ValueError:
            red, green, blue = (16, 24, 21)
        fill = np.empty_like(out)
        fill[:, :] = (blue, green, red)
        opacity = strength / 100.0
        blended = cv2.addWeighted(out, 1.0 - opacity, fill, opacity, 0)
        out[m] = blended[m]
        return out

    if effect == "noise":
        rng = np.random.default_rng()
        noise = rng.integers(0, 256, size=out.shape, dtype=np.uint8)
        opacity = 0.35 + (0.65 * strength / 100.0)
        noisy = cv2.addWeighted(out, 1.0 - opacity, noise, opacity, 0)
        out[m] = noisy[m]
        return out

    if effect == "sticker":
        sticker = out.copy()
        sticker[m] = (42, 113, 78)
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (mask_u8 > 0).astype(np.uint8), 8
        )
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area < 16:
                continue
            font_scale = max(0.35, min(1.2, min(width, height) / 90))
            cv2.putText(
                sticker,
                "PRIVATE",
                (x + 5, y + max(18, height // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        opacity = 0.35 + (0.65 * strength / 100.0)
        blended = cv2.addWeighted(
            out, 1.0 - opacity, sticker, opacity, 0
        )
        out[m] = blended[m]
        return out

    # Default and explicit blur. Kernel scales with the selected percentage.
    min_side = min(out.shape[0], out.shape[1])
    max_kernel = max(15, min(151, (min_side // 6) | 1))
    k = max(3, int(3 + (max_kernel - 3) * strength / 100.0) | 1)
    blurred = cv2.GaussianBlur(out, (k, k), 0)
    out[m] = blurred[m]
    return out


def draw_overlay(
    image_bgr: np.ndarray,
    people: List[PersonInstance],
    face_by_person: Dict[str, Optional[FaceInstance]],
) -> np.ndarray:
    out = image_bgr.copy()

    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for i, p in enumerate(people):
        color = palette[i % len(palette)]
        pmask = p.mask.astype(bool)

        tint = np.zeros_like(out)
        tint[:, :] = color
        out[pmask] = cv2.addWeighted(out, 0.55, tint, 0.45, 0)[pmask]

        x1, y1, x2, y2 = p.bbox_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{p.label} | person {p.score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        face = face_by_person.get(p.label)
        if face is not None:
            fmask = face.mask.astype(bool)
            white = np.zeros_like(out)
            white[:, :] = (255, 255, 255)
            out[fmask] = cv2.addWeighted(out, 0.35, white, 0.65, 0)[fmask]

            fx1, fy1, fx2, fy2 = face.bbox_xyxy
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (255, 255, 255), 2)
            cv2.putText(
                out,
                f"face {face.score:.2f}",
                (fx1, min(out.shape[0] - 10, fy2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return out


# ---------------------------
# Main
# ---------------------------

def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument(
        "--model",
        default=str(Path(__file__).with_name("yolov8n-seg.pt")),
        help="Path to a local YOLO segmentation model",
    )
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold")
    ap.add_argument("--choose", default="middle", help="left|middle|right|p0|1 etc.")
    ap.add_argument("--mask", default="face", choices=["face", "body", "both"], help="What to mask")
    ap.add_argument("--effect", default="blur", choices=["blur", "pixelate", "black"], help="Masking effect")
    ap.add_argument("--prefer_gpu", action="store_true", help="Prefer GPU providers where possible")
    ap.add_argument("--face_expand", type=float, default=1.15, help="Expand face hull for privacy")
    ap.add_argument("--min_face_iou", type=float, default=0.02, help="Min IoU to assign face->person")
    ap.add_argument("--prepare_for_ai", action="store_true",help="Export masked image + mask in AI-friendly resolution with transform metadata")
    ap.add_argument("--ai_size", type=int, default=1024, help="Target size for AI export (square, default 1024)")
    ap.add_argument(
        "--dataveil-json-file",
        default=None,
        help="Optional path for structured DataVeil runtime JSON output"
    )

    args = ap.parse_args()

    ensure_dir(args.outdir)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    # 1) Person segmentation
    yolo = YOLO(args.model)
    people = yolo_person_instances(yolo, img, conf=args.conf)
    people = label_people_left_to_right(people)

    if not people:
        raise SystemExit("No persons detected. Try lowering --conf or use a clearer image.")

    # 2) Face detection (InsightFace)
    face_app = init_face_app(prefer_gpu=args.prefer_gpu)
    faces = detect_faces_insightface(face_app, img, expand=args.face_expand)
    face_by_person = assign_faces_to_people(people, faces, min_iou=args.min_face_iou)

    # 3) Choose target person
    chosen = parse_person_choice(args.choose, people)
    if chosen is None:
        available = ", ".join([p.label for p in people])
        raise SystemExit(f"Invalid --choose '{args.choose}'. Available: {available} or indices 0..{len(people)-1}")

    chosen_face = face_by_person.get(chosen.label)

    # 4) Export per-person masks
    for p in people:
        cv2.imwrite(os.path.join(args.outdir, f"person_{p.label}_mask.png"), p.mask)

    for p in people:
        f = face_by_person.get(p.label)
        if f is None:
            blank = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.imwrite(os.path.join(args.outdir, f"face_{p.label}_mask.png"), blank)
        else:
            cv2.imwrite(os.path.join(args.outdir, f"face_{p.label}_mask.png"), f.mask)

    # 5) Build chosen mask based on --mask
    masks_to_merge: List[np.ndarray] = []
    if args.mask in ("body", "both"):
        masks_to_merge.append(chosen.mask)

    if args.mask in ("face", "both"):
        if chosen_face is None:
            # no face assigned -> blank face mask
            masks_to_merge.append(np.zeros(img.shape[:2], dtype=np.uint8))
        else:
            masks_to_merge.append(chosen_face.mask)

    chosen_mask = combine_masks(masks_to_merge)
    cv2.imwrite(os.path.join(args.outdir, "chosen_mask.png"), chosen_mask)

    # 6) Apply effect to generate final masked image
    masked = apply_effect(img, chosen_mask, effect=args.effect)
    cv2.imwrite(os.path.join(args.outdir, "masked.png"), masked)

    # 7) Debug overlay
    overlay = draw_overlay(img, people, face_by_person)
    cv2.imwrite(os.path.join(args.outdir, "overlay.png"), overlay)

    if args.prepare_for_ai:
        T = args.ai_size

        # Masked image → AI space
        masked_ai, meta_img = letterbox_contain(
            masked, T, interp=cv2.INTER_LINEAR
        )

        # Mask → AI space (single channel)
        mask_3c = cv2.cvtColor(chosen_mask, cv2.COLOR_GRAY2BGR)
        mask_ai_3c, meta_mask = letterbox_contain(
            mask_3c, T, interp=cv2.INTER_NEAREST
        )
        mask_ai = cv2.cvtColor(mask_ai_3c, cv2.COLOR_BGR2GRAY)

        # Save outputs
        cv2.imwrite(os.path.join(args.outdir, "masked_for_ai.png"), masked_ai)
        cv2.imwrite(os.path.join(args.outdir, "mask_for_ai.png"), mask_ai)

        with open(os.path.join(args.outdir, "transform.json"), "w") as f:
            json.dump(meta_img, f, indent=2)

        print("AI-prep exports written:")
        print(" - masked_for_ai.png")
        print(" - mask_for_ai.png")
        print(" - transform.json")

    if args.dataveil_json_file:
        def normalize_artifact(path: str) -> str:
            return os.path.abspath(path).replace("\\", "/")

        original_h, original_w = img.shape[:2]
        original_aspect_ratio = float(original_w) / float(original_h) if original_h > 0 else 1.0
        delivery_aspect_ratio = 1.0

        people_payload = []
        for person in people:
            x1, y1, x2, y2 = person.bbox_xyxy
            people_payload.append({
                "label": person.label,
                "confidence": float(person.score),
                "hasFace": face_by_person.get(person.label) is not None,
                "bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(max(0, x2 - x1)),
                    "height": int(max(0, y2 - y1))
                }
            })

        payload = {
            "artifacts": {
                "mask": normalize_artifact(os.path.join(args.outdir, "chosen_mask.png")),
                "overlay": normalize_artifact(os.path.join(args.outdir, "overlay.png")),
                "masked": normalize_artifact(os.path.join(args.outdir, "masked.png")),
                "transform": normalize_artifact(os.path.join(args.outdir, "transform.json"))
            },
            "geometry": {
                "originalWidth": int(original_w),
                "originalHeight": int(original_h),
                "originalAspectRatio": original_aspect_ratio,
                "deliveryAspectRatio": delivery_aspect_ratio
            },
            "people": people_payload,
            "selectedPersonLabel": chosen.label
        }
        with open(args.dataveil_json_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)


    # Console summary
    print("\nDetected people (left->right):")
    for p in people:
        x1, y1, x2, y2 = p.bbox_xyxy
        has_face = face_by_person.get(p.label) is not None
        print(f" - {p.label}: score={p.score:.2f}, bbox=({x1},{y1},{x2},{y2}), face={has_face}")

    print(f"\nChosen person: {chosen.label}")
    print(f"Mask mode: {args.mask} | Effect: {args.effect}")
    print(f"Outputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
