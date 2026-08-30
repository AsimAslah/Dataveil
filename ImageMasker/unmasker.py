#!/usr/bin/env python3
import os
import json
import argparse
import cv2
import numpy as np

ALIGNMENT_MIN_GOOD_MATCHES = 24
ALIGNMENT_MIN_INLIER_RATIO = 0.42
ALIGNMENT_MAX_MEDIAN_REPROJECTION_ERROR_PX = 7.0
ALIGNMENT_MIN_SCALE = 0.82
ALIGNMENT_MAX_SCALE = 1.22
ALIGNMENT_MAX_ROTATION_DEG = 16.0
ALIGNMENT_MAX_TRANSLATION_RATIO = 0.24


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {path}")
    return img


def load_mask_u8(path: str, target_hw: tuple[int, int], threshold: int = 127) -> np.ndarray:
    """
    Load mask as HxW uint8 {0,255}. If size differs, resize with NEAREST.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"Failed to read mask: {path}")

    H, W = target_hw
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    m = (m > threshold).astype(np.uint8) * 255
    return m


def combine_masks(mask_paths: list[str], target_hw: tuple[int, int]) -> np.ndarray:
    out = np.zeros(target_hw, dtype=np.uint8)
    for p in mask_paths:
        m = load_mask_u8(p, target_hw)
        out = cv2.bitwise_or(out, m)
    return out


def apply_contain_transform_to_image(
    src_bgr: np.ndarray,
    meta: dict,
    interp: int,
) -> np.ndarray:
    """
    Reconstruct the exact "contain/letterbox" transform used in Step 1.
    Output is (target x target).
    """
    H, W = src_bgr.shape[:2]

    # sanity check (not mandatory, but helps catch wrong inputs)
    if int(meta["orig_w"]) != W or int(meta["orig_h"]) != H:
        # We don't crash, but warn via exception message clarity:
        raise SystemExit(
            f"transform.json expects orig {meta['orig_w']}x{meta['orig_h']}, "
            f"but you provided {W}x{H}. Use the matching original image."
        )

    T = int(meta["target"])
    new_w = int(meta["new_w"])
    new_h = int(meta["new_h"])
    pad_x = int(meta["pad_x"])
    pad_y = int(meta["pad_y"])

    resized = cv2.resize(src_bgr, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((T, T, 3), dtype=src_bgr.dtype)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def apply_contain_transform_to_mask(
    src_mask_u8: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """
    Same as image transform but for single-channel mask, NEAREST only.
    """
    H, W = src_mask_u8.shape[:2]

    if int(meta["orig_w"]) != W or int(meta["orig_h"]) != H:
        raise SystemExit(
            f"transform.json expects orig {meta['orig_w']}x{meta['orig_h']}, "
            f"but mask is {W}x{H}. Use masks from the same original."
        )

    T = int(meta["target"])
    new_w = int(meta["new_w"])
    new_h = int(meta["new_h"])
    pad_x = int(meta["pad_x"])
    pad_y = int(meta["pad_y"])

    resized = cv2.resize(src_mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((T, T), dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas


def dilate_binary_mask(mask_u8: np.ndarray, radius: int = 10) -> np.ndarray:
    if radius <= 0:
        return mask_u8
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask_u8, kernel, iterations=1)


def build_alignment_roi(mask_u8: np.ndarray) -> np.ndarray:
    """
    Build a background ROI mask for feature alignment.
    We avoid the masked face region because that content is expected to change.
    """
    protected = dilate_binary_mask((mask_u8 > 127).astype(np.uint8) * 255, radius=12)
    roi = cv2.bitwise_not(protected)
    foreground_pixels = int(np.count_nonzero(roi))
    total_pixels = int(roi.shape[0] * roi.shape[1])
    if total_pixels <= 0:
        return np.full(mask_u8.shape, 255, dtype=np.uint8)
    # If the protected region swallows almost the entire frame, use full frame alignment.
    if foreground_pixels / total_pixels < 0.12:
        return np.full(mask_u8.shape, 255, dtype=np.uint8)
    return roi


def evaluate_affine_sanity(matrix: np.ndarray, width: int, height: int) -> tuple[bool, dict]:
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]

    scale_x = float(np.sqrt(a * a + c * c))
    scale_y = float(np.sqrt(b * b + d * d))
    scale = (scale_x + scale_y) / 2.0
    rotation_deg = float(abs(np.degrees(np.arctan2(c, a))))
    tx_ratio = float(abs(tx) / max(width, 1))
    ty_ratio = float(abs(ty) / max(height, 1))
    translation_ratio = max(tx_ratio, ty_ratio)

    ok = (
        ALIGNMENT_MIN_SCALE <= scale <= ALIGNMENT_MAX_SCALE
        and rotation_deg <= ALIGNMENT_MAX_ROTATION_DEG
        and translation_ratio <= ALIGNMENT_MAX_TRANSLATION_RATIO
    )
    return ok, {
        "scale": scale,
        "rotationDeg": rotation_deg,
        "translationRatio": translation_ratio
    }


def estimate_orb_alignment_transform(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    protected_mask_u8: np.ndarray
) -> tuple[np.ndarray | None, dict]:
    source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    roi = build_alignment_roi(protected_mask_u8)

    orb = cv2.ORB_create(nfeatures=3200, fastThreshold=12)
    kp_source, des_source = orb.detectAndCompute(source_gray, roi)
    kp_target, des_target = orb.detectAndCompute(target_gray, roi)

    if (
        des_source is None
        or des_target is None
        or len(kp_source) < ALIGNMENT_MIN_GOOD_MATCHES
        or len(kp_target) < ALIGNMENT_MIN_GOOD_MATCHES
    ):
        return None, {
            "method": "orb",
            "applied": False,
            "reason": "insufficient_features",
            "sourceKeypoints": len(kp_source) if kp_source is not None else 0,
            "targetKeypoints": len(kp_target) if kp_target is not None else 0
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_source, des_target, k=2)
    good_matches = []
    for pair in knn:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.78 * second.distance:
            good_matches.append(first)

    if len(good_matches) < ALIGNMENT_MIN_GOOD_MATCHES:
        return None, {
            "method": "orb",
            "applied": False,
            "reason": "insufficient_matches",
            "sourceKeypoints": len(kp_source),
            "targetKeypoints": len(kp_target),
            "goodMatches": len(good_matches)
        }

    src_points = np.float32([kp_source[m.queryIdx].pt for m in good_matches])
    dst_points = np.float32([kp_target[m.trainIdx].pt for m in good_matches])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=6000,
        confidence=0.997,
        refineIters=30
    )
    if matrix is None or inliers is None:
        return None, {
            "method": "orb",
            "applied": False,
            "reason": "affine_estimation_failed",
            "goodMatches": len(good_matches)
        }

    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    if inlier_count == 0:
        return None, {
            "method": "orb",
            "applied": False,
            "reason": "zero_inliers",
            "goodMatches": len(good_matches)
        }
    inlier_ratio = float(inlier_count / max(len(good_matches), 1))

    src_points_h = np.hstack([src_points, np.ones((src_points.shape[0], 1), dtype=np.float32)])
    projected = (matrix @ src_points_h.T).T
    reprojection_error = np.linalg.norm(projected - dst_points, axis=1)
    median_reprojection_error = float(np.median(reprojection_error[inlier_mask]))

    sanity_ok, sanity_metrics = evaluate_affine_sanity(
        matrix,
        width=source_bgr.shape[1],
        height=source_bgr.shape[0]
    )
    if (
        inlier_ratio < ALIGNMENT_MIN_INLIER_RATIO
        or median_reprojection_error > ALIGNMENT_MAX_MEDIAN_REPROJECTION_ERROR_PX
        or not sanity_ok
    ):
        return None, {
            "method": "orb",
            "applied": False,
            "reason": "low_alignment_confidence",
            "goodMatches": len(good_matches),
            "inlierRatio": inlier_ratio,
            "medianReprojectionErrorPx": median_reprojection_error,
            **sanity_metrics
        }

    return matrix.astype(np.float32), {
        "method": "orb",
        "applied": True,
        "goodMatches": len(good_matches),
        "inlierRatio": inlier_ratio,
        "medianReprojectionErrorPx": median_reprojection_error,
        **sanity_metrics
    }


def estimate_ecc_alignment_transform(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    protected_mask_u8: np.ndarray
) -> tuple[np.ndarray | None, dict]:
    source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    roi = build_alignment_roi(protected_mask_u8)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-6
    )
    try:
        correlation, warp = cv2.findTransformECC(
            target_gray,
            source_gray,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=roi,
            gaussFiltSize=5
        )
    except cv2.error:
        return None, {
            "method": "ecc",
            "applied": False,
            "reason": "ecc_failed"
        }

    sanity_ok, sanity_metrics = evaluate_affine_sanity(
        warp,
        width=source_bgr.shape[1],
        height=source_bgr.shape[0]
    )
    if correlation < 0.85 or not sanity_ok:
        return None, {
            "method": "ecc",
            "applied": False,
            "reason": "low_alignment_confidence",
            "correlation": float(correlation),
            **sanity_metrics
        }

    return warp.astype(np.float32), {
        "method": "ecc",
        "applied": True,
        "correlation": float(correlation),
        **sanity_metrics
    }


def alignment_confidence_score(alignment: dict) -> float:
    """
    Normalize alignment quality into a comparable scalar score.
    Returns <= 0 when alignment was not applied.
    """
    if not alignment or not alignment.get("applied"):
        return 0.0

    method = str(alignment.get("method", "")).lower()
    if method == "orb":
        inlier_ratio = float(alignment.get("inlierRatio", 0.0))
        reproj = float(alignment.get("medianReprojectionErrorPx", ALIGNMENT_MAX_MEDIAN_REPROJECTION_ERROR_PX))
        reproj_quality = max(
            0.0,
            1.0 - (reproj / max(ALIGNMENT_MAX_MEDIAN_REPROJECTION_ERROR_PX, 1e-6))
        )
        return (1.8 * inlier_ratio) + (0.9 * reproj_quality)

    if method == "ecc":
        correlation = float(alignment.get("correlation", 0.0))
        return max(0.0, correlation)

    # Unknown method but marked as applied.
    return 0.1


def map_source_to_return(
    source_bgr: np.ndarray,
    source_mask_u8: np.ndarray,
    return_hw: tuple[int, int],
    edited_bgr: np.ndarray,
    source_space: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    source space -> return space
    Start with resize, then try robust geometric alignment against the returned image.
    """
    out_h, out_w = return_hw

    resized_img = cv2.resize(source_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(source_mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    resized_mask = (resized_mask > 127).astype(np.uint8) * 255

    matrix, alignment = estimate_orb_alignment_transform(resized_img, edited_bgr, resized_mask)
    if matrix is None:
        matrix, alignment = estimate_ecc_alignment_transform(resized_img, edited_bgr, resized_mask)
    if matrix is None:
        return resized_img, resized_mask, {
            "method": alignment.get("method", "resize"),
            "applied": False,
            "fallback": "resize_only",
            "sourceSpace": source_space,
            "strategy": "resize_then_align",
            **alignment
        }

    aligned_img = cv2.warpAffine(
        resized_img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )
    aligned_mask = cv2.warpAffine(
        resized_mask,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    aligned_mask = (aligned_mask > 127).astype(np.uint8) * 255

    base_area = max(int(np.count_nonzero(resized_mask)), 1)
    aligned_area = int(np.count_nonzero(aligned_mask))
    area_ratio = float(aligned_area / base_area)
    if area_ratio < 0.35 or area_ratio > 2.85:
        return resized_img, resized_mask, {
            "method": alignment.get("method", "affine"),
            "applied": False,
            "fallback": "resize_only",
            "reason": "mask_area_ratio_out_of_range",
            "maskAreaRatio": area_ratio,
            "sourceSpace": source_space,
            "strategy": "resize_then_align",
            **alignment
        }

    return aligned_img, aligned_mask, {
        **alignment,
        "sourceSpace": source_space,
        "strategy": "resize_then_align",
        "fallback": None,
        "maskAreaRatio": area_ratio
    }


def map_delivery_to_return(
    delivery_bgr: np.ndarray,
    delivery_mask_u8: np.ndarray,
    return_hw: tuple[int, int],
    edited_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Backward-compatible alias for callers that still think in delivery-space terms.
    """
    return map_source_to_return(
        delivery_bgr,
        delivery_mask_u8,
        return_hw,
        edited_bgr,
        source_space="delivery"
    )


def build_feather_alpha(
    mapped_mask_u8: np.ndarray,
    *,
    max_radius: int = 16,
) -> np.ndarray:
    """
    Build a solid restoration core with a short outward feather.

    The protected image can contain a strongly colored privacy effect right
    up to the mask boundary. An inside-only feather mixes those contaminated
    pixels back into the restored edge and creates a visible colored halo.
    Keeping the whole protected region opaque removes that contamination;
    the small outward feather then softens the clean source-to-edit seam.
    """
    binary = (mapped_mask_u8 > 127).astype(np.uint8)
    rows, columns = np.where(binary > 0)
    if not rows.size or not columns.size:
        return np.zeros(mapped_mask_u8.shape[:2], dtype=np.float32)

    region_height = int(rows.max() - rows.min() + 1)
    region_width = int(columns.max() - columns.min() + 1)
    short_side = max(1, min(region_height, region_width))
    feather_radius = max(
        3,
        min(max_radius, int(round(short_side * 0.08))),
    )

    distance_outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    alpha = np.where(
        binary > 0,
        1.0,
        np.clip(
            1.0 - (distance_outside / float(feather_radius)),
            0.0,
            1.0,
        ),
    )
    # Smoothstep avoids a visible change in slope at either end of the blend.
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return alpha.astype(np.float32)


def unmask_only_region(
    edited_bgr: np.ndarray,
    mapped_original_bgr: np.ndarray,
    mapped_mask_u8: np.ndarray,
) -> np.ndarray:
    """
    Restore masked pixels with a soft boundary to avoid visible mask shapes.
    """
    if edited_bgr.shape != mapped_original_bgr.shape:
        raise SystemExit("Internal error: mapped original and edited shapes differ.")
    if mapped_mask_u8.shape[:2] != edited_bgr.shape[:2]:
        raise SystemExit("Internal error: mapped mask and edited shapes differ.")

    alpha = build_feather_alpha(mapped_mask_u8)
    if not np.any(alpha):
        return edited_bgr.copy()

    alpha_3d = alpha[:, :, np.newaxis]
    blended = (
        mapped_original_bgr.astype(np.float32) * alpha_3d
        + edited_bgr.astype(np.float32) * (1.0 - alpha_3d)
    )
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edited", required=True, help="AI output image (may be any resolution)")
    ap.add_argument("--original", required=True, help="Original image (source space)")
    ap.add_argument("--mask", nargs="+", required=True, help="One or more masks in original resolution")
    ap.add_argument("--meta", required=True, help="transform.json generated in Step 1")
    ap.add_argument("--outdir", default="out_unmask", help="Output directory")
    ap.add_argument("--outname", default="unmasked.png", help="Output filename")
    ap.add_argument(
        "--dataveil-json-file",
        default=None,
        help="Optional path for structured DataVeil runtime JSON output"
    )
    args = ap.parse_args()

    ensure_dir(args.outdir)

    edited = load_image_bgr(args.edited)
    original = load_image_bgr(args.original)

    H, W = original.shape[:2]

    # 1) combine masks in ORIGINAL space
    combined_mask = combine_masks(args.mask, (H, W))
    cv2.imwrite(os.path.join(args.outdir, "combined_mask_original.png"), combined_mask)

    eh, ew = edited.shape[:2]

    # 2) direct map ORIGINAL -> RETURN (primary path)
    candidates = []
    direct_original_return, direct_mask_return, direct_alignment = map_source_to_return(
        original,
        combined_mask,
        (eh, ew),
        edited,
        source_space="original_direct"
    )
    candidates.append({
        "name": "original_direct",
        "mappedOriginal": direct_original_return,
        "mappedMask": direct_mask_return,
        "alignment": direct_alignment
    })

    # Keep these diagnostics for compatibility with existing artifacts.
    cv2.imwrite(os.path.join(args.outdir, "original_in_delivery_space.png"), direct_original_return)
    cv2.imwrite(os.path.join(args.outdir, "mask_in_delivery_space.png"), direct_mask_return)

    # 3) optional contain candidate (legacy compatibility with prepared-for-AI delivery space)
    meta_issue = None
    contain_issue = None
    try:
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        meta = None
        meta_issue = f"failed_to_read_meta: {str(exc)}"

    if meta is not None:
        if meta.get("mode") != "contain":
            contain_issue = f"unsupported_meta_mode: {meta.get('mode')}"
        else:
            try:
                original_delivery = apply_contain_transform_to_image(original, meta, interp=cv2.INTER_LINEAR)
                mask_delivery = apply_contain_transform_to_mask(combined_mask, meta)
                contain_original_return, contain_mask_return, contain_alignment = map_source_to_return(
                    original_delivery,
                    mask_delivery,
                    (eh, ew),
                    edited,
                    source_space="contain_delivery"
                )
                candidates.append({
                    "name": "contain_delivery",
                    "mappedOriginal": contain_original_return,
                    "mappedMask": contain_mask_return,
                    "alignment": contain_alignment
                })
            except SystemExit as exc:
                contain_issue = f"contain_candidate_failed: {str(exc)}"

    # 4) choose best candidate:
    #    - prefer any candidate with applied alignment, highest confidence score
    #    - otherwise prefer direct original-space resize fallback
    applied_candidates = [c for c in candidates if c["alignment"].get("applied")]
    if applied_candidates:
        selected = max(applied_candidates, key=lambda c: alignment_confidence_score(c["alignment"]))
        selection_reason = "selected_highest_scoring_applied_alignment"
    else:
        selected = next((c for c in candidates if c["name"] == "original_direct"), candidates[0])
        selection_reason = "no_alignment_applied_prefer_direct_original_mapping"

    original_return = selected["mappedOriginal"]
    mask_return = selected["mappedMask"]
    alignment = dict(selected["alignment"])
    alignment["selectedPath"] = selected["name"]
    alignment["selectionReason"] = selection_reason
    alignment["candidateCount"] = len(candidates)
    alignment["candidates"] = {}
    for candidate in candidates:
        candidate_alignment = dict(candidate["alignment"])
        candidate_alignment["score"] = alignment_confidence_score(candidate["alignment"])
        alignment["candidates"][candidate["name"]] = candidate_alignment
    if meta_issue:
        alignment["metaIssue"] = meta_issue
    if contain_issue:
        alignment["containCandidateIssue"] = contain_issue

    cv2.imwrite(os.path.join(args.outdir, "mask_in_return_space.png"), mask_return)
    with open(os.path.join(args.outdir, "alignment_report.json"), "w", encoding="utf-8") as handle:
        json.dump(alignment, handle, ensure_ascii=False, indent=2)

    # 5) unmask only region (paste only within mask)
    unmasked = unmask_only_region(edited, original_return, mask_return)
    out_path = os.path.join(args.outdir, args.outname)
    cv2.imwrite(out_path, unmasked)

    print("Saved:")
    print(f" - {os.path.abspath(out_path)}")
    print(f" - {os.path.abspath(os.path.join(args.outdir, 'mask_in_return_space.png'))}")
    print(f" - {os.path.abspath(os.path.join(args.outdir, 'alignment_report.json'))}")

    if args.dataveil_json_file:
        payload = {
            "artifacts": {
                "unmasked": os.path.abspath(out_path).replace("\\", "/"),
                "maskInReturnSpace": os.path.abspath(os.path.join(args.outdir, "mask_in_return_space.png")).replace("\\", "/"),
                "combinedMaskOriginal": os.path.abspath(os.path.join(args.outdir, "combined_mask_original.png")).replace("\\", "/")
            },
            "alignment": alignment
        }
        with open(args.dataveil_json_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)


if __name__ == "__main__":
    main()


# python unmasker.py --edited ai_output.png --original TestImage.png --mask out\chosen_mask.png --meta out\transform.json --outdir out_unmask --outname final_unmasked.png
