"""
=================================================================
FILE 1/2: GENERATE SAM3 MASKS + COMPUTE ALL QC METRICS
=================================================================
Chạy SAM3 inference, tính tất cả metrics (confidence, coverage,
complexity, jittering, CLIP), lưu kết quả vào LabelMe JSON.

Usage:
    # Mặc định: CHỈ SAM3 + cheap metrics (~1s/ảnh)
    python generate_sam3_masks.py --input_dir ./data/products

    # Bật thêm CLIP verification (chậm hơn)
    python generate_sam3_masks.py --input_dir ./data/products --enable_clip

    # Bật đầy đủ (CLIP + Jittering + Oracle — chậm nhất)
    python generate_sam3_masks.py --input_dir ./data/products --enable_clip --enable_jitter --enable_oracle

    # Chạy lại ảnh đã có JSON
    python generate_sam3_masks.py --input_dir ./data/products --force
=================================================================
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

try:
    from ultralytics import SAM
except ImportError:
    print("pip install ultralytics")
    exit(1)

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPModel, CLIPProcessor = None, None
    print("⚠️ transformers not installed. CLIP verification disabled.")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_xml(xml_path):
    """Parse Pascal VOC XML."""
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return [], 0, 0
    root = tree.getroot()
    size_elem = root.find("size")
    width = int(size_elem.find("width").text) if size_elem is not None else 0
    height = int(size_elem.find("height").text) if size_elem is not None else 0
    objects = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_elem is None or bndbox is None:
            continue
        label = name_elem.text.strip()
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
    return objects, width, height


def calc_iou(mask1, mask2):
    """IoU between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


def get_mask_polygon(mask_bool):
    """Largest contour polygon from boolean mask."""
    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return []
    return largest.reshape(-1, 2).tolist()


def get_connected_components(mask_bool):
    """Number of connected components (excluding background)."""
    num_labels, _ = cv2.connectedComponents(mask_bool.astype(np.uint8))
    return num_labels - 1


def compute_boundary_complexity(mask_bool):
    """P²/A — low for smooth boundaries, high for jagged."""
    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    area = cv2.contourArea(largest)
    if area < 1.0:
        return 0.0
    
    # Standard P²/A complexity
    complexity = (perimeter ** 2) / area
    return round(float(complexity), 2)


# ============================================================
# CLIP SEMANTIC VERIFICATION
# ============================================================

def run_clip_verification(image_path, mask_bool, label, clip_model, clip_processor, device):
    """Crop masked region → CLIP cosine similarity vs label text."""
    image = cv2.imread(image_path)
    if image is None:
        return 0.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masked_img = np.zeros_like(image)
    masked_img[mask_bool] = image[mask_bool]

    y_coords, x_coords = np.where(mask_bool)
    if len(y_coords) == 0:
        return 0.0
    cropped = masked_img[y_coords.min():y_coords.max() + 1, x_coords.min():x_coords.max() + 1]

    try:
        inputs = clip_processor(
            text=[label, "background", "noise"],
            images=[cropped], return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        return round(torch.matmul(img_emb, txt_emb.T)[0, 0].item(), 4)
    except Exception as e:
        print(f"  [CLIP Error] {e}")
        return 0.0


# ============================================================
# BOX JITTERING — Multi-directional Consensus
# ============================================================

def run_jittering(sam_model, image_path, original_mask, bbox, device, jitter_ratio=0.03):
    """3 jitter variations → mean IoU consensus."""
    xmin, ymin, xmax, ymax = bbox
    bw, bh = xmax - xmin, ymax - ymin
    dx, dy = bw * jitter_ratio, bh * jitter_ratio

    jitter_offsets = [
        (-dx, -dy, dx, dy),     # Expand
        (dx, dy, -dx, -dy),     # Shrink
        (-dx, dy, dx, -dy),     # Diagonal skew
    ]

    ious = []
    for d1, d2, d3, d4 in jitter_offsets:
        jbox = [max(0, xmin + d1), max(0, ymin + d2), xmax + d3, ymax + d4]
        try:
            j_res = sam_model(source=image_path, bboxes=[jbox], device=device,
                              verbose=False, imgsz=1036)
            if j_res and j_res[0].masks:
                j_mask = j_res[0].masks.data.cpu().numpy()[0] > 0.5
                ious.append(calc_iou(original_mask, j_mask))
        except Exception:
            continue
    return round(float(np.mean(ious)), 4) if ious else 0.0


# ============================================================
# MULTI-MASK ORACLE
# ============================================================

def run_oracle(sam_model, image_path, bbox, label, clip_model, clip_processor, device, max_candidates=4):
    """Try alternative SAM3 mask candidates, return best CLIP match."""
    if clip_model is None:
        return None, 0.0

    try:
        results = sam_model(source=image_path, bboxes=[bbox], device=device,
                            verbose=False, retina_masks=True, imgsz=1036)
    except Exception:
        return None, 0.0

    if not results or not results[0].masks:
        return None, 0.0

    masks = results[0].masks.data.cpu().numpy()
    best_mask, best_sim = None, -1.0

    for i in range(min(len(masks), max_candidates)):
        candidate = masks[i] > 0.5
        if candidate.sum() < 50:
            continue
        sim = run_clip_verification(image_path, candidate, label, clip_model, clip_processor, device)
        if sim > best_sim:
            best_sim = sim
            best_mask = candidate

    return best_mask, round(best_sim, 4)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="[Step 1/2] Generate SAM3 Masks + Compute ALL QC Metrics")

    BASE_DIR = Path(__file__).parent
    DEFAULT_INPUT = os.path.abspath(BASE_DIR / ".." / "data" / "demo")
    
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--sam_model", type=str, default=os.path.abspath(BASE_DIR / ".." / "sam3.pt"))
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Re-process images with existing JSON")
    parser.add_argument("--imgsz", type=int, default=1036,
                        help="SAM3 input image size (must be multiple of 14)")

    # Toggle expensive operations (OPT-IN, default OFF for speed)
    parser.add_argument("--enable_clip", action="store_true",
                        help="Enable CLIP semantic verification (slower)")
    parser.add_argument("--enable_jitter", action="store_true",
                        help="Enable Box Jittering robustness check (slower)")
    parser.add_argument("--enable_oracle", action="store_true",
                        help="Enable Multi-mask Oracle recovery (requires --enable_clip)")

    # Waterfall triggers (when to run expensive checks if enabled)
    parser.add_argument("--waterfall_conf", type=float, default=0.90,
                        help="Run expensive checks if confidence < this value")
    parser.add_argument("--waterfall_coverage_low", type=float, default=0.15)
    parser.add_argument("--waterfall_coverage_high", type=float, default=0.95)
    parser.add_argument("--waterfall_area", type=int, default=200)

    # Oracle trigger
    parser.add_argument("--oracle_clip_trigger", type=float, default=0.75,
                        help="Run Oracle if CLIP similarity < this value")

    args = parser.parse_args()

    # --- Load SAM3 ---
    print(f"[1/3] Loading SAM3: {args.sam_model}")
    sam_model = SAM(args.sam_model)

    # --- Load CLIP (only if enabled) ---
    clip_model, clip_processor = None, None
    if args.enable_clip and CLIPModel is not None:
        print(f"[2/3] Loading CLIP: {args.clip_model_name}")
        try:
            clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(args.device)
            clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
        except Exception as e:
            print(f"  ⚠️ CLIP load failed: {e}")
    else:
        print("[2/3] CLIP: skipped (use --enable_clip to enable)")

    # Print mode info
    mode_parts = ["SAM3 + cheap metrics"]
    if args.enable_jitter:
        mode_parts.append("Jittering")
    if args.enable_clip:
        mode_parts.append("CLIP")
    if args.enable_oracle and args.enable_clip:
        mode_parts.append("Oracle")
    print(f"     Mode: {' + '.join(mode_parts)}")

    # --- Discover images ---
    root_path = Path(args.input_dir)
    jpg_files = sorted(root_path.rglob("*.jpg"))
    print(f"[3/3] Found {len(jpg_files)} images\n")

    stats = defaultdict(int)

    for jpg_path in tqdm(jpg_files, desc="Generating masks"):
        xml_path = jpg_path.with_suffix(".xml")
        json_path = jpg_path.with_suffix(".json")

        if not xml_path.exists():
            stats["skip_no_xml"] += 1
            continue
        if json_path.exists() and not args.force:
            stats["skip_existing"] += 1
            continue

        objects, w, h = parse_xml(str(xml_path))
        if not objects or w == 0 or h == 0:
            stats["skip_invalid"] += 1
            continue

        # --- SAM3 inference ---
        bboxes_list = [obj["bbox"] for obj in objects]
        try:
            results = sam_model(source=str(jpg_path), bboxes=bboxes_list,
                                device=args.device, verbose=False, imgsz=args.imgsz)
        except Exception as e:
            tqdm.write(f"  [SAM Error] {jpg_path.name}: {e}")
            stats["sam_error"] += 1
            continue

        if not results or not results[0].masks:
            stats["no_masks"] += 1
            continue

        result = results[0]
        mask_data = result.masks.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy() if (result.boxes is not None and result.boxes.conf is not None) else np.ones(len(mask_data))

        shapes = []

        for idx, obj in enumerate(objects):
            if idx >= len(mask_data):
                break

            lbl = obj["label"]
            xmin, ymin, xmax, ymax = obj["bbox"]
            obj_mask = mask_data[idx] > 0.5

            stats["total_masks"] += 1

            # === COMPUTE ALL METRICS (stored in JSON, filtered later in FiftyOne) ===
            conf = round(float(confidences[idx]) if idx < len(confidences) else 0.0, 4)
            mask_area = int(obj_mask.sum())
            box_area = (xmax - xmin) * (ymax - ymin)
            coverage = round(mask_area / box_area, 4) if box_area > 0 else 0.0
            n_components = get_connected_components(obj_mask)

            try:
                complexity = compute_boundary_complexity(obj_mask)
            except Exception:
                complexity = 0.0

            # --- Waterfall: decide whether to run expensive metrics ---
            # Only evaluate if expensive checks are enabled
            needs_expensive = (
                (args.enable_clip or args.enable_jitter)
                and (
                    conf < args.waterfall_conf
                    or coverage < args.waterfall_coverage_low
                    or coverage > args.waterfall_coverage_high
                    or mask_area < args.waterfall_area
                    or n_components > 1
                )
            )

            # Jittering (only if enabled AND waterfall triggered)
            jitter_miou = -1.0  # -1 = not computed
            if needs_expensive and args.enable_jitter:
                jitter_miou = run_jittering(
                    sam_model, str(jpg_path), obj_mask, obj["bbox"], args.device
                )
                stats["jitter_runs"] += 1

            # CLIP (only if enabled AND waterfall triggered)
            clip_sim = -1.0  # -1 = not computed
            if needs_expensive and args.enable_clip and clip_model is not None:
                clip_sim = run_clip_verification(
                    str(jpg_path), obj_mask, lbl,
                    clip_model, clip_processor, args.device,
                )
                stats["clip_runs"] += 1

                # Oracle: if CLIP is low and oracle enabled, try alternative masks
                if clip_sim < args.oracle_clip_trigger and args.enable_oracle:
                    oracle_mask, oracle_sim = run_oracle(
                        sam_model, str(jpg_path), obj["bbox"], lbl,
                        clip_model, clip_processor, args.device,
                    )
                    stats["oracle_runs"] += 1
                    if oracle_mask is not None and oracle_sim > clip_sim:
                        obj_mask = oracle_mask  # Replace with better mask
                        clip_sim = oracle_sim
                        stats["oracle_recovered"] += 1

            # --- Extract polygon from (possibly updated) mask ---
            polygon_pts = get_mask_polygon(obj_mask)
            if not polygon_pts:
                stats["no_polygon"] += 1
                continue

            # --- Save shape with ALL metrics ---
            shapes.append({
                "label": lbl,
                "points": polygon_pts,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
                "metrics": {
                    "confidence": conf,
                    "coverage": coverage,
                    "mask_area": mask_area,
                    "n_components": n_components,
                    "complexity": complexity,
                    "jitter_miou": jitter_miou,
                    "clip_similarity": clip_sim,
                },
            })

        # --- Write LabelMe JSON ---
        json_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": shapes,
            "imagePath": jpg_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        stats["processed"] += 1

    # --- Summary ---
    print("\n" + "=" * 55)
    print("📊 GENERATION COMPLETE")
    print("=" * 55)
    print(f"  Images processed:    {stats['processed']}")
    print(f"  Masks generated:     {stats['total_masks']}")
    print(f"  Skip (no XML):       {stats['skip_no_xml']}")
    print(f"  Skip (existing):     {stats['skip_existing']}")
    print(f"  SAM errors:          {stats['sam_error']}")
    print(f"  Jittering runs:      {stats['jitter_runs']}")
    print(f"  CLIP runs:           {stats['clip_runs']}")
    print(f"  Oracle runs:         {stats['oracle_runs']}")
    print(f"  Oracle recovered:    {stats['oracle_recovered']}")
    print("=" * 55)
    print("\n✅ Done! Now run: python review_fiftyone.py --input_dir " + args.input_dir)


if __name__ == "__main__":
    main()
