import os
import json
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    from ultralytics import SAM
except ImportError:
    SAM = None

from automask_refinery.utils.logger import log

class MaskGenerator:
    def __init__(self, sam_model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        if SAM is None:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        log.info(f"Loading SAM3 model from {sam_model_path} on {device}")
        self.sam_model = SAM(sam_model_path)
        self.device = device

    def parse_xml(self, xml_path: str) -> Tuple[List[Dict], int, int]:
        try:
            tree = ET.parse(xml_path)
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
        except Exception as e:
            log.warning(f"Failed to parse XML {xml_path}: {e}")
            return [], 0, 0

    def get_mask_polygon(self, mask_bool: np.ndarray) -> List[List[float]]:
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            return []
        return largest.reshape(-1, 2).tolist()

    def generate_for_directory(self, input_dir: str, force: bool = False, imgsz: int = 1036):
        root_path = Path(input_dir)
        jpg_files = sorted(list(root_path.rglob("*.jpg")))
        log.info(f"Found {len(jpg_files)} images in {input_dir}")

        stats = defaultdict(int)

        for jpg_path in tqdm(jpg_files, desc="SAM3 Inference"):
            xml_path = jpg_path.with_suffix(".xml")
            json_path = jpg_path.with_suffix(".json")

            if not xml_path.exists():
                stats["skip_no_xml"] += 1
                continue
            if json_path.exists() and not force:
                stats["skip_existing"] += 1
                continue

            objects, w, h = self.parse_xml(str(xml_path))
            if not objects or w == 0 or h == 0:
                stats["skip_invalid"] += 1
                continue

            bboxes_list = [obj["bbox"] for obj in objects]
            try:
                results = self.sam_model(source=str(jpg_path), bboxes=bboxes_list,
                                    device=self.device, verbose=False, imgsz=imgsz)
            except Exception as e:
                log.error(f"SAM Error on {jpg_path.name}: {e}")
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
                if idx >= len(mask_data): break
                
                lbl = obj["label"]
                xmin, ymin, xmax, ymax = obj["bbox"]
                obj_mask = mask_data[idx] > 0.5
                
                polygon_pts = self.get_mask_polygon(obj_mask)
                if not polygon_pts: continue

                mask_area = int(obj_mask.sum())
                box_area = (xmax - xmin) * (ymax - ymin)

                shapes.append({
                    "label": lbl,
                    "points": polygon_pts,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                    "metrics": {
                        "confidence": round(float(confidences[idx]), 4),
                        "coverage": round(mask_area / box_area, 4) if box_area > 0 else 0.0,
                        "mask_area": mask_area
                    },
                })

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
            stats["total_masks"] += len(shapes)

        log.info(f"Generation complete: {stats['processed']} images, {stats['total_masks']} masks.")
        return stats
