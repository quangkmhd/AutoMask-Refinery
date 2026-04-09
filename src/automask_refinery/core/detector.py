import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Tuple

from automask_refinery.utils.geometry import (
    compute_mask_area,
    compute_bbox_area,
    compute_compactness,
    compute_solidity,
    extract_shape_signature
)
from automask_refinery.utils.logger import log

class MaskDetector:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_dataset(self) -> List[Dict]:
        dataset = []
        log.info(f"Scanning directory: {self.data_dir}")
        
        if not os.path.exists(self.data_dir):
            log.error(f"Directory not found: {self.data_dir}")
            return []

        subdirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for subdir in tqdm(subdirs, desc="Loading classes"):
            class_path = os.path.join(self.data_dir, subdir)
            json_files = [f for f in os.listdir(class_path) if f.endswith('.json') and not f.startswith('.')]
            
            for json_file in json_files:
                name = os.path.splitext(json_file)[0]
                xml_path = os.path.join(class_path, name + '.xml')
                json_path = os.path.join(class_path, json_file)
                
                if not os.path.exists(xml_path):
                    continue
                    
                try:
                    with open(json_path, 'r') as f:
                        js_data = json.load(f)
                    
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    obj = root.find('object')
                    if obj is None: continue
                    
                    class_id = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    bbox = [
                        int(bndbox.find('xmin').text),
                        int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text),
                        int(bndbox.find('ymax').text)
                    ]
                    
                    points = None
                    for shape in js_data.get('shapes', []):
                        if shape['label'] == class_id:
                            points = shape['points']
                            break
                    
                    if points:
                        dataset.append({
                            "image_id": f"{subdir}/{name}",
                            "class_id": class_id,
                            "bbox": bbox,
                            "points": points
                        })
                except Exception as e:
                    log.warning(f"Error processing {json_path}: {e}")
                    continue
                    
        return dataset

    def detect_ratio_outliers(self, samples: List[Dict]) -> List[Dict]:
        df = pd.DataFrame(samples)
        results = []
        
        for class_id, group in df.groupby('class_id'):
            ratios = group['ratio'].values
            if len(ratios) < 3:
                group['failed_ratio'] = False
                results.append(group)
                continue
                
            median_ratio = np.median(ratios)
            mad = np.median(np.abs(ratios - median_ratio))
            
            if mad == 0:
                group['failed_ratio'] = (group['ratio'] != median_ratio)
            else:
                group['failed_ratio'] = np.abs(group['ratio'] - median_ratio) > 3 * mad
                
            results.append(group)
            
        return pd.concat(results).to_dict('records')

    def detect_shape_outliers(self, samples: List[Dict]) -> List[Dict]:
        df = pd.DataFrame(samples)
        results = []
        
        for class_id, group in df.groupby('class_id'):
            signatures = np.array(group['signature'].tolist())
            
            if len(signatures) < 10:
                group['shape_anomaly_score'] = 0.0
                group['failed_shape'] = False
                results.append(group)
                continue
                
            clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
            preds = clf.fit_predict(signatures)
            scores = clf.decision_function(signatures)
            
            group['shape_anomaly_score'] = scores
            group['failed_shape'] = (preds == -1)
            results.append(group)
            
        return pd.concat(results).to_dict('records')

    def run_pipeline(self, dataset: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        log.info("Extracting features...")
        for sample in tqdm(dataset, desc="Features"):
            sample['ratio'] = compute_mask_area(sample['points']) / compute_bbox_area(sample['bbox']) if compute_bbox_area(sample['bbox']) > 0 else 0
            sample['compactness'] = compute_compactness(sample['points'])
            sample['solidity'] = compute_solidity(sample['points'])
            sample['signature'] = extract_shape_signature(sample['points'], sample['bbox'])
        
        log.info("Running Method 1 (Ratio Outliers)...")
        dataset = self.detect_ratio_outliers(dataset)
        
        log.info("Running Method 2 (Shape Anomaly)...")
        dataset = self.detect_shape_outliers(dataset)
        
        log.info("Running Method 3 (Absolute Heuristics)...")
        for sample in dataset:
            reasons = []
            if sample['solidity'] < 0.65: reasons.append("low_solidity")
            if sample['compactness'] > 150: reasons.append("high_compactness")
            if sample['ratio'] < 0.1: reasons.append("low_ratio")
            
            sample['failed_heuristic'] = len(reasons) > 0
            sample['heuristic_reason'] = ",".join(reasons)
        
        final_failed = []
        for sample in dataset:
            sample['final_failed'] = sample['failed_ratio'] or sample['failed_shape'] or sample['failed_heuristic']
            if sample['final_failed']:
                final_failed.append(sample)
                
        log.info(f"Pipeline finished. Total: {len(dataset)}, Failures: {len(final_failed)}")
        return dataset, final_failed

    def visualize_results(self, samples: List[Dict], output_dir: str, limit: int = 200):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        log.info(f"Exporting {min(len(samples), limit)} samples for review to: {output_dir}")
        
        for sample in tqdm(samples[:limit], desc="Visualizing"):
            class_id = sample['class_id']
            img_id = sample['image_id']
            
            img_path = os.path.join(self.data_dir, img_id + ".jpg")
            if not os.path.exists(img_path): continue
                
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Draw BBox
            x1, y1, x2, y2 = sample['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw Mask
            points = np.array(sample['points'], dtype=np.int32)
            cv2.polylines(img, [points], True, (0, 0, 255), 2)
            
            label = f"R:{sample['failed_ratio']} S:{sample['failed_shape']} H:{sample['failed_heuristic']}"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            class_out_dir = os.path.join(output_dir, class_id)
            os.makedirs(class_out_dir, exist_ok=True)
                
            out_name = os.path.basename(img_id) + ".jpg"
            cv2.imwrite(os.path.join(class_out_dir, out_name), img)
