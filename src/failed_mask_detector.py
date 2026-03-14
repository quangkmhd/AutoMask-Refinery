import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import pandas as pd
from collections import defaultdict
import argparse

def compute_mask_area(points):
    """Compute area of a polygon defined by points."""
    return cv2.contourArea(np.array(points, dtype=np.float32))

def compute_bbox_area(bbox):
    """Compute area of [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1)) * max(0, (y2 - y1))

def compute_compactness(points):
    """compactness = perimeter^2 / area"""
    area = compute_mask_area(points)
    if area == 0:
        return 0
    perimeter = cv2.arcLength(np.array(points, dtype=np.float32), True)
    return (perimeter ** 2) / area

def compute_aspect_ratio(points):
    """aspect_ratio = mask_width / mask_height"""
    if not points:
        return 0
    points = np.array(points, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(points)
    if h == 0:
        return 0
    return w / h

def compute_hu_moments(points):
    """7 Hu moments from points."""
    moments = cv2.moments(np.array(points, dtype=np.float32))
    hu = cv2.HuMoments(moments).flatten()
    # Log transform to make them more manageable
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log

def compute_solidity(points):
    """solidity = area / convex_hull_area"""
    area = compute_mask_area(points)
    if area == 0:
        return 0
    points_arr = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points_arr)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area

def extract_shape_signature(points, bbox):
    """Extract 11-feature signature vector."""
    mask_area = compute_mask_area(points)
    bbox_area = compute_bbox_area(bbox)
    
    ratio = mask_area / bbox_area if bbox_area > 0 else 0
    compactness = compute_compactness(points)
    aspect_ratio = compute_aspect_ratio(points)
    solidity = compute_solidity(points)
    hu = compute_hu_moments(points)
    
    signature = [ratio, compactness, aspect_ratio, solidity] + hu.tolist()
    return signature

def detect_ratio_outliers(samples):
    """
    Method 1: Median Absolute Deviation (MAD) on mask/box ratio.
    Expects list of dicts with 'ratio' and 'class_id'.
    """
    df = pd.DataFrame(samples)
    results = []
    
    for class_id, group in df.groupby('class_id'):
        ratios = group['ratio'].values
        if len(ratios) < 3:
            # Not enough samples to detect outliers reliably
            group['failed_ratio'] = False
            results.append(group)
            continue
            
        median_ratio = np.median(ratios)
        mad = np.median(np.abs(ratios - median_ratio))
        
        # Avoid division by zero
        if mad == 0:
            # If all ratios are same, no outliers unless they differ from median
            group['failed_ratio'] = (group['ratio'] != median_ratio)
        else:
            group['failed_ratio'] = np.abs(group['ratio'] - median_ratio) > 3 * mad
            
        results.append(group)
        
    return pd.concat(results).to_dict('records')

def detect_shape_outliers(samples):
    """
    Method 2: Isolation Forest on shape signatures.
    Expects list of dicts with 'signature' and 'class_id'.
    """
    df = pd.DataFrame(samples)
    results = []
    
    for class_id, group in df.groupby('class_id'):
        signatures = np.array(group['signature'].tolist())
        
        if len(signatures) < 10:
            # IsolationForest might not be stable with very few samples
            group['shape_anomaly_score'] = 0.0
            group['failed_shape'] = False
            results.append(group)
            continue
            
        clf = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        # Fix: handle potential all-identical signatures
        preds = clf.fit_predict(signatures)
        scores = clf.decision_function(signatures)
        
        group['shape_anomaly_score'] = scores
        group['failed_shape'] = (preds == -1)
        results.append(group)
        
    return pd.concat(results).to_dict('records')

def load_dataset(root_dir):
    dataset = []
    print(f"Scanning directory: {root_dir}")
    
    # Get subdirectories (classes)
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for subdir in tqdm(subdirs, desc="Loading classes"):
        class_path = os.path.join(root_dir, subdir)
        json_files = [f for f in os.listdir(class_path) if f.endswith('.json')]
        
        for json_file in json_files:
            name = os.path.splitext(json_file)[0]
            xml_file = name + '.xml'
            xml_path = os.path.join(class_path, xml_file)
            json_path = os.path.join(class_path, json_file)
            
            if not os.path.exists(xml_path):
                continue
                
            try:
                # Load JSON for mask (polygon)
                with open(json_path, 'r') as f:
                    js_data = json.load(f)
                
                # Load XML for BBox and Class
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Assume one object per file as per example
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
                
                # Find matching shape in JSON
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
                # print(f"Error processing {json_path}: {e}")
                continue
                
    return dataset

def run_filter_pipeline(dataset):
    # Step 1: Feature Extraction
    print("Extracting features...")
    for sample in tqdm(dataset, desc="Features"):
        sample['ratio'] = compute_mask_area(sample['points']) / compute_bbox_area(sample['bbox']) if compute_bbox_area(sample['bbox']) > 0 else 0
        sample['compactness'] = compute_compactness(sample['points'])
        sample['solidity'] = compute_solidity(sample['points'])
        sample['signature'] = extract_shape_signature(sample['points'], sample['bbox'])
    
    # Step 2: Method 1 - Ratio Outliers
    print("Running Method 1 (Ratio Outliers)...")
    dataset = detect_ratio_outliers(dataset)
    
    # Step 3: Method 2 - Shape Anomaly
    print("Running Method 2 (Shape Anomaly)...")
    dataset = detect_shape_outliers(dataset)
    
    # Step 4: Method 3 - Absolute Heuristics (spidery/noise masks)
    print("Running Method 3 (Absolute Heuristics for Noise/Text)...")
    for sample in dataset:
        failed_heuristic = False
        reasons = []
        if sample['solidity'] < 0.65:
            failed_heuristic = True
            reasons.append("low_solidity")
        if sample['compactness'] > 150:
            failed_heuristic = True
            reasons.append("high_compactness")
        if sample['ratio'] < 0.1:
            failed_heuristic = True
            reasons.append("low_ratio")
        
        sample['failed_heuristic'] = failed_heuristic
        sample['heuristic_reason'] = ",".join(reasons)
    
    # Step 5: Final Flag
    final_failed = []
    for sample in dataset:
        sample['final_failed'] = sample['failed_ratio'] or sample['failed_shape'] or sample['failed_heuristic']
        if sample['final_failed']:
            final_failed.append(sample)
            
    # Summary
    total = len(dataset)
    ratio_failures = sum(1 for s in dataset if s['failed_ratio'])
    shape_failures = sum(1 for s in dataset if s['failed_shape'])
    heuristic_failures = sum(1 for s in dataset if s['failed_heuristic'])
    final_failures_count = len(final_failed)
    
    print("\n--- PIPELINE SUMMARY ---")
    print(f"Total samples:  {total}")
    print(f"Ratio failures (MAD): {ratio_failures}")
    print(f"Shape failures (IsoForest): {shape_failures}")
    print(f"Heuristic failures (Spidery/Noise): {heuristic_failures}")
    print(f"Final failures (Combined): {final_failures_count}")
    
    return dataset, final_failed

def visualize_failures(failed_samples, root_data_dir, output_dir):
    """
    Vẽ mask và bbox của các mẫu failed ra thư mục để review.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nExporting {len(failed_samples)} samples for review to: {output_dir}")
    
    for sample in tqdm(failed_samples, desc="Visualizing"):
        class_id = sample['class_id']
        img_id = sample['image_id'] # e.g. "Necessities_ActiPlus/2"
        
        # Đường dẫn ảnh gốc
        img_path = os.path.join(root_data_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Vẽ BBox (Màu xanh dương)
        x1, y1, x2, y2 = sample['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Vẽ Mask Polygon (Màu đỏ)
        points = np.array(sample['points'], dtype=np.int32)
        cv2.polylines(img, [points], True, (0, 0, 255), 2)
        
        # Thêm text chú thích loại lỗi
        label = f"Ratio: {sample['failed_ratio']} | Shape: {sample['failed_shape']}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Lưu vào thư mục class tương ứng
        class_out_dir = os.path.join(output_dir, class_id)
        if not os.path.exists(class_out_dir):
            os.makedirs(class_out_dir)
            
        out_name = os.path.basename(img_id) + "_fail.jpg"
        cv2.imwrite(os.path.join(class_out_dir, out_name), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoMask-Refinery Failed Mask Detector")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'demo'))
    
    parser.add_argument('--data_dir', type=str, default=DEFAULT_ROOT, help='Path to dataset directory')
    parser.add_argument('--review_out', type=str, default=os.path.join(DEFAULT_ROOT, "review_failures"), help='Path to output review failures')
    parser.add_argument('--passed_out', type=str, default=os.path.join(DEFAULT_ROOT, "review_passed"), help='Path to output review passed')
    
    args = parser.parse_args()
    
    root_data_dir = args.data_dir
    output_review_dir = args.review_out
    output_passed_dir = args.passed_out
    
    dataset = load_dataset(root_data_dir)
    if not dataset:
        print("No dataset loaded. Check paths.")
    else:
        results, final_failed = run_filter_pipeline(dataset)
        
        # Lọc ra danh sách các mẫu PASSED (không bị đánh dấu lỗi)
        passed_samples = [s for s in results if not s['final_failed']]
        
        # Xuất 200 mẫu FAILED để kiểm tra lỗi
        visualize_failures(final_failed[:200], root_data_dir, output_review_dir)
        
        # Xuất 200 mẫu PASSED để đối chiếu (xem cái 'ngon' trông thế nào)
        visualize_failures(passed_samples[:200], root_data_dir, output_passed_dir)
        
        print(f"\n--- REVIEW CHANNELS READY ---")
        print(f"1. FAILED samples: {output_review_dir}")
        print(f"2. PASSED samples: {output_passed_dir}")
        print(f"\nHãy so sánh ảnh ở 2 thư mục này để đánh giá độ chính xác của filter.")
