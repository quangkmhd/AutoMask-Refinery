import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pandas as pd
from sklearn.ensemble import IsolationForest
from flask import Flask, render_template, jsonify, request, send_from_directory

import argparse

# Get absolute path of the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'))

# These will be set in main()
ROOT_DIR = ""
SUMMARY_CSV = ""

# --- Re-use Logic from failed_mask_detector.py ---

def compute_mask_area(points):
    return cv2.contourArea(np.array(points, dtype=np.float32))

def compute_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1)) * max(0, (y2 - y1))

def compute_compactness(points):
    area = compute_mask_area(points)
    if area == 0: return 0
    perimeter = cv2.arcLength(np.array(points, dtype=np.float32), True)
    return (perimeter ** 2) / area

def compute_solidity(points):
    area = compute_mask_area(points)
    if area == 0: return 0
    points_arr = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points_arr)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return 0
    return area / hull_area

def is_failed(sample):
    """Simple heuristic logic for quick individual check."""
    # This matches Method 3 in failed_mask_detector.py
    if sample['solidity'] < 0.65: return True
    if sample['compactness'] > 150: return True
    if sample['ratio'] < 0.1: return True
    return False

def get_folder_review_data(folder_name):
    folder_path = os.path.join(ROOT_DIR, folder_name)
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    
    # Load review results if exists
    review_file = os.path.join(folder_path, '.review_results.json')
    overrides = {}
    folder_status = "none" # none, pass_all, not_pass_all
    if os.path.exists(review_file):
        with open(review_file, 'r') as f:
            data = json.load(f)
            overrides = data.get('overrides', {})
            folder_status = data.get('status', "none")

    samples = []
    for jf in json_files:
        name = os.path.splitext(jf)[0]
        xml_path = os.path.join(folder_path, name + '.xml')
        if not os.path.exists(xml_path): continue
        
        try:
            with open(os.path.join(folder_path, jf), 'r') as f:
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
                ratio = compute_mask_area(points) / compute_bbox_area(bbox) if compute_bbox_area(bbox) > 0 else 0
                compactness = compute_compactness(points)
                solidity = compute_solidity(points)
                
                # Default failure by algorithm
                algo_failed = is_failed({
                    'ratio': ratio, 
                    'compactness': compactness, 
                    'solidity': solidity
                })
                
                # Final status based on manual overrides
                final_status = overrides.get(name, "fail" if algo_failed else "pass")
                
                samples.append({
                    "id": name,
                    "img_url": f"/api/render_image/{folder_name}/{name}",
                    "status": final_status, # "pass" or "fail"
                    "algo_failed": algo_failed
                })
        except:
            continue
            
    return samples, folder_status

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/folders')
def get_folders():
    folder_list = []
    for root, dirs, files in os.walk(ROOT_DIR):
        # Check if current directory has at least one .json file (excluding review results)
        if any(f.endswith('.json') and f != '.review_results.json' for f in files):
            # Use relative path from ROOT_DIR
            rel_path = os.path.relpath(root, ROOT_DIR)
            if rel_path == ".":
                continue # Skip root itself if it has jsons directly (unlikely)
            folder_list.append(rel_path)
    
    return jsonify(sorted(folder_list))

@app.route('/api/images/<path:folder_name>')
def get_images(folder_name):
    samples, folder_status = get_folder_review_data(folder_name)
    return jsonify({"images": samples, "status": folder_status})

@app.route('/api/render_image/<path:folder_name>/<image_name>')
def render_image(folder_name, image_name):
    folder_path = os.path.join(ROOT_DIR, folder_name)
    img_path = os.path.join(folder_path, image_name + ".jpg")
    json_path = os.path.join(folder_path, image_name + ".json")
    xml_path = os.path.join(folder_path, image_name + ".xml")

    if not os.path.exists(img_path):
        return "Image not found", 404

    img = cv2.imread(img_path)
    if img is None:
        return "Failed to load image", 500

    # Bounding box coordinates
    bbox_coords = None
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find('object')
            if obj is not None:
                bndbox = obj.find('bndbox')
                bbox_coords = (
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text)
                )
        except: pass

    # Draw Mask (Solid Blue)
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                js_data = json.load(f)

            overlay = img.copy()
            for shape in js_data.get('shapes', []):
                points = np.array(shape['points'], dtype=np.int32)
                # Fill the mask with Bright Red for high visibility
                cv2.fillPoly(overlay, [points], (0, 0, 255))
                # Draw thick border
                cv2.polylines(img, [points], True, (0, 0, 255), 2)

            # Blend with very high opacity (0.9) to make it nearly solid
            cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
        except: pass

    # Draw Bbox (Green) - Draw last to be on top
    if bbox_coords:
        x1, y1, x2, y2 = bbox_coords
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

@app.route('/api/save_review', methods=['POST'])
def save_review():
    data = request.json
    # folder_name can now be a path like "A/B/C"
    folder_name = data.get('folder')
    overrides = data.get('overrides') # dict: { "id": "pass" | "fail" }
    
    folder_path = os.path.join(ROOT_DIR, folder_name)
    review_file = os.path.join(folder_path, '.review_results.json')
    
    # 1. Save Detailed JSON (for app persistence)
    save_data = {
        "status": "reviewed",
        "overrides": overrides
    }
    with open(review_file, 'w') as f:
        json.dump(save_data, f, indent=4)

    # 2. Get all images and their final status
    samples, _ = get_folder_review_data(folder_name)
    
    # 3. Update Global Detailed CSV
    summary_path = SUMMARY_CSV
    
    # Prepare data for this folder
    rows = []
    for s in samples:
        rows.append({
            "Folder": folder_name,
            "File_ID": s['id'],
            "Status": s['status'],
            "Image_Path": os.path.join(ROOT_DIR, folder_name, s['id'] + ".jpg")
        })
    
    new_df = pd.DataFrame(rows)
    
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        # Remove old records for THIS folder to update with new ones
        existing_df = existing_df[existing_df['Folder'] != folder_name]
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df
    
    final_df.to_csv(summary_path, index=False)
        
    return jsonify({"success": True})

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(ROOT_DIR, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoMask-Refinery Review App")
    
    DEFAULT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'demo'))
    
    parser.add_argument('--data_dir', type=str, default=DEFAULT_ROOT, help='Path to dataset directory')
    parser.add_argument('--csv_out', type=str, default=os.path.join(DEFAULT_ROOT, "review_details.csv"), help='Path to output CSV')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the Flask app')
    
    args = parser.parse_args()
    
    ROOT_DIR = args.data_dir
    SUMMARY_CSV = args.csv_out
    
    print(f" * Data directory: {ROOT_DIR}")
    print(f" * Summary CSV: {SUMMARY_CSV}")
    
    app.run(host=args.host, port=args.port, debug=True)
