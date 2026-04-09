import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
from typing import Dict, List, Tuple

from automask_refinery.config.settings import settings
from automask_refinery.utils.geometry import (
    compute_mask_area,
    compute_bbox_area,
    compute_compactness,
    compute_solidity
)
from automask_refinery.utils.logger import log

def create_app():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    app = Flask(__name__, 
                template_folder=os.path.join(BASE_DIR, 'templates'))

    def is_failed(sample: Dict) -> bool:
        """Heuristic logic for quick individual check."""
        if sample['solidity'] < 0.65: return True
        if sample['compactness'] > 150: return True
        if sample['ratio'] < 0.1: return True
        return False

    def get_folder_review_data(folder_name: str) -> Tuple[List[Dict], str]:
        folder_path = os.path.join(settings.DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            return [], "none"
            
        json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json') and not f.startswith('.')])
        
        # Load review results if exists
        review_file = os.path.join(folder_path, '.review_results.json')
        overrides = {}
        folder_status = "none"
        if os.path.exists(review_file):
            try:
                with open(review_file, 'r') as f:
                    data = json.load(f)
                    overrides = data.get('overrides', {})
                    folder_status = data.get('status', "none")
            except Exception as e:
                log.error(f"Error loading review file {review_file}: {e}")

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
                    
                    algo_failed = is_failed({
                        'ratio': ratio, 
                        'compactness': compactness, 
                        'solidity': solidity
                    })
                    
                    final_status = overrides.get(name, "fail" if algo_failed else "pass")
                    
                    samples.append({
                        "id": name,
                        "img_url": f"/api/render_image/{folder_name}/{name}",
                        "status": final_status,
                        "algo_failed": algo_failed
                    })
            except Exception as e:
                log.warning(f"Error processing {name} in {folder_name}: {e}")
                continue
                
        return samples, folder_status

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/folders')
    def get_folders():
        folder_list = []
        for root, dirs, files in os.walk(settings.DATA_DIR):
            if any(f.endswith('.json') and not f.startswith('.') for f in files):
                rel_path = os.path.relpath(root, settings.DATA_DIR)
                if rel_path == ".": continue
                folder_list.append(rel_path)
        return jsonify(sorted(folder_list))

    @app.route('/api/images/<path:folder_name>')
    def get_images(folder_name):
        samples, folder_status = get_folder_review_data(folder_name)
        return jsonify({"images": samples, "status": folder_status})

    @app.route('/api/render_image/<path:folder_name>/<image_name>')
    def render_image(folder_name, image_name):
        folder_path = os.path.join(settings.DATA_DIR, folder_name)
        img_path = os.path.join(folder_path, image_name + ".jpg")
        json_path = os.path.join(folder_path, image_name + ".json")
        xml_path = os.path.join(folder_path, image_name + ".xml")

        if not os.path.exists(img_path):
            return "Image not found", 404

        img = cv2.imread(img_path)
        if img is None:
            return "Failed to load image", 500

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

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    js_data = json.load(f)

                overlay = img.copy()
                for shape in js_data.get('shapes', []):
                    points = np.array(shape['points'], dtype=np.int32)
                    cv2.fillPoly(overlay, [points], (0, 0, 255))
                    cv2.polylines(img, [points], True, (0, 0, 255), 2)
                cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
            except: pass

        if bbox_coords:
            x1, y1, x2, y2 = bbox_coords
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes(), 200, {'Content-Type': 'image/jpeg'}

    @app.route('/api/save_review', methods=['POST'])
    def save_review():
        data = request.json
        folder_name = data.get('folder')
        overrides = data.get('overrides')
        
        folder_path = os.path.join(settings.DATA_DIR, folder_name)
        review_file = os.path.join(folder_path, '.review_results.json')
        
        save_data = {
            "status": "reviewed",
            "overrides": overrides
        }
        with open(review_file, 'w') as f:
            json.dump(save_data, f, indent=4)

        samples, _ = get_folder_review_data(folder_name)
        summary_path = settings.SUMMARY_CSV
        
        rows = []
        for s in samples:
            rows.append({
                "Folder": folder_name,
                "File_ID": s['id'],
                "Status": s['status'],
                "Image_Path": os.path.join(settings.DATA_DIR, folder_name, s['id'] + ".jpg")
            })
        
        new_df = pd.DataFrame(rows)
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            existing_df = existing_df[existing_df['Folder'] != folder_name]
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df
        
        final_df.to_csv(summary_path, index=False)
        return jsonify({"success": True})

    @app.route('/images/<path:filename>')
    def serve_image(filename):
        return send_from_directory(settings.DATA_DIR, filename)

    return app

def run_app():
    app = create_app()
    log.info(f"Starting {settings.APP_NAME} on {settings.HOST}:{settings.PORT}")
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG)

if __name__ == '__main__':
    run_app()
