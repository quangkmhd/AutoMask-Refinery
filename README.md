# AutoMask-Refinery

AutoMask-Refinery is a tool for generating, validating, and refining segmentation masks using SAM (Segment Anything Model) and various quality control metrics.

## Features
- **SAM-based Mask Generation**: Automatically generate masks from bounding boxes.
- **Automated Quality Control**: Filter out "failed" masks using heuristics, statistical outliers, and CLIP-based semantic verification.
- **Interactive Review Web App**: Manually review and override mask statuses through a Flask-based web interface.

![Review App Screenshot](assets/review_app_screenshot.png)

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Masks
Generate masks from images and Pascal VOC XML bounding boxes.
```bash
python src/generate_sam3_masks.py --input_dir data/demo --sam_model sam3.pt
```

### 2. Detect Failed Masks
Run the automated QC pipeline to flag potentially bad masks.
```bash
python src/failed_mask_detector.py --data_dir data/demo
```

### 3. Interactive Review
Launch the web app to manually review and refine the results.
```bash
python src/app.py --data_dir data/demo --csv_out data/demo/review_details.csv
```
Options for `app.py`:
- `--data_dir`: Path to the image dataset (default: `data/demo`).
- `--csv_out`: Path to save the final review results (default: `data/demo/review_details.csv`).
- `--port`: Flask port (default: `5000`).
- `--host`: Flask host (default: `0.0.0.0`).

## Project Structure
- `src/`: Core logic and scripts.
- `data/demo/`: Sample data for demonstration.
- `src/templates/`: HTML templates for the review app.

## License
MIT
