# API & Class Reference: AutoMask Refinery

This document outlines the internal APIs, web endpoints, and CLI tools provided by AutoMask Refinery.

## 1. Unified CLI (`automask`)

The core entrypoint for all features.

### `automask ui`
Launches the interactive review dashboard.
- `--port`: Port for the Flask server (Default: 5000).
- `--host`: Host address (Default: 0.0.0.0).
- `--data-dir`: Path to the image dataset.

### `automask detect`
Runs the automated quality control pipeline.
- `--data-dir`: Path to dataset.
- `--review-out`: Directory to save failed mask visualizations.
- `--passed-out`: Directory to save passed mask visualizations.

### `automask generate`
Batch generates masks using SAM3.
- `--sam-model`: Path to weights.
- `--imgsz`: Inference resolution.

## 2. Flask Web Endpoints (`app.py`)

### `GET /api/folders`
Returns a list of subfolders in the data directory that contain image/mask pairs.

### `GET /api/images/<path:folder_name>`
Returns metadata for all images in a specific folder, including their PASS/FAIL status and algorithmically detected failures.

### `GET /api/render_image/<path:folder_name>/<image_name>`
Generates a real-time visualization snippet of the image with the AI mask overlay (red) and bounding box (green).

### `POST /api/save_review`
Saves manual PASS/FAIL overrides for a specific folder.
- **Request Body:**
  ```json
  {
      "folder": "Necessities_ADCO",
      "overrides": {
          "image_001": "pass",
          "image_002": "fail"
      }
  }
  ```

## 3. Core Package Classes

### `automask_refinery.core.detector.MaskDetector`
Implements the multi-stage QC pipeline.
- `detect_ratio_outliers()`: Statistical MAD check.
- `detect_shape_outliers()`: Isolation Forest anomaly detection.
- `run_pipeline()`: Executes all filters and returns combined results.

### `automask_refinery.core.generator.MaskGenerator`
Interfaces with SAM3 for high-precision mask generation.
- `generate_for_directory()`: Batch processing with detailed progress tracking.

### `automask_refinery.utils.organizer.FileOrganizer`
Handles dataset packaging.
- `organize(move=False)`: Copies or moves files into pass/fail hierarchies.
