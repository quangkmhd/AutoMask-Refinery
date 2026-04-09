# API & Class Reference: AutoMask Refinery

This document outlines the internal APIs, web endpoints, and CLI tools provided by AutoMask Refinery for dataset generation.

## 1. Flask Web Endpoints (`src/app.py`)

### `POST /upload`
Receives the raw image from the user, saves it temporarily, and runs the YOLO model to generate the initial mask.
- **Request Body:** `multipart/form-data` containing a `file` field.
- **Response (JSON):**
  ```json
  {
    "status": "success",
    "image_url": "/static/temp/image123.jpg",
    "mask_url": "/static/temp/mask123.png"
  }
  ```

### `POST /save_mask`
Accepts the final, user-corrected mask from the frontend Canvas and saves it to the permanent dataset directory.
- **Request Body:** JSON containing the Base64 encoded image data.
  ```json
  {
    "filename": "image123.jpg",
    "mask_data": "data:image/png;base64,iVBORw0KGgo..."
  }
  ```
- **Response (JSON):**
  ```json
  {
    "status": "saved",
    "path": "/data/masks/image123.png"
  }
  ```

## 2. Core Python Classes

### `model.yolo_segmenter.YOLOSegmenter`
The primary interface to the Ultralytics framework.

#### `__init__(self, weights: str = "yolov8n-seg.pt", device: str = None)`
Loads the PyTorch model into memory.
- **weights:** Path to a custom `.pt` file or an official Ultralytics model name (e.g., `yolov8s-seg.pt`, `yolov8x-seg.pt`).

#### `predict_mask(self, image_path: str) -> numpy.ndarray`
Runs inference on a single image.
- **Returns:** A binary NumPy array of shape `(H, W)` where `255` denotes the segmented object and `0` denotes the background. If multiple objects are detected, their masks are logically OR'ed together (unless configured for multi-class).

#### `save_mask(self, mask_array: numpy.ndarray, output_path: str)`
Utility to convert the NumPy array to a disk-backed PNG using OpenCV (`cv2.imwrite`).

## 3. Command Line Utilities

### `organize_files.py`
A vital script for preparing the exported masks for machine learning pipelines.

**Usage:**
```bash
python organize_files.py \
    --input_images ./data/raw_images \
    --input_masks ./data/masks \
    --output_dir ./data/dataset \
    --split 0.8
```

**Arguments:**
- `--input_images` (Required): Directory containing the original `.jpg`/`.png` files.
- `--input_masks` (Required): Directory containing the generated binary masks.
- `--output_dir` (Required): Where to build the `train/` and `val/` subdirectories.
- `--split`: (Default: `0.8`) Float representing the percentage of data to allocate to the training set. The remainder goes to validation.
- `--seed`: (Default: `42`) Random seed for reproducible dataset splitting.

**Behavior:**
1. Validates that every image has a mask with the exact same base filename.
2. Shuffles the dataset.
3. Copies files into the new directory structure:
   - `output_dir/train/images/`
   - `output_dir/train/masks/`
   - `output_dir/val/images/`
   - `output_dir/val/masks/`
