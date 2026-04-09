# Architecture Deep Dive: AutoMask Refinery

## 1. System Overview

AutoMask Refinery is a hybrid computer vision system designed to bridge the gap between fully automated segmentation (which often produces small errors) and manual annotation (which is painfully slow). The architecture combines a zero-shot AI backend powered by SAM3/YOLO with an interactive, web-based frontend. This human-in-the-loop design allows data annotators to generate baseline masks instantly, detect potential failures statistically, and refine them before exporting to standard dataset formats.

## 2. Core Architectural Components

The project follows a modular Python package structure located in `src/automask_refinery/`.

### 2.1. The AI Inference Engine (`core/generator.py`)
This module handles the heavy processing using SAM3 or YOLO models.
- **MaskGenerator**: A dedicated class that manages model loading, XML parsing (Pascal VOC), and mask generation. It converts raw model outputs into high-quality LabelMe-compatible polygons.
- **Retina Masks**: Utilizes high-resolution inference (imgsz=1036+) to capture fine details often missed by standard real-time detectors.

### 2.2. The Failure Detection System (`core/detector.py`)
A production-grade pipeline to identify "broken" masks before they reach human review.
- **Statistical Analysis**: Uses Median Absolute Deviation (MAD) to find area-ratio outliers within a class.
- **Machine Learning**: Employs **Isolation Forest** on shape signatures (compactness, solidity, Hu moments) to detect semantic anomalies.
- **Heuristics**: Fast rules to catch "spidery" or noisy masks based on geometric compactness and solidity thresholds.

### 2.3. The Web Interface (`app.py`)
A modular Flask application implemented using the Factory Pattern.
- **Dynamic Data Loading**: Interacts with the `Settings` system to browse datasets dynamically.
- **Interactive Refinement**: Exposes RESTful endpoints for real-time mask rendering and manual PASS/FAIL status overrides.
- **Persistence**: Saves review results to `.review_results.json` within the data folders and updates a global `review_details.csv`.

### 2.4. Unified CLI (`main.py`)
A Typer-powered command-line interface that provides a single entrypoint for all workflows:
- `automask generate`: Batch mask creation.
- `automask detect`: Automated quality control.
- `automask ui`: Interactive refinement dashboard.
- `automask organize`: Final dataset packaging.

### 2.5. Configuration & Utilities (`config/`, `utils/`)
- **Pydantic Settings**: Centralized schema-based configuration with environment variable support.
- **Logger**: Professional logging via Loguru with specialized formatting.
- **Geometric Utils**: Deduplicated math logic for mask metrics (area, compactness, solidity).

## 3. Data Flow Diagram

1. **Generation**: `MaskGenerator` processes raw images and XMLs, saving `labelme.json` files.
2. **QC Check**: `MaskDetector` runs the pipeline, identifying potentially failed masks.
3. **Manual Review**: `app.py` launches the UI, allowing a human to verify the `detect` results.
4. **Outcome**: Human overrides are saved, and `review_details.csv` is updated.
5. **Organization**: `FileOrganizer` moves/copies the "Passed" data into a clean training hierarchy.

## 4. Design Decisions & Trade-offs

- **src/ layout**: Chosen for better package isolation and to prevent accidental imports of test/utility code in production.
- **Pybind/Pydantic**: Used for robust validation of inputs and settings, ensuring the app handles missing data gracefully.
- **Isolation Forest**: Selected for its effectiveness in high-dimensional anomaly detection (Hu moments) without needing a labeled "failure" dataset.
