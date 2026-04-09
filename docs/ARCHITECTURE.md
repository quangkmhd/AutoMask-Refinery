# Architecture Deep Dive: AutoMask Refinery

## 1. System Overview

AutoMask Refinery is a hybrid computer vision system designed to bridge the gap between fully automated segmentation (which often produces small errors) and manual annotation (which is painfully slow). The architecture combines a zero-shot AI backend powered by Ultralytics YOLOv8 with an interactive, web-based frontend. This human-in-the-loop design allows data annotators to generate baseline masks instantly and refine them using brush and eraser tools before exporting them to standard dataset formats.

## 2. Core Architectural Components

### 2.1. The AI Inference Engine (`model/yolo_segmenter.py`)
This is the heavy-lifting module that interfaces with PyTorch.
- **Model:** Ultralytics YOLOv8 (Segmentation variant). It uses a CSPDarknet backbone and a decoupled head to predict both bounding boxes and instance segmentation masks simultaneously.
- **Processing:** When an image is passed to the engine, the image is resized, normalized, and run through the network. The output mask tensor is thresholded (typically > 0.5) and converted into a binary NumPy array (0 for background, 255 for the object).
- **Batch Processing:** For non-interactive workflows, this engine can be detached from the web server and run via a script, leveraging DataLoader multiprocessing to max out GPU utilization.

### 2.2. The Web Backend (`src/app.py`)
A lightweight, stateless Python Flask application.
- **Statelessness:** The backend does not maintain persistent memory of the masks between HTTP requests. Instead, state is passed back and forth via Base64 encoded images or temporary server-side session files, allowing the app to scale easily.
- **Endpoints:** It exposes RESTful endpoints for uploading images, requesting the initial YOLO mask, and saving the final, human-refined mask.
- **Data Conversion:** The backend is responsible for converting raw NumPy arrays into PNG formats with alpha channels so the browser can render them as overlays.

### 2.3. The Frontend Canvas Interface (HTML5/JavaScript)
The user interface where refinement happens.
- **Layering System:** Uses a multi-layered HTML5 `<canvas>` approach.
  - *Layer 0 (Bottom):* The original uploaded image.
  - *Layer 1 (Middle):* The AI-generated mask, rendered in a semi-transparent color (e.g., RGBA 0, 0, 255, 128).
  - *Layer 2 (Top):* The drawing context where the user's brush strokes (addition) or eraser strokes (subtraction) are captured.
- **Compositing:** When the user clicks "Save," the frontend JavaScript composites Layer 1 and Layer 2 using `globalCompositeOperation` to produce the final, definitive binary mask, which is then POSTed back to the server.

### 2.4. Data Management (`organize_files.py`)
A standalone utility architecture component.
- **Matching Logic:** Scans the `images/` and `masks/` directories to ensure every input has a corresponding label.
- **Stratified Splitting:** Uses `sklearn.model_selection.train_test_split` to randomly partition the dataset into `train`, `val`, and `test` directories based on user-defined ratios, ensuring reproducibility via a fixed random seed.
- **Tracking:** Maintains a `review_details.csv` file that logs which masks were purely AI-generated versus which ones required human intervention.

## 3. Data Flow Diagram

1. **Upload:** User uploads `image.jpg` via the browser.
2. **Inference:** Flask routes the image to `YOLOSegmenter`. YOLO returns a binary mask.
3. **Rendering:** Flask converts the mask to a blue, semi-transparent PNG and sends it to the browser.
4. **Interaction:** The user draws on the HTML5 Canvas to fix a missing edge.
5. **Composition:** The frontend merges the AI mask and the human drawings into a single blob.
6. **Save:** The blob is sent to Flask, which saves it as `mask.png` in the output directory.
7. **Export:** The user runs `organize_files.py` to package the data for PyTorch training.

## 4. Design Decisions & Trade-offs

- **Flask vs. FastAPI/Django:** Flask was chosen because the application relies on synchronous OpenCV and UI rendering tasks rather than massive concurrent I/O. The simplicity of Flask outweighs the async benefits of FastAPI for this specific local tool.
- **Browser Canvas vs. OpenCV GUI:** Previous versions of annotation tools used `cv2.imshow()`, which is clunky and platform-dependent. Moving to a web browser ensures cross-platform compatibility (Windows, Mac, Linux) without struggling with X11 forwarding or graphics drivers.
- **YOLOv8 over Mask R-CNN:** YOLOv8 provides real-time performance (often < 20ms per image on GPU) which is crucial for a responsive web UI, whereas Mask R-CNN, while highly accurate, introduces noticeable latency that frustrates annotators.
