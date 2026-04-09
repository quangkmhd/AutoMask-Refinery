# Configuration Guide: AutoMask Refinery

This document details the configuration parameters required to optimize the AutoMask Refinery for different hardware capabilities and dataset requirements.

## 1. Environment Variables (`.env`)

The Flask application behavior is controlled via environment variables. Create a `.env` file in the project root.

| Variable | Description | Recommended Value |
|----------|-------------|-------------------|
| `FLASK_ENV` | Sets the application mode (`development` or `production`). | `development` |
| `PORT` | The port the web interface binds to. | `5000` |
| `MAX_IMAGE_SIZE` | To prevent browser memory crashes, images larger than this width/height (in pixels) will be downscaled before rendering. | `1920` |
| `TEMP_DIR` | Directory for storing intermediate AI masks before the user hits "Save". | `./static/temp` |
| `OUTPUT_DIR` | The final destination for human-approved masks. | `./data/masks` |

## 2. YOLO Model Selection & Configuration

The choice of YOLO weights heavily impacts the speed vs. accuracy trade-off in the UI. Modify the `YOLOSegmenter` initialization in `src/app.py` based on your hardware.

### Official Ultralytics Weights
- **`yolov8n-seg.pt` (Nano):** Extremely fast, runs well on CPU. Good for distinct, high-contrast objects. (Recommended for basic laptops).
- **`yolov8s-seg.pt` (Small):** The default balance. Requires a modest GPU for real-time web UI feel.
- **`yolov8x-seg.pt` (Extra Large):** Highly accurate, catches fine details (like hair or thin wires). Requires a dedicated GPU (e.g., RTX 3080+); otherwise, the web UI will feel sluggish (3-5 seconds per click).

### Custom Weights
If you have fine-tuned YOLO on your specific dataset, you can point the application directly to your `best.pt` file:
```python
# src/app.py
segmenter = YOLOSegmenter(weights="./models/my_custom_dataset_best.pt")
```

## 3. Frontend Canvas Tool Tuning

If you are developing or modifying the HTML/JS frontend, these hardcoded parameters in the JavaScript dictate the user experience.

- **`BRUSH_SIZE` (Default: `20`):** Defines the pixel radius of the manual correction tool. Can be mapped to a UI slider.
- **`MASK_OPACITY` (Default: `0.5`):** The alpha channel value for the blue AI overlay. `0.5` allows the user to see the original image underneath the mask perfectly.
- **`MASK_COLOR` (Default: `rgba(0, 120, 255, 0.5)`):** The hex/rgba color of the mask. Change this if the objects you are segmenting are naturally blue, making the mask hard to see.

## 4. Docker Deployment Configuration

If using the recommended Docker setup, you can override configurations at runtime using docker arguments:

```bash
docker run -p 5000:5000 \
  -e MAX_IMAGE_SIZE=1080 \
  -v /path/to/your/raw/images:/app/data/images \
  -v /path/to/your/approved/masks:/app/data/masks \
  automask-refinery
```
*Note on Docker and GPUs: To utilize GPU acceleration within the container, you must install the NVIDIA Container Toolkit and append the `--gpus all` flag to your `docker run` command.*
