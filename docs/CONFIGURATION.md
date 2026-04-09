# Configuration Guide: AutoMask Refinery

This document details the configuration parameters required to optimize the AutoMask Refinery for different hardware capabilities and dataset requirements.

## 1. Environment Variables (`.env`)

The application consumes settings from environment variables. You can also create a `.env` file in the project root. All variables must be prefixed with `AUTOMASK_`.

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTOMASK_DEBUG` | Enable debug mode for Flask development. | `True` |
| `AUTOMASK_PORT` | The port the web interface binds to. | `5000` |
| `AUTOMASK_HOST` | The host interface to bind to. | `0.0.0.0` |
| `AUTOMASK_DATA_DIR` | Path to the root dataset containing subfolders. | `./data/demo` |
| `AUTOMASK_SUMMARY_CSV` | Path to the output review tracker CSV. | `review_details.csv` |

## 2. Model Configuration

The `MaskGenerator` class handles model loading. You can specify the model path via the CLI:

```bash
automask generate --sam-model ./models/sam3.pt --imgsz 1036
```

### Hardware Acceleration
The system automatically detects CUDA availability. To force a specific device, you can modify the environment or use the CLI (if exposed in specific commands).

## 3. UI Refinement Logic

The interactive review logic uses specific geometric thresholds to flag masks for human attention:

- **Solidity Threshold (`< 0.65`)**: Flags masks that are too "holey" or not convex enough.
- **Compactness Threshold (`> 150`)**: Flags masks with jagged, noisy, or complex edges (likely spidery artifacts).
- **Area Ratio Threshold (`< 0.1`)**: Flags masks that are suspiciously small relative to their bounding box.

These are currently implemented in the `is_failed` helper within `app.py` and the `MaskDetector` class for consistency.

## 4. Advanced Settings (Pydantic)

Behind the scenes, the project uses a `Settings` class in `automask_refinery/config/settings.py`. For advanced customization (e.g., changing output folder names like `review_failures`), modify this class directly or set the corresponding environment variable (e.g., `AUTOMASK_REVIEW_OUT`).

## 5. Docker Usage

If running via Docker, you can pass configurations as environment variables:

```bash
docker run -p 5000:5000 \
  -e AUTOMASK_DATA_DIR=/data/my_project \
  -v /local/path:/data/my_project \
  automask-refinery
```
