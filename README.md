# Object Detection with Faster R-CNN 

An interactive **Streamlit web application** for detecting and identifying objects in images using **Faster R-CNN with a ResNet-50 FPN backbone**, pretrained on the COCO dataset.

The application allows users to upload an image or provide an image URL, detect multiple objects, visualize their locations using bounding boxes, and explore detection results with confidence scores.


## Features

- **Two Image Input Methods:** Upload an image (PNG, JPG, JPEG) or provide an image URL.
- **Object Detection:** Detect multiple objects in a single image using Faster R-CNN.
- **Bounding Box Visualization:** Display detected objects with green bounding boxes and class labels.
- **Confidence Scores:** Display the confidence score associated with each detection.
- **Adjustable Confidence Threshold:** Control the minimum confidence required to display a detection (0.05–1.0).
- **Class Filtering:** Optionally select specific object categories to display.
- **Detection Summary:** View the total number of detected objects and the count of each class.
- **Hardware Acceleration:** Automatically uses CUDA when an available GPU is detected; otherwise, runs on CPU.
- **Model Caching:** Load and cache the model using Streamlit's `st.cache_resource` to avoid reloading it on every interaction.

## Model Information

| Property | Description |
|---|---|
| Model | Faster R-CNN |
| Backbone | ResNet-50 with Feature Pyramid Network (FPN) |
| Framework | PyTorch / Torchvision |
| Pretrained Weights | `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` |
| Dataset | COCO |
| Detectable Classes | 80 object classes |
| Task | Object Detection |
| Execution | Inference only |

The model is based on the `fasterrcnn_resnet50_fpn` architecture provided by `torchvision`.

It is pretrained on COCO, supporting 80 everyday object classes, including people, animals, vehicles, furniture, food, and electronic devices.

## 🛠️ Technologies Used

- **Python** – Application development.
- **PyTorch** – Deep learning framework and model inference.
- **Torchvision** – Pretrained Faster R-CNN implementation and weights.
- **Streamlit** – Interactive web interface.
- **Pillow (PIL)** – Image loading, orientation correction, and bounding box visualization.
- **Requests** – Retrieving images from URLs.

## Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install dependencies

```bash
pip install streamlit torch torchvision pillow requests
```

### 3. Run the application

```bash
streamlit run object_detection_app.py
```

The application will automatically open in your default web browser.

## How to Use

1. Launch the application.
2. Choose one of the two input methods:
   - **Upload Image:** Upload a PNG, JPG, or JPEG image.
   - **Image URL:** Paste a direct URL to an image.
3. Adjust the detection options in the sidebar:
   - Set the confidence threshold.
   - Select specific object classes if needed.
   - Leave the class filter empty to include all detectable classes.
4. View the detection results:
   - The original image with labeled bounding boxes.
   - The total number of detected objects.
   - A summary of detected classes and their counts.
   - A table containing object names and confidence scores.

## Detection Configuration

| Parameter | Default | Description |
|---|---|---|
| Confidence Threshold | 0.70 | Minimum confidence score for a detection to be displayed. |
| Minimum Threshold | 0.05 | Lowest configurable confidence threshold. |
| Maximum Threshold | 1.00 | Highest configurable confidence threshold. |
| Class Filter | All classes | Optionally restrict displayed detections to selected classes. |


## Hardware & Model Weights

The application automatically selects the available execution device:

- **GPU (CUDA):** Used when CUDA is available.
- **CPU:** Used as a fallback when no CUDA-compatible GPU is available.

On the first execution, Torchvision automatically downloads the pretrained model weights (approximately **160 MB**) from `download.pytorch.org`.

The downloaded weights are cached locally by PyTorch, avoiding repeated downloads on subsequent executions.

CPU inference may take a few seconds per image, depending on the hardware and image dimensions.

## Notes & Limitations

- An internet connection is required for the initial model weights download and for retrieving images from URLs.
- Image URLs must point to accessible image resources.
- Only PNG, JPG, and JPEG are accepted through the upload interface.
- Detection results depend on the model's predictions and the selected confidence threshold.
- The application does not perform model training or fine-tuning.
- The model is limited to the 80 object classes defined by the COCO dataset.

## License

This project is intended for educational and demonstration purposes.