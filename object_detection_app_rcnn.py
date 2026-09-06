"""
Object Detection with Faster R-CNN — Streamlit App
====================================================
Detects and names objects in an image using Faster R-CNN pretrained on
COCO (torchvision). This is inference-only — no training required. The
first run downloads the pretrained weights (~160MB) from
download.pytorch.org; after that they're cached locally by torch.

How to run:
    $ pip install streamlit torch torchvision pillow requests
    $ streamlit run object_detection_app.py
"""

import io

import requests
import streamlit as st
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageDraw, ImageFont, ImageOps

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
SELECTABLE_CLASSES = sorted(
    {name for name in COCO_INSTANCE_CATEGORY_NAMES if name not in ("__background__", "N/A")}
)


# ---------------------------------------------------------------------------
# CACHED MODEL LOADER (downloaded/loaded only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Faster R-CNN (pretrained on COCO)...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model, device


# ---------------------------------------------------------------------------
# INFERENCE + POST-PROCESSING
# ---------------------------------------------------------------------------
def get_predictions(pred, threshold=0.8, objects=None):
    """
    Turn one raw Faster R-CNN prediction dict into a clean list of
    detections: class name, confidence score, and box coordinates.
    Mirrors the notebook's get_predictions, minus the torch-tensor
    round trip (done once up front instead of per box).
    """
    boxes = pred[0]["boxes"].detach().cpu().numpy()
    scores = pred[0]["scores"].detach().cpu().numpy()
    labels = pred[0]["labels"].detach().cpu().numpy()

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        name = COCO_INSTANCE_CATEGORY_NAMES[int(label)]
        if objects and name not in objects:
            continue
        detections.append({"class": name, "score": float(score), "box": [float(v) for v in box]})

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def run_detection(pil_img, model, device, threshold, objects=None):
    tensor_img = to_tensor(pil_img.convert("RGB")).to(device)
    with torch.no_grad():
        pred = model([tensor_img])
    return get_predictions(pred, threshold=threshold, objects=objects)


def draw_boxes(pil_img, detections):
    """Draw a labeled box for each detection using PIL (no OpenCV needed)."""
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    line_width = max(2, img.width // 300)
    font_size = max(14, img.width // 70)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f'{det["class"]}: {det["score"]:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=line_width)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        # Nudge the label above the box when there's room, otherwise inside it.
        label_y = text_bbox[1] - (text_bbox[3] - text_bbox[1]) - 4
        if label_y < 0:
            label_y = y1
        offset = label_y - text_bbox[1]
        tag_box = [text_bbox[0], text_bbox[1] + offset, text_bbox[2], text_bbox[3] + offset]

        draw.rectangle(tag_box, fill=(0, 200, 0))
        draw.text((x1, label_y), label, fill=(0, 0, 0), font=font)

    return img


def load_image_from_url(url):
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Faster R-CNN Object Detector", page_icon="🔎", layout="wide")

    st.title("🔎 Object Detection with Faster R-CNN")
    st.markdown(
        "Detects and labels objects in an image using **Faster R-CNN pretrained on COCO** "
        "(80 everyday object classes — people, animals, vehicles, furniture, food, and more). "
    )

    with st.sidebar:
        st.subheader("Options")
        threshold = st.slider(
            "Confidence threshold", min_value=0.05, max_value=1.0, value=0.7, step=0.05,
            help="Only show detections the model is at least this confident about.",
        )
        class_filter = st.multiselect(
            "Only show these classes (optional)",
            options=SELECTABLE_CLASSES,
            help="Leave empty to show every detected class.",
        )
        st.markdown("---")
        model, device = load_model()
        st.caption(f"Running on: **{device.upper()}**")
        if device == "cpu":
            st.caption("CPU inference: a few seconds per image is normal.")
        with st.expander("All 80 detectable classes"):
            st.write(", ".join(SELECTABLE_CLASSES))

    tab_upload, tab_url = st.tabs(["📤 Upload Image", "🔗 Image URL"])
    pil_img = None

    with tab_upload:
        uploaded_file = st.file_uploader("Choose an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            pil_img = ImageOps.exif_transpose(Image.open(uploaded_file))

    with tab_url:
        url = st.text_input("Paste an image URL")
        if url:
            try:
                pil_img = ImageOps.exif_transpose(load_image_from_url(url))
            except Exception as e:
                st.error(f"Couldn't load that image: {e}")

    if pil_img is None:
        st.info("Upload an image or paste a URL to run detection.")
        return

    objects = class_filter if class_filter else None
    with st.spinner("Detecting objects..."):
        detections = run_detection(pil_img, model, device, threshold, objects=objects)

    col_img, col_res = st.columns([3, 2])

    with col_img:
        if detections:
            st.image(draw_boxes(pil_img, detections), use_container_width=True)
        else:
            st.image(pil_img, use_container_width=True)
            st.warning("No objects detected above this confidence threshold. Try lowering it.")

    with col_res:
        st.subheader(f"Found {len(detections)} object(s)")
        if detections:
            counts = {}
            for det in detections:
                counts[det["class"]] = counts.get(det["class"], 0) + 1
            st.write(" • ".join(f"{name} ×{n}" for name, n in sorted(counts.items())))

            st.dataframe(
                [{"Object": d["class"], "Confidence": f'{d["score"]:.1%}'} for d in detections],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.caption("Model: torchvision `fasterrcnn_resnet50_fpn`, pretrained on COCO (91 categories, 80 non-background).")


if __name__ == "__main__":
    main()
