# streamlit_app.py
import os
import io
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO


# =========================
# Streamlit page config
# =========================
st.set_page_config(
    page_title="Image Classification + Similarity",
    page_icon="🧠",
    layout="wide"
    # initial_sidebar_state="expanded",
)
st.title("🧠 Clothes Classification")


# =========================
# Constants / Paths
# =========================

user_selection = st.selectbox("Select Category", ["Mens", "Kids","Womens"], key="category_select")

if( user_selection == "Mens"):
    MODEL_PATH = os.path.normpath("models\mens_model.pt")
    MENS_BASE_DIR = r"IMAGES\MENS"
elif( user_selection == "Womens"):
    MODEL_PATH = os.path.normpath("models\ladies_model.pt")
    MENS_BASE_DIR = r"IMAGES\WOMENS"
elif( user_selection == "Kids"):
    MODEL_PATH = os.path.normpath("models\kids_model.pt")
    MENS_BASE_DIR = r"IMAGES\KIDS"


FCLIP_NAME = "patrickjohncyh/fashion-clip"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

TOP_K = st.sidebar.number_input("Top-K similar images", min_value=1, max_value=20, value=3, step=1)
BATCH_SIZE = st.sidebar.number_input("Batch size (embedding)", min_value=8, max_value=256, value=64, step=8)


# =========================
# Lazy loaders (cached)
# =========================
@st.cache_resource(show_spinner=False)
def load_yolo_classifier(model_path: str):
    return YOLO(os.path.normpath(model_path))


@st.cache_resource(show_spinner=False)
def load_fashionclip(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device


# =========================
# Utilities
# =========================
def load_image_for_model(uploaded_file):
    """
    Streamlit's UploadedFile -> PIL.Image in RGB. Also accepts str | Path.
    """
    if isinstance(uploaded_file, (str, os.PathLike, Path)):
        return Image.open(uploaded_file).convert("RGB")
    data = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return Image.open(io.BytesIO(data)).convert("RGB")


def classify_image(model: YOLO, img: Image.Image) -> Tuple[str, float]:

    results = model.predict(img, verbose=False)
    r0 = results[0]
    probs = r0.probs
    top_idx = probs.top1
    top_conf = float(probs.top1conf)
    top_name = r0.names[top_idx]
    return top_name, top_conf


def iter_image_paths(folder: str) -> List[str]:
    """Recursively collect valid image paths from a folder."""
    paths = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(root, fname))
    return paths


@torch.no_grad()
def encode_images_in_batches(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    paths: List[str],
    batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Returns (features [N, D] on device, kept_paths) for successfully loaded images.
    """
    feats_list = []
    kept_paths = []
    batch_imgs, batch_paths = [], []

    def flush():
        nonlocal batch_imgs, batch_paths, feats_list, kept_paths
        if not batch_imgs:
            return
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        img_feats = model.get_image_features(**inputs).float()
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        feats_list.append(img_feats)
        kept_paths.extend(batch_paths)
        batch_imgs, batch_paths = [], []

    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            # unreadable image; skip
            continue
        batch_imgs.append(im)
        batch_paths.append(p)
        if len(batch_imgs) == batch_size:
            flush()

    flush()

    if not feats_list:
        raise RuntimeError("No images were successfully loaded/encoded from the dataset folder.")
    return torch.cat(feats_list, dim=0), kept_paths


@torch.no_grad()
def encode_single_pil(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    pil_img: Image.Image,
) -> torch.Tensor:
    inputs = processor(images=[pil_img.convert("RGB")], return_tensors="pt").to(device)
    q = model.get_image_features(**inputs).float()
    q = q / q.norm(dim=-1, keepdim=True)
    return q  # [1, D]


def norm_path_case(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


# =========================
# Cache per-class index (paths + embeddings)
# =========================
@st.cache_resource(show_spinner=False)
def build_or_get_class_index(
    class_dir: str,
    model_name: str,
    batch_size: int,
) -> Tuple[torch.Tensor, List[str]]:

    model, processor, device = load_fashionclip(model_name)
    with st.spinner(f"Indexing images in: {class_dir}"):
        paths = iter_image_paths(class_dir)
        if len(paths) == 0:
            raise RuntimeError("No images found in the predicted class directory.")
        feats_dev, kept_paths = encode_images_in_batches(model, processor, device, paths, batch_size)
        feats_cpu = feats_dev.cpu()  # store on CPU
    return feats_cpu, kept_paths


# =========================
# Load models
# =========================
with st.spinner("Loading models…"):
    clf_model = load_yolo_classifier(MODEL_PATH)
    fclip_model, fclip_processor, fclip_device = load_fashionclip(FCLIP_NAME)
st.success(f"Models ready. Device for similarity: {fclip_device}")


# =========================
# File uploader
# =========================
images = st.file_uploader(
    label="Upload one or more images",
    type=["jpg", "jpeg", "png", "JPG", "PNG"],
    accept_multiple_files=True
)

run_button = st.button("Classify & Find Similar")


# =========================
# Main logic
# =========================
if run_button:
    if not images:
        st.warning("Please upload at least one image.")
    else:
        for i, up in enumerate(images, start=1):
            st.markdown("---")
            st.subheader(f"Image #{i}")

            # 1) Load
            try:
                pil_img = load_image_for_model(up)
            except Exception as e:
                st.error(f"Could not read uploaded image: {e}")
                continue

            # 2) Show uploaded image
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(pil_img, caption="Uploaded image", use_container_width=True)

            # 3) Classify with YOLO
            try:
                with st.spinner("Classifying…"):
                    pred_class, conf = classify_image(clf_model, pil_img)
                with c2:
                    st.markdown(f"**Predicted class:** `{pred_class}`")
                    # st.markdown(f"**Confidence:** `{conf:.4f}`")
            except Exception as e:
                st.error(f"Classification failed: {e}")
                continue

            # 4) Build class directory path
            class_dir = os.path.join(MENS_BASE_DIR, pred_class)

            if not os.path.isdir(class_dir):
                st.error(f"Predicted class folder not found:\n`{class_dir}`")
                continue

            # 5) Get/Build index for that class (cached)
            try:
                feats_cpu, kept_paths = build_or_get_class_index(
                    class_dir=class_dir,
                    model_name=FCLIP_NAME,
                    batch_size=int(BATCH_SIZE),
                )
            except Exception as e:
                st.error(f"Indexing failed for class '{pred_class}': {e}")
                continue

            # 6) Encode the query image
            try:
                with torch.no_grad():
                    q = encode_single_pil(fclip_model, fclip_processor, fclip_device, pil_img)
                    feats = feats_cpu.to(fclip_device, non_blocking=True)
                    sims = (feats @ q.T).squeeze(1)  # [N]
                    # Bring scores to CPU for sorting / display
                    sims_cpu = sims.float().cpu()
            except Exception as e:
                st.error(f"Similarity encoding failed: {e}")
                continue

            # 7) Pair, exclude exact same file if it’s somehow in the index
            pairs: List[Tuple[str, float]] = []
            uploaded_identity = ""  # unknown; uploaded file isn’t a path on disk
            for p, s in zip(kept_paths, sims_cpu.tolist()):
                if uploaded_identity and norm_path_case(p) == norm_path_case(uploaded_identity):
                    continue
                pairs.append((p, s))

            # 8) Sort & take top-K
            pairs.sort(key=lambda x: x[1], reverse=True)
            top_matches = pairs[:TOP_K]

            st.markdown(f"**Top {TOP_K} similar in** `.../mens/{pred_class}`:")
            if not top_matches:
                st.info("No similar images found (index empty or all unreadable).")
                continue

            # 9) Display matches
            cols = st.columns(len(top_matches))
            for idx, (p, score) in enumerate(top_matches):
                with cols[idx]:
                    try:
                        thumb = Image.open(p).convert("RGB")
                        st.image(thumb, use_container_width=True)
                    except Exception:
                        st.image(pil_img, caption=f"(Couldn't open) Similarity: {score:.4f}", use_container_width=True)
                    barcode = p.split('\\')[-1].split('.')[0]
                    print("this is print",barcode)
                    st.caption(f"Barcode: {barcode}")
