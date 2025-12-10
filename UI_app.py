# app.py — YOLO + CLIP + FAISS viewer (CPU-friendly)
import json, faiss, torch, open_clip
from PIL import Image
from pathlib import Path
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---- CONFIG ----
YOLO_WEIGHTS = Path("Runs/detect/train2/weights/best.pt")  # adjust if needed
INDEX_PATH   = Path("indexes/v1/index.faiss")
META_PATH    = Path("indexes/v1/meta.json")
MODEL_NAME, PRETRAINED = "ViT-B-32", "openai"
CONF_THRESH = 0.25
SIM_CONF    = 0.30  # below this, show "Low confidence"
TOPK        = 5

st.set_page_config(page_title="Shoe Finder", layout="wide")
st.title("👟 Shoe Detector + Similarity Search")

@st.cache_resource
def load_all():
    # YOLO
    yolo = YOLO(str(YOLO_WEIGHTS))
    # CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device="cpu")
    model.eval()
    # FAISS
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(META_PATH.read_text())
    return yolo, model, preprocess, index, meta

yolo, clip_model, preprocess, index, meta = load_all()

def embed_img(pil: Image.Image):
    with torch.no_grad():
        t = preprocess(pil.convert("RGB")).unsqueeze(0)  # CPU tensor
        z = clip_model.encode_image(t)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy().astype("float32")

def search_sim(pil: Image.Image, k=TOPK):
    q = embed_img(pil)
    sims, ids = index.search(q, k)
    out = []
    for s, i in zip(sims[0], ids[0]):
        m = meta[int(i)]
        out.append((float(s), m["class_name"], m.get("model_name",""), m["image_path"]))
    return out

file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)
    res = yolo.predict(source=np.array(img), conf=CONF_THRESH, verbose=False)[0]

    if len(res.boxes) == 0:
        st.warning("No shoes detected. Try a different image or lower the confidence threshold.")
    else:
        for i, b in enumerate(res.boxes):
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            crop = img.crop((x1,y1,x2,y2))
            cls_idx = int(b.cls.item())
            conf = float(b.conf.item())
            col1, col2 = st.columns([1,2], gap="large")
            with col1:
                st.image(crop, caption=f"Detected box #{i+1} — YOLO class {cls_idx} @ {conf:.2f}")

            with col2:
                st.markdown("**Top matches (CLIP + FAISS):**")
                results = search_sim(crop, k=TOPK)
                for s, cls, name, path in results:
                    badge = "🟢" if s >= 0.45 else ("🟡" if s >= SIM_CONF else "🟠 Low confidence")
                    st.write(f"{badge} **{cls}** — *{name}*  (sim {s:.3f})")
