import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
from tools.smoothing import StableLabel
from tools.retrieval import classify_with_margin, l2_normalize
import os, csv

LOG_PATH = "Runs/debug/retrieval_log.csv"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["frame","det_conf","s1","s2","margin","label","unknown"])

# ---- CONFIG ----
YOLO_WEIGHTS = Path("Runs/detect/train2/weights/best.pt")
CONF = 0.25
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
DRAW_THICK = 2

# ---- OPTIONAL: CLIP + FAISS ----
USE_FAISS = True
INDEX_PATH = Path("indexes/v2/proto.faiss")
METADATA_PATH = Path("indexes/v2/proto_meta.json")
CLIP_MODEL_NAME, CLIP_PRETRAINED = "ViT-B-32", "openai"
RETRIEVE_EVERY_N = 3     # retrieval cadence
SIM_BADGE_THR = 0.35    # "~" badge threshold
TOPK = 5                # neighbors for voting
TAU = 0.25               # similarity threshold gate (loose for testing)
DELTA = 0.05             # margin gate

# ---- load models ----
print("[init] loading YOLO…")
model = YOLO(str(YOLO_WEIGHTS))

preprocess = None
clip_model = None
faiss_index = None
id2sku = None  # list mapping FAISS row -> class_name

if USE_FAISS:
    print("[init] loading CLIP + FAISS…")
    import json, faiss, open_clip
    device = "cpu"  # keep CPU unless you want to push to CUDA

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    clip_model.eval()

    faiss_index = faiss.read_index(str(INDEX_PATH))
    meta = json.loads(METADATA_PATH.read_text())

    # Build id2sku as a list so faiss row id -> class name
    if isinstance(meta, list):
        id2sku = [m["class_name"] for m in meta]
    else:
        # e.g., {"0": {...}, "1": {...}}
        max_id = max(int(k) for k in meta.keys())
        id2sku = [""] * (max_id + 1)
        for k, m in meta.items():
            id2sku[int(k)] = m["class_name"]

    # ---- sanity prints (AFTER loading index & meta)
    metric_name = {0: "METRIC_INNER_PRODUCT", 1: "METRIC_L2"}.get(
        getattr(faiss_index, "metric_type", -1), str(getattr(faiss_index, "metric_type", "unknown"))
    )
    print("[faiss] ntotal:", faiss_index.ntotal)
    print("[faiss] metric_type:", metric_name)
    print("[meta]  len(id2sku):", len(id2sku))
    if faiss_index.ntotal != len(id2sku):
        print("[WARN] FAISS rows != metadata entries. Voting/labels may be wrong.")

def embed_pil(pil_img):
    """Return L2-normalized CLIP embedding (float32 numpy, shape [D]) on CPU."""
    with torch.no_grad():
        t = preprocess(pil_img.convert("RGB")).unsqueeze(0)  # CPU tensor
        z = clip_model.encode_image(t).float()
        z = z / z.norm(dim=-1, keepdim=True)        # L2 normalize
        return z.cpu().numpy().astype("float32")[0] # -> [D]

# ------------- camera -------------
cv2.namedWindow("ShoeCam (q quit, +/- conf, r toggle retrieval)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ShoeCam (q quit, +/- conf, r toggle retrieval)", 960, 540)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
#cap.set(cv2.CAP_PROP_EXPOSURE, -3)          # try -7-/-3 depending on brightness
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

if not cap.isOpened():
    raise SystemExit(f"Could not open camera index {CAM_INDEX}")

print("[info] press 'q' to quit, '+' / '-' to change confidence, 'r' to toggle retrieval")
last_t = time.time()
frame_i = 0

# ---- temporal smoothing OUTSIDE the loop (non-blocking) ----
track_smoothers = defaultdict(lambda: StableLabel(
    maxlen=15, show_thr=0.0, hide_thr=0.0, stick_frames=8
))

# simple cache: crop hash -> (label, conf, s1_dbg, s2_dbg)
from hashlib import blake2b
retr_cache = {}

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_i += 1

    # YOLO inference
    res = model.predict(source=frame, conf=CONF, verbose=False)[0]
    boxes = res.boxes

    # draw boxes
    for bi, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls_idx = int(b.cls.item()) if b.cls is not None else -1
        det_conf = float(b.conf.item()) if b.conf is not None else 0.0

        # Default label text = YOLO info
        label_text = f"id {cls_idx}  {det_conf:.2f}"

        # Track id: if you don't have a tracker yet, use detection index 'bi'
        tid = bi

        # ---- Retrieval (only on some frames to save CPU) ----
        label_for_smoother, conf_for_smoother = None, 0.0
        s1_dbg, s2_dbg = None, None  # metric-neutral (higher = better)

        if USE_FAISS and (frame_i % RETRIEVE_EVERY_N == 0):
            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w - 1, x2), min(h - 1, y2)
            if x2c > x1c and y2c > y1c:
                pad_x = int(0.08 * (x2c - x1c))
                pad_y = int(0.08 * (y2c - y1c))
                x1p = max(0, x1c - pad_x)
                y1p = max(0, y1c - pad_y)
                x2p = min(w - 1, x2c + pad_x)
                y2p = min(h - 1, y2c + pad_y)
                crop_bgr = frame[y1p:y2p, x1p:x2p]

                if crop_bgr.size > 0:
                    hsh = blake2b(crop_bgr.tobytes(), digest_size=8).hexdigest()

                    if hsh in retr_cache:
                        label_for_smoother, conf_for_smoother, s1_dbg, s2_dbg = retr_cache[hsh]
                    else:
                        from PIL import Image
                        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                        q = embed_pil(crop_pil)  # L2-normalized

                        # Classify (handles L2/IP internally)
                        label_for_smoother, conf_for_smoother = classify_with_margin(
                            query_emb=q, faiss_index=faiss_index, id2sku=id2sku,
                            k=TOPK, tau=TAU, delta=DELTA
                        )

                        # HUD: get top-2 and convert to "higher is better"
                        sims2, ids2 = faiss_index.search(q[np.newaxis, :], 2)
                        raw1, raw2 = float(sims2[0][0]), float(sims2[0][1])
                        import faiss
                        metric = getattr(faiss_index, "metric_type", faiss.METRIC_INNER_PRODUCT)
                        s1_dbg = -raw1 if metric == faiss.METRIC_L2 else raw1
                        s2_dbg = -raw2 if metric == faiss.METRIC_L2 else raw2

                        retr_cache[hsh] = (label_for_smoother, conf_for_smoother, s1_dbg, s2_dbg)

        # ---- Temporal smoothing (anti-blink) ----
        stable = track_smoothers[tid].update(label_for_smoother, conf_for_smoother)
        label_val = stable[0] if stable else ""
        unknown = int(stable is None)
        s1v = 0.0 if s1_dbg is None else float(s1_dbg)
        s2v = 0.0 if s2_dbg is None else float(s2_dbg)
        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                frame_i, f"{det_conf:.3f}", f"{s1v:.4f}", f"{s2v:.4f}",
                f"{(s1v - s2v):.4f}", label_val, unknown
            ])

        # ---- Drawing ----
        color_box = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, DRAW_THICK)

        # HUD (always draw if we have numbers)
        if s1_dbg is not None and s2_dbg is not None:
            cv2.putText(frame, f"s1={s1_dbg:.2f} s2={s2_dbg:.2f} (tau={TAU:.2f}, d={DELTA:.2f})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        if stable is not None:
            stable_label, stable_conf = stable
            # similarity badge from s1_dbg
            if s1_dbg is None:
                badge = ""
            else:
                badge = "OK" if s1_dbg >= 0.45 else ("~" if s1_dbg >= SIM_BADGE_THR else "LOW")
                badge = f" {badge}"

            label_text += f" | {stable_label} ({stable_conf:.2f}{badge})"

            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_txt = max(0, y1 - 8)
            cv2.rectangle(frame, (x1, y_txt - th - 4), (x1 + tw + 4, y_txt), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (x1 + 2, y_txt - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # If not stable (Unknown or not confident yet), draw a small gray tag
            gray = (160, 160, 160)
            (tw, th), _ = cv2.getTextSize(label_text + " | …", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_txt = max(0, y1 - 8)
            cv2.rectangle(frame, (x1, y_txt - th - 4), (x1 + tw + 4, y_txt), gray, -1)
            cv2.putText(frame, label_text + " | …", (x1 + 2, y_txt - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)




    # FPS
    now = time.time()
    fps = 1.0 / (now - last_t) if now > last_t else 0.0
    last_t = now
    cv2.putText(frame, f"CONF={CONF:.2f} | RETR={'ON' if USE_FAISS else 'OFF'} | FPS={fps:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("ShoeCam (q quit, +/- conf, r toggle retrieval)", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k in (43, ord('=')):  # '+' or '='
        CONF = min(0.9, CONF + 0.05)
    elif k == ord('-'):
        CONF = max(0.05, CONF - 0.05)
    elif k == ord('r'):
        USE_FAISS = not USE_FAISS

cap.release()
cv2.destroyAllWindows()
print("[done] closed camera.")
