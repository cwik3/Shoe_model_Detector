# tools/build_prototype_index.py
import json, numpy as np, faiss, open_clip
from PIL import Image
from pathlib import Path

# ---- paths ----
METADATA   = Path("indexes/v1/meta.json")   # your existing per-image metadata
IMG_ROOT   = Path(".")                      # change if paths in meta are relative to somewhere else
OUT_INDEX  = Path("indexes/v2/proto.faiss")
OUT_META   = Path("indexes/v2/proto_meta.json")
OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)

# CLIP
MODEL_NAME, PRETRAINED, DEVICE = "ViT-B-32", "openai", "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
model.eval()

# - load meta
meta = json.loads(METADATA.read_text())
# group image paths per class (stable order by sorting class names)
by_class = {}
for m in meta:
    cls = m["class_name"]
    by_class.setdefault(cls, []).append(m["image_path"])
classes = sorted(by_class.keys())

def embed_one(path: Path) -> np.ndarray:
    from torch import no_grad
    with no_grad():
        im = Image.open(path).convert("RGB")
        t = preprocess(im).unsqueeze(0)            # CPU
        z = model.encode_image(t).float()
        z = z / z.norm(dim=-1, keepdim=True)       # L2 normalize
        return z.cpu().numpy().astype("float32")[0]

vecs = []
kept_classes = []
missing = 0

for cls in classes:
    embs = []
    for rel in by_class[cls]:
        p = (IMG_ROOT / rel)
        if not p.exists():
            missing += 1
            continue
        try:
            embs.append(embed_one(p))
        except Exception:
            # bad image file; skip
            missing += 1
            continue
    if not embs:
        continue
    # prototype = mean of normalized embeddings, re-normalized
    proto = np.mean(np.stack(embs, axis=0), axis=0)
    proto = proto / max(1e-12, np.linalg.norm(proto))
    vecs.append(proto.astype("float32"))
    kept_classes.append(cls)

assert len(vecs) > 0, "No valid images found to build prototypes."

X = np.ascontiguousarray(np.stack(vecs, axis=0))  # [C, D]
index = faiss.IndexFlatIP(X.shape[1])             # cosine via dot on L2-normalized vecs
index.add(X)

faiss.write_index(index, str(OUT_INDEX))
# store in same row order as added to faiss
json.dump([{"class_name": c} for c in kept_classes], open(OUT_META, "w"), indent=2)

print(f"[proto] wrote {OUT_INDEX}  rows={index.ntotal}  missing_files_skipped={missing}")
print(f"[proto] meta -> {OUT_META}")
