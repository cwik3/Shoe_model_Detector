import json, faiss, torch, numpy as np
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEVICE = "cpu"  # keep CPU for your setup
MODEL_NAME, PRETRAINED = "ViT-B-32", "openai"
GALLERY = Path("catalog/images")
META_PATH = Path("catalog/metadata.json")
OUT_DIR = Path("indexes/v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert META_PATH.exists(), "catalog/metadata.json not found. Create it first."

print("[index] loading CLIP…")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
model.eval()

meta = json.loads(META_PATH.read_text())
vecs, kept = [], []
with torch.no_grad():
    for m in tqdm(meta, desc="Embedding gallery"):
        p = Path(m["image_path"])
        if not p.exists(): continue
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
        z = model.encode_image(img)
        z = z / z.norm(dim=-1, keepdim=True)
        vecs.append(z.cpu().numpy())
        kept.append(m)

assert vecs, "No gallery images embedded"
X = np.vstack(vecs).astype("float32")

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)
faiss.write_index(index, str(OUT_DIR/"index.faiss"))
np.save(OUT_DIR/"vectors.npy", X)
(OUT_DIR/"meta.json").write_text(json.dumps(kept, indent=2))
print(f"[index] saved {len(kept)} items → {OUT_DIR/'index.faiss'}")