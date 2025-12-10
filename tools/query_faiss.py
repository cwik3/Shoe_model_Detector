
import json, faiss, torch, numpy as np
import open_clip
from PIL import Image
from pathlib import Path

DEVICE = "cpu"
MODEL_NAME, PRETRAINED = "ViT-B-32", "openai"
INDEX = "indexes/v1/index.faiss"
META  = "indexes/v1/meta.json"

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED, device=DEVICE)
model.eval()
index = faiss.read_index(INDEX)
meta  = json.loads(Path(META).read_text())

def embed(p):
    with torch.no_grad():
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
        z = model.encode_image(img)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy().astype("float32")

def search(image_path, k=5):
    q = embed(image_path)
    sims, ids = index.search(q, k)
    out=[]
    for s, i in zip(sims[0], ids[0]):
        m = meta[int(i)]
        out.append((float(s), m["class_name"], m.get("model_name",""), m["image_path"]))
    return out

if __name__ == "__main__":
    # pick one of your crops
    crop = sorted(Path("crops/val").glob("*.*"))[0]
    print(f"Query: {crop}")
    for s, cls, name, path in search(str(crop), k=5):
        print(f"{s:.3f}  {cls:40s}  {name}  <- {path}")
