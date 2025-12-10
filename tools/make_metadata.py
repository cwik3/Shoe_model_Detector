import json
from pathlib import Path

root = Path("img/clean")
items = []

if not root.exists():
    raise SystemExit("img/clean not found. Put your class folders under img/clean/<class>/image.jpg")

for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    cls = d.name
    for p in d.rglob("*.*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        items.append({
            "image_path": str(p).replace("\\", "/"),
            "class_name": cls,
            "model_name": p.stem
        })

Path("catalog").mkdir(exist_ok=True)
meta_path = Path("catalog/metadata.json")
meta_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
print(f"[OK] Wrote {len(items)} entries -> {meta_path}")
