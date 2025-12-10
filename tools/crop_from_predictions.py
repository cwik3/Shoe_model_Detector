
import os, cv2, json
from pathlib import Path
from ultralytics import YOLO

SRC = "data/images/val"
OUT = "crops/val"
MODEL = r"Runs/detect/train2/weights/best.pt"
CONF = 0.25

Path(OUT).mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL)

for img_path in Path(SRC).glob("*.*"):
    if img_path.suffix.lower() not in [".jpg",".jpeg",".png",".bmp",".webp"]:
        continue
    res = model.predict(source=str(img_path), conf=CONF, verbose=False)
    if not res: continue
    im = cv2.imread(str(img_path))
    for i, b in enumerate(res[0].boxes):
        x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
        # optional padding 5%
        h, w = im.shape[:2]
        pad_x = int(0.05*(x2-x1)); pad_y = int(0.05*(y2-y1))
        x1p = max(0, x1-pad_x); y1p = max(0, y1-pad_y)
        x2p = min(w-1, x2+pad_x); y2p = min(h-1, y2+pad_y)
        crop = im[y1p:y2p, x1p:x2p]
        out_name = f"{img_path.stem}_{i}.jpg"
        cv2.imwrite(str(Path(OUT)/out_name), crop)
