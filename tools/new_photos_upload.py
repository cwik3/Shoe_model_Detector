#!/usr/bin/env python3
import argparse, csv, hashlib, os
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg",".jpeg",".png",".webp"}

def sha256(p: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def ok_image(p: Path):
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="staging root with class subfolders")
    ap.add_argument("--dst", default="data/raw", help="dataset root with same class subfolders")
    ap.add_argument("--manifest", default="data/raw/images_manifest.csv")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)

    known = set()
    if manifest.exists():
        for row in csv.DictReader(manifest.open("r", encoding="utf-8")):
            if row.get("hash"): known.add(row["hash"])

    rows = []
    for cls_dir in [p for p in src_root.iterdir() if p.is_dir()]:
        class_name = cls_dir.name
        (dst_root / class_name).mkdir(parents=True, exist_ok=True)

        for p in cls_dir.rglob("*"):
            if not (p.is_file() and p.suffix.lower() in IMG_EXTS): continue
            if not ok_image(p):
                print(f"[skip] corrupt: {p}")
                continue
            h = sha256(p)
            if h in known:
                print(f"[dup]  {p.name} already in manifest")
                continue

            # unique name by content hash (prevents overwrite)
            dst = (dst_root / class_name / f"{p.stem}_{h[:8]}{p.suffix.lower()}")
            if not dst.exists():
                dst.write_bytes(p.read_bytes())
                print(f"[ok]   {dst}")

            rows.append({
                "path": str(dst.as_posix()),
                "hash": h,
                "width": "",
                "height": "",
                "source": class_name,
                "added_at": "",
                "label_status": "unlabeled"
            })
            known.add(h)

    if rows:
        write_header = not manifest.exists()
        with manifest.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path","hash","width","height","source","added_at","label_status"])
            if write_header: w.writeheader()
            w.writerows(rows)
        print(f"\nDone. Added {len(rows)} images. Manifest: {manifest}")
    else:
        print("\nNothing new added.")

if __name__ == "__main__":
    main()
