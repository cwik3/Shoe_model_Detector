import argparse, shutil, random, sys
from pathlib import Path

random.seed(42)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP"}

def gather_images(raw_dir: Path):
    items = []
    if not raw_dir.exists():
        print(f"[ERROR] raw_dir not found: {raw_dir}", file=sys.stderr)
        return items
    for cls_dir in raw_dir.iterdir():
        if cls_dir.is_dir():
            for p in cls_dir.rglob('*'):
                if p.suffix in IMG_EXTS:
                    items.append(p)
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    images_dir = args.out_dir / "images"
    labels_dir = args.out_dir / "labels"
    splits_dir = args.out_dir / "splits"
    (images_dir / "train").mkdir(parents=True, exist_ok=True)
    (images_dir / "val").mkdir(parents=True, exist_ok=True)
    (labels_dir / "train").mkdir(parents=True, exist_ok=True)
    (labels_dir / "val").mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    imgs = gather_images(args.raw_dir)
    print(f"[INFO] Found {len(imgs)} images under {args.raw_dir}")
    if len(imgs) == 0:
        print("[WARN] No images found. Check your folder paths and extensions.")
        return

    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * args.val_ratio))
    val_set = set(imgs[:n_val])

    train_list, val_list = [], []

    for idx, src in enumerate(imgs, 1):
        split = "val" if src in val_set else "train"
        dst_img = images_dir / split / f"{src.stem}{src.suffix.lower()}"
        shutil.copy2(src, dst_img)
        # Create empty label files as placeholders
        (labels_dir / split / f"{src.stem}.txt").write_text("")
        (val_list if split == "val" else train_list).append(dst_img.as_posix())
        if idx % 50 == 0 or idx == len(imgs):
            print(f"[INFO] Copied {idx}/{len(imgs)} -> {split}")

    (splits_dir / "train.txt").write_text("\n".join(train_list))
    (splits_dir / "val.txt").write_text("\n".join(val_list))
    print(f"[DONE] Prepared {len(train_list)} train and {len(val_list)} val images.")
    print(f"[PATH] images/train: {images_dir/'train'}")
    print(f"[PATH] images/val  : {images_dir/'val'}")
    print(f"[PATH] labels/train: {labels_dir/'train'}")
    print(f"[PATH] labels/val  : {labels_dir/'val'}")

if __name__ == "__main__":
    main()
