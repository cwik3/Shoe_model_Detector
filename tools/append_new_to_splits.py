#!/usr/bin/env python3
import argparse, hashlib, os, shutil
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Customize your class → filename prefix here (used in data/images/*)
CLASS_PREFIX = {
    "adidas_ozelia_cream_blue": "aocb",
    "new_balance_550_grey": "nbg",
    "nike_blazer_mid77_darkblue": "blazer",
    "nike_dunk_low_purple_court": "npc",
    "nike_pogo_plus_japanese_roots": "pogo",
}

def sha256(p: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def all_images_under(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="root of class folders with raw images")
    ap.add_argument("--images", default="data/images", help="compiled dataset root (has train/ and val/)")
    ap.add_argument("--splits", default="splits", help="folder containing train.txt / val.txt")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="fraction of NEW images to send to val per class")
    ap.add_argument("--dry_run", action="store_true", help="show what would happen without copying or writing files")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    images_root = Path(args.images)
    train_dir = images_root / "train"
    val_dir   = images_root / "val"
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    # 1) Hash all existing images in data/images/train|val so we know what's already split
    print("Hashing existing dataset images (train/val)...")
    used_hashes = set()
    for p in list(all_images_under(train_dir)) + list(all_images_under(val_dir)):
        try:
            used_hashes.add(sha256(p))
        except Exception as e:
            print(f"[warn] could not hash {p}: {e}")

    # 2) Walk raw dataset and select NEW files (by content hash) per class
    print("Scanning data/raw for new images...")
    new_by_class = []
    for class_dir in [d for d in raw_root.iterdir() if d.is_dir()]:
        class_name = class_dir.name
        for p in all_images_under(class_dir):
            try:
                h = sha256(p)
            except Exception as e:
                print(f"[skip] {p} (hash error: {e})")
                continue
            if h in used_hashes:
                continue
            new_by_class.append((class_name, p, h))

    if not new_by_class:
        print("No new images found. You're up to date ✅")
        return

    # 3) Split per class (80/20 by default)
    # We’ll assign deterministically by sorting by hash (stable) to avoid randomness.
    by_cls = defaultdict(list)
    for cls, p, h in new_by_class:
        by_cls[cls].append((p, h))
    for cls in by_cls:
        by_cls[cls].sort(key=lambda x: x[1])  # sort by hash for stable results

    # 4) Copy and append to splits
    splits_dir = Path(args.splits)
    train_txt = splits_dir / "train.txt"
    val_txt   = splits_dir / "val.txt"
    ensure_dir(splits_dir)

    appended_train = []
    appended_val   = []

    for cls, items in by_cls.items():
        items_count = len(items)
        n_val = max(1, int(items_count * args.val_ratio)) if items_count > 1 else 1 if args.val_ratio > 0 else 0
        n_val = min(n_val, items_count)  # guard
        prefix = CLASS_PREFIX.get(cls, cls[:4].lower())  # fallback: 1st 4 chars

        for i, (src, h) in enumerate(items):
            dst_folder = val_dir if i < n_val else train_dir
            # filename: <prefix>_<shortHash>.<ext>, avoid collisions
            short = h[:8]
            dst_name = f"{prefix}_{short}{src.suffix.lower()}"
            dst_path = dst_folder / dst_name

            # If name collision, add a counter
            c = 1
            while dst_path.exists():
                dst_name = f"{prefix}_{short}_{c}{src.suffix.lower()}"
                dst_path = dst_folder / dst_name
                c += 1

            rel_path = str(Path("data") / "images" / dst_folder.name / dst_name).replace("\\", "/")
            if args.dry_run:
                print(f"[would copy] {src}  ->  {dst_path}")
            else:
                try:
                    shutil.copy2(src, dst_path)
                except Exception as e:
                    print(f"[error] copy failed {src} -> {dst_path}: {e}")
                    continue

            if dst_folder is val_dir:
                appended_val.append(rel_path)
            else:
                appended_train.append(rel_path)

    # 5) Append to split files
    if args.dry_run:
        print("\n[dry-run] Not writing to split files.")
    else:
        if appended_train:
            with train_txt.open("a", encoding="utf-8") as f:
                for line in appended_train:
                    f.write(line + "\n")
        if appended_val:
            with val_txt.open("a", encoding="utf-8") as f:
                for line in appended_val:
                    f.write(line + "\n")

    # 6) Summary
    print("\nSummary:")
    for cls, items in by_cls.items():
        items_count = len(items)
        n_val = min(max(1, int(items_count * args.val_ratio)) if items_count > 1 else (1 if args.val_ratio > 0 else 0), items_count)
        n_train = items_count - n_val
        print(f"  {cls}: {items_count} new  →  train {n_train} / val {n_val}")

    print(f"\nAppended to: {train_txt} (+{len(appended_train)})")
    print(f"           : {val_txt} (+{len(appended_val)})")
    if args.dry_run:
        print("\nRun again without --dry_run to actually copy and append.")
