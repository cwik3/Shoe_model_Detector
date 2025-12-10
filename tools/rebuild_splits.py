#!/usr/bin/env python3
import argparse, hashlib, random, shutil, time
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Optional nice prefixes for filenames (fallback is first 4 chars of class)
CLASS_PREFIX = {
    "adidas_ozelia_cream_blue": "aocb",
    "new_balance_550_grey": "nbg",
    "nike_blazer_mid77_darkblue": "blazer",
    "nike_dunk_low_purple_court": "npc",
    "nike_pogo_plus_japanese_roots": "pogo",
}

def sha8(p: Path, chunk=1<<20):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()[:8]

def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def rm_tree_contents(folder: Path):
    if not folder.exists(): return
    for p in folder.iterdir():
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="root with class subfolders")
    ap.add_argument("--images", default="data/images", help="output root (will create train/ and val/)")
    ap.add_argument("--splits", default="data/splits", help="folder for train.txt/val.txt")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="fraction to put in val per class")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed for reproducibility")
    ap.add_argument("--mode", choices=["copy","move"], default="copy", help="copy (safe) or move files from raw")
    ap.add_argument("--dry_run", action="store_true", help="show actions without changing files")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    images_root = Path(args.images)
    train_dir = images_root / "train"
    val_dir   = images_root / "val"
    splits_dir = Path(args.splits)
    train_txt = splits_dir / "train.txt"
    val_txt   = splits_dir / "val.txt"

    # basic sanity
    if not raw_root.exists():
        print(f"[error] Raw folder not found: {raw_root}")
        return

    # 0) Backup existing images & splits (safe)
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = Path("Runs") / f"backup_{ts}"
    backup_imgs = backup_root / "images"
    backup_splits = backup_root / "splits"
    ensure(backup_root)
    if images_root.exists():
        ensure(backup_imgs)
        if not args.dry_run:
            shutil.copytree(images_root, backup_imgs, dirs_exist_ok=True)
    if splits_dir.exists():
        ensure(backup_splits)
        if not args.dry_run:
            for p in [train_txt, val_txt]:
                if p.exists():
                    ensure(backup_splits)
                    shutil.copy2(p, backup_splits / p.name)
    print(f"[backup] Saved current images/splits to {backup_root}")

    # 1) Collect class -> file list
    classes = [d for d in raw_root.iterdir() if d.is_dir()]
    if not classes:
        print("[error] No class subfolders found in data/raw"); return

    per_class = {}
    for d in classes:
        files = [p for p in iter_images(d)]
        if files:
            per_class[d.name] = files
    if not per_class:
        print("[error] No images found under data/raw/*"); return

    # 2) Shuffle per class with seed
    random.seed(args.seed)
    for cls, files in per_class.items():
        files.sort(key=lambda p: str(p).lower())
        random.shuffle(files)

    # 3) Prepare output folders
    ensure(images_root); ensure(train_dir); ensure(val_dir); ensure(splits_dir)
    if args.dry_run:
        print("[dry-run] Not clearing data/images/*")
    else:
        rm_tree_contents(train_dir)
        rm_tree_contents(val_dir)

    # 4) Split, copy/move, and build split lines
    lines_train, lines_val = [], []
    summary = defaultdict(lambda: {"train":0,"val":0,"total":0})

    for cls, files in per_class.items():
        n = len(files)
        n_val = min(max(1, int(n * args.val_ratio)) if n>1 else (1 if args.val_ratio>0 else 0), n)
        # first part to val, rest to train (shuffle done)
        val_files = files[:n_val]
        train_files = files[n_val:]
        prefix = CLASS_PREFIX.get(cls, cls[:4].lower())

        for src in val_files + train_files:
            short = sha8(src)
            dst_folder = val_dir if src in val_files else train_dir
            name = f"{prefix}_{short}{src.suffix.lower()}"
            dst = dst_folder / name

            # avoid very rare collisions
            c = 1
            while dst.exists():
                dst = dst_folder / f"{prefix}_{short}_{c}{src.suffix.lower()}"; c += 1

            rel = f"data/images/{dst_folder.name}/{dst.name}"

            if args.dry_run:
                print(f"[would {args.mode}] {src} -> {dst}")
            else:
                if args.mode == "copy":
                    shutil.copy2(src, dst)
                else:
                    ensure(dst.parent)
                    shutil.move(str(src), str(dst))

            if dst_folder is val_dir:
                lines_val.append(rel)
                summary[cls]["val"] += 1
            else:
                lines_train.append(rel)
                summary[cls]["train"] += 1
            summary[cls]["total"] += 1

    # 5) Write split files
    if args.dry_run:
        print("[dry-run] Not writing train.txt / val.txt")
    else:
        with train_txt.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_train) + ("\n" if lines_train else ""))
        with val_txt.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_val) + ("\n" if lines_val else ""))

    # 6) Summary
    print("\nSummary per class:")
    for cls, s in summary.items():
        print(f"  {cls:32s} → total {s['total']:4d} | train {s['train']:4d} | val {s['val']:4d}")
    print(f"\nWrote splits to: {train_txt} , {val_txt}")
    print(f"Output images:   {train_dir} , {val_dir}")
    if args.dry_run:
        print("\nThis was a dry run. Run again without --dry_run to apply.")
if __name__ == "__main__":
    main()
