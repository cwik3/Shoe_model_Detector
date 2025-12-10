import argparse, shutil, random, os, glob
from pathlib import Path

def main(src, dst, val_ratio):
    src = Path(src)
    dst = Path(dst)
    (dst/"images/train").mkdir(parents=True, exist_ok=True)
    (dst/"images/val").mkdir(parents=True, exist_ok=True)
    (dst/"labels/train").mkdir(parents=True, exist_ok=True)
    (dst/"labels/val").mkdir(parents=True, exist_ok=True)

    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        images += glob.glob(str(src / "images" / ext))
    images = [Path(p) for p in images]
    random.seed(42)
    random.shuffle(images)

    def lbl_for(img):
        return src/"labels"/(img.stem + ".txt")

    ok = []
    bad = []
    for img in images:
        lbl = lbl_for(img)
        if lbl.exists():
            ok.append((img,lbl))
        else:
            bad.append(img)
    if bad:
        print(f"[WARN] {len(bad)} images have no label .txt (will be ignored). Example:", bad[:3])

    n_val = max(1, int(len(ok)*val_ratio))
    val = ok[:n_val]
    train = ok[n_val:]

    def cp(pair_list, split):
        for img,lbl in pair_list:
            shutil.copy(img, dst/f"images/{split}"/img.name)
            shutil.copy(lbl, dst/f"labels/{split}"/lbl.name)

    cp(train,"train")
    cp(val,"val")

    # orphan check
    def check(split):
        imgs = {Path(p).stem for p in glob.glob(str(dst/f"images/{split}/*"))}
        lbls = {Path(p).stem for p in glob.glob(str(dst/f"labels/{split}/*"))}
        print(f"[{split}] images={len(imgs)} labels={len(lbls)} missing_labels={len(imgs-lbls)} missing_images={len(lbls-imgs)}")
    check("train"); check("val")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    args = ap.parse_args()
    main(args.src, args.dst, args.val_ratio)
