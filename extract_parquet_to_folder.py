import os, io, argparse, pandas as pd
from pathlib import Path
from PIL import Image
from glob import glob

def save_split(parquet_glob, out_root, split, label_names=None):
    files = sorted(glob(parquet_glob))
    assert files, f"No files match {parquet_glob}"
    out_root = Path(out_root) / split
    out_root.mkdir(parents=True, exist_ok=True)

    img_id = 0
    for f in files:
        df = pd.read_parquet(f, columns=["image", "label"])
        for _, row in df.iterrows():
            label = int(row["label"])
            cls = str(label) if label_names is None else label_names.get(label, str(label))
            cls_dir = out_root / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(io.BytesIO(row["image"]['bytes'])).convert("RGB")
            img.save(cls_dir / f"{split}_{img_id:08d}.png", format="PNG")
            img_id += 1
    print(f"{split}: wrote {img_id} images to {out_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with train-*.parquet and test-*.parquet")
    ap.add_argument("--dst", required=True, help="output folder for ImageFolder")
    args = ap.parse_args()

    # optional readable class names; otherwise folders will be 0..9
    label_names = None

    save_split(os.path.join(args.src, "train-*.parquet"), args.dst, "train", label_names)
    save_split(os.path.join(args.src, "test-*.parquet"),  args.dst, "test",   label_names)

if __name__ == "__main__":
    main()
